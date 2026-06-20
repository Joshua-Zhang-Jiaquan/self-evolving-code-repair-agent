#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repair_agent.config import ConfigError, ConfigMap, require_mapping  # noqa: E402
from repair_agent.env.swebench_loader import TaskManifest, load_task_manifest  # noqa: E402
from repair_agent.logging import read_json_object, read_jsonl, write_json_atomic  # noqa: E402
from repair_agent.resources import load_device_inventory, load_resource_config, resolve_resource_plan  # noqa: E402


STATUS_COMPLETED = "completed"
STATUS_BLOCKED = "blocked"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped_existing"

SMOKE_LIMIT = 1
LEARNING_MAIN_EPISODES = 1
ABRATION_EPISODES = 1
SEED = 20260619

INSTANCE_SPLIT_MAIN = "main"
FORCE_ARCHIVE_RUN_IDS = (
    "baseline_main",
    "feedback_main",
    "learning_main",
    "ablation_no_process_reward",
    "ablation_no_feedback_features",
    "ablation_reduced_test_budget",
)

HEAVY_STAGE_OFFICIAL_HARNESS = "official_swebench_harness"
HEAVY_STAGE_FULL_STRICT_EVAL = "full_strict_eval"
HEAVY_STAGE_IDS = (HEAVY_STAGE_OFFICIAL_HARNESS, HEAVY_STAGE_FULL_STRICT_EVAL)
BACKEND_LOCAL = "local"
BACKEND_QZ_PENDING = "qz_pending_approval"
BACKEND_OFFLOAD_UNAVAILABLE = "blocked_offload_unavailable"
QZ_SCHEMA_PATH = Path("outputs/qz/train.CreateJob.schema.yaml")
QZ_FULL_STRICT_JOB_PATH = Path("outputs/qz/full_strict_eval_job.json")
QZ_FULL_STRICT_DRY_RUN_PATH = Path("outputs/qz/full_strict_eval_dry_run.yaml")
QZ_RESOLVE_PLACEHOLDER = "RESOLVE_BEFORE_SUBMISSION"
FULL_STRICT_EVAL_QZ_COMMAND = (
    "python scripts/run_gated_experiments.py --manifest configs/task_manifest.yaml "
    "--out outputs/runs --resources configs/resources.yaml --strict-official --force"
)
STRICT_EVAL_MIN_FREE_GB = 5.0
STRICT_EVAL_MIN_CPU_WORKERS = 1

# JWT-like bearer tokens (qz tokens start with ``eyJ``) and ``key: value`` secret
# lines must never be persisted from qz output. Used to scrub before storage.
_SECRET_PATTERN = re.compile(
    r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"
    + r"|(?:token|secret|password|api[_-]?key|authorization|bearer)\s*[:=]\s*\S+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PipelineArgs:
    manifest: Path
    out: Path
    resources: Path
    dry_run_schedule: bool
    strict_official: bool = False
    force: bool = False


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    kind: str
    run_id: str | None
    config_path: str | None
    task_ids: tuple[str, ...]
    command: tuple[str, ...]
    required_artifacts: tuple[str, ...]
    seed: int = SEED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the gated Task 13 experiment pipeline")
    _ = parser.add_argument("--manifest", required=True, help="SWE-bench task manifest YAML")
    _ = parser.add_argument("--out", required=True, help="Run artifact directory, usually outputs/runs")
    _ = parser.add_argument("--resources", default="configs/resources.yaml", help="Resource YAML for local scheduling")
    _ = parser.add_argument(
        "--dry-run-schedule",
        action="store_true",
        help="Only write outputs/run_schedule.json without executing stages",
    )
    _ = parser.add_argument(
        "--strict-official",
        action="store_true",
        help="Execute the real official SWE-bench harness; blocked status is not completion",
    )
    _ = parser.add_argument(
        "--force",
        action="store_true",
        help="Force fresh runs instead of reusing existing artifacts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _typed_args(build_parser().parse_args(argv))
        return run_from_args(args)
    except (ConfigError, OSError, ValueError) as exc:
        print(f"gated experiment error: {exc}", file=sys.stderr)
        return 2


def run_from_args(args: PipelineArgs) -> int:
    manifest = load_task_manifest(args.manifest)
    stages = build_stage_specs(
        manifest=manifest,
        manifest_path=args.manifest,
        out_dir=args.out,
        resources_path=args.resources,
        strict_official=args.strict_official,
    )
    schedule = build_schedule(stages=stages, resources_path=args.resources)
    write_json_atomic(Path("outputs/run_schedule.json"), schedule)
    if args.dry_run_schedule:
        print("wrote dry-run schedule to outputs/run_schedule.json")
        return 0

    archived_runs = _archive_stale_run_dirs(args.out, FORCE_ARCHIVE_RUN_IDS) if args.force else []
    run_manifest: ConfigMap = _initial_run_manifest(args=args, task_manifest=manifest, stages=stages, schedule=schedule)
    run_manifest["archived_stale_runs"] = archived_runs
    if args.strict_official:
        run_manifest["heavy_stage_offload"] = _heavy_stage_offload_record(args.resources, args.strict_official)
    stage_records: list[ConfigMap] = []
    for stage in stages:
        if stage.kind == "harness_status":
            if args.strict_official:
                record = _finalize_strict_harness_record(_execute_stage(stage))
            else:
                record = _write_controlled_harness_status(stage, args.resources, manifest)
        elif _should_skip_complete(stage, args):
            record = _skipped_stage_record(stage)
        else:
            record = _execute_stage(stage)
            if stage.kind == "gold_smoke" and record["status"] == STATUS_COMPLETED:
                _write_gold_smoke_metrics(stage, manifest)
        stage_records.append(record)
        run_manifest["stages"] = stage_records
        run_manifest["model_gates"] = _load_model_gates()
        run_manifest["status"] = _overall_status(stage_records)
        write_json_atomic(Path("outputs/run_manifest.json"), run_manifest)

    print(f"gated pipeline status={run_manifest['status']}; manifest written to outputs/run_manifest.json")
    return 0


def build_stage_specs(*, manifest: TaskManifest, manifest_path: Path, out_dir: Path, resources_path: Path, strict_official: bool = False) -> list[StageSpec]:
    python = sys.executable
    out = str(out_dir)
    manifest_arg = str(manifest_path)
    resources = str(resources_path)
    gold_dir = out_dir / "gold_patch_smoke"
    gold_predictions = gold_dir / "predictions.jsonl"
    harness_command = _harness_command(python, gold_predictions, resources, strict_official)
    specs = [
        StageSpec(
            stage_id="model_gate_qwable",
            kind="model_gate",
            run_id=None,
            config_path="configs/models.yaml",
            task_ids=(),
            command=(python, "scripts/check_model_gate.py", "--model", "qwable", "--dry-run", "--models-config", "configs/models.yaml", "--resources", resources, "--out-dir", "outputs/model_gates"),
            required_artifacts=("outputs/model_gates/qwable.json",),
        ),
        StageSpec(
            stage_id="model_gate_diffrwkv",
            kind="model_gate",
            run_id=None,
            config_path="configs/models.yaml",
            task_ids=(),
            command=(python, "scripts/check_model_gate.py", "--model", "diffrwkv", "--dry-run", "--models-config", "configs/models.yaml", "--resources", resources, "--out-dir", "outputs/model_gates"),
            required_artifacts=("outputs/model_gates/diffrwkv.json",),
        ),
        StageSpec(
            stage_id="gold_patch_smoke",
            kind="gold_smoke",
            run_id="gold_patch_smoke",
            config_path=manifest_arg,
            task_ids=manifest.smoke_ids,
            command=(python, "scripts/make_gold_smoke.py", "--manifest", manifest_arg, "--out", str(gold_predictions)),
            required_artifacts=(str(gold_predictions),),
        ),
        StageSpec(
            stage_id="validate_gold_patch_smoke",
            kind="validation",
            run_id="gold_patch_smoke",
            config_path=manifest_arg,
            task_ids=manifest.smoke_ids,
            command=(python, "scripts/validate_predictions.py", str(gold_predictions)),
            required_artifacts=(str(gold_predictions),),
        ),
        _run_stage("baseline_smoke", "local_smoke", "baseline_smoke", "configs/baseline.yaml", manifest.smoke_ids, resources, out, limit=SMOKE_LIMIT),
        _run_stage("feedback_smoke", "local_smoke", "feedback_smoke", "configs/feedback.yaml", manifest.smoke_ids, resources, out, limit=SMOKE_LIMIT),
        _train_stage("learning_smoke", "local_smoke", "learning_smoke", "configs/learning.yaml", manifest.smoke_ids, resources, out, episodes=1, limit=SMOKE_LIMIT),
        _run_stage("baseline_main", "local_main_style", "baseline_main", "configs/baseline.yaml", manifest.main_ids, resources, out, limit=None, strict_official=strict_official, manifest_path=manifest_arg),
        _run_stage("feedback_main", "local_main_style", "feedback_main", "configs/feedback.yaml", manifest.main_ids, resources, out, limit=None, strict_official=strict_official, manifest_path=manifest_arg),
        _train_stage("learning_main", "local_main_style", "learning_main", "configs/learning.yaml", manifest.main_ids, resources, out, episodes=LEARNING_MAIN_EPISODES, limit=None, strict_official=strict_official, manifest_path=manifest_arg),
        _train_stage("ablation_no_process_reward", "ablation", "ablation_no_process_reward", "configs/ablations/no_process_reward.yaml", manifest.main_ids, resources, out, episodes=ABRATION_EPISODES, limit=None, strict_official=strict_official, manifest_path=manifest_arg),
        _train_stage("ablation_no_feedback_features", "ablation", "ablation_no_feedback_features", "configs/ablations/no_feedback_features.yaml", manifest.main_ids, resources, out, episodes=ABRATION_EPISODES, limit=None, strict_official=strict_official, manifest_path=manifest_arg),
        _train_stage("ablation_reduced_test_budget", "ablation", "ablation_reduced_test_budget", "configs/ablations/reduced_test_budget.yaml", manifest.main_ids, resources, out, episodes=ABRATION_EPISODES, limit=None, strict_official=strict_official, manifest_path=manifest_arg),
        StageSpec(
            stage_id="official_harness_status",
            kind="harness_status",
            run_id=None,
            config_path=manifest_arg,
            task_ids=manifest.all_ids,
            command=harness_command,
            required_artifacts=("outputs/harness_status.json",),
        ),
    ]
    return specs


def _harness_command(python: str, gold_predictions: Path, resources: str, strict_official: bool) -> tuple[str, ...]:
    base = (
        python,
        "-m",
        "repair_agent.env.harness",
        "--predictions",
        str(gold_predictions),
        "--run-id",
        "official_gold_smoke" if strict_official else "task13_gold_patch_smoke",
        "--auto-workers",
        "--resources",
        resources,
        "--status-out",
        "outputs/harness_status.json",
    )
    if strict_official:
        return base + ("--timeout-seconds", "1800", "--strict-official")
    return base + ("--simulate-docker-failure",)


def build_schedule(*, stages: list[StageSpec], resources_path: Path) -> ConfigMap:
    resources = load_resource_config(resources_path)
    inventory_path = Path("outputs/device_inventory.json")
    inventory = load_device_inventory(inventory_path)
    plan = resolve_resource_plan(resources, inventory, str(inventory_path) if inventory is not None else None)
    expected_gpus = _expected_gpu_ids(resources)
    healthy = list(plan.visible_gpus)
    missing = [gpu_id for gpu_id in expected_gpus if gpu_id not in healthy]
    assignments: list[ConfigMap] = []
    for index, stage in enumerate(stages):
        gpu_id = healthy[index % len(healthy)] if healthy else None
        assignments.append(
            {
                "command": list(stage.command),
                "command_line": _command_line(stage.command),
                "config_path": stage.config_path,
                "cuda_visible_devices": str(gpu_id) if gpu_id is not None else "",
                "gpu_id": gpu_id,
                "kind": stage.kind,
                "required_artifacts": list(stage.required_artifacts),
                "run_id": stage.run_id,
                "seed": stage.seed,
                "stage_id": stage.stage_id,
                "task_ids": list(stage.task_ids),
            }
        )
    used = sorted({cast(int, item["gpu_id"]) for item in assignments if isinstance(item.get("gpu_id"), int)})
    unused = [gpu_id for gpu_id in healthy if gpu_id not in used]
    worker_settings = plan.worker_settings
    return {
        "assignments": assignments,
        "auto_sized_swebench_workers": worker_settings.get("swebench_max_workers"),
        "generated_at": _now(),
        "gpu_coverage": {
            "expected_gpu_ids": expected_gpus,
            "healthy_visible_gpus": healthy,
            "missing_gpu_reasons": plan.fallback,
            "missing_gpus": missing,
            "unused_gpu_reasons": {str(gpu_id): "no_stage_assignment" for gpu_id in unused},
            "unused_healthy_gpus": unused,
            "used_healthy_gpus": used,
        },
        "inventory_source": plan.inventory_source,
        "resource_plan": plan.to_record(),
        "resources_path": str(resources_path),
        "swebench_worker_source": "resources.auto" if worker_settings.get("swebench_max_workers") else "unavailable",
    }


def _strict_or_limit_args(strict_official: bool, manifest_path: str | None, limit: int | None) -> list[str]:
    if strict_official and manifest_path is not None:
        return ["--manifest", manifest_path, "--instance-split", INSTANCE_SPLIT_MAIN, "--strict-official"]
    if limit is not None:
        return ["--limit", str(limit)]
    return []


def _run_stage(
    stage_id: str,
    kind: str,
    run_id: str,
    config_path: str,
    task_ids: tuple[str, ...],
    resources: str,
    out: str,
    *,
    limit: int | None,
    strict_official: bool = False,
    manifest_path: str | None = None,
) -> StageSpec:
    command = [sys.executable, "-m", "repair_agent.run", "--config", config_path, "--resources", resources, "--run-id", run_id]
    command.extend(_strict_or_limit_args(strict_official, manifest_path, limit))
    return StageSpec(
        stage_id=stage_id,
        kind=kind,
        run_id=run_id,
        config_path=config_path,
        task_ids=task_ids,
        command=tuple(command),
        required_artifacts=(f"{out}/{run_id}/predictions.jsonl", f"{out}/{run_id}/trajectories.jsonl", f"{out}/{run_id}/metrics.json"),
    )


def _train_stage(
    stage_id: str,
    kind: str,
    run_id: str,
    config_path: str,
    task_ids: tuple[str, ...],
    resources: str,
    out: str,
    *,
    episodes: int,
    limit: int | None,
    strict_official: bool = False,
    manifest_path: str | None = None,
) -> StageSpec:
    command = [sys.executable, "-m", "repair_agent.training.train", "--config", config_path, "--resources", resources, "--episodes", str(episodes), "--run-id", run_id]
    command.extend(_strict_or_limit_args(strict_official, manifest_path, limit))
    return StageSpec(
        stage_id=stage_id,
        kind=kind,
        run_id=run_id,
        config_path=config_path,
        task_ids=task_ids,
        command=tuple(command),
        required_artifacts=(
            f"{out}/{run_id}/predictions.jsonl",
            f"{out}/{run_id}/trajectories.jsonl",
            f"{out}/{run_id}/metrics.json",
            f"{out}/{run_id}/policy.json",
            f"{out}/{run_id}/rewards.jsonl",
        ),
    )


def _execute_stage(stage: StageSpec) -> ConfigMap:
    started = _now()
    completed = subprocess.run(stage.command, capture_output=True, text=True, check=False)
    status = STATUS_COMPLETED if completed.returncode == 0 else _failure_status(stage)
    if completed.returncode == 0 and stage.kind == "model_gate":
        status = _model_gate_stage_status(stage)
    return {
        "command": list(stage.command),
        "command_line": _command_line(stage.command),
        "completed_at": _now(),
        "config_path": stage.config_path,
        "kind": stage.kind,
        "reason": _stage_reason(status, completed.returncode, completed.stderr or completed.stdout),
        "required_artifacts": list(stage.required_artifacts),
        "returncode": completed.returncode,
        "run_id": stage.run_id,
        "seed": stage.seed,
        "stage_id": stage.stage_id,
        "started_at": started,
        "status": status,
        "stderr_tail": _tail(completed.stderr),
        "stdout_tail": _tail(completed.stdout),
        "task_ids": list(stage.task_ids),
    }


def _write_controlled_harness_status(stage: StageSpec, resources_path: Path, manifest: TaskManifest) -> ConfigMap:
    resources = load_resource_config(resources_path)
    inventory_path = Path("outputs/device_inventory.json")
    inventory = load_device_inventory(inventory_path)
    plan = resolve_resource_plan(resources, inventory, str(inventory_path) if inventory is not None else None)
    status: ConfigMap = {
        "blocked_reason": "official_swebench_harness_not_executed_by_task13_safe_local_pipeline",
        "cache_level": plan.worker_settings.get("docker_cache_level", "env"),
        "command": list(stage.command),
        "dataset_name": manifest.dataset_name,
        "fallback_reason": "official_swebench_harness_not_executed_by_task13_safe_local_pipeline",
        "max_workers": plan.worker_settings.get("swebench_max_workers"),
        "max_workers_source": "resources",
        "official_harness_executed": False,
        "resources": plan.to_record(),
        "run_id": "task13_gold_patch_smoke",
        "split": manifest.split,
        "status": STATUS_BLOCKED,
        "timestamp": _now(),
    }
    write_json_atomic(Path("outputs/harness_status.json"), status)
    return {
        "command": list(stage.command),
        "command_line": _command_line(stage.command),
        "config_path": stage.config_path,
        "kind": stage.kind,
        "reason": cast(str, status["blocked_reason"]),
        "required_artifacts": list(stage.required_artifacts),
        "returncode": 0,
        "run_id": stage.run_id,
        "seed": stage.seed,
        "stage_id": stage.stage_id,
        "status": STATUS_BLOCKED,
        "task_ids": list(stage.task_ids),
    }


def _finalize_strict_harness_record(record: ConfigMap) -> ConfigMap:
    status = read_json_object(Path("outputs/harness_status.json"), {})
    blocked = status.get("status") == STATUS_BLOCKED or status.get("official_harness_executed") is False
    if not blocked:
        return record
    updated = dict(record)
    updated["status"] = STATUS_BLOCKED
    updated["reason"] = _strict_harness_blocked_reason(status)
    return updated


def _strict_harness_blocked_reason(status: ConfigMap) -> str:
    blockers = status.get("blockers")
    if isinstance(blockers, list) and blockers:
        joined = ",".join(str(item) for item in cast(list[object], blockers))
        return f"official_harness_blocked_not_completion:{joined}"
    return "official_harness_blocked_not_completion"


def _write_gold_smoke_metrics(stage: StageSpec, manifest: TaskManifest) -> None:
    if stage.run_id is None:
        return
    run_dir = Path("outputs/runs") / stage.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    trajectories = [
        {
            "actions": ["load_actual_gold_patch", "write_prediction_jsonl"],
            "event": "gold_patch_smoke",
            "instance_id": instance_id,
            "run_id": stage.run_id,
            "run_name": "gold_smoke",
            "status": STATUS_COMPLETED,
            "timestamp": _now(),
        }
        for instance_id in manifest.smoke_ids
    ]
    with (run_dir / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for row in trajectories:
            json.dump(row, handle, sort_keys=True, separators=(",", ":"))
            _ = handle.write("\n")
    write_json_atomic(
        run_dir / "metrics.json",
        {"completed": len(predictions), "instances": [{"instance_id": row.get("instance_id"), "final_status": STATUS_COMPLETED} for row in predictions], "skipped": 0, "total": len(manifest.smoke_ids)},
    )
    write_json_atomic(run_dir / "run_state.json", {"completed_instances": list(manifest.smoke_ids), "status": STATUS_COMPLETED, "run_id": stage.run_id})


def _stage_already_complete(stage: StageSpec) -> bool:
    if stage.kind in {"model_gate", "validation", "harness_status"}:
        return False
    if not stage.required_artifacts or not all(Path(path).is_file() for path in stage.required_artifacts):
        return False
    if stage.run_id is None:
        return True
    state = read_json_object(Path("outputs/runs") / stage.run_id / "run_state.json", {})
    metrics = read_json_object(Path("outputs/runs") / stage.run_id / "metrics.json", {})
    return state.get("status") in {STATUS_COMPLETED, "NO_SIGNAL", "COMPLETED"} or int(cast(int, metrics.get("completed", 0))) > 0


def _should_skip_complete(stage: StageSpec, args: PipelineArgs) -> bool:
    if args.strict_official and args.force:
        return False
    return _stage_already_complete(stage)


def _archive_stale_run_dirs(out_dir: Path, run_ids: Iterable[str]) -> list[str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived: list[str] = []
    for run_id in run_ids:
        run_dir = out_dir / run_id
        if not run_dir.exists():
            continue
        target = out_dir / f"{run_id}.archived.{timestamp}"
        suffix = 1
        while target.exists():
            target = out_dir / f"{run_id}.archived.{timestamp}.{suffix}"
            suffix += 1
        _ = run_dir.rename(target)
        archived.append(str(target))
    return archived


def classify_heavy_stages(resources_path: str | Path, strict_official: bool) -> list[str]:
    stages: list[str] = []
    if not docker_daemon_available():
        stages.append(HEAVY_STAGE_OFFICIAL_HARNESS)
    if strict_official and not _strict_eval_locally_feasible(resources_path):
        stages.append(HEAVY_STAGE_FULL_STRICT_EVAL)
    return stages


def docker_daemon_available() -> bool:
    docker_path = shutil.which("docker")
    if docker_path is None:
        return False
    try:
        completed = subprocess.run([docker_path, "info"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _strict_eval_locally_feasible(resources_path: str | Path) -> bool:
    resources = load_resource_config(resources_path)
    inventory_path = Path("outputs/device_inventory.json")
    inventory = load_device_inventory(inventory_path)
    plan = resolve_resource_plan(resources, inventory, str(inventory_path) if inventory is not None else None)
    cpu_workers = _worker_int(plan.worker_settings.get("cpu_max_workers"))
    free_gb = shutil.disk_usage(_disk_probe_path()).free / float(1024**3)
    return cpu_workers >= STRICT_EVAL_MIN_CPU_WORKERS and free_gb >= STRICT_EVAL_MIN_FREE_GB


def _qz_available() -> bool:
    return shutil.which("qz") is not None


def _heavy_stage_offload_record(resources_path: Path, strict_official: bool) -> ConfigMap:
    requiring_offload = classify_heavy_stages(resources_path, strict_official)
    qz_usable = _qz_available()
    schema_present = QZ_SCHEMA_PATH.is_file()
    job_path: str | None = None
    dry_run_path: str | None = None
    if requiring_offload and qz_usable and schema_present:
        job_path, dry_run_path = _prepare_and_dry_run_qz_job(QZ_SCHEMA_PATH)
    stages: ConfigMap = {}
    for stage_id in HEAVY_STAGE_IDS:
        if stage_id not in requiring_offload:
            stages[stage_id] = {"execution_backend": BACKEND_LOCAL, "requires_offload": False}
        elif job_path is not None:
            stages[stage_id] = {
                "execution_backend": BACKEND_QZ_PENDING,
                "requires_offload": True,
                "qz_job_spec": job_path,
                "qz_dry_run": dry_run_path,
                "submitted": False,
            }
        else:
            stages[stage_id] = {
                "execution_backend": BACKEND_OFFLOAD_UNAVAILABLE,
                "requires_offload": True,
                "qz_available": qz_usable,
                "schema_present": schema_present,
            }
    return {
        "qz_available": qz_usable,
        "schema_present": schema_present,
        "stages": stages,
        "stages_requiring_offload": requiring_offload,
    }


def _prepare_and_dry_run_qz_job(schema_path: Path) -> tuple[str, str]:
    spec = _full_strict_eval_qz_spec(_qz_required_fields(schema_path))
    write_json_atomic(QZ_FULL_STRICT_JOB_PATH, spec)
    _qz_dry_run(QZ_FULL_STRICT_JOB_PATH, QZ_FULL_STRICT_DRY_RUN_PATH)
    return str(QZ_FULL_STRICT_JOB_PATH), str(QZ_FULL_STRICT_DRY_RUN_PATH)


def _qz_required_fields(schema_path: Path) -> list[str]:
    loaded = cast(object, json.loads(schema_path.read_text(encoding="utf-8")))
    if not isinstance(loaded, dict):
        raise ConfigError("qz schema must decode to a JSON object")
    schema = cast(dict[str, object], loaded)
    parameters = schema.get("parameters")
    required: list[str] = []
    if isinstance(parameters, list):
        for parameter in cast(list[object], parameters):
            if not isinstance(parameter, dict):
                continue
            parameter_map = cast(dict[str, object], parameter)
            if parameter_map.get("required") is True:
                field = parameter_map.get("jsonField")
                if isinstance(field, str) and field:
                    required.append(field)
    return required


def _full_strict_eval_qz_spec(required_fields: list[str]) -> ConfigMap:
    spec: ConfigMap = {
        "command": FULL_STRICT_EVAL_QZ_COMMAND,
        "framework": "PyTorch",
        "framework_config": [
            {
                "image": QZ_RESOLVE_PLACEHOLDER,
                "image_type": QZ_RESOLVE_PLACEHOLDER,
                "instance_count": 1,
                "spec_id": QZ_RESOLVE_PLACEHOLDER,
            }
        ],
        "logic_compute_group_id": QZ_RESOLVE_PLACEHOLDER,
        "name": "full-strict-eval-40id",
        "project_id": QZ_RESOLVE_PLACEHOLDER,
        "workspace_id": QZ_RESOLVE_PLACEHOLDER,
    }
    for field in required_fields:
        if field not in spec:
            spec[field] = QZ_RESOLVE_PLACEHOLDER
    return spec


def _qz_dry_run(job_path: Path, dry_run_path: Path) -> None:
    qz_path = shutil.which("qz")
    if qz_path is None:
        return
    completed = subprocess.run(
        [qz_path, "train", "CreateJob", "--data", job_path.read_text(encoding="utf-8"), "--dry-run", "-o", "yaml"],
        capture_output=True,
        text=True,
        check=False,
    )
    content = completed.stdout if completed.stdout.strip() else completed.stderr
    dry_run_path.parent.mkdir(parents=True, exist_ok=True)
    _ = dry_run_path.write_text(_scrub_secrets(content), encoding="utf-8")


def _scrub_secrets(text: str) -> str:
    return _SECRET_PATTERN.sub("***REDACTED***", text)


def _disk_probe_path() -> Path:
    cwd = Path.cwd()
    return cwd if cwd.exists() else PROJECT_ROOT


def _worker_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _skipped_stage_record(stage: StageSpec) -> ConfigMap:
    return {
        "command": list(stage.command),
        "command_line": _command_line(stage.command),
        "config_path": stage.config_path,
        "kind": stage.kind,
        "reason": "required_artifacts_already_present_resume_no_duplicate",
        "required_artifacts": list(stage.required_artifacts),
        "returncode": 0,
        "run_id": stage.run_id,
        "seed": stage.seed,
        "stage_id": stage.stage_id,
        "status": STATUS_SKIPPED,
        "task_ids": list(stage.task_ids),
    }


def _initial_run_manifest(*, args: PipelineArgs, task_manifest: TaskManifest, stages: list[StageSpec], schedule: ConfigMap) -> ConfigMap:
    return {
        "command_line": _command_line(tuple(sys.argv)),
        "configs": sorted({stage.config_path for stage in stages if stage.config_path}),
        "dataset_name": task_manifest.dataset_name,
        "force": args.force,
        "generated_at": _now(),
        "main_ids": list(task_manifest.main_ids),
        "manifest_path": str(args.manifest),
        "model_gates": _load_model_gates(),
        "out_dir": str(args.out),
        "resources_path": str(args.resources),
        "schedule_path": "outputs/run_schedule.json",
        "seeds": {stage.stage_id: stage.seed for stage in stages},
        "smoke_ids": list(task_manifest.smoke_ids),
        "split": task_manifest.split,
        "stages": [],
        "status": "initialized",
        "strict_official": args.strict_official,
        "task_ids": list(task_manifest.all_ids),
        "total_stage_count": len(stages),
        "worker_plan": schedule.get("resource_plan", {}),
    }


def _load_model_gates() -> ConfigMap:
    gates: ConfigMap = {}
    gate_dir = Path("outputs/model_gates")
    for path in sorted(gate_dir.glob("*.json")) if gate_dir.is_dir() else []:
        record = read_json_object(path, {})
        model = record.get("model")
        if isinstance(model, str):
            gates[model] = record
    return gates


def _overall_status(records: list[ConfigMap]) -> str:
    statuses = {str(record.get("status")) for record in records}
    if STATUS_FAILED in statuses:
        return "partial_failed"
    if STATUS_BLOCKED in statuses:
        return "controlled_partial"
    return STATUS_COMPLETED


def _failure_status(stage: StageSpec) -> str:
    return STATUS_BLOCKED if stage.kind in {"gold_smoke", "harness_status", "validation"} else STATUS_FAILED


def _model_gate_stage_status(stage: StageSpec) -> str:
    for artifact in stage.required_artifacts:
        record = read_json_object(artifact, {})
        gate_status = record.get("status")
        if gate_status == "blocked":
            return STATUS_BLOCKED
        if gate_status == "failed":
            return STATUS_FAILED
    return STATUS_COMPLETED


def _stage_reason(status: str, returncode: int, output: str) -> str:
    if status == STATUS_COMPLETED:
        return "command_completed"
    prefix = "controlled_blocked" if status == STATUS_BLOCKED else "command_failed"
    return f"{prefix}_returncode_{returncode}:{_tail(output, limit=300)}"


def _expected_gpu_ids(resources: ConfigMap) -> list[int]:
    gpus = require_mapping(resources.get("gpus"), "Resource config must define gpus")
    expected = gpus.get("expected_ids", [])
    return [item for item in cast(list[object], expected) if isinstance(item, int) and not isinstance(item, bool)] if isinstance(expected, list) else []


def _typed_args(namespace: argparse.Namespace) -> PipelineArgs:
    manifest = cast(object, getattr(namespace, "manifest"))
    out = cast(object, getattr(namespace, "out"))
    resources = cast(object, getattr(namespace, "resources"))
    dry_run_schedule = cast(object, getattr(namespace, "dry_run_schedule"))
    strict_official = cast(object, getattr(namespace, "strict_official"))
    force = cast(object, getattr(namespace, "force"))
    if not isinstance(manifest, str) or not manifest.strip():
        raise ConfigError("--manifest must be a non-empty string")
    if not isinstance(out, str) or not out.strip():
        raise ConfigError("--out must be a non-empty string")
    if not isinstance(resources, str) or not resources.strip():
        raise ConfigError("--resources must be a non-empty string")
    if not isinstance(dry_run_schedule, bool):
        raise ConfigError("--dry-run-schedule must be a boolean")
    if not isinstance(strict_official, bool):
        raise ConfigError("--strict-official must be a boolean")
    if not isinstance(force, bool):
        raise ConfigError("--force must be a boolean")
    return PipelineArgs(
        manifest=Path(manifest),
        out=Path(out),
        resources=Path(resources),
        dry_run_schedule=dry_run_schedule,
        strict_official=strict_official,
        force=force,
    )


def _command_line(command: tuple[str, ...]) -> str:
    return " ".join(_quote(part) for part in command)


def _quote(value: str) -> str:
    if value and all(char.isalnum() or char in "-_=./:" for char in value):
        return value
    return json.dumps(value)


def _tail(value: str, *, limit: int = 2000) -> str:
    stripped = value.strip()
    return stripped[-limit:]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
