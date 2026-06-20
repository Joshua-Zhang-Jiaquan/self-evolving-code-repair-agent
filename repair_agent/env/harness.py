from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from repair_agent.config import ConfigError, ConfigMap, require_string
from repair_agent.env import defects4j_harness
from repair_agent.logging import write_json_atomic
from repair_agent.resources import load_device_inventory, load_resource_config, resolve_resource_plan


DEFAULT_DATASET = "princeton-nlp/SWE-bench_Lite"
DEFAULT_SPLIT = "test"
DEFAULT_STATUS_PATH = Path("outputs/harness_status.json")
INSTANCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*__[A-Za-z0-9][A-Za-z0-9._-]*$")
DEFECTS4J_ID_PATTERN = re.compile(r"^[A-Z][a-zA-Z0-9]*_[0-9]+$")


@dataclass(frozen=True)
class HarnessArgs:
    predictions: Path
    run_id: str
    max_workers: int | None
    auto_workers: bool
    resources: Path | None
    inventory: Path
    cache_level: str | None
    dataset_name: str
    split: str
    status_out: Path
    simulate_docker_failure: bool
    strict_official: bool
    timeout_seconds: int
    defects4j_home: Path | None
    skip_defects4j_fallback: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or preflight the official SWE-bench harness")
    _ = parser.add_argument("--predictions", required=True, help="SWE-bench prediction JSONL path")
    _ = parser.add_argument("--run-id", required=True, help="SWE-bench run id")
    _ = parser.add_argument("--max-workers", type=int, help="Explicit SWE-bench max worker count")
    _ = parser.add_argument("--auto-workers", action="store_true", help="Resolve max workers from resources")
    _ = parser.add_argument("--resources", help="Resource YAML used with --auto-workers")
    _ = parser.add_argument(
        "--inventory",
        default="outputs/device_inventory.json",
        help="Device inventory JSON used with --auto-workers",
    )
    _ = parser.add_argument("--cache-level", help="Override SWE-bench Docker cache level")
    _ = parser.add_argument("--dataset-name", default=DEFAULT_DATASET, help="SWE-bench dataset name")
    _ = parser.add_argument("--split", default=DEFAULT_SPLIT, help="SWE-bench split")
    _ = parser.add_argument("--status-out", default=str(DEFAULT_STATUS_PATH), help="Status JSON path")
    _ = parser.add_argument(
        "--simulate-docker-failure",
        action="store_true",
        help="Record a controlled official-harness blocked status without running Docker",
    )
    _ = parser.add_argument(
        "--strict-official",
        action="store_true",
        help="Require a real official run; exit nonzero when the harness is blocked",
    )
    _ = parser.add_argument(
        "--timeout-seconds",
        "--timeout",
        dest="timeout_seconds",
        type=int,
        default=1800,
        help="Official harness timeout in seconds",
    )
    _ = parser.add_argument(
        "--defects4j-home",
        help="Path to Defects4J installation for non-Docker fallback evaluation",
    )
    _ = parser.add_argument(
        "--skip-defects4j-fallback",
        action="store_true",
        help="Disable Defects4J fallback when the Docker-based harness is blocked",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _typed_args(build_parser().parse_args(argv))
        return run_from_args(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


def run_from_args(args: HarnessArgs) -> int:
    if not args.predictions.is_file():
        raise ConfigError(f"Predictions file not found: {args.predictions}")
    selected_workers, worker_source, cache_level, resource_record = _select_workers(args)
    prediction_ids = _prediction_instance_ids(args.predictions)
    command = build_official_command(
        predictions=args.predictions,
        run_id=args.run_id,
        max_workers=selected_workers,
        cache_level=cache_level,
        dataset_name=args.dataset_name,
        split=args.split,
        instance_ids=prediction_ids,
    )
    base_status = _base_status(
        args=args,
        command=command,
        selected_workers=selected_workers,
        worker_source=worker_source,
        cache_level=cache_level,
        resource_record=resource_record,
    )

    prediction_count = len(prediction_ids) if prediction_ids else _count_predictions(args.predictions)
    report_dir = str(Path("logs/run_evaluation") / args.run_id)

    blocked_reason = _blocked_reason(args)
    if blocked_reason is not None:
        d4j_status = _try_defects4j_fallback(args, base_status, selected_workers, blocked_reason)
        if d4j_status is not None:
            write_json_atomic(args.status_out, d4j_status)
            print(f"SWE-bench harness blocked; Defects4J fallback status={d4j_status['status']}")
            if args.strict_official and d4j_status.get("status") != "completed":
                return 1
            return 0
        status = dict(base_status)
        status.update(
            {
                "blocked_reason": blocked_reason,
                "blockers": [blocked_reason],
                "fallback_reason": blocked_reason,
                "official_harness_executed": False,
                "report_dir": report_dir,
                "resolved": 0,
                "resolved_rate": 0.0,
                "status": "blocked",
                "total": prediction_count,
            }
        )
        write_json_atomic(args.status_out, status)
        print(f"SWE-bench harness blocked: {blocked_reason}; status written to {args.status_out}")
        return 1 if args.strict_official else 0

    result = _run_official(command, args.timeout_seconds)
    if result.get("status") != "completed":
        d4j_status = _try_defects4j_fallback(
            args, base_status, selected_workers, f"official_harness_failed:{result.get('fallback_reason', 'unknown')}"
        )
        if d4j_status is not None:
            write_json_atomic(args.status_out, d4j_status)
            print(f"SWE-bench harness fallback; Defects4J fallback status={d4j_status['status']}")
            if args.strict_official and d4j_status.get("status") != "completed":
                return 1
            return 0

    status = dict(base_status)
    status.update(result)
    status.update(_official_results(args.run_id, prediction_count))
    status["blockers"] = []
    write_json_atomic(args.status_out, status)
    print(f"SWE-bench harness status={status['status']}; status written to {args.status_out}")
    if args.strict_official and status.get("status") != "completed":
        return 1
    return 0


def build_official_command(
    *,
    predictions: Path,
    run_id: str,
    max_workers: int,
    cache_level: str,
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    instance_ids: tuple[str, ...] = (),
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        str(predictions),
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
        "--cache_level",
        cache_level,
        "--namespace",
        "none",
    ]
    if instance_ids:
        command.extend(["--instance_ids", *instance_ids])
    return command


def _select_workers(args: HarnessArgs) -> tuple[int, str, str, ConfigMap]:
    if args.auto_workers:
        if args.resources is None:
            raise ConfigError("--auto-workers requires --resources")
        resources = load_resource_config(args.resources)
        inventory = load_device_inventory(args.inventory)
        plan = resolve_resource_plan(resources, inventory, str(args.inventory) if inventory is not None else None)
        workers_raw = plan.worker_settings.get("swebench_max_workers")
        if not isinstance(workers_raw, int) or workers_raw < 1:
            raise ConfigError("Resolved swebench_max_workers must be a positive integer")
        cache_level = args.cache_level or str(plan.worker_settings.get("docker_cache_level", "env"))
        return workers_raw, "resources", cache_level, plan.to_record()

    if args.max_workers is not None:
        if args.max_workers < 1:
            raise ConfigError("--max-workers must be a positive integer")
        return args.max_workers, "cli", args.cache_level or "env", {}
    return 1, "default", args.cache_level or "env", {}


def _blocked_reason(args: HarnessArgs) -> str | None:
    if args.simulate_docker_failure:
        return "simulated_docker_failure"
    if importlib.util.find_spec("swebench") is None:
        return "swebench_package_unavailable"
    docker_path = shutil.which("docker")
    if docker_path is None:
        return "docker_cli_unavailable"
    try:
        completed = subprocess.run(
            [docker_path, "info"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"docker_daemon_unavailable:{exc.__class__.__name__}"
    if completed.returncode != 0:
        detail = _short_text(completed.stderr or completed.stdout)
        return f"docker_daemon_unavailable:{detail or completed.returncode}"
    return None


def _run_official(command: list[str], timeout_seconds: int) -> ConfigMap:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "fallback_reason": f"official_harness_timeout_after_{timeout_seconds}_seconds",
            "official_harness_executed": True,
            "status": "fallback",
            "stderr_tail": _short_text(exc.stderr),
            "stdout_tail": _short_text(exc.stdout),
        }
    status = "completed" if completed.returncode == 0 else "fallback"
    result: ConfigMap = {
        "official_harness_executed": True,
        "returncode": completed.returncode,
        "status": status,
        "stderr_tail": _short_text(completed.stderr),
        "stdout_tail": _short_text(completed.stdout),
    }
    if completed.returncode != 0:
        result["fallback_reason"] = f"official_harness_returned_{completed.returncode}"
    return result


def _try_defects4j_fallback(
    args: HarnessArgs,
    base_status: ConfigMap,
    selected_workers: int,
    fallback_reason: str,
) -> ConfigMap | None:
    """Run Defects4J evaluation when the Docker-based SWE-bench harness is unavailable.

    Returns None when fallback is disabled, Defects4J is not available, or no
    Defects4J-formatted predictions exist.
    """
    if args.skip_defects4j_fallback:
        return None
    if args.defects4j_home is not None:
        home_bin = args.defects4j_home / "framework" / "bin" / "defects4j"
        if not home_bin.is_file():
            return None
        os.environ["DEFECTS4J_HOME"] = str(args.defects4j_home)
    if not defects4j_harness.is_available():
        return None
    d4j_instances = defects4j_harness.defects4j_ids_in_predictions(args.predictions)
    if not d4j_instances:
        return None
    result = defects4j_harness.evaluate_predictions(
        predictions_path=args.predictions,
        run_id=args.run_id,
        max_workers=selected_workers,
    )
    status = dict(base_status)
    status.update(result)
    status["blockers"] = []
    status["fallback_reason"] = fallback_reason
    status["defects4j_available"] = True
    status["defects4j_instances"] = [i.instance_id for i in d4j_instances]
    return status


def _count_predictions(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return 0
    return sum(1 for line in text.splitlines() if line.strip())


def _prediction_instance_ids(path: Path) -> tuple[str, ...]:
    ids: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    for line in lines:
        if not line.strip():
            continue
        try:
            row_object = cast(object, json.loads(line))
        except json.JSONDecodeError:
            continue
        if not isinstance(row_object, dict):
            continue
        row = cast(dict[object, object], row_object)
        instance_id = row.get("instance_id")
        if isinstance(instance_id, str) and instance_id:
            if not _is_safe_instance_id(instance_id):
                raise ConfigError(f"invalid prediction instance_id for official harness: {instance_id!r}")
            ids.append(instance_id)
    return tuple(dict.fromkeys(ids))


def _is_safe_instance_id(instance_id: str) -> bool:
    return bool(INSTANCE_ID_PATTERN.fullmatch(instance_id) or DEFECTS4J_ID_PATTERN.fullmatch(instance_id))


def _official_results(run_id: str, prediction_count: int) -> ConfigMap:
    report_dir = Path("logs/run_evaluation") / run_id
    report = _load_run_report(run_id, report_dir)
    resolved = 0
    total = prediction_count
    if report is not None:
        resolved = _as_count(report.get("resolved_instances"))
        total_candidate = _as_count(report.get("total_instances"))
        if total_candidate > 0:
            total = total_candidate
    resolved_rate = round(resolved / total, 4) if total > 0 else 0.0
    return {
        "report_dir": str(report_dir),
        "resolved": resolved,
        "resolved_rate": resolved_rate,
        "total": total,
    }


def _load_run_report(run_id: str, report_dir: Path) -> ConfigMap | None:
    for candidate in _report_candidates(run_id):
        report = _read_json_safe(candidate)
        if isinstance(report, dict) and ("resolved_instances" in report or "total_instances" in report):
            return cast(ConfigMap, report)
    return _aggregate_instance_reports(report_dir)


def _report_candidates(run_id: str) -> list[Path]:
    candidates: list[Path] = []
    for root in (Path("."), Path("evaluation_results")):
        if root.is_dir():
            candidates.extend(sorted(root.glob(f"*.{run_id}.json")))
    return candidates


def _aggregate_instance_reports(report_dir: Path) -> ConfigMap | None:
    if not report_dir.is_dir():
        return None
    resolved = 0
    total = 0
    for path in sorted(report_dir.rglob("report.json")):
        data = _read_json_safe(path)
        if not isinstance(data, dict):
            continue
        for payload in cast(dict[str, object], data).values():
            total += 1
            if isinstance(payload, dict) and bool(cast(dict[str, object], payload).get("resolved")):
                resolved += 1
    if total == 0:
        return None
    return {"resolved_instances": resolved, "total_instances": total}


def _read_json_safe(path: Path) -> object:
    try:
        return cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _as_count(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value if value >= 0 else 0
    if isinstance(value, float):
        return int(value) if value >= 0 else 0
    return 0


def _base_status(
    *,
    args: HarnessArgs,
    command: list[str],
    selected_workers: int,
    worker_source: str,
    cache_level: str,
    resource_record: ConfigMap,
) -> ConfigMap:
    status: ConfigMap = {
        "cache_level": cache_level,
        "command": command,
        "dataset_name": args.dataset_name,
        "max_workers": selected_workers,
        "max_workers_source": worker_source,
        "predictions": str(args.predictions),
        "run_id": args.run_id,
        "split": args.split,
        "status": "initialized",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if resource_record:
        status["resources"] = resource_record
        status["cpu_snapshot"] = _cpu_snapshot(resource_record)
        status["memory_snapshot"] = _memory_snapshot(args.inventory)
    return status


def _cpu_snapshot(resource_record: ConfigMap) -> ConfigMap:
    worker_settings = resource_record.get("worker_settings", {})
    return {
        "logical_cores": os.cpu_count() or 1,
        "worker_settings": cast(ConfigMap, worker_settings) if isinstance(worker_settings, dict) else {},
    }


def _memory_snapshot(inventory_path: Path) -> ConfigMap:
    inventory = load_device_inventory(inventory_path)
    if inventory is None:
        return {"source": None}
    memory = inventory.get("memory", {})
    return cast(ConfigMap, memory) if isinstance(memory, dict) else {"source": str(inventory_path)}


def _short_text(value: object, limit: int = 2000) -> str:
    if value is None:
        return ""
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
    stripped = text.strip()
    return stripped[-limit:]


def _typed_args(namespace: argparse.Namespace) -> HarnessArgs:
    predictions = Path(require_string(_namespace_value(namespace, "predictions"), "--predictions must be a string"))
    run_id = require_string(_namespace_value(namespace, "run_id"), "--run-id must be a string")
    resources_raw = _namespace_value(namespace, "resources")
    resources = Path(resources_raw) if isinstance(resources_raw, str) and resources_raw.strip() else None
    max_workers = _namespace_value(namespace, "max_workers")
    if max_workers is not None and not isinstance(max_workers, int):
        raise ConfigError("--max-workers must be an integer")
    auto_workers = bool(_namespace_value(namespace, "auto_workers"))
    if auto_workers and max_workers is not None:
        raise ConfigError("Use either --auto-workers or --max-workers, not both")
    timeout_seconds = _namespace_value(namespace, "timeout_seconds")
    if not isinstance(timeout_seconds, int) or timeout_seconds < 1:
        raise ConfigError("--timeout-seconds must be a positive integer")
    cache_level = _namespace_value(namespace, "cache_level")
    if cache_level is not None and not isinstance(cache_level, str):
        raise ConfigError("--cache-level must be a string")
    defects4j_home_raw = _namespace_value(namespace, "defects4j_home")
    defects4j_home = Path(defects4j_home_raw) if isinstance(defects4j_home_raw, str) and defects4j_home_raw.strip() else None
    return HarnessArgs(
        predictions=predictions,
        run_id=run_id,
        max_workers=max_workers,
        auto_workers=auto_workers,
        resources=resources,
        inventory=Path(require_string(_namespace_value(namespace, "inventory"), "--inventory must be a string")),
        cache_level=cache_level,
        dataset_name=require_string(_namespace_value(namespace, "dataset_name"), "--dataset-name must be a string"),
        split=require_string(_namespace_value(namespace, "split"), "--split must be a string"),
        status_out=Path(require_string(_namespace_value(namespace, "status_out"), "--status-out must be a string")),
        simulate_docker_failure=bool(_namespace_value(namespace, "simulate_docker_failure")),
        strict_official=bool(_namespace_value(namespace, "strict_official")),
        timeout_seconds=timeout_seconds,
        defects4j_home=defects4j_home,
        skip_defects4j_fallback=bool(_namespace_value(namespace, "skip_defects4j_fallback")),
    )


def _namespace_value(namespace: argparse.Namespace, name: str) -> object:
    values = cast(dict[str, object], vars(namespace))
    return values.get(name)


if __name__ == "__main__":
    raise SystemExit(main())
