from __future__ import annotations

import json
import subprocess
from pathlib import Path
from collections.abc import Sequence
from typing import cast

import pytest
import yaml

from repair_agent.logging import read_json_object
from scripts.run_gated_experiments import (
    HEAVY_STAGE_FULL_STRICT_EVAL,
    HEAVY_STAGE_OFFICIAL_HARNESS,
    PipelineArgs,
    build_schedule,
    build_stage_specs,
    classify_heavy_stages,
    run_from_args,
)


def test_dry_run_schedule_covers_all_healthy_gpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    resources = _write_resources(tmp_path)
    _write_inventory(tmp_path, [0, 1, 2, 3], workers=16)
    manifest = _task_manifest_object()

    stages = build_stage_specs(manifest=manifest, manifest_path=Path("configs/task_manifest.yaml"), out_dir=Path("outputs/runs"), resources_path=resources)
    schedule = build_schedule(stages=stages, resources_path=resources)

    coverage = cast(dict[str, object], schedule["gpu_coverage"])
    assert coverage["healthy_visible_gpus"] == [0, 1, 2, 3]
    assert coverage["used_healthy_gpus"] == [0, 1, 2, 3]
    assert coverage["unused_healthy_gpus"] == []
    assert schedule["auto_sized_swebench_workers"] == 16
    assignments = cast(list[dict[str, object]], schedule["assignments"])
    assert len(assignments) == len(stages)


def test_pipeline_manifest_records_controlled_blocked_harness_and_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    resources = _write_resources(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    _write_inventory(tmp_path, [0, 1, 2, 3], workers=16)
    calls: list[tuple[str, ...]] = []

    def fake_run(command: object, capture_output: bool, text: bool, check: bool) -> subprocess.CompletedProcess[str]:
        _ = (capture_output, text, check)
        command_tuple = _command_tuple(command)
        calls.append(command_tuple)
        _materialize_command_outputs(command_tuple)
        return subprocess.CompletedProcess(command_tuple, 0, stdout="ok", stderr="")

    monkeypatch.setattr("scripts.run_gated_experiments.subprocess.run", fake_run)

    result = run_from_args(PipelineArgs(manifest=manifest_path, out=Path("outputs/runs"), resources=resources, dry_run_schedule=False))

    assert result == 0
    run_manifest = read_json_object("outputs/run_manifest.json")
    assert run_manifest["status"] == "controlled_partial"
    gates = cast(dict[str, dict[str, object]], run_manifest["model_gates"])
    assert gates["qwable"]["status"] == "pass"
    assert gates["diffrwkv"]["status"] == "blocked"
    harness = read_json_object("outputs/harness_status.json")
    assert harness["official_harness_executed"] is False
    assert harness["status"] == "blocked"
    assert any("scripts/make_gold_smoke.py" in command for command in calls)


def test_pipeline_resume_skips_existing_stage_without_duplicate_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    resources = _write_resources(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    _write_inventory(tmp_path, [0, 1, 2, 3], workers=16)
    _write_run_artifacts(Path("outputs/runs/baseline_smoke"), completed=1, learning=False)
    calls: list[tuple[str, ...]] = []

    def fake_run(command: object, capture_output: bool, text: bool, check: bool) -> subprocess.CompletedProcess[str]:
        _ = (capture_output, text, check)
        command_tuple = _command_tuple(command)
        calls.append(command_tuple)
        _materialize_command_outputs(command_tuple)
        return subprocess.CompletedProcess(command_tuple, 0, stdout="ok", stderr="")

    monkeypatch.setattr("scripts.run_gated_experiments.subprocess.run", fake_run)

    result = run_from_args(PipelineArgs(manifest=manifest_path, out=Path("outputs/runs"), resources=resources, dry_run_schedule=False))

    assert result == 0
    run_manifest = read_json_object("outputs/run_manifest.json")
    stages = cast(list[dict[str, object]], run_manifest["stages"])
    baseline_stage = next(stage for stage in stages if stage["stage_id"] == "baseline_smoke")
    assert baseline_stage["status"] == "skipped_existing"
    assert not any("baseline_smoke" in command and "repair_agent.run" in command for command in calls)
    prediction_rows = (Path("outputs/runs/baseline_smoke/predictions.jsonl").read_text(encoding="utf-8").strip().splitlines())
    assert len(prediction_rows) == 1


def test_strict_official_pipeline_marks_blocked_harness_not_completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    resources = _write_resources(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    _write_inventory(tmp_path, [0, 1, 2, 3], workers=16)
    calls: list[tuple[str, ...]] = []

    def fake_run(command: object, capture_output: bool, text: bool, check: bool) -> subprocess.CompletedProcess[str]:
        _ = (capture_output, text, check)
        command_tuple = _command_tuple(command)
        calls.append(command_tuple)
        if "repair_agent.env.harness" in command_tuple:
            _write_blocked_harness_status()
            return subprocess.CompletedProcess(command_tuple, 1, stdout="", stderr="docker_cli_unavailable")
        _materialize_command_outputs(command_tuple)
        return subprocess.CompletedProcess(command_tuple, 0, stdout="ok", stderr="")

    monkeypatch.setattr("scripts.run_gated_experiments.subprocess.run", fake_run)
    # Force the docker daemon probe deterministically so the test does not depend on whether the
    # host has a docker binary on PATH; the strict pipeline must still mark the harness blocked.
    monkeypatch.setattr("scripts.run_gated_experiments.docker_daemon_available", lambda: False)

    result = run_from_args(
        PipelineArgs(manifest=manifest_path, out=Path("outputs/runs"), resources=resources, dry_run_schedule=False, strict_official=True)
    )

    assert result == 0
    run_manifest = read_json_object("outputs/run_manifest.json")
    assert run_manifest["status"] == "controlled_partial"
    assert run_manifest["strict_official"] is True
    stages = cast(list[dict[str, object]], run_manifest["stages"])
    harness_stage = next(stage for stage in stages if stage["kind"] == "harness_status")
    assert harness_stage["status"] == "blocked"
    harness_command = cast(list[str], harness_stage["command"])
    assert "--strict-official" in harness_command
    assert "--simulate-docker-failure" not in harness_command
    assert "official_gold_smoke" in harness_command
    harness_status = read_json_object("outputs/harness_status.json")
    assert harness_status["official_harness_executed"] is False
    assert harness_status["status"] == "blocked"
    assert any("repair_agent.env.harness" in command and "--strict-official" in command for command in calls)


def test_strict_schedule_has_manifest_and_instance_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    resources = _write_resources(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    _write_inventory(tmp_path, [0, 1, 2, 3], workers=16)

    result = run_from_args(
        PipelineArgs(manifest=manifest_path, out=Path("outputs/runs"), resources=resources, dry_run_schedule=True, strict_official=True)
    )

    assert result == 0
    schedule = read_json_object("outputs/run_schedule.json")
    assignments = cast(list[dict[str, object]], schedule["assignments"])
    main_ablation = {
        "baseline_main",
        "feedback_main",
        "learning_main",
        "ablation_no_process_reward",
        "ablation_no_feedback_features",
        "ablation_reduced_test_budget",
    }
    main_commands = [cast(str, item["command_line"]) for item in assignments if item["stage_id"] in main_ablation]
    assert len(main_commands) == 6
    for command_line in main_commands:
        assert "--manifest" in command_line
        assert "--instance-split main" in command_line
        assert "--strict-official" in command_line
        assert "--limit" not in command_line
    smoke_commands = [cast(str, item["command_line"]) for item in assignments if item["stage_id"] in {"baseline_smoke", "feedback_smoke", "learning_smoke"}]
    assert len(smoke_commands) == 3
    for command_line in smoke_commands:
        assert "--strict-official" not in command_line
        assert "--limit 1" in command_line
    all_commands = "\n".join(cast(str, item["command_line"]) for item in assignments)
    assert "--simulate-docker-failure" not in all_commands


def test_strict_force_archives_stale_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    resources = _write_resources(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    _write_inventory(tmp_path, [0, 1, 2, 3], workers=16)
    stale_dir = Path("outputs/runs/baseline_main")
    _write_run_artifacts(stale_dir, completed=1, learning=False)
    _ = (stale_dir / "STALE_MARKER.txt").write_text("stale-one-row", encoding="utf-8")

    monkeypatch.setattr("scripts.run_gated_experiments.docker_daemon_available", _docker_unavailable)
    monkeypatch.setattr("scripts.run_gated_experiments._qz_available", _qz_unavailable)
    monkeypatch.setattr("scripts.run_gated_experiments.subprocess.run", _strict_fake_run([]))

    result = run_from_args(
        PipelineArgs(manifest=manifest_path, out=Path("outputs/runs"), resources=resources, dry_run_schedule=False, strict_official=True, force=True)
    )

    assert result == 0
    archives = list(Path("outputs/runs").glob("baseline_main.archived.*"))
    assert len(archives) >= 1
    assert any((archive / "STALE_MARKER.txt").is_file() for archive in archives)
    assert Path("outputs/runs/baseline_main").is_dir()
    assert not (Path("outputs/runs/baseline_main") / "STALE_MARKER.txt").is_file()
    run_manifest = read_json_object("outputs/run_manifest.json")
    archived_runs = cast(list[str], run_manifest["archived_stale_runs"])
    assert any("baseline_main.archived." in path for path in archived_runs)


def test_strict_no_skipped_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    resources = _write_resources(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    _write_inventory(tmp_path, [0, 1, 2, 3], workers=16)
    for run_id in ("baseline_main", "feedback_main", "learning_main"):
        _write_run_artifacts(Path("outputs/runs") / run_id, completed=2, learning=run_id == "learning_main")

    monkeypatch.setattr("scripts.run_gated_experiments.docker_daemon_available", _docker_unavailable)
    monkeypatch.setattr("scripts.run_gated_experiments._qz_available", _qz_unavailable)
    monkeypatch.setattr("scripts.run_gated_experiments.subprocess.run", _strict_fake_run([]))

    result = run_from_args(
        PipelineArgs(manifest=manifest_path, out=Path("outputs/runs"), resources=resources, dry_run_schedule=False, strict_official=True, force=True)
    )

    assert result == 0
    run_manifest = read_json_object("outputs/run_manifest.json")
    stages = cast(list[dict[str, object]], run_manifest["stages"])
    assert stages
    assert all(stage["status"] != "skipped_existing" for stage in stages)
    by_id = {cast(str, stage["stage_id"]): stage for stage in stages}
    for stage_id in (
        "baseline_main",
        "feedback_main",
        "learning_main",
        "ablation_no_process_reward",
        "ablation_no_feedback_features",
        "ablation_reduced_test_budget",
    ):
        assert by_id[stage_id]["status"] == "completed"


def test_heavy_stage_classifier_docker_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    resources = _write_resources(tmp_path)
    _write_inventory(tmp_path, [0, 1, 2, 3], workers=16)
    monkeypatch.setattr("scripts.run_gated_experiments.docker_daemon_available", _docker_unavailable)

    stages = classify_heavy_stages(resources, strict_official=True)

    assert HEAVY_STAGE_OFFICIAL_HARNESS in stages
    assert HEAVY_STAGE_FULL_STRICT_EVAL not in stages


def test_heavy_stage_classifier_local_feasible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    resources = _write_resources(tmp_path)
    _write_inventory(tmp_path, [0, 1, 2, 3], workers=16)
    monkeypatch.setattr("scripts.run_gated_experiments.docker_daemon_available", _docker_available)

    stages = classify_heavy_stages(resources, strict_official=True)

    assert HEAVY_STAGE_FULL_STRICT_EVAL not in stages
    assert stages == []


def _docker_unavailable() -> bool:
    return False


def _docker_available() -> bool:
    return True


def _qz_unavailable() -> bool:
    return False


def _strict_fake_run(calls: list[tuple[str, ...]]):
    def fake_run(command: object, capture_output: bool, text: bool, check: bool) -> subprocess.CompletedProcess[str]:
        _ = (capture_output, text, check)
        command_tuple = _command_tuple(command)
        calls.append(command_tuple)
        if "repair_agent.env.harness" in command_tuple:
            _write_blocked_harness_status()
            return subprocess.CompletedProcess(command_tuple, 1, stdout="", stderr="docker_cli_unavailable")
        _materialize_command_outputs(command_tuple)
        return subprocess.CompletedProcess(command_tuple, 0, stdout="ok", stderr="")

    return fake_run


def _write_blocked_harness_status() -> None:
    path = Path("outputs/harness_status.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "blocked_reason": "docker_cli_unavailable",
        "blockers": ["docker_cli_unavailable"],
        "official_harness_executed": False,
        "report_dir": "logs/run_evaluation/official_gold_smoke",
        "resolved": 0,
        "resolved_rate": 0.0,
        "status": "blocked",
        "total": 2,
    }
    _ = path.write_text(json.dumps(payload), encoding="utf-8")


def _write_resources(tmp_path: Path) -> Path:
    path = tmp_path / "configs" / "resources.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cpu": {"max_workers": 32},
        "device_policy": "maximize_local",
        "docker_cache_level": "env",
        "fallback": {"on_cpu_saturated": "wait_and_retry", "on_gpu_oom": "reduce_batch_and_retry", "on_gpu_unavailable": "record_and_continue"},
        "gpus": {"expected_ids": [0, 1, 2, 3], "per_device": {"fallback": None, "min_memory_mb": 4096}},
        "memory": {"per_swebench_worker_mb": 8192, "reserve_mb": 8192},
        "model_shards": {"max_gpus_per_model": 4, "per_gpu_batch_size": 1, "strategy": "device_map_auto"},
        "trainer_devices": {"policy_device": 0, "rollout_gpus": [0, 1, 2, 3], "rollout_parallelism": 4},
        "swebench_max_workers": "auto",
    }
    _ = path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _write_inventory(tmp_path: Path, gpus: list[int], *, workers: int) -> None:
    path = tmp_path / "outputs" / "device_inventory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gpus": [{"index": gpu_id, "memory_free_mb": 48000} for gpu_id in gpus],
        "swebench_workers": {"recommended_swebench_max_workers": workers},
    }
    _ = path.write_text(json.dumps(payload), encoding="utf-8")


def _write_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "configs" / "task_manifest.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": "princeton-nlp/SWE-bench_Lite",
        "split": "test",
        "smoke_ids": ["smoke__case-1", "smoke__case-2"],
        "main_ids": [f"main__case-{index}" for index in range(30)],
    }
    _ = path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _task_manifest_object():
    from repair_agent.env.swebench_loader import TaskManifest

    return TaskManifest(
        dataset_name="princeton-nlp/SWE-bench_Lite",
        split="test",
        smoke_ids=("smoke__case-1", "smoke__case-2"),
        main_ids=tuple(f"main__case-{index}" for index in range(30)),
    )


def _materialize_command_outputs(command: tuple[str, ...]) -> None:
    if "scripts/check_model_gate.py" in command:
        model = command[command.index("--model") + 1]
        status = "pass" if model == "qwable" else "blocked"
        gate_dir = Path("outputs/model_gates")
        gate_dir.mkdir(parents=True, exist_ok=True)
        _ = (gate_dir / f"{model}.json").write_text(json.dumps({"model": model, "status": status, "reason": f"{model}_unit"}), encoding="utf-8")
    elif "scripts/make_gold_smoke.py" in command:
        out_path = Path(command[command.index("--out") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"instance_id": "smoke__case-1", "model_name_or_path": "gold-smoke", "model_patch": "diff --git a/a b/a\n"},
            {"instance_id": "smoke__case-2", "model_name_or_path": "gold-smoke", "model_patch": "diff --git a/b b/b\n"},
        ]
        _ = out_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    elif "repair_agent.run" in command or "repair_agent.training.train" in command:
        run_id = command[command.index("--run-id") + 1]
        _write_run_artifacts(Path("outputs/runs") / run_id, completed=2, learning="repair_agent.training.train" in command)


def _write_run_artifacts(run_dir: Path, *, completed: int, learning: bool) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction = {"instance_id": f"{run_dir.name}-case", "model_name_or_path": "unit", "model_patch": "diff --git a/x b/x\n"}
    _ = (run_dir / "predictions.jsonl").write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    trajectory = {"instance_id": prediction["instance_id"], "run_id": run_dir.name, "run_name": "learning" if learning else run_dir.name, "status": "completed"}
    _ = (run_dir / "trajectories.jsonl").write_text(json.dumps(trajectory) + "\n", encoding="utf-8")
    _ = (run_dir / "metrics.json").write_text(json.dumps({"completed": completed, "instances": [{"instance_id": prediction["instance_id"]}], "skipped": 0, "total": completed}), encoding="utf-8")
    _ = (run_dir / "run_state.json").write_text(json.dumps({"completed_instances": [prediction["instance_id"]], "status": "completed"}), encoding="utf-8")
    if learning:
        _ = (run_dir / "policy.json").write_text(json.dumps({"schema": "safe-tool-selection-v1"}), encoding="utf-8")
        _ = (run_dir / "rewards.jsonl").write_text(json.dumps({"reward_total": 1.0}) + "\n", encoding="utf-8")


def _command_tuple(command: object) -> tuple[str, ...]:
    if isinstance(command, list | tuple):
        sequence = cast(Sequence[object], command)
        return tuple(str(item) for item in sequence)
    raise TypeError("fake subprocess command must be a sequence")
