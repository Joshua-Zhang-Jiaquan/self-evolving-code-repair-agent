from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
import yaml

from repair_agent.env import harness as harness_module
from repair_agent.env.harness import main as harness_main
from repair_agent.logging import read_json_object
from scripts.run_gated_experiments import PipelineArgs, run_from_args


def test_strict_schedule_has_no_simulate_docker_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    schedule = _strict_schedule(tmp_path)
    assignments = cast(list[dict[str, object]], schedule["assignments"])
    commands = "\n".join(cast(str, assignment["command_line"]) for assignment in assignments)

    assert "--simulate-docker-failure" not in commands
    assert "repair_agent.env.harness" in commands


def test_strict_schedule_has_timeout_1800(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    schedule = _strict_schedule(tmp_path)
    harness_line = _harness_command_line(schedule)

    assert "--timeout-seconds 1800" in harness_line
    assert "--run-id official_gold_smoke" in harness_line
    assert "--simulate-docker-failure" not in harness_line


def test_missing_docker_is_not_success_in_strict_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    _ = predictions.write_text(
        '{"instance_id":"case__one","model_name_or_path":"unit","model_patch":""}\n',
        encoding="utf-8",
    )
    status_path = tmp_path / "harness_status.json"
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package: str | None = None) -> importlib.machinery.ModuleSpec | None:
        if name == "swebench":
            return importlib.machinery.ModuleSpec("swebench", None)
        return real_find_spec(name, package)

    def fake_which(name: str) -> str | None:
        _ = name
        return None

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(shutil, "which", fake_which)

    result = harness_main(
        [
            "--predictions",
            str(predictions),
            "--run-id",
            "official_gold_smoke",
            "--max-workers",
            "1",
            "--strict-official",
            "--status-out",
            str(status_path),
        ]
    )
    status = cast(dict[str, object], json.loads(status_path.read_text(encoding="utf-8")))

    assert result != 0
    assert status["status"] == "blocked"
    assert status["official_harness_executed"] is False
    assert status["blocked_reason"] == "docker_cli_unavailable"
    assert "docker_cli_unavailable" in cast(list[str], status["blockers"])


def test_harness_status_has_required_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    prediction_rows = [
        '{"instance_id":"case__1","model_name_or_path":"unit","model_patch":"diff"}',
        '{"instance_id":"case__2","model_name_or_path":"unit","model_patch":"diff"}',
    ]
    predictions = tmp_path / "predictions.jsonl"
    _ = predictions.write_text("\n".join(prediction_rows) + "\n", encoding="utf-8")
    status_path = tmp_path / "harness_status.json"

    def never_blocked(args: object) -> str | None:
        _ = args
        return None

    def fake_run(command: object, capture_output: bool, text: bool, timeout: int, check: bool) -> subprocess.CompletedProcess[str]:
        _ = (capture_output, text, timeout, check)
        cmd = _command_list(command)
        run_id = cmd[cmd.index("--run_id") + 1]
        report = Path(f"unit.{run_id}.json")
        _ = report.write_text(json.dumps({"resolved_instances": 1, "total_instances": 2}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="done", stderr="")

    monkeypatch.setattr(harness_module, "_blocked_reason", never_blocked)
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = harness_main(
        [
            "--predictions",
            str(predictions),
            "--run-id",
            "official_gold_smoke",
            "--max-workers",
            "4",
            "--status-out",
            str(status_path),
        ]
    )
    status = cast(dict[str, object], json.loads(status_path.read_text(encoding="utf-8")))

    for field in ("official_harness_executed", "status", "resolved", "total", "resolved_rate", "report_dir", "blockers"):
        assert field in status
    assert result == 0
    assert status["official_harness_executed"] is True
    assert status["status"] == "completed"
    assert status["resolved"] == 1
    assert status["total"] == 2
    assert status["resolved_rate"] == 0.5
    assert status["blockers"] == []
    assert "official_gold_smoke" in cast(str, status["report_dir"])
    command = cast(list[str], status["command"])
    assert command[command.index("--namespace") + 1] == "none"
    assert command[command.index("--instance_ids") + 1 :] == ["case__1", "case__2"]


def test_strict_mode_fails_when_official_execution_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    _ = predictions.write_text(
        '{"instance_id":"case__one","model_name_or_path":"unit","model_patch":"diff"}\n',
        encoding="utf-8",
    )
    status_path = tmp_path / "harness_status.json"

    def never_blocked(args: object) -> str | None:
        _ = args
        return None

    def failing_run(command: object, capture_output: bool, text: bool, timeout: int, check: bool) -> subprocess.CompletedProcess[str]:
        _ = (capture_output, text, timeout, check)
        return subprocess.CompletedProcess(_command_list(command), 1, stdout="", stderr="unshare: operation not permitted")

    monkeypatch.setattr(harness_module, "_blocked_reason", never_blocked)
    monkeypatch.setattr(subprocess, "run", failing_run)

    result = harness_main(
        [
            "--predictions",
            str(predictions),
            "--run-id",
            "official_failure",
            "--max-workers",
            "1",
            "--strict-official",
            "--status-out",
            str(status_path),
        ]
    )
    status = cast(dict[str, object], json.loads(status_path.read_text(encoding="utf-8")))

    assert result == 1
    assert status["status"] == "fallback"
    assert status["official_harness_executed"] is True
    assert status["fallback_reason"] == "official_harness_returned_1"


def test_prediction_instance_ids_must_be_safe(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    predictions = tmp_path / "predictions.jsonl"
    _ = predictions.write_text(
        '{"instance_id":"--cache_level","model_name_or_path":"unit","model_patch":"diff"}\n',
        encoding="utf-8",
    )
    status_path = tmp_path / "harness_status.json"

    result = harness_main(
        [
            "--predictions",
            str(predictions),
            "--run-id",
            "bad_ids",
            "--max-workers",
            "1",
            "--status-out",
            str(status_path),
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert "invalid prediction instance_id" in captured.err


def test_non_strict_mode_still_supports_simulate(tmp_path: Path):
    predictions = tmp_path / "predictions.jsonl"
    _ = predictions.write_text(
        '{"instance_id":"case__one","model_name_or_path":"unit","model_patch":""}\n',
        encoding="utf-8",
    )
    status_path = tmp_path / "harness_status.json"

    result = harness_main(
        [
            "--predictions",
            str(predictions),
            "--run-id",
            "task13_gold_patch_smoke",
            "--max-workers",
            "1",
            "--simulate-docker-failure",
            "--status-out",
            str(status_path),
        ]
    )
    status = cast(dict[str, object], json.loads(status_path.read_text(encoding="utf-8")))

    assert result == 0
    assert status["status"] == "blocked"
    assert status["blocked_reason"] == "simulated_docker_failure"
    assert status["official_harness_executed"] is False
    assert "simulated_docker_failure" in cast(list[str], status["blockers"])
    command = cast(list[str], status["command"])
    assert "swebench.harness.run_evaluation" in command


def _strict_schedule(tmp_path: Path) -> dict[str, object]:
    resources = _write_resources(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    _write_inventory(tmp_path, [0, 1, 2, 3], workers=16)
    result = run_from_args(
        PipelineArgs(
            manifest=manifest_path,
            out=Path("outputs/runs"),
            resources=resources,
            dry_run_schedule=True,
            strict_official=True,
        )
    )
    assert result == 0
    return read_json_object("outputs/run_schedule.json")


def _harness_command_line(schedule: dict[str, object]) -> str:
    assignments = cast(list[dict[str, object]], schedule["assignments"])
    for assignment in assignments:
        if assignment.get("kind") == "harness_status":
            return cast(str, assignment["command_line"])
    raise AssertionError("schedule has no harness_status stage")


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


def _command_list(command: object) -> list[str]:
    if isinstance(command, list | tuple):
        sequence = cast(Sequence[object], command)
        return [str(item) for item in sequence]
    raise TypeError("fake subprocess command must be a sequence")
