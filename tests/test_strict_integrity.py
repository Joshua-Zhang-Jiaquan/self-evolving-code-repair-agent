from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from scripts.check_strict_integrity import main


def _read_json_object(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


REAL_FORBIDDEN_PATHS = (
    "homework1",
    "homework2",
    "homework3",
    "homework4",
    "submissions",
    ".venv",
    "/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/DiffRWKV-RELAY/releases/traj32x16-2.9B-s2-rwkv7-v3-ddpm",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(text, encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_valid_layout(root: Path) -> None:
    """Create a minimal artifact tree that satisfies all eight integrity checks."""
    outputs = root / "outputs"
    runs = outputs / "runs"

    manifest = {
        "strict_official": True,
        "model_gates": {
            "qwable": {"status": "pass", "details": {"dry_run": False}},
            "diffrwkv": {"status": "blocked", "details": {"checkpoint": "traj32x16-2.9B-s2-rwkv7-v3-ddpm"}},
        },
        "stages": [
            {
                "stage_id": "baseline_main",
                "run_id": "baseline_main",
                "status": "completed",
                "command": ["python", "-m", "repair_agent.run", "--instance-split", "main", "--strict-official"],
            },
            {
                "stage_id": "official_harness_status",
                "run_id": None,
                "status": "blocked",
                "command": ["python", "-m", "repair_agent.env.harness", "--strict-official"],
            },
        ],
    }
    _write_json(outputs / "run_manifest.json", manifest)
    _write_json(outputs / "summary.json", {"aggregate": {"run_count": 1}})
    _write_json(
        outputs / "harness_status.json",
        {
            "official_harness_executed": False,
            "status": "blocked",
            "blockers": ["swebench_package_unavailable", "docker_cli_unavailable"],
        },
    )
    _write_json(outputs / "model_gates" / "qwable.json", {"status": "pass", "details": {"dry_run": False}})
    _write_json(outputs / "run_schedule.json", {"assignments": [{"command": ["python", "-m", "repair_agent.env.harness", "--strict-official"]}]})

    _write_jsonl(
        runs / "baseline_main" / "predictions.jsonl",
        [
            {"instance_id": "astropy__astropy-12907", "model_name_or_path": "rule_based_local", "model_patch": ""},
            {"instance_id": "django__django-10914", "model_name_or_path": "rule_based_local", "model_patch": ""},
        ],
    )
    _write_jsonl(
        runs / "baseline_main" / "trajectories.jsonl",
        [
            {"instance_id": "astropy__astropy-12907", "tool": "search", "patch_path": "x.patch", "patch_sha256": "deadbeef"},
        ],
    )

    _write_text(root / "README.md", "# Repair Agent\nSafe strict official commands only.\n")
    _write_text(root / "report" / "report.md", "# Report\nStrict official mode, blocked harness.\n")
    _write_json(root / "report" / "figures" / "results.json", {"aggregate": {"run_count": 1}})
    _write_text(root / "report" / "tables" / "results_table.md", "| Run |\n|---|\n")
    _write_text(root / "report" / "tables" / "ablation_comparison.md", "| Run |\n|---|\n")
    _write_text(root / "report" / "tables" / "device_utilization.md", "| Run |\n|---|\n")


def _argv(root: Path, forbidden: tuple[str, ...] = REAL_FORBIDDEN_PATHS) -> list[str]:
    outputs = root / "outputs"
    argv = [
        "--manifest", str(outputs / "run_manifest.json"),
        "--summary", str(outputs / "summary.json"),
        "--harness", str(outputs / "harness_status.json"),
        "--qwable", str(outputs / "model_gates" / "qwable.json"),
    ]
    for path in forbidden:
        argv.extend(["--forbidden-path", path])
    return argv


def test_valid_layout_passes(tmp_path: Path):
    _build_valid_layout(tmp_path)
    assert main(_argv(tmp_path)) == 0


def test_catches_fixture_instance_ids(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _build_valid_layout(tmp_path)
    _write_jsonl(
        tmp_path / "outputs" / "runs" / "baseline_main" / "predictions.jsonl",
        [{"instance_id": "baseline-local-0001", "model_name_or_path": "rule_based_local", "model_patch": ""}],
    )
    assert main(_argv(tmp_path)) == 1
    captured = capsys.readouterr()
    assert "fixture-style instance_id" in captured.err
    assert "baseline-local-0001" in captured.err


def test_catches_missing_official_separator(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _build_valid_layout(tmp_path)
    _write_jsonl(
        tmp_path / "outputs" / "runs" / "baseline_main" / "predictions.jsonl",
        [{"instance_id": "astropy-12907", "model_name_or_path": "rule_based_local", "model_patch": ""}],
    )
    assert main(_argv(tmp_path)) == 1
    assert "fixture-style instance_id" in capsys.readouterr().err


def test_catches_skipped_existing_strict_stage(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _build_valid_layout(tmp_path)
    manifest_path = tmp_path / "outputs" / "run_manifest.json"
    manifest = _read_json_object(manifest_path)
    stages = cast(list[dict[str, object]], manifest["stages"])
    stages[0]["status"] = "skipped_existing"
    _write_json(manifest_path, manifest)
    assert main(_argv(tmp_path)) == 1
    assert "skipped_existing" in capsys.readouterr().err


def test_catches_simulate_docker_failure_in_schedule(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _build_valid_layout(tmp_path)
    _write_json(
        tmp_path / "outputs" / "run_schedule.json",
        {"assignments": [{"command": ["python", "-m", "repair_agent.env.harness", "--simulate-docker-failure"]}]},
    )
    assert main(_argv(tmp_path)) == 1
    captured = capsys.readouterr()
    assert "--simulate-docker-failure" in captured.err
    assert "run schedule" in captured.err


def test_catches_simulate_docker_failure_in_readme(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _build_valid_layout(tmp_path)
    _write_text(tmp_path / "README.md", "# Repair Agent\nrun with --simulate-docker-failure\n")
    assert main(_argv(tmp_path)) == 1
    assert "README" in capsys.readouterr().err


def test_catches_dry_run_qwable(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _build_valid_layout(tmp_path)
    _write_json(
        tmp_path / "outputs" / "model_gates" / "qwable.json",
        {"status": "pass", "details": {"dry_run": True}},
    )
    assert main(_argv(tmp_path)) == 1
    captured = capsys.readouterr()
    assert "dry_run" in captured.err
    assert "real inference" in captured.err


def test_catches_harness_not_executed_without_blockers(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _build_valid_layout(tmp_path)
    _write_json(
        tmp_path / "outputs" / "harness_status.json",
        {"official_harness_executed": False, "status": "blocked", "blockers": []},
    )
    assert main(_argv(tmp_path)) == 1
    assert "official_harness_executed" in capsys.readouterr().err


def test_harness_executed_true_passes(tmp_path: Path):
    _build_valid_layout(tmp_path)
    _write_json(
        tmp_path / "outputs" / "harness_status.json",
        {"official_harness_executed": True, "status": "completed", "resolved": 3, "total": 40, "blockers": []},
    )
    assert main(_argv(tmp_path)) == 0


def test_catches_forbidden_path_in_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _build_valid_layout(tmp_path)
    manifest_path = tmp_path / "outputs" / "run_manifest.json"
    manifest = _read_json_object(manifest_path)
    forbidden = REAL_FORBIDDEN_PATHS[-1]
    gates = cast(dict[str, dict[str, dict[str, object]]], manifest["model_gates"])
    gates["diffrwkv"]["details"]["checkpoint"] = forbidden
    _write_json(manifest_path, manifest)
    assert main(_argv(tmp_path)) == 1
    captured = capsys.readouterr()
    assert "forbidden path" in captured.err
    assert "run manifest" in captured.err


def test_catches_forbidden_path_in_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _build_valid_layout(tmp_path)
    _write_json(tmp_path / "outputs" / "summary.json", {"note": "wrote homework1/leak.json by mistake"})
    assert main(_argv(tmp_path)) == 1
    assert "forbidden path" in capsys.readouterr().err


def test_catches_hidden_patch_key_in_trajectory(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _build_valid_layout(tmp_path)
    _write_jsonl(
        tmp_path / "outputs" / "runs" / "baseline_main" / "trajectories.jsonl",
        [{"instance_id": "astropy__astropy-12907", "metadata": {"patch": "--- a/x\n+++ b/x\n"}}],
    )
    assert main(_argv(tmp_path)) == 1
    captured = capsys.readouterr()
    assert "hidden gold key" in captured.err
    assert "patch" in captured.err


def test_catches_hidden_test_patch_key_in_trajectory(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _build_valid_layout(tmp_path)
    _write_jsonl(
        tmp_path / "outputs" / "runs" / "baseline_main" / "trajectories.jsonl",
        [{"instance_id": "astropy__astropy-12907", "test_patch": "gold-test-patch"}],
    )
    assert main(_argv(tmp_path)) == 1
    assert "test_patch" in capsys.readouterr().err


def test_patch_path_key_is_not_flagged(tmp_path: Path):
    _build_valid_layout(tmp_path)
    _write_jsonl(
        tmp_path / "outputs" / "runs" / "baseline_main" / "trajectories.jsonl",
        [{"instance_id": "astropy__astropy-12907", "patch_path": "p.patch", "patch_sha256": "abc123"}],
    )
    assert main(_argv(tmp_path)) == 0


def test_catches_missing_result_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _build_valid_layout(tmp_path)
    (tmp_path / "report" / "tables" / "device_utilization.md").unlink()
    assert main(_argv(tmp_path)) == 1
    captured = capsys.readouterr()
    assert "missing required result file" in captured.err
    assert "device_utilization.md" in captured.err


def test_archived_runs_are_ignored(tmp_path: Path):
    _build_valid_layout(tmp_path)
    _write_jsonl(
        tmp_path / "outputs" / "runs" / "baseline_main.archived.20260619T173515Z" / "predictions.jsonl",
        [{"instance_id": "baseline-local-0001", "model_patch": ""}],
    )
    _write_jsonl(
        tmp_path / "outputs" / "runs" / "baseline_main.archived.20260619T173515Z" / "trajectories.jsonl",
        [{"instance_id": "baseline-local-0001", "patch": "leaked-but-archived"}],
    )
    assert main(_argv(tmp_path)) == 0


def test_real_artifacts_pass_with_deliverable_cli(project_root: Path):
    manifest = project_root / "outputs" / "run_manifest.json"
    if not manifest.is_file():
        pytest.skip("real outputs/run_manifest.json not present in this environment")
    argv = [
        "--manifest", str(manifest),
        "--summary", str(project_root / "outputs" / "summary.json"),
        "--harness", str(project_root / "outputs" / "harness_status.json"),
        "--qwable", str(project_root / "outputs" / "model_gates" / "qwable.json"),
    ]
    for path in REAL_FORBIDDEN_PATHS:
        argv.extend(["--forbidden-path", path])
    assert main(argv) == 0
