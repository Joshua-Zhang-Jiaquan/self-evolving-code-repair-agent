"""Tests for Task 7 official SWE-bench harness execution and qz offload preparation.

The official harness cannot run locally (Docker and the ``swebench`` package are both
unavailable), so the harness wrapper must honestly record a blocked status for the gold
smoke set and for every agent stage, and the heavy official run must be offloaded to a
schema-validated, dry-run-only qz job that is never submitted without approval.

These tests monkeypatch the environment so the blocked path is deterministic regardless
of whether the host happens to have Docker, and they lock in the comprehensive status
summary plus the qz offload artifacts produced by Task 7.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import re
import shutil
from pathlib import Path
from typing import cast

import pytest

from repair_agent.env.harness import main as harness_main


PLACEHOLDER = "RESOLVE_BEFORE_SUBMISSION"
EXECUTION_BACKEND = "qz_pending_approval"
AGENT_STAGES = (
    "baseline_main",
    "feedback_main",
    "learning_main",
    "ablation_no_process_reward",
    "ablation_no_feedback_features",
    "ablation_reduced_test_budget",
)
SECTION_FIELDS = (
    "predictions",
    "resolved",
    "official_harness_executed",
    "blockers",
    "execution_backend",
)
# qz auth tokens are JWTs that start with ``eyJ`` (three base64url segments). No output
# file may ever contain one.
_JWT_PATTERN = re.compile(r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+")


def test_gold_smoke_harness_writes_blocked_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, project_root: Path
):
    _force_docker_unavailable(monkeypatch)
    predictions = project_root / "outputs" / "runs" / "gold_patch_smoke" / "predictions.jsonl"
    assert predictions.is_file()
    expected = _count_rows(predictions)
    status_path = tmp_path / "gold_status.json"

    result = _run_blocked_harness(predictions, "official_gold_smoke", status_path)
    status = _read_json(status_path)

    assert result == 0
    assert status["status"] == "blocked"
    assert status["official_harness_executed"] is False
    assert status["blocked_reason"] == "docker_cli_unavailable"
    assert "docker_cli_unavailable" in cast(list[str], status["blockers"])
    assert status["resolved"] == 0
    assert status["total"] == expected
    assert expected == 2


def test_agent_harness_writes_blocked_status_per_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, project_root: Path
):
    _force_docker_unavailable(monkeypatch)
    for stage in AGENT_STAGES:
        predictions = project_root / "outputs" / "runs" / stage / "predictions.jsonl"
        assert predictions.is_file(), f"missing predictions for {stage}"
        expected = _count_rows(predictions)
        status_path = tmp_path / f"{stage}_status.json"

        result = _run_blocked_harness(predictions, f"official_{stage}", status_path)
        status = _read_json(status_path)

        assert result == 0, stage
        assert status["status"] == "blocked", stage
        assert status["official_harness_executed"] is False, stage
        assert status["blocked_reason"] == "docker_cli_unavailable", stage
        assert status["resolved"] == 0, stage
        assert status["total"] == expected, stage
        assert expected == 40, stage


def test_qz_offload_spec_prepared(project_root: Path):
    job_path = project_root / "outputs" / "qz" / "official_harness_job.json"
    assert job_path.is_file()
    spec = _read_json(job_path)

    for field in ("name", "project_id", "workspace_id", "logic_compute_group_id", "framework"):
        assert spec[field] == PLACEHOLDER, field

    command = cast(str, spec["command"])
    assert command != PLACEHOLDER
    assert "repair_agent.env.harness" in command
    assert "--strict-official" in command

    framework_config = cast(list[dict[str, object]], spec["framework_config"])
    assert framework_config
    entry = framework_config[0]
    assert entry["image"] == PLACEHOLDER
    assert entry["image_type"] == PLACEHOLDER
    assert entry["spec_id"] == PLACEHOLDER
    assert entry["instance_count"] == 1


def test_qz_dry_run_executed(project_root: Path):
    dry_path = project_root / "outputs" / "qz" / "official_harness_dry_run.yaml"
    assert dry_path.is_file()
    content = dry_path.read_text(encoding="utf-8")

    assert "DRY RUN" in content
    assert "CreateJob" in content
    assert PLACEHOLDER in content
    assert _JWT_PATTERN.search(content) is None


def test_no_qz_submission_without_approval(project_root: Path):
    status = _read_json(project_root / "outputs" / "harness_status.json")
    qz_offload = cast(dict[str, object], status["qz_offload"])

    assert qz_offload["submitted"] is False
    assert status["execution_backend"] == EXECUTION_BACKEND
    # A dry-run-only artifact must not record a real cluster job id.
    dry_path = project_root / "outputs" / "qz" / "official_harness_dry_run.yaml"
    dry_text = dry_path.read_text(encoding="utf-8")
    assert "DRY RUN" in dry_text
    assert '"job_id"' not in dry_text


def test_comprehensive_harness_status_has_required_fields(project_root: Path):
    status = _read_json(project_root / "outputs" / "harness_status.json")

    assert status["official_harness_executed"] is False
    assert status["status"] == "blocked"
    assert status["execution_backend"] == EXECUTION_BACKEND
    blockers = cast(list[str], status["blockers"])
    assert "swebench_package_unavailable" in blockers
    assert "docker_cli_unavailable" in blockers

    gold = cast(dict[str, object], status["gold_smoke"])
    for field in SECTION_FIELDS:
        assert field in gold, field
    assert gold["predictions"] == 2
    assert gold["resolved"] == 0
    assert gold["official_harness_executed"] is False
    assert gold["execution_backend"] == EXECUTION_BACKEND

    agent_runs = cast(dict[str, object], status["agent_runs"])
    assert set(agent_runs.keys()) == set(AGENT_STAGES)
    for stage in AGENT_STAGES:
        section = cast(dict[str, object], agent_runs[stage])
        for field in SECTION_FIELDS:
            assert field in section, (stage, field)
        assert section["predictions"] == 40, stage
        assert section["resolved"] == 0, stage
        assert section["official_harness_executed"] is False, stage
        assert section["execution_backend"] == EXECUTION_BACKEND, stage

    qz_offload = cast(dict[str, object], status["qz_offload"])
    for field in ("available", "schema_path", "job_spec", "dry_run", "submitted"):
        assert field in qz_offload, field
    assert qz_offload["available"] is True
    assert qz_offload["submitted"] is False
    assert qz_offload["job_spec"] == "outputs/qz/official_harness_job.json"
    assert qz_offload["dry_run"] == "outputs/qz/official_harness_dry_run.yaml"


def test_no_token_leakage(project_root: Path):
    outputs_dir = project_root / "outputs"
    critical = [
        outputs_dir / "qz" / "official_harness_job.json",
        outputs_dir / "qz" / "official_harness_dry_run.yaml",
        outputs_dir / "harness_status.json",
    ]
    for path in critical:
        assert path.is_file(), path

    text_suffixes = {".json", ".yaml", ".yml", ".txt"}
    scanned: list[Path] = []
    for path in sorted(outputs_dir.rglob("*")):
        if path.is_file() and path.suffix in text_suffixes:
            text = path.read_text(encoding="utf-8", errors="replace")
            assert _JWT_PATTERN.search(text) is None, f"JWT token leaked in {path}"
            scanned.append(path)

    for path in critical:
        assert path in scanned, f"{path} was not scanned for token leakage"


def _force_docker_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``swebench`` importable but the docker CLI absent -> docker_cli_unavailable."""
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


def _run_blocked_harness(predictions: Path, run_id: str, status_out: Path) -> int:
    return harness_main(
        [
            "--predictions",
            str(predictions),
            "--run-id",
            run_id,
            "--max-workers",
            "1",
            "--status-out",
            str(status_out),
        ]
    )


def _count_rows(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _read_json(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
