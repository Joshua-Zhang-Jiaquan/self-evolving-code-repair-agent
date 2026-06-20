from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

import pytest
import yaml

from repair_agent.agent.interface import AgentFinalAnswer
from repair_agent.evaluation.metrics import REQUIRED_PREDICTION_KEYS, validate_prediction_row
from repair_agent.logging import read_jsonl
from repair_agent.run import main
from scripts.validate_predictions import validate_predictions_file


def _fake_official_row(instance_id: str) -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "repo": f"{instance_id.split('__', 1)[0]}/repo",
        "problem_statement": f"Official problem statement for {instance_id}.",
        "source": "swebench_lite_official",
        "model_patch": "",
    }


def _patch_official_loader(
    monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]]
) -> None:
    def fake_load_task_instances(
        manifest_path: str | Path,
        split: str = "test",
        ids: Iterable[str] | None = None,
        strict: bool = True,
    ) -> dict[str, list[dict[str, object]]]:
        _ = (manifest_path, split, ids, strict)
        return {"requested": [dict(row) for row in rows]}

    monkeypatch.setattr("repair_agent.run.load_task_instances", fake_load_task_instances)


def _write_strict_config(tmp_path: Path, *, agent_type: str) -> Path:
    config = {
        "agent": {
            "type": agent_type,
            "model_name_or_path": "rule_based_local",
            "max_steps": 12,
            "max_test_runs": 1,
            "test_timeout_seconds": 5.0,
            "max_output_chars": 4000,
        },
        "dry_run": {"instances": [{"instance_id": "dry-0001", "repo": "local/dry"}]},
        "run": {"name": "prediction-validation-test", "output_dir": str(tmp_path / "runs")},
    }
    path = tmp_path / f"{agent_type}.yaml"
    _ = path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return path


def _run_strict(
    monkeypatch: pytest.MonkeyPatch,
    project_root: Path,
    tmp_path: Path,
    *,
    agent_type: str,
    run_id: str,
    official_ids: list[str],
) -> Path:
    config_path = _write_strict_config(tmp_path, agent_type=agent_type)
    _patch_official_loader(monkeypatch, [_fake_official_row(instance_id) for instance_id in official_ids])
    argv = [
        "--config", str(config_path),
        "--manifest", str(project_root / "configs" / "task_manifest.yaml"),
        "--instance-split", "main",
        "--strict-official",
        "--limit", str(len(official_ids)),
        "--run-id", run_id,
        "--force",
    ]
    assert main(argv) == 0
    return tmp_path / "runs" / run_id


def test_baseline_strict_predictions_validate_as_official_rows(
    monkeypatch: pytest.MonkeyPatch, project_root: Path, tmp_path: Path
):
    official_ids = ["astropy__astropy-12907", "django__django-10914"]
    run_dir = _run_strict(
        monkeypatch, project_root, tmp_path,
        agent_type="baseline", run_id="pred_baseline", official_ids=official_ids,
    )

    summary = validate_predictions_file(run_dir / "predictions.jsonl")
    assert summary.row_count == 2
    assert summary.instance_ids == tuple(official_ids)
    assert all("__" in instance_id for instance_id in summary.instance_ids)
    assert all("local-" not in instance_id for instance_id in summary.instance_ids)


def test_feedback_strict_predictions_validate_as_official_rows(
    monkeypatch: pytest.MonkeyPatch, project_root: Path, tmp_path: Path
):
    official_ids = ["sympy__sympy-11870", "pallets__flask-4045"]
    run_dir = _run_strict(
        monkeypatch, project_root, tmp_path,
        agent_type="feedback", run_id="pred_feedback", official_ids=official_ids,
    )

    summary = validate_predictions_file(run_dir / "predictions.jsonl")
    assert summary.row_count == 2
    assert summary.instance_ids == tuple(official_ids)


def test_strict_prediction_rows_have_only_official_required_keys(
    monkeypatch: pytest.MonkeyPatch, project_root: Path, tmp_path: Path
):
    official_ids = ["astropy__astropy-12907", "django__django-10914"]
    run_dir = _run_strict(
        monkeypatch, project_root, tmp_path,
        agent_type="baseline", run_id="pred_keys", official_ids=official_ids,
    )

    rows = read_jsonl(run_dir / "predictions.jsonl")
    assert len(rows) == 2
    for row in rows:
        assert set(row.keys()) == set(REQUIRED_PREDICTION_KEYS)
        assert isinstance(row["instance_id"], str)
        assert "__" in str(row["instance_id"])
        assert isinstance(row["model_name_or_path"], str)
        assert isinstance(row["model_patch"], str)


def test_empty_model_patch_is_valid_prediction_row():
    final = AgentFinalAnswer(
        instance_id="astropy__astropy-12907",
        model_name_or_path="rule_based_local",
        model_patch="",
        status="no_patch",
        explanation="agent produced no patch on empty official checkout",
    )

    row = final.prediction_row()

    validate_prediction_row(cast("Mapping[object, object]", row), 1)
    assert set(row.keys()) == set(REQUIRED_PREDICTION_KEYS)
    assert row["instance_id"] == "astropy__astropy-12907"
    assert row["model_patch"] == ""
