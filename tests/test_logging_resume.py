from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from repair_agent.config import ConfigError, load_run_config
from repair_agent.logging import read_jsonl
from repair_agent.run import main


def write_dry_config(tmp_path: Path) -> Path:
    path = tmp_path / "baseline.yaml"
    _ = path.write_text(
        yaml.safe_dump(
            {
                "dry_run": {
                    "instances": [
                        {"instance_id": "case-1", "repo": "local/repo"},
                        {"instance_id": "case-2", "repo": "local/repo"},
                    ]
                },
                "run": {"name": "baseline-test", "output_dir": str(tmp_path / "runs")},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_dry_run_creates_artifacts_and_resume_skips_duplicate_rows(tmp_path: Path):
    config_path = write_dry_config(tmp_path)
    argv = ["--config", str(config_path), "--dry-run", "--limit", "1", "--run-id", "resume"]

    assert main(argv) == 0
    run_dir = tmp_path / "runs" / "resume"
    expected_files = [
        "config.yaml",
        "trajectories.jsonl",
        "predictions.jsonl",
        "metrics.json",
        "run_state.json",
    ]
    for filename in expected_files:
        assert (run_dir / filename).is_file(), f"missing run artifact {filename}"

    first_trajectories = read_jsonl(run_dir / "trajectories.jsonl")
    first_predictions = read_jsonl(run_dir / "predictions.jsonl")
    assert [row["instance_id"] for row in first_trajectories] == ["case-1"]
    assert [row["model_name_or_path"] for row in first_predictions] == ["dry-run"]

    assert main(argv) == 0
    second_trajectories = read_jsonl(run_dir / "trajectories.jsonl")
    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert len(second_trajectories) == 1
    assert state["completed_instances"] == ["case-1"]
    assert metrics["skipped"] == 1
    assert metrics["newly_completed"] == 0


def test_force_resets_existing_dry_run_rows(tmp_path: Path):
    config_path = write_dry_config(tmp_path)
    argv = ["--config", str(config_path), "--dry-run", "--limit", "1", "--run-id", "forced"]
    assert main(argv) == 0
    force_argv = [*argv, "--force"]
    assert main(force_argv) == 0

    run_dir = tmp_path / "runs" / "forced"
    trajectories = read_jsonl(run_dir / "trajectories.jsonl")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert len(trajectories) == 1
    assert metrics["newly_completed"] == 1


def test_config_errors_name_missing_dry_run_instances(tmp_path: Path):
    bad_config = tmp_path / "bad.yaml"
    _ = bad_config.write_text("run:\n  name: broken\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="dry_run"):
        load_run_config(bad_config)
