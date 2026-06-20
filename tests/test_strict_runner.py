from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import pytest
import yaml

from repair_agent.logging import read_jsonl
from repair_agent.run import main


def _fake_official_row(instance_id: str) -> dict[str, object]:
    org = instance_id.split("__", 1)[0]
    repo = f"{org}/repo"
    base_commit = f"basecommit-{instance_id}"
    return {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": base_commit,
        "problem_statement": f"Official problem statement for {instance_id}.",
        "hints_text": "",
        "visible_test_metadata": {
            "oracle_tests_hidden": True,
            "fail_to_pass_count": 1,
            "pass_to_pass_count": 0,
        },
        "workspace_setup": {
            "repo": repo,
            "base_commit": base_commit,
            "environment_setup_commit": "",
            "version": "1.0",
            "created_at": "2020-01-01T00:00:00Z",
        },
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


def _write_config(tmp_path: Path, *, agent_type: str = "baseline") -> Path:
    config = {
        "agent": {
            "type": agent_type,
            "model_name_or_path": "rule_based_local",
            "max_steps": 12,
            "max_test_runs": 1,
            "test_timeout_seconds": 5.0,
            "max_output_chars": 4000,
            "instances": [
                {
                    "instance_id": "baseline-local-0001",
                    "repo": "local/baseline-fixture",
                    "problem_statement": "Config fixture that strict mode must ignore; add two numbers.",
                    "visible_tests": ["tests/test_math_utils.py"],
                    "visible_failures": {"visible-failure": "AssertionError: add_numbers(2, 3) should equal 5"},
                    "fixture": {
                        "files": {
                            "README.md": "baseline fixture\n",
                            "math_utils.py": "def add_numbers(left, right):\n    return left - right\n",
                            "tests/test_math_utils.py": "from math_utils import add_numbers\n\n\ndef test_add_numbers_visible():\n    assert add_numbers(2, 3) == 5\n",
                        }
                    },
                }
            ],
        },
        "dry_run": {"instances": [{"instance_id": "dry-0001", "repo": "local/dry"}]},
        "run": {"name": "strict-runner-test", "output_dir": str(tmp_path / "runs")},
    }
    path = tmp_path / "config.yaml"
    _ = path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return path


def _manifest_path(project_root: Path) -> Path:
    return project_root / "configs" / "task_manifest.yaml"


def test_strict_bridge_smoke_uses_official_ids(
    monkeypatch: pytest.MonkeyPatch, project_root: Path, tmp_path: Path
):
    config_path = _write_config(tmp_path)
    official_ids = ["astropy__astropy-12907", "django__django-10914", "sympy__sympy-11870"]
    _patch_official_loader(monkeypatch, [_fake_official_row(instance_id) for instance_id in official_ids])

    argv = [
        "--config", str(config_path),
        "--manifest", str(_manifest_path(project_root)),
        "--instance-split", "main",
        "--strict-official",
        "--limit", "2",
        "--run-id", "strict_smoke",
        "--force",
    ]
    assert main(argv) == 0

    run_dir = tmp_path / "runs" / "strict_smoke"
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    prediction_ids = [str(row["instance_id"]) for row in predictions]

    assert prediction_ids == official_ids[:2]
    assert all("__" in instance_id for instance_id in prediction_ids)
    assert all("local-" not in instance_id for instance_id in prediction_ids)
    assert "baseline-local-0001" not in prediction_ids
    for row in predictions:
        assert row["model_name_or_path"] == "rule_based_local"
        assert isinstance(row["model_patch"], str)


def test_strict_mode_refuses_config_fixtures(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    config_path = _write_config(tmp_path)
    argv = [
        "--config", str(config_path),
        "--strict-official",
        "--run-id", "strict_no_manifest",
        "--force",
    ]

    assert main(argv) == 2
    captured = capsys.readouterr()
    assert "strict_official_requires_manifest" in captured.err

    run_dir = tmp_path / "runs" / "strict_no_manifest"
    assert read_jsonl(run_dir / "predictions.jsonl") == []


def test_strict_mode_rejects_fixture_ids(
    monkeypatch: pytest.MonkeyPatch, project_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    config_path = _write_config(tmp_path)
    for index, bad_id in enumerate(("baseline-local-0001", "no-double-underscore")):
        _patch_official_loader(monkeypatch, [_fake_official_row(bad_id)])
        argv = [
            "--config", str(config_path),
            "--manifest", str(_manifest_path(project_root)),
            "--instance-split", "main",
            "--strict-official",
            "--run-id", f"strict_reject_{index}",
            "--force",
        ]

        assert main(argv) == 2
        captured = capsys.readouterr()
        assert "strict_official_rejects_fixture_id" in captured.err
        run_dir = tmp_path / "runs" / f"strict_reject_{index}"
        assert read_jsonl(run_dir / "predictions.jsonl") == []


def test_strict_metadata_records_official_source(
    monkeypatch: pytest.MonkeyPatch, project_root: Path, tmp_path: Path
):
    config_path = _write_config(tmp_path)
    official_ids = ["astropy__astropy-12907", "django__django-10914"]
    _patch_official_loader(monkeypatch, [_fake_official_row(instance_id) for instance_id in official_ids])

    argv = [
        "--config", str(config_path),
        "--manifest", str(_manifest_path(project_root)),
        "--instance-split", "main",
        "--strict-official",
        "--limit", "2",
        "--run-id", "strict_meta",
        "--force",
    ]
    assert main(argv) == 0

    run_dir = tmp_path / "runs" / "strict_meta"
    state = cast(dict[str, object], json.loads((run_dir / "run_state.json").read_text(encoding="utf-8")))
    metadata = cast(dict[str, object], state["metadata"])

    assert metadata["strict_official"] is True
    assert metadata["official_instance_source"] == "swebench_lite"
    assert metadata["instance_count"] == 2
    assert metadata["instance_split"] == "main"
    assert state["dry_run"] is False


def test_non_strict_mode_still_uses_config_instances(tmp_path: Path):
    config_path = _write_config(tmp_path)
    argv = [
        "--config", str(config_path),
        "--limit", "1",
        "--run-id", "non_strict",
        "--force",
    ]
    assert main(argv) == 0

    run_dir = tmp_path / "runs" / "non_strict"
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    assert [str(row["instance_id"]) for row in predictions] == ["baseline-local-0001"]

    state = cast(dict[str, object], json.loads((run_dir / "run_state.json").read_text(encoding="utf-8")))
    assert "metadata" not in state
