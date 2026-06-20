from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import yaml
import pytest

from repair_agent.config import ConfigError
from repair_agent.env.harness import main as harness_main
from repair_agent.env.swebench_loader import (
    agent_records,
    load_dataset_gold_patches,
    load_gold_patch_source,
    load_manifest_records,
    load_task_manifest,
)
from scripts.make_gold_smoke import write_gold_smoke_predictions
from scripts.validate_predictions import validate_predictions_file


def test_manifest_shape_has_fixed_smoke_and_main_ids(project_root: Path):
    manifest_path = project_root / "configs" / "task_manifest.yaml"
    manifest = load_task_manifest(manifest_path)
    raw = cast(dict[str, object], yaml.safe_load(manifest_path.read_text(encoding="utf-8")))

    assert raw["dataset_name"] == "princeton-nlp/SWE-bench_Lite"
    assert raw["split"] == "test"
    assert 1 <= len(manifest.smoke_ids) <= 3
    assert 30 <= len(manifest.main_ids) <= 50
    assert len(set(manifest.all_ids)) == len(manifest.all_ids)
    assert "smoke_gold_patches" not in raw


def test_loader_keeps_gold_patch_out_of_agent_records(project_root: Path):
    manifest = load_task_manifest(project_root / "configs" / "task_manifest.yaml")
    records = load_manifest_records(project_root / "configs" / "task_manifest.yaml", include_main=False)
    visible_rows = agent_records(records)
    encoded = json.dumps(visible_rows, sort_keys=True)

    assert len(visible_rows) == len(manifest.smoke_ids)
    assert "patch" not in visible_rows[0]
    assert "test_patch" not in visible_rows[0]
    for record in records:
        assert record.gold is None
    assert "model_patch" not in encoded


def test_validate_predictions_accepts_official_rows_and_rejects_bad_types(tmp_path: Path):
    good_path = tmp_path / "predictions.jsonl"
    _ = good_path.write_text(
        json.dumps(
            {
                "instance_id": "django__django-11099",
                "model_name_or_path": "unit-test",
                "model_patch": "diff --git a/a.py b/a.py\n",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = validate_predictions_file(good_path)

    assert summary.row_count == 1
    assert summary.instance_ids == ("django__django-11099",)

    bad_path = tmp_path / "bad.jsonl"
    _ = bad_path.write_text(
        json.dumps(
            {"instance_id": "bad", "model_name_or_path": "unit-test", "model_patch": 3},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        _ = validate_predictions_file(bad_path)
    except ValueError as exc:
        assert "model_patch" in str(exc)
    else:
        raise AssertionError("non-string model_patch must be rejected")


def test_dataset_gold_loader_collects_requested_patch_rows(project_root: Path):
    manifest = load_task_manifest(project_root / "configs" / "task_manifest.yaml")
    patch_by_id = {instance_id: f"actual patch for {instance_id}" for instance_id in manifest.smoke_ids}

    def fake_loader(path: str, *, split: str, streaming: bool):
        assert path == manifest.dataset_name
        assert split == manifest.split
        assert streaming is True
        yield {"instance_id": "unrelated", "patch": "ignored"}
        for instance_id, patch in patch_by_id.items():
            yield {"instance_id": instance_id, "patch": patch, "test_patch": "hidden"}

    patches = load_dataset_gold_patches(
        manifest.dataset_name, manifest.split, manifest.smoke_ids, loader=fake_loader
    )

    assert patches == patch_by_id


def test_gold_smoke_generation_uses_default_dataset_loader_fixture(
    monkeypatch: pytest.MonkeyPatch, project_root: Path, tmp_path: Path
):
    manifest = load_task_manifest(project_root / "configs" / "task_manifest.yaml")
    output = tmp_path / "gold_smoke.jsonl"
    patch_by_id = {instance_id: f"actual dataset patch for {instance_id}" for instance_id in manifest.smoke_ids}

    def fake_dataset_patches(dataset_name: str, split: str, instance_ids: tuple[str, ...]):
        assert dataset_name == manifest.dataset_name
        assert split == manifest.split
        assert instance_ids == manifest.smoke_ids
        return patch_by_id

    monkeypatch.setattr("scripts.make_gold_smoke.load_dataset_gold_patches", fake_dataset_patches)

    count = write_gold_smoke_predictions(
        manifest_path=project_root / "configs" / "task_manifest.yaml",
        out_path=output,
        model_name="gold-smoke-test",
    )
    rows = [cast(dict[str, object], json.loads(line)) for line in output.read_text(encoding="utf-8").splitlines()]

    assert count == len(manifest.smoke_ids)
    assert [row["model_patch"] for row in rows] == [patch_by_id[instance_id] for instance_id in manifest.smoke_ids]


def test_gold_smoke_generation_reports_unavailable_dataset(
    monkeypatch: pytest.MonkeyPatch, project_root: Path, tmp_path: Path
):
    output = tmp_path / "gold_smoke.jsonl"

    def blocked_dataset_patches(dataset_name: str, split: str, instance_ids: tuple[str, ...]):
        _ = (dataset_name, split, instance_ids)
        raise ConfigError("gold_patch_unavailable: dataset access blocked")

    monkeypatch.setattr("scripts.make_gold_smoke.load_dataset_gold_patches", blocked_dataset_patches)

    with pytest.raises(ConfigError, match="gold_patch_unavailable"):
        _ = write_gold_smoke_predictions(
            manifest_path=project_root / "configs" / "task_manifest.yaml",
            out_path=output,
            model_name="gold-smoke-test",
        )

    assert not output.exists()


def test_gold_smoke_generation_uses_local_gold_source_fixture(project_root: Path, tmp_path: Path):
    manifest = load_task_manifest(project_root / "configs" / "task_manifest.yaml")
    output = tmp_path / "gold_smoke.jsonl"
    gold_source = tmp_path / "gold_source.jsonl"
    patch_by_id = {
        instance_id: f"diff --git a/{instance_id}.py b/{instance_id}.py\n--- a/{instance_id}.py\n+++ b/{instance_id}.py\n"
        for instance_id in manifest.smoke_ids
    }
    _ = gold_source.write_text(
        "".join(
            json.dumps({"instance_id": instance_id, "patch": patch}, sort_keys=True) + "\n"
            for instance_id, patch in patch_by_id.items()
        ),
        encoding="utf-8",
    )

    count = write_gold_smoke_predictions(
        manifest_path=project_root / "configs" / "task_manifest.yaml",
        out_path=output,
        model_name="gold-smoke-test",
        gold_source=gold_source,
    )
    summary = validate_predictions_file(output)
    rows = [cast(dict[str, object], json.loads(line)) for line in output.read_text(encoding="utf-8").splitlines()]
    loaded_gold = load_gold_patch_source(gold_source)

    assert count == len(manifest.smoke_ids)
    assert summary.instance_ids == manifest.smoke_ids
    assert [row["model_name_or_path"] for row in rows] == ["gold-smoke-test"] * count
    assert [row["model_patch"] for row in rows] == [loaded_gold[instance_id] for instance_id in manifest.smoke_ids]
    banned_synthetic_name = "".join(["repair_agent", "_smoke.py"])
    assert banned_synthetic_name not in output.read_text(encoding="utf-8")


def test_harness_simulated_docker_failure_writes_blocked_status(tmp_path: Path):
    predictions = tmp_path / "predictions.jsonl"
    _ = predictions.write_text(
        '{"instance_id":"case","model_name_or_path":"unit","model_patch":""}\n',
        encoding="utf-8",
    )
    status_path = tmp_path / "harness_status.json"

    result = harness_main(
        [
            "--predictions",
            str(predictions),
            "--run-id",
            "simulated",
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
    assert status["max_workers"] == 1
    assert status["max_workers_source"] == "cli"
    assert status["cache_level"] == "env"
    command = cast(list[str], status["command"])
    assert "swebench.harness.run_evaluation" in command


def test_harness_auto_workers_record_resources_and_cache(project_root: Path, tmp_path: Path):
    predictions = tmp_path / "predictions.jsonl"
    _ = predictions.write_text(
        '{"instance_id":"case","model_name_or_path":"unit","model_patch":""}\n',
        encoding="utf-8",
    )
    inventory = tmp_path / "device_inventory.json"
    _ = inventory.write_text(
        json.dumps(
            {
                "gpus": [{"index": 0, "memory_free_mb": 12000}],
                "memory": {"available_mb": 65536, "total_mb": 131072},
                "swebench_workers": {"recommended_swebench_max_workers": 7},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    status_path = tmp_path / "harness_status.json"

    result = harness_main(
        [
            "--predictions",
            str(predictions),
            "--run-id",
            "auto",
            "--resources",
            str(project_root / "configs" / "resources.yaml"),
            "--inventory",
            str(inventory),
            "--auto-workers",
            "--simulate-docker-failure",
            "--status-out",
            str(status_path),
        ]
    )
    status = cast(dict[str, object], json.loads(status_path.read_text(encoding="utf-8")))
    resources = cast(dict[str, object], status["resources"])
    worker_settings = cast(dict[str, object], resources["worker_settings"])
    memory_snapshot = cast(dict[str, object], status["memory_snapshot"])

    assert result == 0
    assert status["status"] == "blocked"
    assert status["max_workers"] == 7
    assert status["max_workers_source"] == "resources"
    assert status["cache_level"] == "env"
    assert worker_settings["swebench_max_workers"] == 7
    assert memory_snapshot["available_mb"] == 65536
