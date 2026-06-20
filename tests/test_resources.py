from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from repair_agent.config import ConfigError, ConfigMap, require_mapping
from repair_agent.logging import read_jsonl
from repair_agent.resources import load_resource_config, resolve_resource_plan
from repair_agent.run import main


def write_resource_run_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "resource-run.yaml"
    _ = config_path.write_text(
        yaml.safe_dump(
            {
                "dry_run": {"instances": [{"instance_id": "resource-case", "repo": "local/repo"}]},
                "run": {"name": "resource-test", "output_dir": str(tmp_path / "runs")},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return config_path


def test_resource_plan_uses_inventory_and_worker_recommendation(project_root: Path):
    resources = load_resource_config(project_root / "configs" / "resources.yaml")
    inventory: ConfigMap = {
        "gpus": [
            {"index": 0, "memory_free_mb": 12000},
            {"index": 1, "memory_free_mb": 12000},
            {"index": 2, "memory_free_mb": 12000},
            {"index": 3, "memory_free_mb": 12000},
        ],
        "swebench_workers": {"recommended_swebench_max_workers": 16},
    }

    plan = resolve_resource_plan(resources, inventory, "inventory.json")

    assert plan.device_policy == "maximize_local"
    assert plan.visible_gpus == [0, 1, 2, 3]
    assert plan.assigned_device == "cuda:0"
    assert plan.worker_settings["swebench_max_workers"] == 16
    assert plan.fallback["missing_gpus"] == []


def test_cli_writes_resource_usage_jsonl(tmp_path: Path, project_root: Path):
    config_path = write_resource_run_config(tmp_path)
    result = main(
        [
            "--config",
            str(config_path),
            "--dry-run",
            "--limit",
            "1",
            "--resources",
            str(project_root / "configs" / "resources.yaml"),
            "--run-id",
            "resources",
        ]
    )

    assert result == 0
    rows = read_jsonl(tmp_path / "runs" / "resources" / "resource_usage.jsonl")
    assert len(rows) == 1
    row = rows[0]
    worker_settings = require_mapping(row.get("worker_settings"), "worker settings must be present")
    assert row["device_policy"] == "maximize_local"
    assert row["assigned_device"] in {"cuda:0", "cpu"}
    assert "visible_gpus" in row
    assert isinstance(worker_settings["swebench_max_workers"], int)
    assert worker_settings["swebench_max_workers"] >= 1


def test_resource_env_fallback_when_inventory_missing(monkeypatch: pytest.MonkeyPatch, project_root: Path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    resources = load_resource_config(project_root / "configs" / "resources.yaml")

    plan = resolve_resource_plan(resources, inventory=None, inventory_source=None)

    assert plan.visible_gpus == [2, 3]
    assert plan.assigned_device == "cuda:2"
    assert plan.inventory_source is None


def test_resource_config_rejects_unknown_policy(tmp_path: Path):
    path = tmp_path / "resources.yaml"
    _ = path.write_text(
        yaml.safe_dump(
            {
                "cpu": {},
                "device_policy": "unknown",
                "fallback": {},
                "gpus": {},
                "memory": {},
                "model_shards": {},
                "trainer_devices": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="device_policy"):
        load_resource_config(path)
