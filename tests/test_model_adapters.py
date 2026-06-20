from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import yaml

from repair_agent.agent.models import (
    DEFAULT_DIFFRWKV_CHECKPOINT,
    QwableAdapter,
    check_diffrwkv_gate,
    check_qwable_gate,
    inspect_diffrwkv_checkpoint,
    load_models_config,
    normalize_tool_name,
    parse_qwable_tool_use,
    write_gate_record,
)
from repair_agent.config import ConfigMap, require_mapping


def test_qwable_xml_tool_parser_extracts_and_normalizes_read():
    result = parse_qwable_tool_use(
        'prefix <tool_use>{"name":"read","arguments":{"path":"repair_agent/config.py"}}</tool_use>'
    )

    assert result.ok is True
    assert result.tool_name == "read_file"
    assert result.raw_tool_name == "read"
    assert result.arguments == {"path": "repair_agent/config.py"}
    assert result.error is None


def test_qwable_parser_handles_malformed_xml_json_without_crashing():
    result = parse_qwable_tool_use('<tool_use>{"name":"read","arguments":</tool_use>')

    assert result.ok is False
    assert result.tool_name is None
    assert result.error is not None
    assert "invalid_tool_use_json" in result.error


def test_tool_name_normalization_registry_aliases():
    assert normalize_tool_name("read") == "read_file"
    assert normalize_tool_name("pytest") == "run_tests"
    assert normalize_tool_name("edit") == "edit_file"
    assert normalize_tool_name("not-a-tool") is None


def test_qwable_adapter_dry_run_generation_is_parseable():
    generated = QwableAdapter().generate([{"role": "user", "content": "read file"}], {"dry_run": True})
    parsed = QwableAdapter.parse_tool_use(generated.text)

    assert generated.model == "qwable"
    assert parsed.ok is True
    assert parsed.tool_name == "read_file"


def test_qwable_gate_json_shape_records_resources(tmp_path: Path, project_root: Path):
    models_config = load_models_config(project_root / "configs" / "models.yaml")
    record = check_qwable_gate(models_config, project_root / "configs" / "resources.yaml", dry_run=True)
    output = write_gate_record(record, tmp_path)
    loaded = cast(ConfigMap, json.loads(output.read_text(encoding="utf-8")))
    details = require_mapping(loaded.get("details"), "details must be present")
    parser_record = require_mapping(details.get("parser"), "parser record must be present")

    assert loaded["status"] in {"pass", "blocked", "skipped"}
    assert loaded["status"] == "pass"
    assert loaded["model"] == "qwable"
    assert loaded["reason"]
    assert cast(str, loaded["timestamp"]).endswith("Z")
    assert loaded["device_strategy"] in {
        "device_map_auto",
        "tensor_parallel",
        "per_worker_cuda_visible_devices",
        "cpu_fallback",
        "fallback",
    }
    assert isinstance(loaded["device_ids"], list)
    assert "memory" in loaded
    assert parser_record["tool_name"] == "read_file"


def test_qwable_gate_records_intended_multi_gpu_strategy(tmp_path: Path, project_root: Path):
    resources_path = tmp_path / "resources.yaml"
    _ = resources_path.write_text(
        yaml.safe_dump(
            {
                "cpu": {"max_workers": 8},
                "device_policy": "maximize_local",
                "docker_cache_level": "env",
                "fallback": {"on_gpu_unavailable": "record_and_continue"},
                "gpus": {"expected_ids": [0, 1, 2, 3], "per_device": {"min_memory_mb": 4096}},
                "memory": {"reserve_mb": 8192},
                "model_shards": {"strategy": "device_map_auto"},
                "trainer_devices": {"rollout_parallelism": 4},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    models_config = load_models_config(project_root / "configs" / "models.yaml")
    record = check_qwable_gate(models_config, resources_path, dry_run=True)
    details = require_mapping(record.get("details"), "details must be present")

    assert record["device_strategy"] in {"device_map_auto", "cpu_fallback"}
    assert "visible_gpus" in require_mapping(details.get("resource_plan"), "resource plan must be present")
    if record["device_ids"]:
        assert record["device_ids"] == [0, 1, 2, 3]


def test_diffrwkv_missing_checkpoint_is_controlled(tmp_path: Path, project_root: Path):
    models_config = load_models_config(project_root / "configs" / "models.yaml")
    missing = tmp_path / "missing-checkpoint"
    record = check_diffrwkv_gate(
        models_config,
        checkpoint=missing,
        resources_path=project_root / "configs" / "resources.yaml",
        dry_run=True,
    )

    assert record["status"] == "skipped"
    assert record["reason"] == "checkpoint_path_missing"
    assert record["model"] == "diffrwkv"


def test_diffrwkv_checkpoint_inspection_blocks_trajectory_ddpm(project_root: Path):
    models_config = load_models_config(project_root / "configs" / "models.yaml")
    checkpoint = DEFAULT_DIFFRWKV_CHECKPOINT
    report = inspect_diffrwkv_checkpoint(checkpoint)
    record = check_diffrwkv_gate(
        models_config,
        checkpoint=checkpoint,
        resources_path=project_root / "configs" / "resources.yaml",
        dry_run=True,
    )

    if checkpoint.exists():
        existing_artifacts = cast(list[object], report["existing_artifacts"])
        reason = cast(str, record["reason"])
        details = require_mapping(record.get("details"), "details must be present")
        artifact_report = require_mapping(details.get("artifact_report"), "artifact report must be present")
        assert {"README.md", "config.yaml", "manifest.json", "model.safetensors"}.issubset(
            set(existing_artifacts)
        )
        assert record["status"] == "blocked"
        assert "trajectory" in reason or "instruction" in reason
        assert artifact_report["model_safetensors_present"] is True
    else:
        assert record["status"] == "skipped"


def test_models_yaml_defines_local_only_entries(project_root: Path):
    config = load_models_config(project_root / "configs" / "models.yaml")
    models = require_mapping(config.get("models"), "models must be present")
    qwable = require_mapping(models.get("qwable"), "qwable must be present")
    diffrwkv = require_mapping(models.get("diffrwkv"), "diffrwkv must be present")

    assert qwable["local_only"] is True
    assert qwable["commercial_api"] is False
    assert diffrwkv["local_only"] is True
    assert cast(str, diffrwkv["checkpoint_path"]).endswith("traj32x16-2.9B-s2-rwkv7-v3-ddpm")
