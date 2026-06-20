from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import torch
import transformers

from repair_agent.agent import models
from repair_agent.agent.models import (
    QWABLE_MODEL_ID,
    check_qwable_gate,
    classify_qwable_resource_safety,
    load_models_config,
    prepare_qz_qwable_job,
    verify_qwable_real_gate,
    write_gate_record,
)
from repair_agent.config import ConfigMap, require_mapping


class _MockTokenizer:
    eos_token_id = 0
    pad_token_id = None
    chat_template = None

    def __call__(self, text: str, return_tensors: str | None = None) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[5, 6, 7]])}

    def decode(self, token_ids: object, skip_special_tokens: bool = False) -> str:
        return '<tool_use>{"name":"read","arguments":{"path":"README.md"}}</tool_use>'


class _MockModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.hf_device_map = {"": 0}

    def generate(self, input_ids: object, **kwargs: object) -> torch.Tensor:
        return torch.tensor([[5, 6, 7, 8, 9, 10]])


def _patch_real_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _MockTokenizer()
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: _MockModel()
    )


def _real_gate_record(monkeypatch: pytest.MonkeyPatch, project_root: Path) -> ConfigMap:
    _patch_real_loaders(monkeypatch)
    models_config = load_models_config(project_root / "configs" / "models.yaml")
    return check_qwable_gate(
        models_config,
        project_root / "configs" / "resources.yaml",
        dry_run=False,
        max_new_tokens=16,
        inventory_path=project_root / "outputs" / "device_inventory.json",
    )


def test_qwable_real_gate_artifact_shape(
    monkeypatch: pytest.MonkeyPatch, project_root: Path, tmp_path: Path
):
    record = _real_gate_record(monkeypatch, project_root)
    output = write_gate_record(record, tmp_path)
    loaded = cast(ConfigMap, json.loads(output.read_text(encoding="utf-8")))
    details = require_mapping(loaded.get("details"), "details must be present")
    memory = require_mapping(loaded.get("memory"), "memory must be present")

    assert loaded["status"] == "pass"
    assert details["dry_run"] is False
    assert details["generated_text_nonempty"] is True
    assert details["device_map"] == "auto"
    assert details["license"] == "AGPL-3.0"
    assert details["canonical_model_id"] == QWABLE_MODEL_ID
    assert memory["loading"] == "loaded"
    assert "gpu_memory_peak" in memory
    assert loaded["device_ids"] == [0, 1, 2, 3]


def test_qwable_real_gate_passes_strict_verifier(monkeypatch: pytest.MonkeyPatch, project_root: Path):
    record = _real_gate_record(monkeypatch, project_root)

    verdict = verify_qwable_real_gate(record)

    assert verdict["ok"] is True
    assert verdict["reasons"] == []


def test_qwable_offload_records_qz_status(
    monkeypatch: pytest.MonkeyPatch, project_root: Path, tmp_path: Path
):
    monkeypatch.setattr(
        models,
        "_qwable_resource_assessment",
        lambda *a, **k: {
            "classification": "unsafe_for_4090",
            "reasons": ["gpu_0_free_below_4096mb"],
            "visible_gpus": [],
        },
    )
    job_path = tmp_path / "qz" / "qwable_gate_job.json"
    models_config = load_models_config(project_root / "configs" / "models.yaml")

    record = check_qwable_gate(
        models_config,
        project_root / "configs" / "resources.yaml",
        dry_run=False,
        qz_schema_path=project_root / "outputs" / "qz" / "train.CreateJob.schema.yaml",
        qz_job_out_path=job_path,
    )
    memory = require_mapping(record.get("memory"), "memory must be present")
    offload = require_mapping(record.get("qz_offload_status"), "qz_offload_status must be present")
    fallback_reasons = record.get("fallback_reasons")

    assert record["status"] == "skipped"
    assert memory["loading"] == "offloaded_qz"
    assert isinstance(fallback_reasons, list) and fallback_reasons
    assert offload["submitted"] is False
    assert offload["requires_resolution_before_submission"] is True
    assert verify_qwable_real_gate(record)["ok"] is False


def test_prepare_qz_qwable_job_creates_valid_spec(project_root: Path, tmp_path: Path):
    job_path = tmp_path / "qwable_gate_job.json"

    written = prepare_qz_qwable_job(
        project_root / "outputs" / "qz" / "train.CreateJob.schema.yaml", job_path
    )
    spec = cast(ConfigMap, json.loads(job_path.read_text(encoding="utf-8")))

    assert written == job_path
    for field in ("name", "project_id", "workspace_id", "logic_compute_group_id", "framework", "command"):
        assert field in spec
    assert spec["project_id"] == "RESOLVE_BEFORE_SUBMISSION"
    assert spec["workspace_id"] == "RESOLVE_BEFORE_SUBMISSION"
    assert spec["logic_compute_group_id"] == "RESOLVE_BEFORE_SUBMISSION"
    assert isinstance(spec["command"], str) and spec["command"]


def test_classifier_unsafe_when_total_gpu_memory_below_floor(project_root: Path, tmp_path: Path):
    inventory = {"gpus": [{"index": 0, "memory_total_mb": 8000, "memory_free_mb": 8000}]}
    inventory_path = tmp_path / "small_inventory.json"
    _ = inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    classification = classify_qwable_resource_safety(
        project_root / "configs" / "resources.yaml", inventory_path
    )

    assert classification == "unsafe_for_4090"
