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
    QwableAdapter,
    check_qwable_gate,
    classify_qwable_resource_safety,
    load_models_config,
    verify_qwable_real_gate,
)
from repair_agent.config import ConfigMap, require_mapping


class _MockTokenizer:
    eos_token_id = 0
    pad_token_id = None
    chat_template = None

    def __call__(self, text: str, return_tensors: str | None = None) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[1, 2, 3, 4]])}

    def decode(self, token_ids: object, skip_special_tokens: bool = False) -> str:
        return 'Inspect first. <tool_use>{"name":"read","arguments":{"path":"README.md"}}</tool_use>'


class _MockModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.hf_device_map = {"": 0}

    def generate(self, input_ids: object, **kwargs: object) -> torch.Tensor:
        return torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])


def _patch_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _MockTokenizer()
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: _MockModel()
    )


def test_qwable_real_inference_succeeds(monkeypatch: pytest.MonkeyPatch):
    _patch_transformers(monkeypatch)

    result = QwableAdapter().generate(
        [{"role": "user", "content": "inspect the failing test"}],
        {"dry_run": False, "max_new_tokens": 16, "visible_gpus": [0, 1, 2, 3]},
    )
    metadata = result.metadata or {}

    assert result.finish_reason == "stop"
    assert result.text.strip() != ""
    assert metadata["dry_run"] is False
    assert metadata["loading"] == "loaded"
    assert metadata["device_map"] == "auto"
    assert metadata["model_id"] == QWABLE_MODEL_ID


def test_dry_run_cannot_satisfy_real_gate(project_root: Path):
    models_config = load_models_config(project_root / "configs" / "models.yaml")
    dry_run_record = check_qwable_gate(
        models_config, project_root / "configs" / "resources.yaml", dry_run=True
    )

    verdict = verify_qwable_real_gate(dry_run_record)

    assert verdict["ok"] is False
    assert "qwable_real_inference_required" in cast(list[str], verdict["reasons"])


def test_unsafe_local_path_prepares_qz_dry_run(
    monkeypatch: pytest.MonkeyPatch, project_root: Path, tmp_path: Path
):
    monkeypatch.setattr(
        models,
        "_qwable_resource_assessment",
        lambda *a, **k: {
            "classification": "unsafe_for_4090",
            "reasons": ["total_visible_gpu_memory_8000mb_below_required_12288mb"],
            "total_visible_gpu_memory_mb": 8000,
            "visible_gpus": [0],
        },
    )

    def _fail_loader(*args: object, **kwargs: object) -> object:
        raise AssertionError("real inference must not run on the unsafe offload path")

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", _fail_loader)

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

    assert record["status"] == "skipped"
    assert memory["loading"] == "offloaded_qz"
    assert offload["submitted"] is False
    assert "fallback_reasons" in record
    assert job_path.is_file()

    spec = cast(ConfigMap, json.loads(job_path.read_text(encoding="utf-8")))
    assert spec["project_id"] == "RESOLVE_BEFORE_SUBMISSION"


def test_resource_classifier_safe_with_4_gpus(project_root: Path):
    classification = classify_qwable_resource_safety(
        project_root / "configs" / "resources.yaml",
        project_root / "outputs" / "device_inventory.json",
    )

    assert classification == "safe_for_4090"
