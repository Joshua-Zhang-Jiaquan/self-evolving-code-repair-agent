from __future__ import annotations

import ast
import html
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, cast

from repair_agent.config import ConfigError, ConfigMap, load_yaml_config, require_mapping, require_string
from repair_agent.resources import ResourcePlan, load_device_inventory, load_resource_config, resolve_resource_plan


DEFAULT_DIFFRWKV_CHECKPOINT = Path("/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/DiffRWKV-RELAY/releases/traj32x16-2.9B-s2-rwkv7-v3-ddpm")

STATUS_VALUES = {"pass", "blocked", "skipped"}
SUPPORTED_DEVICE_STRATEGIES = {
    "device_map_auto",
    "tensor_parallel",
    "per_worker_cuda_visible_devices",
    "cpu_fallback",
    "fallback",
}

QWABLE_MODEL_ID = "lordx64/Qwable-v1"
QWABLE_INFERENCE_SEED = 20260619
QWABLE_DEFAULT_MAX_NEW_TOKENS = 1024
QWABLE_VISIBLE_GPUS: tuple[int, ...] = (0, 1, 2, 3)
# 12 GB floor: a ~2.9B-active model needs this much for device_map=auto inference.
QWABLE_MIN_INFERENCE_MEMORY_MB = 12 * 1024
QWABLE_GATE_PROMPT = (
    "You are a code repair agent. Name one safe local tool you would call "
    "first to inspect a failing test, and reply with a single <tool_use> call."
)
DEFAULT_QZ_SCHEMA_PATH = Path("outputs/qz/train.CreateJob.schema.yaml")
DEFAULT_QZ_QWABLE_JOB_PATH = Path("outputs/qz/qwable_gate_job.json")
QZ_RESOLVE_PLACEHOLDER = "RESOLVE_BEFORE_SUBMISSION"
QWABLE_QZ_OFFLOAD_COMMAND = (
    "python scripts/check_model_gate.py --model qwable --no-dry-run "
    "--models-config configs/models.yaml --resources configs/resources.yaml "
    "--out-dir outputs/model_gates --max-new-tokens 1024"
)

TOOL_NAME_ALIASES = {
    "read": "read_file",
    "readfile": "read_file",
    "read_file": "read_file",
    "file_read": "read_file",
    "open": "read_file",
    "search": "search",
    "grep": "search",
    "find": "search",
    "inspect_test": "inspect_test",
    "inspect_tests": "inspect_test",
    "test": "run_tests",
    "tests": "run_tests",
    "run_test": "run_tests",
    "run_tests": "run_tests",
    "pytest": "run_tests",
    "edit": "edit_file",
    "write": "edit_file",
    "edit_file": "edit_file",
    "rollback": "rollback",
    "revert": "rollback",
    "final": "final_answer",
    "final_answer": "final_answer",
    "answer": "final_answer",
}


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    finish_reason: str = "stop"
    metadata: ConfigMap | None = None


class ModelAdapter(Protocol):
    name: str

    def generate(self, messages: list[ConfigMap], config: ConfigMap) -> GenerationResult:
        ...


@dataclass(frozen=True)
class ToolUseParseResult:
    ok: bool
    tool_name: str | None
    arguments: ConfigMap
    raw_tool_name: str | None
    raw_payload: str | None
    error: str | None = None

    def to_record(self) -> ConfigMap:
        return {
            "arguments": self.arguments,
            "error": self.error,
            "ok": self.ok,
            "raw_payload": self.raw_payload,
            "raw_tool_name": self.raw_tool_name,
            "tool_name": self.tool_name,
        }


class RuleBasedAdapter:
    name: str = "rule_based_local"

    def generate(self, messages: list[ConfigMap], config: ConfigMap) -> GenerationResult:
        prompt = _last_message_content(messages)
        if "read" in prompt.lower() or "file" in prompt.lower():
            text = '<tool_use>{"name":"read","arguments":{"path":"README.md"}}</tool_use>'
        else:
            text = '<tool_use>{"name":"search","arguments":{"query":"failure"}}</tool_use>'
        return GenerationResult(
            text=text,
            model=self.name,
            metadata={"dry_run": bool(config.get("dry_run", True)), "local_only": True},
        )


class QwableAdapter:
    name: str = "qwable"

    def generate(self, messages: list[ConfigMap], config: ConfigMap) -> GenerationResult:
        if config.get("dry_run", True):
            target = str(config.get("dry_run_read_path", "README.md"))
            return GenerationResult(
                text=f'<tool_use>{{"name":"read","arguments":{{"path":"{target}"}}}}</tool_use>',
                model=self.name,
                metadata={"dry_run": True, "local_only": True, "parser": "xml_tool_use_json"},
            )
        return self._generate_real(messages, config)

    def _generate_real(self, messages: list[ConfigMap], config: ConfigMap) -> GenerationResult:
        import os
        import sys

        model_id = str(config.get("model_id") or QWABLE_MODEL_ID)
        max_new_tokens = _positive_int(config.get("max_new_tokens"), QWABLE_DEFAULT_MAX_NEW_TOKENS)
        seed = _positive_int(config.get("seed"), QWABLE_INFERENCE_SEED)
        visible_gpus = _int_items(config.get("visible_gpus")) or list(QWABLE_VISIBLE_GPUS)
        if "torch" not in sys.modules:
            _ = os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(index) for index in visible_gpus))
        try:
            import torch
            import transformers

            transformers.set_seed(seed)
            _ = torch.manual_seed(seed)
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                for index in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(index)

            tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", torch_dtype="auto"
            )

            input_ids = _qwable_input_ids(tokenizer, messages)
            target_device = getattr(model, "device", None)
            if target_device is not None:
                input_ids = input_ids.to(target_device)
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(tokenizer, "eos_token_id", None)
            output_ids = model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_token_id
            )
            prompt_length = int(input_ids.shape[-1])
            generated = tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True).strip()
            peak_total, peak_per_device = _peak_gpu_memory(torch, visible_gpus, cuda_available)
            return GenerationResult(
                text=generated,
                model=self.name,
                finish_reason="stop",
                metadata={
                    "device_map": "auto",
                    "device_map_resolved": _stringify_device_map(getattr(model, "hf_device_map", None)),
                    "dry_run": False,
                    "gpu_memory_peak": peak_total,
                    "gpu_memory_peak_per_device": peak_per_device,
                    "loading": "loaded",
                    "local_only": True,
                    "max_new_tokens": max_new_tokens,
                    "model_id": model_id,
                    "seed": seed,
                },
            )
        except Exception as exc:  # any load/inference failure must be recorded, not claimed as loaded
            return GenerationResult(
                text="",
                model=self.name,
                finish_reason="blocked",
                metadata={
                    "device_map": "auto",
                    "dry_run": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "gpu_memory_peak": None,
                    "loading": "failed",
                    "local_only": True,
                    "model_id": model_id,
                },
            )

    @staticmethod
    def parse_tool_use(text: str) -> ToolUseParseResult:
        return parse_qwable_tool_use(text)


class DiffRWKVAdapter:
    name: str = "diffrwkv"

    def generate(self, messages: list[ConfigMap], config: ConfigMap) -> GenerationResult:
        _ = messages
        checkpoint = str(config.get("checkpoint", DEFAULT_DIFFRWKV_CHECKPOINT))
        return GenerationResult(
            text="DiffRWKV dry-run adapter did not load weights; instruction-following code repair is gated.",
            model=self.name,
            finish_reason="blocked",
            metadata={
                "checkpoint": checkpoint,
                "dry_run": bool(config.get("dry_run", True)),
                "local_only": True,
                "supports_instruction_following_code_repair": False,
            },
        )


def normalize_tool_name(name: object) -> str | None:
    if not isinstance(name, str) or not name.strip():
        return None
    key = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower()).strip("_")
    return TOOL_NAME_ALIASES.get(key, key if key in set(TOOL_NAME_ALIASES.values()) else None)


def parse_qwable_tool_use(text: str) -> ToolUseParseResult:
    if not text.strip():
        return _parse_error("empty_model_output", None)

    payload = _extract_tool_payload(text)
    if payload is None:
        return _parse_error("tool_use_payload_not_found", None)

    loaded, error = _load_payload(payload)
    if error is not None:
        return _parse_error(error, payload)
    if not isinstance(loaded, dict):
        return _parse_error("tool_use_payload_must_be_object", payload)

    record = _string_key_mapping(cast(Mapping[object, object], loaded))
    raw_name = record.get("name", record.get("tool", record.get("tool_name", record.get("function"))))
    normalized = normalize_tool_name(raw_name)
    if normalized is None:
        return ToolUseParseResult(
            ok=False,
            tool_name=None,
            arguments={},
            raw_tool_name=str(raw_name) if raw_name is not None else None,
            raw_payload=payload,
            error="unknown_or_missing_tool_name",
        )

    args_value = record.get("arguments", record.get("args", record.get("parameters")))
    if args_value is None:
        args_value = {
            key: value
            for key, value in record.items()
            if key not in {"name", "tool", "tool_name", "function"}
        }
    if not isinstance(args_value, dict):
        return ToolUseParseResult(
            ok=False,
            tool_name=normalized,
            arguments={},
            raw_tool_name=str(raw_name),
            raw_payload=payload,
            error="tool_arguments_must_be_object",
        )

    return ToolUseParseResult(
        ok=True,
        tool_name=normalized,
        arguments=_string_key_mapping(cast(Mapping[object, object], args_value)),
        raw_tool_name=str(raw_name),
        raw_payload=payload,
        error=None,
    )


def load_models_config(path: str | Path = "configs/models.yaml") -> ConfigMap:
    config = load_yaml_config(path)
    models = require_mapping(config.get("models"), "Models config must define a 'models' mapping")
    for name in ("qwable", "diffrwkv"):
        _ = require_mapping(models.get(name), f"Models config must define models.{name}")
    return config


def check_qwable_gate(
    models_config: ConfigMap,
    resources_path: str | Path | None = "configs/resources.yaml",
    dry_run: bool = True,
    max_new_tokens: int = QWABLE_DEFAULT_MAX_NEW_TOKENS,
    inventory_path: str | Path = "outputs/device_inventory.json",
    qz_schema_path: str | Path = DEFAULT_QZ_SCHEMA_PATH,
    qz_job_out_path: str | Path = DEFAULT_QZ_QWABLE_JOB_PATH,
) -> ConfigMap:
    model_cfg = _model_entry(models_config, "qwable")
    adapter = QwableAdapter()
    resource_record = _resource_record(resources_path)
    strategy = _select_device_strategy(resource_record, model_cfg)
    visible_gpus = _int_items(resource_record.get("visible_gpus", []))
    if dry_run:
        return _qwable_dry_run_gate(adapter, model_cfg, resource_record, strategy, visible_gpus)

    assessment = _qwable_resource_assessment(resources_path, inventory_path, QWABLE_MIN_INFERENCE_MEMORY_MB)
    if str(assessment["classification"]) == "unsafe_for_4090":
        return _qwable_offload_gate(
            adapter, model_cfg, resource_record, strategy, visible_gpus, assessment, qz_schema_path, qz_job_out_path
        )
    return _qwable_local_inference_gate(
        adapter, model_cfg, resource_record, strategy, visible_gpus, assessment, max_new_tokens
    )


def _qwable_dry_run_gate(
    adapter: QwableAdapter,
    model_cfg: ConfigMap,
    resource_record: ConfigMap,
    strategy: str,
    visible_gpus: list[int],
) -> ConfigMap:
    sample = adapter.generate([{"role": "user", "content": "read a project file"}], {"dry_run": True})
    parsed = adapter.parse_tool_use(sample.text)
    status = "pass" if parsed.ok else "skipped"
    reason = "dry_run_parser_and_resource_gate_passed" if parsed.ok else str(parsed.error)
    return _gate_record(
        model="qwable",
        status=status,
        reason=reason,
        device_ids=visible_gpus,
        device_strategy=strategy,
        memory={
            "estimate": str(model_cfg.get("memory_estimate", "unknown")),
            "fallback": resource_record.get("fallback", {}),
            "loading": "not_loaded_in_dry_run",
        },
        details={
            "adapter": adapter.name,
            "canonical_model_id": model_cfg.get("id"),
            "dry_run": True,
            "generation_finish_reason": sample.finish_reason,
            "license": model_cfg.get("license"),
            "local_only": bool(model_cfg.get("local_only", True)),
            "parser": parsed.to_record(),
            "resource_plan": resource_record,
        },
    )


def _qwable_local_inference_gate(
    adapter: QwableAdapter,
    model_cfg: ConfigMap,
    resource_record: ConfigMap,
    strategy: str,
    visible_gpus: list[int],
    assessment: ConfigMap,
    max_new_tokens: int,
) -> ConfigMap:
    model_id = str(model_cfg.get("id") or QWABLE_MODEL_ID)
    device_ids = visible_gpus or list(QWABLE_VISIBLE_GPUS)
    sample = adapter.generate(
        [{"role": "user", "content": QWABLE_GATE_PROMPT}],
        {
            "dry_run": False,
            "max_new_tokens": max_new_tokens,
            "model_id": model_id,
            "seed": QWABLE_INFERENCE_SEED,
            "visible_gpus": device_ids,
        },
    )
    meta = sample.metadata or {}
    loading = str(meta.get("loading", "failed"))
    generated_nonempty = bool(sample.text.strip())
    loaded_ok = loading == "loaded" and generated_nonempty and sample.finish_reason != "blocked"
    status = "pass" if loaded_ok else "blocked"
    reason = "real_local_inference_succeeded" if loaded_ok else f"qwable_local_inference_failed:{meta.get('error', 'unknown')}"
    return _gate_record(
        model="qwable",
        status=status,
        reason=reason,
        device_ids=device_ids,
        device_strategy=strategy,
        memory={
            "estimate": str(model_cfg.get("memory_estimate", "unknown")),
            "fallback": resource_record.get("fallback", {}),
            "gpu_memory_peak": meta.get("gpu_memory_peak"),
            "gpu_memory_peak_per_device": meta.get("gpu_memory_peak_per_device", {}),
            "loading": loading,
        },
        details={
            "adapter": adapter.name,
            "canonical_model_id": model_id,
            "device_map": meta.get("device_map", "auto"),
            "device_map_resolved": meta.get("device_map_resolved"),
            "dry_run": False,
            "generated_text_nonempty": generated_nonempty,
            "generated_text_preview": sample.text[:200],
            "generation_error": meta.get("error"),
            "generation_finish_reason": sample.finish_reason,
            "license": model_cfg.get("license"),
            "local_only": bool(model_cfg.get("local_only", True)),
            "max_new_tokens": max_new_tokens,
            "resource_classification": assessment,
            "resource_plan": resource_record,
            "seed": QWABLE_INFERENCE_SEED,
        },
    )


def _qwable_offload_gate(
    adapter: QwableAdapter,
    model_cfg: ConfigMap,
    resource_record: ConfigMap,
    strategy: str,
    visible_gpus: list[int],
    assessment: ConfigMap,
    qz_schema_path: str | Path,
    qz_job_out_path: str | Path,
) -> ConfigMap:
    job_path = prepare_qz_qwable_job(qz_schema_path, qz_job_out_path)
    raw_reasons = assessment.get("reasons")
    reasons = (
        [item for item in cast(list[object], raw_reasons) if isinstance(item, str)]
        if isinstance(raw_reasons, list)
        else []
    )
    fallback_reasons = reasons or ["local_gpu_memory_insufficient_for_qwable"]
    record = _gate_record(
        model="qwable",
        status="skipped",
        reason="qwable_local_unsafe_offloaded_to_qz_dry_run",
        device_ids=visible_gpus,
        device_strategy=strategy,
        memory={
            "estimate": str(model_cfg.get("memory_estimate", "unknown")),
            "fallback": resource_record.get("fallback", {}),
            "gpu_memory_peak": None,
            "loading": "offloaded_qz",
        },
        details={
            "adapter": adapter.name,
            "canonical_model_id": model_cfg.get("id") or QWABLE_MODEL_ID,
            "device_map": "offloaded_qz",
            "dry_run": False,
            "generated_text_nonempty": False,
            "license": model_cfg.get("license"),
            "local_only": bool(model_cfg.get("local_only", True)),
            "resource_classification": assessment,
            "resource_plan": resource_record,
        },
    )
    record["fallback_reasons"] = fallback_reasons
    record["qz_offload_status"] = {
        "dry_run_only": True,
        "job_spec_path": str(job_path),
        "requires_resolution_before_submission": True,
        "schema_source": str(qz_schema_path),
        "submitted": False,
    }
    return record


def classify_qwable_resource_safety(
    resources_path: str | Path | None = "configs/resources.yaml",
    inventory_path: str | Path = "outputs/device_inventory.json",
    min_memory_mb: int = QWABLE_MIN_INFERENCE_MEMORY_MB,
) -> str:
    assessment = _qwable_resource_assessment(resources_path, inventory_path, min_memory_mb)
    return str(assessment["classification"])


def prepare_qz_qwable_job(
    schema_path: str | Path = DEFAULT_QZ_SCHEMA_PATH,
    out_path: str | Path = DEFAULT_QZ_QWABLE_JOB_PATH,
) -> Path:
    required_fields = _qz_required_fields(schema_path)
    spec = _qwable_qz_job_spec(required_fields)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _ = target.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def verify_qwable_real_gate(record: ConfigMap) -> ConfigMap:
    details = record.get("details")
    details_map = details if isinstance(details, dict) else {}
    memory = record.get("memory")
    memory_map = memory if isinstance(memory, dict) else {}
    if details_map.get("dry_run") is not False:
        return {"ok": False, "reasons": ["qwable_real_inference_required"]}
    reasons: list[str] = []
    if memory_map.get("loading") != "loaded":
        reasons.append("qwable_weights_not_loaded")
    if details_map.get("generated_text_nonempty") is not True:
        reasons.append("qwable_generated_text_empty")
    if record.get("status") != "pass":
        reasons.append("qwable_gate_status_not_pass")
    return {"ok": not reasons, "reasons": reasons}


def check_diffrwkv_gate(
    models_config: ConfigMap,
    checkpoint: str | Path | None = None,
    resources_path: str | Path | None = "configs/resources.yaml",
    dry_run: bool = True,
) -> ConfigMap:
    model_cfg = _model_entry(models_config, "diffrwkv")
    configured_checkpoint = checkpoint if checkpoint is not None else model_cfg.get("checkpoint_path")
    if not isinstance(configured_checkpoint, str | Path):
        configured_checkpoint = DEFAULT_DIFFRWKV_CHECKPOINT
    checkpoint_path = Path(configured_checkpoint)
    resource_record = _resource_record(resources_path)
    strategy = _select_device_strategy(resource_record, model_cfg)
    artifact_report = inspect_diffrwkv_checkpoint(checkpoint_path)
    visible_gpus = _int_items(resource_record.get("visible_gpus", []))

    if not checkpoint_path.exists():
        status = "skipped"
        reason = "checkpoint_path_missing"
    elif artifact_report["missing_artifacts"]:
        status = "blocked"
        reason = "checkpoint_missing_required_artifacts"
    elif not bool(artifact_report["instruction_following_candidate"]):
        status = "blocked"
        reason = "diffrwkv_is_ddpm_rwkv_trajectory_model_not_instruction_code_repair"
    elif dry_run:
        status = "skipped"
        reason = "dry_run_skipped_weight_loading_smoke"
    else:
        status = "blocked"
        reason = "non_dry_run_smoke_loader_not_enabled_in_safe_gate"

    adapter = DiffRWKVAdapter()
    sample = adapter.generate([], {"checkpoint": str(checkpoint_path), "dry_run": dry_run})
    return _gate_record(
        model="diffrwkv",
        status=status,
        reason=reason,
        device_ids=visible_gpus,
        device_strategy=strategy,
        memory={
            "checkpoint_model_safetensors_size_bytes": artifact_report.get("model_safetensors_size_bytes"),
            "fallback": resource_record.get("fallback", {}),
            "loading": "not_loaded_in_dry_run" if dry_run else "not_attempted_after_gate",
        },
        details={
            "adapter": adapter.name,
            "checkpoint": str(checkpoint_path),
            "dry_run": dry_run,
            "gate_policy": model_cfg.get("gate_policy"),
            "local_only": bool(model_cfg.get("local_only", True)),
            "resource_plan": resource_record,
            "smoke_feasibility": {
                "one_turn_generation": "not_attempted_heavy_weight_load_avoided" if dry_run else "blocked",
                "prompt_to_patch_or_tool_selection": bool(artifact_report["instruction_following_candidate"]),
            },
            "artifact_report": artifact_report,
            "generation_metadata": sample.metadata or {},
        },
    )


def inspect_diffrwkv_checkpoint(checkpoint_path: Path) -> ConfigMap:
    expected = ["README.md", "config.yaml", "manifest.json"]
    existing = [name for name in expected if (checkpoint_path / name).is_file()]
    missing = [name for name in expected if name not in existing]
    has_model = (checkpoint_path / "model.safetensors").is_file()
    if not has_model:
        missing.append("model.safetensors")

    readme_text = _safe_read_text(checkpoint_path / "README.md")
    config_data = _safe_load_yaml(checkpoint_path / "config.yaml")
    manifest_data = _safe_load_json(checkpoint_path / "manifest.json")
    model_section = require_mapping(config_data.get("model", {}), "DiffRWKV model section must be a mapping") if config_data else {}
    training_section = require_mapping(config_data.get("training", {}), "DiffRWKV training section must be a mapping") if config_data else {}
    text = f"{readme_text}\n{json.dumps(config_data, sort_keys=True)}".lower()
    diffusion_markers = [marker for marker in ("ddpm", "diffusion", "trajectory", "state_hijack") if marker in text]
    instruction_markers = [marker for marker in ("instruction", "tool_use", "code repair", "swe-bench", "patch") if marker in text]
    manifest_size = manifest_data.get("model_safetensors_size_bytes")

    return {
        "backbone": model_section.get("rwkv_name"),
        "backbone_local_path": model_section.get("rwkv_local_path"),
        "config_model_type": model_section.get("type"),
        "existing_artifacts": existing + (["model.safetensors"] if has_model else []),
        "generation_type": training_section.get("gen_type"),
        "instruction_following_candidate": bool(instruction_markers) and not diffusion_markers,
        "instruction_markers": instruction_markers,
        "diffusion_trajectory_markers": diffusion_markers,
        "manifest_variant": manifest_data.get("variant"),
        "missing_artifacts": missing,
        "model_safetensors_present": has_model,
        "model_safetensors_size_bytes": manifest_size,
        "read_only_inspection": True,
    }


def write_gate_record(record: ConfigMap, out_dir: str | Path = "outputs/model_gates") -> Path:
    model = require_string(record.get("model"), "Gate record must include model")
    status = require_string(record.get("status"), "Gate record must include status")
    if status not in STATUS_VALUES:
        raise ConfigError(f"Gate status must be one of {sorted(STATUS_VALUES)}")
    target = Path(out_dir) / f"{model}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    _ = target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _extract_tool_payload(text: str) -> str | None:
    tag_match = re.search(r"<tool_use\b[^>]*>(.*?)</tool_use>", text, flags=re.IGNORECASE | re.DOTALL)
    if tag_match:
        return html.unescape(tag_match.group(1)).strip()
    alt_match = re.search(r"<tool_call\b[^>]*>(.*?)</tool_call>", text, flags=re.IGNORECASE | re.DOTALL)
    if alt_match:
        return html.unescape(alt_match.group(1)).strip()
    start = text.lower().find("<tool_use")
    if start >= 0:
        tail = text[text.find(">", start) + 1 :] if ">" in text[start:] else text[start + len("<tool_use") :]
        candidate = _first_json_object(tail)
        if candidate is not None:
            return html.unescape(candidate).strip()
    return _first_json_object(text)


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1].strip()
    return text[start:].strip()


def _load_payload(payload: str) -> tuple[object | None, str | None]:
    try:
        return json.loads(payload), None
    except json.JSONDecodeError as json_exc:
        try:
            return ast.literal_eval(payload), None
        except (ValueError, SyntaxError) as ast_exc:
            return None, f"invalid_tool_use_json: {json_exc.msg}; literal_eval: {ast_exc}"


def _parse_error(error: str, payload: str | None) -> ToolUseParseResult:
    return ToolUseParseResult(
        ok=False,
        tool_name=None,
        arguments={},
        raw_tool_name=None,
        raw_payload=payload,
        error=error,
    )


def _string_key_mapping(value: Mapping[object, object]) -> ConfigMap:
    return {str(key): item for key, item in value.items()}


def _last_message_content(messages: list[ConfigMap]) -> str:
    for message in reversed(messages):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""


def _model_entry(models_config: ConfigMap, name: str) -> ConfigMap:
    models = require_mapping(models_config.get("models"), "Models config must define models")
    return require_mapping(models.get(name), f"Models config must define models.{name}")


def _resource_record(resources_path: str | Path | None) -> ConfigMap:
    if resources_path is None:
        return {
            "assigned_device": "cpu",
            "device_policy": "cpu_only",
            "fallback": {"reasons": ["resources_config_not_provided"]},
            "inventory_source": None,
            "visible_gpus": [],
            "worker_settings": {},
        }
    resource_config = load_resource_config(resources_path)
    inventory = load_device_inventory("outputs/device_inventory.json")
    plan: ResourcePlan = resolve_resource_plan(resource_config, inventory, "outputs/device_inventory.json")
    return plan.to_record()


def _select_device_strategy(resource_record: ConfigMap, model_cfg: ConfigMap) -> str:
    visible = _int_items(resource_record.get("visible_gpus", []))
    configured = str(model_cfg.get("device_strategy", "device_map_auto"))
    if not visible:
        return "cpu_fallback"
    if configured in SUPPORTED_DEVICE_STRATEGIES:
        return configured
    if len(visible) >= 2:
        return "device_map_auto"
    return "per_worker_cuda_visible_devices"


def _int_items(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    items = cast(list[object], value)
    for item in items:
        if isinstance(item, int) and not isinstance(item, bool):
            result.append(item)
    return result


def _positive_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdecimal() and int(value) > 0:
        return int(value)
    return default


def _qwable_input_ids(tokenizer: Any, messages: list[ConfigMap]) -> Any:
    prompt_text = _last_message_content(messages) or QWABLE_GATE_PROMPT
    chat_template = getattr(tokenizer, "chat_template", None)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if chat_template and callable(apply_chat_template):
        prompt_text = apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return tokenizer(prompt_text, return_tensors="pt")["input_ids"]


def _peak_gpu_memory(
    torch_module: Any, visible_gpus: list[int], cuda_available: bool
) -> tuple[int | None, dict[str, int]]:
    if not cuda_available:
        return None, {}
    device_count = int(torch_module.cuda.device_count())
    per_device: dict[str, int] = {}
    total = 0
    for index in visible_gpus:
        if 0 <= index < device_count:
            device_peak = int(torch_module.cuda.max_memory_allocated(index))
            per_device[str(index)] = device_peak
            total += device_peak
    return total, per_device


def _stringify_device_map(device_map: object) -> object:
    if isinstance(device_map, Mapping):
        mapping = cast(Mapping[object, object], device_map)
        return {str(key): str(value) for key, value in mapping.items()}
    if device_map is None:
        return None
    return str(device_map)


def _resource_gpu_policy(resources_path: str | Path | None) -> tuple[list[int], int]:
    if resources_path is None:
        return [], 0
    config = load_resource_config(resources_path)
    gpus_cfg = require_mapping(config.get("gpus"), "Resource config must define a 'gpus' mapping")
    expected = _int_items(gpus_cfg.get("expected_ids", []))
    per_device = require_mapping(gpus_cfg.get("per_device", {}), "gpus.per_device must be a mapping")
    min_memory_value = per_device.get("min_memory_mb", 0)
    min_memory = min_memory_value if isinstance(min_memory_value, int) and not isinstance(min_memory_value, bool) else 0
    return expected, min_memory


def _inventory_gpus(inventory_path: str | Path) -> list[tuple[int, int, int]]:
    try:
        inventory = load_device_inventory(inventory_path)
    except ConfigError:
        return []
    if inventory is None:
        return []
    gpus = inventory.get("gpus", [])
    if not isinstance(gpus, list):
        return []
    result: list[tuple[int, int, int]] = []
    for gpu in cast(list[object], gpus):
        if not isinstance(gpu, dict):
            continue
        gpu_map = _string_key_mapping(cast(Mapping[object, object], gpu))
        index = gpu_map.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            continue
        total = gpu_map.get("memory_total_mb")
        total_mb = total if isinstance(total, int) and not isinstance(total, bool) else 0
        free = gpu_map.get("memory_free_mb")
        free_mb = free if isinstance(free, int) and not isinstance(free, bool) else total_mb
        result.append((index, total_mb, free_mb))
    return result


def _qwable_resource_assessment(
    resources_path: str | Path | None,
    inventory_path: str | Path,
    min_memory_mb: int,
) -> ConfigMap:
    expected_ids, gpu_min_free = _resource_gpu_policy(resources_path)
    inventory_by_index = {index: (total, free) for index, total, free in _inventory_gpus(inventory_path)}
    candidate_ids = expected_ids if expected_ids else sorted(inventory_by_index.keys())
    visible: list[int] = []
    per_gpu: dict[str, int] = {}
    reasons: list[str] = []
    total_memory = 0
    for index in candidate_ids:
        entry = inventory_by_index.get(index)
        if entry is None:
            reasons.append(f"gpu_{index}_absent_from_inventory")
            continue
        total_mb, free_mb = entry
        if free_mb < gpu_min_free:
            reasons.append(f"gpu_{index}_free_below_{gpu_min_free}mb")
            continue
        visible.append(index)
        per_gpu[str(index)] = total_mb
        total_memory += total_mb
    if not visible:
        reasons.append("no_visible_gpus_for_local_inference")
    if total_memory < min_memory_mb:
        reasons.append(f"total_visible_gpu_memory_{total_memory}mb_below_required_{min_memory_mb}mb")
    classification = "safe_for_4090" if total_memory >= min_memory_mb else "unsafe_for_4090"
    return {
        "classification": classification,
        "min_required_mb": min_memory_mb,
        "per_gpu_memory_mb": per_gpu,
        "reasons": reasons,
        "total_visible_gpu_memory_mb": total_memory,
        "visible_gpus": visible,
    }


def _load_qz_schema(schema_path: str | Path) -> ConfigMap:
    path = Path(schema_path)
    if not path.is_file():
        raise ConfigError(f"qz schema not found: {path}")
    try:
        loaded = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError) as exc:
        raise ConfigError(f"qz schema is not valid JSON: {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"qz schema must decode to a JSON object: {path}")
    return _string_key_mapping(cast(Mapping[object, object], loaded))


def _qz_required_fields(schema_path: str | Path) -> list[str]:
    schema = _load_qz_schema(schema_path)
    parameters = schema.get("parameters")
    if not isinstance(parameters, list):
        raise ConfigError("qz schema must define a 'parameters' list")
    required: list[str] = []
    for parameter in cast(list[object], parameters):
        if not isinstance(parameter, dict):
            continue
        parameter_map = _string_key_mapping(cast(Mapping[object, object], parameter))
        if parameter_map.get("required") is True:
            field = parameter_map.get("jsonField")
            if isinstance(field, str) and field:
                required.append(field)
    return required


def _qwable_qz_job_spec(required_fields: list[str]) -> ConfigMap:
    spec: ConfigMap = {
        "command": QWABLE_QZ_OFFLOAD_COMMAND,
        "framework": "PyTorch",
        "framework_config": [
            {
                "image": QZ_RESOLVE_PLACEHOLDER,
                "image_type": QZ_RESOLVE_PLACEHOLDER,
                "instance_count": 1,
                "spec_id": QZ_RESOLVE_PLACEHOLDER,
            }
        ],
        "logic_compute_group_id": QZ_RESOLVE_PLACEHOLDER,
        "name": "qwable-gate-local-inference",
        "project_id": QZ_RESOLVE_PLACEHOLDER,
        "workspace_id": QZ_RESOLVE_PLACEHOLDER,
    }
    for field in required_fields:
        if field not in spec:
            spec[field] = QZ_RESOLVE_PLACEHOLDER
    return spec


def _gate_record(
    model: str,
    status: str,
    reason: str,
    device_ids: list[int],
    device_strategy: str,
    memory: ConfigMap,
    details: ConfigMap,
) -> ConfigMap:
    return {
        "details": details,
        "device_ids": device_ids,
        "device_strategy": device_strategy,
        "fallback": memory.get("fallback", {}),
        "memory": memory,
        "model": model,
        "reason": reason,
        "status": status,
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else ""
    except OSError:
        return ""


def _safe_load_yaml(path: Path) -> ConfigMap:
    if not path.is_file():
        return {}
    try:
        loaded = load_yaml_config(path)
    except ConfigError:
        return {}
    return loaded


def _safe_load_json(path: Path) -> ConfigMap:
    if not path.is_file():
        return {}
    try:
        loaded = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return {}
    return _string_key_mapping(cast(Mapping[object, object], loaded)) if isinstance(loaded, dict) else {}
