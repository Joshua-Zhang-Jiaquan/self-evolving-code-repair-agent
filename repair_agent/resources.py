from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from repair_agent.config import ConfigError, ConfigMap, load_yaml_config, require_mapping


@dataclass(frozen=True)
class ResourcePlan:
    device_policy: str
    visible_gpus: list[int]
    assigned_device: str
    worker_settings: dict[str, int | str]
    fallback: ConfigMap
    inventory_source: str | None

    def to_record(self) -> ConfigMap:
        return {
            "assigned_device": self.assigned_device,
            "device_policy": self.device_policy,
            "fallback": self.fallback,
            "inventory_source": self.inventory_source,
            "visible_gpus": self.visible_gpus,
            "worker_settings": self.worker_settings,
        }


def load_resource_config(path: str | Path) -> ConfigMap:
    config = load_yaml_config(path)
    policy = config.get("device_policy")
    if policy not in {"maximize_local", "single_device", "cpu_only"}:
        raise ConfigError("Resource config 'device_policy' must be maximize_local, single_device, or cpu_only")
    for key in ["gpus", "model_shards", "trainer_devices", "cpu", "memory", "fallback"]:
        if not isinstance(config.get(key), dict):
            raise ConfigError(f"Resource config must define a '{key}' mapping")
    return config


def load_device_inventory(path: str | Path = "outputs/device_inventory.json") -> ConfigMap | None:
    inventory_path = Path(path)
    if not inventory_path.is_file():
        return None
    loaded: object = json.loads(inventory_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ConfigError(f"Device inventory must contain a JSON object: {inventory_path}")
    return _string_key_mapping(loaded, f"Device inventory {inventory_path}")


def resolve_resource_plan(
    resource_config: ConfigMap,
    inventory: ConfigMap | None = None,
    inventory_source: str | None = "outputs/device_inventory.json",
) -> ResourcePlan:
    policy = str(resource_config["device_policy"])
    visible_gpus, fallback = _visible_gpus(resource_config, inventory)
    if policy == "cpu_only":
        assigned_device = "cpu"
    elif visible_gpus:
        assigned_device = f"cuda:{visible_gpus[0]}"
    else:
        assigned_device = "cpu"
        fallback["reasons"] = _with_reason(fallback.get("reasons"), "no_visible_gpus")

    worker_settings = _worker_settings(resource_config, inventory)
    source = inventory_source if inventory is not None else None
    return ResourcePlan(
        device_policy=policy,
        visible_gpus=visible_gpus,
        assigned_device=assigned_device,
        worker_settings=worker_settings,
        fallback=fallback,
        inventory_source=source,
    )


def _visible_gpus(
    resource_config: ConfigMap, inventory: ConfigMap | None
) -> tuple[list[int], ConfigMap]:
    gpus_cfg = require_mapping(resource_config.get("gpus"), "Resource config must define a 'gpus' mapping")
    expected_ids = _int_list(gpus_cfg.get("expected_ids", []))
    per_device = require_mapping(gpus_cfg.get("per_device", {}), "gpus.per_device must be a mapping")
    min_memory = _int_value(per_device.get("min_memory_mb", 0), "gpus.per_device.min_memory_mb")
    missing_gpus: list[int] = []
    reasons: list[str] = []

    if inventory is not None:
        inventory_gpus = inventory.get("gpus", [])
        if not isinstance(inventory_gpus, list):
            raise ConfigError("Device inventory field 'gpus' must be a list")
        visible: list[int] = []
        seen: set[int] = set()
        for gpu in inventory_gpus:
            if not isinstance(gpu, dict):
                continue
            gpu_map = _string_key_mapping(gpu, "GPU inventory entry")
            index_value = gpu_map.get("index")
            if index_value is None:
                continue
            index = _int_value(index_value, "GPU index")
            seen.add(index)
            if expected_ids and index not in expected_ids:
                continue
            free_memory = _int_value(gpu_map.get("memory_free_mb", 0), "GPU free memory")
            if free_memory >= min_memory:
                visible.append(index)
            else:
                reasons.append(f"gpu_{index}_memory_below_{min_memory}_mb")
        missing_gpus = [gpu_id for gpu_id in expected_ids if gpu_id not in seen]
        if missing_gpus:
            reasons.append("expected_gpus_missing")
        return visible, {"missing_gpus": missing_gpus, "reasons": reasons}

    env_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not env_devices:
        reasons.append("device_inventory_missing_and_cuda_visible_devices_unset")
        return [], {"missing_gpus": missing_gpus, "reasons": reasons}
    if env_devices.lower() in {"none", "no", "-1"}:
        reasons.append("cuda_visible_devices_disables_gpu")
        return [], {"missing_gpus": missing_gpus, "reasons": reasons}
    visible = []
    for chunk in env_devices.split(","):
        chunk = chunk.strip()
        if chunk.isdigit():
            visible.append(int(chunk))
    if not visible:
        reasons.append("cuda_visible_devices_has_no_numeric_ids")
    return visible, {"missing_gpus": missing_gpus, "reasons": reasons}


def _worker_settings(resource_config: ConfigMap, inventory: ConfigMap | None) -> dict[str, int | str]:
    configured_swebench = resource_config.get("swebench_max_workers", "auto")
    if configured_swebench == "auto":
        recommended: object = None
        if inventory is not None:
            swebench_section = require_mapping(
                inventory.get("swebench_workers", {}), "swebench_workers must be a mapping"
            )
            recommended = swebench_section.get("recommended_swebench_max_workers")
        swebench_workers = _int_value(recommended or max(1, (os.cpu_count() or 1) // 4), "swebench workers")
    else:
        swebench_workers = _int_value(configured_swebench, "swebench_max_workers")

    cpu_section = require_mapping(resource_config.get("cpu"), "Resource config must define a 'cpu' mapping")
    trainer_section = require_mapping(
        resource_config.get("trainer_devices"), "Resource config must define a 'trainer_devices' mapping"
    )

    return {
        "cpu_max_workers": _int_value(cpu_section.get("max_workers", os.cpu_count() or 1), "cpu.max_workers"),
        "docker_cache_level": str(resource_config.get("docker_cache_level", "env")),
        "rollout_parallelism": _int_value(
            trainer_section.get("rollout_parallelism", 1), "trainer_devices.rollout_parallelism"
        ),
        "swebench_max_workers": swebench_workers,
    }


def _int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        raise ConfigError("Expected a list of integer values")
    return [_int_value(item, "integer list item") for item in value]


def _int_value(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    raise ConfigError(f"{label} must be an integer")


def _string_key_mapping(value: dict[object, object], error: str) -> ConfigMap:
    result: ConfigMap = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ConfigError(f"{error} must use string keys")
        result[key] = item
    return result


def _with_reason(value: object, reason: str) -> list[str]:
    reasons = [item for item in value if isinstance(item, str)] if isinstance(value, list) else []
    reasons.append(reason)
    return reasons
