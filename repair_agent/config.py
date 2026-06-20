from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import yaml


ConfigMap: TypeAlias = dict[str, object]


class ConfigError(ValueError):
    pass


def load_yaml_config(path: str | Path) -> ConfigMap:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        loaded: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read config file {config_path}: {exc}") from exc

    if loaded is None:
        raise ConfigError(f"Config file is empty: {config_path}")
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config file must contain a YAML mapping: {config_path}")
    return _string_key_mapping(loaded, f"Config file {config_path}")


def load_run_config(path: str | Path) -> ConfigMap:
    config = load_yaml_config(path)
    run_section = require_mapping(config.get("run"), "Run config must define a 'run' mapping")
    run_name = run_section.get("name")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ConfigError("Run config field 'run.name' must be a non-empty string")

    dry_run = require_mapping(config.get("dry_run"), "Run config must define a 'dry_run' mapping")
    instances = dry_run.get("instances")
    if not isinstance(instances, list) or not instances:
        raise ConfigError("Run config field 'dry_run.instances' must be a non-empty list")

    seen_ids: set[str] = set()
    for index, instance in enumerate(instances):
        instance_map = require_mapping(instance, f"dry_run.instances[{index}] must be a mapping")
        instance_id = instance_map.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ConfigError(f"dry_run.instances[{index}].instance_id must be a non-empty string")
        if instance_id in seen_ids:
            raise ConfigError(f"Duplicate dry-run instance_id: {instance_id}")
        seen_ids.add(instance_id)
    return config


def dry_run_instances(config: ConfigMap, limit: int | None = None) -> list[ConfigMap]:
    dry_run = require_mapping(config.get("dry_run"), "Run config must define a 'dry_run' mapping")
    instances = dry_run.get("instances")
    if not isinstance(instances, list):
        raise ConfigError("Run config field 'dry_run.instances' must be a list")
    instance_maps = [require_mapping(item, "Each dry-run instance must be a mapping") for item in instances]
    if limit is None:
        return instance_maps
    if limit < 1:
        raise ConfigError("--limit must be a positive integer")
    return instance_maps[:limit]


def output_root(config: ConfigMap) -> Path:
    run_section = require_mapping(config.get("run"), "Run config must define a 'run' mapping")
    configured = run_section.get("output_dir", "outputs/runs")
    if not isinstance(configured, str) or not configured.strip():
        raise ConfigError("Run config field 'run.output_dir' must be a non-empty string")
    return Path(configured)


def require_mapping(value: object, error: str) -> ConfigMap:
    if not isinstance(value, dict):
        raise ConfigError(error)
    return _string_key_mapping(value, error)


def require_string(value: object, error: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(error)
    return value


def _string_key_mapping(value: dict[object, object], error: str) -> ConfigMap:
    result: ConfigMap = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ConfigError(f"{error} must use string keys")
        result[key] = item
    return result
