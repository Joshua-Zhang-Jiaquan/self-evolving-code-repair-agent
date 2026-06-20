from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from repair_agent.agent.baseline import BaselineAgent
from repair_agent.agent.feedback import FeedbackAgent
from repair_agent.agent.interface import AgentTask, RepairAgent
from repair_agent.agent.learning import LearningAgent
from repair_agent.config import (
    ConfigError,
    ConfigMap,
    dry_run_instances,
    load_run_config,
    output_root,
    require_mapping,
    require_string,
)
from repair_agent.env.swebench_loader import load_task_instances, load_task_manifest
from repair_agent.logging import (
    append_jsonl,
    ensure_run_dir,
    initialize_run_files,
    read_json_object,
    write_json_atomic,
)
from repair_agent.resources import load_device_inventory, load_resource_config, resolve_resource_plan


INSTANCE_SPLIT_CHOICES = ("main", "smoke")
OFFICIAL_INSTANCE_SOURCE = "swebench_lite_official"
OFFICIAL_METADATA_SOURCE = "swebench_lite"


@dataclass(frozen=True)
class CliArgs:
    config: str
    resources: str | None
    dry_run: bool
    limit: int | None
    run_id: str
    force: bool
    manifest: str | None = None
    instance_split: str | None = None
    strict_official: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repair_agent experiments")
    _ = parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    _ = parser.add_argument("--resources", help="Optional path to resource YAML config")
    _ = parser.add_argument("--dry-run", action="store_true", help="Run deterministic local dry-run instances")
    _ = parser.add_argument("--limit", type=int, help="Limit number of instances")
    _ = parser.add_argument("--run-id", required=True, help="Output run identifier under outputs/runs")
    _ = parser.add_argument("--force", action="store_true", help="Reset existing run artifacts before running")
    _ = parser.add_argument("--manifest", help="Path to SWE-bench Lite task manifest YAML (required with --strict-official)")
    _ = parser.add_argument("--instance-split", choices=list(INSTANCE_SPLIT_CHOICES), help="Official manifest split to evaluate in strict mode")
    _ = parser.add_argument("--strict-official", action="store_true", help="Load official SWE-bench Lite instances and ignore config fixtures")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _typed_args(build_parser().parse_args(argv))
    try:
        return run_from_args(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


def run_from_args(args: CliArgs) -> int:
    config_path = Path(args.config)
    config = load_run_config(config_path)
    run_dir = ensure_run_dir(output_root(config), args.run_id)
    paths = initialize_run_files(run_dir, force=bool(args.force))
    _ = shutil.copyfile(config_path, run_dir / "config.yaml")

    if args.resources:
        _record_resource_usage(Path(args.resources), run_dir)

    if args.strict_official:
        instances, metadata = _strict_official_instances(args)
        _run_agent(
            config=config,
            run_id=args.run_id,
            run_dir=run_dir,
            paths=paths,
            limit=None,
            force=args.force,
            instances_override=instances,
            metadata=metadata,
        )
        return 0

    if not args.dry_run:
        _run_agent(config=config, run_id=args.run_id, run_dir=run_dir, paths=paths, limit=args.limit, force=args.force)
        return 0
    _run_dry(config=config, run_id=args.run_id, paths=paths, limit=args.limit)
    return 0


def _strict_official_instances(args: CliArgs) -> tuple[list[ConfigMap], ConfigMap]:
    if not args.manifest:
        raise ConfigError("strict_official_requires_manifest")
    manifest_path = Path(args.manifest)
    manifest = load_task_manifest(manifest_path)
    split_choice = args.instance_split or "main"
    selected_ids = list(manifest.smoke_ids if split_choice == "smoke" else manifest.main_ids)
    loaded = load_task_instances(manifest_path, split="test", ids=selected_ids, strict=True)
    rows = [row for group in loaded.values() for row in group]
    instances: list[ConfigMap] = []
    for row in rows:
        instance_id = require_string(row.get("instance_id"), "Official instance_id must be a string")
        _assert_strict_official_id(instance_id)
        instances.append(_official_instance_record(instance_id, row))
    limited = _apply_limit(instances, args.limit)
    metadata: ConfigMap = {
        "official_instance_source": OFFICIAL_METADATA_SOURCE,
        "strict_official": True,
        "instance_split": split_choice,
        "instance_count": len(limited),
    }
    return limited, metadata


def _assert_strict_official_id(instance_id: str) -> None:
    if "local-" in instance_id or "__" not in instance_id:
        raise ConfigError("strict_official_rejects_fixture_id")


def _official_instance_record(instance_id: str, row: ConfigMap) -> ConfigMap:
    return {
        "instance_id": instance_id,
        "repo": str(row.get("repo", "")),
        "problem_statement": require_string(row.get("problem_statement"), "Official problem_statement must be a string"),
        "visible_tests": [],
        "visible_failures": {},
        "source": OFFICIAL_INSTANCE_SOURCE,
    }


def _apply_limit(instances: list[ConfigMap], limit: int | None) -> list[ConfigMap]:
    if limit is None:
        return instances
    if limit < 1:
        raise ConfigError("--limit must be a positive integer")
    return instances[:limit]


def _record_resource_usage(resources_path: Path, run_dir: Path) -> None:
    resources = load_resource_config(resources_path)
    inventory_path = Path("outputs/device_inventory.json")
    inventory = load_device_inventory(inventory_path)
    plan = resolve_resource_plan(resources, inventory, str(inventory_path))
    record = plan.to_record()
    record.update({"event": "resource_plan", "resources_path": str(resources_path), "timestamp": _now()})
    append_jsonl(run_dir / "resource_usage.jsonl", record)


def _run_dry(config: ConfigMap, run_id: str, paths: dict[str, Path], limit: int | None) -> None:
    selected = dry_run_instances(config, limit)
    state = read_json_object(paths["state"], {"completed_instances": []})
    completed = set(_completed_instances(state))
    run_section = require_mapping(config.get("run"), "Run config must define a 'run' mapping")
    run_name = require_string(run_section.get("name"), "Run config field 'run.name' must be a string")
    appended = 0
    skipped = 0

    for instance in selected:
        instance_id = require_string(instance.get("instance_id"), "Dry-run instance_id must be a string")
        if instance_id in completed:
            skipped += 1
            continue
        trajectory = _dry_trajectory_row(run_id, run_name, instance)
        prediction = _dry_prediction_row(run_id, run_name, instance)
        append_jsonl(paths["trajectories"], trajectory)
        append_jsonl(paths["predictions"], prediction)
        completed.add(instance_id)
        appended += 1

    completed_sorted = sorted(completed)
    status = "completed" if all(_instance_id(item) in completed for item in selected) else "partial"
    write_json_atomic(
        paths["state"],
        {
            "completed_instances": completed_sorted,
            "dry_run": True,
            "last_updated": _now(),
            "run_id": run_id,
            "status": status,
        },
    )
    write_json_atomic(
        paths["metrics"],
        {
            "completed": len([item for item in selected if _instance_id(item) in completed]),
            "newly_completed": appended,
            "skipped": skipped,
            "total": len(selected),
        },
    )


def _run_agent(
    config: ConfigMap,
    run_id: str,
    run_dir: Path,
    paths: dict[str, Path],
    limit: int | None,
    force: bool,
    *,
    instances_override: list[ConfigMap] | None = None,
    metadata: ConfigMap | None = None,
) -> None:
    selected = instances_override if instances_override is not None else _agent_instances(config, limit)
    state = read_json_object(paths["state"], {"completed_instances": []})
    completed = set(_completed_instances(state))
    run_section = require_mapping(config.get("run"), "Run config must define a 'run' mapping")
    run_name = require_string(run_section.get("name"), "Run config field 'run.name' must be a string")
    agent_section = require_mapping(config.get("agent"), "Run config must define an 'agent' mapping")
    model_name = str(agent_section.get("model_name_or_path", "rule_based_local"))
    agent = _agent_from_config(agent_section)
    patches_dir = run_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    appended = 0
    skipped = 0
    instance_metrics: list[ConfigMap] = []

    for instance in selected:
        instance_id = require_string(instance.get("instance_id"), "Baseline instance_id must be a string")
        if instance_id in completed:
            skipped += 1
            continue
        checkout_root = _prepare_task_checkout(run_dir, instance_id, instance, force=force)
        task = _agent_task_from_config(instance, agent_section, checkout_root, model_name)
        result = agent.run(task, run_id)
        patch_path = patches_dir / f"{_safe_filename(instance_id)}.patch"
        _ = patch_path.write_text(result.final.model_patch, encoding="utf-8")
        patch_sha = hashlib.sha256(result.final.model_patch.encode("utf-8")).hexdigest()
        for row in result.trajectory_rows():
            row["run_name"] = run_name
            row["patch_path"] = str(patch_path)
            row["patch_sha256"] = patch_sha
            append_jsonl(paths["trajectories"], row)
        append_jsonl(paths["predictions"], result.final.prediction_row())
        completed.add(instance_id)
        appended += 1
        instance_metrics.append(
            {
                "final_status": result.final.status,
                "instance_id": instance_id,
                "patch_path": str(patch_path),
                "patch_sha256": patch_sha,
                "patch_status": "non_empty" if result.final.model_patch else "empty",
                **dict(result.metrics),
            }
        )

    completed_sorted = sorted(completed)
    status = "completed" if all(_instance_id(item) in completed for item in selected) else "partial"
    state_payload: ConfigMap = {
        "completed_instances": completed_sorted,
        "dry_run": False,
        "last_updated": _now(),
        "run_id": run_id,
        "status": status,
    }
    if metadata is not None:
        state_payload["metadata"] = dict(metadata)
    write_json_atomic(paths["state"], state_payload)
    write_json_atomic(
        paths["metrics"],
        {
            "completed": len([item for item in selected if _instance_id(item) in completed]),
            "instances": instance_metrics,
            "newly_completed": appended,
            "skipped": skipped,
            "total": len(selected),
        },
    )


def _agent_instances(config: ConfigMap, limit: int | None) -> list[ConfigMap]:
    agent = require_mapping(config.get("agent"), "Run config must define an 'agent' mapping")
    instances = agent.get("instances", agent.get("tasks"))
    if not isinstance(instances, list) or not instances:
        raise ConfigError("Run config field 'agent.instances' must be a non-empty list")
    raw_instances = cast(list[object], instances)
    instance_maps = [require_mapping(_strip_hidden_fields(item), "Each baseline instance must be a mapping") for item in raw_instances]
    seen_ids: set[str] = set()
    for item in instance_maps:
        instance_id = require_string(item.get("instance_id"), "Agent instance_id must be a string")
        if instance_id in seen_ids:
            raise ConfigError(f"Duplicate agent instance_id: {instance_id}")
        seen_ids.add(instance_id)
    return _apply_limit(instance_maps, limit)


def _agent_from_config(agent_section: ConfigMap) -> RepairAgent:
    agent_type = str(agent_section.get("type", "baseline")).strip().lower()
    if agent_type == "baseline":
        return BaselineAgent()
    if agent_type in {"feedback", "+feedback"}:
        return FeedbackAgent()
    if agent_type in {"learning", "reinforce", "+learning"}:
        return LearningAgent()
    raise ConfigError(f"Unsupported agent.type: {agent_type}")


def _agent_task_from_config(instance: ConfigMap, agent_section: ConfigMap, checkout_root: Path, model_name: str) -> AgentTask:
    visible_tests_raw = instance.get("visible_tests", [])
    if not isinstance(visible_tests_raw, list):
        raise ConfigError("Baseline instance visible_tests must be a list when supplied")
    visible_failures = instance.get("visible_failures", {})
    if not isinstance(visible_failures, dict):
        raise ConfigError("Baseline instance visible_failures must be a mapping when supplied")
    visible_tests = cast(list[object], visible_tests_raw)
    failure_map = cast(dict[object, object], visible_failures)
    official = instance.get("source") == OFFICIAL_INSTANCE_SOURCE
    # Zero test budget for official empty checkouts: a bare pytest under the project
    # root would discover the project's own testpaths and run the whole suite.
    max_test_runs = 0 if official else _bounded_int(agent_section.get("max_test_runs"), default=1, minimum=0)
    return AgentTask(
        instance_id=require_string(instance.get("instance_id"), "Baseline instance_id must be a string"),
        repo=str(instance.get("repo", "local/baseline")),
        problem_statement=require_string(instance.get("problem_statement"), "Baseline problem_statement must be a string"),
        checkout_root=checkout_root,
        visible_tests=tuple(str(item) for item in visible_tests),
        visible_failures={str(key): str(value) for key, value in failure_map.items()},
        model_name_or_path=model_name,
        max_steps=_bounded_int(agent_section.get("max_steps"), default=12, minimum=1),
        max_test_runs=max_test_runs,
        test_timeout_seconds=_bounded_float(agent_section.get("test_timeout_seconds"), default=10.0, minimum=0.1),
        max_output_chars=_bounded_int(agent_section.get("max_output_chars"), default=4000, minimum=128),
        metadata={"repo": instance.get("repo", "local/baseline")},
    )


def _prepare_task_checkout(run_dir: Path, instance_id: str, instance: ConfigMap, *, force: bool) -> Path:
    checkouts = run_dir / "checkouts"
    checkouts.mkdir(parents=True, exist_ok=True)
    checkout = checkouts / _safe_filename(instance_id)
    official = instance.get("source") == OFFICIAL_INSTANCE_SOURCE
    if checkout.exists() and (force or official or bool(instance.get("fixture"))):
        shutil.rmtree(checkout)
    checkout.mkdir(parents=True, exist_ok=True)
    if official:
        return checkout
    fixture = instance.get("fixture")
    files = _fixture_files(fixture)
    for raw_path, content in files.items():
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ConfigError(f"Unsafe fixture path for baseline checkout: {raw_path}")
        target = (checkout / relative).resolve(strict=False)
        try:
            _ = target.relative_to(checkout.resolve())
        except ValueError as exc:
            raise ConfigError(f"Fixture path escapes baseline checkout: {raw_path}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        _ = target.write_text(str(content), encoding="utf-8")
    return checkout


def _fixture_files(fixture: object) -> dict[str, str]:
    if isinstance(fixture, dict):
        fixture_map = cast(dict[object, object], fixture)
        files = fixture_map.get("files")
        if isinstance(files, dict) and files:
            file_map = cast(dict[object, object], files)
            return {str(path): str(content) for path, content in file_map.items()}
    return {
        "README.md": "Local baseline smoke fixture for safe tool-only code repair.\n",
        "math_utils.py": "def add_numbers(left, right):\n    return left - right\n",
        "tests/test_math_utils.py": "from math_utils import add_numbers\n\n\ndef test_add_numbers_visible():\n    assert add_numbers(2, 3) == 5\n",
    }


def _strip_hidden_fields(value: object) -> object:
    if isinstance(value, dict):
        value_map = cast(dict[object, object], value)
        return {str(key): _strip_hidden_fields(item) for key, item in value_map.items() if str(key) not in {"patch", "test_patch"}}
    if isinstance(value, list):
        value_list = cast(list[object], value)
        return [_strip_hidden_fields(item) for item in value_list]
    return value


def _completed_instances(state: ConfigMap) -> list[str]:
    completed = state.get("completed_instances", [])
    if not isinstance(completed, list):
        return []
    completed_values = cast(list[object], completed)
    return [item for item in completed_values if isinstance(item, str)]


def _dry_trajectory_row(run_id: str, run_name: str, instance: ConfigMap) -> ConfigMap:
    instance_id = _instance_id(instance)
    return {
        "actions": ["load_config", "dry_run_noop", "write_prediction"],
        "event": "trajectory",
        "instance_id": instance_id,
        "run_id": run_id,
        "run_name": run_name,
        "status": "completed",
        "timestamp": _now(),
    }


def _dry_prediction_row(run_id: str, run_name: str, instance: ConfigMap) -> ConfigMap:
    instance_id = _instance_id(instance)
    return {
        "instance_id": instance_id,
        "model_name_or_path": "dry-run",
        "model_patch": "",
        "run_id": run_id,
        "run_name": run_name,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _instance_id(instance: ConfigMap) -> str:
    return require_string(instance.get("instance_id"), "Dry-run instance_id must be a string")


def _safe_filename(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    return cleaned or "instance"


def _bounded_int(value: object, *, default: int, minimum: int) -> int:
    if not isinstance(value, int):
        return default
    return max(minimum, value)


def _bounded_float(value: object, *, default: float, minimum: float) -> float:
    if not isinstance(value, int | float):
        return default
    return max(minimum, float(value))


def _typed_args(namespace: argparse.Namespace) -> CliArgs:
    config = require_string(cast(object, getattr(namespace, "config")), "--config must be a string")
    resources_value = cast(object, getattr(namespace, "resources"))
    if resources_value is not None and not isinstance(resources_value, str):
        raise ConfigError("--resources must be a string")
    resources = resources_value if isinstance(resources_value, str) else None
    dry_run_value = cast(object, getattr(namespace, "dry_run"))
    limit_value = cast(object, getattr(namespace, "limit"))
    run_id = require_string(cast(object, getattr(namespace, "run_id")), "--run-id must be a string")
    force_value = cast(object, getattr(namespace, "force"))
    if not isinstance(dry_run_value, bool):
        raise ConfigError("--dry-run must be a boolean")
    if limit_value is not None and not isinstance(limit_value, int):
        raise ConfigError("--limit must be an integer")
    if not isinstance(force_value, bool):
        raise ConfigError("--force must be a boolean")
    manifest_value = cast(object, getattr(namespace, "manifest"))
    if manifest_value is not None and not isinstance(manifest_value, str):
        raise ConfigError("--manifest must be a string")
    manifest = manifest_value if isinstance(manifest_value, str) else None
    split_value = cast(object, getattr(namespace, "instance_split"))
    if split_value is not None and split_value not in INSTANCE_SPLIT_CHOICES:
        raise ConfigError("--instance-split must be one of: main, smoke")
    instance_split = split_value if isinstance(split_value, str) else None
    strict_value = cast(object, getattr(namespace, "strict_official"))
    if not isinstance(strict_value, bool):
        raise ConfigError("--strict-official must be a boolean")
    return CliArgs(
        config=config,
        resources=resources,
        dry_run=dry_run_value,
        limit=limit_value,
        run_id=run_id,
        force=force_value,
        manifest=manifest,
        instance_split=instance_split,
        strict_official=strict_value,
    )


if __name__ == "__main__":
    raise SystemExit(main())
