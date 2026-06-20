from __future__ import annotations
# pyright: reportPrivateUsage=false, reportUnknownVariableType=false, reportUnusedCallResult=false

import argparse
import json
import shutil
import struct
import sys
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from repair_agent.agent.learning import LearningAgent
from repair_agent.config import ConfigError, ConfigMap, load_run_config, output_root, require_mapping, require_string
from repair_agent.env.swebench_loader import load_task_instances, load_task_manifest
from repair_agent.logging import append_jsonl, ensure_run_dir, initialize_run_files, write_json_atomic
from repair_agent.resources import load_device_inventory, load_resource_config, resolve_resource_plan
from repair_agent.run import (
    INSTANCE_SPLIT_CHOICES,
    OFFICIAL_METADATA_SOURCE,
    _agent_instances,
    _agent_task_from_config,
    _apply_limit,
    _assert_strict_official_id,
    _now,
    _official_instance_record,
    _prepare_task_checkout,
    _safe_filename,
)
from repair_agent.training.policy import ACTION_SCHEMA_VERSION, LinearSoftmaxPolicy, assert_frozen_tool_schema, policy_from_config
from repair_agent.training.pomdp import WeightedReward, load_reward_weights


NO_SIGNAL = "NO_SIGNAL"
COMPLETED = "COMPLETED"
RUN_STATE_COMPLETED = "completed"


@dataclass(frozen=True)
class TrainArgs:
    config: str
    limit: int | None
    episodes: int
    run_id: str
    resources: str | None
    dry_run_devices: bool
    manifest: str | None = None
    instance_split: str | None = None
    strict_official: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train REINFORCE+baseline tool-selection policy")
    _ = parser.add_argument("--config", required=True, help="Path to learning YAML config")
    _ = parser.add_argument("--limit", type=int, help="Limit task instances per episode")
    _ = parser.add_argument("--episodes", type=int, required=True, help="Number of local smoke training episodes")
    _ = parser.add_argument("--run-id", required=True, help="Output run identifier under outputs/runs")
    _ = parser.add_argument("--resources", help="Optional resources YAML path")
    _ = parser.add_argument("--dry-run-devices", action="store_true", help="Only write rollout GPU allocation plan")
    _ = parser.add_argument("--manifest", help="Path to SWE-bench Lite task manifest YAML (required with --strict-official)")
    _ = parser.add_argument("--instance-split", choices=list(INSTANCE_SPLIT_CHOICES), help="Official manifest split to evaluate in strict mode")
    _ = parser.add_argument("--strict-official", action="store_true", help="Load official SWE-bench Lite instances and ignore config fixtures")
    return parser


def main(argv: list[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    try:
        return train_from_args(_typed_args(namespace))
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


def train_from_args(args: TrainArgs) -> int:
    if args.episodes < 1:
        raise ConfigError("--episodes must be a positive integer")
    config_path = Path(args.config)
    config = load_run_config(config_path)
    _validate_learning_config(config)
    run_dir = ensure_run_dir(output_root(config), args.run_id)
    paths = initialize_run_files(run_dir, force=True)
    _ = shutil.copyfile(config_path, run_dir / "config.yaml")
    resources_path = Path(args.resources or "configs/resources.yaml")
    allocation = allocate_rollout_workers(resources_path, rollout_count=max(1, args.episodes * (args.limit or 1)))
    write_json_atomic(run_dir / "rollout_allocation.json", allocation)
    append_jsonl(run_dir / "resource_usage.jsonl", _resource_usage_row(allocation, resources_path))

    if args.dry_run_devices:
        status: ConfigMap = {"status": "DRY_RUN_DEVICES", "run_id": args.run_id, "rollout_allocation": allocation}
        write_json_atomic(run_dir / "status.json", status)
        write_json_atomic(paths["metrics"], {"status": "DRY_RUN_DEVICES", "rollout_allocation": allocation})
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0

    instances_override: list[ConfigMap] | None = None
    metadata: ConfigMap | None = None
    if args.strict_official:
        instances_override, metadata = _strict_official_instances(args)

    policy = policy_from_config(config)
    reward = WeightedReward(load_reward_weights(_reward_config_path(config)))
    curve = run_training_loop(
        config=config,
        policy=policy,
        reward=reward,
        run_dir=run_dir,
        paths=paths,
        run_id=args.run_id,
        episodes=args.episodes,
        limit=args.limit,
        allocation=allocation,
        instances_override=instances_override,
    )
    policy.save_json(run_dir / "policy.json")
    write_json_atomic(run_dir / "learning_curve.json", curve)
    write_learning_curve_png(run_dir / "learning_curve.png", cast(Sequence[float], curve["episode_returns"]))
    status = training_status(curve)
    write_json_atomic(run_dir / "status.json", status)
    metrics = {
        "completed": curve["rollout_count"],
        "learning": curve,
        "policy_checkpoint": str(run_dir / "policy.json"),
        "skipped": 0,
        "status": status["status"],
        "total": curve["rollout_count"],
    }
    write_json_atomic(paths["metrics"], metrics)
    if metadata is not None:
        _write_strict_run_state(paths["state"], args.run_id, instances_override or [], metadata)
    print(json.dumps(status, sort_keys=True))
    return 0


def _strict_official_instances(args: TrainArgs) -> tuple[list[ConfigMap], ConfigMap]:
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


def _write_strict_run_state(state_path: Path, run_id: str, instances: list[ConfigMap], metadata: ConfigMap) -> None:
    completed = sorted(require_string(instance.get("instance_id"), "Official instance_id must be a string") for instance in instances)
    payload: ConfigMap = {
        "completed_instances": completed,
        "dry_run": False,
        "last_updated": _now(),
        "metadata": dict(metadata),
        "run_id": run_id,
        "status": RUN_STATE_COMPLETED,
    }
    write_json_atomic(state_path, payload)


def run_training_loop(
    *,
    config: ConfigMap,
    policy: LinearSoftmaxPolicy,
    reward: WeightedReward,
    run_dir: Path,
    paths: Mapping[str, Path],
    run_id: str,
    episodes: int,
    limit: int | None,
    allocation: ConfigMap,
    instances_override: list[ConfigMap] | None = None,
) -> ConfigMap:
    selected = instances_override if instances_override is not None else _agent_instances(config, limit)
    agent_section = require_mapping(config.get("agent"), "Run config must define an 'agent' mapping")
    model_name = str(agent_section.get("model_name_or_path", "rule_based_local"))
    run_section = require_mapping(config.get("run"), "Run config must define a 'run' mapping")
    run_name = require_string(run_section.get("name"), "Run config field 'run.name' must be a string")
    rewards_path = run_dir / "rewards.jsonl"
    _ = rewards_path.write_text("", encoding="utf-8")
    episode_returns: list[float] = []
    update_summaries: list[ConfigMap] = []
    rollout_count = 0
    all_step_rewards: list[float] = []

    for episode_index in range(episodes):
        for instance_index, instance in enumerate(selected):
            instance_id = require_string(instance.get("instance_id"), "Learning instance_id must be a string")
            checkout_id = f"{instance_id}-episode-{episode_index}"
            checkout_root = _prepare_task_checkout(run_dir, checkout_id, instance, force=True)
            task = _agent_task_from_config(instance, agent_section, checkout_root, model_name)
            agent = LearningAgent(
                policy=policy,
                reward=reward,
                deterministic=True,
                visible_gpu_count=len(cast(list[object], allocation.get("healthy_visible_gpus", []))),
                rollout_parallelism=_int(allocation.get("rollout_parallelism"), 1),
                model_gate_status=str(config.get("model_gate_status", "pass")),
            )
            episode = agent.run_episode(task, run_id)
            transitions = [step.transition for step in episode.steps]
            update = policy.update_episode(transitions)
            rollout_return = sum(step.reward.total for step in episode.steps)
            episode_returns.append(rollout_return)
            update_summaries.append({str(key): value for key, value in update.items()})
            for step in episode.steps:
                all_step_rewards.append(step.reward.total)
                append_jsonl(
                    rewards_path,
                    {
                        "action": step.transition.action,
                        "episode_index": episode_index,
                        "event": "learning_reward",
                        "instance_id": instance_id,
                        "reward_total": step.reward.total,
                        "status": step.status,
                        "step_index": step.step_index,
                        "tool": step.tool,
                        "weighted_components": dict(step.reward.weighted_components),
                    },
                )
            patch_path = run_dir / "patches" / f"{_safe_filename(instance_id)}-ep{episode_index}.patch"
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            _ = patch_path.write_text(episode.result.final.model_patch, encoding="utf-8")
            for row in episode.result.trajectory_rows():
                row["episode_index"] = episode_index
                row["run_name"] = run_name
                append_jsonl(paths["trajectories"], row)
            prediction = episode.result.final.prediction_row()
            prediction["episode_index"] = episode_index
            append_jsonl(paths["predictions"], prediction)
            rollout_count += 1
            _ = instance_index

    return {
        "all_step_rewards_zero": all(abs(value) <= 1e-12 for value in all_step_rewards),
        "episode_returns": episode_returns,
        "mean_return": (sum(episode_returns) / len(episode_returns)) if episode_returns else 0.0,
        "policy_baseline_value": policy.baseline_value,
        "rollout_count": rollout_count,
        "schema": ACTION_SCHEMA_VERSION,
        "update_summaries": update_summaries,
    }


def training_status(curve: Mapping[str, object]) -> ConfigMap:
    returns = [float(item) for item in cast(Sequence[object], curve.get("episode_returns", [])) if isinstance(item, int | float)]
    no_signal = bool(curve.get("all_step_rewards_zero")) or not returns or all(abs(value) <= 1e-12 for value in returns)
    if no_signal:
        return {
            "recommendation": "Reward signal is all zero; inspect configs/rewards.yaml, smoke fixtures, and model/tool gates before scaling learning.",
            "status": NO_SIGNAL,
        }
    return {"recommendation": "Learning smoke produced non-zero shaped rewards and a policy checkpoint.", "status": COMPLETED}


def allocate_rollout_workers(
    resources_path: str | Path,
    *,
    rollout_count: int,
    inventory_path: str | Path = "outputs/device_inventory.json",
) -> ConfigMap:
    resources = load_resource_config(resources_path)
    inventory = load_device_inventory(inventory_path)
    plan = resolve_resource_plan(resources, inventory, str(inventory_path))
    trainer = require_mapping(resources.get("trainer_devices"), "Resource config must define trainer_devices")
    rollout_gpus = _int_list(trainer.get("rollout_gpus", []))
    visible = [gpu for gpu in plan.visible_gpus if not rollout_gpus or gpu in rollout_gpus]
    fallback = dict(plan.fallback)
    expected = _int_list(require_mapping(resources.get("gpus"), "Resource config must define gpus").get("expected_ids", []))
    workers: list[ConfigMap] = []
    if visible:
        for index, gpu_id in enumerate(visible):
            workers.append(
                {
                    "assigned_device": f"cuda:{gpu_id}",
                    "cuda_visible_devices": str(gpu_id),
                    "gpu_id": gpu_id,
                    "worker_id": f"rollout-{index}",
                }
            )
    else:
        fallback["reasons"] = [*cast(list[object], fallback.get("reasons", [])), "no_healthy_visible_gpus_for_rollout"]
        workers.append({"assigned_device": "cpu", "cuda_visible_devices": "", "gpu_id": None, "worker_id": "rollout-cpu"})
    for index, worker in enumerate(workers):
        worker["planned_rollouts"] = rollout_count // len(workers) + (1 if index < rollout_count % len(workers) else 0)
    return {
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "expected_gpu_ids": expected,
        "fallback": fallback,
        "healthy_visible_gpus": visible,
        "missing_gpus": [gpu_id for gpu_id in expected if gpu_id not in plan.visible_gpus],
        "resource_plan": plan.to_record(),
        "rollout_parallelism": min(len(workers), _int(plan.worker_settings.get("rollout_parallelism"), len(workers))),
        "workers": workers,
    }


def write_learning_curve_png(path: str | Path, returns: Sequence[float]) -> None:
    width, height = 160, 80
    pixels = bytearray([255, 255, 255] * width * height)
    values = list(returns) or [0.0]
    low, high = min(values), max(values)
    spread = high - low if high != low else 1.0
    points: list[tuple[int, int]] = []
    for index, value in enumerate(values):
        x = 5 + int(index * (width - 10) / max(1, len(values) - 1))
        y = height - 6 - int(((value - low) / spread) * (height - 12))
        points.append((x, y))
    for left, right in zip(points, points[1:], strict=False):
        _draw_line(pixels, width, height, left, right, (31, 119, 180))
    for point in points:
        _draw_point(pixels, width, height, point, (214, 39, 40))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _ = target.write_bytes(_png_bytes(width, height, pixels))


def _validate_learning_config(config: Mapping[str, object]) -> None:
    agent = require_mapping(config.get("agent"), "Learning config must define agent")
    if str(agent.get("type", "")).lower() != "learning":
        raise ConfigError("Learning trainer requires agent.type: learning")
    learning = require_mapping(config.get("learning"), "Learning config must define learning")
    assert_frozen_tool_schema([str(item) for item in cast(list[object], learning.get("action_vocabulary", []))])
    if learning.get("tool_schema_version") != ACTION_SCHEMA_VERSION:
        raise ConfigError("learning.tool_schema_version must match safe-tool-selection-v1")


def _reward_config_path(config: Mapping[str, object]) -> str:
    learning = require_mapping(config.get("learning"), "Learning config must define learning")
    value = learning.get("reward_config", "configs/rewards.yaml")
    return str(value) if isinstance(value, str) else "configs/rewards.yaml"


def _resource_usage_row(allocation: Mapping[str, object], resources_path: Path) -> ConfigMap:
    plan = cast(dict[str, object], allocation.get("resource_plan", {}))
    row = {str(key): value for key, value in plan.items()}
    row.update({"event": "learning_rollout_allocation", "resources_path": str(resources_path), "rollout_allocation": dict(allocation)})
    return row


def _typed_args(namespace: argparse.Namespace) -> TrainArgs:
    config = require_string(cast(object, getattr(namespace, "config")), "--config must be a string")
    run_id = require_string(cast(object, getattr(namespace, "run_id")), "--run-id must be a string")
    episodes = cast(object, getattr(namespace, "episodes"))
    limit = cast(object, getattr(namespace, "limit"))
    resources = cast(object, getattr(namespace, "resources"))
    dry_run_devices = cast(object, getattr(namespace, "dry_run_devices"))
    if not isinstance(episodes, int):
        raise ConfigError("--episodes must be an integer")
    if limit is not None and not isinstance(limit, int):
        raise ConfigError("--limit must be an integer")
    if resources is not None and not isinstance(resources, str):
        raise ConfigError("--resources must be a string")
    if not isinstance(dry_run_devices, bool):
        raise ConfigError("--dry-run-devices must be a boolean")
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
    return TrainArgs(
        config=config,
        limit=limit,
        episodes=episodes,
        run_id=run_id,
        resources=resources,
        dry_run_devices=dry_run_devices,
        manifest=manifest,
        instance_split=instance_split,
        strict_official=strict_value,
    )


def _int(value: object, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, int) and not isinstance(item, bool)]


def _draw_point(pixels: bytearray, width: int, height: int, point: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = point
    for yy in range(max(0, y - 1), min(height, y + 2)):
        for xx in range(max(0, x - 1), min(width, x + 2)):
            offset = (yy * width + xx) * 3
            pixels[offset:offset + 3] = bytes(color)


def _draw_line(pixels: bytearray, width: int, height: int, left: tuple[int, int], right: tuple[int, int], color: tuple[int, int, int]) -> None:
    x0, y0 = left
    x1, y1 = right
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for step in range(steps + 1):
        x = round(x0 + (x1 - x0) * step / steps)
        y = round(y0 + (y1 - y0) * step / steps)
        if 0 <= x < width and 0 <= y < height:
            offset = (y * width + x) * 3
            pixels[offset:offset + 3] = bytes(color)


def _png_bytes(width: int, height: int, pixels: bytearray) -> bytes:
    raw = b"".join(b"\x00" + bytes(pixels[y * width * 3:(y + 1) * width * 3]) for y in range(height))
    def chunk(name: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


if __name__ == "__main__":
    raise SystemExit(main())
