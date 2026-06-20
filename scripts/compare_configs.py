#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repair_agent.config import ConfigError, ConfigMap, load_yaml_config, require_mapping


BUDGET_FIELDS = (
    "agent.max_steps",
    "agent.max_test_runs",
    "agent.test_timeout_seconds",
    "agent.max_output_chars",
    "agent.model_name_or_path",
)


@dataclass(frozen=True)
class ConfigDiff:
    path: str
    left: object
    right: object


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare repair-agent YAML configs for fair budgets")
    _ = parser.add_argument("left", help="Reference YAML config")
    _ = parser.add_argument("right", help="Candidate YAML config")
    _ = parser.add_argument("--check-budget-equal", action="store_true", help="Require agent budgets and model setting to match")
    _ = parser.add_argument("--allowed-diff", action="append", default=[], help="Config dotted path allowed to differ for general comparisons")
    return parser


def main(argv: list[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    left_path = Path(cast(str, namespace.left))
    right_path = Path(cast(str, namespace.right))
    check_budget: bool = bool(cast(bool, namespace.check_budget_equal))
    allowed_diffs: list[str] = cast(list[str], namespace.allowed_diff)
    try:
        left = load_yaml_config(left_path)
        right = load_yaml_config(right_path)
        diffs: list[ConfigDiff] = []
        if check_budget:
            diffs.extend(compare_budget_fields(left, right))
            diffs.extend(compare_task_budget_shape(left, right))
        elif allowed_diffs:
            diffs.extend(diff_configs(left, right, allowed=set(allowed_diffs)))
    except (ConfigError, ValueError) as exc:
        print(f"config comparison error: {exc}", file=sys.stderr)
        return 2
    if diffs:
        for diff in diffs:
            print(f"diff {diff.path}: {diff.left!r} != {diff.right!r}", file=sys.stderr)
        return 1
    checked = ", ".join(BUDGET_FIELDS) if check_budget else "requested fields"
    print(f"configs comparable: {left_path} vs {right_path}; checked {checked}")
    return 0


def compare_budget_fields(left: ConfigMap, right: ConfigMap) -> list[ConfigDiff]:
    diffs: list[ConfigDiff] = []
    for path in BUDGET_FIELDS:
        left_value = get_dotted(left, path)
        right_value = get_dotted(right, path)
        if left_value != right_value:
            diffs.append(ConfigDiff(path, left_value, right_value))
    return diffs


def compare_task_budget_shape(left: ConfigMap, right: ConfigMap) -> list[ConfigDiff]:
    left_agent = require_mapping(left.get("agent"), "left config must define agent mapping")
    right_agent = require_mapping(right.get("agent"), "right config must define agent mapping")
    left_instances = _instances(left_agent)
    right_instances = _instances(right_agent)
    diffs: list[ConfigDiff] = []
    if len(left_instances) != len(right_instances):
        diffs.append(ConfigDiff("agent.instances.length", len(left_instances), len(right_instances)))
    left_tests = [_visible_test_count(item) for item in left_instances]
    right_tests = [_visible_test_count(item) for item in right_instances]
    if left_tests != right_tests:
        diffs.append(ConfigDiff("agent.instances.visible_tests.lengths", left_tests, right_tests))
    return diffs


def diff_configs(left: object, right: object, *, allowed: set[str], prefix: str = "") -> list[ConfigDiff]:
    if prefix in allowed or _parent_of_allowed(prefix, allowed):
        return []
    if isinstance(left, dict) and isinstance(right, dict):
        left_d = cast(dict[object, object], left)
        right_d = cast(dict[object, object], right)
        result: list[ConfigDiff] = []
        keys = sorted(set(left_d) | set(right_d), key=str)
        for key in keys:
            path = f"{prefix}.{key}" if prefix else str(key)
            result.extend(diff_configs(left_d.get(key), right_d.get(key), allowed=allowed, prefix=path))
        return result
    if isinstance(left, list) and isinstance(right, list):
        left_l = cast(list[object], left)
        right_l = cast(list[object], right)
        list_result: list[ConfigDiff] = []
        for index, (left_item, right_item) in enumerate(zip(left_l, right_l, strict=False)):
            path = f"{prefix}[{index}]"
            list_result.extend(diff_configs(left_item, right_item, allowed=allowed, prefix=path))
        if len(left_l) != len(right_l):
            list_result.append(ConfigDiff(f"{prefix}.length", len(left_l), len(right_l)))
        return list_result
    return [] if left == right else [ConfigDiff(prefix, left, right)]  # pyright: ignore[reportUnknownArgumentType]


def _parent_of_allowed(prefix: str, allowed: set[str]) -> bool:
    """Return True when *prefix* is an ancestor of any path in *allowed*.

    Example: prefix="reward" and allowed={"reward.process_weight"} → True.
    """
    if not prefix:
        return False
    dot_prefix = prefix + "."
    return any(a.startswith(dot_prefix) for a in allowed)


def get_dotted(config: ConfigMap, path: str) -> object:
    current: object = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"missing config field: {path}")
        current = cast(object, current[part])  # pyright: ignore[reportUnnecessaryCast]
    return current


def _instances(agent: ConfigMap) -> list[ConfigMap]:
    value = agent.get("instances", agent.get("tasks"))
    if not isinstance(value, list):
        raise ValueError("agent.instances must be a list")
    return [require_mapping(item, "agent instance must be a mapping") for item in cast(list[object], value)]


def _visible_test_count(instance: ConfigMap) -> int:
    value = instance.get("visible_tests", [])
    if not isinstance(value, list):
        return 0
    return len(cast(list[object], value))


if __name__ == "__main__":
    raise SystemExit(main())
