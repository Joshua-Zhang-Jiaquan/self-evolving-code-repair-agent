from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from repair_agent.evaluation.metrics import RESOLVED_STATUSES
from repair_agent.logging import JsonMap, read_jsonl, write_json_atomic
from repair_agent.training.policy import ACTION_VOCABULARY, compute_returns


CREDIT_SCHEMA_VERSION = "credit-assignment-diagnostic-v1"
ANALYSIS_LABEL = "diagnostic_correlational_not_causal"
REWARD_COMPONENT_KEYS = (
    "pass",
    "visible_test_pass",
    "visible_test_failure",
    "hidden_regression_ready",
    "partial_progress",
    "test_runs",
    "timeout",
)


@dataclass(frozen=True)
class StepCredit:
    run_id: str
    instance_id: str
    episode_index: str
    step_index: int
    action: str
    tool: str
    status: str
    final_status: str
    reward: float
    reward_to_go: float
    success: bool
    partial_progress: float
    test_signal: float
    components: Mapping[str, float]
    reward_source: str

    @property
    def trajectory_key(self) -> tuple[str, str, str]:
        return (self.run_id, self.instance_id, self.episode_index)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute long-horizon credit-assignment diagnostics for repair_agent runs")
    _ = parser.add_argument("--runs", required=True, help="Directory containing run subdirectories")
    _ = parser.add_argument("--out", required=True, help="Credit-assignment JSON output path")
    _ = parser.add_argument("--gamma", type=float, default=1.0, help="Discount for reward-to-go; defaults to Task 9 gamma=1.0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        out = Path(cast(str, getattr(args, "out")))
        gamma = _float(cast(object, getattr(args, "gamma")), 1.0)
        summary = summarize_credit_assignment(Path(cast(str, getattr(args, "runs"))), gamma=gamma)
        artifacts = write_credit_artifacts(out, summary)
        summary["artifacts"] = artifacts
        write_json_atomic(out, summary)
    except (OSError, ValueError) as exc:
        print(f"credit assignment error: {exc}", file=sys.stderr)
        return 2
    return 0


def summarize_credit_assignment(runs_root: Path, *, gamma: float = 1.0) -> JsonMap:
    if not runs_root.is_dir():
        raise ValueError(f"runs directory not found: {runs_root}")
    steps = load_step_credits(runs_root, gamma=gamma)
    trajectories = _trajectory_groups(steps)
    action_summary = action_type_summary(steps, trajectories)
    leave_one_out = leave_one_action_type_out(steps, trajectories)
    for row in action_summary:
        action = str(row["action"])
        row["leave_one_out"] = leave_one_out.get(action, _empty_leave_one_out())
    position_rows = position_summary(steps)
    action_position_rows = action_position_summary(steps)
    summary: JsonMap = {
        "schema_version": CREDIT_SCHEMA_VERSION,
        "analysis_label": ANALYSIS_LABEL,
        "interpretation": "Reward-to-go, leave-one-action-type-out, and correlations are diagnostic summaries of logged trajectories; they are not causal proof.",
        "runs_root": str(runs_root),
        "gamma": gamma,
        "trajectory_count": len(trajectories),
        "step_count": len(steps),
        "reward_source_counts": dict(sorted(Counter(step.reward_source for step in steps).items())),
        "top_positive_action_types": _rank_actions(action_summary, reverse=True),
        "top_negative_action_types": _rank_actions(action_summary, reverse=False),
        "action_type_summary": action_summary,
        "leave_one_action_type_out": leave_one_out,
        "position_summary": position_rows,
        "per_position_contribution_summary": position_rows,
        "action_position_summary": action_position_rows,
        "action_position_correlations": action_position_correlations(steps),
        "correlations": correlations(steps),
        "tool_contribution_table": tool_contribution_table(steps),
        "test_contribution_table": component_contribution_table(steps),
        "skipped_note": "Runs or steps without reward_total in rewards.jsonl or trajectory metadata are skipped for credit diagnostics.",
    }
    return summary


def annotate_reward_to_go(rows: Sequence[Mapping[str, object]], *, gamma: float = 1.0) -> list[JsonMap]:
    ordered = sorted(rows, key=lambda row: _int(row.get("step_index")))
    rewards = [_float(row.get("reward_total"), 0.0) for row in ordered]
    returns = compute_returns(rewards, gamma=gamma)
    annotated: list[JsonMap] = []
    for row, reward_to_go in zip(ordered, returns, strict=True):
        item = dict(row)
        item["reward_to_go"] = reward_to_go
        annotated.append(item)
    return annotated


def load_step_credits(runs_root: Path, *, gamma: float = 1.0) -> list[StepCredit]:
    steps: list[StepCredit] = []
    for run_dir in _run_directories(runs_root):
        raw_rows = _reward_rows_for_run(run_dir)
        for rows in _raw_trajectory_groups(raw_rows).values():
            annotated = annotate_reward_to_go(rows, gamma=gamma)
            for row in annotated:
                steps.append(_step_credit_from_row(row))
    return sorted(steps, key=lambda step: (step.run_id, step.instance_id, step.episode_index, step.step_index, step.action))


def action_type_summary(steps: Sequence[StepCredit], trajectories: Mapping[tuple[str, str, str], Sequence[StepCredit]]) -> list[JsonMap]:
    by_action: dict[str, list[StepCredit]] = defaultdict(list)
    for step in steps:
        by_action[step.action].append(step)
    rows: list[JsonMap] = []
    for action in _ordered_actions(by_action):
        action_steps = by_action[action]
        trajectory_keys = {step.trajectory_key for step in action_steps}
        success_count = sum(1 for key in trajectory_keys if _trajectory_success(trajectories[key]))
        rows.append(
            {
                "action": action,
                "step_count": len(action_steps),
                "trajectory_count": len(trajectory_keys),
                "success_rate_when_present": _safe_rate(success_count, len(trajectory_keys)),
                "mean_step_reward": _mean(step.reward for step in action_steps),
                "total_step_reward": round(sum(step.reward for step in action_steps), 6),
                "mean_reward_to_go": _mean(step.reward_to_go for step in action_steps),
                "partial_progress_mean": _mean(step.partial_progress for step in action_steps),
                "test_signal_mean": _mean(step.test_signal for step in action_steps),
            }
        )
    return rows


def leave_one_action_type_out(
    steps: Sequence[StepCredit],
    trajectories: Mapping[tuple[str, str, str], Sequence[StepCredit]],
) -> JsonMap:
    if not trajectories:
        return {}
    total_returns = {key: sum(step.reward for step in rows) for key, rows in trajectories.items()}
    baseline_mean = _mean(total_returns.values())
    actions = _ordered_actions({step.action: [] for step in steps})
    output: JsonMap = {}
    for action in actions:
        removed_by_trajectory: dict[tuple[str, str, str], float] = defaultdict(float)
        for step in steps:
            if step.action == action:
                removed_by_trajectory[step.trajectory_key] += step.reward
        without = [total - removed_by_trajectory.get(key, 0.0) for key, total in total_returns.items()]
        without_mean = _mean(without)
        output[action] = {
            "analysis_label": ANALYSIS_LABEL,
            "mean_return_without_action_type": without_mean,
            "mean_return_delta_vs_logged": round(baseline_mean - without_mean, 6),
            "total_removed_reward": round(sum(removed_by_trajectory.values()), 6),
            "trajectory_count_with_action": len(removed_by_trajectory),
        }
    return output


def position_summary(steps: Sequence[StepCredit]) -> list[JsonMap]:
    by_position: dict[int, list[StepCredit]] = defaultdict(list)
    for step in steps:
        by_position[step.step_index].append(step)
    rows: list[JsonMap] = []
    for position in sorted(by_position):
        position_steps = by_position[position]
        rows.append(
            {
                "step_index": position,
                "step_count": len(position_steps),
                "success_rate": _safe_rate(sum(1 for step in position_steps if step.success), len(position_steps)),
                "mean_step_reward": _mean(step.reward for step in position_steps),
                "mean_reward_to_go": _mean(step.reward_to_go for step in position_steps),
                "partial_progress_mean": _mean(step.partial_progress for step in position_steps),
                "test_signal_mean": _mean(step.test_signal for step in position_steps),
                "top_action": _counter_top(step.action for step in position_steps),
            }
        )
    return rows


def action_position_summary(steps: Sequence[StepCredit]) -> list[JsonMap]:
    by_pair: dict[tuple[str, int], list[StepCredit]] = defaultdict(list)
    for step in steps:
        by_pair[(step.action, step.step_index)].append(step)
    rows: list[JsonMap] = []
    for action, position in sorted(by_pair, key=lambda item: (_action_sort_key(item[0]), item[1])):
        pair_steps = by_pair[(action, position)]
        rows.append(
            {
                "action": action,
                "step_index": position,
                "step_count": len(pair_steps),
                "success_rate": _safe_rate(sum(1 for step in pair_steps if step.success), len(pair_steps)),
                "mean_reward_to_go": _mean(step.reward_to_go for step in pair_steps),
                "partial_progress_mean": _mean(step.partial_progress for step in pair_steps),
            }
        )
    return rows


def action_position_correlations(steps: Sequence[StepCredit]) -> list[JsonMap]:
    by_action: dict[str, list[StepCredit]] = defaultdict(list)
    for step in steps:
        by_action[step.action].append(step)
    rows: list[JsonMap] = []
    for action in _ordered_actions(by_action):
        action_steps = by_action[action]
        positions = [float(step.step_index) for step in action_steps]
        rows.append(
            {
                "action": action,
                "step_count": len(action_steps),
                "position_vs_success": _pearson(positions, [1.0 if step.success else 0.0 for step in action_steps]),
                "position_vs_partial_progress": _pearson(positions, [step.partial_progress for step in action_steps]),
                "position_vs_reward_to_go": _pearson(positions, [step.reward_to_go for step in action_steps]),
            }
        )
    return rows


def correlations(steps: Sequence[StepCredit]) -> JsonMap:
    positions = [float(step.step_index) for step in steps]
    success = [1.0 if step.success else 0.0 for step in steps]
    partial = [step.partial_progress for step in steps]
    reward_to_go = [step.reward_to_go for step in steps]
    return {
        "analysis_label": ANALYSIS_LABEL,
        "position_vs_success": _pearson(positions, success),
        "position_vs_partial_progress": _pearson(positions, partial),
        "position_vs_reward_to_go": _pearson(positions, reward_to_go),
        "reward_to_go_vs_success": _pearson(reward_to_go, success),
        "reward_to_go_vs_partial_progress": _pearson(reward_to_go, partial),
    }


def tool_contribution_table(steps: Sequence[StepCredit]) -> list[JsonMap]:
    by_tool: dict[str, list[StepCredit]] = defaultdict(list)
    for step in steps:
        by_tool[step.tool].append(step)
    rows: list[JsonMap] = []
    for tool in sorted(by_tool):
        tool_steps = by_tool[tool]
        rows.append(
            {
                "tool": tool,
                "step_count": len(tool_steps),
                "mean_step_reward": _mean(step.reward for step in tool_steps),
                "total_step_reward": round(sum(step.reward for step in tool_steps), 6),
                "mean_reward_to_go": _mean(step.reward_to_go for step in tool_steps),
                "success_rate": _safe_rate(sum(1 for step in tool_steps if step.success), len(tool_steps)),
            }
        )
    return rows


def component_contribution_table(steps: Sequence[StepCredit]) -> list[JsonMap]:
    rows: list[JsonMap] = []
    for component in REWARD_COMPONENT_KEYS:
        values = [float(step.components.get(component, 0.0)) for step in steps]
        rows.append(
            {
                "component": component,
                "nonzero_count": sum(1 for value in values if abs(value) > 1e-12),
                "mean_weighted_value": _mean(values),
                "total_weighted_value": round(sum(values), 6),
            }
        )
    return rows


def write_credit_artifacts(out: Path, summary: Mapping[str, object]) -> JsonMap:
    out.parent.mkdir(parents=True, exist_ok=True)
    table_path = out.with_name(f"{out.stem}_tables.md")
    _ = table_path.write_text(_markdown_tables(summary), encoding="utf-8")
    return {"table": str(table_path)}


def _reward_rows_for_run(run_dir: Path) -> list[JsonMap]:
    trajectories = read_jsonl(run_dir / "trajectories.jsonl")
    trajectory_by_key = {_row_key(row): row for row in trajectories}
    rewards = read_jsonl(run_dir / "rewards.jsonl")
    rows: list[JsonMap] = []
    if rewards:
        for reward in rewards:
            key = _row_key(reward)
            trajectory = trajectory_by_key.get(key, {})
            rows.append(_merged_reward_row(run_dir.name, reward, trajectory, "rewards.jsonl"))
        return rows
    for trajectory in trajectories:
        metadata = trajectory.get("metadata")
        metadata_map = cast(dict[object, object], metadata) if isinstance(metadata, dict) else {}
        if not isinstance(metadata_map.get("reward_total"), int | float):
            continue
        rows.append(_merged_reward_row(run_dir.name, trajectory, trajectory, "trajectory_metadata"))
    return rows


def _merged_reward_row(run_id: str, reward: Mapping[str, object], trajectory: Mapping[str, object], source: str) -> JsonMap:
    metadata = trajectory.get("metadata")
    metadata_map = cast(dict[object, object], metadata) if isinstance(metadata, dict) else {}
    components: object = reward.get("weighted_components")
    if not isinstance(components, dict):
        components = metadata_map.get("weighted_reward_components")
    if not isinstance(components, dict):
        components = metadata_map.get("reward_components")
    reward_total = reward.get("reward_total")
    if not isinstance(reward_total, int | float):
        reward_total = metadata_map.get("reward_total")
    row: JsonMap = {
        "action": str(reward.get("action") or trajectory.get("action") or reward.get("tool") or trajectory.get("tool") or "unknown"),
        "components": _components_from_object(cast(object, components)),
        "episode_index": str(reward.get("episode_index", trajectory.get("episode_index", "0"))),
        "final_status": str(trajectory.get("final_status") or reward.get("final_status") or reward.get("status") or ""),
        "instance_id": str(reward.get("instance_id") or trajectory.get("instance_id") or ""),
        "reward_source": source,
        "reward_total": _float(reward_total, 0.0),
        "run_id": str(reward.get("run_id") or trajectory.get("run_id") or run_id),
        "status": str(reward.get("status") or trajectory.get("status") or ""),
        "step_index": _int(reward.get("step_index", trajectory.get("step_index", 0))),
        "tool": str(reward.get("tool") or trajectory.get("tool") or reward.get("action") or trajectory.get("action") or "unknown"),
    }
    return row


def _step_credit_from_row(row: Mapping[str, object]) -> StepCredit:
    component_map = _components_from_object(row.get("components"))
    final_status = str(row.get("final_status", "")).lower()
    return StepCredit(
        run_id=str(row.get("run_id", "")),
        instance_id=str(row.get("instance_id", "")),
        episode_index=str(row.get("episode_index", "0")),
        step_index=_int(row.get("step_index")),
        action=str(row.get("action") or "unknown"),
        tool=str(row.get("tool") or row.get("action") or "unknown"),
        status=str(row.get("status", "")),
        final_status=final_status,
        reward=_float(row.get("reward_total"), 0.0),
        reward_to_go=_float(row.get("reward_to_go"), 0.0),
        success=final_status in RESOLVED_STATUSES,
        partial_progress=float(component_map.get("partial_progress", 0.0)),
        test_signal=sum(float(component_map.get(key, 0.0)) for key in ("visible_test_pass", "visible_test_failure", "hidden_regression_ready", "test_runs", "pass")),
        components=component_map,
        reward_source=str(row.get("reward_source", "unknown")),
    )


def _run_directories(runs_root: Path) -> list[Path]:
    return sorted(path for path in runs_root.iterdir() if path.is_dir() and ((path / "rewards.jsonl").exists() or (path / "trajectories.jsonl").exists()))


def _row_key(row: Mapping[str, object]) -> tuple[str, str, int]:
    return (str(row.get("instance_id", "")), str(row.get("episode_index", "0")), _int(row.get("step_index")))


def _raw_trajectory_groups(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str, str], list[Mapping[str, object]]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("run_id", "")), str(row.get("instance_id", "")), str(row.get("episode_index", "0")))].append(row)
    return grouped


def _trajectory_groups(steps: Sequence[StepCredit]) -> dict[tuple[str, str, str], list[StepCredit]]:
    grouped: dict[tuple[str, str, str], list[StepCredit]] = defaultdict(list)
    for step in steps:
        grouped[step.trajectory_key].append(step)
    return {key: sorted(value, key=lambda step: step.step_index) for key, value in grouped.items()}


def _trajectory_success(steps: Sequence[StepCredit]) -> bool:
    return any(step.success for step in steps)


def _rank_actions(action_summary_rows: Sequence[Mapping[str, object]], *, reverse: bool) -> list[JsonMap]:
    ranked = sorted(
        action_summary_rows,
        key=lambda row: (_float(_leave_one_out_delta(row), 0.0), str(row.get("action"))),
        reverse=reverse,
    )
    if not reverse:
        ranked = sorted(action_summary_rows, key=lambda row: (_float(_leave_one_out_delta(row), 0.0), str(row.get("action"))))
    output: list[JsonMap] = []
    for row in ranked[:5]:
        output.append(
            {
                "action": str(row.get("action", "")),
                "mean_return_delta_vs_logged": _float(_leave_one_out_delta(row), 0.0),
                "mean_reward_to_go": _float(row.get("mean_reward_to_go"), 0.0),
                "step_count": _int(row.get("step_count")),
            }
        )
    return output


def _leave_one_out_delta(row: Mapping[str, object]) -> object:
    leave_one = row.get("leave_one_out")
    if isinstance(leave_one, dict):
        return cast(dict[object, object], leave_one).get("mean_return_delta_vs_logged")
    return 0.0


def _empty_leave_one_out() -> JsonMap:
    return {
        "analysis_label": ANALYSIS_LABEL,
        "mean_return_without_action_type": 0.0,
        "mean_return_delta_vs_logged": 0.0,
        "total_removed_reward": 0.0,
        "trajectory_count_with_action": 0,
    }


def _markdown_tables(summary: Mapping[str, object]) -> str:
    action_rows = cast(list[Mapping[str, object]], summary.get("action_type_summary", []))
    position_rows = cast(list[Mapping[str, object]], summary.get("position_summary", []))
    component_rows = cast(list[Mapping[str, object]], summary.get("test_contribution_table", []))
    lines = [
        "# Long-Horizon Credit Assignment Diagnostics",
        "",
        "These tables are diagnostic/correlational summaries of logged rewards, not causal proof.",
        "",
        "## Action types",
        "",
        "| action | steps | mean reward-to-go | leave-one-out Δ | partial progress |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in action_rows:
        leave_one = row.get("leave_one_out")
        delta = cast(dict[object, object], leave_one).get("mean_return_delta_vs_logged") if isinstance(leave_one, dict) else 0.0
        lines.append(
            f"| {row.get('action', '')} | {_int(row.get('step_count'))} | {_float(row.get('mean_reward_to_go'), 0.0):.6f} | {_float(delta, 0.0):.6f} | {_float(row.get('partial_progress_mean'), 0.0):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Positions",
            "",
            "| step index | steps | mean reward-to-go | success rate | partial progress | top action |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in position_rows:
        lines.append(
            f"| {_int(row.get('step_index'))} | {_int(row.get('step_count'))} | {_float(row.get('mean_reward_to_go'), 0.0):.6f} | {_display_optional(row.get('success_rate'))} | {_float(row.get('partial_progress_mean'), 0.0):.6f} | {row.get('top_action', '')} |"
        )
    lines.extend(
        [
            "",
            "## Test/reward components",
            "",
            "| component | nonzero count | total weighted value | mean weighted value |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in component_rows:
        lines.append(
            f"| {row.get('component', '')} | {_int(row.get('nonzero_count'))} | {_float(row.get('total_weighted_value'), 0.0):.6f} | {_float(row.get('mean_weighted_value'), 0.0):.6f} |"
        )
    return "\n".join(lines) + "\n"


def _float_components(value: Mapping[object, object]) -> dict[str, float]:
    components: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, int | float) and not isinstance(item, bool):
            components[str(key)] = float(item)
    return components


def _components_from_object(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return _float_components(cast(dict[object, object], value))


def _ordered_actions(by_action: Mapping[str, object]) -> list[str]:
    return sorted(by_action, key=_action_sort_key)


def _action_sort_key(action: str) -> tuple[int, str]:
    try:
        return (ACTION_VOCABULARY.index(action), action)
    except ValueError:
        return (len(ACTION_VOCABULARY), action)


def _counter_top(values: Iterable[str]) -> str | None:
    counts = Counter(values)
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], _action_sort_key(item[0])))[0][0]


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return round(sum(items) / len(items), 6) if items else 0.0


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    if left_var <= 1e-12 or right_var <= 1e-12:
        return None
    return round(numerator / math.sqrt(left_var * right_var), 6)


def _display_optional(value: object) -> str:
    return "n/a" if value is None else f"{_float(value, 0.0):.6f}"


def _int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    return int(value) if isinstance(value, int | float) else default


def _float(value: object, default: float) -> float:
    if isinstance(value, bool):
        return default
    return float(value) if isinstance(value, int | float) else default


if __name__ == "__main__":
    raise SystemExit(main())
