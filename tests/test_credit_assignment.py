from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from repair_agent.evaluation.credit import (
    StepCredit,
    action_type_summary,
    annotate_reward_to_go,
    leave_one_action_type_out,
    load_step_credits,
    position_summary,
    summarize_credit_assignment,
    write_credit_artifacts,
)


def test_annotate_reward_to_go_uses_policy_return_semantics():
    rows = [
        {"step_index": 2, "reward_total": 3.0},
        {"step_index": 0, "reward_total": 1.0},
        {"step_index": 1, "reward_total": 2.0},
    ]

    annotated = annotate_reward_to_go(rows, gamma=0.5)

    assert [row["step_index"] for row in annotated] == [0, 1, 2]
    assert [row["reward_to_go"] for row in annotated] == [2.75, 3.5, 3.0]


def test_credit_summary_aggregates_actions_and_leave_one_out(tmp_path: Path):
    runs = tmp_path / "runs"
    run_dir = runs / "learning_unit"
    run_dir.mkdir(parents=True)
    _write_jsonl(
        run_dir / "rewards.jsonl",
        [
            _reward("case-1", 0, "search", 1.0, partial=0.2),
            _reward("case-1", 1, "read_file", 2.0, partial=0.4),
            _reward("case-1", 2, "run_tests", -1.0, visible=-1.0),
            _reward("case-2", 0, "search", 0.5, partial=0.1),
            _reward("case-2", 1, "edit_file", 3.0, partial=1.0),
        ],
    )
    _write_jsonl(
        run_dir / "trajectories.jsonl",
        [
            _trajectory("case-1", 0, "search", "passed"),
            _trajectory("case-1", 1, "read_file", "passed"),
            _trajectory("case-1", 2, "run_tests", "passed"),
            _trajectory("case-2", 0, "search", "no_patch"),
            _trajectory("case-2", 1, "edit_file", "no_patch"),
        ],
    )

    summary = summarize_credit_assignment(runs)

    assert summary["analysis_label"] == "diagnostic_correlational_not_causal"
    assert summary["trajectory_count"] == 2
    assert summary["step_count"] == 5
    action_rows = {str(row["action"]): row for row in cast(list[dict[str, object]], summary["action_type_summary"])}
    assert action_rows["search"]["step_count"] == 2
    assert action_rows["search"]["success_rate_when_present"] == 0.5
    assert action_rows["edit_file"]["mean_reward_to_go"] == 3.0
    leave_one = cast(dict[str, dict[str, object]], summary["leave_one_action_type_out"])
    assert leave_one["edit_file"]["mean_return_delta_vs_logged"] == 1.5
    top_positive = cast(list[dict[str, object]], summary["top_positive_action_types"])
    assert top_positive[0]["action"] == "edit_file"
    assert "per_position_contribution_summary" in summary


def test_position_summary_reports_success_and_partial_progress(tmp_path: Path):
    runs = tmp_path / "runs"
    run_dir = runs / "learning_positions"
    run_dir.mkdir(parents=True)
    _write_jsonl(
        run_dir / "rewards.jsonl",
        [
            _reward("case-a", 0, "search", 1.0, partial=0.0),
            _reward("case-a", 1, "run_tests", 2.0, partial=0.5, visible=1.0),
            _reward("case-b", 0, "search", 0.0, partial=0.0),
            _reward("case-b", 1, "run_tests", 0.0, partial=0.0),
        ],
    )
    _write_jsonl(
        run_dir / "trajectories.jsonl",
        [
            _trajectory("case-a", 0, "search", "passed"),
            _trajectory("case-a", 1, "run_tests", "passed"),
            _trajectory("case-b", 0, "search", "no_patch"),
            _trajectory("case-b", 1, "run_tests", "no_patch"),
        ],
    )

    steps = load_step_credits(runs)
    positions = position_summary(steps)

    assert positions[0]["step_index"] == 0
    assert positions[0]["success_rate"] == 0.5
    assert positions[0]["top_action"] == "search"
    assert positions[1]["partial_progress_mean"] == 0.25
    assert positions[1]["test_signal_mean"] == 0.5


def test_all_zero_failed_trajectories_do_not_divide_by_zero(tmp_path: Path):
    runs = tmp_path / "runs"
    run_dir = runs / "zero_learning"
    run_dir.mkdir(parents=True)
    _write_jsonl(
        run_dir / "rewards.jsonl",
        [
            _reward("case-zero", 0, "search", 0.0),
            _reward("case-zero", 1, "run_tests", 0.0),
        ],
    )
    _write_jsonl(
        run_dir / "trajectories.jsonl",
        [_trajectory("case-zero", 0, "search", "failed"), _trajectory("case-zero", 1, "run_tests", "failed")],
    )

    summary = summarize_credit_assignment(runs)

    assert summary["trajectory_count"] == 1
    assert cast(dict[str, object], summary["correlations"])["reward_to_go_vs_success"] is None
    action_rows = cast(list[dict[str, object]], summary["action_type_summary"])
    assert all(row["success_rate_when_present"] == 0.0 for row in action_rows)
    negative = cast(list[dict[str, object]], summary["top_negative_action_types"])
    assert negative[0]["mean_return_delta_vs_logged"] == 0.0


def test_action_helpers_and_table_artifact_are_deterministic(tmp_path: Path):
    runs = tmp_path / "runs"
    run_dir = runs / "learning_tables"
    run_dir.mkdir(parents=True)
    _write_jsonl(
        run_dir / "rewards.jsonl",
        [_reward("case", 0, "search", 1.0), _reward("case", 1, "read_file", -0.25)],
    )
    _write_jsonl(
        run_dir / "trajectories.jsonl",
        [_trajectory("case", 0, "search", "passed"), _trajectory("case", 1, "read_file", "passed")],
    )
    steps = load_step_credits(runs)
    trajectories: dict[tuple[str, str, str], list[StepCredit]] = {}
    for step in steps:
        trajectories.setdefault(step.trajectory_key, []).append(step)

    actions = action_type_summary(steps, trajectories)
    leave_one = leave_one_action_type_out(steps, trajectories)
    summary = summarize_credit_assignment(runs)
    artifacts = write_credit_artifacts(tmp_path / "figures" / "credit_assignment.json", summary)

    assert [row["action"] for row in actions] == ["search", "read_file"]
    assert cast(dict[str, object], leave_one["read_file"])["mean_return_delta_vs_logged"] == -0.25
    table_path = Path(cast(str, artifacts["table"]))
    table_text = table_path.read_text(encoding="utf-8")
    assert "diagnostic/correlational" in table_text
    assert "| search |" in table_text


def _reward(instance_id: str, step_index: int, action: str, reward: float, *, partial: float = 0.0, visible: float = 0.0) -> dict[str, object]:
    return {
        "action": action,
        "episode_index": 0,
        "event": "learning_reward",
        "instance_id": instance_id,
        "reward_total": reward,
        "status": "ok",
        "step_index": step_index,
        "tool": action,
        "weighted_components": {
            "hidden_regression_ready": 0.0,
            "partial_progress": partial,
            "pass": 0.0,
            "test_runs": 0.0,
            "visible_test_failure": min(visible, 0.0),
            "visible_test_pass": max(visible, 0.0),
        },
    }


def _trajectory(instance_id: str, step_index: int, action: str, final_status: str) -> dict[str, object]:
    return {
        "action": action,
        "episode_index": 0,
        "event": "trajectory_step",
        "final_status": final_status,
        "instance_id": instance_id,
        "metadata": {},
        "run_id": "learning_unit",
        "status": "ok",
        "step_index": step_index,
        "tool": action,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
