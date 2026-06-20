from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from repair_agent.training.pomdp import (
    ACTION_SCHEMAS,
    REQUIRED_ACTIONS,
    REQUIRED_REWARD_WEIGHTS,
    Action,
    ActionType,
    RepairState,
    RewardSignal,
    TerminationReason,
    TerminationSpec,
    ToolOutcome,
    WeightedReward,
    build_observation,
    check_termination,
    load_reward_weights,
    transition,
)


def test_reward_yaml_has_required_numeric_weights(project_root: Path):
    cfg = yaml.safe_load((project_root / "configs" / "rewards.yaml").read_text())
    weights = cfg["weights"]

    for name in REQUIRED_REWARD_WEIGHTS:
        assert name in weights
        assert isinstance(weights[name], int | float)

    assert weights["pass"] > 0
    assert weights["partial_progress"] > 0
    assert weights["relevant_file"] > 0
    assert weights["tool_calls"] < 0
    assert weights["test_runs"] < 0
    assert weights["unsafe_edit"] < 0
    assert weights["test_deletion"] < 0
    assert weights["timeout"] < 0


def test_weighted_reward_uses_named_components(project_root: Path):
    weights = load_reward_weights(project_root / "configs" / "rewards.yaml")
    signal = RewardSignal(
        pass_result=1.0,
        visible_test_pass=2.0,
        hidden_regression_ready=1.0,
        partial_progress=3.0,
        relevant_file=1.0,
        tool_calls=4.0,
        test_runs=2.0,
        unsafe_edit=1.0,
        test_deletion=0.0,
        timeout=0.0,
    )

    breakdown = WeightedReward(weights).score(signal)
    expected = (
        weights["pass"]
        + 2.0 * weights["visible_test_pass"]
        + weights["hidden_regression_ready"]
        + 3.0 * weights["partial_progress"]
        + weights["relevant_file"]
        + 4.0 * weights["tool_calls"]
        + 2.0 * weights["test_runs"]
        + weights["unsafe_edit"]
    )

    assert breakdown.total == pytest.approx(expected)
    assert breakdown.weighted_components["tool_calls"] == pytest.approx(-0.2)
    assert breakdown.weighted_components["unsafe_edit"] == pytest.approx(-20.0)
    assert breakdown.components["pass"] == 1.0


def test_observation_redacts_gold_patch_and_hidden_test_patch():
    task = {
        "instance_id": "demo__repo-1",
        "repo": "demo/repo",
        "base_commit": "abc123",
        "problem_statement": "Fix the parser crash for empty input.",
        "patch": "GOLD_PATCH_SHOULD_NOT_LEAK",
        "test_patch": "HIDDEN_TEST_PATCH_SHOULD_NOT_LEAK",
        "metadata": {"patch": "NESTED_PATCH_SHOULD_NOT_LEAK"},
    }
    observation = build_observation(
        task,
        files={"parser.py": "def parse(value): return value"},
        visible_test_results={"pytest tests/test_parser.py": "failed"},
        tool_feedback={"stderr": "GOLD_PATCH_SHOULD_NOT_LEAK"},
        step_count=3,
        test_run_count=1,
    )

    serialized = observation.to_json()
    parsed = json.loads(serialized)

    assert parsed["task"]["problem_statement"] == "Fix the parser crash for empty input."
    assert parsed["task"]["repo"] == "demo/repo"
    assert "patch" not in parsed["task"]
    assert "test_patch" not in parsed["task"]
    assert parsed["task"]["metadata"] == {}
    assert "GOLD_PATCH_SHOULD_NOT_LEAK" not in serialized
    assert "HIDDEN_TEST_PATCH_SHOULD_NOT_LEAK" not in serialized
    assert "NESTED_PATCH_SHOULD_NOT_LEAK" not in serialized
    assert parsed["step_count"] == 3
    assert parsed["test_run_count"] == 1


def test_required_action_schemas_validate_core_tools():
    assert set(REQUIRED_ACTIONS).issubset(ACTION_SCHEMAS)

    Action(ActionType.SEARCH, {"query": "Parser"}).validate()
    Action(ActionType.READ_FILE, {"path": "src/parser.py"}).validate()
    Action(ActionType.INSPECT_TEST, {"target": "tests/test_parser.py"}).validate()
    Action(ActionType.EDIT_FILE, {"path": "src/parser.py", "replacement": "return []"}).validate()
    Action(ActionType.RUN_TESTS, {"target": "pytest tests/test_parser.py"}).validate()
    Action(ActionType.ROLLBACK, {}).validate()
    Action(ActionType.GIT_DIFF, {}).validate()
    Action(ActionType.FINAL_ANSWER, {"answer": "Implemented parser guard."}).validate()

    with pytest.raises(ValueError, match="missing required arguments"):
        Action(ActionType.EDIT_FILE, {"path": "src/parser.py"}).validate()


def test_transition_updates_test_counts_files_and_termination():
    state = RepairState(
        instance_id="demo__repo-1",
        repo="demo/repo",
        problem_statement="Parser fails on empty input.",
    )
    edited = transition(
        state,
        Action(ActionType.EDIT_FILE, {"path": "src/parser.py", "replacement": "return []"}),
        ToolOutcome(patch="diff --git a/src/parser.py b/src/parser.py", relevant_file="src/parser.py"),
    )
    tested = transition(
        edited,
        Action(ActionType.RUN_TESTS, {"target": "pytest tests/test_parser.py"}),
        ToolOutcome(visible_tests_passed=5, visible_tests_failed=0, elapsed_seconds=2.5),
    )

    assert tested.step_count == 2
    assert tested.test_run_count == 1
    assert tested.touched_files == ("src/parser.py",)
    assert tested.relevant_files == ("src/parser.py",)
    assert tested.visible_tests_passed == 5
    assert tested.visible_tests_failed == 0
    assert tested.elapsed_seconds == pytest.approx(2.5)
    assert not check_termination(tested, TerminationSpec(max_steps=4, max_test_runs=3)).done

    submitted = transition(tested, Action(ActionType.PATCH_SUBMISSION, {}))
    result = check_termination(submitted, TerminationSpec(max_steps=4, max_test_runs=3))
    assert result.done
    assert result.reasons == (TerminationReason.PATCH_SUBMISSION,)


def test_termination_conditions_cover_limits_failures_and_final_answer():
    spec = TerminationSpec(max_steps=3, max_test_runs=2, timeout_seconds=10.0)
    base = RepairState(instance_id="i", repo="r", problem_statement="p")

    assert check_termination(base, spec).reasons == (TerminationReason.NOT_TERMINATED,)
    assert check_termination(replace(base, final_answer=True), spec).reasons == (
        TerminationReason.FINAL_ANSWER,
    )
    limit_state = replace(base, step_count=3, test_run_count=2, elapsed_seconds=10.0)
    limit_result = check_termination(limit_state, spec)
    assert limit_result.done
    assert set(limit_result.reasons) == {
        TerminationReason.MAX_STEPS,
        TerminationReason.MAX_TEST_RUNS,
        TerminationReason.TIMEOUT,
    }
    failure_state = replace(base, unrecoverable_tool_failure=True)
    assert check_termination(failure_state, spec).reasons == (
        TerminationReason.UNRECOVERABLE_TOOL_FAILURE,
    )


def test_print_spec_cli_includes_required_sections(project_root: Path):
    result = subprocess.run(
        [sys.executable, "-m", "repair_agent.training.pomdp", "--print-spec"],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    for section in ["## State", "## Action", "## Reward", "## Termination"]:
        assert section in stdout
    for action_name in REQUIRED_ACTIONS:
        assert action_name in stdout
    assert "unsafe_edit" in stdout
    assert "timeout" in stdout
