from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import yaml

from repair_agent.agent.feedback import FeedbackAgent
from repair_agent.agent.interface import AgentTask
from repair_agent.logging import read_jsonl
from repair_agent.run import main
from repair_agent.tools.core import ERROR, OK, ToolResult
from scripts.compare_configs import compare_budget_fields, compare_task_budget_shape, main as compare_main
from scripts.validate_predictions import validate_predictions_file


class FeedbackRecordingRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def execute(self, name: str, workspace: object, args: dict[str, object] | None = None) -> ToolResult:
        _ = workspace
        payload = args or {}
        self.calls.append((name, payload))
        if name == "search":
            query = str(payload.get("query", ""))
            if "add_numbers" in query:
                return ToolResult(name, OK, output="math_utils.py:1: def add_numbers(left, right):")
            return ToolResult(name, OK, output="README.md:1: local fixture")
        if name == "read_file" and payload.get("path") == "math_utils.py":
            return ToolResult(name, OK, output="1: def add_numbers(left, right):\n2:     return left - right")
        if name == "run_tests":
            return ToolResult(name, ERROR, output="E assert add_numbers(2, 3) == 5\nFAILED tests/test_math_utils.py::test_add_numbers_visible")
        if name == "git_diff":
            return ToolResult(name, OK, output="--- a/math_utils.py\n+++ b/math_utils.py\n@@\n-    return left - right\n+    return left + right\n")
        return ToolResult(name, OK, output="ok")


def write_repair_fixture(root: Path) -> None:
    (root / "tests").mkdir(parents=True)
    _ = (root / "README.md").write_text("feedback fixture\n", encoding="utf-8")
    _ = (root / "math_utils.py").write_text("def add_numbers(left, right):\n    return left - right\n", encoding="utf-8")
    _ = (root / "tests" / "test_math_utils.py").write_text(
        "from math_utils import add_numbers\n\n\ndef test_add_numbers_visible():\n    assert add_numbers(2, 3) == 5\n",
        encoding="utf-8",
    )


def make_task(root: Path, *, max_test_runs: int = 1) -> AgentTask:
    return AgentTask(
        instance_id="feedback-local-case",
        repo="local/feedback-fixture",
        problem_statement="The visible test for add_numbers fails because the helper should add two numbers.",
        checkout_root=root,
        visible_tests=("tests/test_math_utils.py",),
        visible_failures={"visible-failure": "AssertionError: add_numbers(2, 3) should equal 5"},
        max_steps=12,
        max_test_runs=max_test_runs,
        test_timeout_seconds=5.0,
    )


def test_failed_test_output_changes_next_action_context(tmp_path: Path):
    write_repair_fixture(tmp_path)
    registry = FeedbackRecordingRegistry()

    result = FeedbackAgent(registry=registry).run(make_task(tmp_path), "feedback-unit")

    search_calls = [args for name, args in registry.calls if name == "search"]
    assert len(search_calls) >= 2
    assert search_calls[1]["query"] == "add_numbers"
    reflected_steps = [step for step in result.trajectory if step.metadata.get("reflection_used") is True]
    assert reflected_steps
    assert any("retry_search:add_numbers" == step.metadata.get("action_context") for step in reflected_steps)
    assert any(step.tool == "edit_file" for step in reflected_steps)
    assert result.metrics["learning_updates"] == 0


def test_feedback_metadata_recorded_after_real_visible_test_failure(tmp_path: Path):
    write_repair_fixture(tmp_path)

    result = FeedbackAgent().run(make_task(tmp_path), "feedback-real")

    failure_steps = [step for step in result.trajectory if step.tool == "run_tests" and step.status == "error"]
    assert len(failure_steps) == 1
    metadata = dict(failure_steps[0].metadata)
    assert metadata["previous_test_status"] == "error"
    assert "feedback_summary" in metadata
    assert metadata["reflection_used"] is False
    used_later = [step for step in result.trajectory if step.step_index > failure_steps[0].step_index and step.metadata.get("reflection_used") is True]
    assert used_later
    assert result.final.metadata["feedback_summary"]


def test_cli_feedback_writes_artifacts_and_valid_prediction(tmp_path: Path):
    config_path = write_feedback_config(tmp_path, include_hidden=False)
    argv = ["--config", str(config_path), "--limit", "1", "--run-id", "feedback_unit", "--force"]

    assert main(argv) == 0
    run_dir = tmp_path / "runs" / "feedback_unit"
    assert (run_dir / "predictions.jsonl").is_file()
    assert (run_dir / "trajectories.jsonl").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "patches").is_dir()
    summary = validate_predictions_file(run_dir / "predictions.jsonl")
    assert summary.instance_ids == ("feedback-unit-0001",)
    rows = read_jsonl(run_dir / "trajectories.jsonl")
    assert any(cast(dict[str, object], row.get("metadata", {})).get("feedback_summary") for row in rows)
    patch_files = sorted((run_dir / "patches").glob("*.patch"))
    assert len(patch_files) == 1


def test_cli_feedback_force_does_not_leak_hidden_patch_fields(tmp_path: Path):
    marker = "SECRET_FEEDBACK_GOLD_MARKER"
    config_path = write_feedback_config(tmp_path, include_hidden=True, hidden_marker=marker)
    assert main(["--config", str(config_path), "--limit", "1", "--run-id", "feedback_hidden", "--force"]) == 0
    run_dir = tmp_path / "runs" / "feedback_hidden"
    serialized = json.dumps(
        {
            "predictions": read_jsonl(run_dir / "predictions.jsonl"),
            "trajectories": read_jsonl(run_dir / "trajectories.jsonl"),
            "patches": [path.read_text(encoding="utf-8") for path in (run_dir / "patches").glob("*.patch")],
        },
        sort_keys=True,
    )
    assert marker not in serialized


def test_compare_configs_budget_fairness_passes_for_feedback(tmp_path: Path):
    baseline = write_named_config(tmp_path, name="baseline", agent_type="baseline")
    feedback = write_named_config(tmp_path, name="feedback", agent_type="feedback")
    left = yaml.safe_load(baseline.read_text(encoding="utf-8"))
    right = yaml.safe_load(feedback.read_text(encoding="utf-8"))

    assert compare_budget_fields(left, right) == []
    assert compare_task_budget_shape(left, right) == []
    assert compare_main([str(baseline), str(feedback), "--check-budget-equal"]) == 0


def test_compare_configs_budget_fairness_fails_on_increased_budget(tmp_path: Path):
    baseline = write_named_config(tmp_path, name="baseline", agent_type="baseline")
    feedback = write_named_config(tmp_path, name="feedback", agent_type="feedback", max_steps=13)

    assert compare_main([str(baseline), str(feedback), "--check-budget-equal"]) == 1


def test_feedback_agent_keeps_policy_state_non_learning(tmp_path: Path):
    agent = FeedbackAgent()
    before = dict(agent.__dict__)

    write_repair_fixture(tmp_path / "case_a")
    write_repair_fixture(tmp_path / "case_b")
    result_a = agent.run(make_task(tmp_path / "case_a"), "run-a")
    result_b = agent.run(make_task(tmp_path / "case_b"), "run-b")

    assert dict(agent.__dict__) == before
    assert result_a.metrics["learning_updates"] == 0
    assert result_b.metrics["learning_updates"] == 0
    assert result_a.final.model_patch == result_b.final.model_patch


def write_feedback_config(tmp_path: Path, *, include_hidden: bool, hidden_marker: str = "") -> Path:
    config = feedback_config(output_dir=str(tmp_path / "runs"), run_name="feedback-test", instance_id="feedback-unit-0001")
    agent = cast(dict[str, object], config["agent"])
    instances = cast(list[dict[str, object]], agent["instances"])
    instance = instances[0]
    if include_hidden:
        instance["patch"] = hidden_marker
        instance["test_patch"] = hidden_marker
    path = tmp_path / "feedback.yaml"
    _ = path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return path


def write_named_config(tmp_path: Path, *, name: str, agent_type: str, max_steps: int = 12) -> Path:
    config = feedback_config(output_dir=str(tmp_path / "runs"), run_name=name, instance_id=f"{name}-unit-0001")
    agent = cast(dict[str, object], config["agent"])
    agent["type"] = agent_type
    agent["max_steps"] = max_steps
    path = tmp_path / f"{name}.yaml"
    _ = path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return path


def feedback_config(*, output_dir: str, run_name: str, instance_id: str) -> dict[str, object]:
    return {
        "agent": {
            "instances": [
                {
                    "fixture": {
                        "files": {
                            "README.md": "feedback fixture\n",
                            "math_utils.py": "def add_numbers(left, right):\n    return left - right\n",
                            "tests/test_math_utils.py": "from math_utils import add_numbers\n\n\ndef test_add_numbers_visible():\n    assert add_numbers(2, 3) == 5\n",
                        }
                    },
                    "instance_id": instance_id,
                    "problem_statement": "The visible test for add_numbers fails because the helper should add two numbers.",
                    "repo": "local/feedback-fixture",
                    "visible_failures": {"visible-failure": "AssertionError: add_numbers(2, 3) should equal 5"},
                    "visible_tests": ["tests/test_math_utils.py"],
                }
            ],
            "max_output_chars": 4000,
            "max_steps": 12,
            "max_test_runs": 1,
            "model_name_or_path": "rule_based_local",
            "test_timeout_seconds": 5.0,
            "type": "feedback",
        },
        "dry_run": {"instances": [{"instance_id": "dry-unit", "repo": "local/dry"}]},
        "run": {"name": run_name, "output_dir": output_dir},
    }
