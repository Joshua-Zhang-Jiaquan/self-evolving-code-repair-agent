from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import yaml

from repair_agent.agent.baseline import BaselineAgent
from repair_agent.agent.interface import AgentTask
from repair_agent.agent.models import GenerationResult
from repair_agent.logging import read_jsonl
from repair_agent.run import main
from repair_agent.tools.core import OK, ToolResult
from scripts.validate_predictions import validate_predictions_file


class MalformedAdapter:
    name: str = "malformed-local"

    def generate(self, messages: list[dict[str, object]], config: dict[str, object]) -> GenerationResult:
        _ = messages, config
        return GenerationResult(text="not a tool call", model=self.name)


class RecordingRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def execute(self, name: str, workspace: object, args: dict[str, object] | None = None) -> ToolResult:
        _ = workspace
        payload = args or {}
        self.calls.append((name, payload))
        if name == "search":
            return ToolResult(name, OK, output="math_utils.py:1: def add_numbers(left, right):")
        if name == "read_file" and payload.get("path") == "math_utils.py":
            return ToolResult(name, OK, output="1: def add_numbers(left, right):\n2:     return left - right")
        if name == "git_diff":
            return ToolResult(name, OK, output="--- a/math_utils.py\n+++ b/math_utils.py\n@@\n-return left - right\n+return left + right\n")
        return ToolResult(name, OK, output="ok")


def write_repair_fixture(root: Path) -> None:
    (root / "tests").mkdir(parents=True)
    _ = (root / "README.md").write_text("baseline fixture\n", encoding="utf-8")
    _ = (root / "math_utils.py").write_text("def add_numbers(left, right):\n    return left - right\n", encoding="utf-8")
    _ = (root / "tests" / "test_math_utils.py").write_text(
        "from math_utils import add_numbers\n\n\ndef test_add_numbers_visible():\n    assert add_numbers(2, 3) == 5\n",
        encoding="utf-8",
    )


def make_task(root: Path, *, instance_id: str = "local-case") -> AgentTask:
    return AgentTask(
        instance_id=instance_id,
        repo="local/baseline-fixture",
        problem_statement="The visible test for add_numbers fails because the helper should add two numbers.",
        checkout_root=root,
        visible_tests=("tests/test_math_utils.py",),
        visible_failures={"visible-failure": "AssertionError: add_numbers(2, 3) should equal 5"},
        max_steps=12,
        max_test_runs=1,
        test_timeout_seconds=5.0,
    )


def test_fixed_policy_repairs_local_fixture_and_emits_prediction(tmp_path: Path):
    write_repair_fixture(tmp_path)
    result = BaselineAgent().run(make_task(tmp_path), "unit-run")

    assert result.final.prediction_row()["instance_id"] == "local-case"
    assert result.final.prediction_row()["model_patch"] == result.final.model_patch
    assert result.final.status == "passed"
    assert "+    return left + right" in result.final.model_patch
    assert [step.tool for step in result.trajectory if step.tool == "edit_file"]
    assert [step.tool for step in result.trajectory if step.tool == "run_tests"]
    assert all(step.final_status == "passed" for step in result.trajectory)


def test_malformed_tool_call_records_error_and_continues(tmp_path: Path):
    write_repair_fixture(tmp_path)
    result = BaselineAgent(model=MalformedAdapter()).run(make_task(tmp_path), "malformed-run")

    parse_steps = [step for step in result.trajectory if step.action == "model_tool_parse"]
    assert len(parse_steps) == 1
    assert parse_steps[0].status == "malformed"
    assert parse_steps[0].error
    assert isinstance(result.final.model_patch, str)
    assert result.final.status in {"passed", "failed", "no_patch", "patch_unverified", "rolled_back"}


def test_baseline_uses_supplied_safe_registry_for_all_actions(tmp_path: Path):
    tmp_path.mkdir(exist_ok=True)
    registry = RecordingRegistry()
    task = make_task(tmp_path)

    result = BaselineAgent(registry=registry).run(task, "registry-run")

    called_tools = [name for name, _ in registry.calls]
    assert called_tools[:2] == ["search", "read_file"]
    assert "edit_file" in called_tools
    assert called_tools[-1] == "final_answer"
    assert result.final.model_patch.startswith("--- a/math_utils.py")


def test_cli_baseline_writes_artifacts_validates_prediction_and_resumes(tmp_path: Path):
    config_path = write_baseline_config(tmp_path, include_hidden=False)
    argv = ["--config", str(config_path), "--limit", "1", "--run-id", "baseline_unit"]

    assert main(argv) == 0
    run_dir = tmp_path / "runs" / "baseline_unit"
    assert (run_dir / "predictions.jsonl").is_file()
    assert (run_dir / "trajectories.jsonl").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "patches").is_dir()
    patch_files = sorted((run_dir / "patches").glob("*.patch"))
    assert len(patch_files) == 1
    assert "+    return left + right" in patch_files[0].read_text(encoding="utf-8")
    summary = validate_predictions_file(run_dir / "predictions.jsonl")
    assert summary.instance_ids == ("baseline-unit-0001",)
    first_rows = read_jsonl(run_dir / "trajectories.jsonl")
    assert first_rows
    assert {"instance_id", "run_id", "model_name_or_path", "agent_version", "action", "tool", "status", "output_summary", "tool_call_count", "test_run_count", "final_status"}.issubset(first_rows[0])

    assert main(argv) == 0
    second_rows = read_jsonl(run_dir / "trajectories.jsonl")
    metrics = cast(dict[str, object], json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")))
    assert len(second_rows) == len(first_rows)
    assert metrics["skipped"] == 1
    assert metrics["newly_completed"] == 0


def test_cli_force_resets_baseline_run_and_does_not_leak_gold_fields(tmp_path: Path):
    marker = "SECRET_GOLD_PATCH_MARKER"
    config_path = write_baseline_config(tmp_path, include_hidden=True, hidden_marker=marker)
    argv = ["--config", str(config_path), "--limit", "1", "--run-id", "force_unit"]

    assert main(argv) == 0
    assert main([*argv, "--force"]) == 0
    run_dir = tmp_path / "runs" / "force_unit"
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    trajectories = read_jsonl(run_dir / "trajectories.jsonl")
    patch_text = "\n".join(path.read_text(encoding="utf-8") for path in (run_dir / "patches").glob("*.patch"))
    assert len(predictions) == 1
    assert trajectories
    serialized_agent_outputs = json.dumps({"predictions": predictions, "trajectories": trajectories, "patch_text": patch_text}, sort_keys=True)
    assert marker not in serialized_agent_outputs


def write_baseline_config(tmp_path: Path, *, include_hidden: bool, hidden_marker: str = "") -> Path:
    instance: dict[str, object] = {
        "fixture": {
            "files": {
                "README.md": "baseline fixture\n",
                "math_utils.py": "def add_numbers(left, right):\n    return left - right\n",
                "tests/test_math_utils.py": "from math_utils import add_numbers\n\n\ndef test_add_numbers_visible():\n    assert add_numbers(2, 3) == 5\n",
            }
        },
        "instance_id": "baseline-unit-0001",
        "problem_statement": "The visible test for add_numbers fails because the helper should add two numbers.",
        "repo": "local/baseline-fixture",
        "visible_failures": {"visible-failure": "AssertionError: add_numbers(2, 3) should equal 5"},
        "visible_tests": ["tests/test_math_utils.py"],
    }
    if include_hidden:
        instance["patch"] = hidden_marker
        instance["test_patch"] = hidden_marker
    config = {
        "agent": {
            "instances": [instance],
            "max_output_chars": 4000,
            "max_steps": 12,
            "max_test_runs": 1,
            "model_name_or_path": "rule_based_local",
            "test_timeout_seconds": 5.0,
            "type": "baseline",
        },
        "dry_run": {"instances": [{"instance_id": "dry-unit", "repo": "local/dry"}]},
        "run": {"name": "baseline-test", "output_dir": str(tmp_path / "runs")},
    }
    path = tmp_path / "baseline.yaml"
    _ = path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return path
