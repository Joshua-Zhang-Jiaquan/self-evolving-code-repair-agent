from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from time import monotonic
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REWARD_CONFIG = PROJECT_ROOT / "configs" / "rewards.yaml"

SENSITIVE_TASK_KEYS = frozenset({"patch", "test_patch"})
REQUIRED_ACTIONS = (
    "search",
    "read_file",
    "inspect_test",
    "edit_file",
    "run_tests",
    "rollback",
    "final_answer",
)
REQUIRED_REWARD_WEIGHTS = (
    "pass",
    "visible_test_pass",
    "hidden_regression_ready",
    "partial_progress",
    "relevant_file",
    "tool_calls",
    "test_runs",
    "unsafe_edit",
    "test_deletion",
    "timeout",
)


class ActionType(str, Enum):
    SEARCH = "search"
    READ_FILE = "read_file"
    INSPECT_TEST = "inspect_test"
    EDIT_FILE = "edit_file"
    RUN_TESTS = "run_tests"
    ROLLBACK = "rollback"
    GIT_DIFF = "git_diff"
    FINAL_ANSWER = "final_answer"
    PATCH_SUBMISSION = "submit_patch"


class TerminationReason(str, Enum):
    NOT_TERMINATED = "not_terminated"
    FINAL_ANSWER = "final_answer"
    MAX_STEPS = "max_steps"
    MAX_TEST_RUNS = "max_test_runs"
    TIMEOUT = "timeout"
    UNRECOVERABLE_TOOL_FAILURE = "unrecoverable_tool_failure"
    PATCH_SUBMISSION = "patch_submission"


ACTION_SCHEMAS: dict[str, dict[str, Any]] = {
    ActionType.SEARCH.value: {
        "description": "Search repository text or symbols for likely repair locations.",
        "required": ["query"],
        "properties": {"query": "string search expression"},
    },
    ActionType.READ_FILE.value: {
        "description": "Read a source file or directory listing from the repository.",
        "required": ["path"],
        "properties": {"path": "repository-relative path"},
    },
    ActionType.INSPECT_TEST.value: {
        "description": "Inspect visible tests or failure traces supplied by the task environment.",
        "required": ["target"],
        "properties": {"target": "test path, test name, or failure identifier"},
    },
    ActionType.EDIT_FILE.value: {
        "description": "Apply a bounded source edit to a repository file.",
        "required": ["path", "replacement"],
        "properties": {
            "path": "repository-relative path",
            "replacement": "new source text or patch hunk for the selected span",
            "start_line": "optional one-indexed start line",
            "end_line": "optional one-indexed inclusive end line",
        },
    },
    ActionType.RUN_TESTS.value: {
        "description": "Run visible tests, smoke tests, or a targeted regression command.",
        "required": ["target"],
        "properties": {"target": "test command, file, marker, or suite name"},
    },
    ActionType.ROLLBACK.value: {
        "description": "Revert the last unsafe or unhelpful edit checkpoint.",
        "required": [],
        "properties": {"reason": "optional reason recorded in trajectory metadata"},
    },
    ActionType.GIT_DIFF.value: {
        "description": "Inspect the current working diff before testing or submission.",
        "required": [],
        "properties": {},
    },
    ActionType.FINAL_ANSWER.value: {
        "description": "End the dialogue with a concise repair summary.",
        "required": ["answer"],
        "properties": {"answer": "final human-readable response"},
    },
    ActionType.PATCH_SUBMISSION.value: {
        "description": "Submit the current patch to the evaluator as the candidate repair.",
        "required": [],
        "properties": {"summary": "optional patch summary"},
    },
}


@dataclass(frozen=True)
class Observation:
    task: Mapping[str, Any]
    files: Mapping[str, str] = field(default_factory=dict)
    visible_test_results: Mapping[str, Any] = field(default_factory=dict)
    tool_feedback: Mapping[str, Any] = field(default_factory=dict)
    step_count: int = 0
    test_run_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": dict(self.task),
            "files": dict(self.files),
            "visible_test_results": dict(self.visible_test_results),
            "tool_feedback": dict(self.tool_feedback),
            "step_count": self.step_count,
            "test_run_count": self.test_run_count,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass(frozen=True)
class RepairState:
    instance_id: str
    repo: str
    problem_statement: str
    base_commit: str | None = None
    relevant_files: tuple[str, ...] = ()
    touched_files: tuple[str, ...] = ()
    visible_tests_passed: int = 0
    visible_tests_failed: int = 0
    current_patch: str = ""
    step_count: int = 0
    test_run_count: int = 0
    elapsed_seconds: float = 0.0
    started_at: float = field(default_factory=monotonic)
    last_tool_success: bool = True
    unrecoverable_tool_failure: bool = False
    submitted_patch: bool = False
    final_answer: bool = False


@dataclass(frozen=True)
class Action:
    kind: ActionType
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        schema = ACTION_SCHEMAS[self.kind.value]
        missing = [name for name in schema["required"] if name not in self.arguments]
        if missing:
            raise ValueError(f"{self.kind.value} missing required arguments: {', '.join(missing)}")


@dataclass(frozen=True)
class ToolOutcome:
    success: bool = True
    elapsed_seconds: float = 0.0
    patch: str | None = None
    relevant_file: str | None = None
    touched_file: str | None = None
    visible_tests_passed: int | None = None
    visible_tests_failed: int | None = None
    unrecoverable_failure: bool = False
    submitted_patch: bool = False


@dataclass(frozen=True)
class RewardSignal:
    pass_result: float = 0.0
    visible_test_pass: float = 0.0
    visible_test_failure: float = 0.0
    hidden_regression_ready: float = 0.0
    partial_progress: float = 0.0
    relevant_file: float = 0.0
    tool_calls: float = 0.0
    test_runs: float = 0.0
    unsafe_edit: float = 0.0
    test_deletion: float = 0.0
    timeout: float = 0.0

    def to_components(self) -> dict[str, float]:
        return {
            "pass": self.pass_result,
            "visible_test_pass": self.visible_test_pass,
            "visible_test_failure": self.visible_test_failure,
            "hidden_regression_ready": self.hidden_regression_ready,
            "partial_progress": self.partial_progress,
            "relevant_file": self.relevant_file,
            "tool_calls": self.tool_calls,
            "test_runs": self.test_runs,
            "unsafe_edit": self.unsafe_edit,
            "test_deletion": self.test_deletion,
            "timeout": self.timeout,
        }


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    components: Mapping[str, float]
    weighted_components: Mapping[str, float]


@dataclass(frozen=True)
class WeightedReward:
    weights: Mapping[str, float]

    def score(self, signal: RewardSignal | Mapping[str, float]) -> RewardBreakdown:
        components = signal.to_components() if isinstance(signal, RewardSignal) else dict(signal)
        weighted = {
            name: float(self.weights.get(name, 0.0)) * float(value)
            for name, value in components.items()
        }
        return RewardBreakdown(
            total=sum(weighted.values()),
            components=components,
            weighted_components=weighted,
        )


@dataclass(frozen=True)
class TerminationSpec:
    max_steps: int = 64
    max_test_runs: int = 8
    timeout_seconds: float = 1800.0


@dataclass(frozen=True)
class TerminationResult:
    done: bool
    reasons: tuple[TerminationReason, ...]


def _sensitive_values(task: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key in SENSITIVE_TASK_KEYS:
        value = task.get(key)
        if isinstance(value, str) and value:
            values.append(value)
    return tuple(values)


def _sanitize(value: Any, sensitive_values: Sequence[str]) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize(item, sensitive_values)
            for key, item in value.items()
            if str(key) not in SENSITIVE_TASK_KEYS
        }
    if isinstance(value, list):
        return [_sanitize(item, sensitive_values) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize(item, sensitive_values) for item in value)
    if isinstance(value, str):
        sanitized = value
        for secret in sensitive_values:
            sanitized = sanitized.replace(secret, "[redacted hidden repair target]")
        return sanitized
    return value


def build_observation(
    task: Mapping[str, Any],
    *,
    files: Mapping[str, str] | None = None,
    visible_test_results: Mapping[str, Any] | None = None,
    tool_feedback: Mapping[str, Any] | None = None,
    step_count: int = 0,
    test_run_count: int = 0,
) -> Observation:
    sensitive_values = _sensitive_values(task)
    return Observation(
        task=_sanitize(task, sensitive_values),
        files=_sanitize(files or {}, sensitive_values),
        visible_test_results=_sanitize(visible_test_results or {}, sensitive_values),
        tool_feedback=_sanitize(tool_feedback or {}, sensitive_values),
        step_count=step_count,
        test_run_count=test_run_count,
    )


def initial_state_from_task(task: Mapping[str, Any]) -> RepairState:
    observation = build_observation(task)
    safe_task = observation.task
    return RepairState(
        instance_id=str(safe_task.get("instance_id", safe_task.get("id", "unknown"))),
        repo=str(safe_task.get("repo", "unknown")),
        problem_statement=str(safe_task.get("problem_statement", "")),
        base_commit=(str(safe_task["base_commit"]) if safe_task.get("base_commit") else None),
    )


def transition(state: RepairState, action: Action, outcome: ToolOutcome | None = None) -> RepairState:
    action.validate()
    result = outcome or ToolOutcome()
    relevant_files = state.relevant_files
    if result.relevant_file and result.relevant_file not in relevant_files:
        relevant_files = (*relevant_files, result.relevant_file)
    touched_files = state.touched_files
    touched_file = result.touched_file or action.arguments.get("path")
    if action.kind == ActionType.EDIT_FILE and touched_file and touched_file not in touched_files:
        touched_files = (*touched_files, str(touched_file))
    return replace(
        state,
        relevant_files=relevant_files,
        touched_files=touched_files,
        visible_tests_passed=(
            state.visible_tests_passed
            if result.visible_tests_passed is None
            else result.visible_tests_passed
        ),
        visible_tests_failed=(
            state.visible_tests_failed
            if result.visible_tests_failed is None
            else result.visible_tests_failed
        ),
        current_patch=state.current_patch if result.patch is None else result.patch,
        step_count=state.step_count + 1,
        test_run_count=state.test_run_count + (1 if action.kind == ActionType.RUN_TESTS else 0),
        elapsed_seconds=state.elapsed_seconds + result.elapsed_seconds,
        last_tool_success=result.success,
        unrecoverable_tool_failure=state.unrecoverable_tool_failure or result.unrecoverable_failure,
        submitted_patch=(
            state.submitted_patch
            or result.submitted_patch
            or action.kind == ActionType.PATCH_SUBMISSION
        ),
        final_answer=state.final_answer or action.kind == ActionType.FINAL_ANSWER,
    )


def check_termination(state: RepairState, spec: TerminationSpec) -> TerminationResult:
    reasons: list[TerminationReason] = []
    if state.final_answer:
        reasons.append(TerminationReason.FINAL_ANSWER)
    if state.step_count >= spec.max_steps:
        reasons.append(TerminationReason.MAX_STEPS)
    if state.test_run_count >= spec.max_test_runs:
        reasons.append(TerminationReason.MAX_TEST_RUNS)
    if state.elapsed_seconds >= spec.timeout_seconds:
        reasons.append(TerminationReason.TIMEOUT)
    if state.unrecoverable_tool_failure:
        reasons.append(TerminationReason.UNRECOVERABLE_TOOL_FAILURE)
    if state.submitted_patch:
        reasons.append(TerminationReason.PATCH_SUBMISSION)
    if not reasons:
        reasons.append(TerminationReason.NOT_TERMINATED)
    return TerminationResult(done=reasons != [TerminationReason.NOT_TERMINATED], reasons=tuple(reasons))


def load_reward_weights(path: str | Path = DEFAULT_REWARD_CONFIG) -> dict[str, float]:
    config_path = Path(path)
    data: object = yaml.safe_load(config_path.read_text())
    if not isinstance(data, Mapping):
        raise ValueError(f"reward config must contain a 'weights' mapping: {config_path}")
    weights_data = data.get("weights")
    if not isinstance(weights_data, Mapping):
        raise ValueError(f"reward config must contain a 'weights' mapping: {config_path}")
    weights: dict[str, float] = {}
    for name, raw_value in weights_data.items():
        if not isinstance(raw_value, int | float):
            raise ValueError(f"reward weight {name!r} must be numeric, got {type(raw_value).__name__}")
        weights[str(name)] = float(raw_value)
    missing = [name for name in REQUIRED_REWARD_WEIGHTS if name not in weights]
    if missing:
        raise ValueError(f"reward config missing required weights: {', '.join(missing)}")
    return weights


def default_reward() -> WeightedReward:
    return WeightedReward(load_reward_weights())


def spec_sections(weights: Mapping[str, float] | None = None) -> dict[str, str]:
    reward_weights = weights or load_reward_weights()
    action_lines = [
        f"- {name}: requires {schema['required'] or 'no required args'}; {schema['description']}"
        for name, schema in ACTION_SCHEMAS.items()
    ]
    reward_lines = [f"- {name}: {value:g}" for name, value in sorted(reward_weights.items())]
    return {
        "State": "\n".join(
            [
                "RepairState tracks instance_id, repo, problem_statement, base_commit,",
                "relevant_files, touched_files, current_patch, visible test counts,",
                "step_count, test_run_count, elapsed_seconds, tool failure flags,",
                "submitted_patch, and final_answer.",
            ]
        ),
        "Action": "\n".join(action_lines),
        "Observation": "\n".join(
            [
                "build_observation(task, files, visible_test_results, tool_feedback)",
                "recursively removes patch and test_patch fields and redacts their values.",
            ]
        ),
        "Reward": "Weighted sum over named components from configs/rewards.yaml:\n"
        + "\n".join(reward_lines),
        "Transition": "transition(state, action, outcome) validates action schemas, increments",
        "Termination": "final_answer, max_steps, max_test_runs, timeout, unrecoverable_tool_failure,",
    } | {
        "TransitionDetail": "test counters, records relevant/touched files, stores current patch metadata, and sets submission/failure flags.",
        "TerminationDetail": "and patch_submission are checked by check_termination(state, spec).",
    }


def render_spec(weights: Mapping[str, float] | None = None) -> str:
    sections = spec_sections(weights)
    return "\n\n".join(f"## {title}\n{body}" for title, body in sections.items())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repair-agent POMDP and reward specification")
    _ = parser.add_argument("--print-spec", action="store_true", help="print state/action/reward sections")
    _ = parser.add_argument(
        "--reward-config",
        type=Path,
        default=DEFAULT_REWARD_CONFIG,
        help="path to rewards.yaml",
    )
    args = parser.parse_args(argv)
    if args.print_spec:
        print(render_spec(load_reward_weights(args.reward_config)))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
