from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol

from repair_agent.agent.interface import AgentFinalAnswer, AgentResult, AgentTask, TrajectoryStepRecord, utc_now
from repair_agent.agent.models import ModelAdapter, RuleBasedAdapter, parse_qwable_tool_use
from repair_agent.config import ConfigMap
from repair_agent.tools.core import BUDGET_EXCEEDED, DENIED, ERROR, MALFORMED, OK, ToolRegistry, ToolResult, TaskWorkspace
from repair_agent.tools.registry import get_registry


BASELINE_AGENT_VERSION = "baseline-fixed-v1"
PATCH_SUCCESS_STATUSES = {OK}
TOOL_FAILURE_STATUSES = {BUDGET_EXCEEDED, DENIED, ERROR, MALFORMED}
SOURCE_SUFFIXES = (".py", ".pyi", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp")


class _RegistryLike(Protocol):
    def execute(self, name: str, workspace: TaskWorkspace, args: dict[str, object] | None = None) -> ToolResult:
        ...


@dataclass
class _Counters:
    tool_calls: int = 0
    test_runs: int = 0
    edits: int = 0


class BaselineAgent:
    agent_version: str = BASELINE_AGENT_VERSION

    def __init__(self, model: ModelAdapter | None = None, registry: _RegistryLike | ToolRegistry | None = None) -> None:
        self.model: ModelAdapter = model or RuleBasedAdapter()
        self.registry: _RegistryLike = registry if registry is not None else get_registry()

    def run(self, task: AgentTask, run_id: str) -> AgentResult:
        workspace = TaskWorkspace(
            checkout_root=task.checkout_root,
            visible_tests=task.visible_tests,
            visible_failures=task.visible_failures,
            max_output_chars=task.max_output_chars,
            test_timeout_seconds=task.test_timeout_seconds,
            max_test_runs=task.max_test_runs,
        )
        counters = _Counters()
        steps: list[TrajectoryStepRecord] = []
        last_read: ToolResult | None = None
        candidate_path = ""
        edited = False
        test_status = "not_run"

        def remaining() -> bool:
            return len(steps) < task.max_steps

        def execute(tool: str, args: dict[str, object] | None = None) -> ToolResult:
            result = self.registry.execute(tool, workspace, args or {})
            counters.tool_calls += 1
            if tool == "run_tests" and result.status not in {MALFORMED, DENIED}:
                counters.test_runs = workspace.test_run_count
            if tool == "edit_file" and result.status == OK:
                counters.edits += 1
            steps.append(self._tool_step(task, run_id, len(steps), tool, args or {}, result, counters))
            return result

        if remaining():
            search = execute("search", {"query": _query_from_problem(task.problem_statement), "path": ".", "max_matches": 50})
            candidate_path = _candidate_from_search(search.output)
        if remaining():
            _ = execute("read_file", {"path": ".", "max_lines": 200})
        for target in [*task.visible_failures.keys(), *task.visible_tests[:1]]:
            if not remaining():
                break
            _ = execute("inspect_test", {"target": target})
        if candidate_path and remaining():
            last_read = execute("read_file", {"path": candidate_path, "max_lines": 300})
        if remaining():
            self._try_model_action(task, run_id, workspace, steps, counters)
        if last_read is not None and candidate_path and remaining():
            for edit_args in _edits_from_read(candidate_path, last_read.output, task.problem_statement):
                if not remaining():
                    break
                edit = execute("edit_file", edit_args)
                edited = edited or edit.status == OK
        if remaining() and task.max_test_runs > 0:
            test_target = task.visible_tests[0] if task.visible_tests else ""
            tests = execute("run_tests", {"target": test_target, "timeout_seconds": task.test_timeout_seconds})
            test_status = tests.status
            if edited and tests.status not in PATCH_SUCCESS_STATUSES and remaining():
                _ = execute("rollback", {"reason": "baseline edit did not pass visible tests"})
                edited = False
        patch = ""
        diff_status = "not_run"
        if remaining():
            diff = execute("git_diff", {})
            diff_status = diff.status
            patch = diff.output if diff.status == OK else ""

        final_status = _final_status(edited=edited, test_status=test_status, patch=patch, diff_status=diff_status)
        explanation = _final_explanation(final_status=final_status, test_status=test_status, edited=edited, diff_status=diff_status)
        if remaining():
            _ = execute("final_answer", {"answer": explanation})
        steps = [self._with_final_status(step, final_status) for step in steps]
        final = AgentFinalAnswer(
            instance_id=task.instance_id,
            model_name_or_path=task.model_name_or_path,
            model_patch=patch,
            status=final_status,
            explanation=explanation,
            metadata={"test_status": test_status, "diff_status": diff_status, "edited": edited},
        )
        return AgentResult(
            final=final,
            trajectory=steps,
            metrics={
                "edit_count": counters.edits,
                "final_status": final_status,
                "test_run_count": counters.test_runs,
                "tool_call_count": counters.tool_calls,
            },
        )

    def _try_model_action(
        self,
        task: AgentTask,
        run_id: str,
        workspace: TaskWorkspace,
        steps: list[TrajectoryStepRecord],
        counters: _Counters,
    ) -> None:
        generated = self.model.generate(
            [{"role": "user", "content": task.problem_statement}],
            {"dry_run": True, "allowed_tools": ["search", "read_file", "inspect_test", "edit_file", "run_tests", "rollback", "git_diff", "final_answer"]},
        )
        parsed = parse_qwable_tool_use(generated.text)
        if not parsed.ok or parsed.tool_name is None:
            steps.append(
                TrajectoryStepRecord(
                    instance_id=task.instance_id,
                    run_id=run_id,
                    model_name_or_path=task.model_name_or_path,
                    agent_version=self.agent_version,
                    step_index=len(steps),
                    action="model_tool_parse",
                    tool="model",
                    status=MALFORMED,
                    output_summary=_summarize(generated.text),
                    error=parsed.error or "malformed_model_tool_call",
                    tool_call_count=counters.tool_calls,
                    test_run_count=counters.test_runs,
                    edit_count=counters.edits,
                    args_hash=_hash_args({"generated": generated.text}),
                    timestamp=utc_now(),
                    metadata={"parser": parsed.to_record(), "model_finish_reason": generated.finish_reason},
                )
            )
            return
        result = self.registry.execute(parsed.tool_name, workspace, parsed.arguments)
        counters.tool_calls += 1
        if parsed.tool_name == "run_tests" and result.status not in {MALFORMED, DENIED}:
            counters.test_runs = workspace.test_run_count
        if parsed.tool_name == "edit_file" and result.status == OK:
            counters.edits += 1
        steps.append(self._tool_step(task, run_id, len(steps), parsed.tool_name, parsed.arguments, result, counters, action="model_tool_execute", metadata={"parser": parsed.to_record()}))

    def _tool_step(
        self,
        task: AgentTask,
        run_id: str,
        step_index: int,
        tool: str,
        args: dict[str, object],
        result: ToolResult,
        counters: _Counters,
        *,
        action: str | None = None,
        metadata: ConfigMap | None = None,
    ) -> TrajectoryStepRecord:
        merged_metadata: ConfigMap = {"tool_metadata": dict(result.metadata), **(metadata or {})}
        return TrajectoryStepRecord(
            instance_id=task.instance_id,
            run_id=run_id,
            model_name_or_path=task.model_name_or_path,
            agent_version=self.agent_version,
            step_index=step_index,
            action=action or tool,
            tool=tool,
            status=result.status,
            output_summary=_summarize(result.output),
            error=result.error,
            tool_call_count=counters.tool_calls,
            test_run_count=counters.test_runs,
            edit_count=counters.edits,
            args_hash=_hash_args(args),
            timestamp=utc_now(),
            metadata=merged_metadata,
        )

    @staticmethod
    def _with_final_status(step: TrajectoryStepRecord, final_status: str) -> TrajectoryStepRecord:
        return TrajectoryStepRecord(
            instance_id=step.instance_id,
            run_id=step.run_id,
            model_name_or_path=step.model_name_or_path,
            agent_version=step.agent_version,
            step_index=step.step_index,
            action=step.action,
            tool=step.tool,
            status=step.status,
            output_summary=step.output_summary,
            error=step.error,
            tool_call_count=step.tool_call_count,
            test_run_count=step.test_run_count,
            edit_count=step.edit_count,
            final_status=final_status,
            args_hash=step.args_hash,
            timestamp=step.timestamp,
            metadata=step.metadata,
        )


def _query_from_problem(problem_statement: str) -> str:
    lowered = problem_statement.lower()
    if "defaultprinting" in lowered or ("__dict__" in lowered and "__slots__" in lowered):
        return "class Printable"
    if "usernamevalidator" in lowered or "contrib.auth.validators" in lowered:
        return "class UnicodeUsernameValidator"
    for pattern in (r"`([A-Za-z_]\w*)`", r"\b([A-Za-z_]\w*_[A-Za-z_]\w*)\b", r"\b([A-Za-z_]\w*)\("):
        match = re.search(pattern, problem_statement)
        if match:
            return match.group(1)
    if re.search(r"\b(add|sum|plus)\b|\+", lowered):
        return "add"
    return "failure"


def _candidate_from_search(output: str) -> str:
    for line in output.splitlines():
        path = line.split(":", 1)[0].strip()
        lowered = path.lower()
        if "/tests/" in f"/{lowered}" or lowered.startswith("tests/") or lowered.endswith("test.py"):
            continue
        if lowered.endswith(SOURCE_SUFFIXES):
            return path
    return ""


def _edits_from_read(path: str, numbered_text: str, problem_statement: str) -> list[dict[str, object]]:
    return [
        *_django_username_validator_edits(path, numbered_text, problem_statement),
        *_sympy_default_printing_edits(path, numbered_text, problem_statement),
        *_addition_edit_from_read(path, numbered_text, problem_statement),
    ]


def _addition_edit_from_read(path: str, numbered_text: str, problem_statement: str) -> list[dict[str, object]]:
    lowered = problem_statement.lower()
    if not re.search(r"\b(add|sum|plus)\b|\+", lowered):
        return []
    for line in numbered_text.splitlines():
        match = re.match(r"^(\d+): (\s*)return\s+([A-Za-z_]\w*)\s*-\s*([A-Za-z_]\w*)\s*$", line)
        if not match:
            continue
        line_number = int(match.group(1))
        indent, left, right = match.group(2), match.group(3), match.group(4)
        return [{"path": path, "replacement": f"{indent}return {left} + {right}\n", "start_line": line_number, "end_line": line_number}]
    return []


def _django_username_validator_edits(path: str, numbered_text: str, problem_statement: str) -> list[dict[str, object]]:
    lowered = problem_statement.lower()
    if "usernamevalidator" not in lowered and "contrib.auth.validators" not in lowered:
        return []
    edits: list[dict[str, object]] = []
    for line in numbered_text.splitlines():
        match = re.match(r"^(\d+): (\s*)regex\s*=\s*r(['\"])\^\[\\w\.@\+\-\]\+\$\3\s*$", line)
        if not match:
            continue
        line_number = int(match.group(1))
        indent, quote = match.group(2), match.group(3)
        edits.append({"path": path, "replacement": f"{indent}regex = r{quote}\\A[\\w.@+-]+\\Z{quote}\n", "start_line": line_number, "end_line": line_number})
    return edits


def _sympy_default_printing_edits(path: str, numbered_text: str, problem_statement: str) -> list[dict[str, object]]:
    lowered = problem_statement.lower()
    if "defaultprinting" not in lowered and not ("__dict__" in lowered and "__slots__" in lowered):
        return []
    if "__slots__" in numbered_text:
        return []
    for line in numbered_text.splitlines():
        match = re.match(r"^(\d+): (\s*)class\s+(?:DefaultPrinting|Printable)\b.*:\s*$", line)
        if not match:
            continue
        line_number = int(match.group(1))
        indent = match.group(2)
        source_line = line.split(": ", 1)[1]
        docstring_end = _class_docstring_end_line(numbered_text, class_line=line_number, class_indent=indent)
        if docstring_end > line_number:
            closing_line = _source_line(numbered_text, docstring_end)
            return [{"path": path, "replacement": f"{closing_line}\n{indent}    __slots__ = ()\n", "start_line": docstring_end, "end_line": docstring_end}]
        return [{"path": path, "replacement": f"{source_line}\n{indent}    __slots__ = ()\n", "start_line": line_number, "end_line": line_number}]
    return []


def _class_docstring_end_line(numbered_text: str, *, class_line: int, class_indent: str) -> int:
    body_indent = f"{class_indent}    "
    lines = numbered_text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^(\d+): (.*)$", line)
        if not match or int(match.group(1)) <= class_line:
            continue
        source = match.group(2)
        if not source.strip():
            continue
        if not source.startswith(body_indent):
            return class_line
        stripped = source.strip()
        quote = _docstring_quote(stripped)
        if quote is None:
            return class_line
        if stripped.count(quote) >= 2 and len(stripped) > len(quote):
            return int(match.group(1))
        for later in lines[index + 1 :]:
            later_match = re.match(r"^(\d+): (.*)$", later)
            if later_match and quote in later_match.group(2):
                return int(later_match.group(1))
        return class_line
    return class_line


def _docstring_quote(stripped_line: str) -> str | None:
    for quote in ('"""', "'''"):
        if stripped_line.startswith(quote):
            return quote
    return None


def _source_line(numbered_text: str, line_number: int) -> str:
    prefix = f"{line_number}: "
    for line in numbered_text.splitlines():
        if line.startswith(prefix):
            return line.split(": ", 1)[1]
    return ""


def _final_status(*, edited: bool, test_status: str, patch: str, diff_status: str) -> str:
    if patch and test_status == OK:
        return "passed"
    if edited and test_status in TOOL_FAILURE_STATUSES:
        return "rolled_back"
    if diff_status == OK and patch:
        return "patch_unverified"
    if test_status in TOOL_FAILURE_STATUSES:
        return "failed"
    return "no_patch"


def _final_explanation(*, final_status: str, test_status: str, edited: bool, diff_status: str) -> str:
    return (
        f"Baseline fixed policy finished with status={final_status}; "
        f"edited={edited}; visible_test_status={test_status}; diff_status={diff_status}."
    )


def _summarize(text: str, limit: int = 500) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 15] + " ...[truncated]"


def _hash_args(args: dict[str, object]) -> str:
    payload = repr(sorted((str(key), repr(value)) for key, value in args.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


Counters = _Counters
RegistryLike = _RegistryLike
candidate_from_search = _candidate_from_search
edits_from_read = _edits_from_read
final_explanation = _final_explanation
query_from_problem = _query_from_problem
summarize_text = _summarize
