from __future__ import annotations

import re
from dataclasses import dataclass
from typing import override

from repair_agent.agent.baseline import (
    PATCH_SUCCESS_STATUSES,
    TOOL_FAILURE_STATUSES,
    BaselineAgent,
    Counters,
    RegistryLike,
    candidate_from_search,
    edits_from_read,
    final_explanation,
    query_from_problem,
    summarize_text,
)
from repair_agent.agent.interface import AgentFinalAnswer, AgentResult, AgentTask, TrajectoryStepRecord
from repair_agent.agent.models import ModelAdapter
from repair_agent.config import ConfigMap
from repair_agent.tools.core import MALFORMED, OK, ToolRegistry, ToolResult, TaskWorkspace


FEEDBACK_AGENT_VERSION = "feedback-fixed-v1"


@dataclass(frozen=True)
class FeedbackReflection:
    summary: str
    previous_test_status: str
    retry_reason: str
    search_query: str
    source_step_index: int

    def to_metadata(self, *, used: bool, action_context: str) -> ConfigMap:
        return {
            "action_context": action_context,
            "feedback_summary": self.summary,
            "previous_test_status": self.previous_test_status,
            "reflection_source_step_index": self.source_step_index,
            "reflection_used": used,
            "retry_reason": self.retry_reason,
        }


class FeedbackAgent(BaselineAgent):
    agent_version: str = FEEDBACK_AGENT_VERSION

    def __init__(self, model: ModelAdapter | None = None, registry: RegistryLike | ToolRegistry | None = None) -> None:
        super().__init__(model=model, registry=registry)

    @override
    def run(self, task: AgentTask, run_id: str) -> AgentResult:
        workspace = TaskWorkspace(
            checkout_root=task.checkout_root,
            visible_tests=task.visible_tests,
            visible_failures=task.visible_failures,
            max_output_chars=task.max_output_chars,
            test_timeout_seconds=task.test_timeout_seconds,
            max_test_runs=task.max_test_runs,
        )
        counters = Counters()
        steps: list[TrajectoryStepRecord] = []
        candidate_path = ""
        initial_candidate_path = ""
        edited = False
        test_status = "not_run"
        diff_status = "not_run"
        patch = ""
        reflections: list[FeedbackReflection] = []

        def remaining() -> bool:
            return len(steps) < task.max_steps

        def reflection_metadata(tool: str, args: dict[str, object]) -> ConfigMap:
            if not reflections or tool not in {"search", "read_file", "edit_file", "rollback", "final_answer"}:
                return {}
            return reflections[-1].to_metadata(used=True, action_context=_action_context(tool, args))

        def execute(tool: str, args: dict[str, object] | None = None, metadata: ConfigMap | None = None) -> ToolResult:
            payload = args or {}
            result = self.registry.execute(tool, workspace, payload)
            counters.tool_calls += 1
            if tool == "run_tests" and result.status not in {MALFORMED}:
                counters.test_runs = workspace.test_run_count
            if tool == "edit_file" and result.status == OK:
                counters.edits += 1

            merged_metadata: ConfigMap = {**reflection_metadata(tool, payload), **(metadata or {})}
            if tool == "run_tests" and result.status not in PATCH_SUCCESS_STATUSES:
                reflection = _reflection_from_test_failure(result, step_index=len(steps), problem_statement=task.problem_statement)
                reflections.append(reflection)
                merged_metadata.update(reflection.to_metadata(used=False, action_context="observe_failed_visible_test"))
            steps.append(self._tool_step(task, run_id, len(steps), tool, payload, result, counters, metadata=merged_metadata))
            return result

        if remaining():
            search = execute("search", {"query": query_from_problem(task.problem_statement), "path": ".", "max_matches": 50})
            initial_candidate_path = candidate_from_search(search.output)
            candidate_path = initial_candidate_path
        if remaining():
            _ = execute("read_file", {"path": ".", "max_lines": 200})
        for target in [*task.visible_failures.keys(), *task.visible_tests[:1]]:
            if not remaining():
                break
            _ = execute("inspect_test", {"target": target})

        if task.visible_tests and remaining():
            tests = execute("run_tests", {"target": task.visible_tests[0], "timeout_seconds": task.test_timeout_seconds})
            test_status = tests.status

        reflection = reflections[-1] if reflections else None
        if reflection is not None and remaining():
            feedback_search = execute(
                "search",
                {"query": reflection.search_query, "path": ".", "max_matches": 50},
                metadata={"feedback_search_query": reflection.search_query, "baseline_candidate_path": initial_candidate_path},
            )
            candidate_path = candidate_from_search(feedback_search.output) or candidate_path

        last_read: ToolResult | None = None
        if candidate_path and remaining():
            last_read = execute("read_file", {"path": candidate_path, "max_lines": 300})

        if last_read is not None and candidate_path and remaining():
            for edit_args in edits_from_read(candidate_path, last_read.output, _problem_with_feedback(task.problem_statement, reflection)):
                if not remaining():
                    break
                edit = execute("edit_file", edit_args)
                edited = edited or edit.status == OK

        if not reflections and remaining():
            self._try_model_action(task, run_id, workspace, steps, counters)

        if edited and not reflections and remaining() and task.visible_tests:
            tests = execute("run_tests", {"target": task.visible_tests[0], "timeout_seconds": task.test_timeout_seconds})
            test_status = tests.status
            if tests.status in TOOL_FAILURE_STATUSES and remaining():
                _ = execute("rollback", {"reason": "feedback agent edit failed visible tests without usable reflection"})
                edited = False

        if remaining():
            diff = execute("git_diff", {})
            diff_status = diff.status
            patch = diff.output if diff.status == OK else ""

        reflection = reflections[-1] if reflections else None
        final_status = _feedback_final_status(edited=edited, test_status=test_status, patch=patch, diff_status=diff_status, reflection=reflection)
        explanation = _feedback_final_explanation(final_status=final_status, test_status=test_status, edited=edited, diff_status=diff_status, reflection=reflection)
        if remaining():
            _ = execute("final_answer", {"answer": explanation})
        steps = [self._with_final_status(step, final_status) for step in steps]
        final_metadata: ConfigMap = {
            "diff_status": diff_status,
            "edited": edited,
            "learning_updates": 0,
            "test_status": test_status,
        }
        if reflection is not None:
            final_metadata.update(reflection.to_metadata(used=True, action_context="final_answer"))
        final = AgentFinalAnswer(
            instance_id=task.instance_id,
            model_name_or_path=task.model_name_or_path,
            model_patch=patch,
            status=final_status,
            explanation=explanation,
            metadata=final_metadata,
        )
        metrics: ConfigMap = {
            "edit_count": counters.edits,
            "feedback_reflections": 1 if reflection is not None else 0,
            "final_status": final_status,
            "learning_updates": 0,
            "test_run_count": counters.test_runs,
            "tool_call_count": counters.tool_calls,
        }
        if reflection is not None:
            metrics.update({"feedback_summary": reflection.summary, "previous_test_status": reflection.previous_test_status, "retry_reason": reflection.retry_reason})
        return AgentResult(final=final, trajectory=steps, metrics=metrics)


def _reflection_from_test_failure(result: ToolResult, *, step_index: int, problem_statement: str) -> FeedbackReflection:
    summary = _summarize_test_failure(result.output or result.error or "visible test failed")
    search_query = _query_from_failure(summary) or query_from_problem(problem_statement)
    return FeedbackReflection(
        summary=summary,
        previous_test_status=result.status,
        retry_reason=f"visible test returned {result.status}; retry search/edit using failure token '{search_query}'",
        search_query=search_query,
        source_step_index=step_index,
    )


def _summarize_test_failure(output: str) -> str:
    interesting: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if any(marker in lowered for marker in ("assert", "error", "failed", "expected", "traceback", "pytest")):
            interesting.append(stripped)
        if len(interesting) >= 6:
            break
    return summarize_text(" | ".join(interesting) if interesting else output, limit=600)


def _query_from_failure(summary: str) -> str:
    for pattern in (r"\b([A-Za-z_]\w*)\(", r"\btest_([A-Za-z_]\w*)\b", r"\b([A-Za-z_]\w*_[A-Za-z_]\w*)\b"):
        match = re.search(pattern, summary)
        if match:
            value = match.group(1)
            return value if not value.startswith("test_") else value.removeprefix("test_")
    return ""


def _problem_with_feedback(problem_statement: str, reflection: FeedbackReflection | None) -> str:
    if reflection is None:
        return problem_statement
    return f"{problem_statement}\nVisible test feedback: {reflection.summary}"


def _action_context(tool: str, args: dict[str, object]) -> str:
    if tool == "search":
        return f"retry_search:{args.get('query', '')}"
    if tool == "read_file":
        return f"retry_read:{args.get('path', '')}"
    if tool == "edit_file":
        return f"retry_edit:{args.get('path', '')}"
    return f"retry_{tool}"


def _feedback_final_status(*, edited: bool, test_status: str, patch: str, diff_status: str, reflection: FeedbackReflection | None) -> str:
    if patch and test_status == OK and reflection is None:
        return "passed"
    if patch and edited:
        return "patch_unverified"
    if diff_status == OK and patch:
        return "patch_unverified"
    if test_status in TOOL_FAILURE_STATUSES:
        return "failed"
    return "no_patch"


def _feedback_final_explanation(*, final_status: str, test_status: str, edited: bool, diff_status: str, reflection: FeedbackReflection | None) -> str:
    base = final_explanation(final_status=final_status, test_status=test_status, edited=edited, diff_status=diff_status).replace("Baseline", "Feedback")
    if reflection is None:
        return base
    return f"{base} Reflection used: {reflection.summary}"
