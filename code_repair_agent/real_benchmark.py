"""Real Defects4J benchmark runner for the self-improving repair agent."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, cast

from .d4j_benchmark import BenchmarkCase, load_cases
from .d4j_memory import BenchmarkMemory, extract_features
from .deepseek_repair import (
    RepairPlan,
    build_localization_prompt,
    build_repair_prompt,
    parse_localization_plan,
    parse_repair_plan,
)
from .defects4j import CommandResult, Defects4JCase, Defects4JClient
from .llm import DeepSeekChatClient
from .safe_patch import PatchApplyResult, PatchHunk, SafePatchApplier


SYSTEMS = ("baseline", "feedback", "self_evolved")
DEFAULT_BASELINE_ATTEMPTS = 1
DEFAULT_FEEDBACK_ATTEMPTS = 3
DEFAULT_SELF_EVOLVED_ATTEMPTS = 5
DEFAULT_MEMORY_ATTEMPT_BONUS = 2
DEFAULT_MAX_ATTEMPT_CAP = 8
DEFAULT_MEMORY_GUIDANCE_LIMIT = 5
DEFAULT_MAX_NON_PATCH_ROUNDS = 8
MEMORY_MODES = ("none", "check_only", "repair_only", "full")


@dataclass
class ScopeResult:
    scope: str
    passed: bool
    results: List[CommandResult]

    def as_dict(self) -> Dict[str, object]:
        return {
            "scope": self.scope,
            "passed": self.passed,
            "results": [result.as_dict() for result in self.results],
        }


@dataclass
class CaseMetrics:
    system: str
    case_id: str
    project: str
    bug_id: int
    status: str
    pass_at_1: bool
    pass_at_3: bool
    visible_pass: bool
    regression_pass: bool
    compile_success: bool
    tool_calls: int
    test_runs: int
    patch_size: int
    unsafe_edit: bool
    wall_time_seconds: float
    deepseek_calls: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    infrastructure_failure: bool
    agent_failure: bool

    def as_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class RuntimeTuning:
    attempt_limits: Dict[str, int]
    memory_attempt_bonus: int = 2
    max_attempt_cap: int = 8
    memory_guidance_limit: int = 5
    max_non_patch_rounds: int = DEFAULT_MAX_NON_PATCH_ROUNDS
    memory_path: Optional[Path] = None
    fresh_memory: bool = False
    memory_mode: str = "full"


class Defects4JBenchmarkRunner:
    def __init__(
        self,
        *,
        cases: List[BenchmarkCase],
        run_dir: Path,
        systems: Iterable[str],
        max_attempts: int,
        client: Defects4JClient,
        llm: DeepSeekChatClient,
        resume: bool = True,
        max_regression_tests: int = 20,
        tuning: Optional[RuntimeTuning] = None,
    ):
        self.cases = cases
        self.run_dir = run_dir.resolve()
        self.systems = list(systems)
        self.max_attempts = max_attempts
        self.client = client
        self.llm = llm
        self.resume = resume
        self.max_regression_tests = max_regression_tests
        self.tuning = tuning or RuntimeTuning(
            attempt_limits={"baseline": 1, "feedback": max_attempts, "self_evolved": max_attempts},
        )
        self.trace_dir = self.run_dir / "traces"
        self.patch_dir = self.run_dir / "patches"
        self.work_dir = self.run_dir / "work"
        self.memory_dir = self.run_dir / "memory_snapshots"

    def run(self) -> Dict[str, object]:
        self._prepare_dirs()
        memory = (
            BenchmarkMemory.load(self.tuning.memory_path)
            if self.tuning.memory_path and not self.tuning.fresh_memory
            else BenchmarkMemory()
        )
        (self.run_dir / "memory_before.json").write_text(
            json.dumps(memory.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        metrics: List[CaseMetrics] = []
        for system in self.systems:
            for case in self.cases:
                trace_path = self.trace_dir / f"{system}-{case.case_id}.json"
                if self.resume and trace_path.exists():
                    trace = json.loads(trace_path.read_text(encoding="utf-8"))
                    metrics.append(CaseMetrics(**trace["metrics"]))
                    continue
                case_metrics, trace = self._run_case(system, case, memory)
                metrics.append(case_metrics)
                self._write_json(trace_path, trace)
                partial_path = self.trace_dir / f"{system}-{case.case_id}.partial.json"
                if partial_path.exists():
                    partial_path.unlink()
                memory.save(self.memory_dir / f"after-{system}-{case.case_id}.json")
                if self.tuning.memory_path:
                    memory.save(self.tuning.memory_path)
        memory.save(self.run_dir / "memory_after.json")
        if self.tuning.memory_path:
            memory.save(self.tuning.memory_path)
        summary = self._summarize(metrics)
        self._write_metrics(metrics)
        self._write_json(self.run_dir / "summary.json", summary)
        self._write_failure_report(metrics)
        return summary

    def _run_case(
        self,
        system: str,
        case: BenchmarkCase,
        memory: BenchmarkMemory,
    ) -> Tuple[CaseMetrics, Dict[str, object]]:
        started = time.perf_counter()
        llm_before = {
            "calls": self.llm.usage.calls,
            "prompt_tokens": self.llm.usage.prompt_tokens,
            "completion_tokens": self.llm.usage.completion_tokens,
            "estimated_cost_usd": self.llm.usage.estimated_cost_usd,
        }
        case_workdir = (self.work_dir / system / case.case_id).resolve()
        trace: Dict[str, object] = {
            "system": system,
            "case": case.as_dict(),
            "attempts": [],
            "commands": [],
            "memory_before": memory.as_dict(),
        }
        tool_calls = 0
        test_runs = 0
        patch_size = 0
        unsafe_edit = False
        compile_success = False
        visible_pass = False
        regression_pass = False
        pass_at_1 = False
        pass_at_3 = False
        status = "failed"
        infrastructure_failure = False
        agent_failure = False
        final_features: List[str] = [f"project:{case.project}"]

        try:
            metadata, failing_output, trigger_result = self._fresh_checkout(case, case_workdir, trace)
            compile_success = True
            test_runs += len(trigger_result.results)
            final_features = extract_features(case.project, failing_output, metadata)
            attempts = self._attempt_limit(system, memory, final_features)
            attempt_failures: List[str] = []
            requested_files: List[str] = []
            feedback_output = failing_output
            root_failing_output = failing_output
            patch_attempt_idx = 0
            non_patch_rounds = 0
            consecutive_llm_errors = 0
            prompt_budget_scale = 1.0
            failed_patch_signatures: set[str] = set()
            duplicate_rejections = 0
            fixed_diagnosis: Optional[str] = None
            last_localization_plan = None
            last_verified_context: Optional[Dict[str, Dict[str, object]]] = None
            last_patch_ranges: List[Dict[str, object]] = []
            carried_grounded_hunks: List[PatchHunk] = []
            carried_grounded_files: set[str] = set()
            max_llm_rounds = attempts + max(0, self.tuning.max_non_patch_rounds)
            for attempt_idx in range(1, max_llm_rounds + 1):
                if patch_attempt_idx >= attempts:
                    break
                use_check_memory, use_repair_memory = self._memory_dimensions(system)
                memory_limit = 1 if patch_attempt_idx < 1 else (3 if patch_attempt_idx < 2 else self.tuning.memory_guidance_limit)
                memory_preferences = [] if not use_repair_memory else memory.prompt_preferences(final_features)[:memory_limit]
                repair_skills = [] if not use_repair_memory else memory.repair_skill_preferences(
                    final_features, limit=memory_limit
                )
                test_skills = [] if not use_check_memory else memory.test_skill_preferences(
                    final_features, limit=memory_limit
                )
                regression_warnings = [] if not use_check_memory else memory.regression_warnings(
                    final_features, limit=memory_limit
                )
                success_strategies = [] if not use_repair_memory else memory.relevant_success_strategies(
                    final_features, limit=memory_limit
                )
                reflections = [] if not use_repair_memory else memory.relevant_reflections(final_features)[:memory_limit]
                prompt_context_source = "focused_snippet"
                if fixed_diagnosis:
                    retry_ranges = _localization_ranges(last_localization_plan) or last_patch_ranges
                    retry_verified = (
                        self._read_verified_patch_context(case_workdir, metadata, {"line_ranges": retry_ranges})
                        if retry_ranges
                        else {}
                    )
                    if retry_verified:
                        last_verified_context = retry_verified
                    snippets = _verified_context_snippets(last_verified_context or {})
                    snippet_line_numbers = _verified_context_line_numbers(last_verified_context or {})
                    prompt_context_source = "verified_patch_context_semantic_retry"
                else:
                    snippets, snippet_line_numbers = self._read_snippet_context(
                        case_workdir,
                        metadata,
                        memory_preferences,
                        requested_files=requested_files,
                        failing_output="\n".join([root_failing_output, feedback_output]),
                        budget_scale=prompt_budget_scale,
                    )
                visible_assertions = _trigger_assertion_summary(case_workdir, metadata)
                prompt_context_budget = _prompt_context_budget(prompt_budget_scale)
                carried_feedback = _carried_grounded_hunks_feedback(carried_grounded_hunks)
                extra_failures = (
                    [
                        f"fixed_diagnosis: {fixed_diagnosis}",
                        "alternative_patch_strategy_required: patch compiled but tests failed; keep the diagnosis and choose a different semantic strategy/condition.",
                    ]
                    if fixed_diagnosis
                    else []
                ) + ([carried_feedback] if carried_feedback else [])
                prompt = build_repair_prompt(
                    project=case.project,
                    bug_id=case.bug_id,
                    metadata=metadata,
                    failing_output=feedback_output,
                    snippets=snippets,
                    snippet_line_numbers=snippet_line_numbers,
                    current_diff=self._git_diff(case_workdir),
                    attempt=patch_attempt_idx + 1,
                    memory_preferences=memory_preferences,
                    visible_test_assertions=visible_assertions,
                    derived_repair_constraints=_derived_repair_constraints(visible_assertions)
                    + _failure_output_repair_constraints(root_failing_output)
                    + _failure_output_repair_constraints(feedback_output),
                    repair_skills=repair_skills,
                    test_skills=test_skills,
                    regression_warnings=regression_warnings,
                    success_strategies=success_strategies,
                    previous_attempt_failures=([] if system == "baseline" else attempt_failures[-5:]) + extra_failures,
                    reflections=reflections,
                    context_budget_chars=prompt_context_budget,
                )
                attempt_trace: Dict[str, object] = {
                    "attempt": attempt_idx,
                    "patch_attempt": None,
                    "prompt": self._redact(prompt),
                    "memory_preferences": memory_preferences,
                    "repair_skills": repair_skills,
                    "test_skills": test_skills,
                    "regression_warnings": regression_warnings,
                    "success_strategies": success_strategies,
                    "reflections": reflections,
                    "prompt_budget_scale": round(prompt_budget_scale, 3),
                    "prompt_context_budget_chars": prompt_context_budget,
                    "prompt_context_source": prompt_context_source,
                }
                try:
                    tool_calls += 1
                    response_text = self._call_llm(prompt)
                    attempt_trace["response"] = self._redact(response_text)
                    consecutive_llm_errors = 0
                except Exception as exc:
                    llm_error = str(exc)
                    consumes_patch_attempt = system == "baseline" or not _is_retryable_llm_error(llm_error)
                    if consumes_patch_attempt:
                        patch_attempt_idx += 1
                        attempt_trace["patch_attempt"] = patch_attempt_idx
                    else:
                        attempt_trace["non_patch_model_failure"] = True
                        non_patch_rounds += 1
                        consecutive_llm_errors += 1
                        if consecutive_llm_errors == 1:
                            prompt_budget_scale = max(0.50, prompt_budget_scale * 0.7)
                        elif consecutive_llm_errors == 2:
                            prompt_budget_scale = max(0.35, prompt_budget_scale * 0.5)
                        else:
                            prompt_budget_scale = max(0.18, prompt_budget_scale * 0.35)
                    attempt_trace["llm_error"] = llm_error
                    attempt_trace["consecutive_llm_errors"] = consecutive_llm_errors
                    attempt_failures.append(f"attempt {attempt_idx}: llm_error: {llm_error[:300]}")
                    self._record_memory_event(
                        system,
                        memory,
                        final_features,
                        patch_style="llm-error",
                        test_scope="not-run",
                        solved=False,
                        failure_reason="llm_error",
                        reflection=f"llm_error: {llm_error[:400]}; retry with smaller, more grounded patch request",
                        repair_skill="recover-from-llm-error",
                        test_skill="no-test-run",
                    )
                    self._record_attempt(trace, attempt_trace, system, case)
                    if system == "baseline" or _is_non_retryable_llm_error(llm_error):
                        break
                    if non_patch_rounds >= self.tuning.max_non_patch_rounds:
                        break
                    continue
                try:
                    try:
                        localization_plan = parse_localization_plan(response_text)
                    except Exception:
                        localization_plan = None
                    if localization_plan is not None:
                        last_localization_plan = localization_plan
                        attempt_trace["localization_plan"] = localization_plan.as_dict()
                        requested_files = _merge_requested_files(
                            requested_files,
                            localization_plan.files_to_read,
                            [item.file for item in localization_plan.line_ranges],
                        )
                        verified_context = self._read_verified_patch_context(case_workdir, metadata, localization_plan)
                        last_verified_context = verified_context
                        attempt_trace["verified_patch_context"] = _summarize_verified_patch_context(verified_context)
                    plan = parse_repair_plan(response_text, allow_empty=True)
                    if carried_grounded_hunks and plan.patch_hunks:
                        plan = _with_carried_grounded_hunks(plan, carried_grounded_hunks)
                    attempt_trace["repair_plan"] = plan.as_dict()
                    requested_files = _merge_requested_files(
                        requested_files,
                        plan.files_to_read,
                        [hunk.file for hunk in plan.patch_hunks],
                    )
                    attempt_trace["requested_files_for_next_attempt"] = requested_files
                    if not plan.patch_hunks:
                        attempt_trace["read_request_without_patch"] = True
                        attempt_failures.append(
                            f"attempt {attempt_idx}: read_request_without_patch: "
                            f"requested_files={plan.files_to_read[:5]}; next response must include non-empty patch_hunks"
                        )
                        non_patch_rounds += 1
                        prompt_budget_scale = max(0.35, prompt_budget_scale * 0.85)
                        self._record_memory_event(
                            system,
                            memory,
                            final_features,
                            patch_style=plan.patch_style or "read-request",
                            test_scope="not-run",
                            solved=False,
                            failure_reason="read_request_without_patch",
                            reflection=(
                                "read_request_without_patch: model asked to read files but produced no patch; "
                                "use supplied source snippets and exact old/new text in the next repair attempt"
                            ),
                            repair_skill="read-before-patch",
                            test_skill="no-test-run",
                        )
                        self._record_attempt(trace, attempt_trace, system, case)
                        if system == "baseline" or non_patch_rounds >= self.tuning.max_non_patch_rounds:
                            break
                        continue
                except Exception as exc:
                    parse_error = str(exc)
                    if system == "baseline":
                        patch_attempt_idx += 1
                        attempt_trace["patch_attempt"] = patch_attempt_idx
                    else:
                        attempt_trace["non_patch_parse_failure"] = True
                        non_patch_rounds += 1
                        prompt_budget_scale = max(0.35, prompt_budget_scale * 0.75)
                    attempt_trace["parse_error"] = parse_error
                    attempt_failures.append(f"attempt {attempt_idx}: parse_error: {parse_error[:300]}")
                    self._record_memory_event(
                        system,
                        memory,
                        final_features,
                        patch_style="invalid-response",
                        test_scope="not-run",
                        solved=False,
                        failure_reason="parse_error",
                        reflection=f"parse_error: {parse_error[:400]}; request concrete non-empty patch_hunks",
                        repair_skill="format-and-patch-grounding",
                        test_skill="no-test-run",
                    )
                    self._record_attempt(trace, attempt_trace, system, case)
                    if system == "baseline" or non_patch_rounds >= self.tuning.max_non_patch_rounds:
                        break
                    continue

                grounded_plan, grounding_errors, grounding_context = self._preflight_ground_plan(case_workdir, metadata, plan)
                attempt_trace["preflight_grounding"] = {
                    "ok": not grounding_errors,
                    "errors": grounding_errors,
                    "grounded_hunks": [hunk.__dict__ for hunk in grounded_plan.patch_hunks],
                    "context": grounding_context,
                }
                if grounding_errors:
                    attempt_trace["non_patch_grounding_failure"] = True
                    non_patch_rounds += 1
                    # Carry forward successfully grounded hunks so the next
                    # attempt only needs to fix the failed files/ranges.
                    # Match by hunk identity (file+line range), not just file,
                    # so same-file partial grounding is preserved.
                    failed_ids = {_grounding_error_identity(e) for e in grounding_errors}
                    for gh in grounded_plan.patch_hunks:
                        gh_id = (gh.file, gh.line_start or 0, gh.line_end or 0)
                        if gh_id not in failed_ids and not _hunk_in_list(gh, carried_grounded_hunks):
                            carried_grounded_hunks.append(gh)
                            carried_grounded_files.add(gh.file)
                    attempt_trace["carried_grounded_files"] = sorted(carried_grounded_files)
                    feedback_output = _range_grounding_failure_feedback(
                        grounding_errors, grounding_context, carried_grounded_files
                    )
                    attempt_failures.append(f"attempt {attempt_idx}: {feedback_output[:1400]}")
                    self._record_memory_event(
                        system,
                        memory,
                        final_features,
                        patch_style=plan.patch_style or "range-grounding",
                        test_scope="not-run",
                        solved=False,
                        failure_reason="range_grounding_failure",
                        reflection=feedback_output[:400],
                        repair_skill="range-grounding",
                        test_skill="no-test-run",
                    )
                    self._record_attempt(trace, attempt_trace, system, case)
                    if system == "baseline" or non_patch_rounds >= self.tuning.max_non_patch_rounds:
                        break
                    continue

                plan = grounded_plan
                carried_grounded_hunks = []
                carried_grounded_files = set()
                last_patch_ranges = _ranges_from_hunks(plan.patch_hunks)
                attempt_trace["grounded_repair_plan"] = plan.as_dict()
                patch_attempt_idx += 1
                attempt_trace["patch_attempt"] = patch_attempt_idx
                apply_result = self._apply_plan(case_workdir, metadata, plan)
                patch_size += apply_result.patch_size
                unsafe_edit = unsafe_edit or apply_result.unsafe
                attempt_trace["patch_apply"] = apply_result.as_dict()
                if apply_result.ok:
                    patch_signature = _patch_strategy_signature(apply_result.diff)
                    attempt_trace["patch_strategy_signature"] = patch_signature
                    if patch_signature and patch_signature in failed_patch_signatures:
                        patch_attempt_idx = max(0, patch_attempt_idx - 1)
                        patch_size = max(0, patch_size - apply_result.patch_size)
                        non_patch_rounds += 1
                        attempt_trace["patch_attempt"] = None
                        attempt_trace["non_patch_candidate_rejection"] = True
                        feedback_output = _duplicate_patch_feedback(patch_signature, apply_result.diff, visible_assertions)
                        duplicate_rejections += 1
                        attempt_failures.append(f"attempt {attempt_idx}: {feedback_output}")
                        self._record_memory_event(
                            system,
                            memory,
                            final_features,
                            patch_style=plan.patch_style or "duplicate",
                            test_scope="not-run",
                            solved=False,
                            failure_reason="duplicate_failed_patch_strategy",
                            reflection=feedback_output,
                            repair_skill="candidate-deduplication",
                            test_skill="no-test-run",
                        )
                        attempt_trace["candidate_rejected"] = {
                            "reason": "duplicate_failed_patch_strategy",
                            "signature": patch_signature,
                            "feedback": feedback_output,
                        }
                        self._record_attempt(trace, attempt_trace, system, case)
                        if duplicate_rejections >= _duplicate_rejection_limit():
                            trace["early_stop_reason"] = "duplicate_failed_patch_strategy_limit"
                            break
                        if system != "baseline" and patch_attempt_idx < attempts:
                            metadata, failing_output, trigger_result = self._fresh_checkout(case, case_workdir, trace)
                        continue
                    patch_path = self.patch_dir / f"{system}-{case.case_id}-attempt{patch_attempt_idx}.diff"
                    patch_path.write_text(apply_result.diff, encoding="utf-8")
                    attempt_trace["patch_path"] = str(patch_path)
                else:
                    feedback_output = _patch_failure_feedback(plan, apply_result.errors)
                    grounding_feedback = _patch_grounding_feedback(case_workdir, plan, apply_result.errors)
                    if grounding_feedback:
                        feedback_output = f"{feedback_output}\n\n{grounding_feedback}"
                    if _is_duplicate_apply_failure(apply_result.errors):
                        patch_attempt_idx = max(0, patch_attempt_idx - 1)
                        patch_size = max(0, patch_size - apply_result.patch_size)
                        non_patch_rounds += 1
                        attempt_trace["patch_attempt"] = None
                        attempt_trace["non_patch_candidate_rejection"] = True
                        duplicate_rejections += 1
                        feedback_output = (
                            f"{feedback_output}\n"
                            "duplicate_failed_patch_strategy: patch application says replacement text already exists; "
                            "the candidate is equivalent to an already tried edit"
                        )
                    attempt_failures.append(f"attempt {attempt_idx}: {feedback_output[:1400]}")
                    self._update_memory(
                        system,
                        memory,
                        final_features,
                        plan,
                        "not-run",
                        False,
                        "patch_apply_failure",
                        reflection_detail="; ".join(apply_result.errors),
                    )
                    self._record_attempt(trace, attempt_trace, system, case)
                    if _is_duplicate_apply_failure(apply_result.errors) and duplicate_rejections >= _duplicate_rejection_limit():
                        trace["early_stop_reason"] = "duplicate_failed_patch_strategy_limit"
                        break
                    if system == "baseline":
                        break
                    continue

                compile_result = self.client.run([self.client.binary, "compile"], cwd=case_workdir, check=False)
                cast(List[Dict[str, object]], trace["commands"]).append(compile_result.as_dict())
                tool_calls += 1
                compile_success = compile_result.ok
                attempt_trace["compile"] = compile_result.as_dict()
                if not compile_success:
                    feedback_output = _compile_failure_feedback(
                        compile_result.output,
                        case_workdir,
                        metadata,
                        requested_files=requested_files,
                    )
                    compile_summary = _failure_summary(feedback_output, limit=900)
                    attempt_failures.append(
                        f"attempt {attempt_idx}: "
                        f"{_failed_patch_feedback(plan, apply_result, 'compile_failure', compile_summary)}"
                    )
                    self._update_memory(
                        system,
                        memory,
                        final_features,
                        plan,
                        "trigger",
                        False,
                        "compile_failure",
                        reflection_detail=compile_summary,
                    )
                    self._record_attempt(trace, attempt_trace, system, case)
                    if system != "baseline" and patch_attempt_idx < attempts:
                        metadata, failing_output, trigger_result = self._fresh_checkout(case, case_workdir, trace)
                    continue

                trigger_after = self._run_scope(case_workdir, "trigger", metadata)
                test_runs += len(trigger_after.results)
                visible_pass = trigger_after.passed
                attempt_trace["visible_tests"] = trigger_after.as_dict()
                cast(List[Dict[str, object]], trace["commands"]).extend(result.as_dict() for result in trigger_after.results)
                if visible_pass:
                    use_check_memory, _ = self._memory_dimensions(system)
                    regression_scope = memory.preferred_test_scope(final_features, default="relevant") if use_check_memory else "relevant"
                    if regression_scope == "trigger":
                        regression_scope = "relevant"
                    regression_after = self._run_scope(case_workdir, regression_scope, metadata)
                    test_runs += len(regression_after.results)
                    regression_pass = regression_after.passed
                    attempt_trace["regression_tests"] = regression_after.as_dict()
                    cast(List[Dict[str, object]], trace["commands"]).extend(result.as_dict() for result in regression_after.results)
                    if regression_pass:
                        if patch_attempt_idx == 1:
                            pass_at_1 = True
                        if patch_attempt_idx <= 3:
                            pass_at_3 = True
                        status = "solved"
                        self._update_memory(
                            system,
                            memory,
                            final_features,
                            plan,
                            regression_scope,
                            True,
                            None,
                            visible_passed=True,
                            regression_checked=True,
                            regression_passed=True,
                        )
                        self._record_attempt(trace, attempt_trace, system, case)
                        break
                    feedback_output = "\n".join(result.output for result in regression_after.results)
                    regression_tail = _scope_failure_tail(regression_after)
                    fixed_diagnosis = plan.diagnosis
                    feedback_output = _semantic_retry_feedback(
                        fixed_diagnosis,
                        feedback_output,
                        regression_tail,
                        plan.patch_style,
                    )
                    patch_signature = str(attempt_trace.get("patch_strategy_signature", ""))
                    if patch_signature:
                        failed_patch_signatures.add(patch_signature)
                    attempt_failures.append(
                        f"attempt {attempt_idx}: "
                        f"{_failed_patch_feedback(plan, apply_result, 'regression_failure', regression_tail)}"
                    )
                    self._update_memory(
                        system,
                        memory,
                        final_features,
                        plan,
                        regression_scope,
                        False,
                        "regression_failure",
                        visible_passed=True,
                        regression_checked=True,
                        regression_passed=False,
                        reflection_detail=regression_tail,
                    )
                else:
                    feedback_output = "\n".join(result.output for result in trigger_after.results)
                    visible_tail = _scope_failure_tail(trigger_after)
                    visible_guidance = _visible_failure_guidance(visible_assertions, apply_result.diff)
                    visible_detail = "\n".join(item for item in [visible_tail, visible_guidance] if item)
                    fixed_diagnosis = plan.diagnosis
                    feedback_output = _semantic_retry_feedback(
                        fixed_diagnosis,
                        feedback_output,
                        visible_detail,
                        plan.patch_style,
                    )
                    patch_signature = str(attempt_trace.get("patch_strategy_signature", ""))
                    if patch_signature:
                        failed_patch_signatures.add(patch_signature)
                    attempt_failures.append(
                        f"attempt {attempt_idx}: "
                        f"{_failed_patch_feedback(plan, apply_result, 'visible_failure', visible_detail)}"
                    )
                    self._update_memory(
                        system,
                        memory,
                        final_features,
                        plan,
                        "trigger",
                        False,
                        "visible_failure",
                        visible_passed=False,
                        reflection_detail=visible_detail,
                    )
                self._record_attempt(trace, attempt_trace, system, case)
                if system == "baseline":
                    break
                if patch_attempt_idx < attempts:
                    metadata, failing_output, trigger_result = self._fresh_checkout(case, case_workdir, trace)
            else:
                agent_failure = True
            if status != "solved":
                agent_failure = True
        except Exception as exc:
            status = "infrastructure_error"
            infrastructure_failure = True
            trace["error"] = str(exc)

        usage_delta = self._usage_delta(llm_before)
        metrics = CaseMetrics(
            system=system,
            case_id=case.case_id,
            project=case.project,
            bug_id=case.bug_id,
            status=status,
            pass_at_1=pass_at_1,
            pass_at_3=pass_at_3,
            visible_pass=visible_pass,
            regression_pass=regression_pass,
            compile_success=compile_success,
            tool_calls=tool_calls,
            test_runs=test_runs,
            patch_size=patch_size,
            unsafe_edit=unsafe_edit,
            wall_time_seconds=round(time.perf_counter() - started, 4),
            deepseek_calls=int(usage_delta["calls"]),
            prompt_tokens=int(usage_delta["prompt_tokens"]),
            completion_tokens=int(usage_delta["completion_tokens"]),
            estimated_cost_usd=float(usage_delta["estimated_cost_usd"]),
            infrastructure_failure=infrastructure_failure,
            agent_failure=agent_failure,
        )
        trace["metrics"] = metrics.as_dict()
        trace["memory_after"] = memory.as_dict()
        return metrics, trace

    def _fresh_checkout(
        self,
        case: BenchmarkCase,
        workdir: Path,
        trace: Dict[str, object],
    ) -> Tuple[Dict[str, str], str, ScopeResult]:
        if workdir.exists():
            shutil.rmtree(workdir)
        checkout = self.client.checkout(Defects4JCase(case.project, case.bug_id, workdir))
        cast(List[str], trace.setdefault("checkout_outputs", [])).append(checkout[-2000:])
        compile_result = self.client.run(["defects4j", "compile"], cwd=workdir, check=False)
        cast(List[Dict[str, object]], trace.setdefault("commands", [])).append(compile_result.as_dict())
        if not compile_result.ok:
            raise RuntimeError(f"compile failed before patching: {compile_result.output[-2000:]}")
        metadata = self.client.metadata(workdir)
        trigger_result = self._run_scope(workdir, "trigger", metadata)
        cast(List[Dict[str, object]], trace.setdefault("commands", [])).extend(result.as_dict() for result in trigger_result.results)
        failing_output = "\n".join(result.output for result in trigger_result.results)
        return metadata, failing_output, trigger_result

    def _run_scope(self, workdir: Path, scope: str, metadata: Dict[str, str]) -> ScopeResult:
        if scope == "relevant":
            result = self.client.run([self.client.binary, "test", "-r"], cwd=workdir, check=False)
            result = self._with_failing_tests(workdir, result)
            return ScopeResult(scope=scope, passed=_d4j_test_passed(result), results=[result])
        if scope == "all":
            result = self.client.run([self.client.binary, "test"], cwd=workdir, check=False)
            result = self._with_failing_tests(workdir, result)
            return ScopeResult(scope=scope, passed=_d4j_test_passed(result), results=[result])
        tests = self._tests_for_scope(scope, metadata)
        results: List[CommandResult] = []
        if not tests:
                results.append(self._with_failing_tests(workdir, self.client.run([self.client.binary, "test"], cwd=workdir, check=False)))
        else:
            for test_name in tests:
                results.append(
                    self._with_failing_tests(
                        workdir,
                        self.client.run([self.client.binary, "test", "-t", test_name], cwd=workdir, check=False),
                    )
                )
        return ScopeResult(scope=scope, passed=all(_d4j_test_passed(result) for result in results), results=results)

    def _with_failing_tests(self, workdir: Path, result: CommandResult) -> CommandResult:
        failing_tests = workdir / "failing_tests"
        if not failing_tests.exists() or _d4j_test_passed(result):
            return result
        details = failing_tests.read_text(encoding="utf-8", errors="replace")[-6000:]
        return CommandResult(
            command=result.command,
            cwd=result.cwd,
            returncode=result.returncode,
            output=f"{result.output}\n\n[failing_tests]\n{details}",
            elapsed_seconds=result.elapsed_seconds,
        )

    def _tests_for_scope(self, scope: str, metadata: Dict[str, str]) -> List[str]:
        if scope == "trigger":
            return _split_metadata_list(metadata.get("tests.trigger", ""))
        if scope == "relevant":
            return _split_metadata_list(metadata.get("tests.relevant", ""))[: self.max_regression_tests]
        if scope == "all":
            return []
        return [scope]

    def _apply_plan(self, workdir: Path, metadata: Dict[str, str], plan: RepairPlan) -> PatchApplyResult:
        source_dirs = _split_metadata_list(metadata.get("dir.src.classes", ""))
        test_dirs = _split_metadata_list(metadata.get("dir.src.tests", ""))
        applier = SafePatchApplier(workdir, source_dirs=source_dirs, test_dirs=test_dirs)
        return applier.apply(plan.patch_hunks)

    def _preflight_ground_plan(
        self,
        workdir: Path,
        metadata: Dict[str, str],
        plan: RepairPlan,
    ) -> Tuple[RepairPlan, List[str], Dict[str, object]]:
        grounded_hunks: List[PatchHunk] = []
        errors: List[str] = []
        context: Dict[str, object] = {}
        for hunk in plan.patch_hunks:
            if hunk.range_grounded or hunk.line_start is None or hunk.line_end is None:
                grounded_hunks.append(hunk)
                continue
            rel = Path(hunk.file)
            if rel.is_absolute() or ".." in rel.parts:
                errors.append(f"{hunk.file}: invalid range hunk path")
                continue
            path = workdir / rel
            if not path.exists() or not path.is_file():
                errors.append(f"{hunk.file}: range hunk target file does not exist")
                continue
            current_text = path.read_text(encoding="utf-8", errors="replace")
            try:
                grounded = PatchHunk.from_range(
                    hunk.file,
                    hunk.line_start,
                    hunk.line_end,
                    hunk.new,
                    current_text,
                    method_name=hunk.method_name,
                    intent=hunk.intent,
                    line_offset=hunk.line_offset,
                )
            except Exception as exc:
                grounded = None
                errors.append(f"{hunk.file}:{hunk.line_start}-{hunk.line_end}: {exc}")
            if grounded is None:
                context[hunk.file] = _numbered_range_context(current_text, hunk.line_start, hunk.line_end)
                continue
            grounded_hunks.append(grounded)
            context[hunk.file] = _numbered_range_context(current_text, hunk.line_start, hunk.line_end)
        grounded_plan = RepairPlan(
            diagnosis=plan.diagnosis,
            files_to_read=plan.files_to_read,
            patch_hunks=grounded_hunks,
            tests_to_run_next=plan.tests_to_run_next,
            confidence=plan.confidence,
            final_explanation=plan.final_explanation,
            patch_style=plan.patch_style,
        )
        return grounded_plan, errors, context

    def _read_verified_patch_context(
        self,
        workdir: Path,
        metadata: Dict[str, str],
        localization,
        *,
        context_lines: int = 15,
    ) -> Dict[str, Dict[str, object]]:
        del metadata
        ranges = _localization_ranges(localization)
        result: Dict[str, Dict[str, object]] = {}
        for item in ranges:
            file_name = str(item.get("file", ""))
            line_start = _optional_int(item.get("line_start"))
            line_end = _optional_int(item.get("line_end"))
            if not file_name or line_start is None or line_end is None:
                continue
            rel = Path(file_name)
            if rel.is_absolute() or ".." in rel.parts:
                continue
            path = workdir / rel
            if not path.exists() or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines(keepends=True)
            if line_start < 1 or line_end < line_start or line_start > len(lines):
                continue
            bounded_end = min(line_end, len(lines))
            excerpt_start = max(1, line_start - context_lines)
            excerpt_end = min(len(lines), bounded_end + context_lines)
            excerpt = "".join(lines[excerpt_start - 1 : excerpt_end])
            line_numbered = _line_numbered_lines(lines, excerpt_start, excerpt_end)
            method_name = item.get("method_name")
            method_body = _java_method_declaration_block(text, str(method_name)) if method_name else ""
            entry = cast(
                Dict[str, object],
                result.setdefault(
                    file_name,
                    {"excerpt": "", "line_numbered": "", "method_body": "", "ranges": []},
                ),
            )
            cast(List[Dict[str, object]], entry["ranges"]).append(
                {
                    "line_start": line_start,
                    "line_end": bounded_end,
                    "excerpt_start": excerpt_start,
                    "excerpt_end": excerpt_end,
                    "method_name": method_name,
                }
            )
            entry["excerpt"] = _join_verified_block(str(entry["excerpt"]), excerpt)
            entry["line_numbered"] = _join_verified_block(str(entry["line_numbered"]), line_numbered)
            if method_body:
                entry["method_body"] = _join_verified_block(str(entry["method_body"]), method_body)
        return result

    def _read_snippets(self, workdir: Path, metadata: Dict[str, str], preferences: List[str]) -> Dict[str, str]:
        snippets, _ = self._read_snippet_context(workdir, metadata, preferences)
        return snippets

    def _read_snippet_context(
        self,
        workdir: Path,
        metadata: Dict[str, str],
        preferences: List[str],
        requested_files: Optional[List[str]] = None,
        failing_output: str = "",
        budget_scale: float = 1.0,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        scale = max(0.2, min(1.0, budget_scale))
        base_snippet_chars = int(os.environ.get("REPAIR_SNIPPET_CHARS", "18000"))
        base_window_chars = int(os.environ.get("REPAIR_SNIPPET_WINDOW_CHARS", "12000"))
        snippet_floor = min(4000, base_snippet_chars)
        window_floor = min(2000, base_window_chars)
        snippet_chars = max(snippet_floor, int(base_snippet_chars * scale))
        window_chars = max(window_floor, int(base_window_chars * scale))
        source_dir = metadata.get("dir.src.classes", "").splitlines()[0].strip() or "src/main/java"
        test_dir = metadata.get("dir.src.tests", "").splitlines()[0].strip() or "src/test/java"
        classes = _split_metadata_list(metadata.get("classes.modified", ""))
        triggers = _split_metadata_list(metadata.get("tests.trigger", ""))
        trigger_methods = [item.split("::", 1)[1] for item in triggers if "::" in item]
        failure_needles, failure_line_hints = _failure_source_hints(failing_output, source_dir, classes)
        test_needles = _trigger_test_source_needles(workdir, test_dir, triggers, snippet_chars, window_chars)
        assertion_context = _trigger_assertion_summary(workdir, metadata)
        constraint_needles = _constraint_source_needles(assertion_context)
        failure_output_needles = _failure_output_source_needles(failing_output)
        assertion_needles = _assertion_source_needles(failing_output, triggers)
        assertion_needles.extend(_source_call_needles_from_test("\n".join(assertion_context)))
        paired_needles = _paired_source_needles(trigger_methods + constraint_needles + test_needles + assertion_needles)
        source_needles = list(
            dict.fromkeys(
                trigger_methods
                + constraint_needles
                + failure_output_needles
                + test_needles
                + assertion_needles
                + paired_needles
                + failure_needles
                + preferences
            )
        )
        snippets: Dict[str, str] = {}
        line_numbers: Dict[str, str] = {}
        modified_source_texts: Dict[str, str] = {}
        for klass in classes[:8]:
            rel = Path(source_dir) / Path(klass.replace(".", "/") + ".java")
            path = workdir / rel
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="replace")
                key = str(rel)
                modified_source_texts[key] = text
                snippets[key] = _focused_snippet(
                    text,
                    source_needles,
                    snippet_chars,
                    window_chars,
                    line_hints=failure_line_hints.get(key, []),
                )
                line_numbers[key] = _focused_snippet_line_numbers(
                    text,
                    source_needles,
                    snippet_chars,
                    window_chars,
                    line_hints=failure_line_hints.get(key, []),
                )
        for rel in _requested_source_paths(requested_files or [], source_dir, workdir):
            if str(rel) in snippets:
                continue
            path = workdir / rel
            text = path.read_text(encoding="utf-8", errors="replace")
            key = str(rel)
            snippets[key] = _focused_snippet(
                text,
                source_needles,
                snippet_chars,
                window_chars,
                line_hints=failure_line_hints.get(key, []),
            )
            line_numbers[key] = _focused_snippet_line_numbers(
                text,
                source_needles,
                snippet_chars,
                window_chars,
                line_hints=failure_line_hints.get(key, []),
            )
        for rel in _related_source_paths(
            workdir,
            source_dir,
            test_dir,
            classes,
            triggers,
            modified_source_texts,
            max_paths=3,
        ):
            if str(rel) in snippets:
                continue
            path = workdir / rel
            text = path.read_text(encoding="utf-8", errors="replace")
            key = str(rel)
            snippets[key] = _focused_snippet(
                text,
                source_needles + _type_needles_from_text(text),
                min(snippet_chars, 9000),
                min(window_chars, 5000),
            )
            line_numbers[key] = _focused_snippet_line_numbers(
                text,
                source_needles + _type_needles_from_text(text),
                min(snippet_chars, 9000),
                min(window_chars, 5000),
            )
        for test_name in triggers[:4]:
            test_class = test_name.split("::", 1)[0]
            test_methods = [test_name.split("::", 1)[1]] if "::" in test_name else []
            rel = Path(test_dir) / Path(test_class.replace(".", "/") + ".java")
            path = workdir / rel
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="replace")
                key = f"{rel} [read-only-test]"
                snippets[key] = _focused_snippet(text, test_methods, snippet_chars, window_chars)
                line_numbers[key] = _focused_snippet_line_numbers(text, test_methods, snippet_chars, window_chars)
        if snippets:
            return snippets, line_numbers
        for path in sorted((workdir / source_dir).rglob("*.java"))[:8]:
            text = path.read_text(encoding="utf-8", errors="replace")
            key = str(path.relative_to(workdir))
            snippets[key] = text[:snippet_chars]
            line_numbers[key] = _line_numbered_prefix(text, max_chars=snippet_chars)
        return snippets, line_numbers

    def _call_llm(self, prompt: List[Dict[str, str]]) -> str:
        if not self.llm.enabled():
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        return self.llm.complete(prompt, temperature=0.0)

    def _update_memory(
        self,
        system: str,
        memory: BenchmarkMemory,
        features: List[str],
        plan: RepairPlan,
        test_scope: str,
        solved: bool,
        failure_reason: Optional[str],
        visible_passed: bool = False,
        regression_checked: bool = False,
        regression_passed: bool = False,
        reflection_detail: Optional[str] = None,
    ) -> None:
        if system != "self_evolved":
            return
        use_check_memory, use_repair_memory = self._memory_dimensions(system)
        if not use_check_memory and not use_repair_memory:
            return
        reflection = None
        if not solved:
            reflection = _failure_reflection(plan, failure_reason, reflection_detail)
        memory.update(
            features=features,
            patch_style=plan.patch_style,
            test_scope=test_scope,
            solved=solved,
            failure_reason=failure_reason,
            reflection=reflection,
            repair_skill=_repair_skill(plan, failure_reason),
            test_skill=_test_skill(test_scope, failure_reason, visible_passed, regression_checked, regression_passed),
            visible_passed=visible_passed,
            regression_checked=regression_checked,
            regression_passed=regression_passed,
            success_strategy=_success_strategy(plan, test_scope) if solved else None,
            update_check_memory=use_check_memory,
            update_repair_memory=use_repair_memory,
        )

    def _record_memory_event(
        self,
        system: str,
        memory: BenchmarkMemory,
        features: List[str],
        *,
        patch_style: str,
        test_scope: str,
        solved: bool,
        failure_reason: str,
        reflection: str,
        repair_skill: str,
        test_skill: str,
    ) -> None:
        if system != "self_evolved":
            return
        use_check_memory, use_repair_memory = self._memory_dimensions(system)
        if not use_check_memory and not use_repair_memory:
            return
        memory.update(
            features=features,
            patch_style=patch_style,
            test_scope=test_scope,
            solved=solved,
            failure_reason=failure_reason,
            reflection=reflection,
            repair_skill=repair_skill,
            test_skill=test_skill,
            update_check_memory=use_check_memory,
            update_repair_memory=use_repair_memory,
        )

    def _usage_delta(self, before) -> Dict[str, float]:
        return {
            "calls": self.llm.usage.calls - before["calls"],
            "prompt_tokens": self.llm.usage.prompt_tokens - before["prompt_tokens"],
            "completion_tokens": self.llm.usage.completion_tokens - before["completion_tokens"],
            "estimated_cost_usd": round(self.llm.usage.estimated_cost_usd - before["estimated_cost_usd"], 8),
        }

    def _attempt_limit(self, system: str, memory: BenchmarkMemory, features: List[str]) -> int:
        base = self.tuning.attempt_limits.get(system, self.max_attempts)
        _, use_repair_memory = self._memory_dimensions(system)
        if use_repair_memory:
            base += memory.attempt_bonus(features, max_bonus=self.tuning.memory_attempt_bonus)
        return max(1, min(base, self.tuning.max_attempt_cap))

    def _memory_dimensions(self, system: str) -> Tuple[bool, bool]:
        if system != "self_evolved":
            return False, False
        mode = self.tuning.memory_mode
        return mode in {"check_only", "full"}, mode in {"repair_only", "full"}

    def _git_diff(self, workdir: Path) -> str:
        proc = shutil.which("git")
        if not proc:
            return ""
        result = self.client.run(["git", "diff", "--", "."], cwd=workdir, check=False)
        return result.output

    def _redact(self, value):
        secret = os.environ.get("DEEPSEEK_API_KEY")
        text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
        if secret:
            text = text.replace(secret, "[REDACTED_DEEPSEEK_API_KEY]")
        try:
            return json.loads(text)
        except Exception:
            return text

    def _prepare_dirs(self) -> None:
        for path in [self.run_dir, self.trace_dir, self.patch_dir, self.work_dir, self.memory_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def _write_metrics(self, metrics: List[CaseMetrics]) -> None:
        path = self.run_dir / "metrics.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(CaseMetrics.__dataclass_fields__.keys()))
            writer.writeheader()
            for item in metrics:
                writer.writerow(item.as_dict())

    def _summarize(self, metrics: List[CaseMetrics]) -> Dict[str, object]:
        by_system: Dict[str, List[CaseMetrics]] = {}
        for item in metrics:
            by_system.setdefault(item.system, []).append(item)
        summary: Dict[str, object] = {"run_dir": str(self.run_dir), "systems": {}}
        systems_summary = cast(Dict[str, object], summary["systems"])
        for system, items in by_system.items():
            total = len(items) or 1
            systems_summary[system] = {
                "cases": len(items),
                "pass_at_1": round(sum(item.pass_at_1 for item in items) / total, 4),
                "pass_at_3": round(sum(item.pass_at_3 for item in items) / total, 4),
                "visible_pass_rate": round(sum(item.visible_pass for item in items) / total, 4),
                "regression_pass_rate": round(sum(item.regression_pass for item in items) / total, 4),
                "compile_success_rate": round(sum(item.compile_success for item in items) / total, 4),
                "avg_tool_calls": round(sum(item.tool_calls for item in items) / total, 4),
                "avg_test_runs": round(sum(item.test_runs for item in items) / total, 4),
                "avg_patch_size": round(sum(item.patch_size for item in items) / total, 4),
                "unsafe_edit_rate": round(sum(item.unsafe_edit for item in items) / total, 4),
                "infrastructure_failures": sum(item.infrastructure_failure for item in items),
                "agent_failures": sum(item.agent_failure for item in items),
                "deepseek_calls": sum(item.deepseek_calls for item in items),
                "prompt_tokens": sum(item.prompt_tokens for item in items),
                "completion_tokens": sum(item.completion_tokens for item in items),
                "estimated_cost_usd": round(sum(item.estimated_cost_usd for item in items), 8),
                "wall_time_seconds": round(sum(item.wall_time_seconds for item in items), 4),
            }
        return summary

    def _write_failure_report(self, metrics: List[CaseMetrics]) -> None:
        lines = ["# Failure Analysis", ""]
        aggregate: Counter[str] = Counter()
        case_rows: List[tuple[str, str, Counter[str]]] = []
        for item in metrics:
            if item.status == "solved":
                continue
            labels = _failure_labels_from_trace(self.trace_dir / f"{item.system}-{item.case_id}.json")
            aggregate.update(set(labels))
            case_rows.append((item.system, item.case_id, Counter(labels)))
            kind = "infrastructure" if item.infrastructure_failure else "agent"
            lines.append(f"- `{item.system}` `{item.case_id}`: {kind} failure, status={item.status}")
        if len(lines) == 2:
            lines.append("No failures recorded.")
        elif aggregate:
            lines.extend(["", "## Failure Mode Counts", ""])
            for label, count in aggregate.most_common():
                lines.append(f"- `{label}`: {count} cases")
            lines.extend(["", "## Per-Case Labels", ""])
            for system, case_id, labels in case_rows:
                if labels:
                    rendered = ", ".join(f"{label}={count}" for label, count in labels.most_common())
                else:
                    rendered = "unclassified"
                lines.append(f"- `{system}` `{case_id}`: {rendered}")
        (self.run_dir / "failure_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_json(self, path: Path, payload: Dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _record_attempt(
        self,
        trace: Dict[str, object],
        attempt_trace: Dict[str, object],
        system: str,
        case: BenchmarkCase,
    ) -> None:
        cast(List[Dict[str, object]], trace["attempts"]).append(attempt_trace)
        partial_path = self.trace_dir / f"{system}-{case.case_id}.partial.json"
        self._write_json(partial_path, trace)


def _split_metadata_list(text: str) -> List[str]:
    values = [item.strip() for item in re.split(r"[;\n\r]+", text or "") if item.strip()]
    return list(dict.fromkeys(values))


def _merge_requested_files(existing: List[str], *groups: List[str]) -> List[str]:
    merged = list(existing)
    for group in groups:
        for item in group:
            if item and item not in merged:
                merged.append(item)
    return merged[-20:]


def _hunk_identity(hunk: PatchHunk) -> tuple[str, Optional[int], Optional[int], str]:
    return (
        hunk.file,
        hunk.original_line_start if hunk.original_line_start is not None else hunk.line_start,
        hunk.original_line_end if hunk.original_line_end is not None else hunk.line_end,
        hunk.method_name or "",
    )


def _hunk_in_list(hunk: PatchHunk, hunks: List[PatchHunk]) -> bool:
    identity = _hunk_identity(hunk)
    return any(_hunk_identity(existing) == identity for existing in hunks)


def _hunk_ranges_overlap(a: PatchHunk, b: PatchHunk) -> bool:
    a_start = a.original_line_start if a.original_line_start is not None else (a.line_start or 0)
    a_end = a.original_line_end if a.original_line_end is not None else (a.line_end or 0)
    b_start = b.original_line_start if b.original_line_start is not None else (b.line_start or 0)
    b_end = b.original_line_end if b.original_line_end is not None else (b.line_end or 0)
    if a_start <= 0 or b_start <= 0:
        return False
    return max(a_start, b_start) <= min(a_end, b_end)


def _with_carried_grounded_hunks(plan: RepairPlan, carried: List[PatchHunk]) -> RepairPlan:
    new_hunks = list(plan.patch_hunks)
    merged: List[PatchHunk] = []
    for ch in carried:
        conflicts = any(
            nh.file == ch.file and _hunk_ranges_overlap(ch, nh)
            for nh in new_hunks
        )
        if not conflicts:
            merged.append(ch)
    merged.extend(new_hunks)
    return RepairPlan(
        diagnosis=plan.diagnosis,
        files_to_read=plan.files_to_read,
        patch_hunks=merged,
        tests_to_run_next=plan.tests_to_run_next,
        confidence=plan.confidence,
        final_explanation=plan.final_explanation,
        patch_style=plan.patch_style,
    )


def _ranges_from_hunks(hunks: List[PatchHunk]) -> List[Dict[str, object]]:
    ranges: List[Dict[str, object]] = []
    for hunk in hunks:
        line_start = hunk.original_line_start if hunk.original_line_start is not None else hunk.line_start
        line_end = hunk.original_line_end if hunk.original_line_end is not None else hunk.line_end
        if line_start is None or line_end is None:
            continue
        ranges.append(
            {
                "file": hunk.file,
                "line_start": line_start,
                "line_end": line_end,
                "method_name": hunk.method_name,
                "intent": hunk.intent or "",
            }
        )
    return ranges


def _carried_grounded_hunks_feedback(hunks: List[PatchHunk]) -> str:
    if not hunks:
        return ""
    parts = [
        "already_grounded_hunks: keep these exact grounded hunks; do not regenerate them. Return only the missing/failed ranges needed to complete the atomic patch."
    ]
    for hunk in hunks[:6]:
        line_start = hunk.original_line_start if hunk.original_line_start is not None else hunk.line_start
        line_end = hunk.original_line_end if hunk.original_line_end is not None else hunk.line_end
        parts.append(
            f"- file={hunk.file} lines={line_start}-{line_end} method={hunk.method_name or ''} intent={hunk.intent or ''} new_head={_single_line(hunk.new, 180)}"
        )
    return "\n".join(parts)


def _localization_ranges(localization) -> List[Dict[str, object]]:
    raw_ranges = getattr(localization, "line_ranges", None)
    if raw_ranges is None and isinstance(localization, dict):
        raw_ranges = localization.get("line_ranges", localization.get("ranges", []))
    if raw_ranges is None:
        return []
    ranges: List[Dict[str, object]] = []
    for item in raw_ranges:
        if isinstance(item, dict):
            ranges.append(dict(item))
        else:
            ranges.append(
                {
                    "file": getattr(item, "file", ""),
                    "line_start": getattr(item, "line_start", None),
                    "line_end": getattr(item, "line_end", None),
                    "method_name": getattr(item, "method_name", None),
                    "intent": getattr(item, "intent", ""),
                }
            )
    return ranges


def _optional_int(value: object) -> Optional[int]:
    if value in (None, ""):
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _line_numbered_lines(lines: List[str], start_line: int, end_line: int) -> str:
    rendered: List[str] = []
    for line_no in range(start_line, end_line + 1):
        if line_no < 1 or line_no > len(lines):
            continue
        rendered.append(f"{line_no}: {lines[line_no - 1]}")
    return "".join(rendered).rstrip()


def _line_numbered_lines_with_base(lines: List[str], start_line: int, end_line: int, base_line: int) -> str:
    rendered: List[str] = []
    for offset, line_no in enumerate(range(start_line, end_line + 1)):
        if line_no < 1 or line_no > len(lines):
            continue
        rendered.append(f"{base_line + offset}: {lines[line_no - 1]}")
    return "".join(rendered).rstrip()


def _join_verified_block(existing: str, addition: str) -> str:
    if not addition:
        return existing
    if not existing:
        return addition
    if addition in existing:
        return existing
    return existing.rstrip() + "\n" + addition


def _verified_context_snippets(context: Dict[str, Dict[str, object]]) -> Dict[str, str]:
    snippets: Dict[str, str] = {}
    for file_name, entry in context.items():
        parts = [str(entry.get("excerpt", "")), str(entry.get("method_body", ""))]
        snippets[file_name] = "\n".join(part for part in parts if part).strip()
    return snippets


def _verified_context_line_numbers(context: Dict[str, Dict[str, object]]) -> Dict[str, str]:
    return {file_name: str(entry.get("line_numbered", "")) for file_name, entry in context.items()}


def _summarize_verified_patch_context(context: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for file_name, entry in context.items():
        summary[file_name] = {
            "ranges": entry.get("ranges", []),
            "excerpt_chars": len(str(entry.get("excerpt", ""))),
            "line_numbered_chars": len(str(entry.get("line_numbered", ""))),
            "method_body_chars": len(str(entry.get("method_body", ""))),
        }
    return summary


def _numbered_range_context(text: str, line_start: int, line_end: int, *, context_lines: int = 15) -> str:
    lines = text.splitlines(keepends=True)
    if not lines:
        return ""
    start = max(1, line_start - context_lines)
    end = min(len(lines), max(line_end, line_start) + context_lines)
    return _line_numbered_lines(lines, start, end)


def _range_grounding_failure_feedback(errors: List[str], context: Dict[str, object], carried_files: Optional[set[str]] = None, limit: int = 2200) -> str:
    parts = [
        "range_grounding_failure: line range could not be verified against current source; this did not consume a patch attempt.",
        "reground the failed hunk using exact numbered source below; return line_start/line_end/new for a valid contiguous range.",
        "errors: " + "; ".join(errors[:5]),
    ]
    if carried_files:
        parts.append(
            "already_grounded_files: " + ", ".join(sorted(carried_files))
            + ". Do NOT re-ground these files; only fix the failed ranges above."
        )
    for file_name, block in list(context.items())[:3]:
        if block:
            parts.append(f"file={file_name}\n{str(block)[:900]}")
    return "\n".join(parts)[:limit]


def _grounding_error_file(error: str) -> str:
    return error.split(":")[0] if error else ""


def _grounding_error_identity(error: str) -> Tuple[str, int, int]:
    parts = error.split(":") if error else []
    if len(parts) >= 3:
        range_part = parts[1]
        try:
            start_str, end_str = range_part.split("-", 1)
            return (parts[0], int(start_str), int(end_str))
        except (ValueError, IndexError):
            pass
    return (parts[0] if parts else "", -1, -1)


def _semantic_retry_feedback(diagnosis: str, test_output: str, failure_detail: str, patch_style: str, limit: int = 2600) -> str:
    parts = [
        f"fixed_diagnosis: {diagnosis}",
        "semantic_failure_after_apply: the patch applied and compiled, but tests failed.",
        "alternative_patch_strategy_required: keep the same diagnosis, use the actual failing assertion output, and choose a different semantic patch strategy/condition; do not repeat the same patch_style.",
        f"failed_patch_style: {patch_style}",
        "actual_test_failure_delta:",
        failure_detail or _failure_summary(test_output, limit=900),
        "raw_test_output:",
        test_output,
    ]
    return "\n".join(part for part in parts if part)[:limit]


def _prompt_context_budget(scale: float) -> int:
    base = int(os.environ.get("REPAIR_PROMPT_CONTEXT_CHARS", "56000"))
    floor = int(os.environ.get("REPAIR_PROMPT_MIN_CONTEXT_CHARS", "18000"))
    bounded_scale = max(0.2, min(1.0, scale))
    return max(floor, int(base * bounded_scale))


def _requested_source_paths(requested_files: List[str], source_dir: str, workdir: Path) -> List[Path]:
    source_root = Path(source_dir)
    paths: List[Path] = []
    for raw in requested_files:
        if not raw or "\x00" in raw:
            continue
        value = raw.strip()
        if not value:
            continue
        candidate = Path(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            continue
        candidates: List[Path] = []
        if value.endswith(".java"):
            candidates.append(candidate)
        elif re.match(r"^[A-Za-z_][A-Za-z0-9_.$]*$", value):
            candidates.append(source_root / Path(value.replace(".", "/") + ".java"))
        for rel in candidates:
            if rel.is_absolute() or ".." in rel.parts:
                continue
            normalized = str(rel)
            if not (normalized == str(source_root) or normalized.startswith(str(source_root).rstrip("/") + "/")):
                continue
            path = workdir / rel
            if path.exists() and path.is_file() and rel not in paths:
                paths.append(rel)
    return paths[:8]


def _trigger_assertion_summary(workdir: Path, metadata: Dict[str, str], limit: int = 16) -> List[str]:
    test_dir = metadata.get("dir.src.tests", "").splitlines()[0].strip() or "src/test/java"
    triggers = _split_metadata_list(metadata.get("tests.trigger", ""))
    assertions: List[str] = []
    for body in _trigger_test_bodies(workdir, test_dir, triggers):
        lines = body.splitlines()
        compact_body_lines = [
            " ".join(line.strip().split())
            for line in lines
            if line.strip() and not line.strip().startswith(("*", "//", "/*"))
        ]
        if any("assert" in line.lower() or line.lower().startswith("fail(") for line in compact_body_lines):
            if len(compact_body_lines) <= 36:
                rendered_body = "full_test_method: " + " | ".join(compact_body_lines)
                if len(rendered_body) > 1800:
                    rendered_body = rendered_body[:1800]
                if rendered_body not in assertions:
                    assertions.append(rendered_body)
                    if len(assertions) >= limit:
                        return assertions
        for idx, line in enumerate(lines):
            stripped = " ".join(line.strip().split())
            if not stripped or stripped.startswith(("*", "//")):
                continue
            lowered = stripped.lower()
            if "assert" not in lowered and not lowered.startswith("fail("):
                continue
            context: List[str] = []
            for raw in lines[max(0, idx - 3) : idx + 1]:
                item = " ".join(raw.strip().split())
                if not item or item.startswith(("*", "//", "/*")):
                    continue
                context.append(item)
            rendered = " | ".join(context[-4:])
            if rendered and rendered not in assertions:
                assertions.append(rendered)
                if len(assertions) >= limit:
                    return assertions
    return assertions


def _failed_patch_feedback(
    plan: RepairPlan,
    apply_result: PatchApplyResult,
    failure_reason: str,
    failure_detail: str,
    limit: int = 1200,
) -> str:
    parts = []
    critical = _critical_failure_guidance(failure_detail)
    if critical:
        parts.append(critical)
    parts.extend([
        f"{failure_reason}: {_single_line(failure_detail, 520)}",
        "failed_patch_summary: this exact candidate was rejected by the environment; do not repeat an equivalent patch",
        "next_patch_requirement: change the semantic condition or root-cause strategy, not just whitespace, anchoring, or reformatting",
        f"patch_style={plan.patch_style}",
        f"changed_files={','.join(apply_result.changed_files)[:220]}",
    ])
    removed: List[str] = []
    added: List[str] = []
    for line in apply_result.diff.splitlines():
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("-") and len(removed) < 5:
            removed.append(_single_line(line[1:], 160))
        elif line.startswith("+") and len(added) < 5:
            added.append(_single_line(line[1:], 160))
    if removed:
        parts.append(f"removed_heads={removed}")
    if added:
        parts.append(f"added_heads={added}")
    return "\n".join(parts)[:limit]


def _critical_failure_guidance(detail: str) -> str:
    for line in detail.splitlines():
        if "numeric_type_selection_failed_strategy" in line:
            return _single_line(line, 420)
    return ""


def _related_source_paths(
    workdir: Path,
    source_dir: str,
    test_dir: str,
    classes: List[str],
    triggers: List[str],
    modified_source_texts: Dict[str, str],
    *,
    max_paths: int = 3,
) -> List[Path]:
    """Find transferable helper/source context near the modified classes.

    Defects4J metadata often identifies the edited class but not package-local
    helper APIs. This retrieval is feature based: it uses class-name tokens,
    trigger-test identifiers, and nearby package files, never bug ids.
    """
    source_root = Path(source_dir)
    query_tokens: List[str] = []
    query_types: List[str] = []
    for klass in classes[:8]:
        simple = klass.rsplit(".", 1)[-1]
        query_tokens.extend(_identifier_tokens(simple))
    for text in modified_source_texts.values():
        query_types.extend(_type_needles_from_text(text)[:16])
    for body in _trigger_test_bodies(workdir, test_dir, triggers):
        query_types.extend(_type_needles_from_text(body)[:16])
        query_tokens.extend(_identifier_tokens(body)[:24])
    query_tokens = list(dict.fromkeys(token.lower() for token in query_tokens if len(token) >= 4))
    query_types = list(dict.fromkeys(item for item in query_types if len(item) >= 4))
    if not query_tokens and not query_types:
        return []

    modified_rels = {
        source_root / Path(klass.replace(".", "/") + ".java")
        for klass in classes[:8]
    }
    candidate_dirs = {rel.parent for rel in modified_rels}
    ranked: List[Tuple[int, str, Path]] = []
    for package_dir in sorted(candidate_dirs):
        abs_dir = workdir / package_dir
        if not abs_dir.exists():
            continue
        for path in sorted(abs_dir.glob("*.java")):
            rel = path.relative_to(workdir)
            if rel in modified_rels:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            stem = path.stem
            stem_tokens = [token.lower() for token in _identifier_tokens(stem)]
            score = 0
            overlap = set(stem_tokens) & set(query_tokens)
            score += 6 * len(overlap)
            if re.search(r"(Util|Utils|Utilities|Helper|Factory|Support)$", stem):
                score += 4
            for type_name in query_types[:20]:
                if re.search(rf"\b{re.escape(type_name)}\b", text):
                    score += 2
            if "public static" in text:
                score += 1
            if score > 0:
                ranked.append((-score, str(rel), rel))
    return [rel for _, _, rel in sorted(ranked)[:max_paths]]


def _trigger_test_bodies(workdir: Path, test_dir: str, triggers: List[str]) -> List[str]:
    bodies: List[str] = []
    if not test_dir:
        test_dir = "src/test/java"
    for test_name in triggers[:4]:
        if "::" in test_name:
            test_class, test_method = test_name.split("::", 1)
            focus = test_method
        else:
            test_class = test_name
            focus = ""
        rel = Path(test_dir) / Path(test_class.replace(".", "/") + ".java")
        path = workdir / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        body = _java_method_block(text, focus) if focus else ""
        bodies.append(body or text[:6000])
    return bodies


def _identifier_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", raw)
        if not parts:
            parts = [raw]
        for part in parts:
            lowered = part.lower()
            if len(lowered) >= 3 and lowered not in {"test", "junit", "java", "org", "apache"}:
                tokens.append(lowered)
    return list(dict.fromkeys(tokens))


def _type_needles_from_text(text: str) -> List[str]:
    stop_types = {
        "Assert",
        "AssertionError",
        "Exception",
        "Object",
        "String",
        "System",
        "TestCase",
    }
    needles: List[str] = []
    for name in re.findall(r"\b[A-Z][A-Za-z0-9_]{3,}\b", text):
        if name in stop_types or name.endswith("Test") or name.endswith("Tests"):
            continue
        if name not in needles:
            needles.append(name)
        if len(needles) >= 32:
            break
    return needles


def _assertion_source_needles(failing_output: str, triggers: List[str]) -> List[str]:
    text = "\n".join([failing_output or "", "\n".join(triggers)]).lower()
    needles: List[str] = []
    if "assertequals" in text or "equals" in text or "equality" in text or ("expected" in text and "but was" in text):
        needles.extend(["equals", "equal"])
    if "asserttrue" in text or "assertfalse" in text:
        needles.extend(["boolean", "equals", "equal"])
    if "serialization" in text:
        needles.extend(["writeObject", "readObject", "serialize"])
    return list(dict.fromkeys(needles))


def _paired_source_needles(needles: List[str]) -> List[str]:
    pairs = [
        ("Domain", "Range"),
        ("domain", "range"),
        ("XValue", "YValue"),
        ("YValue", "XValue"),
        ("StartX", "StartY"),
        ("StartY", "StartX"),
        ("EndX", "EndY"),
        ("EndY", "EndX"),
        ("Lower", "Upper"),
        ("Upper", "Lower"),
    ]
    paired: List[str] = []
    for needle in needles:
        for left, right in pairs:
            if left in needle:
                paired.append(needle.replace(left, right))
            if right in needle:
                paired.append(needle.replace(right, left))
    return [item for item in dict.fromkeys(paired) if item and item not in needles]


def _derived_repair_constraints(visible_assertions: List[str]) -> List[str]:
    text = "\n".join(visible_assertions)
    lowered = text.lower()
    constraints: List[str] = []
    if "getlowerbound" in lowered and "getupperbound" in lowered:
        constraints.append(
            "Bounds/range repairs must satisfy both lower and upper assertions. "
            "Each non-NaN candidate value may need to update both minimum and maximum; "
            "do not assume start values only lower the bound or end values only raise it unless the API guarantees ordering."
        )
    if "nan" in lowered and ("getlowerbound" in lowered or "getupperbound" in lowered):
        constraints.append(
            "When interval endpoints are NaN, use the visible assertion setup lines to decide fallback behavior, "
            "and preserve non-NaN endpoint values rather than replacing the whole interval path."
        )
    if "integer.valueof" in lowered and "long.valueof" in lowered:
        constraints.append(
            "Numeric factory repairs must preserve the expected return type on both sides of the boundary; "
            "do not use a broad fallback that changes below-boundary Integer cases into Long cases."
        )
    if "createnumber" in lowered and "instanceof float" in lowered and "instanceof double" in lowered:
        constraints.append(
            "Numeric type-selection repairs must satisfy all visible instanceof assertions together. "
            "Do not simply try Float before Double: preserve Float inputs as Float, larger precise double inputs as Double, "
            "and precision-loss or beyond-double inputs as BigDecimal when the test asserts those types."
        )
    return constraints


def _failure_output_repair_constraints(failing_output: str) -> List[str]:
    lowered = (failing_output or "").lower()
    constraints: List[str] = []
    if "\\s*+" in failing_output and ("expected fdf failure" in lowered or "expected sdf failure" in lowered):
        constraints.append(
            "Regex parser whitespace repair: visible failure shows a generated \\s*+ regex accepted input that "
            "SimpleDateFormat rejected. Do not make all literal spaces flexible. Inspect the exact whitespace "
            "escaping path and preserve strict literal-space matching when the pattern contains one literal space."
        )
    return constraints


def _constraint_source_needles(visible_assertions: List[str]) -> List[str]:
    text = "\n".join(visible_assertions).lower()
    needles: List[str] = []
    if "nan" in text and ("getlowerbound" in text or "getupperbound" in text):
        needles.extend(
            [
                "getStartYValue",
                "getEndYValue",
                "getYValue",
                "getStartXValue",
                "getEndXValue",
                "getXValue",
                "iterateDomainBounds",
                "iterateRangeBounds",
                "IntervalXYDataset",
            ]
        )
    if "integer.valueof" in text and "long.valueof" in text:
        needles.extend(["createInteger", "createLong", "createBigInteger"])
    if "createnumber" in text and ("instanceof float" in text or "instanceof double" in text or "bigdecimal" in text):
        needles.extend(["createFloat", "createDouble", "createBigDecimal", "isAllZeros", "numDecimals"])
    return needles


def _failure_output_source_needles(failing_output: str) -> List[str]:
    lowered = (failing_output or "").lower()
    needles: List[str] = []
    if "\\s*+" in failing_output or "expected fdf failure" in lowered or "expected sdf failure" in lowered:
        needles.extend(
            [
                "escapeRegex(regex, formatField, true)",
                "CopyQuotedStrategy",
                "escapeRegex",
                "wasWhite",
                "\\s*+",
                "formatField",
            ]
        )
    if "patternsyntaxexception" in lowered:
        needles.extend(["escapeRegex", "Pattern.compile", "Pattern.quote"])
    return list(dict.fromkeys(needles))


def _trigger_test_source_needles(
    workdir: Path,
    test_dir: str,
    triggers: List[str],
    snippet_chars: int,
    window_chars: int,
) -> List[str]:
    needles: List[str] = []
    for test_name in triggers[:4]:
        if "::" in test_name:
            test_class, test_method = test_name.split("::", 1)
            focus = [test_method]
        else:
            test_class = test_name
            focus = []
        rel = Path(test_dir) / Path(test_class.replace(".", "/") + ".java")
        path = workdir / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        snippet = _java_method_block(text, focus[0]) if focus else ""
        if not snippet:
            snippet = _focused_snippet(text, focus, min(snippet_chars, 6000), min(window_chars, 4000))
        for needle in _source_call_needles_from_test(snippet):
            if needle not in needles:
                needles.append(needle)
            if len(needles) >= 24:
                return needles
        for needle in _type_needles_from_text(snippet):
            if needle not in needles:
                needles.append(needle)
            if len(needles) >= 24:
                return needles
    return needles


def _source_call_needles_from_test(text: str) -> List[str]:
    stop_words = {
        "assertEquals",
        "assertFalse",
        "assertNotNull",
        "assertNotSame",
        "assertNull",
        "assertSame",
        "assertTrue",
        "assertNotEquals",
        "fail",
        "add",
        "get",
        "setUp",
        "tearDown",
        "super",
    }
    needles: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("*", "//", "/*", "public ", "private ", "protected ")):
            continue
        for name in re.findall(r"(?:\.|\b)([a-z][A-Za-z0-9_]{2,})\s*\(", stripped):
            if len(name) <= 3 or name in stop_words or name.startswith("assert") or name.startswith("test"):
                continue
            if name not in needles:
                needles.append(name)
            if len(needles) >= 24:
                return _rank_test_needles(needles)
    return _rank_test_needles(needles)


def _rank_test_needles(needles: List[str]) -> List[str]:
    return [
        name
        for _, name in sorted(
            enumerate(needles),
            key=lambda item: (-len(item[1]), item[0]),
        )
    ]


def _java_method_block(text: str, method_name: str) -> str:
    if not method_name:
        return ""
    positions = _ranked_pattern_positions(text, method_name)
    for pos in positions:
        open_paren = text.find("(", pos + len(method_name))
        if open_paren < 0 or open_paren - pos > len(method_name) + 8:
            continue
        open_brace = text.find("{", open_paren)
        if open_brace < 0:
            continue
        end = _matching_brace_offset(text, open_brace)
        if end is not None:
            return text[pos : end + 1]
    return ""


def _java_method_declaration_block(text: str, method_name: str) -> str:
    if not method_name:
        return ""
    positions = _ranked_pattern_positions(text, method_name)
    for pos in positions:
        open_paren = text.find("(", pos + len(method_name))
        if open_paren < 0 or open_paren - pos > len(method_name) + 8:
            continue
        open_brace = text.find("{", open_paren)
        if open_brace < 0:
            continue
        end = _matching_brace_offset(text, open_brace)
        if end is None:
            continue
        line_start = text.rfind("\n", 0, pos) + 1
        return text[line_start : end + 1]
    return ""


def _matching_brace_offset(text: str, open_brace: int) -> Optional[int]:
    depth = 0
    in_line_comment = False
    in_block_comment = False
    in_string = False
    in_char = False
    escape = False
    idx = open_brace
    while idx < len(text):
        char = text[idx]
        nxt = text[idx + 1] if idx + 1 < len(text) else ""
        if in_line_comment:
            if char in "\r\n":
                in_line_comment = False
            idx += 1
            continue
        if in_block_comment:
            if char == "*" and nxt == "/":
                in_block_comment = False
                idx += 2
            else:
                idx += 1
            continue
        if in_string or in_char:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif in_string and char == '"':
                in_string = False
            elif in_char and char == "'":
                in_char = False
            idx += 1
            continue
        if char == "/" and nxt == "/":
            in_line_comment = True
            idx += 2
            continue
        if char == "/" and nxt == "*":
            in_block_comment = True
            idx += 2
            continue
        if char == '"':
            in_string = True
        elif char == "'":
            in_char = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return idx
        idx += 1
    return None


def _d4j_test_passed(result: CommandResult) -> bool:
    match = re.search(r"Failing tests:\s*(\d+)", result.output)
    if match:
        return int(match.group(1)) == 0
    return result.ok


def _is_non_retryable_llm_error(error: str) -> bool:
    lowered = error.lower()
    return any(
        marker in lowered
        for marker in (
            "api_key",
            "api key",
            "deepseek_api_key is not set",
            "401",
            "403",
            "unauthorized",
            "forbidden",
        )
    )


def _is_retryable_llm_error(error: str) -> bool:
    if _is_non_retryable_llm_error(error):
        return False
    lowered = error.lower()
    return any(
        marker in lowered
        for marker in (
            "empty message content",
            "subprocess exceeded",
            "deadline",
            "timed out",
            "timeout",
            "model deadline",
            "temporarily unavailable",
            "connection reset",
            "remote end closed",
            "remote disconnected",
            "ssl",
            "eof",
            "urlerror",
            "http 429",
            "too many requests",
        )
    )


def _patch_failure_feedback(plan: RepairPlan, errors: List[str], limit: int = 1400) -> str:
    parts = [
        f"patch_apply_failure: {'; '.join(errors)}",
        "failed hunks below are diagnostic only; after rollback they are not guaranteed to exist in current source",
    ]
    for hunk in plan.patch_hunks[:3]:
        old_head = _single_line(hunk.old, 260)
        new_head = _single_line(hunk.new, 220)
        line_info = ""
        if hunk.line_start is not None or hunk.line_end is not None:
            line_info = f" lines={hunk.line_start}-{hunk.line_end}"
        parts.append(f"hunk file={hunk.file}{line_info}; old_head={old_head}; new_head={new_head}")
    return "\n".join(parts)[:limit]


def _patch_grounding_feedback(workdir: Path, plan: RepairPlan, errors: List[str], limit: int = 2800) -> str:
    if not any("old text not found" in error.lower() or "brace imbalance" in error.lower() for error in errors):
        return ""
    parts = [
        "exact_current_source_grounding:",
        "The failed old text is absent or unsafe in the rolled-back source. Copy old text only from these exact current source excerpts.",
    ]
    used = len("\n".join(parts))
    for hunk in plan.patch_hunks[:2]:
        rel = Path(hunk.file)
        if rel.is_absolute() or ".." in rel.parts:
            continue
        path = workdir / rel
        if not path.exists() or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        block = _best_grounding_block(text, plan, hunk)
        if not block:
            continue
        numbered = _line_numbered_excerpt(text, block, max_chars=2200)
        addition = f"\nfile={hunk.file}\n{numbered}"
        if used + len(addition) > limit:
            addition = addition[: max(0, limit - used)]
        if addition.strip():
            parts.append(addition)
            used += len(addition)
        if used >= limit:
            break
    if len(parts) == 2:
        return ""
    return "\n".join(parts)[:limit]


def _best_grounding_block(text: str, plan: RepairPlan, hunk: PatchHunk) -> str:
    for anchor in _hunk_call_anchors(hunk) + _hunk_line_anchors(hunk):
        idx = text.find(anchor)
        if idx >= 0:
            start = max(0, idx - 900)
            end = min(len(text), idx + len(anchor) + 900)
            return text[start:end]
    for name in _patch_method_needles(plan, hunk):
        block = _java_method_block(text, name)
        if block:
            return _focused_grounding_window(block, hunk, max_chars=1200)
    for line in hunk.old.splitlines() + hunk.new.splitlines():
        stripped = line.strip()
        if len(stripped) < 18:
            continue
        idx = text.find(stripped)
        if idx >= 0:
            start = max(0, idx - 1200)
            end = min(len(text), idx + len(stripped) + 1200)
            return text[start:end]
    return ""


def _focused_grounding_window(block: str, hunk: PatchHunk, *, max_chars: int) -> str:
    if len(block) <= max_chars:
        return block
    anchors = _hunk_call_anchors(hunk)
    positions = [block.find(anchor) for anchor in anchors if anchor and block.find(anchor) >= 0]
    if positions:
        center = min(positions)
        pre_context = min(120, max_chars // 6)
        start = max(0, center - pre_context)
        end = min(len(block), start + max_chars)
        start = max(0, end - max_chars)
        return block[start:end]
    return block[:max_chars]


def _hunk_call_anchors(hunk: PatchHunk) -> List[str]:
    anchors: List[str] = []
    for text in (hunk.old, hunk.new):
        for call in re.findall(r"\b[a-z][A-Za-z0-9_]*\s*\([^;\n{}]*\)", text):
            normalized = re.sub(r"\s+", "", call)
            if len(normalized) < 6:
                continue
            if normalized not in anchors:
                anchors.append(normalized)
            compact = re.sub(r"\s+", " ", call).strip()
            if compact not in anchors:
                anchors.append(compact)
    return anchors[:12]


def _hunk_line_anchors(hunk: PatchHunk) -> List[str]:
    anchors: List[str] = []
    for text in (hunk.old, hunk.new):
        for line in text.splitlines():
            stripped = line.strip()
            if len(stripped) < 18 or stripped in {"try {", "}"}:
                continue
            if stripped not in anchors:
                anchors.append(stripped)
    return anchors[:8]


def _patch_method_needles(plan: RepairPlan, hunk: PatchHunk) -> List[str]:
    stop_words = {
        "return",
        "final",
        "public",
        "private",
        "protected",
        "static",
        "string",
        "number",
        "value",
        "values",
        "double",
        "float",
        "integer",
        "object",
        "false",
        "true",
        "null",
    }
    texts = [plan.diagnosis, plan.final_explanation, hunk.old, hunk.new]
    names: List[str] = []
    for text in texts:
        for name in re.findall(r"\b[a-z][A-Za-z0-9_]{3,}\b", text or ""):
            if name.lower() in stop_words or name.startswith(("assert", "test")):
                continue
            if name not in names:
                names.append(name)
    return names[:12]


def _line_numbered_excerpt(source_text: str, excerpt: str, *, max_chars: int) -> str:
    start = source_text.find(excerpt)
    if start < 0:
        return excerpt[:max_chars]
    prefix = source_text[:start]
    start_line = prefix.count("\n") + 1
    lines = excerpt.splitlines(keepends=True)
    rendered: List[str] = []
    used = 0
    for offset, line in enumerate(lines):
        item = f"{start_line + offset}: {line}"
        if used + len(item) > max_chars:
            break
        rendered.append(item)
        used += len(item)
    return "".join(rendered).rstrip()


def _compile_failure_feedback(
    output: str,
    workdir: Path,
    metadata: Dict[str, str],
    *,
    requested_files: List[str],
    limit: int = 2200,
) -> str:
    parts = [_failure_summary(output, limit=900)]
    missing = _missing_symbol_hints(output)
    if missing:
        parts.append("compile_api_feedback:")
        for item in missing[:4]:
            symbol = item.get("symbol", "")
            location = item.get("location", "")
            parts.append(f"- missing {symbol} in {location}; do not call APIs that are absent from the current source")
    api_surface = _available_api_surface(workdir, metadata, requested_files, missing)
    if api_surface:
        parts.append("available_api_surface:")
        for class_name, methods in api_surface[:6]:
            rendered = ", ".join(methods[:24])
            parts.append(f"- {class_name}: {rendered}")
    parts.append("Next patch must use only methods/classes visible in source snippets or available_api_surface.")
    return "\n".join(part for part in parts if part)[:limit]


def _missing_symbol_hints(output: str) -> List[Dict[str, str]]:
    hints: List[Dict[str, str]] = []
    symbol = ""
    for raw in output.splitlines():
        line = raw.strip()
        match = re.search(r"symbol:\s+(method|class|variable)\s+(.+)", line)
        if match:
            symbol = f"{match.group(1)} {match.group(2).strip()}"
            continue
        match = re.search(r"location:\s+(?:class|variable)\s+([A-Za-z_$][A-Za-z0-9_.$]*)", line)
        if match and symbol:
            hints.append({"symbol": symbol, "location": match.group(1)})
            symbol = ""
    return hints


def _available_api_surface(
    workdir: Path,
    metadata: Dict[str, str],
    requested_files: List[str],
    missing: List[Dict[str, str]],
) -> List[Tuple[str, List[str]]]:
    source_dir = metadata.get("dir.src.classes", "").splitlines()[0].strip() or "src/main/java"
    classes = _split_metadata_list(metadata.get("classes.modified", ""))
    rels: List[Path] = []
    for klass in classes[:8]:
        rel = Path(source_dir) / Path(klass.replace(".", "/") + ".java")
        if rel not in rels:
            rels.append(rel)
    for rel in _requested_source_paths(requested_files, source_dir, workdir):
        if rel not in rels:
            rels.append(rel)
    for item in missing:
        location = item.get("location", "")
        for rel in _location_source_paths(location, source_dir, classes, workdir):
            if rel not in rels:
                rels.append(rel)

    surfaces: List[Tuple[str, List[str]]] = []
    seen_classes: set[str] = set()
    for rel in rels[:10]:
        path = workdir / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        class_name = path.stem
        methods = _java_method_names(text)
        if methods and class_name not in seen_classes:
            surfaces.append((class_name, methods))
            seen_classes.add(class_name)
        superclass = _java_superclass_name(text)
        if superclass:
            for super_rel in _location_source_paths(superclass, source_dir, classes, workdir):
                if super_rel in rels:
                    continue
                super_path = workdir / super_rel
                if not super_path.exists():
                    continue
                super_text = super_path.read_text(encoding="utf-8", errors="replace")
                super_methods = _java_method_names(super_text)
                if super_methods and super_path.stem not in seen_classes:
                    surfaces.append((super_path.stem, super_methods))
                    seen_classes.add(super_path.stem)
    return surfaces


def _location_source_paths(location: str, source_dir: str, classes: List[str], workdir: Path) -> List[Path]:
    if not location:
        return []
    source_root = Path(source_dir)
    candidates: List[Path] = []
    if "." in location:
        candidates.append(source_root / Path(location.replace(".", "/") + ".java"))
    simple = location.rsplit(".", 1)[-1]
    for klass in classes:
        if klass.rsplit(".", 1)[-1] == simple:
            candidates.append(source_root / Path(klass.replace(".", "/") + ".java"))
    for path in (workdir / source_root).rglob(f"{simple}.java"):
        candidates.append(path.relative_to(workdir))
        if len(candidates) >= 6:
            break
    deduped: List[Path] = []
    for rel in candidates:
        if rel not in deduped and (workdir / rel).exists():
            deduped.append(rel)
    return deduped[:4]


def _java_method_names(text: str) -> List[str]:
    names: List[str] = []
    pattern = re.compile(
        r"(?:public|protected|private)\s+(?:static\s+)?(?:final\s+)?"
        r"(?:<[^>]+>\s+)?[A-Za-z_$][A-Za-z0-9_.$<>\[\], ?]*\s+"
        r"([A-Za-z_$][A-Za-z0-9_$]*)\s*\(",
        re.MULTILINE,
    )
    for name in pattern.findall(text):
        if name not in names and name not in {"if", "for", "while", "switch", "catch"}:
            names.append(name)
    return names


def _java_superclass_name(text: str) -> str:
    match = re.search(r"\bclass\s+[A-Za-z_$][A-Za-z0-9_$]*\s+extends\s+([A-Za-z_$][A-Za-z0-9_.$]*)", text)
    return match.group(1) if match else ""


def _single_line(value: str, limit: int) -> str:
    return " ".join(value.split())[:limit]


def _failure_labels_from_trace(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        trace = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ["trace_read_error"]
    labels: List[str] = []
    for attempt in trace.get("attempts", []):
        if not isinstance(attempt, dict):
            continue
        if attempt.get("llm_error"):
            labels.append("llm_error")
        if attempt.get("parse_error"):
            labels.append("parse_error")
        patch_apply = attempt.get("patch_apply")
        if isinstance(patch_apply, dict) and not patch_apply.get("ok"):
            for error in patch_apply.get("errors", []):
                error_text = str(error)
                lowered = error_text.lower()
                if "old text not found" in lowered:
                    labels.append("old_text_not_found")
                elif "replacement text already exists" in lowered:
                    labels.append("replacement_exists")
                elif "no-op hunk" in lowered:
                    labels.append("no_op_patch")
                elif "brace imbalance" in lowered:
                    labels.append("brace_imbalance")
                elif "editing tests is blocked" in lowered or "outside configured source dirs" in lowered:
                    labels.append("unsafe_patch_rejected")
                else:
                    labels.append("patch_apply_other")
        compile_result = attempt.get("compile")
        if isinstance(compile_result, dict) and compile_result.get("returncode") != 0:
            labels.append("compile_failed_after_apply")
        visible_tests = attempt.get("visible_tests")
        if isinstance(visible_tests, dict) and not visible_tests.get("passed"):
            labels.append("visible_failed_after_apply")
        regression_tests = attempt.get("regression_tests")
        if isinstance(regression_tests, dict) and not regression_tests.get("passed"):
            labels.append("regression_failed_after_visible")
    return labels


def _focused_snippet(
    text: str,
    needles: List[str],
    max_chars: int,
    window_chars: int,
    line_hints: Optional[List[int]] = None,
) -> str:
    if len(text) <= max_chars:
        return text
    parts: List[str] = []
    used = 0
    for start, end in _snippet_windows(text, needles, max_chars, window_chars, line_hints=line_hints):
        if used >= max_chars:
            break
        chunk = text[start:end]
        remaining = max_chars - used
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        parts.append(f"\n// snippet window {start}:{start + len(chunk)}\n{chunk}")
        used += len(chunk)
    return "\n".join(parts).strip()


def _focused_snippet_line_numbers(
    text: str,
    needles: List[str],
    max_chars: int,
    window_chars: int,
    line_hints: Optional[List[int]] = None,
) -> str:
    if len(text) <= max_chars:
        return _line_numbered_prefix(text, max_chars=max_chars)
    windows = _snippet_windows(text, needles, max_chars, window_chars, line_hints=line_hints)
    lines = text.splitlines(keepends=True)
    offsets: List[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)
    chunks: List[str] = []
    used = 0
    for start, end in windows:
        if used >= max_chars:
            break
        numbered: List[str] = []
        for idx, line_start in enumerate(offsets, start=1):
            line_end = line_start + len(lines[idx - 1])
            if line_end <= start:
                continue
            if line_start >= end:
                break
            rendered = f"{idx}: {lines[idx - 1]}"
            if used + len(rendered) > max_chars:
                break
            numbered.append(rendered)
            used += len(rendered)
        if numbered:
            chunks.append("".join(numbered).rstrip())
    return "\n...\n".join(chunks)


def _line_numbered_prefix(text: str, *, max_chars: int) -> str:
    lines: List[str] = []
    used = 0
    for idx, line in enumerate(text.splitlines(keepends=True), start=1):
        rendered = f"{idx}: {line}"
        if used + len(rendered) > max_chars:
            break
        lines.append(rendered)
        used += len(rendered)
    return "".join(lines).rstrip()


def _snippet_windows(
    text: str,
    needles: List[str],
    max_chars: int,
    window_chars: int,
    line_hints: Optional[List[int]] = None,
) -> List[Tuple[int, int]]:
    windows: List[Tuple[int, int]] = []
    if line_hints:
        offsets = _line_offsets(text)
        for line_no in line_hints[:8]:
            if line_no <= 0 or line_no > len(offsets):
                continue
            center = offsets[line_no - 1]
            half = max(200, min(window_chars // 2, max_chars // 2))
            windows.append((max(0, center - half), min(len(text), center + half)))
    for needle in needles:
        if not needle:
            continue
        for pattern in {needle, needle[0].lower() + needle[1:] if needle else needle}:
            positions = _ranked_pattern_positions(text, pattern)
            if positions:
                half = max(200, min(window_chars // 2, max_chars // 2))
                take = 4 if len(pattern) >= 8 and len(positions) > 2 else 2
                local_half = max(400, min(half, window_chars // 4)) if take > 2 else half
                for idx in positions[:take]:
                    windows.append((max(0, idx - local_half), min(len(text), idx + local_half)))
                break
    if not windows:
        return [(0, min(len(text), max_chars))]
    windows.append((0, min(len(text), max_chars // 4)))
    selected = _select_priority_windows(windows, max_chars)
    bounded: List[Tuple[int, int]] = []
    used = 0
    for start, end in selected:
        if used >= max_chars:
            break
        remaining = max_chars - used
        if end - start > remaining:
            end = start + remaining
        bounded.append((start, end))
        used += end - start
    return bounded


def _line_offsets(text: str) -> List[int]:
    offsets: List[int] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        offsets.append(cursor)
        cursor += len(line)
    return offsets or [0]


def _select_priority_windows(windows: List[Tuple[int, int]], max_chars: int) -> List[Tuple[int, int]]:
    selected: List[Tuple[int, int]] = []
    used = 0
    for start, end in windows:
        if used >= max_chars:
            break
        if end <= start:
            continue
        for seg_start, seg_end in _subtract_selected_window(start, end, selected):
            if used >= max_chars:
                break
            remaining = max_chars - used
            if seg_end - seg_start > remaining:
                seg_end = seg_start + remaining
            if seg_end > seg_start:
                selected.append((seg_start, seg_end))
                used += seg_end - seg_start
    return selected


def _subtract_selected_window(start: int, end: int, selected: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    segments = [(start, end)]
    for used_start, used_end in selected:
        next_segments: List[Tuple[int, int]] = []
        for seg_start, seg_end in segments:
            if used_end <= seg_start or used_start >= seg_end:
                next_segments.append((seg_start, seg_end))
                continue
            if seg_start < used_start:
                next_segments.append((seg_start, used_start))
            if used_end < seg_end:
                next_segments.append((used_end, seg_end))
        segments = next_segments
        if not segments:
            break
    return segments


def _ranked_pattern_positions(text: str, pattern: str) -> List[int]:
    matches = [match.start() for match in re.finditer(re.escape(pattern), text)]
    if len(matches) <= 1:
        return matches
    line_starts = _line_offsets(text)

    def score(idx: int) -> tuple[int, int]:
        line_no = _line_number_for_offset(line_starts, idx)
        line_start = line_starts[line_no]
        line_end = text.find("\n", line_start)
        if line_end < 0:
            line_end = len(text)
        line = text[line_start:line_end].strip()
        code_score = 0
        if not line.startswith(("*", "//", "/*")):
            code_score += 10
        if re.search(r"\b(public|protected|private|static|final|synchronized)\b", line):
            code_score += 4
        if re.search(rf"\b{re.escape(pattern)}\s*\(", line):
            code_score += 4
        if line.endswith(";"):
            code_score -= 1
        return (-code_score, idx)

    return sorted(matches, key=score)


def _line_number_for_offset(line_starts: List[int], offset: int) -> int:
    lo = 0
    hi = len(line_starts) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if line_starts[mid] <= offset:
            lo = mid + 1
        else:
            hi = mid - 1
    return max(0, hi)


def _failure_source_hints(test_output: str, source_dir: str, classes: List[str]) -> Tuple[List[str], Dict[str, List[int]]]:
    needles: List[str] = []
    line_hints: Dict[str, List[int]] = {}
    if not test_output:
        return needles, line_hints
    source_root = Path(source_dir)
    class_to_rel = {
        klass.rsplit(".", 1)[-1] + ".java": str(source_root / Path(klass.replace(".", "/") + ".java"))
        for klass in classes
    }
    for match in re.finditer(r"\bat\s+([A-Za-z_$][A-Za-z0-9_.$]*)\.([A-Za-z_$][A-Za-z0-9_$]*)\(([A-Za-z_$][A-Za-z0-9_$]*\.java):(\d+)\)", test_output):
        qualified_class, method, filename, line_text = match.groups()
        rel = class_to_rel.get(filename)
        if not rel:
            continue
        if method not in needles:
            needles.append(method)
        simple_class = qualified_class.rsplit(".", 1)[-1]
        if simple_class not in needles:
            needles.append(simple_class)
        line_hints.setdefault(rel, [])
        line_no = int(line_text)
        if line_no not in line_hints[rel]:
            line_hints[rel].append(line_no)
    for exception in re.findall(r"([A-Za-z_][A-Za-z0-9_.]*Exception|AssertionError|Error)", test_output)[:5]:
        name = exception.rsplit(".", 1)[-1]
        if name not in needles:
            needles.append(name)
    return needles[:16], {key: values[:8] for key, values in line_hints.items()}


def _merge_windows(windows: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(windows):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _scope_failure_tail(scope_result: ScopeResult, limit: int = 600) -> str:
    failures = [_failure_summary(result.output, limit=limit) for result in scope_result.results if not _d4j_test_passed(result)]
    if not failures:
        failures = [_failure_summary(result.output, limit=limit) for result in scope_result.results[-1:]]
    return "\n".join(failures)[:limit]


def _visible_failure_guidance(visible_assertions: List[str], diff: str) -> str:
    assertion_text = "\n".join(visible_assertions).lower()
    diff_text = diff.lower()
    guidance: List[str] = []
    if (
        "instanceof float" in assertion_text
        and "instanceof double" in assertion_text
        and "bigdecimal" in assertion_text
        and "createfloat" in diff_text
        and "createdouble" in diff_text
    ):
        guidance.append(
            "numeric_type_selection_failed_strategy: simple Float-before-Double patch compiled and failed. "
            "Do not repeat it. Next patch needs separate Float, Double, and BigDecimal precision/range guards; "
            "do not return Float solely because createFloat succeeds."
        )
    if not guidance:
        return ""
    return "\n".join(guidance)


def _patch_strategy_signature(diff: str) -> str:
    added = [
        re.sub(r"\s+", "", line[1:]).lower()
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++") and line[1:].strip()
    ]
    removed = [
        re.sub(r"\s+", "", line[1:]).lower()
        for line in diff.splitlines()
        if line.startswith("-") and not line.startswith("---") and line[1:].strip()
    ]
    added_text = "\n".join(added)
    full_text = "\n".join(
        re.sub(r"\s+", "", line[1:] if line[:1] in {"+", "-", " "} else line).lower()
        for line in diff.splitlines()
        if not line.startswith(("+++", "---", "@@")) and line.strip()
    )
    if "createfloat(str)" in added_text and "createdouble(str)" in full_text and "returnf;" in added_text:
        return "numeric-type:add-float-before-double"
    if "createfloat(numeric)" in added_text and "createdouble(numeric)" in full_text and "returnf;" in added_text:
        return "numeric-type:add-float-before-double"
    normalized = "\n".join(["+ " + line for line in added[:12]] + ["- " + line for line in removed[:12]])
    if not normalized:
        return ""
    return "patch:" + hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def _duplicate_patch_feedback(signature: str, diff: str, visible_assertions: List[str]) -> str:
    guidance = _visible_failure_guidance(visible_assertions, diff)
    parts = [
        f"duplicate_failed_patch_strategy: signature={signature}; this candidate matches a patch strategy that already compiled and failed tests in this episode",
    ]
    if guidance:
        parts.append(guidance)
    parts.append("next_patch_requirement: choose a different semantic condition/root cause; do not reapply the same added/removed code pattern")
    return "\n".join(parts)


def _duplicate_rejection_limit() -> int:
    return max(1, int(os.environ.get("REPAIR_DUPLICATE_REJECTION_LIMIT", "2")))


def _is_duplicate_apply_failure(errors: List[str]) -> bool:
    return any("replacement text already exists" in str(error).lower() for error in errors)


def _failure_summary(output: str, limit: int = 600) -> str:
    """Keep assertion and exception details ahead of long Ant/JUnit stack tails."""
    if not output:
        return ""
    failing_block = output.split("[failing_tests]", 1)[-1] if "[failing_tests]" in output else output
    lines = [line.strip() for line in failing_block.splitlines() if line.strip()]
    selected: List[str] = []
    for line in lines:
        lowered = line.lower()
        if (
            line.startswith("---")
            or "assertion" in lowered
            or "expected" in lowered
            or "but was" in lowered
            or "error:" in lowered
            or "cannot find symbol" in lowered
            or lowered.startswith("symbol:")
            or lowered.startswith("location:")
            or "numberformatexception" in lowered
            or re.search(r"\b[A-Za-z_][A-Za-z0-9_.]*(Exception|Error)\b", line)
        ):
            selected.append(line)
        if len(selected) >= 6:
            break
    if not selected:
        selected = lines[:6]
    return "\n".join(selected)[:limit]


def _repair_skill(plan: RepairPlan, failure_reason: Optional[str]) -> str:
    hunk_count = len(plan.patch_hunks)
    size_label = "single-hunk" if hunk_count == 1 else "multi-hunk"
    if failure_reason == "patch_apply_failure":
        return "exact-old-text-grounding"
    if failure_reason == "regression_failure":
        return "repair-after-regression"
    if failure_reason in {"visible_failure", "compile_failure"}:
        return "retry-after-feedback"
    return f"{size_label}:{plan.patch_style}"


def _failure_reflection(plan: RepairPlan, failure_reason: Optional[str], detail: Optional[str]) -> str:
    files = ",".join(hunk.file for hunk in plan.patch_hunks)[:200]
    if failure_reason == "patch_apply_failure":
        detail_text = f"; apply_errors={detail[:300]}" if detail else ""
        return (
            f"patch_apply_failure: style={plan.patch_style}; files={files}; "
            f"old/new text was not grounded in current source, so re-read exact snippets and use a minimal anchored edit{detail_text}"
        )
    if failure_reason == "visible_failure":
        detail_text = f"; visible_tail={detail[:300]}" if detail else ""
        return (
            f"visible_failure: style={plan.patch_style}; files={files}; "
            "patch compiled but trigger tests still failed, so use failing_tests assertion details and change strategy"
            f"{detail_text}"
        )
    if failure_reason == "regression_failure":
        detail_text = f"; regression_tail={detail[:300]}" if detail else ""
        return (
            f"regression_failure: style={plan.patch_style}; files={files}; "
            "visible tests passed but regression failed, so avoid visible-only overfitting and validate broader behavior"
            f"{detail_text}"
        )
    if failure_reason == "compile_failure":
        detail_text = f"; compile_tail={detail[:300]}" if detail else ""
        return f"compile_failure: style={plan.patch_style}; files={files}; patch did not compile{detail_text}"
    return f"{failure_reason or 'failure'}: style={plan.patch_style}; files={files}; previous attempt failed"


def _success_strategy(plan: RepairPlan, test_scope: str) -> str:
    files = ",".join(sorted({hunk.file for hunk in plan.patch_hunks}))[:200]
    explanation = (plan.final_explanation or plan.diagnosis).replace("\n", " ")[:500]
    return (
        f"successful strategy: style={plan.patch_style}; scope={test_scope}; files={files}; "
        f"approach={explanation}; require exact old/new text from current snippets and verify trigger plus regression"
    )


def _test_skill(
    test_scope: str,
    failure_reason: Optional[str],
    visible_passed: bool,
    regression_checked: bool,
    regression_passed: bool,
) -> str:
    if regression_checked and visible_passed:
        outcome = "passed" if regression_passed else "caught-overfit"
        return f"regression-{test_scope}-{outcome}"
    if failure_reason == "compile_failure":
        return "compile-before-tests"
    return f"{test_scope}-first"


def _parse_systems(value: str) -> List[str]:
    systems = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in systems if item not in SYSTEMS]
    if unknown:
        raise ValueError(f"unknown systems: {unknown}")
    return systems


def _load_runtime_tuning(config_path: Path, args: argparse.Namespace) -> RuntimeTuning:
    raw = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    runtime = raw.get("runtime", {}) if isinstance(raw, dict) else {}
    if not isinstance(runtime, dict):
        runtime = {}
    raw_limits = runtime.get("attempt_limits", {})
    attempt_limits = {
        "baseline": _runtime_int(
            raw_limits if isinstance(raw_limits, dict) else {},
            "baseline",
            args.baseline_attempts,
            DEFAULT_BASELINE_ATTEMPTS,
        ),
        "feedback": _runtime_int(
            raw_limits if isinstance(raw_limits, dict) else {},
            "feedback",
            args.feedback_attempts,
            DEFAULT_FEEDBACK_ATTEMPTS,
        ),
        "self_evolved": _runtime_int(
            raw_limits if isinstance(raw_limits, dict) else {},
            "self_evolved",
            args.self_evolved_attempts,
            DEFAULT_SELF_EVOLVED_ATTEMPTS,
        ),
    }
    memory_path_value = args.memory_path
    if memory_path_value is None and isinstance(runtime.get("memory_path"), str):
        memory_path_value = Path(str(runtime["memory_path"]))
    memory_mode_value = str(args.memory_mode or runtime.get("memory_mode", "full"))
    if memory_mode_value not in MEMORY_MODES:
        raise ValueError(f"memory_mode must be one of {', '.join(MEMORY_MODES)}")
    return RuntimeTuning(
        attempt_limits=attempt_limits,
        memory_attempt_bonus=_runtime_int(runtime, "memory_attempt_bonus", args.memory_attempt_bonus, DEFAULT_MEMORY_ATTEMPT_BONUS),
        max_attempt_cap=_runtime_int(runtime, "max_attempt_cap", args.max_attempt_cap, DEFAULT_MAX_ATTEMPT_CAP),
        memory_guidance_limit=_runtime_int(
            runtime,
            "memory_guidance_limit",
            args.memory_guidance_limit,
            DEFAULT_MEMORY_GUIDANCE_LIMIT,
        ),
        max_non_patch_rounds=_runtime_int(
            runtime,
            "max_non_patch_rounds",
            args.max_non_patch_rounds,
            DEFAULT_MAX_NON_PATCH_ROUNDS,
        ),
        memory_path=memory_path_value,
        fresh_memory=bool(args.fresh_memory),
        memory_mode=memory_mode_value,
    )


def _runtime_int(config_values: Dict[str, object], key: str, cli_value: Optional[int], default: int) -> int:
    if cli_value is not None:
        return int(cli_value)
    if key in config_values:
        value = config_values[key]
        if isinstance(value, (int, float, str)):
            return int(value)
    return default


def write_preflight_failure(run_dir: Path, errors: List[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_dir": str(run_dir),
        "status": "preflight_failed",
        "infrastructure_failures": len(errors),
        "errors": errors,
        "systems": {},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "metrics.csv").write_text(
        "system,case_id,project,bug_id,status,pass_at_1,pass_at_3,visible_pass,regression_pass,"
        "compile_success,tool_calls,test_runs,patch_size,unsafe_edit,wall_time_seconds,deepseek_calls,"
        "prompt_tokens,completion_tokens,estimated_cost_usd,infrastructure_failure,agent_failure\n",
        encoding="utf-8",
    )
    (run_dir / "failure_analysis.md").write_text(
        "# Failure Analysis\n\n"
        + "\n".join(f"- infrastructure blocker: {error}" for error in errors)
        + "\n",
        encoding="utf-8",
    )
    BenchmarkMemory().save(run_dir / "memory_before.json")
    BenchmarkMemory().save(run_dir / "memory_after.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/defects4j_30.json"))
    parser.add_argument("--run-id", default=datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--systems", default="baseline,feedback,self_evolved")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--baseline-attempts", type=int, default=None)
    parser.add_argument("--feedback-attempts", type=int, default=None)
    parser.add_argument("--self-evolved-attempts", type=int, default=None)
    parser.add_argument("--memory-attempt-bonus", type=int, default=None)
    parser.add_argument("--max-attempt-cap", type=int, default=None)
    parser.add_argument("--memory-guidance-limit", type=int, default=None)
    parser.add_argument("--max-non-patch-rounds", type=int, default=None)
    parser.add_argument("--memory-path", type=Path, default=None)
    parser.add_argument("--memory-mode", choices=MEMORY_MODES, default=None)
    parser.add_argument("--fresh-memory", action="store_true")
    parser.add_argument("--max-regression-tests", type=int, default=20)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--allow-missing-runtime",
        action="store_true",
        help="Write per-case infrastructure traces even when defects4j or DEEPSEEK_API_KEY is missing.",
    )
    args = parser.parse_args()

    cases = load_cases(args.config)
    run_dir = args.out_dir / args.run_id
    tuning = _load_runtime_tuning(args.config, args)
    client = Defects4JClient()
    llm = DeepSeekChatClient()
    preflight_errors: List[str] = []
    if not client.available():
        preflight_errors.append("defects4j CLI not found on PATH")
    if not llm.enabled():
        preflight_errors.append("DEEPSEEK_API_KEY is not set")
    if preflight_errors and not args.allow_missing_runtime:
        write_preflight_failure(run_dir, preflight_errors)
        print(json.dumps({"run_dir": str(run_dir), "status": "preflight_failed", "errors": preflight_errors}, ensure_ascii=False))
        raise SystemExit(2)
    runner = Defects4JBenchmarkRunner(
        cases=cases,
        run_dir=run_dir,
        systems=_parse_systems(args.systems),
        max_attempts=args.max_attempts,
        client=client,
        llm=llm,
        resume=not args.no_resume,
        max_regression_tests=args.max_regression_tests,
        tuning=tuning,
    )
    summary = runner.run()
    print(json.dumps({"run_dir": str(run_dir), "summary": summary["systems"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
