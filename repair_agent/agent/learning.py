from __future__ import annotations
# pyright: reportImplicitOverride=false, reportPrivateUsage=false, reportUnannotatedClassAttribute=false

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from repair_agent.agent.baseline import (
    PATCH_SUCCESS_STATUSES,
    TOOL_FAILURE_STATUSES,
    BaselineAgent,
    _Counters,
    _RegistryLike,
    _addition_edit_from_read,
    _candidate_from_search,
    _final_explanation,
    _final_status,
    _query_from_problem,
)
from repair_agent.agent.interface import AgentFinalAnswer, AgentResult, AgentTask, TrajectoryStepRecord
from repair_agent.config import ConfigMap
from repair_agent.tools.core import MALFORMED, OK, ToolRegistry, ToolResult, TaskWorkspace
from repair_agent.training.policy import (
    ACTION_VOCABULARY,
    FeatureExtractor,
    LearningContext,
    LinearSoftmaxPolicy,
    PolicyTransition,
)
from repair_agent.training.pomdp import RewardBreakdown, RewardSignal, WeightedReward, default_reward


LEARNING_AGENT_VERSION = "learning-reinforce-baseline-v1"


@dataclass(frozen=True)
class LearningStep:
    transition: PolicyTransition
    reward: RewardBreakdown
    tool: str
    status: str
    step_index: int


@dataclass(frozen=True)
class LearningEpisode:
    result: AgentResult
    steps: Sequence[LearningStep]


class LearningAgent(BaselineAgent):
    agent_version: str = LEARNING_AGENT_VERSION

    def __init__(
        self,
        policy: LinearSoftmaxPolicy | None = None,
        reward: WeightedReward | None = None,
        registry: _RegistryLike | ToolRegistry | None = None,
        *,
        deterministic: bool = True,
        visible_gpu_count: int = 0,
        rollout_parallelism: int = 1,
        model_gate_status: str = "unknown",
    ) -> None:
        super().__init__(registry=registry)
        self.policy = policy or LinearSoftmaxPolicy()
        self.reward = reward or default_reward()
        self.extractor = FeatureExtractor()
        self.deterministic = deterministic
        self.visible_gpu_count = visible_gpu_count
        self.rollout_parallelism = rollout_parallelism
        self.model_gate_status = model_gate_status

    def run(self, task: AgentTask, run_id: str) -> AgentResult:
        return self.run_episode(task, run_id).result

    def run_episode(self, task: AgentTask, run_id: str) -> LearningEpisode:
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
        learning_steps: list[LearningStep] = []
        candidate_path = ""
        last_read: ToolResult | None = None
        edited = False
        test_status = "not_run"
        diff_status = "not_run"
        patch = ""
        relevant_file_score = 0.0
        last_action = "none"
        repeated_action_count = 0

        while len(steps) < task.max_steps:
            context = LearningContext(
                step_index=len(steps),
                max_steps=task.max_steps,
                test_run_count=counters.test_runs,
                max_test_runs=task.max_test_runs,
                last_action_type=last_action,
                last_test_status=test_status,
                relevant_file_score=relevant_file_score,
                patch_exists=bool(patch or edited),
                repeated_action_count=repeated_action_count,
                model_gate_status=self.model_gate_status,
                tool_call_count=counters.tool_calls,
                visible_gpu_count=self.visible_gpu_count,
                rollout_parallelism=self.rollout_parallelism,
            )
            features = self.extractor.extract(context)
            available = _available_actions(
                step_index=len(steps),
                task=task,
                candidate_path=candidate_path,
                last_read=last_read,
                edited=edited,
                test_status=test_status,
                diff_status=diff_status,
            )
            action = self.policy.select_action(features, available_actions=available, deterministic=self.deterministic)
            args = _action_args(action, task, candidate_path, last_read, test_status, diff_status)
            result = self.registry.execute(action, workspace, args)
            counters.tool_calls += 1
            if action == "run_tests" and result.status not in {MALFORMED}:
                counters.test_runs = workspace.test_run_count
                test_status = result.status
            if action == "edit_file" and result.status == OK:
                counters.edits += 1
                edited = True
            if action == "search" and result.status == OK:
                candidate_path = _candidate_from_search(result.output) or candidate_path
                relevant_file_score = 1.0 if candidate_path else relevant_file_score
            if action == "read_file" and result.status == OK and str(args.get("path")) != ".":
                last_read = result
            if action == "rollback" and result.status == OK:
                edited = False
                patch = ""
            if action == "git_diff":
                diff_status = result.status
                patch = result.output if result.status == OK else ""

            breakdown = self.reward.score(_reward_signal(action, result, edited=edited, patch=patch, test_status=test_status, candidate_path=candidate_path))
            metadata: ConfigMap = {
                "learning_policy": "reinforce_baseline",
                "policy_action_vocabulary": list(ACTION_VOCABULARY),
                "reward_components": dict(breakdown.components),
                "reward_total": breakdown.total,
                "weighted_reward_components": dict(breakdown.weighted_components),
            }
            steps.append(self._tool_step(task, run_id, len(steps), action, args, result, counters, metadata=metadata))
            learning_steps.append(
                LearningStep(
                    transition=PolicyTransition(features=features, action=action, reward=breakdown.total),
                    reward=breakdown,
                    tool=action,
                    status=result.status,
                    step_index=len(steps) - 1,
                )
            )
            repeated_action_count = repeated_action_count + 1 if action == last_action else 0
            last_action = action
            if action == "final_answer":
                break

        if patch and test_status == OK:
            final_status = "passed"
        else:
            final_status = _final_status(edited=edited, test_status=test_status, patch=patch, diff_status=diff_status)
        explanation = _final_explanation(final_status=final_status, test_status=test_status, edited=edited, diff_status=diff_status).replace("Baseline", "Learning")
        steps = [self._with_final_status(step, final_status) for step in steps]
        final = AgentFinalAnswer(
            instance_id=task.instance_id,
            model_name_or_path=task.model_name_or_path,
            model_patch=patch,
            status=final_status,
            explanation=explanation,
            metadata={
                "diff_status": diff_status,
                "edited": edited,
                "learning_updates": 0,
                "policy": "reinforce_baseline",
                "test_status": test_status,
            },
        )
        return LearningEpisode(
            result=AgentResult(
                final=final,
                trajectory=steps,
                metrics={
                    "edit_count": counters.edits,
                    "episode_return": sum(step.reward.total for step in learning_steps),
                    "final_status": final_status,
                    "learning_updates": 0,
                    "test_run_count": counters.test_runs,
                    "tool_call_count": counters.tool_calls,
                },
            ),
            steps=learning_steps,
        )


def _available_actions(
    *,
    step_index: int,
    task: AgentTask,
    candidate_path: str,
    last_read: ToolResult | None,
    edited: bool,
    test_status: str,
    diff_status: str,
) -> tuple[str, ...]:
    if step_index == 0:
        return ("search",)
    if not candidate_path:
        return ("read_file", "search")
    if last_read is None:
        if task.visible_failures or task.visible_tests:
            return ("inspect_test", "read_file")
        return ("read_file",)
    if not edited:
        return ("edit_file", "read_file", "search")
    if test_status == "not_run" and task.max_test_runs > 0:
        return ("run_tests",)
    if edited and test_status in TOOL_FAILURE_STATUSES:
        return ("rollback", "git_diff", "final_answer")
    if diff_status == "not_run":
        return ("git_diff",)
    return ("final_answer",)


def _action_args(action: str, task: AgentTask, candidate_path: str, last_read: ToolResult | None, test_status: str, diff_status: str) -> dict[str, object]:
    if action == "search":
        return {"query": _query_from_problem(task.problem_statement), "path": ".", "max_matches": 50}
    if action == "read_file":
        return {"path": candidate_path or ".", "max_lines": 300 if candidate_path else 200}
    if action == "inspect_test":
        target = next(iter(task.visible_failures.keys()), task.visible_tests[0] if task.visible_tests else "visible-tests")
        return {"target": target}
    if action == "edit_file":
        edit_args = _addition_edit_from_read(candidate_path, last_read.output if last_read is not None else "", task.problem_statement)
        return edit_args or {"path": candidate_path, "replacement": "", "start_line": 1, "end_line": 1}
    if action == "run_tests":
        return {"target": task.visible_tests[0] if task.visible_tests else "", "timeout_seconds": task.test_timeout_seconds}
    if action == "rollback":
        return {"reason": f"learning policy rollback after {test_status}"}
    if action == "final_answer":
        return {"answer": f"Learning policy finished with test_status={test_status}, diff_status={diff_status}."}
    return {}


def _reward_signal(action: str, result: ToolResult, *, edited: bool, patch: str, test_status: str, candidate_path: str) -> RewardSignal:
    return RewardSignal(
        pass_result=1.0 if action == "final_answer" and patch and test_status in PATCH_SUCCESS_STATUSES else 0.0,
        visible_test_pass=1.0 if action == "run_tests" and result.status in PATCH_SUCCESS_STATUSES else 0.0,
        visible_test_failure=1.0 if action == "run_tests" and result.status not in PATCH_SUCCESS_STATUSES else 0.0,
        hidden_regression_ready=1.0 if action == "git_diff" and patch and test_status in PATCH_SUCCESS_STATUSES else 0.0,
        partial_progress=1.0 if action == "edit_file" and result.status == OK and edited else 0.0,
        relevant_file=1.0 if action == "search" and candidate_path else 0.0,
        tool_calls=1.0,
        test_runs=1.0 if action == "run_tests" else 0.0,
        unsafe_edit=1.0 if action == "edit_file" and result.status != OK else 0.0,
        timeout=1.0 if result.status == "timeout" else 0.0,
    )


def policy_hash(policy: LinearSoftmaxPolicy) -> str:
    payload = repr(policy.to_checkpoint()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
