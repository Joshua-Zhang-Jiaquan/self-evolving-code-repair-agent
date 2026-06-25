"""Training loop for trajectory-level self-evolution."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .agent import EpisodeResult, PolicyMemory, SelfEvolvingRepairAgent
from .environment import CodeRepairEnvironment
from .tasks import RepairTask


def run_agent_on_tasks(
    tasks: Iterable[RepairTask],
    memory: PolicyMemory,
    learn: bool,
    *,
    use_test_feedback: bool = True,
    max_patch_attempts: Optional[int] = None,
    max_strategy_scans: Optional[int] = None,
) -> List[EpisodeResult]:
    agent = SelfEvolvingRepairAgent(
        memory=memory,
        use_test_feedback=use_test_feedback,
        max_patch_attempts=max_patch_attempts,
        max_strategy_scans=max_strategy_scans,
    )
    results: List[EpisodeResult] = []
    for task in tasks:
        with CodeRepairEnvironment(task) as env:
            results.append(agent.run_episode(env, learn=learn))
    return results


def train_policy_memory(
    tasks: Iterable[RepairTask],
    episodes: int,
    memory_path: Path,
) -> PolicyMemory:
    memory = PolicyMemory()
    for _ in range(episodes):
        run_agent_on_tasks(tasks, memory=memory, learn=True)
    memory.save(memory_path)
    return memory


def summarize(results: List[EpisodeResult]) -> Dict[str, float]:
    if not results:
        return {
            "tasks": 0,
            "visible_solve_rate": 0.0,
            "hidden_solve_rate": 0.0,
            "avg_reward": 0.0,
            "avg_tool_calls": 0.0,
            "avg_test_runs": 0.0,
            "pass_at_1": 0.0,
            "pass_at_k": 0.0,
            "avg_patch_size": 0.0,
            "unsafe_edit_rate": 0.0,
        }
    total = len(results)
    return {
        "tasks": total,
        "visible_solve_rate": round(sum(r.solved_visible for r in results) / total, 4),
        "hidden_solve_rate": round(sum(r.solved_hidden for r in results) / total, 4),
        "avg_reward": round(sum(r.reward.reward for r in results) / total, 4),
        "avg_tool_calls": round(sum(r.tool_calls for r in results) / total, 4),
        "avg_test_runs": round(sum(r.test_runs for r in results) / total, 4),
        "pass_at_1": round(sum(r.pass_at_1 for r in results) / total, 4),
        "pass_at_k": round(sum(r.solved_hidden for r in results) / total, 4),
        "avg_patch_size": round(sum(r.patch_size for r in results) / total, 4),
        "unsafe_edit_rate": round(sum(bool(r.reward.unsafe_edit) for r in results) / total, 4),
    }
