"""Command-line evaluation protocol for the repair agent."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .agent import PolicyMemory
from .evolution import run_agent_on_tasks, summarize, train_policy_memory
from .tasks import eval_tasks, training_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episodes", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("artifacts/eval.json"))
    parser.add_argument("--memory", type=Path, default=Path("artifacts/policy_memory.json"))
    args = parser.parse_args()

    started = time.perf_counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    baseline_results = run_agent_on_tasks(
        eval_tasks(),
        PolicyMemory(),
        learn=False,
        use_test_feedback=False,
        max_patch_attempts=1,
        max_strategy_scans=1,
    )
    feedback_results = run_agent_on_tasks(
        eval_tasks(),
        PolicyMemory(),
        learn=False,
        use_test_feedback=True,
        max_patch_attempts=None,
        max_strategy_scans=None,
    )
    evolved_memory = train_policy_memory(training_tasks(), args.train_episodes, args.memory)
    evolved_results = run_agent_on_tasks(
        eval_tasks(),
        evolved_memory,
        learn=False,
        use_test_feedback=True,
        max_patch_attempts=1,
        max_strategy_scans=1,
    )
    budget_2_results = run_agent_on_tasks(
        eval_tasks(),
        evolved_memory,
        learn=False,
        use_test_feedback=True,
        max_patch_attempts=2,
        max_strategy_scans=2,
    )

    payload = {
        "protocol": {
            "train_tasks": [task.task_id for task in training_tasks()],
            "eval_tasks": [task.task_id for task in eval_tasks()],
            "train_episodes": args.train_episodes,
            "visible_tests": "python3 -B -m unittest discover -s tests -p test_visible.py -v",
            "hidden_tests": "python3 -B -m unittest discover -s tests -p test_hidden.py -v",
            "llm_backend": "offline-rule-backend by default; DeepSeek adapter reads DEEPSEEK_API_KEY when enabled",
        },
        "baseline": {
            "summary": summarize(baseline_results),
            "episodes": [result.as_dict() for result in baseline_results],
        },
        "feedback": {
            "summary": summarize(feedback_results),
            "episodes": [result.as_dict() for result in feedback_results],
        },
        "self_evolved": {
            "summary": summarize(evolved_results),
            "episodes": [result.as_dict() for result in evolved_results],
        },
        "ablations": {
            "evolved_patch_budget_2": {
                "summary": summarize(budget_2_results),
                "episodes": [result.as_dict() for result in budget_2_results],
            },
            "remove_test_feedback": "see baseline",
            "remove_long_term_memory": "see feedback",
        },
        "policy_memory": {
            "strategy_scores": evolved_memory.strategy_scores,
            "keyword_strategy_scores": evolved_memory.keyword_strategy_scores,
        },
        "cost": {
            "closed_model_api_calls": 0,
            "estimated_api_cost_usd": 0.0,
            "wall_time_seconds": round(time.perf_counter() - started, 4),
        },
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "summary": payload["self_evolved"]["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
