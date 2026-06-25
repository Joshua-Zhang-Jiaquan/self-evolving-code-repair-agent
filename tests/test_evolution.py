"""Tests for the evolution module: summarize, run_agent_on_tasks, train_policy_memory."""

import tempfile
import unittest
from pathlib import Path

from code_repair_agent.agent import EpisodeResult, PolicyMemory
from code_repair_agent.evolution import run_agent_on_tasks, summarize, train_policy_memory
from code_repair_agent.tasks import eval_tasks, training_tasks


class SummarizeTest(unittest.TestCase):
    def test_summarize_empty_returns_all_zeros(self):
        result = summarize([])
        self.assertEqual(result["tasks"], 0)
        self.assertEqual(result["visible_solve_rate"], 0.0)
        self.assertEqual(result["hidden_solve_rate"], 0.0)
        self.assertEqual(result["avg_reward"], 0.0)
        self.assertEqual(result["avg_tool_calls"], 0.0)
        self.assertEqual(result["avg_test_runs"], 0.0)
        self.assertEqual(result["pass_at_1"], 0.0)
        self.assertEqual(result["pass_at_k"], 0.0)
        self.assertEqual(result["avg_patch_size"], 0.0)
        self.assertEqual(result["unsafe_edit_rate"], 0.0)
        # Verify all expected keys are present and no extras
        expected_keys = {
            "tasks",
            "visible_solve_rate",
            "hidden_solve_rate",
            "avg_reward",
            "avg_tool_calls",
            "avg_test_runs",
            "pass_at_1",
            "pass_at_k",
            "avg_patch_size",
            "unsafe_edit_rate",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_summarize_populated_returns_correct_rates(self):
        results = run_agent_on_tasks(eval_tasks(), PolicyMemory(), learn=False)
        summary = summarize(results)
        # All eval tasks are designed to be solvable by the agent
        self.assertEqual(summary["tasks"], 3)
        self.assertEqual(summary["hidden_solve_rate"], 1.0)
        self.assertEqual(summary["visible_solve_rate"], 1.0)
        self.assertEqual(summary["pass_at_1"], 1.0)
        self.assertEqual(summary["unsafe_edit_rate"], 0.0)
        self.assertGreater(summary["avg_reward"], 0)
        self.assertGreater(summary["avg_tool_calls"], 0)
        self.assertGreater(summary["avg_test_runs"], 0)
        self.assertGreater(summary["avg_patch_size"], 0)
        # pass_at_k mirrors hidden_solve_rate in current implementation
        self.assertEqual(summary["pass_at_k"], summary["hidden_solve_rate"])

    def test_summarize_with_partial_solves(self):
        """Summarize when only some tasks are solved."""
        # Use max_strategy_scans=1 which limits to first strategy (slugify_split_join).
        # Only eval_slugify matches slugify; other two eval tasks won't match.
        # Use learning to avoid any ordering changes from memory affecting results.
        memory = PolicyMemory()
        results = run_agent_on_tasks(
            eval_tasks(), memory, learn=False, max_strategy_scans=1
        )
        summary = summarize(results)
        # 1 out of 3 tasks solved by slugify strategy
        self.assertEqual(summary["tasks"], 3)
        # The exact rate depends on the agent behavior with strategy scan limit
        self.assertGreaterEqual(summary["hidden_solve_rate"], 0.0)
        self.assertLess(summary["hidden_solve_rate"], 1.0)
        # unsolved tasks have 0 reward, solved tasks have positive reward
        self.assertGreater(summary["avg_reward"], 0.0)


class RunAgentOnTasksTest(unittest.TestCase):
    def test_learn_false_memory_not_updated(self):
        memory = PolicyMemory()
        run_agent_on_tasks(eval_tasks(), memory, learn=False)
        self.assertEqual(memory.strategy_scores, {})

    def test_learn_true_memory_is_updated(self):
        memory = PolicyMemory()
        run_agent_on_tasks(eval_tasks(), memory, learn=True)
        self.assertNotEqual(memory.strategy_scores, {})
        # At least one strategy should have been tried and scored
        self.assertGreater(len(memory.strategy_scores), 0)

    def test_use_test_feedback_false_single_shot_patch(self):
        """Without test feedback, agent applies one patch and stops."""
        results = run_agent_on_tasks(
            eval_tasks(), PolicyMemory(), learn=False, use_test_feedback=False
        )
        for result in results:
            if result.patch_attempts > 0:
                self.assertEqual(
                    result.patch_attempts, 1,
                    f"Expected patch_attempts=1 without feedback for {result.task_id}",
                )

    def test_max_patch_attempts_1_limits_attempts(self):
        results = run_agent_on_tasks(
            training_tasks(), PolicyMemory(), learn=False, max_patch_attempts=1
        )
        for result in results:
            self.assertLessEqual(result.patch_attempts, 1)

    def test_max_strategy_scans_1_limits_strategies(self):
        """With only 1 strategy scan, at most one strategy is used across all tasks."""
        results = run_agent_on_tasks(
            eval_tasks(), PolicyMemory(), learn=False, max_strategy_scans=1,
        )
        strategies_used = {r.strategy for r in results if r.strategy is not None}
        # With max_strategy_scans=1 and empty memory, only slugify_split_join
        # (the first in default order) can be tried
        self.assertLessEqual(len(strategies_used), 1)
        if strategies_used:
            self.assertIn(list(strategies_used)[0], {"slugify_split_join"})

    def test_run_agent_learn_true_on_single_task(self):
        """Run on a single task with learning; memory picks up the successful strategy."""
        memory = PolicyMemory()
        results = run_agent_on_tasks(
            training_tasks()[:1], memory, learn=True,
        )
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].solved_hidden)
        # After learning, the strategy used should be scored in memory
        strategy = results[0].strategy
        self.assertIsNotNone(strategy)
        self.assertIn(strategy, memory.strategy_scores)
        self.assertGreater(memory.strategy_scores[strategy], 0)

    def test_returns_episode_result_list(self):
        results = run_agent_on_tasks(training_tasks()[:1], PolicyMemory(), learn=False)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertIsInstance(r, EpisodeResult)
        self.assertIsInstance(r.task_id, str)
        self.assertIsInstance(r.solved_visible, bool)
        self.assertIsInstance(r.solved_hidden, bool)
        self.assertIsInstance(r.reward.reward, float)
        self.assertIsInstance(r.strategy, (str, type(None)))
        self.assertIsInstance(r.diff, str)
        self.assertIsInstance(r.tool_calls, int)
        self.assertIsInstance(r.test_runs, int)
        self.assertIsInstance(r.patch_attempts, int)
        self.assertIsInstance(r.pass_at_1, bool)
        self.assertIsInstance(r.patch_size, int)
        self.assertIsInstance(r.events, list)
        # A solved task should have positive reward and pass_at_1=True
        self.assertTrue(r.solved_hidden)
        self.assertTrue(r.pass_at_1)
        self.assertGreater(r.reward.reward, 0)
        self.assertGreater(r.patch_size, 0)
        self.assertGreater(r.tool_calls, 0)
        self.assertGreater(r.test_runs, 0)

    def test_diff_produced_for_solved_task(self):
        results = run_agent_on_tasks(training_tasks()[:1], PolicyMemory(), learn=False)
        self.assertGreater(len(results[0].diff), 0)
        self.assertIn("+", results[0].diff)

    def test_no_learn_preserves_empty_keyword_scores(self):
        memory = PolicyMemory()
        run_agent_on_tasks(eval_tasks(), memory, learn=False)
        self.assertEqual(memory.strategy_scores, {})
        self.assertEqual(memory.keyword_strategy_scores, {})

    def test_learn_populates_keyword_scores(self):
        memory = PolicyMemory()
        run_agent_on_tasks(eval_tasks(), memory, learn=True)
        self.assertNotEqual(memory.keyword_strategy_scores, {})


class TrainPolicyMemoryTest(unittest.TestCase):
    def test_single_episode_creates_memory_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "policy_memory.json"
            memory = train_policy_memory(training_tasks(), 1, memory_path)
            self.assertTrue(memory_path.exists())
            self.assertTrue(memory.strategy_scores)

    def test_three_episodes_accumulates_more_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem1 = train_policy_memory(
                training_tasks(), 1, Path(tmp) / "mem1.json",
            )
            mem3 = train_policy_memory(
                training_tasks(), 3, Path(tmp) / "mem3.json",
            )
            # After 3 episodes, same strategies get higher scores (if solved consistently)
            self.assertTrue(mem1.strategy_scores)
            self.assertTrue(mem3.strategy_scores)
            # Sum of absolute scores should be >= for 3 episodes
            sum1 = sum(abs(v) for v in mem1.strategy_scores.values())
            sum3 = sum(abs(v) for v in mem3.strategy_scores.values())
            self.assertGreaterEqual(sum3, sum1)


if __name__ == "__main__":
    unittest.main()
