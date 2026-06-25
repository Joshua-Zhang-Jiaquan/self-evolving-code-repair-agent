import tempfile
import unittest
from pathlib import Path

from code_repair_agent.agent import PolicyMemory
from code_repair_agent.evolution import run_agent_on_tasks, summarize, train_policy_memory
from code_repair_agent.tasks import eval_tasks, training_tasks


class EvaluationTest(unittest.TestCase):
    def test_agent_solves_eval_tasks(self):
        results = run_agent_on_tasks(eval_tasks(), PolicyMemory(), learn=False)
        summary = summarize(results)
        self.assertEqual(summary["hidden_solve_rate"], 1.0)

    def test_training_writes_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "policy_memory.json"
            memory = train_policy_memory(training_tasks(), 1, memory_path)
            self.assertTrue(memory_path.exists())
            self.assertTrue(memory.strategy_scores)


if __name__ == "__main__":
    unittest.main()
