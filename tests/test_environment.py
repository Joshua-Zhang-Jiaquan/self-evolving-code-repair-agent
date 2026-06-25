import subprocess
import unittest
from unittest.mock import patch

from code_repair_agent.environment import CodeRepairEnvironment
from code_repair_agent.tasks import training_tasks


class EnvironmentTest(unittest.TestCase):
    def test_edit_and_reward(self):
        task = training_tasks()[0]
        with CodeRepairEnvironment(task) as env:
            initial = env.inspect_test()
            self.assertFalse(initial.passed)
            env.edit_file("calculator.py", "return a - b", "return a + b")
            self.assertTrue(env.run_tests("visible").passed)
            reward = env.reward()
            self.assertEqual(reward.pass_visible_tests, 1)
            self.assertEqual(reward.pass_hidden_tests, 1)
            self.assertGreater(reward.reward, 0)

    def test_blocks_test_edits(self):
        task = training_tasks()[0]
        with CodeRepairEnvironment(task) as env:
            detail = env.edit_file("tests/test_visible.py", "5", "4")
            self.assertIn("blocked", detail)
            reward = env.reward()
            self.assertEqual(reward.unsafe_edit, 1)
            self.assertEqual(reward.test_deletion, 1)

    def test_blocks_path_traversal_tools(self):
        task = training_tasks()[0]
        with CodeRepairEnvironment(task) as env:
            read_detail = env.read_file("../outside.py")
            edit_detail = env.edit_file("../outside.py", "a", "b")
            self.assertIn("blocked", read_detail)
            self.assertIn("blocked", edit_detail)
            reward = env.reward()
            self.assertEqual(reward.unsafe_edit, 1)
            self.assertEqual(reward.test_deletion, 0)

    def test_timeout_becomes_failed_run_result(self):
        task = training_tasks()[0]
        with CodeRepairEnvironment(task, test_timeout=17) as env:
            with patch("code_repair_agent.environment.subprocess.run") as run:
                run.side_effect = subprocess.TimeoutExpired(["python"], 17, output="partial")
                result = env.run_tests("visible")
            self.assertEqual(result.returncode, 124)
            self.assertIn("timed out after 17s", result.output)
            self.assertIn("timed out", env.state.last_test_output)


if __name__ == "__main__":
    unittest.main()
