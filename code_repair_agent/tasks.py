"""Benchmark task definitions for the code repair environment."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List


VISIBLE_CMD = [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_visible.py", "-v"]
HIDDEN_CMD = [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_hidden.py", "-v"]


@dataclass(frozen=True)
class RepairTask:
    task_id: str
    issue: str
    files: Dict[str, str]
    visible_test_command: List[str]
    hidden_test_command: List[str]
    optional_hints: str = ""


def _task(task_id: str, issue: str, files: Dict[str, str], hints: str = "") -> RepairTask:
    return RepairTask(
        task_id=task_id,
        issue=issue,
        files=files,
        visible_test_command=list(VISIBLE_CMD),
        hidden_test_command=list(HIDDEN_CMD),
        optional_hints=hints,
    )


def make_tasks() -> List[RepairTask]:
    """Return deterministic training and held-out evaluation tasks."""

    return [
        _task(
            "train_add_operator",
            "The add(a, b) helper returns the wrong arithmetic result.",
            {
                "calculator.py": """def add(a, b):\n    return a - b\n""",
                "tests/test_visible.py": """import unittest\nfrom calculator import add\n\n\nclass TestAddVisible(unittest.TestCase):\n    def test_positive(self):\n        self.assertEqual(add(2, 3), 5)\n\n\nif __name__ == "__main__":\n    unittest.main()\n""",
                "tests/test_hidden.py": """import unittest\nfrom calculator import add\n\n\nclass TestAddHidden(unittest.TestCase):\n    def test_negative(self):\n        self.assertEqual(add(-4, 10), 6)\n\n\nif __name__ == "__main__":\n    unittest.main()\n""",
            },
            hints="Visible failure: expected 5 but got -1.",
        ),
        _task(
            "train_factorial_base",
            "factorial(0) should be 1, but the base case is wrong.",
            {
                "math_utils.py": """def factorial(n):\n    if n < 0:\n        raise ValueError(\"n must be non-negative\")\n    if n == 0:\n        return 0\n    return n * factorial(n - 1)\n""",
                "tests/test_visible.py": """import unittest\nfrom math_utils import factorial\n\n\nclass TestFactorialVisible(unittest.TestCase):\n    def test_zero(self):\n        self.assertEqual(factorial(0), 1)\n\n\nif __name__ == "__main__":\n    unittest.main()\n""",
                "tests/test_hidden.py": """import unittest\nfrom math_utils import factorial\n\n\nclass TestFactorialHidden(unittest.TestCase):\n    def test_five(self):\n        self.assertEqual(factorial(5), 120)\n\n\nif __name__ == "__main__":\n    unittest.main()\n""",
            },
        ),
        _task(
            "train_normalize_zero",
            "normalize(values) crashes on an all-zero list; return zeros instead.",
            {
                "stats.py": """def normalize(values):\n    total = sum(values)\n    return [v / total for v in values]\n""",
                "tests/test_visible.py": """import unittest\nfrom stats import normalize\n\n\nclass TestNormalizeVisible(unittest.TestCase):\n    def test_all_zero(self):\n        self.assertEqual(normalize([0, 0]), [0, 0])\n\n\nif __name__ == "__main__":\n    unittest.main()\n""",
                "tests/test_hidden.py": """import unittest\nfrom stats import normalize\n\n\nclass TestNormalizeHidden(unittest.TestCase):\n    def test_regular_values(self):\n        self.assertEqual(normalize([1, 1, 2]), [0.25, 0.25, 0.5])\n\n\nif __name__ == "__main__":\n    unittest.main()\n""",
            },
        ),
        _task(
            "eval_add_operator",
            "sum_pair should add both inputs, but subtraction appears in the implementation.",
            {
                "numbers.py": """def sum_pair(left, right):\n    return left - right\n""",
                "tests/test_visible.py": """import unittest\nfrom numbers import sum_pair\n\n\nclass TestSumVisible(unittest.TestCase):\n    def test_small_numbers(self):\n        self.assertEqual(sum_pair(7, 8), 15)\n\n\nif __name__ == "__main__":\n    unittest.main()\n""",
                "tests/test_hidden.py": """import unittest\nfrom numbers import sum_pair\n\n\nclass TestSumHidden(unittest.TestCase):\n    def test_mixed_signs(self):\n        self.assertEqual(sum_pair(-3, 9), 6)\n\n\nif __name__ == "__main__":\n    unittest.main()\n""",
            },
        ),
        _task(
            "eval_factorial_base",
            "The product_down function uses the wrong identity for n == 0.",
            {
                "product.py": """def product_down(n):\n    if n < 0:\n        raise ValueError(\"n must be non-negative\")\n    if n == 0:\n        return 0\n    return n * product_down(n - 1)\n""",
                "tests/test_visible.py": """import unittest\nfrom product import product_down\n\n\nclass TestProductVisible(unittest.TestCase):\n    def test_identity(self):\n        self.assertEqual(product_down(0), 1)\n\n\nif __name__ == "__main__":\n    unittest.main()\n""",
                "tests/test_hidden.py": """import unittest\nfrom product import product_down\n\n\nclass TestProductHidden(unittest.TestCase):\n    def test_four(self):\n        self.assertEqual(product_down(4), 24)\n\n\nif __name__ == "__main__":\n    unittest.main()\n""",
            },
        ),
        _task(
            "eval_slugify",
            "slugify should trim leading/trailing whitespace and collapse internal spaces.",
            {
                "text_tools.py": """def slugify(text):\n    return text.lower().replace(\" \", \"-\")\n""",
                "tests/test_visible.py": """import unittest\nfrom text_tools import slugify\n\n\nclass TestSlugVisible(unittest.TestCase):\n    def test_trim(self):\n        self.assertEqual(slugify(\"  Hello World  \"), \"hello-world\")\n\n\nif __name__ == \"__main__\":\n    unittest.main()\n""",
                "tests/test_hidden.py": """import unittest\nfrom text_tools import slugify\n\n\nclass TestSlugHidden(unittest.TestCase):\n    def test_collapse(self):\n        self.assertEqual(slugify(\"Many   Spaces\"), \"many-spaces\")\n\n\nif __name__ == \"__main__\":\n    unittest.main()\n""",
            },
        ),
    ]


def training_tasks() -> List[RepairTask]:
    return [task for task in make_tasks() if task.task_id.startswith("train_")]


def eval_tasks() -> List[RepairTask]:
    return [task for task in make_tasks() if task.task_id.startswith("eval_")]
