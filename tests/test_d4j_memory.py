import copy
import json
import tempfile
import unittest
from pathlib import Path

from code_repair_agent.d4j_memory import BenchmarkMemory


class BenchmarkMemoryDecayTest(unittest.TestCase):
    def test_apply_decay_multiplies_all_tables(self):
        memory = BenchmarkMemory(
            patch_ranking={"feature": {"boundary": 2.0}},
            test_selection={"feature": {"trigger": 4.0}},
            repair_skill_memory={"feature": {"repair": 6.0}},
            test_skill_memory={"feature": {"test": 8.0}},
            regression_outcomes={"feature": {"style:boundary|scope:trigger": -10.0}},
        )

        memory.apply_decay(0.5)

        self.assertEqual(memory.patch_ranking["feature"]["boundary"], 1.0)
        self.assertEqual(memory.test_selection["feature"]["trigger"], 2.0)
        self.assertEqual(memory.repair_skill_memory["feature"]["repair"], 3.0)
        self.assertEqual(memory.test_skill_memory["feature"]["test"], 4.0)
        self.assertEqual(memory.regression_outcomes["feature"]["style:boundary|scope:trigger"], -5.0)

    def test_apply_decay_removes_near_zero_scores(self):
        memory = BenchmarkMemory(
            patch_ranking={"small": {"boundary": 0.05}, "keep": {"boundary": 0.2}},
            test_selection={"small": {"trigger": -0.05}},
            repair_skill_memory={"small": {"repair": 0.09}},
            test_skill_memory={"small": {"test": -0.09}},
            regression_outcomes={"small": {"style:boundary|scope:trigger": 0.05}},
        )

        memory.apply_decay(0.1)

        self.assertNotIn("small", memory.patch_ranking)
        self.assertEqual(memory.patch_ranking["keep"]["boundary"], 0.020000000000000004)
        self.assertEqual(memory.test_selection, {})
        self.assertEqual(memory.repair_skill_memory, {})
        self.assertEqual(memory.test_skill_memory, {})
        self.assertEqual(memory.regression_outcomes, {})

    def test_apply_decay_preserves_reflections_and_strategies(self):
        memory = BenchmarkMemory(
            patch_ranking={"feature": {"boundary": 2.0}},
            failure_reflections=[{"features": "feature", "failure_reason": "visible_failure", "reflection": "try again"}],
            success_strategies=[{"features": "feature", "strategy": "use boundary"}],
        )
        reflections = copy.deepcopy(memory.failure_reflections)
        strategies = copy.deepcopy(memory.success_strategies)

        memory.apply_decay(0.5)

        self.assertEqual(memory.failure_reflections, reflections)
        self.assertEqual(memory.success_strategies, strategies)

    def test_apply_decay_with_factor_1_is_noop(self):
        memory = BenchmarkMemory(
            patch_ranking={"feature": {"boundary": 2.0}},
            test_selection={"feature": {"trigger": 4.0}},
            repair_skill_memory={"feature": {"repair": 6.0}},
            test_skill_memory={"feature": {"test": 8.0}},
            regression_outcomes={"feature": {"style:boundary|scope:trigger": -10.0}},
        )
        before = copy.deepcopy(memory.as_dict())

        memory.apply_decay(1.0)

        self.assertEqual(memory.as_dict(), before)

    def test_apply_decay_ignores_invalid_factors(self):
        memory = BenchmarkMemory(
            patch_ranking={"feature": {"boundary": 2.0}},
            test_selection={"feature": {"trigger": 4.0}},
            repair_skill_memory={"feature": {"repair": 6.0}},
            test_skill_memory={"feature": {"test": 8.0}},
            regression_outcomes={"feature": {"style:boundary|scope:trigger": -10.0}},
        )
        before = copy.deepcopy(memory.as_dict())

        memory.apply_decay(0)
        memory.apply_decay(-1)
        memory.apply_decay(2)

        self.assertEqual(memory.as_dict(), before)

    def test_apply_decay_empty_memory(self):
        memory = BenchmarkMemory()
        before = copy.deepcopy(memory.as_dict())

        memory.apply_decay(0.5)

        self.assertEqual(memory.as_dict(), before)


class BenchmarkMemoryDedupTest(unittest.TestCase):
    def test_failure_reflection_dedup_replaces_same_features_and_reason(self):
        memory = BenchmarkMemory()
        features = ["project:Lang", "class:parser"]

        memory.update(
            features=features,
            patch_style="boundary",
            test_scope="trigger",
            solved=False,
            failure_reason="visible_failure",
            reflection="old reflection",
        )
        memory.update(
            features=features,
            patch_style="boundary",
            test_scope="trigger",
            solved=False,
            failure_reason="visible_failure",
            reflection="new reflection",
        )

        self.assertEqual(len(memory.failure_reflections), 1)
        self.assertEqual(memory.failure_reflections[0]["reflection"], "new reflection")

    def test_failure_reflection_keeps_different_reasons(self):
        memory = BenchmarkMemory()
        features = ["project:Lang", "class:parser"]

        memory.update(
            features=features,
            patch_style="boundary",
            test_scope="trigger",
            solved=False,
            failure_reason="visible_failure",
            reflection="visible failed",
        )
        memory.update(
            features=features,
            patch_style="boundary",
            test_scope="trigger",
            solved=False,
            failure_reason="compile_failure",
            reflection="compile failed",
        )

        self.assertEqual(len(memory.failure_reflections), 2)
        self.assertEqual(
            {item["failure_reason"] for item in memory.failure_reflections},
            {"visible_failure", "compile_failure"},
        )

    def test_failure_reflection_cap_removes_oldest(self):
        memory = BenchmarkMemory(max_failure_reflections=3)

        for index in range(4):
            memory.update(
                features=[f"project:Lang{index}"],
                patch_style="boundary",
                test_scope="trigger",
                solved=False,
                failure_reason="visible_failure",
                reflection=f"reflection {index}",
            )

        self.assertEqual(len(memory.failure_reflections), 3)
        self.assertEqual([item["features"] for item in memory.failure_reflections], ["project:Lang1", "project:Lang2", "project:Lang3"])

    def test_success_strategy_dedup_replaces_same_features(self):
        memory = BenchmarkMemory()
        features = ["project:Lang", "class:parser"]

        memory.update(
            features=features,
            patch_style="boundary",
            test_scope="trigger",
            solved=True,
            failure_reason=None,
            reflection=None,
            success_strategy="old strategy",
        )
        memory.update(
            features=features,
            patch_style="boundary",
            test_scope="trigger",
            solved=True,
            failure_reason=None,
            reflection=None,
            success_strategy="new strategy",
        )

        self.assertEqual(len(memory.success_strategies), 1)
        self.assertEqual(memory.success_strategies[0]["strategy"], "new strategy")

    def test_success_strategy_cap_removes_oldest(self):
        memory = BenchmarkMemory(max_success_strategies=3)

        for index in range(4):
            memory.update(
                features=[f"project:Lang{index}"],
                patch_style="boundary",
                test_scope="trigger",
                solved=True,
                failure_reason=None,
                reflection=None,
                success_strategy=f"strategy {index}",
            )

        self.assertEqual(len(memory.success_strategies), 3)
        self.assertEqual([item["features"] for item in memory.success_strategies], ["project:Lang1", "project:Lang2", "project:Lang3"])


class BenchmarkMemorySerializationTest(unittest.TestCase):
    def test_full_serialization_round_trip(self):
        memory = BenchmarkMemory(max_failure_reflections=11, max_success_strategies=12)
        memory.update(
            features=["project:Lang", "class:parser"],
            patch_style="boundary",
            test_scope="relevant",
            solved=True,
            failure_reason=None,
            reflection=None,
            repair_skill="repair-boundary",
            test_skill="run-relevant",
            regression_checked=True,
            regression_passed=True,
            success_strategy="patch the parser boundary",
        )
        memory.update(
            features=["project:Math", "exception:assertionerror"],
            patch_style="guard",
            test_scope="all",
            solved=False,
            failure_reason="regression_failure",
            reflection="visible passed but regression failed",
            repair_skill="repair-after-regression",
            test_skill="run-all",
            visible_passed=True,
            regression_checked=True,
            regression_passed=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.json"
            memory.save(path)
            loaded = BenchmarkMemory.load(path)

        self.assertEqual(loaded.as_dict(), memory.as_dict())

    def test_serialization_preserves_caps(self):
        memory = BenchmarkMemory(max_failure_reflections=7, max_success_strategies=8)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.json"
            memory.save(path)
            loaded = BenchmarkMemory.load(path)

        self.assertEqual(loaded.max_failure_reflections, 7)
        self.assertEqual(loaded.max_success_strategies, 8)

    def test_load_backward_compatibility(self):
        raw = {
            "patch_ranking": {"feature": {"boundary": 1}},
            "test_selection": {"feature": {"trigger": 2}},
            "repair_skill_memory": {"feature": {"repair": 3}},
            "test_skill_memory": {"feature": {"test": 4}},
            "regression_outcomes": {"feature": {"style:boundary|scope:trigger": -1}},
            "failure_reflections": [],
            "success_strategies": [],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            loaded = BenchmarkMemory.load(path)

        self.assertEqual(loaded.max_failure_reflections, 100)
        self.assertEqual(loaded.max_success_strategies, 100)
        self.assertEqual(loaded.patch_ranking["feature"]["boundary"], 1.0)


class BenchmarkMemoryEdgeCaseTest(unittest.TestCase):
    def test_empty_features_update(self):
        memory = BenchmarkMemory()

        memory.update(
            features=[],
            patch_style="boundary",
            test_scope="trigger",
            solved=False,
            failure_reason="visible_failure",
            reflection="no feature reflection",
            repair_skill="repair",
            test_skill="test",
            regression_checked=True,
            regression_passed=False,
        )

        self.assertEqual(memory.as_dict(), BenchmarkMemory().as_dict())

    def test_update_with_only_check_memory(self):
        memory = BenchmarkMemory()
        features = ["project:Lang", "class:parser"]

        memory.update(
            features=features,
            patch_style="regex-boundary",
            test_scope="relevant",
            solved=True,
            failure_reason=None,
            reflection=None,
            repair_skill="use-regex-boundary",
            test_skill="run-relevant",
            regression_checked=True,
            regression_passed=True,
            success_strategy="successful repair strategy",
            update_check_memory=True,
            update_repair_memory=False,
        )

        self.assertEqual(memory.patch_ranking, {})
        self.assertEqual(memory.repair_skill_memory, {})
        self.assertEqual(memory.failure_reflections, [])
        self.assertEqual(memory.success_strategies, [])
        self.assertEqual(memory.preferred_test_scope(features), "relevant")
        self.assertEqual(memory.test_skill_preferences(features), ["run-relevant"])
        self.assertIn("style:regex-boundary|scope:relevant", memory.regression_outcomes["project:Lang"])

    def test_update_with_only_repair_memory(self):
        memory = BenchmarkMemory()
        features = ["project:Lang", "class:parser"]

        memory.update(
            features=features,
            patch_style="regex-boundary",
            test_scope="relevant",
            solved=True,
            failure_reason=None,
            reflection=None,
            repair_skill="use-regex-boundary",
            test_skill="run-relevant",
            regression_checked=True,
            regression_passed=True,
            success_strategy="patch regex boundary and validate relevant tests",
            update_check_memory=False,
            update_repair_memory=True,
        )

        self.assertEqual(memory.test_selection, {})
        self.assertEqual(memory.test_skill_memory, {})
        self.assertEqual(memory.regression_outcomes, {})
        self.assertEqual(memory.prompt_preferences(features), ["regex-boundary"])
        self.assertEqual(memory.repair_skill_preferences(features), ["use-regex-boundary"])
        self.assertIn("regex boundary", memory.relevant_success_strategies(features)[0])

    def test_feature_weighting_specific_vs_broad(self):
        memory = BenchmarkMemory(
            patch_ranking={
                "project:Lang": {"broad-project-style": 3.0},
                "class:parser": {"specific-class-style": 2.0},
            }
        )

        preferences = memory.prompt_preferences(["project:Lang", "class:parser"])

        self.assertEqual(preferences[0], "specific-class-style")
        self.assertEqual(preferences[1], "broad-project-style")


if __name__ == "__main__":
    unittest.main()
