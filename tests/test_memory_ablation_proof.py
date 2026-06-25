"""Tests for the memory ablation proof experiment."""

import json
import tempfile
import unittest
from pathlib import Path

from code_repair_agent.d4j_memory import BenchmarkMemory
from code_repair_agent.memory_ablation_proof import (
    EVAL_CASES,
    TRAIN_CASES,
    VARIANTS,
    SyntheticCase,
    evaluate_case,
    main,
    summarize,
    train_case,
    write_csv,
    write_report,
)


class VariantsDictTest(unittest.TestCase):
    def test_variants_contains_four_entries(self):
        self.assertEqual(len(VARIANTS), 4)

    def test_variants_have_correct_memory_flags(self):
        self.assertEqual(VARIANTS["feedback_only"], (False, False))
        self.assertEqual(VARIANTS["check_memory_only"], (True, False))
        self.assertEqual(VARIANTS["repair_memory_only"], (False, True))
        self.assertEqual(VARIANTS["two_dimensional_memory"], (True, True))

    def test_variant_keys_are_ordered(self):
        keys = list(VARIANTS.keys())
        self.assertEqual(keys, ["feedback_only", "check_memory_only", "repair_memory_only", "two_dimensional_memory"])


class SyntheticCaseTest(unittest.TestCase):
    def test_train_cases_have_five_entries(self):
        self.assertEqual(len(TRAIN_CASES), 5)

    def test_eval_cases_have_four_entries(self):
        self.assertEqual(len(EVAL_CASES), 4)

    def test_eval_case_is_frozen_dataclass(self):
        case = EVAL_CASES[0]
        with self.assertRaises(Exception):
            case.required_scope = "different"

    def test_all_train_cases_have_valid_required_scope(self):
        for case in TRAIN_CASES:
            self.assertIn(case.required_scope, {"trigger", "relevant", "all"})

    def test_all_eval_cases_have_valid_required_scope(self):
        for case in EVAL_CASES:
            self.assertIn(case.required_scope, {"trigger", "relevant", "all"})

    def test_all_train_cases_have_required_patch_style(self):
        for case in TRAIN_CASES:
            self.assertTrue(case.required_patch_style)
            self.assertTrue(case.bad_patch_style)
            self.assertNotEqual(case.required_patch_style, case.bad_patch_style)

    def test_train_and_eval_cases_have_matching_shape(self):
        # TRAIN_CASES[2] is a repeat of TRAIN_CASES[1]; skip it for shape check
        train_eval_pairs = [
            (TRAIN_CASES[0], EVAL_CASES[0]),
            (TRAIN_CASES[1], EVAL_CASES[1]),
            (TRAIN_CASES[3], EVAL_CASES[2]),
            (TRAIN_CASES[4], EVAL_CASES[3]),
        ]
        for train_case_item, eval_case_item in train_eval_pairs:
            self.assertEqual(train_case_item.features, eval_case_item.features)
            self.assertEqual(train_case_item.required_scope, eval_case_item.required_scope)
            self.assertEqual(train_case_item.required_patch_style, eval_case_item.required_patch_style)
            self.assertEqual(train_case_item.bad_patch_style, eval_case_item.bad_patch_style)

    def test_synthetic_case_stores_all_fields(self):
        case = SyntheticCase(
            case_id="test-id",
            features=["project:Test", "class:tester"],
            required_scope="relevant",
            required_patch_style="boundary",
            bad_patch_style="null-check",
        )
        self.assertEqual(case.case_id, "test-id")
        self.assertEqual(case.features, ["project:Test", "class:tester"])
        self.assertEqual(case.required_scope, "relevant")
        self.assertEqual(case.required_patch_style, "boundary")
        self.assertEqual(case.bad_patch_style, "null-check")


class TrainCaseTest(unittest.TestCase):
    def test_train_check_memory_only_populates_test_selection_not_patch_ranking(self):
        memory = BenchmarkMemory()
        case = TRAIN_CASES[0]  # train-regex-parser, scope=relevant
        train_case(memory, case, use_check_memory=True, use_repair_memory=False)
        self.assertTrue(memory.test_selection)
        self.assertEqual(memory.patch_ranking, {})
        self.assertEqual(memory.repair_skill_memory, {})
        # test_selection should have "relevant" for this case's features
        for feature in case.features:
            self.assertIn(feature, memory.test_selection)
            self.assertIn("relevant", memory.test_selection[feature])

    def test_train_repair_memory_only_populates_patch_ranking_not_test_selection(self):
        memory = BenchmarkMemory()
        case = TRAIN_CASES[0]
        train_case(memory, case, use_check_memory=False, use_repair_memory=True)
        self.assertTrue(memory.patch_ranking)
        self.assertEqual(memory.test_selection, {})
        self.assertEqual(memory.test_skill_memory, {})
        for feature in case.features:
            self.assertIn(feature, memory.patch_ranking)
            self.assertIn(case.required_patch_style, memory.patch_ranking[feature])

    def test_train_both_dimensions_populates_both(self):
        memory = BenchmarkMemory()
        case = TRAIN_CASES[0]
        train_case(memory, case, use_check_memory=True, use_repair_memory=True)
        self.assertTrue(memory.test_selection)
        self.assertTrue(memory.patch_ranking)
        for feature in case.features:
            self.assertIn(feature, memory.test_selection)
            self.assertIn(feature, memory.patch_ranking)

    def test_train_feedback_only_leaves_memory_empty(self):
        memory = BenchmarkMemory()
        case = TRAIN_CASES[1]  # train-numeric-types, scope=trigger
        train_case(memory, case, use_check_memory=False, use_repair_memory=False)
        self.assertEqual(memory.test_selection, {})
        self.assertEqual(memory.patch_ranking, {})
        self.assertEqual(memory.repair_skill_memory, {})
        self.assertEqual(memory.test_skill_memory, {})

    def test_train_successful_patch_gets_positive_score(self):
        memory = BenchmarkMemory()
        case = TRAIN_CASES[0]
        train_case(memory, case, use_check_memory=True, use_repair_memory=True)
        for feature in case.features:
            score = memory.patch_ranking[feature][case.required_patch_style]
            self.assertGreater(score, 0)

    def test_train_bad_patch_gets_negative_or_zero_score(self):
        memory = BenchmarkMemory()
        case = TRAIN_CASES[0]  # scope=relevant -> bad patch is regression_failure
        train_case(memory, case, use_check_memory=True, use_repair_memory=True)
        for feature in case.features:
            score = memory.patch_ranking[feature][case.bad_patch_style]
            self.assertLess(score, 0)


class EvaluateCaseTest(unittest.TestCase):
    def test_evaluate_feedback_only_uses_defaults_and_never_solves(self):
        memory = BenchmarkMemory()
        case = EVAL_CASES[0]  # required_scope=relevant, required_patch_style=regex-boundary
        row = evaluate_case(memory, case, "feedback_only", use_check_memory=False, use_repair_memory=False)
        self.assertEqual(row["selected_scope"], "trigger")
        self.assertEqual(row["selected_patch_style"], "direct-default")
        self.assertFalse(row["check_correct"])
        self.assertFalse(row["repair_correct"])
        self.assertFalse(row["solved"])

    def test_evaluate_check_memory_only_selects_correct_scope(self):
        memory = BenchmarkMemory()
        tc = TRAIN_CASES[0]  # scope=relevant, features=["project:Lang","class:fastdateparser"]
        train_case(memory, tc, use_check_memory=True, use_repair_memory=False)
        eval_case = EVAL_CASES[0]  # same features, scope=relevant
        row = evaluate_case(memory, eval_case, "check_memory_only", use_check_memory=True, use_repair_memory=False)
        self.assertEqual(row["selected_scope"], "relevant")
        self.assertTrue(row["check_correct"])
        self.assertEqual(row["selected_patch_style"], "direct-default")
        self.assertFalse(row["repair_correct"])
        self.assertFalse(row["solved"])

    def test_evaluate_repair_memory_only_selects_correct_patch_style(self):
        memory = BenchmarkMemory()
        tc = TRAIN_CASES[0]
        train_case(memory, tc, use_check_memory=False, use_repair_memory=True)
        eval_case = EVAL_CASES[0]
        row = evaluate_case(memory, eval_case, "repair_memory_only", use_check_memory=False, use_repair_memory=True)
        self.assertEqual(row["selected_scope"], "trigger")
        self.assertFalse(row["check_correct"])
        self.assertEqual(row["selected_patch_style"], "regex-boundary")
        self.assertTrue(row["repair_correct"])
        self.assertFalse(row["solved"])

    def test_evaluate_repair_memory_only_solves_trigger_scope_case(self):
        memory = BenchmarkMemory()
        tc = TRAIN_CASES[1]  # scope=trigger, patch=numeric-conversion
        train_case(memory, tc, use_check_memory=False, use_repair_memory=True)
        eval_case = EVAL_CASES[1]
        row = evaluate_case(memory, eval_case, "repair_memory_only", use_check_memory=False, use_repair_memory=True)
        self.assertEqual(row["selected_scope"], "trigger")
        self.assertTrue(row["check_correct"])  # required_scope is "trigger"
        self.assertEqual(row["selected_patch_style"], "numeric-conversion")
        self.assertTrue(row["repair_correct"])
        self.assertTrue(row["solved"])

    def test_evaluate_two_dimensional_memory_solves_all(self):
        memory = BenchmarkMemory()
        for tc in TRAIN_CASES:
            train_case(memory, tc, use_check_memory=True, use_repair_memory=True)
        for ec in EVAL_CASES:
            row = evaluate_case(memory, ec, "two_dimensional_memory", use_check_memory=True, use_repair_memory=True)
            self.assertTrue(row["check_correct"], f"{ec.case_id}: check not correct")
            self.assertTrue(row["repair_correct"], f"{ec.case_id}: repair not correct")
            self.assertTrue(row["solved"], f"{ec.case_id}: not solved")

    def test_evaluate_duplicate_risk_is_true_for_default_and_bad_styles(self):
        memory = BenchmarkMemory()
        case = EVAL_CASES[0]
        # feedback_only: selected_patch_style = "direct-default"
        row = evaluate_case(memory, case, "feedback_only", use_check_memory=False, use_repair_memory=False)
        self.assertTrue(row["duplicate_risk"])

    def test_evaluate_regression_overfit_risk_when_scope_is_trigger_but_required_is_broader(self):
        memory = BenchmarkMemory()
        case = EVAL_CASES[0]  # required_scope=relevant
        # feedback_only returns scope=trigger
        row = evaluate_case(memory, case, "feedback_only", use_check_memory=False, use_repair_memory=False)
        self.assertTrue(row["regression_overfit_risk"])

    def test_evaluate_no_regression_overfit_risk_when_scope_matches(self):
        memory = BenchmarkMemory()
        for tc in TRAIN_CASES:
            train_case(memory, tc, use_check_memory=True, use_repair_memory=True)
        case = EVAL_CASES[0]  # required_scope=relevant
        row = evaluate_case(memory, case, "two_dimensional_memory", use_check_memory=True, use_repair_memory=True)
        self.assertFalse(row["regression_overfit_risk"])

    def test_evaluate_simulated_tool_calls_and_test_runs(self):
        memory = BenchmarkMemory()
        case = EVAL_CASES[0]  # required_scope=relevant, so check_correct=False → tool_calls=5
        row = evaluate_case(memory, case, "feedback_only", use_check_memory=False, use_repair_memory=False)
        self.assertEqual(row["simulated_tool_calls"], 5)
        self.assertEqual(row["simulated_test_runs"], 1)  # selected_scope=trigger → 1


class SummarizeTest(unittest.TestCase):
    def test_summarize_empty_rows_returns_zero_counts(self):
        summary = summarize([])
        self.assertEqual(summary["cases"], 0)
        self.assertEqual(summary["solved"], 0)
        self.assertEqual(summary["pass_rate"], 0.0)

    def test_summarize_all_solved(self):
        rows = [
            {"solved": True, "check_correct": True, "repair_correct": True,
             "duplicate_risk": False, "regression_overfit_risk": False,
             "simulated_tool_calls": 3, "simulated_test_runs": 2},
            {"solved": True, "check_correct": True, "repair_correct": True,
             "duplicate_risk": False, "regression_overfit_risk": False,
             "simulated_tool_calls": 3, "simulated_test_runs": 2},
        ]
        summary = summarize(rows)
        self.assertEqual(summary["cases"], 2)
        self.assertEqual(summary["solved"], 2)
        self.assertEqual(summary["pass_rate"], 1.0)
        self.assertEqual(summary["check_correct_rate"], 1.0)
        self.assertEqual(summary["repair_correct_rate"], 1.0)
        self.assertEqual(summary["duplicate_risk_count"], 0)
        self.assertEqual(summary["regression_overfit_risk_count"], 0)
        self.assertEqual(summary["avg_tool_calls"], 3.0)
        self.assertEqual(summary["avg_test_runs"], 2.0)

    def test_summarize_mixed_results(self):
        rows = [
            {"solved": True, "check_correct": True, "repair_correct": True,
             "duplicate_risk": False, "regression_overfit_risk": False,
             "simulated_tool_calls": 3, "simulated_test_runs": 2},
            {"solved": False, "check_correct": True, "repair_correct": False,
             "duplicate_risk": True, "regression_overfit_risk": False,
             "simulated_tool_calls": 5, "simulated_test_runs": 2},
            {"solved": False, "check_correct": False, "repair_correct": True,
             "duplicate_risk": False, "regression_overfit_risk": True,
             "simulated_tool_calls": 5, "simulated_test_runs": 1},
            {"solved": False, "check_correct": False, "repair_correct": False,
             "duplicate_risk": True, "regression_overfit_risk": True,
             "simulated_tool_calls": 5, "simulated_test_runs": 1},
        ]
        summary = summarize(rows)
        self.assertEqual(summary["cases"], 4)
        self.assertEqual(summary["solved"], 1)
        self.assertAlmostEqual(summary["pass_rate"], 0.25)
        self.assertAlmostEqual(summary["check_correct_rate"], 0.5)
        self.assertAlmostEqual(summary["repair_correct_rate"], 0.5)
        self.assertEqual(summary["duplicate_risk_count"], 2)
        self.assertEqual(summary["regression_overfit_risk_count"], 2)
        self.assertAlmostEqual(summary["avg_tool_calls"], (3 + 5 + 5 + 5) / 4, places=4)
        self.assertAlmostEqual(summary["avg_test_runs"], (2 + 2 + 1 + 1) / 4, places=4)

    def test_summarize_returns_dict_keys(self):
        rows = [{
            "solved": False, "check_correct": False, "repair_correct": False,
            "duplicate_risk": True, "regression_overfit_risk": True,
            "simulated_tool_calls": 5, "simulated_test_runs": 1,
        }]
        summary = summarize(rows)
        expected_keys = {
            "cases", "solved", "pass_rate", "check_correct_rate",
            "repair_correct_rate", "duplicate_risk_count",
            "regression_overfit_risk_count", "avg_tool_calls", "avg_test_runs",
        }
        self.assertEqual(set(summary.keys()), expected_keys)


class WriteCsvTest(unittest.TestCase):
    def test_write_csv_creates_file_with_header_and_rows(self):
        rows = [
            {"variant": "feedback_only", "case_id": "eval-1", "solved": False, "check_correct": False},
            {"variant": "two_dimensional", "case_id": "eval-2", "solved": True, "check_correct": True},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            write_csv(path, rows)
            self.assertTrue(path.exists())
            content = path.read_text(encoding="utf-8")
            self.assertIn("variant,case_id,solved,check_correct", content)
            self.assertIn("feedback_only,eval-1,False,False", content)
            self.assertIn("two_dimensional,eval-2,True,True", content)


class WriteReportTest(unittest.TestCase):
    def test_write_report_creates_markdown_file(self):
        summary = {
            "feedback_only": {
                "cases": 4, "solved": 0, "check_correct_rate": 0.25,
                "repair_correct_rate": 0.0, "duplicate_risk_count": 4,
                "regression_overfit_risk_count": 3, "avg_tool_calls": 5.0, "avg_test_runs": 1.0,
            },
            "two_dimensional_memory": {
                "cases": 4, "solved": 4, "check_correct_rate": 1.0,
                "repair_correct_rate": 1.0, "duplicate_risk_count": 0,
                "regression_overfit_risk_count": 0, "avg_tool_calls": 3.0, "avg_test_runs": 2.0,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analysis.md"
            write_report(path, summary)
            self.assertTrue(path.exists())
            content = path.read_text(encoding="utf-8")
            self.assertIn("# Two-Dimensional Memory Proof Experiment", content)
            self.assertIn("feedback_only", content)
            self.assertIn("two_dimensional_memory", content)
            self.assertIn("0/4", content)
            self.assertIn("4/4", content)


class FullAblationProofEndToEndTest(unittest.TestCase):
    def test_full_proof_two_dimensional_memory_solves_all_four(self):
        for variant, (use_check, use_repair) in VARIANTS.items():
            memory = BenchmarkMemory()
            for case in TRAIN_CASES:
                train_case(memory, case, use_check_memory=use_check, use_repair_memory=use_repair)
            rows = [
                evaluate_case(memory, case, variant, use_check_memory=use_check, use_repair_memory=use_repair)
                for case in EVAL_CASES
            ]
            summary = summarize(rows)
            if variant == "two_dimensional_memory":
                self.assertEqual(summary["solved"], 4, f"{variant}: expected 4 solved, got {summary['solved']}")
                self.assertEqual(summary["duplicate_risk_count"], 0)
                self.assertEqual(summary["regression_overfit_risk_count"], 0)
            elif variant == "feedback_only":
                self.assertEqual(summary["solved"], 0,
                                 f"{variant}: expected 0 solved, got {summary['solved']}")
            elif variant == "check_memory_only":
                self.assertEqual(summary["solved"], 0,
                                 f"{variant}: expected 0 solved, got {summary['solved']}")
            elif variant == "repair_memory_only":
                self.assertLessEqual(summary["solved"], 1,
                                     f"{variant}: expected <=1 solved, got {summary['solved']}")

    def test_full_proof_feedback_only_has_duplicate_and_overfit_risks(self):
        memory = BenchmarkMemory()
        for case in TRAIN_CASES:
            train_case(memory, case, use_check_memory=False, use_repair_memory=False)
        rows = [
            evaluate_case(memory, case, "feedback_only", use_check_memory=False, use_repair_memory=False)
            for case in EVAL_CASES
        ]
        summary = summarize(rows)
        self.assertEqual(summary["duplicate_risk_count"], 4)
        self.assertGreater(summary["regression_overfit_risk_count"], 0)

    def test_full_proof_check_memory_only_fixes_scope_not_patch(self):
        memory = BenchmarkMemory()
        for case in TRAIN_CASES:
            train_case(memory, case, use_check_memory=True, use_repair_memory=False)
        rows = [
            evaluate_case(memory, case, "check_memory_only", use_check_memory=True, use_repair_memory=False)
            for case in EVAL_CASES
        ]
        summary = summarize(rows)
        self.assertEqual(summary["check_correct_rate"], 1.0)
        self.assertEqual(summary["repair_correct_rate"], 0.0)
        self.assertEqual(summary["solved"], 0)

    def test_full_proof_repair_memory_only_fixes_patch_not_scope(self):
        memory = BenchmarkMemory()
        for case in TRAIN_CASES:
            train_case(memory, case, use_check_memory=False, use_repair_memory=True)
        rows = [
            evaluate_case(memory, case, "repair_memory_only", use_check_memory=False, use_repair_memory=True)
            for case in EVAL_CASES
        ]
        summary = summarize(rows)
        self.assertEqual(summary["repair_correct_rate"], 1.0)
        # check_correct_rate is only 0.25 because only 1 of 4 has required_scope=trigger
        self.assertEqual(summary["check_correct_rate"], 0.25)
        self.assertLessEqual(summary["solved"], 1)


class MainEntryPointTest(unittest.TestCase):
    def test_main_creates_summary_json_csv_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "proof_out"
            # Patch sys.argv to pass --out-dir
            import sys
            original_argv = sys.argv
            try:
                sys.argv = ["memory_ablation_proof.py", "--out-dir", str(out_dir)]
                main()
            finally:
                sys.argv = original_argv

            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "metrics.csv").exists())
            self.assertTrue((out_dir / "analysis.md").exists())

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("two_dimensional_memory", summary)
            self.assertEqual(summary["two_dimensional_memory"]["solved"], 4)
            self.assertEqual(summary["feedback_only"]["solved"], 0)

    def test_main_memory_files_are_saved_for_each_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "proof_out"
            import sys
            original_argv = sys.argv
            try:
                sys.argv = ["memory_ablation_proof.py", "--out-dir", str(out_dir)]
                main()
            finally:
                sys.argv = original_argv

            for variant in VARIANTS:
                memory_file = out_dir / f"{variant}_memory_after_train.json"
                self.assertTrue(memory_file.exists(), f"Missing memory file for {variant}")
                loaded = BenchmarkMemory.load(memory_file)
                if variant == "feedback_only":
                    self.assertEqual(loaded.test_selection, {})
                    self.assertEqual(loaded.patch_ranking, {})
                else:
                    # At least one dimension should be non-empty
                    self.assertTrue(
                        loaded.test_selection or loaded.patch_ranking,
                        f"{variant}: both dimensions empty",
                    )


if __name__ == "__main__":
    unittest.main()
