import argparse
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from code_repair_agent import llm as llm_module
from code_repair_agent.d4j_benchmark import default_30_cases, load_cases
from code_repair_agent.d4j_memory import BenchmarkMemory
from code_repair_agent.deepseek_repair import (
    RepairPlan,
    build_localization_prompt,
    build_repair_prompt,
    parse_localization_plan,
    parse_repair_plan,
)
from code_repair_agent.defects4j import CommandResult, Defects4JCase, Defects4JClient, _clean_export_output
from code_repair_agent.d4j_test_sweep import SweepMetrics, _summarize as summarize_d4j_sweep
from code_repair_agent.llm import DeepSeekChatClient
from code_repair_agent.real_benchmark import (
    Defects4JBenchmarkRunner,
    RuntimeTuning,
    ScopeResult,
    _compile_failure_feedback,
    _failed_patch_feedback,
    _failure_summary,
    _failure_source_hints,
    _is_non_retryable_llm_error,
    _is_retryable_llm_error,
    _is_duplicate_apply_failure,
    _load_runtime_tuning,
    _patch_grounding_feedback,
    _patch_strategy_signature,
    _prompt_context_budget,
    _constraint_source_needles,
    _derived_repair_constraints,
    _requested_source_paths,
    _scope_failure_tail,
    _trigger_assertion_summary,
    _visible_failure_guidance,
    _duplicate_patch_feedback,
    _duplicate_rejection_limit,
    _failure_output_repair_constraints,
    _failure_output_source_needles,
    _with_carried_grounded_hunks,
    _hunk_ranges_overlap,
    _grounding_error_identity,
)
from code_repair_agent.safe_patch import PatchApplyResult, PatchHunk, RangeHunk, SafePatchApplier


class Defects4JBenchmarkConfigTest(unittest.TestCase):
    def test_default_case_selection(self):
        cases = default_30_cases()
        self.assertEqual(len(cases), 30)
        self.assertEqual(cases[0].case_id, "Chart-1")
        self.assertEqual(cases[10].case_id, "Lang-1")
        self.assertNotIn("Lang-2", [case.case_id for case in cases])
        self.assertEqual(cases[-1].case_id, "Math-10")

    def test_load_config(self):
        cases = load_cases(Path("configs/defects4j_30.json"))
        self.assertEqual(len(cases), 30)
        self.assertEqual(cases[20].case_id, "Math-1")

    def test_runtime_cli_attempts_override_config_without_losing_config_defaults(self):
        args = argparse.Namespace(
            baseline_attempts=None,
            feedback_attempts=None,
            self_evolved_attempts=2,
            memory_attempt_bonus=None,
            max_attempt_cap=2,
            memory_guidance_limit=None,
            max_non_patch_rounds=None,
            memory_path=None,
            memory_mode=None,
            fresh_memory=True,
        )
        tuning = _load_runtime_tuning(Path("configs/defects4j_smoke.json"), args)
        self.assertEqual(tuning.attempt_limits["baseline"], 1)
        self.assertEqual(tuning.attempt_limits["feedback"], 3)
        self.assertEqual(tuning.attempt_limits["self_evolved"], 2)
        self.assertEqual(tuning.max_attempt_cap, 2)
        self.assertGreaterEqual(tuning.max_non_patch_rounds, 1)
        self.assertEqual(tuning.memory_path, Path("artifacts/memory/benchmark_memory.json"))
        self.assertEqual(tuning.memory_mode, "full")

    def test_runtime_cli_memory_mode_override(self):
        args = argparse.Namespace(
            baseline_attempts=None,
            feedback_attempts=None,
            self_evolved_attempts=None,
            memory_attempt_bonus=None,
            max_attempt_cap=None,
            memory_guidance_limit=None,
            max_non_patch_rounds=None,
            memory_path=None,
            memory_mode="check_only",
            fresh_memory=True,
        )
        tuning = _load_runtime_tuning(Path("configs/defects4j_smoke.json"), args)
        self.assertEqual(tuning.memory_mode, "check_only")

    def test_d4j_sweep_summary_counts_infrastructure_separately(self):
        metrics = [
            SweepMetrics("Lang-1", "Lang", 1, "completed", True, True, True, 1, False, True, False, True, False, False, 1.0),
            SweepMetrics("Math-1", "Math", 1, "compile_failed", True, False, False, 0, False, False, False, False, False, True, 2.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            summary = summarize_d4j_sweep(Path(tmp), metrics, jobs=2, scopes=["trigger", "all"])
        self.assertEqual(summary["cases"], 2)
        self.assertEqual(summary["infrastructure_failures"], 1)
        self.assertEqual(summary["compile_success_rate"], 0.5)


class Defects4JClientTest(unittest.TestCase):
    def test_default_timeout_is_relaxed(self):
        client = Defects4JClient()
        self.assertGreaterEqual(client.timeout, 3600)

    def test_clean_export_output_removes_cli_progress(self):
        raw = (
            "Running ant (export.tests.trigger)......................................... OK\n"
            "org.example.FooTest::testA\n"
            "org.example.BarTest\n"
        )
        self.assertEqual(_clean_export_output(raw), "org.example.FooTest::testA\norg.example.BarTest")

    def test_checkout_uses_absolute_workdir(self):
        captured = {}

        class FakeClient(Defects4JClient):
            def require(self):
                return None

            def run(self, command, cwd, check=False):
                captured["command"] = command
                captured["cwd"] = cwd
                return CommandResult(command, str(cwd), 0, "ok", 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            rel_root = Path(tmp).name
            client = FakeClient()
            client.checkout(Defects4JCase("Lang", 1, Path(rel_root) / "Lang-1"))
        workdir_arg = captured["command"][captured["command"].index("-w") + 1]
        self.assertTrue(Path(workdir_arg).is_absolute())
        self.assertEqual(captured["cwd"], Path(workdir_arg).parent)


class SafePatchApplierTest(unittest.TestCase):
    def test_applies_source_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src/main/java/Foo.java"
            src.parent.mkdir(parents=True)
            src.write_text("class Foo { int x() { return 0; } }\n", encoding="utf-8")
            applier = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"])
            result = applier.apply([PatchHunk("src/main/java/Foo.java", "return 0", "return 1")])
            self.assertTrue(result.ok)
            self.assertFalse(result.unsafe)
            self.assertIn("return 1", src.read_text(encoding="utf-8"))

    def test_rejects_unsafe_paths_and_test_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_file = root / "src/test/java/FooTest.java"
            test_file.parent.mkdir(parents=True)
            test_file.write_text("assertEquals(0, x);\n", encoding="utf-8")
            source_file = root / "src/main/java/Foo.java"
            source_file.parent.mkdir(parents=True)
            source_file.write_text("class Foo {}\n", encoding="utf-8")
            applier = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"])
            result = applier.apply(
                [
                    PatchHunk("../secret.txt", "a", "b"),
                    PatchHunk("src/test/java/FooTest.java", "0", "1"),
                    PatchHunk("src/main/java/Foo.java", "class Foo {}", "class Foo {}"),
                ]
            )
            self.assertFalse(result.ok)
            self.assertTrue(result.unsafe)
            self.assertEqual(source_file.read_text(encoding="utf-8"), "class Foo {}\n")

    def test_applies_unique_whitespace_normalized_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src/main/java/Foo.java"
            src.parent.mkdir(parents=True)
            src.write_text("class Foo {\n  int x() {\n    return 0;\n  }\n}\n", encoding="utf-8")
            applier = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"])
            result = applier.apply([PatchHunk("src/main/java/Foo.java", "int x(){return 0;}", "int x() { return 1; }")])
            self.assertTrue(result.ok)
            self.assertIn("return 1", src.read_text(encoding="utf-8"))

    def test_rejects_ambiguous_whitespace_normalized_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src/main/java/Foo.java"
            src.parent.mkdir(parents=True)
            src.write_text("class Foo { int x(){return 0;} int y(){return 0;} }\n", encoding="utf-8")
            applier = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"])
            result = applier.apply([PatchHunk("src/main/java/Foo.java", "return    0;", "return 1;")])
            self.assertFalse(result.ok)
            self.assertIn("old text not found", result.errors[0])

    def test_rejects_duplicate_replacement_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src/main/java/Foo.java"
            src.parent.mkdir(parents=True)
            src.write_text("class Foo { void x() { guard(); target(); } }\n", encoding="utf-8")
            applier = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"])
            result = applier.apply([PatchHunk("src/main/java/Foo.java", "target();", "guard();")])
            self.assertFalse(result.ok)
            self.assertIn("already exists", result.errors[0])

    def test_rejects_ambiguous_exact_patch_without_line_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src/main/java/Foo.java"
            src.parent.mkdir(parents=True)
            src.write_text(
                "class Foo {\n"
                "  int a() {\n"
                "    return value;\n"
                "  }\n"
                "  int b() {\n"
                "    return value;\n"
                "  }\n"
                "}\n",
                encoding="utf-8",
            )
            applier = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"])
            result = applier.apply([PatchHunk("src/main/java/Foo.java", "    return value;\n", "    return value + 1;\n")])
            self.assertFalse(result.ok)
            self.assertIn("old text not found", result.errors[0])

    def test_line_anchor_disambiguates_repeated_exact_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src/main/java/Foo.java"
            src.parent.mkdir(parents=True)
            src.write_text(
                "class Foo {\n"
                "  int a() {\n"
                "    return value;\n"
                "  }\n"
                "  int b() {\n"
                "    return value;\n"
                "  }\n"
                "}\n",
                encoding="utf-8",
            )
            hunk = PatchHunk(
                "src/main/java/Foo.java",
                "    return value;\n",
                "    return value + 1;\n",
                line_start=6,
                line_end=6,
            )
            result = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"]).apply([hunk])
            self.assertTrue(result.ok)
            patched = src.read_text(encoding="utf-8")
            self.assertIn("  int a() {\n    return value;\n  }", patched)
            self.assertIn("  int b() {\n    return value + 1;\n  }", patched)

    def test_bad_line_anchor_falls_back_to_unique_normalized_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src/main/java/Foo.java"
            src.parent.mkdir(parents=True)
            src.write_text("class Foo {\n  int x() {\n    return 0;\n  }\n}\n", encoding="utf-8")
            result = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"]).apply(
                [PatchHunk("src/main/java/Foo.java", "int x(){return 0;}", "int x() { return 1; }", line_start=1, line_end=1)]
            )
            self.assertTrue(result.ok)
            self.assertIn("return 1", src.read_text(encoding="utf-8"))

    def test_near_line_anchor_disambiguates_short_repeated_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src/main/java/Foo.java"
            src.parent.mkdir(parents=True)
            src.write_text(
                "class Foo {\n"
                "  void a() {\n"
                "    while (true) {\n"
                "      break;\n"
                "    }\n"
                "  }\n"
                "  void b() {\n"
                "    while (true) {\n"
                "      break;\n"
                "    }\n"
                "  }\n"
                "}\n",
                encoding="utf-8",
            )
            result = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"]).apply(
                [PatchHunk("src/main/java/Foo.java", "      break;\n", "      continue;\n", line_start=9, line_end=9)]
            )
            self.assertTrue(result.ok)
            patched = src.read_text(encoding="utf-8")
            self.assertIn("  void a() {\n    while (true) {\n      break;\n    }\n  }", patched)
            self.assertIn("  void b() {\n    while (true) {\n      continue;\n    }\n  }", patched)

    def test_unique_exact_text_beats_inaccurate_line_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src/main/java/Foo.java"
            src.parent.mkdir(parents=True)
            old = (
                "  int x() {\n"
                "    if (value > 0) {\n"
                "      return value;\n"
                "    }\n"
                "  }\n"
            )
            src.write_text("class Foo {\n" + old + "}\n", encoding="utf-8")
            new = (
                "  int x() {\n"
                "    return value;\n"
                "  }\n"
            )
            result = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"]).apply(
                [PatchHunk("src/main/java/Foo.java", old, new, line_start=2, line_end=4)]
            )
            self.assertTrue(result.ok)
            patched = src.read_text(encoding="utf-8")
            self.assertIn("  int x() {\n    return value;\n  }\n}", patched)
            self.assertNotIn("}\n  }\n}", patched)

    def test_rejects_java_brace_imbalance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src/main/java/Foo.java"
            src.parent.mkdir(parents=True)
            original = "class Foo {\n  int x() {\n    return 0;\n  }\n}\n"
            src.write_text(original, encoding="utf-8")
            result = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"]).apply(
                [PatchHunk("src/main/java/Foo.java", "    return 0;\n", "    return 1;\n  }\n")]
            )
            self.assertFalse(result.ok)
            self.assertIn("java brace imbalance", result.errors[0])
            self.assertEqual(src.read_text(encoding="utf-8"), original)

    def test_allows_multi_hunk_when_final_java_braces_balance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src/main/java/Foo.java"
            src.parent.mkdir(parents=True)
            src.write_text("class Foo {\n  int x() {\n    return 0;\n  }\n}\n", encoding="utf-8")
            result = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"]).apply(
                [
                    PatchHunk("src/main/java/Foo.java", "  int x() {\n", "  int x() {\n    if (true) {\n"),
                    PatchHunk("src/main/java/Foo.java", "    return 0;\n", "      return 1;\n    }\n"),
                ]
            )
            self.assertTrue(result.ok)
            self.assertIn("if (true)", src.read_text(encoding="utf-8"))

    def test_applies_unique_similar_line_window_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src/main/java/Foo.java"
            src.parent.mkdir(parents=True)
            src.write_text(
                "class Foo {\n"
                "  Items getItems() {\n"
                "    Items result = new Items();\n"
                "    Dataset dataset = plot.getDataset();\n"
                "    if (dataset != null) {\n"
                "      return result;\n"
                "    }\n"
                "    int count = dataset.getRowCount();\n"
                "    return build(count);\n"
                "  }\n"
                "}\n",
                encoding="utf-8",
            )
            old = (
                "    Dataset dataset = plot.getDataset();\n"
                "    if (dataset == null) {\n"
                "      return result;\n"
                "    }\n"
                "    int count = this.rowCount;"
            )
            new = (
                "    Dataset dataset = plot.getDataset();\n"
                "    if (dataset == null) {\n"
                "      return result;\n"
                "    }\n"
                "    int count = dataset.getRowCount();"
            )
            applier = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"])
            result = applier.apply([PatchHunk("src/main/java/Foo.java", old, new)])
            self.assertTrue(result.ok)
            patched = src.read_text(encoding="utf-8")
            self.assertIn("if (dataset == null)", patched)
            self.assertIn("int count = dataset.getRowCount();", patched)

    def test_rejects_ambiguous_similar_line_window_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src/main/java/Foo.java"
            src.parent.mkdir(parents=True)
            block = (
                "    Dataset dataset = plot.getDataset();\n"
                "    if (dataset != null) {\n"
                "      return result;\n"
                "    }\n"
                "    int count = dataset.getRowCount();\n"
            )
            src.write_text(f"class Foo {{\n  void a() {{\n{block}  }}\n  void b() {{\n{block}  }}\n}}\n", encoding="utf-8")
            old = (
                "    Dataset dataset = plot.getDataset();\n"
                "    if (dataset == null) {\n"
                "      return result;\n"
                "    }\n"
                "    int count = this.rowCount;"
            )
            new = old.replace("this.rowCount", "dataset.getRowCount()")
            applier = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"])
            result = applier.apply([PatchHunk("src/main/java/Foo.java", old, new)])
            self.assertFalse(result.ok)
            self.assertIn("old text not found", result.errors[0])

    def test_rejects_very_large_unanchored_fuzzy_patch_without_slow_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src/main/java/Foo.java"
            src.parent.mkdir(parents=True)
            src.write_text("class Foo {\n" + "\n".join(f"  int v{i} = {i};" for i in range(800)) + "\n}\n", encoding="utf-8")
            old = "\n".join(f"  int missing{i} = {i};" for i in range(500))
            result = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"]).apply(
                [PatchHunk("src/main/java/Foo.java", old, "  int replacement = 1;")]
            )
            self.assertFalse(result.ok)
            self.assertIn("old text not found", result.errors[0])

    def test_applies_line_anchored_patch_when_old_text_is_similar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src/main/java/Foo.java"
            src.parent.mkdir(parents=True)
            src.write_text(
                "class Foo {\n"
                "  int x() {\n"
                "    int value = source();\n"
                "    return value;\n"
                "  }\n"
                "}\n",
                encoding="utf-8",
            )
            hunk = PatchHunk(
                "src/main/java/Foo.java",
                "    int value = otherSource();\n    return value;",
                "    int value = source();\n    return value + 1;",
                line_start=3,
                line_end=4,
            )
            result = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"]).apply([hunk])
            self.assertTrue(result.ok)
            self.assertIn("return value + 1", src.read_text(encoding="utf-8"))

    def test_rejects_line_anchored_patch_when_line_range_does_not_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src/main/java/Foo.java"
            src.parent.mkdir(parents=True)
            src.write_text("class Foo {\n  int x() { return 0; }\n}\n", encoding="utf-8")
            hunk = PatchHunk(
                "src/main/java/Foo.java",
                "totally unrelated old text that should not match",
                "return 1;",
                line_start=2,
                line_end=2,
            )
            result = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"]).apply([hunk])
            self.assertFalse(result.ok)

    def test_range_hunk_from_dict_preserves_intent_and_offset(self):
        hunk = RangeHunk.from_dict(
            {
                "file": "src/main/java/Foo.java",
                "line_start": "3",
                "line_end": 5,
                "new": "fixed();\n",
                "method_name": "target",
                "intent": "fix boundary",
                "line_offset": "2",
            }
        )
        self.assertEqual(hunk.file, "src/main/java/Foo.java")
        self.assertEqual(hunk.line_start, 3)
        self.assertEqual(hunk.line_end, 5)
        self.assertEqual(hunk.method_name, "target")
        self.assertEqual(hunk.intent, "fix boundary")
        self.assertEqual(hunk.line_offset, 2)

    def test_patch_hunk_from_range_slices_exact_source_lines(self):
        text = "line1\nline2\nline3\nline4\n"
        hunk = PatchHunk.from_range("src/main/java/Foo.java", 2, 3, "replacement\n", text)
        self.assertEqual(hunk.old, "line2\nline3\n")
        self.assertEqual(hunk.line_start, 2)
        self.assertEqual(hunk.line_end, 3)
        self.assertEqual(hunk.original_line_start, 2)
        self.assertTrue(hunk.range_grounded)

    def test_patch_hunk_from_range_rejects_out_of_bounds_range(self):
        with self.assertRaises(ValueError):
            PatchHunk.from_range("src/main/java/Foo.java", 1, 99, "replacement\n", "line1\n")

    def test_patch_hunk_from_range_rejects_empty_slice(self):
        with self.assertRaises(ValueError):
            PatchHunk.from_range("src/main/java/Foo.java", 1, 1, "replacement\n", "\n")

    def test_patch_hunk_from_range_accepts_method_scoped_slice(self):
        text = (
            "class Foo {\n"
            "  int target() {\n"
            "    return 0;\n"
            "  }\n"
            "}\n"
        )
        hunk = PatchHunk.from_range("src/main/java/Foo.java", 3, 3, "    return 1;\n", text, method_name="target")
        self.assertEqual(hunk.old, "    return 0;\n")
        self.assertEqual(hunk.method_name, "target")

    def test_patch_hunk_from_range_allows_slice_outside_named_method_with_warning(self):
        text = (
            "class Foo {\n"
            "  int target() {\n"
            "    return 0;\n"
            "  }\n"
            "  int other() { return 2; }\n"
            "}\n"
        )
        # Method name is a hint; if the line range is outside the named method,
        # grounding should still succeed using the verified line range.
        hunk = PatchHunk.from_range("src/main/java/Foo.java", 5, 5, "  int other() { return 3; }\n", text, method_name="target")
        self.assertEqual(hunk.old, "  int other() { return 2; }\n")
        self.assertTrue(hunk.range_grounded)

    def test_range_grounded_patch_uses_exact_line_span_for_duplicate_old_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src/main/java/Foo.java"
            src.parent.mkdir(parents=True)
            text = "class Foo {\n  int a() {\n    return value;\n  }\n  int b() {\n    return value;\n  }\n}\n"
            src.write_text(text, encoding="utf-8")
            hunk = PatchHunk.from_range("src/main/java/Foo.java", 6, 6, "    return value + 1;\n", text)
            result = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"]).apply([hunk])
            self.assertTrue(result.ok)
            patched = src.read_text(encoding="utf-8")
            self.assertIn("  int a() {\n    return value;\n  }", patched)
            self.assertIn("  int b() {\n    return value + 1;\n  }", patched)

    def test_range_grounded_multi_hunk_applies_bottom_up_without_line_shift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src/main/java/Foo.java"
            src.parent.mkdir(parents=True)
            text = "class Foo {\n  int a = 0;\n  int b = 0;\n  int c = 0;\n}\n"
            src.write_text(text, encoding="utf-8")
            lower = PatchHunk.from_range("src/main/java/Foo.java", 2, 2, "  int a = 1;\n  int inserted = 9;\n", text)
            higher = PatchHunk.from_range("src/main/java/Foo.java", 4, 4, "  int c = 3;\n", text)
            result = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"]).apply([lower, higher])
            self.assertTrue(result.ok, result.errors)
            patched = src.read_text(encoding="utf-8")
            self.assertIn("int inserted = 9", patched)
            self.assertIn("int c = 3", patched)

    def test_range_grounded_overlapping_hunks_rejected_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src/main/java/Foo.java"
            src.parent.mkdir(parents=True)
            text = "class Foo {\n  int a = 0;\n  int b = 0;\n}\n"
            src.write_text(text, encoding="utf-8")
            first = PatchHunk.from_range("src/main/java/Foo.java", 2, 3, "  int both = 1;\n", text)
            second = PatchHunk.from_range("src/main/java/Foo.java", 3, 3, "  int b = 2;\n", text)
            result = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"]).apply([first, second])
            self.assertFalse(result.ok)
            self.assertIn("overlapping patch ranges", "\n".join(result.errors))
            self.assertEqual(src.read_text(encoding="utf-8"), text)

    def test_patch_hunk_from_range_applies_line_offset_to_current_checkout(self):
        original_range_start = 1
        current_text = "package pkg;\nclass Foo {\n  int value() { return 0; }\n}\n"
        hunk = PatchHunk.from_range(
            "src/main/java/Foo.java",
            original_range_start,
            original_range_start,
            "class Foo {\n",
            current_text,
            line_offset=1,
        )
        self.assertEqual(hunk.old, "class Foo {\n")
        self.assertEqual(hunk.line_start, 2)
        self.assertEqual(hunk.original_line_start, 1)
        self.assertEqual(hunk.line_offset, 1)


class RangeGroundedPatchTest(unittest.TestCase):
    def test_range_hunk_records_location_new_text_method_and_intent(self):
        from code_repair_agent.safe_patch import RangeHunk

        hunk = RangeHunk(
            file="src/main/java/Foo.java",
            line_start=7,
            line_end=9,
            new="    return fixed;\n",
            method_name="compute",
            intent="replace buggy denominator update",
        )
        self.assertEqual(hunk.file, "src/main/java/Foo.java")
        self.assertEqual(hunk.line_start, 7)
        self.assertEqual(hunk.line_end, 9)
        self.assertEqual(hunk.new, "    return fixed;\n")
        self.assertEqual(hunk.method_name, "compute")
        self.assertEqual(hunk.intent, "replace buggy denominator update")

    def test_from_range_slices_exact_one_based_inclusive_lines(self):
        source = (
            "class Foo {\n"
            "  int value() {\n"
            "    int x = 1;\n"
            "    return x;\n"
            "  }\n"
            "}\n"
        )
        hunk = PatchHunk.from_range(
            "src/main/java/Foo.java",
            line_start=3,
            line_end=4,
            new="    return 2;\n",
            current_text=source,
        )
        self.assertIsNotNone(hunk)
        assert hunk is not None
        self.assertEqual(hunk.old, "    int x = 1;\n    return x;\n")
        self.assertEqual(hunk.new, "    return 2;\n")
        self.assertEqual(hunk.line_start, 3)
        self.assertEqual(hunk.line_end, 4)

    def test_from_range_rejects_invalid_or_empty_ranges(self):
        source = "class Foo {\n\n}\n"
        invalid_ranges = [(0, 1), (2, 1), (1, 99), (2, 2)]
        for line_start, line_end in invalid_ranges:
            with self.subTest(line_start=line_start, line_end=line_end):
                try:
                    hunk = PatchHunk.from_range(
                        "src/main/java/Foo.java",
                        line_start=line_start,
                        line_end=line_end,
                        new="replacement\n",
                        current_text=source,
                    )
                except ValueError:
                    hunk = None
                self.assertIsNone(hunk)

    def test_from_range_enforces_method_scope_when_provided(self):
        source = (
            "class Foo {\n"
            "  public int target() {\n"
            "    return 1;\n"
            "  }\n"
            "  public int other() {\n"
            "    return 2;\n"
            "  }\n"
            "}\n"
        )
        hunk = PatchHunk.from_range(
            "src/main/java/Foo.java",
            line_start=3,
            line_end=3,
            new="    return 10;\n",
            current_text=source,
            method_name="target",
        )
        self.assertIsNotNone(hunk)
        assert hunk is not None
        self.assertEqual(hunk.method_name, "target")
        self.assertEqual(hunk.old, "    return 1;\n")
        # Method name is a hint; out-of-scope ranges still ground using
        # the verified line range rather than failing.
        out_of_scope = PatchHunk.from_range(
            "src/main/java/Foo.java",
            line_start=6,
            line_end=6,
            new="    return 20;\n",
            current_text=source,
            method_name="target",
        )
        self.assertIsNotNone(out_of_scope)
        self.assertEqual(out_of_scope.old, "    return 2;\n")
        self.assertTrue(out_of_scope.range_grounded)

    def test_range_hunk_applies_at_requested_span_even_when_old_text_repeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src/main/java/Foo.java"
            src.parent.mkdir(parents=True)
            source = (
                "class Foo {\n"
                "  int a() {\n"
                "    return value;\n"
                "  }\n"
                "  int b() {\n"
                "    return value;\n"
                "  }\n"
                "}\n"
            )
            src.write_text(source, encoding="utf-8")
            hunk = PatchHunk.from_range(
                "src/main/java/Foo.java",
                line_start=6,
                line_end=6,
                new="    return value + 1;\n",
                current_text=source,
            )
            self.assertIsNotNone(hunk)
            assert hunk is not None
            result = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"]).apply([hunk])
            self.assertTrue(result.ok, result.errors)
            patched = src.read_text(encoding="utf-8")
            self.assertIn("  int a() {\n    return value;\n  }", patched)
            self.assertIn("  int b() {\n    return value + 1;\n  }", patched)

    def test_range_hunks_apply_same_file_in_descending_original_line_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src/main/java/Foo.java"
            src.parent.mkdir(parents=True)
            source = (
                "class Foo {\n"
                "  int a() { return 0; }\n"
                "  int keep = 7;\n"
                "  int b() { return 0; }\n"
                "}\n"
            )
            src.write_text(source, encoding="utf-8")
            hunks = [
                PatchHunk.from_range(
                    "src/main/java/Foo.java",
                    line_start=2,
                    line_end=2,
                    new="  int a() {\n    return 1;\n  }\n",
                    current_text=source,
                ),
                PatchHunk.from_range(
                    "src/main/java/Foo.java",
                    line_start=4,
                    line_end=4,
                    new="  int b() { return 2; }\n",
                    current_text=source,
                ),
            ]
            self.assertTrue(all(hunk is not None for hunk in hunks))
            hunks = [hunk for hunk in hunks if hunk is not None]
            result = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"]).apply(hunks)
            self.assertTrue(result.ok, result.errors)
            patched = src.read_text(encoding="utf-8")
            self.assertIn("  int a() {\n    return 1;\n  }", patched)
            self.assertIn("  int keep = 7;\n  int b() { return 2; }", patched)

    def test_range_hunks_reject_overlapping_original_spans(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src/main/java/Foo.java"
            src.parent.mkdir(parents=True)
            source = "class Foo {\n  int a = 1;\n  int b = 2;\n  int c = 3;\n}\n"
            src.write_text(source, encoding="utf-8")
            hunks = [
                PatchHunk.from_range("src/main/java/Foo.java", 2, 3, "  int ab = 12;\n", source),
                PatchHunk.from_range("src/main/java/Foo.java", 3, 4, "  int bc = 23;\n", source),
            ]
            self.assertTrue(all(hunk is not None for hunk in hunks))
            hunks = [hunk for hunk in hunks if hunk is not None]
            result = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"]).apply(hunks)
            self.assertFalse(result.ok)
            self.assertTrue(any("overlap" in error.lower() for error in result.errors), result.errors)
            self.assertEqual(src.read_text(encoding="utf-8"), source)

    def test_multi_file_range_patch_rolls_back_when_any_hunk_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "src/main/java/Foo.java"
            second = root / "src/main/java/Bar.java"
            first.parent.mkdir(parents=True)
            first_source = "class Foo {\n  int value() { return 0; }\n}\n"
            second_source = "class Bar {\n  int value() { return 0; }\n}\n"
            first.write_text(first_source, encoding="utf-8")
            second.write_text(second_source, encoding="utf-8")
            hunks = [
                PatchHunk.from_range("src/main/java/Foo.java", 2, 2, "  int value() { return 1; }\n", first_source),
                PatchHunk("src/main/java/Bar.java", "missing exact old", "  int value() { return 2; }\n", line_start=2, line_end=2),
            ]
            self.assertIsNotNone(hunks[0])
            hunks = [hunk for hunk in hunks if hunk is not None]
            result = SafePatchApplier(root, source_dirs=["src/main/java"], test_dirs=["src/test/java"]).apply(hunks)
            self.assertFalse(result.ok)
            self.assertEqual(first.read_text(encoding="utf-8"), first_source)
            self.assertEqual(second.read_text(encoding="utf-8"), second_source)

    def test_parse_repair_plan_accepts_range_hunk_without_old_text(self):
        raw = json.dumps(
            {
                "diagnosis": "off by one",
                "patch_hunks": [
                    {
                        "file": "src/main/java/Foo.java",
                        "line_start": 12,
                        "line_end": 14,
                        "new": "    return max + 1;\n",
                        "method_name": "compute",
                        "intent": "fix inclusive upper bound",
                    }
                ],
                "patch_style": "range-grounded-boundary",
            }
        )
        plan = parse_repair_plan(raw)
        self.assertEqual(plan.patch_hunks[0].file, "src/main/java/Foo.java")
        self.assertEqual(plan.patch_hunks[0].old, "")
        self.assertEqual(plan.patch_hunks[0].line_start, 12)
        self.assertEqual(plan.patch_hunks[0].line_end, 14)
        self.assertEqual(plan.patch_hunks[0].method_name, "compute")


class DeepSeekParserTest(unittest.TestCase):
    def test_parse_valid_fenced_json(self):
        raw = """```json
        {
          "diagnosis": "boundary bug",
          "files_to_read": ["src/main/java/Foo.java"],
          "patch_hunks": [{"file": "src/main/java/Foo.java", "old": "return 0", "new": "return 1", "line_start": 10, "line_end": 10}],
          "tests_to_run_next": ["trigger"],
          "confidence": 0.7,
          "final_explanation": "fix boundary",
          "patch_style": "boundary"
        }
        ```"""
        plan = parse_repair_plan(raw)
        self.assertEqual(plan.patch_style, "boundary")
        self.assertEqual(plan.patch_hunks[0].file, "src/main/java/Foo.java")
        self.assertEqual(plan.patch_hunks[0].line_start, 10)

    def test_rejects_missing_patch_hunks(self):
        with self.assertRaises(ValueError):
            parse_repair_plan(json.dumps({"diagnosis": "no patch"}))

    def test_allow_empty_patch_for_read_request(self):
        plan = parse_repair_plan(
            json.dumps(
                {
                    "diagnosis": "need exact file",
                    "files_to_read": ["src/main/java/Foo.java"],
                    "patch_hunks": [],
                    "patch_style": "read-request",
                }
            ),
            allow_empty=True,
        )
        self.assertEqual(plan.files_to_read, ["src/main/java/Foo.java"])
        self.assertEqual(plan.patch_hunks, [])

    def test_parse_prefers_final_json_after_thinking_draft(self):
        raw = (
            '{"diagnosis": "draft", "patch_hunks": [{"file": "src/main/java/Foo.java", "old": "bad\\n'
            'broken", "new": "x"}]}\n'
            '<｜end▁of▁thinking｜>\n'
            '{"diagnosis": "final", "patch_hunks": [{"file": "src/main/java/Foo.java", '
            '"old": "return 0", "new": "return 1"}], "patch_style": "boundary"}'
        )
        plan = parse_repair_plan(raw)
        self.assertEqual(plan.diagnosis, "final")
        self.assertEqual(plan.patch_hunks[0].old, "return 0")

    def test_parse_repairs_raw_newlines_inside_json_strings(self):
        raw = (
            '{\n'
            '  "diagnosis": "newline in patch text",\n'
            '  "patch_hunks": [{\n'
            '    "file": "src/main/java/Foo.java",\n'
            '    "old": "int x() {\n'
            '      return 0;\n'
            '    }",\n'
            '    "new": "int x() {\n'
            '      return 1;\n'
            '    }"\n'
            '  }],\n'
            '  "patch_style": "boundary"\n'
            '}\n'
        )
        plan = parse_repair_plan(raw)
        self.assertIn("return 0", plan.patch_hunks[0].old)
        self.assertIn("return 1", plan.patch_hunks[0].new)

    def test_parse_salvages_complete_patch_hunks_from_truncated_tail(self):
        raw = (
            '{\n'
            '  "diagnosis": "truncated after useful patch",\n'
            '  "files_to_read": ["src/main/java/Foo.java"],\n'
            '  "patch_hunks": [\n'
            '    {"file": "src/main/java/Foo.java", "old": "return 0", "new": "return 1"}\n'
            '  ],\n'
            '  "tests_to_run_next": ["pkg.FooTest::testBug", "pkg.FooTest::testOther"\n'
        )
        plan = parse_repair_plan(raw)
        self.assertEqual(plan.diagnosis, "truncated after useful patch")
        self.assertEqual(plan.patch_hunks[0].old, "return 0")
        self.assertEqual(plan.files_to_read, ["src/main/java/Foo.java"])

    def test_parse_delimiter_wrapped_json(self):
        raw = """<<<PATCH_JSON>>>
        {
          "diagnosis": "boundary bug via delimiter",
          "patch_hunks": [{"file": "src/Foo.java", "old": "return 0", "new": "return 1"}],
          "patch_style": "boundary"
        }
        <<<END_PATCH_JSON>>>"""
        plan = parse_repair_plan(raw)
        self.assertEqual(plan.diagnosis, "boundary bug via delimiter")
        self.assertEqual(plan.patch_style, "boundary")
        self.assertEqual(plan.patch_hunks[0].old, "return 0")

    def test_parse_trailing_comma_in_array(self):
        raw = """{
          "diagnosis": "test",
          "patch_hunks": [
            {"file": "src/Foo.java", "old": "x", "new": "y"},
          ],
          "patch_style": "direct"
        }"""
        plan = parse_repair_plan(raw)
        self.assertEqual(plan.patch_hunks[0].old, "x")

    def test_parse_trailing_comma_after_last_field(self):
        raw = """{
          "diagnosis": "test",
          "patch_hunks": [{"file": "src/Foo.java", "old": "x", "new": "y"}],
          "patch_style": "direct",
        }"""
        plan = parse_repair_plan(raw)
        self.assertEqual(plan.patch_hunks[0].old, "x")

    def test_parse_trailing_comma_and_escaped_newlines(self):
        raw = '{\n  "diagnosis": "test",\n  "patch_hunks": [{"file": "src/Foo.java", "old": "line1\\nline2", "new": "fixed"}],\n  "patch_style": "guard",\n}'
        plan = parse_repair_plan(raw)
        self.assertIn("line1", plan.patch_hunks[0].old)

    def test_parse_method_name_and_anchors_in_hunk(self):
        raw = json.dumps({
            "diagnosis": "needs method anchoring",
            "patch_hunks": [{
                "file": "src/Foo.java",
                "old": "return 0",
                "new": "return 1",
                "method_name": "calculate",
                "anchor_before": "int x = 5;",
                "anchor_after": "System.out.println(x);"
            }],
            "patch_style": "boundary"
        })
        plan = parse_repair_plan(raw)
        self.assertEqual(plan.patch_hunks[0].file, "src/Foo.java")

    def test_parse_localization_plan_extracts_ranges_and_intent(self):
        raw = json.dumps(
            {
                "files_to_read": ["src/main/java/Foo.java"],
                "line_ranges": [
                    {
                        "file": "src/main/java/Foo.java",
                        "line_start": 10,
                        "line_end": 14,
                        "method_name": "target",
                        "intent": "fix boundary",
                    }
                ],
                "hypothesis": "off by one",
                "patch_intent": "adjust upper bound",
            }
        )
        plan = parse_localization_plan(raw)
        self.assertEqual(plan.files_to_read, ["src/main/java/Foo.java"])
        self.assertEqual(plan.line_ranges[0].line_start, 10)
        self.assertEqual(plan.line_ranges[0].line_end, 14)
        self.assertEqual(plan.line_ranges[0].method_name, "target")
        self.assertEqual(plan.line_ranges[0].intent, "fix boundary")
        self.assertEqual(plan.hypothesis, "off by one")

    def test_parse_localization_plan_rejects_bad_range(self):
        raw = json.dumps(
            {
                "line_ranges": [
                    {"file": "src/main/java/Foo.java", "line_start": 20, "line_end": 10, "method_name": "target"}
                ],
                "hypothesis": "bad range",
            }
        )
        with self.assertRaises(ValueError):
            parse_localization_plan(raw)

    def test_build_localization_prompt_schema_excludes_old_new_text(self):
        messages = build_localization_prompt(
            project="Lang",
            bug_id=1,
            metadata={"classes.modified": "pkg.Foo"},
            failing_output="AssertionError expected:<1> but was:<0>",
            snippets={"src/main/java/pkg/Foo.java": "class Foo {}"},
            snippet_line_numbers={"src/main/java/pkg/Foo.java": "1: class Foo {}"},
        )
        user = json.loads(messages[1]["content"])
        schema_range = user["required_json_schema"]["line_ranges"][0]
        self.assertIn("line_start", schema_range)
        self.assertIn("line_end", schema_range)
        self.assertNotIn("old", schema_range)
        self.assertNotIn("new", schema_range)
        self.assertIn("Do not include old/new", user["localization_instructions"])

    def test_build_repair_prompt_prefers_range_hunks_without_old_in_schema(self):
        messages = build_repair_prompt(
            project="Lang",
            bug_id=1,
            metadata={"classes.modified": "pkg.Foo"},
            failing_output="AssertionError",
            snippets={"src/main/java/pkg/Foo.java": "class Foo {}"},
            snippet_line_numbers={"src/main/java/pkg/Foo.java": "1: class Foo {}"},
            current_diff="",
            attempt=1,
            memory_preferences=[],
        )
        self.assertIn("Prefer exact old/new replacement", messages[0]["content"])
        user = json.loads(messages[1]["content"])
        schema_hunk = user["required_json_schema"]["patch_hunks"][0]
        self.assertEqual(["file", "old", "new", "line_start", "line_end", "method_name", "intent", "anchor_before", "anchor_after"], list(schema_hunk.keys()))
        self.assertIn("Prefer exact old/new replacement", user["patch_grounding_instructions"])

    def test_parse_repair_plan_accepts_range_only_hunk_for_preflight(self):
        raw = json.dumps(
            {
                "diagnosis": "range grounded fix",
                "patch_hunks": [
                    {
                        "file": "src/main/java/Foo.java",
                        "line_start": 3,
                        "line_end": 3,
                        "new": "    return 1;\n",
                        "method_name": "value",
                        "intent": "fix return value",
                    }
                ],
                "patch_style": "boundary",
            }
        )
        plan = parse_repair_plan(raw)
        hunk = plan.patch_hunks[0]
        self.assertEqual(hunk.old, "")
        self.assertEqual(hunk.line_start, 3)
        self.assertEqual(hunk.method_name, "value")
        self.assertEqual(hunk.intent, "fix return value")


class BenchmarkMemoryTest(unittest.TestCase):
    def test_memory_ranks_successful_patch_styles(self):
        memory = BenchmarkMemory()
        features = ["project:Lang", "exception:assertionerror"]
        memory.update(
            features=features,
            patch_style="boundary",
            test_scope="trigger",
            solved=True,
            failure_reason=None,
            reflection=None,
        )
        memory.update(
            features=features,
            patch_style="null-check",
            test_scope="all",
            solved=False,
            failure_reason="visible_failure",
            reflection="avoid unrelated null check",
        )
        self.assertEqual(memory.prompt_preferences(features)[0], "boundary")
        self.assertEqual(memory.preferred_test_scope(features), "trigger")
        self.assertIn("trigger tests", memory.relevant_reflections(features)[0])

    def test_regression_failure_penalizes_visible_only_strategy(self):
        memory = BenchmarkMemory()
        features = ["project:Math", "exception:assertionerror"]
        memory.update(
            features=features,
            patch_style="boundary",
            test_scope="relevant",
            solved=False,
            failure_reason="regression_failure",
            reflection="visible passed but relevant regression failed",
            repair_skill="repair-after-regression",
            test_skill="regression-relevant-caught-overfit",
            visible_passed=True,
            regression_checked=True,
            regression_passed=False,
        )
        warnings = memory.regression_warnings(features)
        self.assertIn("style:boundary|scope:relevant", warnings[0])
        self.assertEqual(memory.repair_skill_preferences(features)[0], "repair-after-regression")
        self.assertEqual(memory.test_skill_preferences(features)[0], "regression-relevant-caught-overfit")

    def test_memory_update_can_isolate_check_dimension(self):
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
            update_check_memory=True,
            update_repair_memory=False,
        )
        self.assertEqual(memory.preferred_test_scope(features), "relevant")
        self.assertEqual(memory.test_skill_preferences(features), ["run-relevant"])
        self.assertEqual(memory.prompt_preferences(features), [])
        self.assertEqual(memory.repair_skill_preferences(features), [])

    def test_memory_update_can_isolate_repair_dimension(self):
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
        self.assertEqual(memory.preferred_test_scope(features), "trigger")
        self.assertEqual(memory.test_skill_preferences(features), [])
        self.assertEqual(memory.prompt_preferences(features), ["regex-boundary"])
        self.assertEqual(memory.repair_skill_preferences(features), ["use-regex-boundary"])
        self.assertIn("regex boundary", memory.relevant_success_strategies(features)[0])

    def test_attempt_bonus_uses_transferable_memory(self):
        memory = BenchmarkMemory()
        features = ["project:Lang", "class:StringUtils"]
        for _ in range(3):
            memory.update(
                features=features,
                patch_style="api-contract",
                test_scope="trigger",
                solved=True,
                failure_reason=None,
                reflection=None,
                repair_skill="retry-after-feedback",
                test_skill="trigger-first",
            )
        self.assertGreaterEqual(memory.attempt_bonus(features, max_bonus=2), 1)

    def test_preferred_test_scope_ignores_negative_and_invalid_scopes(self):
        memory = BenchmarkMemory()
        features = ["project:Lang"]
        memory.update(
            features=features,
            patch_style="invalid-response",
            test_scope="not-run",
            solved=False,
            failure_reason="parse_error",
            reflection="bad response",
        )
        memory.update(
            features=features,
            patch_style="boundary",
            test_scope="relevant",
            solved=False,
            failure_reason="regression_failure",
            reflection="regression failed",
            visible_passed=True,
            regression_checked=True,
            regression_passed=False,
        )
        self.assertEqual(memory.preferred_test_scope(features, default="relevant"), "relevant")

    def test_success_strategy_memory_is_retrieved_for_similar_features(self):
        memory = BenchmarkMemory()
        features = ["project:Lang", "class:numberutils"]
        memory.update(
            features=features,
            patch_style="boundary",
            test_scope="relevant",
            solved=True,
            failure_reason=None,
            reflection=None,
            success_strategy="successful strategy: try narrow numeric type then fallback and run relevant regression",
        )
        self.assertIn("try narrow numeric", memory.relevant_success_strategies(["class:numberutils"])[0])

    def test_patch_apply_failure_reflection_is_sanitized(self):
        memory = BenchmarkMemory()
        memory.failure_reflections.append(
            {
                "features": "project:Chart,class:renderer",
                "failure_reason": "patch_apply_failure",
                "reflection": (
                    "patch_apply_failure: diagnosis=rowCount is the root cause; "
                    "detail=source/File.java: old text not found"
                ),
            }
        )
        reflection = memory.relevant_reflections(["class:renderer"])[0]
        self.assertIn("old text did not match", reflection)
        self.assertNotIn("rowCount is the root cause", reflection)

    def test_visible_failure_reflection_is_sanitized(self):
        memory = BenchmarkMemory()
        memory.failure_reflections.append(
            {
                "features": "project:Chart,class:renderer",
                "failure_reason": "visible_failure",
                "reflection": "visible_failure: diagnosis=null check is root cause",
            }
        )
        reflection = memory.relevant_reflections(["project:Chart"])[0]
        self.assertIn("trigger tests", reflection)
        self.assertNotIn("null check is root cause", reflection)

    def test_prompt_includes_regression_aware_memory_fields(self):
        messages = build_repair_prompt(
            project="Lang",
            bug_id=4,
            metadata={"tests.trigger": "FooTest::test"},
            failing_output="AssertionError",
            snippets={"src/main/java/Foo.java": "class Foo {}"},
            snippet_line_numbers={"src/main/java/Foo.java": "1: class Foo {}"},
            current_diff="",
            attempt=2,
            memory_preferences=["boundary"],
            visible_test_assertions=["assertEquals(Integer.valueOf(1), Foo.parse(\"1\"));"],
            derived_repair_constraints=["preserve both sides of numeric boundary"],
            repair_skills=["retry-after-feedback"],
            test_skills=["regression-relevant-caught-overfit"],
            regression_warnings=["style:boundary|scope:trigger has failed regression before"],
            success_strategies=["successful strategy: try narrow parser then fallback"],
            previous_attempt_failures=["attempt 1: patch_apply_failure: old text not found"],
            reflections=["avoid visible-only patch"],
        )
        user = json.loads(messages[1]["content"])
        self.assertEqual(user["failing_summary"], "AssertionError")
        self.assertEqual(user["memory_preferred_repair_skills"], ["retry-after-feedback"])
        self.assertEqual(user["memory_preferred_test_skills"], ["regression-relevant-caught-overfit"])
        self.assertIn("failed regression", user["memory_regression_warnings"][0])
        self.assertIn("narrow parser", user["memory_successful_strategies"][0])
        self.assertIn("old text not found", user["previous_attempt_failures"][0])
        self.assertEqual(user["source_snippet_line_numbers"]["src/main/java/Foo.java"], "1: class Foo {}")
        self.assertIn("Integer.valueOf(1)", user["visible_test_assertions"][0])
        self.assertIn("numeric boundary", user["derived_repair_constraints"][0])

    def test_scope_failure_summary_keeps_failing_test_header(self):
        output = (
            "long ant prelude\n"
            "\n[failing_tests]\n"
            "--- pkg.NumberUtilsTest::testPrecision\n"
            "junit.framework.AssertionFailedError\n"
            + "\n".join(f"at stack.Line{idx}" for idx in range(50))
        )
        scope = ScopeResult(
            scope="trigger",
            passed=False,
            results=[
                CommandResult(
                    command=["defects4j", "test", "-t", "pkg.NumberUtilsTest::testPrecision"],
                    cwd="/tmp",
                    returncode=0,
                    output=output,
                    elapsed_seconds=0.1,
                )
            ],
        )
        summary = _scope_failure_tail(scope, limit=120)
        self.assertIn("--- pkg.NumberUtilsTest::testPrecision", summary)
        self.assertIn("AssertionFailedError", summary)

    def test_visible_failure_guidance_blocks_repeated_numeric_type_strategy(self):
        guidance = _visible_failure_guidance(
            [
                "assertTrue(NumberUtils.createNumber(shouldBeFloat) instanceof Float);",
                "assertTrue(NumberUtils.createNumber(shouldBeDouble) instanceof Double);",
                "assertTrue(NumberUtils.createNumber(shouldBeBigDecimal) instanceof BigDecimal);",
            ],
            (
                "+ final Float f = createFloat(str);\n"
                "+ return f;\n"
                "  final Double d = createDouble(str);\n"
            ),
        )
        self.assertIn("numeric_type_selection_failed_strategy", guidance)
        self.assertIn("do not return Float solely", guidance)

    def test_failed_patch_feedback_keeps_critical_guidance_first(self):
        plan = RepairPlan(
            diagnosis="numeric type selection",
            patch_style="fallback",
            patch_hunks=[
                PatchHunk(
                    "src/main/java/Foo.java",
                    "return createDouble(str);",
                    "return createFloat(str);",
                )
            ],
        )
        result = PatchApplyResult(
            ok=True,
            unsafe=False,
            changed_files=["src/main/java/Foo.java"],
            diff="+ final Float f = createFloat(str);\n+ return f;\n  final Double d = createDouble(str);\n",
            errors=[],
        )
        feedback = _failed_patch_feedback(
            plan,
            result,
            "visible_failure",
            (
                "numeric_type_selection_failed_strategy: simple Float-before-Double patch compiled and failed. "
                "Do not repeat it. Next patch needs separate Float, Double, and BigDecimal precision/range guards.\n"
                + "stack\n" * 200
            ),
        )
        self.assertTrue(feedback.startswith("numeric_type_selection_failed_strategy"))
        self.assertIn("Do not repeat it", feedback[:240])

    def test_patch_strategy_signature_collapses_numeric_duplicates(self):
        first = (
            "+ final Float f = createFloat(str);\n"
            "+ if (!(f.isInfinite())) {\n"
            "+     return f;\n"
            "+ }\n"
            "  final Double d = createDouble(str);\n"
        )
        second = (
            "+final Float f = createFloat(str);\n"
            "+return f;\n"
            "+final Double d = createDouble(str);\n"
        )
        self.assertEqual(_patch_strategy_signature(first), "numeric-type:add-float-before-double")
        self.assertEqual(_patch_strategy_signature(first), _patch_strategy_signature(second))

    def test_duplicate_patch_feedback_includes_visible_guidance(self):
        feedback = _duplicate_patch_feedback(
            "numeric-type:add-float-before-double",
            "+ final Float f = createFloat(str);\n+ return f;\n final Double d = createDouble(str);\n",
            [
                "assertTrue(NumberUtils.createNumber(shouldBeFloat) instanceof Float);",
                "assertTrue(NumberUtils.createNumber(shouldBeDouble) instanceof Double);",
                "assertTrue(NumberUtils.createNumber(shouldBeBigDecimal) instanceof BigDecimal);",
            ],
        )
        self.assertIn("duplicate_failed_patch_strategy", feedback)
        self.assertIn("numeric_type_selection_failed_strategy", feedback)

    def test_duplicate_rejection_limit_is_env_configurable(self):
        import os

        old_value = os.environ.get("REPAIR_DUPLICATE_REJECTION_LIMIT")
        try:
            os.environ["REPAIR_DUPLICATE_REJECTION_LIMIT"] = "3"
            self.assertEqual(_duplicate_rejection_limit(), 3)
            os.environ["REPAIR_DUPLICATE_REJECTION_LIMIT"] = "0"
            self.assertEqual(_duplicate_rejection_limit(), 1)
        finally:
            if old_value is None:
                os.environ.pop("REPAIR_DUPLICATE_REJECTION_LIMIT", None)
            else:
                os.environ["REPAIR_DUPLICATE_REJECTION_LIMIT"] = old_value

    def test_replacement_exists_counts_as_duplicate_apply_failure(self):
        self.assertTrue(_is_duplicate_apply_failure(["src/Foo.java: replacement text already exists"]))
        self.assertFalse(_is_duplicate_apply_failure(["src/Foo.java: old text not found"]))

    def test_regex_whitespace_failure_adds_constraints_and_needles(self):
        output = (
            "junit.framework.AssertionFailedError: Expected FDF failure, but got Mon Mar 02 "
            "for [M E,3  Tue] using (\\p{IsNd}++)\\s*+(Tue|Tuesday)"
        )
        constraints = _failure_output_repair_constraints(output)
        needles = _failure_output_source_needles(output)
        self.assertTrue(any("literal spaces" in item for item in constraints))
        self.assertIn("escapeRegex(regex, formatField, true)", needles)
        self.assertIn("CopyQuotedStrategy", needles)

    def test_prompt_context_budget_trims_bulk_fields(self):
        long_snippet = "class Foo {\n" + "\n".join(f"  int value{i} = {i};" for i in range(500)) + "\n}"
        messages = build_repair_prompt(
            project="Lang",
            bug_id=4,
            metadata={"tests.trigger": "FooTest::test"},
            failing_output="AssertionError\n" + ("stack line\n" * 1000),
            snippets={
                "src/main/java/Foo.java": long_snippet,
                "src/main/java/Bar.java": long_snippet,
            },
            snippet_line_numbers={"src/main/java/Foo.java": "1: class Foo\n" * 500},
            current_diff="",
            attempt=2,
            memory_preferences=["boundary"],
            previous_attempt_failures=["compile failed " + ("detail " * 300)],
            reflections=["visible failed " + ("detail " * 300)],
            context_budget_chars=6000,
        )
        user = json.loads(messages[1]["content"])
        self.assertEqual(user["context_budget_chars"], 6000)
        self.assertIn("...[truncated]...", user["source_snippets"]["src/main/java/Foo.java"])
        self.assertLess(len(messages[1]["content"]), 12000)

    def test_prompt_budget_prioritizes_editable_source_over_read_only_tests(self):
        long_source = "class Foo {\n" + "\n".join(f"  int source{i} = {i};" for i in range(800)) + "\n}"
        long_test = "class FooTest {\n" + "\n".join(f"  void test{i}() {{}}" for i in range(800)) + "\n}"
        messages = build_repair_prompt(
            project="Lang",
            bug_id=4,
            metadata={"tests.trigger": "FooTest::test"},
            failing_output="AssertionError",
            snippets={
                "src/main/java/Foo.java": long_source,
                "src/main/java/Helper.java": long_source,
                "src/test/java/FooTest.java [read-only-test]": long_test,
            },
            snippet_line_numbers={},
            current_diff="",
            attempt=1,
            memory_preferences=[],
            context_budget_chars=9000,
        )
        snippets = json.loads(messages[1]["content"])["source_snippets"]
        self.assertGreater(
            len(snippets["src/main/java/Foo.java"]),
            len(snippets["src/test/java/FooTest.java [read-only-test]"]),
        )
        self.assertGreater(
            len(snippets["src/main/java/Foo.java"]),
            len(snippets["src/main/java/Helper.java"]),
        )

    def test_prompt_budget_keeps_recent_attempt_failures(self):
        messages = build_repair_prompt(
            project="Lang",
            bug_id=4,
            metadata={},
            failing_output="AssertionError",
            snippets={"src/main/java/Foo.java": "class Foo {}"},
            snippet_line_numbers={},
            current_diff="",
            attempt=5,
            memory_preferences=[],
            previous_attempt_failures=[f"attempt {idx}: failure {idx}" for idx in range(1, 6)],
            context_budget_chars=9000,
        )
        failures = json.loads(messages[1]["content"])["previous_attempt_failures"]
        self.assertTrue(failures[0].startswith("attempt 4"))
        self.assertTrue(failures[1].startswith("attempt 5"))

    def test_prompt_context_budget_scales_from_environment(self):
        import os

        old_base = os.environ.get("REPAIR_PROMPT_CONTEXT_CHARS")
        old_floor = os.environ.get("REPAIR_PROMPT_MIN_CONTEXT_CHARS")
        try:
            os.environ["REPAIR_PROMPT_CONTEXT_CHARS"] = "10000"
            os.environ["REPAIR_PROMPT_MIN_CONTEXT_CHARS"] = "3000"
            self.assertEqual(_prompt_context_budget(1.0), 10000)
            self.assertEqual(_prompt_context_budget(0.35), 3500)
            self.assertEqual(_prompt_context_budget(0.1), 3000)
        finally:
            if old_base is None:
                os.environ.pop("REPAIR_PROMPT_CONTEXT_CHARS", None)
            else:
                os.environ["REPAIR_PROMPT_CONTEXT_CHARS"] = old_base
            if old_floor is None:
                os.environ.pop("REPAIR_PROMPT_MIN_CONTEXT_CHARS", None)
            else:
                os.environ["REPAIR_PROMPT_MIN_CONTEXT_CHARS"] = old_floor

    def test_feedback_system_does_not_write_long_term_memory(self):
        memory = BenchmarkMemory()
        plan = RepairPlan(
            diagnosis="boundary",
            patch_hunks=[PatchHunk("src/main/java/Foo.java", "return 0", "return 1")],
            patch_style="boundary",
        )
        with tempfile.TemporaryDirectory() as tmp:
            runner = Defects4JBenchmarkRunner(
                cases=[],
                run_dir=Path(tmp) / "run",
                systems=[],
                max_attempts=1,
                client=None,
                llm=None,
            )
            runner._update_memory("feedback", memory, ["project:Lang"], plan, "trigger", False, "visible_failure")
            self.assertEqual(memory.as_dict(), BenchmarkMemory().as_dict())
            runner._update_memory("self_evolved", memory, ["project:Lang"], plan, "trigger", False, "visible_failure")
        self.assertTrue(memory.failure_reflections)
        self.assertIn("boundary", memory.patch_ranking["project:Lang"])


class BenchmarkRunnerFailureClassificationTest(unittest.TestCase):
    def test_relevant_scope_uses_defects4j_relevant_flag(self):
        class FakeClient:
            binary = "defects4j"

            def __init__(self):
                self.commands = []

            def run(self, command, cwd, check=False):
                self.commands.append(command)
                return CommandResult(command, str(cwd), 0, "Failing tests: 0\n", 0.0)

        client = FakeClient()
        with tempfile.TemporaryDirectory() as tmp:
            runner = Defects4JBenchmarkRunner(
                cases=[],
                run_dir=Path(tmp) / "run",
                systems=[],
                max_attempts=1,
                client=client,
                llm=None,
            )
            result = runner._run_scope(Path(tmp), "relevant", {"tests.relevant": "pkg.FooTest"})
        self.assertTrue(result.passed)
        self.assertEqual(client.commands, [["defects4j", "test", "-r"]])

    def test_failing_tests_file_is_appended_to_failed_test_output(self):
        class FakeClient:
            binary = "defects4j"

            def run(self, command, cwd, check=False):
                return CommandResult(command, str(cwd), 0, "Failing tests: 1\n  - T::m\n", 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "failing_tests").write_text("--- T::m\nAssertionFailedError: expected:<1> but was:<0>\n", encoding="utf-8")
            runner = Defects4JBenchmarkRunner(
                cases=[],
                run_dir=root / "run",
                systems=[],
                max_attempts=1,
                client=FakeClient(),
                llm=None,
            )
            result = runner._run_scope(root, "trigger", {"tests.trigger": "T::m"})
        self.assertIn("[failing_tests]", result.results[0].output)
        self.assertIn("expected:<1> but was:<0>", result.results[0].output)

    def test_failure_summary_preserves_assertion_details_over_stack_tail(self):
        output = (
            "Running ant...\nFailing tests: 1\n\n[failing_tests]\n"
            "--- T::m\n"
            "junit.framework.AssertionFailedError: expected: java.lang.Integer<1> but was: java.lang.Long<1>\n"
            "\tat org.junit.Assert.fail(Assert.java:88)\n"
            "\tat org.apache.tools.ant.Project.executeTarget(Project.java:1368)\n"
            "\tat org.apache.tools.ant.Main.startAnt(Main.java:217)\n"
        )
        summary = _failure_summary(output)
        self.assertIn("T::m", summary)
        self.assertIn("expected: java.lang.Integer<1> but was: java.lang.Long<1>", summary)
        self.assertNotIn("Main.startAnt", summary)

    def test_failure_stack_line_focuses_modified_source_window(self):
        output = (
            "[failing_tests]\n--- pkg.FooTest::testBug\n"
            "java.lang.NumberFormatException: bad\n"
            "\tat pkg.Foo.target(Foo.java:520)\n"
        )
        needles, line_hints = _failure_source_hints(output, "src/main/java", ["pkg.Foo"])
        self.assertIn("target", needles)
        self.assertEqual(line_hints["src/main/java/pkg/Foo.java"], [520])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src/main/java/pkg/Foo.java"
            source.parent.mkdir(parents=True)
            lines = [f"// filler {idx}\n" for idx in range(1, 700)]
            lines[519] = "class Foo { void target() { int boundaryValue = 1; } }\n"
            source.write_text("".join(lines), encoding="utf-8")
            runner = Defects4JBenchmarkRunner(
                cases=[],
                run_dir=root / "run",
                systems=[],
                max_attempts=1,
                client=None,
                llm=None,
            )
            import os

            old_chars = os.environ.get("REPAIR_SNIPPET_CHARS")
            old_window = os.environ.get("REPAIR_SNIPPET_WINDOW_CHARS")
            try:
                os.environ["REPAIR_SNIPPET_CHARS"] = "800"
                os.environ["REPAIR_SNIPPET_WINDOW_CHARS"] = "400"
                snippets, line_numbers = runner._read_snippet_context(
                    root,
                    {
                        "dir.src.classes": "src/main/java",
                        "dir.src.tests": "src/test/java",
                        "classes.modified": "pkg.Foo",
                        "tests.trigger": "pkg.FooTest::testBug",
                    },
                    [],
                    failing_output=output,
                )
            finally:
                if old_chars is None:
                    os.environ.pop("REPAIR_SNIPPET_CHARS", None)
                else:
                    os.environ["REPAIR_SNIPPET_CHARS"] = old_chars
                if old_window is None:
                    os.environ.pop("REPAIR_SNIPPET_WINDOW_CHARS", None)
                else:
                    os.environ["REPAIR_SNIPPET_WINDOW_CHARS"] = old_window
        self.assertIn("boundaryValue", snippets["src/main/java/pkg/Foo.java"])
        self.assertIn("520: class Foo", line_numbers["src/main/java/pkg/Foo.java"])

    def test_read_snippets_includes_long_source_and_trigger_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src/main/java/pkg/Foo.java"
            source.parent.mkdir(parents=True)
            source.write_text("a" * 25000 + " void testBug() { targetMethod(); }\n", encoding="utf-8")
            test = root / "src/test/java/pkg/FooTest.java"
            test.parent.mkdir(parents=True)
            test.write_text("b" * 25000 + "class FooTest { void testBug() { assert true; } }\n", encoding="utf-8")
            runner = Defects4JBenchmarkRunner(
                cases=[],
                run_dir=root / "run",
                systems=[],
                max_attempts=1,
                client=None,
                llm=None,
            )
            snippets = runner._read_snippets(
                root,
                {
                    "dir.src.classes": "src/main/java",
                    "dir.src.tests": "src/test/java",
                    "classes.modified": "pkg.Foo",
                    "tests.trigger": "pkg.FooTest::testBug",
                },
                [],
            )
        self.assertIn("targetMethod", snippets["src/main/java/pkg/Foo.java"])
        self.assertIn("src/test/java/pkg/FooTest.java [read-only-test]", snippets)
        self.assertIn("assert true", snippets["src/test/java/pkg/FooTest.java [read-only-test]"])

    def test_snippet_focus_prefers_code_occurrence_over_changelog_comment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src/main/java/pkg/Foo.java"
            source.parent.mkdir(parents=True)
            source.write_text(
                "/**\n"
                " * 01-Jan-2000: Mentioned targetMethod in release notes.\n"
                " */\n"
                + "\n".join(f"// filler {idx}" for idx in range(400))
                + "\nclass Foo {\n"
                "  int targetMethod() {\n"
                "    return importantValue();\n"
                "  }\n"
                "}\n",
                encoding="utf-8",
            )
            test = root / "src/test/java/pkg/FooTest.java"
            test.parent.mkdir(parents=True)
            test.write_text(
                "package pkg;\n"
                "class FooTest {\n"
                "  void testBug() {\n"
                "    Foo foo = new Foo();\n"
                "    assertEquals(1, foo.targetMethod());\n"
                "  }\n"
                "}\n",
                encoding="utf-8",
            )
            runner = Defects4JBenchmarkRunner(
                cases=[],
                run_dir=root / "run",
                systems=[],
                max_attempts=1,
                client=None,
                llm=None,
            )
            import os

            old_chars = os.environ.get("REPAIR_SNIPPET_CHARS")
            old_window = os.environ.get("REPAIR_SNIPPET_WINDOW_CHARS")
            try:
                os.environ["REPAIR_SNIPPET_CHARS"] = "900"
                os.environ["REPAIR_SNIPPET_WINDOW_CHARS"] = "500"
                snippets, _ = runner._read_snippet_context(
                    root,
                    {
                        "dir.src.classes": "src/main/java",
                        "dir.src.tests": "src/test/java",
                        "classes.modified": "pkg.Foo",
                        "tests.trigger": "pkg.FooTest::testBug",
                    },
                    [],
                )
            finally:
                if old_chars is None:
                    os.environ.pop("REPAIR_SNIPPET_CHARS", None)
                else:
                    os.environ["REPAIR_SNIPPET_CHARS"] = old_chars
                if old_window is None:
                    os.environ.pop("REPAIR_SNIPPET_WINDOW_CHARS", None)
                else:
                    os.environ["REPAIR_SNIPPET_WINDOW_CHARS"] = old_window
        self.assertIn("importantValue", snippets["src/main/java/pkg/Foo.java"])

    def test_line_numbered_snippet_prioritizes_code_window_over_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src/main/java/pkg/Foo.java"
            source.parent.mkdir(parents=True)
            source.write_text(
                "\n".join(f"// changelog targetMethod {idx}" for idx in range(300))
                + "\nclass Foo {\n"
                "  int targetMethod() {\n"
                "    return importantValue();\n"
                "  }\n"
                "}\n",
                encoding="utf-8",
            )
            test = root / "src/test/java/pkg/FooTest.java"
            test.parent.mkdir(parents=True)
            test.write_text(
                "package pkg;\n"
                "class FooTest { void testBug() { assertEquals(1, new Foo().targetMethod()); } }\n",
                encoding="utf-8",
            )
            runner = Defects4JBenchmarkRunner(
                cases=[],
                run_dir=root / "run",
                systems=[],
                max_attempts=1,
                client=None,
                llm=None,
            )
            import os

            old_chars = os.environ.get("REPAIR_SNIPPET_CHARS")
            old_window = os.environ.get("REPAIR_SNIPPET_WINDOW_CHARS")
            try:
                os.environ["REPAIR_SNIPPET_CHARS"] = "700"
                os.environ["REPAIR_SNIPPET_WINDOW_CHARS"] = "500"
                snippets, line_numbers = runner._read_snippet_context(
                    root,
                    {
                        "dir.src.classes": "src/main/java",
                        "dir.src.tests": "src/test/java",
                        "classes.modified": "pkg.Foo",
                        "tests.trigger": "pkg.FooTest::testBug",
                    },
                    [],
                )
            finally:
                if old_chars is None:
                    os.environ.pop("REPAIR_SNIPPET_CHARS", None)
                else:
                    os.environ["REPAIR_SNIPPET_CHARS"] = old_chars
                if old_window is None:
                    os.environ.pop("REPAIR_SNIPPET_WINDOW_CHARS", None)
                else:
                    os.environ["REPAIR_SNIPPET_WINDOW_CHARS"] = old_window
        self.assertIn("importantValue", snippets["src/main/java/pkg/Foo.java"])
        self.assertIn("importantValue", line_numbers["src/main/java/pkg/Foo.java"])

    def test_paired_source_needles_include_sibling_domain_range_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src/main/java/pkg/Bounds.java"
            source.parent.mkdir(parents=True)
            source.write_text(
                "class Bounds {\n"
                "  Range iterateDomainBounds(Data d) {\n"
                "    double x = d.getXValue();\n"
                "    return new Range(x, x);\n"
                "  }\n"
                + "\n".join(f"// filler {idx}" for idx in range(220))
                + "\n  Range iterateRangeBounds(Data d) {\n"
                "    double y = d.getYValue();\n"
                "    return new Range(y, y);\n"
                "  }\n"
                "}\n",
                encoding="utf-8",
            )
            test = root / "src/test/java/pkg/BoundsTest.java"
            test.parent.mkdir(parents=True)
            test.write_text(
                "class BoundsTest {\n"
                "  void testBounds() {\n"
                "    Range r = Bounds.iterateDomainBounds(d);\n"
                "    assertEquals(1.0, r.getLowerBound(), EPSILON);\n"
                "    assertEquals(2.0, r.getUpperBound(), EPSILON);\n"
                "  }\n"
                "}\n",
                encoding="utf-8",
            )
            runner = Defects4JBenchmarkRunner(
                cases=[],
                run_dir=root / "run",
                systems=[],
                max_attempts=1,
                client=None,
                llm=None,
            )
            import os

            old_chars = os.environ.get("REPAIR_SNIPPET_CHARS")
            old_window = os.environ.get("REPAIR_SNIPPET_WINDOW_CHARS")
            try:
                os.environ["REPAIR_SNIPPET_CHARS"] = "1200"
                os.environ["REPAIR_SNIPPET_WINDOW_CHARS"] = "500"
                snippets, line_numbers = runner._read_snippet_context(
                    root,
                    {
                        "dir.src.classes": "src/main/java",
                        "dir.src.tests": "src/test/java",
                        "classes.modified": "pkg.Bounds",
                        "tests.trigger": "pkg.BoundsTest::testBounds",
                    },
                    [],
                )
            finally:
                if old_chars is None:
                    os.environ.pop("REPAIR_SNIPPET_CHARS", None)
                else:
                    os.environ["REPAIR_SNIPPET_CHARS"] = old_chars
                if old_window is None:
                    os.environ.pop("REPAIR_SNIPPET_WINDOW_CHARS", None)
                else:
                    os.environ["REPAIR_SNIPPET_WINDOW_CHARS"] = old_window
        self.assertIn("iterateDomainBounds", snippets["src/main/java/pkg/Bounds.java"])
        self.assertIn("iterateRangeBounds", snippets["src/main/java/pkg/Bounds.java"])
        self.assertIn("iterateRangeBounds", line_numbers["src/main/java/pkg/Bounds.java"])

    def test_bounds_nan_assertions_pull_interval_xy_api_needles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src/main/java/pkg/DatasetUtilities.java"
            source.parent.mkdir(parents=True)
            source.write_text(
                "class DatasetUtilities {\n"
                "  Range iterateDomainBounds(XYDataset dataset) {\n"
                "    IntervalXYDataset ixyd = (IntervalXYDataset) dataset;\n"
                "    double lvalue = ixyd.getStartXValue(0, 0);\n"
                "    double uvalue = ixyd.getEndXValue(0, 0);\n"
                "    return new Range(lvalue, uvalue);\n"
                "  }\n"
                + "\n".join(f"// filler {idx}" for idx in range(260))
                + "\n  Range iterateRangeBounds(XYDataset dataset) {\n"
                "    IntervalXYDataset ixyd = (IntervalXYDataset) dataset;\n"
                "    double lvalue = ixyd.getStartYValue(0, 0);\n"
                "    double uvalue = ixyd.getEndYValue(0, 0);\n"
                "    return new Range(lvalue, uvalue);\n"
                "  }\n"
                "}\n",
                encoding="utf-8",
            )
            test = root / "src/test/java/pkg/DatasetUtilitiesTest.java"
            test.parent.mkdir(parents=True)
            test.write_text(
                "class DatasetUtilitiesTest {\n"
                "  void testBounds() {\n"
                "    s.add(1.0, Double.NaN, Double.NaN, Double.NaN, 1.5, Double.NaN);\n"
                "    Range r = DatasetUtilities.iterateDomainBounds(d);\n"
                "    assertEquals(1.0, r.getLowerBound(), EPSILON);\n"
                "    assertEquals(1.0, r.getUpperBound(), EPSILON);\n"
                "  }\n"
                "}\n",
                encoding="utf-8",
            )
            runner = Defects4JBenchmarkRunner(
                cases=[],
                run_dir=root / "run",
                systems=[],
                max_attempts=1,
                client=None,
                llm=None,
            )
            import os

            old_chars = os.environ.get("REPAIR_SNIPPET_CHARS")
            old_window = os.environ.get("REPAIR_SNIPPET_WINDOW_CHARS")
            try:
                os.environ["REPAIR_SNIPPET_CHARS"] = "1400"
                os.environ["REPAIR_SNIPPET_WINDOW_CHARS"] = "500"
                _, line_numbers = runner._read_snippet_context(
                    root,
                    {
                        "dir.src.classes": "src/main/java",
                        "dir.src.tests": "src/test/java",
                        "classes.modified": "pkg.DatasetUtilities",
                        "tests.trigger": "pkg.DatasetUtilitiesTest::testBounds",
                    },
                    [],
                )
            finally:
                if old_chars is None:
                    os.environ.pop("REPAIR_SNIPPET_CHARS", None)
                else:
                    os.environ["REPAIR_SNIPPET_CHARS"] = old_chars
                if old_window is None:
                    os.environ.pop("REPAIR_SNIPPET_WINDOW_CHARS", None)
                else:
                    os.environ["REPAIR_SNIPPET_WINDOW_CHARS"] = old_window
        rendered = line_numbers["src/main/java/pkg/DatasetUtilities.java"]
        self.assertIn("getStartXValue", rendered)
        self.assertIn("getStartYValue", rendered)

    def test_requested_files_extend_next_prompt_context_source_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src/main/java/pkg/Bar.java"
            source.parent.mkdir(parents=True)
            source.write_text("package pkg;\nclass Bar { int target() { return 1; } }\n", encoding="utf-8")
            test = root / "src/test/java/pkg/BarTest.java"
            test.parent.mkdir(parents=True)
            test.write_text("class BarTest {}\n", encoding="utf-8")
            paths = _requested_source_paths(
                ["pkg.Bar", "src/main/java/pkg/Bar.java", "src/test/java/pkg/BarTest.java", "../secret.java"],
                "src/main/java",
                root,
            )
            runner = Defects4JBenchmarkRunner(
                cases=[],
                run_dir=root / "run",
                systems=[],
                max_attempts=1,
                client=None,
                llm=None,
            )
            snippets, line_numbers = runner._read_snippet_context(
                root,
                {
                    "dir.src.classes": "src/main/java",
                    "dir.src.tests": "src/test/java",
                    "classes.modified": "",
                    "tests.trigger": "",
                },
                [],
                requested_files=["pkg.Bar", "src/test/java/pkg/BarTest.java", "../secret.java"],
            )
        self.assertEqual(paths, [Path("src/main/java/pkg/Bar.java")])
        self.assertIn("src/main/java/pkg/Bar.java", snippets)
        self.assertIn("target()", snippets["src/main/java/pkg/Bar.java"])
        self.assertIn("1: package pkg;", line_numbers["src/main/java/pkg/Bar.java"])
        self.assertNotIn("src/test/java/pkg/BarTest.java", snippets)

    def test_read_snippets_includes_same_package_helper_utility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "src/main/java/pkg"
            source_dir.mkdir(parents=True)
            (source_dir / "ShapeList.java").write_text(
                "package pkg;\n"
                "import java.awt.Shape;\n"
                "class ShapeList {\n"
                "  void setShape(int index, Shape shape) {}\n"
                "  boolean same(Shape a, Shape b) { return a.equals(b); }\n"
                "}\n",
                encoding="utf-8",
            )
            (source_dir / "ShapeUtilities.java").write_text(
                "package pkg;\n"
                "import java.awt.Shape;\n"
                "import java.awt.geom.Line2D;\n"
                "class ShapeUtilities {\n"
                "  public static boolean equal(Shape s1, Shape s2) { return true; }\n"
                "  public static boolean equal(Line2D l1, Line2D l2) { return true; }\n"
                "}\n",
                encoding="utf-8",
            )
            test = root / "src/test/java/pkg/ShapeListTests.java"
            test.parent.mkdir(parents=True)
            test.write_text(
                "package pkg;\n"
                "import java.awt.geom.Line2D;\n"
                "class ShapeListTests {\n"
                "  void testEquals() {\n"
                "    ShapeList l1 = new ShapeList();\n"
                "    l1.setShape(1, new Line2D.Double(1, 2, 3, 4));\n"
                "  }\n"
                "}\n",
                encoding="utf-8",
            )
            runner = Defects4JBenchmarkRunner(
                cases=[],
                run_dir=root / "run",
                systems=[],
                max_attempts=1,
                client=None,
                llm=None,
            )
            snippets, _ = runner._read_snippet_context(
                root,
                {
                    "dir.src.classes": "src/main/java",
                    "dir.src.tests": "src/test/java",
                    "classes.modified": "pkg.ShapeList",
                    "tests.trigger": "pkg.ShapeListTests::testEquals",
                },
                [],
            )
        self.assertIn("src/main/java/pkg/ShapeUtilities.java", snippets)
        self.assertIn("equal(Line2D", snippets["src/main/java/pkg/ShapeUtilities.java"])

    def test_trigger_assertion_summary_extracts_boundary_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test = root / "src/test/java/pkg/NumberUtilsTest.java"
            test.parent.mkdir(parents=True)
            test.write_text(
                "package pkg;\n"
                "class NumberUtilsTest {\n"
                "  void testHex() {\n"
                "    assertEquals(Integer.valueOf(0x7FFFFFFF), NumberUtils.createNumber(\"0x7FFFFFFF\"));\n"
                "    assertEquals(Long.valueOf(0x80000000L), NumberUtils.createNumber(\"0x80000000\"));\n"
                "  }\n"
                "}\n",
                encoding="utf-8",
            )
            summary = _trigger_assertion_summary(
                root,
                {
                    "dir.src.tests": "src/test/java",
                    "tests.trigger": "pkg.NumberUtilsTest::testHex",
                },
            )
        self.assertGreaterEqual(len(summary), 2)
        self.assertTrue(summary[0].startswith("full_test_method:"))
        self.assertIn("0x7FFFFFFF", summary[0])
        self.assertIn("0x80000000", summary[0])

    def test_trigger_assertion_summary_keeps_arrange_lines_before_asserts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test = root / "src/test/java/pkg/DatasetUtilitiesTest.java"
            test.parent.mkdir(parents=True)
            test.write_text(
                "package pkg;\n"
                "class DatasetUtilitiesTest {\n"
                "  void testBounds() {\n"
                "    s.add(1.0, 1.5, Double.NaN, Double.NaN, 1.5, Double.NaN);\n"
                "    r = DatasetUtilities.iterateDomainBounds(d);\n"
                "    assertEquals(1.5, r.getUpperBound(), EPSILON);\n"
                "    s.add(1.0, Double.NaN, 0.5, Double.NaN, 1.5, Double.NaN);\n"
                "    r = DatasetUtilities.iterateDomainBounds(d);\n"
                "    assertEquals(0.5, r.getLowerBound(), EPSILON);\n"
                "  }\n"
                "}\n",
                encoding="utf-8",
            )
            summary = _trigger_assertion_summary(
                root,
                {
                    "dir.src.tests": "src/test/java",
                    "tests.trigger": "pkg.DatasetUtilitiesTest::testBounds",
                },
            )
        self.assertTrue(summary[0].startswith("full_test_method:"))
        self.assertIn("s.add(1.0, 1.5", summary[0])
        self.assertIn("getUpperBound", summary[0])
        self.assertIn("Double.NaN, 0.5", summary[0])
        self.assertIn("getLowerBound", summary[0])

    def test_trigger_assertion_summary_keeps_all_short_method_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test = root / "src/test/java/pkg/NumberUtilsTest.java"
            test.parent.mkdir(parents=True)
            test.write_text(
                "package pkg;\n"
                "class NumberUtilsTest {\n"
                "  void testStringCreateNumberEnsureNoPrecisionLoss() {\n"
                "    String shouldBeFloat = \"1.23e+2\";\n"
                "    String shouldBeDouble = \"3.40282354e+38\";\n"
                "    String shouldBeBigDecimal = \"1.797693134862315759e+308\";\n"
                "    assertTrue(NumberUtils.createNumber(shouldBeFloat) instanceof Float);\n"
                "    assertTrue(NumberUtils.createNumber(shouldBeDouble) instanceof Double);\n"
                "    assertTrue(NumberUtils.createNumber(shouldBeBigDecimal) instanceof java.math.BigDecimal);\n"
                "  }\n"
                "}\n",
                encoding="utf-8",
            )
            summary = _trigger_assertion_summary(
                root,
                {
                    "dir.src.tests": "src/test/java",
                    "tests.trigger": "pkg.NumberUtilsTest::testStringCreateNumberEnsureNoPrecisionLoss",
                },
            )
        self.assertIn("shouldBeFloat", summary[0])
        self.assertIn("shouldBeDouble", summary[0])
        self.assertIn("shouldBeBigDecimal", summary[0])

    def test_numeric_instanceof_assertions_create_type_selection_constraints(self):
        assertions = [
            "full_test_method: String shouldBeFloat = \"1.23\"; | "
            "String shouldBeDouble = \"3.40282354e+38\"; | "
            "String shouldBeBigDecimal = \"1.797693134862315759e+308\"; | "
            "assertTrue(NumberUtils.createNumber(shouldBeFloat) instanceof Float); | "
            "assertTrue(NumberUtils.createNumber(shouldBeDouble) instanceof Double); | "
            "assertTrue(NumberUtils.createNumber(shouldBeBigDecimal) instanceof java.math.BigDecimal);"
        ]
        constraints = _derived_repair_constraints(assertions)
        needles = _constraint_source_needles(assertions)
        self.assertTrue(any("Do not simply try Float before Double" in item for item in constraints))
        self.assertIn("createFloat", needles)
        self.assertIn("createDouble", needles)
        self.assertIn("createBigDecimal", needles)

    def test_compile_feedback_lists_missing_symbol_and_available_api_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "src/main/java/pkg"
            source_dir.mkdir(parents=True)
            (source_dir / "ShapeList.java").write_text(
                "package pkg;\n"
                "class ShapeList extends AbstractObjectList {\n"
                "  public void setShape(int index, Object shape) { set(index, shape); }\n"
                "}\n",
                encoding="utf-8",
            )
            (source_dir / "AbstractObjectList.java").write_text(
                "package pkg;\n"
                "class AbstractObjectList {\n"
                "  public void clear() {}\n"
                "  protected void set(int index, Object value) {}\n"
                "  public int size() { return 0; }\n"
                "}\n",
                encoding="utf-8",
            )
            output = (
                "[javac] ShapeList.java:10: error: cannot find symbol\n"
                "[javac]             remove(i);\n"
                "[javac]             ^\n"
                "[javac]   symbol:   method remove(int)\n"
                "[javac]   location: class ShapeList\n"
            )
            feedback = _compile_failure_feedback(
                output,
                root,
                {
                    "dir.src.classes": "src/main/java",
                    "classes.modified": "pkg.ShapeList",
                },
                requested_files=[],
            )
        self.assertIn("missing method remove(int)", feedback)
        self.assertIn("ShapeList", feedback)
        self.assertIn("AbstractObjectList", feedback)
        self.assertIn("clear", feedback)
        self.assertIn("set", feedback)

    def test_patch_grounding_feedback_includes_exact_current_method_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src/main/java/pkg/NumberUtils.java"
            source.parent.mkdir(parents=True)
            source.write_text(
                "package pkg;\n"
                "class NumberUtils {\n"
                "  static Number createNumber(String str) {\n"
                "    Double d = Double.valueOf(str);\n"
                "    if (!d.isInfinite()) {\n"
                "      return d;\n"
                "    }\n"
                "    return createBigDecimal(str);\n"
                "  }\n"
                "  static Number createBigDecimal(String str) { return null; }\n"
                "}\n",
                encoding="utf-8",
            )
            plan = RepairPlan(
                diagnosis="createNumber should avoid precision loss",
                patch_hunks=[
                    PatchHunk(
                        "src/main/java/pkg/NumberUtils.java",
                        "return inventedOld;",
                        "return createBigDecimal(str);",
                    )
                ],
            )
            feedback = _patch_grounding_feedback(root, plan, ["src/main/java/pkg/NumberUtils.java: old text not found"])
        self.assertIn("exact_current_source_grounding", feedback)
        self.assertIn("createNumber", feedback)
        self.assertIn("return createBigDecimal(str);", feedback)

    def test_patch_grounding_feedback_focuses_anchor_inside_long_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src/main/java/pkg/NumberUtils.java"
            source.parent.mkdir(parents=True)
            filler = "\n".join(f"    int filler{i} = {i};" for i in range(140))
            source.write_text(
                "package pkg;\n"
                "class NumberUtils {\n"
                "  static Number createNumber(String str) {\n"
                f"{filler}\n"
                "    Double d = createDouble(str);\n"
                "    return d;\n"
                "  }\n"
                "  static Double createDouble(String str) { return null; }\n"
                "}\n",
                encoding="utf-8",
            )
            plan = RepairPlan(
                diagnosis="createNumber should avoid precision loss",
                patch_hunks=[
                    PatchHunk(
                        "src/main/java/pkg/NumberUtils.java",
                        "return createDouble(str);",
                        "return createBigDecimal(str);",
                    )
                ],
            )
            feedback = _patch_grounding_feedback(root, plan, ["src/main/java/pkg/NumberUtils.java: old text not found"])
        self.assertIn("createDouble(str)", feedback)
        self.assertNotIn("int filler0", feedback)

    def test_patch_grounding_feedback_prefers_hunk_anchor_over_first_overload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src/main/java/pkg/FastDatePrinter.java"
            source.parent.mkdir(parents=True)
            filler = "\n".join(f"  int filler{idx};" for idx in range(120))
            source.write_text(
                "package pkg;\n"
                "class FastDatePrinter {\n"
                "  void appendTo(StringBuffer buffer, Calendar calendar) {\n"
                "    buffer.append(mValue);\n"
                "  }\n"
                f"{filler}\n"
                "  void appendTo(StringBuffer buffer, Calendar calendar, boolean daylight) {\n"
                "    buffer.append(mTimeZone.getDisplayName(daylight, mStyle, mLocale));\n"
                "  }\n"
                "}\n",
                encoding="utf-8",
            )
            plan = RepairPlan(
                diagnosis="appendTo should use calendar timezone",
                patch_hunks=[
                    PatchHunk(
                        "src/main/java/pkg/FastDatePrinter.java",
                        "buffer.append(mTimeZone.getDisplayName(daylight, mStyle, mLocale));",
                        "buffer.append(calendar.getTimeZone().getDisplayName(daylight, mStyle, mLocale));",
                    )
                ],
            )
            feedback = _patch_grounding_feedback(root, plan, ["src/main/java/pkg/FastDatePrinter.java: old text not found"])
        self.assertIn("mTimeZone.getDisplayName", feedback)
        self.assertNotIn("buffer.append(mValue)", feedback)

    def test_llm_timeout_is_agent_failure_not_infrastructure(self):
        class TimeoutLLM:
            class Usage:
                calls = 0
                prompt_tokens = 0
                completion_tokens = 0
                estimated_cost_usd = 0.0

            usage = Usage()

            def enabled(self):
                return True

            def complete(self, messages, temperature=0.0):
                raise TimeoutError("model deadline")

        class FakeClient:
            binary = "defects4j"

            def run(self, command, cwd, check=False):
                return CommandResult(command, str(cwd), 0, "Failing tests: 1\n  - T::m\n", 0.0)

        class FakeRunner(Defects4JBenchmarkRunner):
            def _fresh_checkout(self, case, workdir, trace):
                metadata = {
                    "dir.src.classes": "src/main/java",
                    "dir.src.tests": "src/test/java",
                    "tests.trigger": "T::m",
                    "tests.relevant": "T",
                    "classes.modified": "pkg.Foo",
                }
                return metadata, "AssertionError", ScopeResult(
                    "trigger",
                    False,
                    [CommandResult(["defects4j", "test", "-t", "T::m"], str(workdir), 0, "Failing tests: 1", 0.0)],
                )

        with tempfile.TemporaryDirectory() as tmp:
            case = load_cases(Path("configs/defects4j_smoke.json"))[0]
            runner = FakeRunner(
                cases=[case],
                run_dir=Path(tmp) / "run",
                systems=["feedback"],
                max_attempts=1,
                client=FakeClient(),
                llm=TimeoutLLM(),
                tuning=RuntimeTuning(attempt_limits={"feedback": 1}, max_non_patch_rounds=2),
            )
            metrics, trace = runner._run_case("feedback", case, BenchmarkMemory())
        self.assertFalse(metrics.infrastructure_failure)
        self.assertTrue(metrics.agent_failure)
        self.assertEqual(metrics.tool_calls, 2)
        self.assertIn("llm_error", trace["attempts"][0])
        self.assertTrue(trace["attempts"][0]["non_patch_model_failure"])

    def test_missing_api_key_llm_error_is_non_retryable(self):
        self.assertTrue(_is_non_retryable_llm_error("DEEPSEEK_API_KEY is not set"))
        self.assertTrue(_is_non_retryable_llm_error("HTTPError: 401 Unauthorized"))
        self.assertFalse(_is_non_retryable_llm_error("model deadline"))
        self.assertTrue(_is_retryable_llm_error("DeepSeek returned empty message content"))
        self.assertTrue(_is_retryable_llm_error("URLError: SSL EOF occurred"))
        self.assertTrue(_is_retryable_llm_error("DeepSeek subprocess exceeded 180s deadline"))
        self.assertFalse(_is_retryable_llm_error("DEEPSEEK_API_KEY is not set"))

    def test_read_request_round_does_not_consume_patch_attempt_budget(self):
        class FakeLLM:
            class Usage:
                calls = 0
                prompt_tokens = 0
                completion_tokens = 0
                estimated_cost_usd = 0.0

            usage = Usage()

            def __init__(self):
                self.responses = [
                    json.dumps(
                        {
                            "diagnosis": "need exact file",
                            "files_to_read": ["src/main/java/pkg/Foo.java"],
                            "patch_hunks": [],
                            "patch_style": "read-request",
                        }
                    ),
                    json.dumps(
                        {
                            "diagnosis": "change return value",
                            "patch_hunks": [
                                {
                                    "file": "src/main/java/pkg/Foo.java",
                                    "old": "return 0",
                                    "new": "return 1",
                                }
                            ],
                            "patch_style": "boundary",
                        }
                    ),
                ]

            def enabled(self):
                return True

            def complete(self, messages, temperature=0.0):
                self.usage.calls += 1
                return self.responses.pop(0)

        class FakeClient:
            binary = "defects4j"

            def run(self, command, cwd, check=False):
                return CommandResult(command, str(cwd), 0, "Failing tests: 0\n", 0.0)

        class FakeRunner(Defects4JBenchmarkRunner):
            def _fresh_checkout(self, case, workdir, trace):
                if workdir.exists():
                    shutil.rmtree(workdir)
                source = workdir / "src/main/java/pkg/Foo.java"
                source.parent.mkdir(parents=True)
                source.write_text("package pkg;\nclass Foo { int value() { return 0; } }\n", encoding="utf-8")
                metadata = {
                    "dir.src.classes": "src/main/java",
                    "dir.src.tests": "src/test/java",
                    "tests.trigger": "pkg.FooTest::testBug",
                    "tests.relevant": "pkg.FooTest",
                    "classes.modified": "pkg.Foo",
                }
                failure = (
                    "Failing tests: 1\n\n[failing_tests]\n--- pkg.FooTest::testBug\n"
                    "junit.framework.AssertionFailedError: expected:<1> but was:<0>\n"
                    "\tat pkg.Foo.value(Foo.java:2)\n"
                )
                return metadata, failure, ScopeResult(
                    "trigger",
                    False,
                    [CommandResult(["defects4j", "test", "-t", "pkg.FooTest::testBug"], str(workdir), 0, failure, 0.0)],
                )

        with tempfile.TemporaryDirectory() as tmp:
            case = load_cases(Path("configs/defects4j_smoke.json"))[0]
            runner = FakeRunner(
                cases=[case],
                run_dir=Path(tmp) / "run",
                systems=["self_evolved"],
                max_attempts=1,
                client=FakeClient(),
                llm=FakeLLM(),
                tuning=RuntimeTuning(attempt_limits={"self_evolved": 1}),
            )
            runner._prepare_dirs()
            metrics, trace = runner._run_case("self_evolved", case, BenchmarkMemory())
        self.assertEqual(len(trace["attempts"]), 2)
        self.assertTrue(trace["attempts"][0]["read_request_without_patch"])
        self.assertEqual(trace["attempts"][1]["patch_attempt"], 1)
        self.assertTrue(metrics.pass_at_1)
        self.assertEqual(metrics.deepseek_calls, 2)

    def test_parse_error_round_does_not_consume_self_evolved_patch_attempt_budget(self):
        class FakeLLM:
            class Usage:
                calls = 0
                prompt_tokens = 0
                completion_tokens = 0
                estimated_cost_usd = 0.0

            usage = Usage()

            def __init__(self):
                self.responses = [
                    "not json",
                    json.dumps(
                        {
                            "diagnosis": "change return value",
                            "patch_hunks": [
                                {
                                    "file": "src/main/java/pkg/Foo.java",
                                    "old": "return 0",
                                    "new": "return 1",
                                }
                            ],
                            "patch_style": "boundary",
                        }
                    ),
                ]

            def enabled(self):
                return True

            def complete(self, messages, temperature=0.0):
                self.usage.calls += 1
                return self.responses.pop(0)

        class FakeClient:
            binary = "defects4j"

            def run(self, command, cwd, check=False):
                return CommandResult(command, str(cwd), 0, "Failing tests: 0\n", 0.0)

        class FakeRunner(Defects4JBenchmarkRunner):
            def _fresh_checkout(self, case, workdir, trace):
                if workdir.exists():
                    shutil.rmtree(workdir)
                source = workdir / "src/main/java/pkg/Foo.java"
                source.parent.mkdir(parents=True)
                source.write_text("package pkg;\nclass Foo { int value() { return 0; } }\n", encoding="utf-8")
                metadata = {
                    "dir.src.classes": "src/main/java",
                    "dir.src.tests": "src/test/java",
                    "tests.trigger": "pkg.FooTest::testBug",
                    "tests.relevant": "pkg.FooTest",
                    "classes.modified": "pkg.Foo",
                }
                failure = (
                    "Failing tests: 1\n\n[failing_tests]\n--- pkg.FooTest::testBug\n"
                    "junit.framework.AssertionFailedError: expected:<1> but was:<0>\n"
                )
                return metadata, failure, ScopeResult(
                    "trigger",
                    False,
                    [CommandResult(["defects4j", "test", "-t", "pkg.FooTest::testBug"], str(workdir), 0, failure, 0.0)],
                )

        with tempfile.TemporaryDirectory() as tmp:
            case = load_cases(Path("configs/defects4j_smoke.json"))[0]
            runner = FakeRunner(
                cases=[case],
                run_dir=Path(tmp) / "run",
                systems=["self_evolved"],
                max_attempts=1,
                client=FakeClient(),
                llm=FakeLLM(),
                tuning=RuntimeTuning(attempt_limits={"self_evolved": 1}),
            )
            runner._prepare_dirs()
            metrics, trace = runner._run_case("self_evolved", case, BenchmarkMemory())
        self.assertTrue(trace["attempts"][0]["non_patch_parse_failure"])
        self.assertEqual(trace["attempts"][1]["patch_attempt"], 1)
        self.assertTrue(metrics.pass_at_1)
        self.assertEqual(metrics.deepseek_calls, 2)

    def test_retryable_llm_errors_have_separate_non_patch_budget(self):
        class FakeLLM:
            class Usage:
                calls = 0
                prompt_tokens = 0
                completion_tokens = 0
                estimated_cost_usd = 0.0

            usage = Usage()

            def __init__(self):
                self.failures_left = 3

            def enabled(self):
                return True

            def complete(self, messages, temperature=0.0):
                self.usage.calls += 1
                if self.failures_left:
                    self.failures_left -= 1
                    raise TimeoutError("model deadline")
                return json.dumps(
                    {
                        "diagnosis": "change return value",
                        "patch_hunks": [
                            {
                                "file": "src/main/java/pkg/Foo.java",
                                "old": "return 0",
                                "new": "return 1",
                            }
                        ],
                        "patch_style": "boundary",
                    }
                )

        class FakeClient:
            binary = "defects4j"

            def run(self, command, cwd, check=False):
                return CommandResult(command, str(cwd), 0, "Failing tests: 0\n", 0.0)

        class FakeRunner(Defects4JBenchmarkRunner):
            def _fresh_checkout(self, case, workdir, trace):
                if workdir.exists():
                    shutil.rmtree(workdir)
                source = workdir / "src/main/java/pkg/Foo.java"
                source.parent.mkdir(parents=True)
                source.write_text("package pkg;\nclass Foo { int value() { return 0; } }\n", encoding="utf-8")
                metadata = {
                    "dir.src.classes": "src/main/java",
                    "dir.src.tests": "src/test/java",
                    "tests.trigger": "pkg.FooTest::testBug",
                    "tests.relevant": "pkg.FooTest",
                    "classes.modified": "pkg.Foo",
                }
                failure = "Failing tests: 1\n\n[failing_tests]\n--- pkg.FooTest::testBug\nAssertionError: expected:<1> but was:<0>\n"
                return metadata, failure, ScopeResult(
                    "trigger",
                    False,
                    [CommandResult(["defects4j", "test", "-t", "pkg.FooTest::testBug"], str(workdir), 0, failure, 0.0)],
                )

        with tempfile.TemporaryDirectory() as tmp:
            case = load_cases(Path("configs/defects4j_smoke.json"))[0]
            runner = FakeRunner(
                cases=[case],
                run_dir=Path(tmp) / "run",
                systems=["feedback"],
                max_attempts=1,
                client=FakeClient(),
                llm=FakeLLM(),
                tuning=RuntimeTuning(attempt_limits={"feedback": 1}, max_non_patch_rounds=4),
            )
            runner._prepare_dirs()
            metrics, trace = runner._run_case("feedback", case, BenchmarkMemory())
        self.assertEqual([item.get("non_patch_model_failure") for item in trace["attempts"][:3]], [True, True, True])
        self.assertEqual([item.get("prompt_budget_scale") for item in trace["attempts"][:4]], [1.0, 0.7, 0.35, 0.18])
        self.assertEqual(trace["attempts"][3]["patch_attempt"], 1)
        self.assertTrue(metrics.pass_at_1)
        self.assertEqual(metrics.deepseek_calls, 4)

    def test_fresh_memory_ignores_existing_memory_file(self):
        class DummyLLM:
            class Usage:
                calls = 0
                prompt_tokens = 0
                completion_tokens = 0
                estimated_cost_usd = 0.0

            usage = Usage()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_path = root / "memory.json"
            seeded = BenchmarkMemory()
            seeded.update(
                features=["project:Lang"],
                patch_style="seeded-style",
                test_scope="trigger",
                solved=True,
                failure_reason=None,
                reflection=None,
            )
            seeded.save(memory_path)
            runner = Defects4JBenchmarkRunner(
                cases=[],
                run_dir=root / "run",
                systems=[],
                max_attempts=1,
                client=None,
                llm=DummyLLM(),
                tuning=RuntimeTuning(attempt_limits={}, memory_path=memory_path, fresh_memory=True),
            )
            runner.run()
            before = json.loads((root / "run" / "memory_before.json").read_text(encoding="utf-8"))
        self.assertEqual(before["patch_ranking"], {})

    def test_verified_patch_context_returns_contiguous_range_and_full_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src/main/java/pkg/Foo.java"
            source.parent.mkdir(parents=True)
            lines = ["package pkg;\n", "class Foo {\n"]
            lines.extend(f"  int filler{idx} = {idx};\n" for idx in range(1, 18))
            lines.extend(
                [
                    "  public int target(int input) {\n",
                    "    int base = input;\n",
                    "    int guard = base + 1;\n",
                    "    int answer = guard - 1;\n",
                    "    return answer;\n",
                    "  }\n",
                ]
            )
            lines.extend(f"  int tail{idx} = {idx};\n" for idx in range(1, 31))
            lines.append("}\n")
            source.write_text("".join(lines), encoding="utf-8")
            runner = Defects4JBenchmarkRunner(cases=[], run_dir=root / "run", systems=[], max_attempts=1, client=None, llm=None)
            context = runner._read_verified_patch_context(
                root,
                {"dir.src.classes": "src/main/java", "dir.src.tests": "src/test/java"},
                {
                    "line_ranges": [
                        {
                            "file": "src/main/java/pkg/Foo.java",
                            "line_start": 21,
                            "line_end": 23,
                            "method_name": "target",
                        }
                    ]
                },
            )
        entry = context["src/main/java/pkg/Foo.java"]
        excerpt = entry["excerpt"] if isinstance(entry, dict) else entry[0]
        line_numbered = entry["line_numbered"] if isinstance(entry, dict) else entry[1]
        method_body = entry["method_body"] if isinstance(entry, dict) else entry[2]
        self.assertEqual(excerpt, "".join(lines[5:38]))
        self.assertNotIn("snippet window", excerpt.lower())
        self.assertIn("22:     int guard = base + 1;", line_numbered)
        self.assertIn("public int target", method_body)
        self.assertIn("return answer;", method_body)

    def test_preflight_grounding_failure_does_not_consume_patch_attempt_budget(self):
        class FakeLLM:
            class Usage:
                calls = 0
                prompt_tokens = 0
                completion_tokens = 0
                estimated_cost_usd = 0.0

            usage = Usage()

            def __init__(self):
                self.responses = [
                    json.dumps(
                        {
                            "diagnosis": "value should be one",
                            "patch_hunks": [
                                {
                                    "file": "src/main/java/pkg/Foo.java",
                                    "line_start": 99,
                                    "line_end": 99,
                                    "new": "  int value() { return 1; }\n",
                                    "method_name": "value",
                                }
                            ],
                            "patch_style": "range-grounded-boundary",
                        }
                    ),
                    json.dumps(
                        {
                            "diagnosis": "value should be one",
                            "patch_hunks": [
                                {
                                    "file": "src/main/java/pkg/Foo.java",
                                    "line_start": 3,
                                    "line_end": 3,
                                    "new": "  int value() { return 1; }\n",
                                    "method_name": "value",
                                }
                            ],
                            "patch_style": "range-grounded-boundary",
                        }
                    ),
                ]

            def enabled(self):
                return True

            def complete(self, messages, temperature=0.0):
                self.usage.calls += 1
                return self.responses.pop(0)

        class FakeClient:
            binary = "defects4j"

            def run(self, command, cwd, check=False):
                return CommandResult(command, str(cwd), 0, "Failing tests: 0\n", 0.0)

        class FakeRunner(Defects4JBenchmarkRunner):
            def _fresh_checkout(self, case, workdir, trace):
                if workdir.exists():
                    shutil.rmtree(workdir)
                source = workdir / "src/main/java/pkg/Foo.java"
                source.parent.mkdir(parents=True)
                source.write_text("package pkg;\nclass Foo {\n  int value() { return 0; }\n}\n", encoding="utf-8")
                metadata = {
                    "dir.src.classes": "src/main/java",
                    "dir.src.tests": "src/test/java",
                    "tests.trigger": "pkg.FooTest::testBug",
                    "tests.relevant": "pkg.FooTest",
                    "classes.modified": "pkg.Foo",
                }
                failure = "Failing tests: 1\n--- pkg.FooTest::testBug\nAssertionError: expected:<1> but was:<0>\n"
                return metadata, failure, ScopeResult(
                    "trigger",
                    False,
                    [CommandResult(["defects4j", "test", "-t", "pkg.FooTest::testBug"], str(workdir), 0, failure, 0.0)],
                )

        with tempfile.TemporaryDirectory() as tmp:
            case = load_cases(Path("configs/defects4j_smoke.json"))[0]
            runner = FakeRunner(
                cases=[case],
                run_dir=Path(tmp) / "run",
                systems=["self_evolved"],
                max_attempts=1,
                client=FakeClient(),
                llm=FakeLLM(),
                tuning=RuntimeTuning(attempt_limits={"self_evolved": 1}, max_non_patch_rounds=2),
            )
            runner._prepare_dirs()
            metrics, trace = runner._run_case("self_evolved", case, BenchmarkMemory())
        self.assertIsNone(trace["attempts"][0]["patch_attempt"])
        self.assertTrue(trace["attempts"][0]["non_patch_grounding_failure"])
        self.assertEqual(trace["attempts"][1]["patch_attempt"], 1)
        self.assertTrue(metrics.pass_at_1)
        self.assertEqual(metrics.deepseek_calls, 2)

    def test_semantic_failure_retry_keeps_diagnosis_and_requests_alternative_strategy(self):
        class FakeLLM:
            class Usage:
                calls = 0
                prompt_tokens = 0
                completion_tokens = 0
                estimated_cost_usd = 0.0

            usage = Usage()

            def __init__(self):
                self.prompts = []
                self.responses = [
                    json.dumps(
                        {
                            "diagnosis": "value should be one",
                            "patch_hunks": [
                                {
                                    "file": "src/main/java/pkg/Foo.java",
                                    "line_start": 3,
                                    "line_end": 3,
                                    "new": "  int value() { return 2; }\n",
                                    "method_name": "value",
                                    "intent": "try changing observed value",
                                }
                            ],
                            "patch_style": "wrong-constant",
                        }
                    ),
                    json.dumps(
                        {
                            "diagnosis": "value should be one",
                            "patch_hunks": [
                                {
                                    "file": "src/main/java/pkg/Foo.java",
                                    "line_start": 3,
                                    "line_end": 3,
                                    "new": "  int value() { return 1; }\n",
                                    "method_name": "value",
                                    "intent": "use expected semantic constant",
                                }
                            ],
                            "patch_style": "expected-constant",
                        }
                    ),
                ]

            def enabled(self):
                return True

            def complete(self, messages, temperature=0.0):
                self.usage.calls += 1
                self.prompts.append(messages[-1]["content"])
                return self.responses.pop(0)

        class FakeClient:
            binary = "defects4j"

            def run(self, command, cwd, check=False):
                if "compile" in command:
                    return CommandResult(command, str(cwd), 0, "compile ok", 0.0)
                source = Path(cwd) / "src/main/java/pkg/Foo.java"
                if "return 1;" in source.read_text(encoding="utf-8"):
                    return CommandResult(command, str(cwd), 0, "Failing tests: 0\n", 0.0)
                failure = "Failing tests: 1\n--- pkg.FooTest::testBug\nAssertionError: expected:<1> but was:<2>\n"
                return CommandResult(command, str(cwd), 0, failure, 0.0)

        class FakeRunner(Defects4JBenchmarkRunner):
            def _fresh_checkout(self, case, workdir, trace):
                if workdir.exists():
                    shutil.rmtree(workdir)
                source = workdir / "src/main/java/pkg/Foo.java"
                source.parent.mkdir(parents=True)
                source.write_text("package pkg;\nclass Foo {\n  int value() { return 0; }\n}\n", encoding="utf-8")
                metadata = {
                    "dir.src.classes": "src/main/java",
                    "dir.src.tests": "src/test/java",
                    "tests.trigger": "pkg.FooTest::testBug",
                    "tests.relevant": "pkg.FooTest",
                    "classes.modified": "pkg.Foo",
                }
                failure = "Failing tests: 1\n--- pkg.FooTest::testBug\nAssertionError: expected:<1> but was:<0>\n"
                return metadata, failure, ScopeResult(
                    "trigger",
                    False,
                    [CommandResult(["defects4j", "test", "-t", "pkg.FooTest::testBug"], str(workdir), 0, failure, 0.0)],
                )

        llm = FakeLLM()
        with tempfile.TemporaryDirectory() as tmp:
            case = load_cases(Path("configs/defects4j_smoke.json"))[0]
            runner = FakeRunner(
                cases=[case],
                run_dir=Path(tmp) / "run",
                systems=["self_evolved"],
                max_attempts=2,
                client=FakeClient(),
                llm=llm,
                tuning=RuntimeTuning(attempt_limits={"self_evolved": 2}, max_non_patch_rounds=2),
            )
            runner._prepare_dirs()
            metrics, trace = runner._run_case("self_evolved", case, BenchmarkMemory())
        self.assertFalse(trace["attempts"][0]["visible_tests"]["passed"])
        self.assertEqual(trace["attempts"][1]["patch_attempt"], 2)
        self.assertTrue(metrics.pass_at_3)
        self.assertIn("value should be one", llm.prompts[1])
        self.assertRegex(llm.prompts[1].lower(), r"alternative|different semantic|different .*strategy")


class DeepSeekUsageTest(unittest.TestCase):
    def test_default_model_is_deepseek_v4_pro(self):
        client = DeepSeekChatClient()
        self.assertEqual(client.model, "deepseek-v4-pro")

    def test_default_timeout_is_relaxed(self):
        client = DeepSeekChatClient()
        self.assertGreaterEqual(client.timeout, 300)

    def test_default_retries_do_not_hide_runner_level_failures(self):
        client = DeepSeekChatClient()
        self.assertEqual(client.retries, 0)
        self.assertGreaterEqual(client.transport_retries, 1)

    def test_cost_proxy_uses_env_rates(self):
        import os

        old_input = os.environ.get("DEEPSEEK_INPUT_USD_PER_MTOK")
        old_output = os.environ.get("DEEPSEEK_OUTPUT_USD_PER_MTOK")
        try:
            os.environ["DEEPSEEK_INPUT_USD_PER_MTOK"] = "1.0"
            os.environ["DEEPSEEK_OUTPUT_USD_PER_MTOK"] = "2.0"
            client = DeepSeekChatClient()
            client.record_usage({"prompt_tokens": 1000, "completion_tokens": 500})
            self.assertEqual(client.usage.calls, 1)
            self.assertAlmostEqual(client.usage.estimated_cost_usd, 0.002)
        finally:
            if old_input is None:
                os.environ.pop("DEEPSEEK_INPUT_USD_PER_MTOK", None)
            else:
                os.environ["DEEPSEEK_INPUT_USD_PER_MTOK"] = old_input
            if old_output is None:
                os.environ.pop("DEEPSEEK_OUTPUT_USD_PER_MTOK", None)
            else:
                os.environ["DEEPSEEK_OUTPUT_USD_PER_MTOK"] = old_output

    def test_deepseek_child_deadline(self):
        import os

        old_key = os.environ.get("DEEPSEEK_API_KEY")
        old_child = llm_module._DEEPSEEK_CHILD
        try:
            os.environ["DEEPSEEK_API_KEY"] = "test-key"
            llm_module._DEEPSEEK_CHILD = "import time; time.sleep(5)"
            client = DeepSeekChatClient(timeout=1)
            with self.assertRaises(TimeoutError):
                client.complete([{"role": "user", "content": "return json"}])
        finally:
            llm_module._DEEPSEEK_CHILD = old_child
            if old_key is None:
                os.environ.pop("DEEPSEEK_API_KEY", None)
            else:
                os.environ["DEEPSEEK_API_KEY"] = old_key

    def test_deepseek_deadline_uses_request_timeout(self):
        old_child = llm_module._DEEPSEEK_CHILD
        try:
            llm_module._DEEPSEEK_CHILD = "import time; time.sleep(5)"
            client = DeepSeekChatClient(timeout=5)
            spec = {
                "api_key_env": "DEEPSEEK_API_KEY",
                "base_url": "https://example.invalid",
                "timeout": 0.25,
                "body": {"model": "deepseek-v4-pro", "messages": []},
            }
            with self.assertRaisesRegex(TimeoutError, "0.25s deadline"):
                client._post_with_deadline(spec)
        finally:
            llm_module._DEEPSEEK_CHILD = old_child

    def test_deepseek_payload_requests_json_object(self):
        import os

        captured = {}
        old_key = os.environ.get("DEEPSEEK_API_KEY")
        try:
            os.environ["DEEPSEEK_API_KEY"] = "test-key"
            client = DeepSeekChatClient(timeout=1)

            def fake_post(spec):
                captured.update(spec)
                return {"usage": {"prompt_tokens": 1, "completion_tokens": 2}, "choices": [{"message": {"content": "{}"}}]}

            client._post_with_deadline = fake_post
            self.assertEqual(client.complete([{"role": "user", "content": "x"}]), "{}")
        finally:
            if old_key is None:
                os.environ.pop("DEEPSEEK_API_KEY", None)
            else:
                os.environ["DEEPSEEK_API_KEY"] = old_key
        self.assertEqual(captured["body"]["response_format"], {"type": "json_object"})

    def test_deepseek_empty_content_retries_and_records_usage(self):
        import os

        old_key = os.environ.get("DEEPSEEK_API_KEY")
        old_retries = os.environ.get("DEEPSEEK_RETRIES")
        old_backoff = os.environ.get("DEEPSEEK_RETRY_BACKOFF")
        try:
            os.environ["DEEPSEEK_API_KEY"] = "test-key"
            os.environ["DEEPSEEK_RETRIES"] = "1"
            os.environ["DEEPSEEK_RETRY_BACKOFF"] = "0"
            client = DeepSeekChatClient(timeout=1)
            responses = [
                {"usage": {"prompt_tokens": 3, "completion_tokens": 1}, "choices": [{"message": {"content": ""}}]},
                {"usage": {"prompt_tokens": 5, "completion_tokens": 2}, "choices": [{"message": {"content": "{}"}}]},
            ]

            def fake_post(spec):
                return responses.pop(0)

            client._post_with_deadline = fake_post
            self.assertEqual(client.complete([{"role": "user", "content": "x"}]), "{}")
            self.assertEqual(client.usage.calls, 2)
            self.assertEqual(client.usage.prompt_tokens, 8)
            self.assertEqual(client.usage.completion_tokens, 3)
        finally:
            if old_key is None:
                os.environ.pop("DEEPSEEK_API_KEY", None)
            else:
                os.environ["DEEPSEEK_API_KEY"] = old_key
            if old_retries is None:
                os.environ.pop("DEEPSEEK_RETRIES", None)
            else:
                os.environ["DEEPSEEK_RETRIES"] = old_retries
            if old_backoff is None:
                os.environ.pop("DEEPSEEK_RETRY_BACKOFF", None)
            else:
                os.environ["DEEPSEEK_RETRY_BACKOFF"] = old_backoff

    def test_deepseek_empty_content_can_extract_json_fallback(self):
        import os

        old_key = os.environ.get("DEEPSEEK_API_KEY")
        try:
            os.environ["DEEPSEEK_API_KEY"] = "test-key"
            client = DeepSeekChatClient(timeout=1)

            def fake_post(spec):
                return {
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "reasoning_content": (
                                    "analysis text that must not be returned "
                                    '{"diagnosis":"boundary","patch_hunks":[]}'
                                ),
                            }
                        }
                    ],
                }

            client._post_with_deadline = fake_post
            self.assertEqual(client.complete([{"role": "user", "content": "x"}]), '{"diagnosis":"boundary","patch_hunks":[]}')
            self.assertEqual(client.usage.calls, 1)
            self.assertEqual(client.usage.prompt_tokens, 7)
        finally:
            if old_key is None:
                os.environ.pop("DEEPSEEK_API_KEY", None)
            else:
                os.environ["DEEPSEEK_API_KEY"] = old_key

    def test_deepseek_empty_json_mode_response_falls_back_without_json_mode(self):
        import os

        old_key = os.environ.get("DEEPSEEK_API_KEY")
        old_retries = os.environ.get("DEEPSEEK_RETRIES")
        try:
            os.environ["DEEPSEEK_API_KEY"] = "test-key"
            os.environ["DEEPSEEK_RETRIES"] = "0"
            client = DeepSeekChatClient(timeout=1)
            seen_response_formats = []

            def fake_post(spec):
                seen_response_formats.append(spec["body"].get("response_format"))
                if len(seen_response_formats) == 1:
                    return {"usage": {"prompt_tokens": 11, "completion_tokens": 0}, "choices": [{"message": {"content": ""}}]}
                return {"usage": {"prompt_tokens": 11, "completion_tokens": 4}, "choices": [{"message": {"content": "{}"}}]}

            client._post_with_deadline = fake_post
            self.assertEqual(client.complete([{"role": "user", "content": "x"}]), "{}")
            self.assertEqual(seen_response_formats, [{"type": "json_object"}, None])
            self.assertEqual(client.usage.calls, 2)
            self.assertEqual(client.usage.prompt_tokens, 22)
        finally:
            if old_key is None:
                os.environ.pop("DEEPSEEK_API_KEY", None)
            else:
                os.environ["DEEPSEEK_API_KEY"] = old_key
            if old_retries is None:
                os.environ.pop("DEEPSEEK_RETRIES", None)
            else:
                os.environ["DEEPSEEK_RETRIES"] = old_retries

    def test_deepseek_empty_content_uses_compact_recovery_call(self):
        import os

        old_key = os.environ.get("DEEPSEEK_API_KEY")
        old_retries = os.environ.get("DEEPSEEK_RETRIES")
        old_empty_retries = os.environ.get("DEEPSEEK_EMPTY_CONTENT_RETRIES")
        old_empty_max_tokens = os.environ.get("DEEPSEEK_EMPTY_RETRY_MAX_TOKENS")
        old_empty_timeout = os.environ.get("DEEPSEEK_EMPTY_RETRY_TIMEOUT")
        old_empty_prompt_chars = os.environ.get("DEEPSEEK_EMPTY_RETRY_PROMPT_CHARS")
        try:
            os.environ["DEEPSEEK_API_KEY"] = "test-key"
            os.environ["DEEPSEEK_RETRIES"] = "0"
            os.environ["DEEPSEEK_EMPTY_CONTENT_RETRIES"] = "1"
            os.environ["DEEPSEEK_EMPTY_RETRY_MAX_TOKENS"] = "16000"
            os.environ["DEEPSEEK_EMPTY_RETRY_TIMEOUT"] = "9"
            os.environ["DEEPSEEK_EMPTY_RETRY_PROMPT_CHARS"] = "5000"
            client = DeepSeekChatClient(timeout=1, max_tokens=2048)
            seen = []
            large_prompt = json.dumps(
                {
                    "source_snippets": {"src/main/java/Foo.java": "A" * 20000},
                    "source_snippet_line_numbers": {"src/main/java/Foo.java": "1: " + "B" * 12000},
                    "failing_output": "C" * 6000,
                    "current_diff": "D" * 6000,
                    "required_json_schema": {"patch_hunks": []},
                }
            )

            def fake_post(spec):
                seen.append(
                    {
                        "timeout": spec.get("timeout"),
                        "response_format": spec["body"].get("response_format"),
                        "max_tokens": spec["body"].get("max_tokens"),
                        "messages": spec["body"].get("messages"),
                    }
                )
                if len(seen) < 3:
                    return {"usage": {"prompt_tokens": 17, "completion_tokens": 0}, "choices": [{"message": {"content": ""}}]}
                return {"usage": {"prompt_tokens": 19, "completion_tokens": 5}, "choices": [{"message": {"content": "{}"}}]}

            client._post_with_deadline = fake_post
            self.assertEqual(
                client.complete(
                    [
                        {"role": "system", "content": "return json"},
                        {"role": "user", "content": large_prompt},
                    ]
                ),
                "{}",
            )
            self.assertEqual([item["response_format"] for item in seen], [{"type": "json_object"}, None, None])
            self.assertEqual([item["max_tokens"] for item in seen], [2048, 2048, 16000])
            self.assertEqual([item["timeout"] for item in seen], [1, 1, 9])
            self.assertEqual(len(seen[2]["messages"]), 3)
            self.assertLess(len(seen[2]["messages"][1]["content"]), len(large_prompt))
            self.assertLessEqual(len(seen[2]["messages"][1]["content"]), 5000)
            self.assertIn("empty final content", seen[2]["messages"][-1]["content"])
            self.assertEqual(client.usage.calls, 3)
        finally:
            if old_key is None:
                os.environ.pop("DEEPSEEK_API_KEY", None)
            else:
                os.environ["DEEPSEEK_API_KEY"] = old_key
            if old_retries is None:
                os.environ.pop("DEEPSEEK_RETRIES", None)
            else:
                os.environ["DEEPSEEK_RETRIES"] = old_retries
            if old_empty_retries is None:
                os.environ.pop("DEEPSEEK_EMPTY_CONTENT_RETRIES", None)
            else:
                os.environ["DEEPSEEK_EMPTY_CONTENT_RETRIES"] = old_empty_retries
            if old_empty_max_tokens is None:
                os.environ.pop("DEEPSEEK_EMPTY_RETRY_MAX_TOKENS", None)
            else:
                os.environ["DEEPSEEK_EMPTY_RETRY_MAX_TOKENS"] = old_empty_max_tokens
            if old_empty_timeout is None:
                os.environ.pop("DEEPSEEK_EMPTY_RETRY_TIMEOUT", None)
            else:
                os.environ["DEEPSEEK_EMPTY_RETRY_TIMEOUT"] = old_empty_timeout
            if old_empty_prompt_chars is None:
                os.environ.pop("DEEPSEEK_EMPTY_RETRY_PROMPT_CHARS", None)
            else:
                os.environ["DEEPSEEK_EMPTY_RETRY_PROMPT_CHARS"] = old_empty_prompt_chars

    def test_deepseek_transport_errors_retry_without_consuming_runner_rounds(self):
        import os

        old_key = os.environ.get("DEEPSEEK_API_KEY")
        old_transport_retries = os.environ.get("DEEPSEEK_TRANSPORT_RETRIES")
        old_backoff = os.environ.get("DEEPSEEK_TRANSPORT_RETRY_BACKOFF")
        try:
            os.environ["DEEPSEEK_API_KEY"] = "test-key"
            os.environ["DEEPSEEK_TRANSPORT_RETRIES"] = "2"
            os.environ["DEEPSEEK_TRANSPORT_RETRY_BACKOFF"] = "0"
            client = DeepSeekChatClient(timeout=1)
            calls = {"count": 0}

            def fake_post(spec):
                calls["count"] += 1
                if calls["count"] < 3:
                    raise RuntimeError("URLError: <urlopen error [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred>")
                return {"usage": {"prompt_tokens": 13, "completion_tokens": 5}, "choices": [{"message": {"content": "{}"}}]}

            client._post_with_deadline = fake_post
            self.assertEqual(client.complete([{"role": "user", "content": "x"}]), "{}")
            self.assertEqual(calls["count"], 3)
            self.assertEqual(client.usage.calls, 1)
            self.assertEqual(client.usage.prompt_tokens, 13)
        finally:
            if old_key is None:
                os.environ.pop("DEEPSEEK_API_KEY", None)
            else:
                os.environ["DEEPSEEK_API_KEY"] = old_key
            if old_transport_retries is None:
                os.environ.pop("DEEPSEEK_TRANSPORT_RETRIES", None)
            else:
                os.environ["DEEPSEEK_TRANSPORT_RETRIES"] = old_transport_retries
            if old_backoff is None:
                os.environ.pop("DEEPSEEK_TRANSPORT_RETRY_BACKOFF", None)
            else:
                os.environ["DEEPSEEK_TRANSPORT_RETRY_BACKOFF"] = old_backoff


class CarriedGroundedHunksMergeTest(unittest.TestCase):
    def _make_hunk(self, file, line_start, line_end, new="pass", old="old"):
        return PatchHunk(
            file=file,
            old=old,
            new=new,
            line_start=line_start,
            line_end=line_end,
            original_line_start=line_start,
            original_line_end=line_end,
            range_grounded=True,
            method_name="",
            intent="",
        )

    def test_same_file_non_overlapping_carried_kept(self):
        carried = [self._make_hunk("Foo.java", 10, 15)]
        plan = RepairPlan(
            patch_hunks=[self._make_hunk("Foo.java", 40, 45)],
            diagnosis="",
            files_to_read=[],
            tests_to_run_next=[],
            confidence=0.5,
            final_explanation="",
            patch_style="",
        )
        merged = _with_carried_grounded_hunks(plan, carried)
        self.assertEqual(len(merged.patch_hunks), 2)
        self.assertEqual(merged.patch_hunks[0].line_start, 10)
        self.assertEqual(merged.patch_hunks[1].line_start, 40)

    def test_same_file_overlapping_new_wins(self):
        carried = [self._make_hunk("Foo.java", 10, 20)]
        plan = RepairPlan(
            patch_hunks=[self._make_hunk("Foo.java", 15, 25)],
            diagnosis="",
            files_to_read=[],
            tests_to_run_next=[],
            confidence=0.5,
            final_explanation="",
            patch_style="",
        )
        merged = _with_carried_grounded_hunks(plan, carried)
        self.assertEqual(len(merged.patch_hunks), 1)
        self.assertEqual(merged.patch_hunks[0].line_start, 15)

    def test_different_file_carried_kept(self):
        carried = [self._make_hunk("A.java", 1, 5)]
        plan = RepairPlan(
            patch_hunks=[self._make_hunk("B.java", 10, 15)],
            diagnosis="",
            files_to_read=[],
            tests_to_run_next=[],
            confidence=0.5,
            final_explanation="",
            patch_style="",
        )
        merged = _with_carried_grounded_hunks(plan, carried)
        self.assertEqual(len(merged.patch_hunks), 2)
        self.assertEqual(merged.patch_hunks[0].file, "A.java")
        self.assertEqual(merged.patch_hunks[1].file, "B.java")

    def test_grounding_error_identity(self):
        self.assertEqual(
            _grounding_error_identity("Foo.java:10-20: ValueError"),
            ("Foo.java", 10, 20),
        )
        self.assertEqual(
            _grounding_error_identity("Bad format"),
            ("Bad format", -1, -1),
        )

    def test_hunk_ranges_overlap(self):
        a = self._make_hunk("Foo.java", 10, 20)
        b = self._make_hunk("Foo.java", 15, 25)
        self.assertTrue(_hunk_ranges_overlap(a, b))
        c = self._make_hunk("Foo.java", 25, 30)
        self.assertFalse(_hunk_ranges_overlap(a, c))


if __name__ == "__main__":
    unittest.main()
