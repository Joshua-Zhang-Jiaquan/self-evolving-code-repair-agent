"""Run a deterministic proof experiment for two-dimensional memory.

This is not a replacement for the real Defects4J benchmark. It is a fast,
model-free controller experiment that uses the same BenchmarkMemory tables as
the real runner to show how check memory and repair memory affect behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .d4j_memory import BenchmarkMemory


VARIANTS = {
    "feedback_only": (False, False),
    "check_memory_only": (True, False),
    "repair_memory_only": (False, True),
    "two_dimensional_memory": (True, True),
}


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    features: List[str]
    required_scope: str
    required_patch_style: str
    bad_patch_style: str


TRAIN_CASES = [
    SyntheticCase("train-regex-parser", ["project:Lang", "class:fastdateparser"], "relevant", "regex-boundary", "literal-space-relaxation"),
    SyntheticCase("train-numeric-types", ["project:Lang", "class:numberutils"], "trigger", "numeric-conversion", "float-before-double"),
    SyntheticCase("train-numeric-types-repeat", ["project:Lang", "class:numberutils"], "trigger", "numeric-conversion", "float-before-double"),
    SyntheticCase("train-math-boundary", ["project:Math", "exception:assertionerror"], "relevant", "boundary-condition", "constant-return"),
    SyntheticCase("train-api-contract", ["project:Chart", "class:renderer"], "all", "api-contract", "delete-validation"),
]

EVAL_CASES = [
    SyntheticCase("eval-regex-parser-transfer", ["project:Lang", "class:fastdateparser"], "relevant", "regex-boundary", "literal-space-relaxation"),
    SyntheticCase("eval-numeric-types-transfer", ["project:Lang", "class:numberutils"], "trigger", "numeric-conversion", "float-before-double"),
    SyntheticCase("eval-math-boundary-transfer", ["project:Math", "exception:assertionerror"], "relevant", "boundary-condition", "constant-return"),
    SyntheticCase("eval-api-contract-transfer", ["project:Chart", "class:renderer"], "all", "api-contract", "delete-validation"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/proof_experiments/two_dimensional_memory"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    summary: Dict[str, Dict[str, object]] = {}
    for variant, (use_check_memory, use_repair_memory) in VARIANTS.items():
        memory = BenchmarkMemory()
        for case in TRAIN_CASES:
            train_case(memory, case, use_check_memory=use_check_memory, use_repair_memory=use_repair_memory)
        memory.save(args.out_dir / f"{variant}_memory_after_train.json")
        rows = [
            evaluate_case(memory, case, variant, use_check_memory=use_check_memory, use_repair_memory=use_repair_memory)
            for case in EVAL_CASES
        ]
        all_rows.extend(rows)
        summary[variant] = summarize(rows)

    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.out_dir / "metrics.csv", all_rows)
    write_report(args.out_dir / "analysis.md", summary)
    print(json.dumps({"out_dir": str(args.out_dir), "summary": summary}, ensure_ascii=False))


def train_case(memory: BenchmarkMemory, case: SyntheticCase, *, use_check_memory: bool, use_repair_memory: bool) -> None:
    bad_patch_reaches_regression = case.required_scope != "trigger"
    memory.update(
        features=case.features,
        patch_style=case.bad_patch_style,
        test_scope="trigger",
        solved=False,
        failure_reason="regression_failure" if bad_patch_reaches_regression else "visible_failure",
        reflection=(
            f"{case.bad_patch_style} passed visible checks but failed regression"
            if bad_patch_reaches_regression
            else f"{case.bad_patch_style} failed the trigger test and should not be retried"
        ),
        repair_skill=f"avoid:{case.bad_patch_style}",
        test_skill=f"use:{case.required_scope}",
        visible_passed=bad_patch_reaches_regression,
        regression_checked=bad_patch_reaches_regression,
        regression_passed=False,
        update_check_memory=use_check_memory,
        update_repair_memory=use_repair_memory,
    )
    memory.update(
        features=case.features,
        patch_style=case.required_patch_style,
        test_scope=case.required_scope,
        solved=True,
        failure_reason=None,
        reflection=None,
        repair_skill=f"use:{case.required_patch_style}",
        test_skill=f"use:{case.required_scope}",
        visible_passed=True,
        regression_checked=True,
        regression_passed=True,
        success_strategy=f"Use {case.required_patch_style} and validate with {case.required_scope}",
        update_check_memory=use_check_memory,
        update_repair_memory=use_repair_memory,
    )


def evaluate_case(
    memory: BenchmarkMemory,
    case: SyntheticCase,
    variant: str,
    *,
    use_check_memory: bool,
    use_repair_memory: bool,
) -> Dict[str, object]:
    selected_scope = memory.preferred_test_scope(case.features, default="trigger") if use_check_memory else "trigger"
    repair_preferences = memory.prompt_preferences(case.features) if use_repair_memory else []
    selected_patch_style = repair_preferences[0] if repair_preferences else "direct-default"
    check_correct = selected_scope == case.required_scope
    repair_correct = selected_patch_style == case.required_patch_style
    solved = check_correct and repair_correct
    duplicate_risk = selected_patch_style in {case.bad_patch_style, "direct-default"}
    regression_overfit_risk = selected_scope == "trigger" and case.required_scope in {"relevant", "all"}
    return {
        "variant": variant,
        "case_id": case.case_id,
        "selected_scope": selected_scope,
        "required_scope": case.required_scope,
        "selected_patch_style": selected_patch_style,
        "required_patch_style": case.required_patch_style,
        "check_correct": check_correct,
        "repair_correct": repair_correct,
        "solved": solved,
        "duplicate_risk": duplicate_risk,
        "regression_overfit_risk": regression_overfit_risk,
        "simulated_tool_calls": 3 if check_correct else 5,
        "simulated_test_runs": 1 if selected_scope == "trigger" else 2,
    }


def summarize(rows: Iterable[Dict[str, object]]) -> Dict[str, object]:
    rows = list(rows)
    total = len(rows) or 1
    return {
        "cases": len(rows),
        "solved": sum(1 for row in rows if row["solved"]),
        "pass_rate": round(sum(1 for row in rows if row["solved"]) / total, 4),
        "check_correct_rate": round(sum(1 for row in rows if row["check_correct"]) / total, 4),
        "repair_correct_rate": round(sum(1 for row in rows if row["repair_correct"]) / total, 4),
        "duplicate_risk_count": sum(1 for row in rows if row["duplicate_risk"]),
        "regression_overfit_risk_count": sum(1 for row in rows if row["regression_overfit_risk"]),
        "avg_tool_calls": round(sum(int(row["simulated_tool_calls"]) for row in rows) / total, 4),
        "avg_test_runs": round(sum(int(row["simulated_test_runs"]) for row in rows) / total, 4),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: Dict[str, Dict[str, object]]) -> None:
    lines = [
        "# Two-Dimensional Memory Proof Experiment",
        "",
        "This deterministic experiment uses the real BenchmarkMemory update/retrieval code.",
        "It isolates four variants: feedback-only, check-memory-only, repair-memory-only, and full two-dimensional memory.",
        "",
        "| Variant | Solved | Check Correct | Repair Correct | Duplicate Risk | Regression Overfit Risk | Avg Tool Calls | Avg Test Runs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, item in summary.items():
        lines.append(
            f"| `{variant}` | {item['solved']}/{item['cases']} | {item['check_correct_rate']} | "
            f"{item['repair_correct_rate']} | {item['duplicate_risk_count']} | "
            f"{item['regression_overfit_risk_count']} | {item['avg_tool_calls']} | {item['avg_test_runs']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: check memory improves what evidence/tests are selected; repair memory improves patch-style selection; "
            "the two-dimensional variant combines both benefits.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
