"""Run trace-based experiments from existing v16 D4J benchmark traces.

Experiments: E1, E2, E3, E4, B2, B3, F1, F3
No DeepSeek or Docker required — analyzes existing trace files only.
"""
import json, math, os, sys
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

TRACE_DIR = Path("artifacts/runs/d4j30-self-evolved-real-v16/traces")
RUN_DIR = Path("artifacts/runs/d4j30-self-evolved-real-v16")
OUT_DIR = Path("artifacts/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_traces():
    traces = {}
    for f in sorted(TRACE_DIR.glob("*.json")):
        t = json.loads(f.read_text())
        case_id = t["case"]["case_id"]
        traces[case_id] = t
    return traces


def visible_passed(attempt):
    vt = attempt.get("visible_tests")
    return vt and vt.get("passed", False)


def regression_data(attempt):
    rt = attempt.get("regression_tests")
    if rt is None:
        return None
    return rt.get("passed", None)


def patch_style(attempt):
    rp = attempt.get("repair_plan")
    if rp:
        return rp.get("patch_style", "unknown")
    return "no-plan"


def run_e1_regression_catch(traces):
    """E1: How often does the Examiner catch Solver overfits?"""
    total_visible_pass = 0
    regression_caught = 0
    regression_passed = 0
    trigger_only_would_miss = 0

    for cid, t in traces.items():
        for a in t.get("attempts", []):
            if a.get("patch_attempt") is None:
                continue
            if not visible_passed(a):
                continue
            total_visible_pass += 1
            reg = regression_data(a)
            if reg is False:
                regression_caught += 1
            elif reg is True:
                regression_passed += 1

    return {
        "total_visible_pass_attempts": total_visible_pass,
        "regression_caught_overfit": regression_caught,
        "regression_passed": regression_passed,
        "catch_rate": round(regression_caught / max(1, total_visible_pass), 4),
        "interpretation": f"Examiner caught {regression_caught} overfits out of {total_visible_pass} visible-pass attempts ({regression_caught/max(1,total_visible_pass)*100:.1f}%)",
    }


def run_e2_post_regression_recovery(traces):
    """E2: After Examiner catches overfit, does Solver recover?"""
    regression_catches = []
    recoveries = 0
    never_recovered = 0
    attempts_to_recover = []

    for cid, t in traces.items():
        attempts = [a for a in t.get("attempts", []) if a.get("patch_attempt") is not None]
        for i, a in enumerate(attempts):
            if visible_passed(a) and regression_data(a) is False:
                regression_catches.append((cid, i))
                solved = t["metrics"].get("status") == "solved"
                if solved:
                    recoveries += 1
                    remaining = len(attempts) - i - 1
                    attempts_to_recover.append(remaining)
                else:
                    never_recovered += 1

    avg_recovery = sum(attempts_to_recover) / max(1, len(attempts_to_recover))
    return {
        "regression_catch_events": len(regression_catches),
        "eventually_solved": recoveries,
        "never_recovered": never_recovered,
        "post_regression_solve_rate": round(recoveries / max(1, len(regression_catches)), 4),
        "avg_additional_attempts_to_solve": round(avg_recovery, 2),
        "catch_case_ids": [c for c, _ in regression_catches],
    }


def run_e3_scope_escalation(traces):
    """E3: Does the Examiner's scope choice evolve over the run?"""
    case_order = list(traces.keys())
    scope_by_third = {"first_10": Counter(), "mid_10": Counter(), "last_10": Counter()}

    for idx, cid in enumerate(case_order):
        t = traces[cid]
        third = "first_10" if idx < 10 else ("mid_10" if idx < 20 else "last_10")
        for a in t.get("attempts", []):
            if a.get("patch_attempt") is None:
                continue
            if visible_passed(a):
                rt = a.get("regression_tests")
                if rt:
                    scope = rt.get("scope", "unknown")
                    scope_by_third[third][scope] += 1

    return {
        "scope_distribution_by_run_third": {k: dict(v) for k, v in scope_by_third.items()},
        "interpretation": "If 'relevant' or 'all' scope increases in later thirds, Examiner learned to escalate",
    }


def run_e4_strategy_diversity(traces):
    """E4: Solver patch style entropy over time."""
    case_order = list(traces.keys())
    windows = {"bugs_1_10": [], "bugs_11_20": [], "bugs_21_30": []}

    for idx, cid in enumerate(case_order):
        t = traces[cid]
        window = "bugs_1_10" if idx < 10 else ("bugs_11_20" if idx < 20 else "bugs_21_30")
        for a in t.get("attempts", []):
            if a.get("patch_attempt") is not None and a.get("patch_apply", {}).get("ok"):
                windows[window].append(patch_style(a))

    results = {}
    for window, styles in windows.items():
        c = Counter(styles)
        total = sum(c.values()) or 1
        entropy = -sum((n / total) * math.log2(n / total) for n in c.values() if n > 0)
        results[window] = {
            "unique_styles": len(c),
            "total_patches": sum(c.values()),
            "entropy_bits": round(entropy, 3),
            "top_3": c.most_common(3),
        }
    return results


def run_b2_memory_growth(traces):
    """B2: Memory growth trajectory across the 30-bug run."""
    case_order = list(traces.keys())
    growth = []

    prev_features = 0
    for idx, cid in enumerate(case_order):
        t = traces[cid]
        mem_after = t.get("memory_after", {})
        patch_features = len(mem_after.get("patch_ranking", {}))
        test_features = len(mem_after.get("test_selection", {}))
        reflections = len(mem_after.get("failure_reflections", []))
        strategies = len(mem_after.get("success_strategies", []))
        new_features = patch_features - prev_features
        prev_features = patch_features
        growth.append({
            "bug_index": idx + 1,
            "case_id": cid,
            "patch_ranking_features": patch_features,
            "test_selection_features": test_features,
            "failure_reflections": reflections,
            "success_strategies": strategies,
            "new_features_this_bug": new_features,
        })

    first = growth[0] if growth else {}
    last = growth[-1] if growth else {}
    return {
        "trajectory": growth,
        "summary": {
            "first_bug_features": first.get("patch_ranking_features", 0),
            "last_bug_features": last.get("patch_ranking_features", 0),
            "first_bug_reflections": first.get("failure_reflections", 0),
            "last_bug_reflections": last.get("failure_reflections", 0),
            "first_bug_strategies": first.get("success_strategies", 0),
            "last_bug_strategies": last.get("success_strategies", 0),
        },
    }


def run_b3_transfer_benefit(traces):
    """B3: Correlation between feature overlap / retrieval richness and solve outcome."""
    case_order = list(traces.keys())
    seen_features = set()
    results = []

    for idx, cid in enumerate(case_order):
        t = traces[cid]
        mem_before = t.get("memory_before", {})
        mem_features = set(mem_before.get("patch_ranking", {}).keys())
        overlap = len(mem_features & seen_features) if seen_features else 0

        first_attempt = None
        for a in t.get("attempts", []):
            if a.get("patch_attempt") is not None:
                first_attempt = a
                break

        richness = 0
        if first_attempt:
            richness = (
                len(first_attempt.get("memory_preferences", []))
                + len(first_attempt.get("repair_skills", []))
                + len(first_attempt.get("test_skills", []))
                + len(first_attempt.get("regression_warnings", []))
                + len(first_attempt.get("success_strategies", []))
                + len(first_attempt.get("reflections", []))
            )

        solved = t["metrics"].get("status") == "solved"
        results.append({
            "bug_index": idx + 1,
            "case_id": cid,
            "feature_overlap": overlap,
            "retrieval_richness": richness,
            "solved": solved,
        })

        # Update seen features with this case's memory_after features
        mem_after = t.get("memory_after", {})
        seen_features.update(mem_after.get("patch_ranking", {}).keys())

    # Compute correlation
    solved_richness = [r["retrieval_richness"] for r in results if r["solved"]]
    unsolved_richness = [r["retrieval_richness"] for r in results if not r["solved"]]
    solved_overlap = [r["feature_overlap"] for r in results if r["solved"]]
    unsolved_overlap = [r["feature_overlap"] for r in results if not r["solved"]]

    def avg(lst):
        return round(sum(lst) / max(1, len(lst)), 2)

    return {
        "per_bug": results,
        "summary": {
            "solved_count": len(solved_richness),
            "unsolved_count": len(unsolved_richness),
            "avg_retrieval_richness_solved": avg(solved_richness),
            "avg_retrieval_richness_unsolved": avg(unsolved_richness),
            "avg_feature_overlap_solved": avg(solved_overlap),
            "avg_feature_overlap_unsolved": avg(unsolved_overlap),
        },
    }


def run_f1_call_efficiency(traces):
    """F1: LLM call efficiency over time."""
    case_order = list(traces.keys())
    windows = {"bugs_1_10": [], "bugs_11_20": [], "bugs_21_30": []}

    for idx, cid in enumerate(case_order):
        t = traces[cid]
        window = "bugs_1_10" if idx < 10 else ("bugs_11_20" if idx < 20 else "bugs_21_30")
        calls = t["metrics"].get("deepseek_calls", 0)
        tokens = t["metrics"].get("prompt_tokens", 0) + t["metrics"].get("completion_tokens", 0)
        windows[window].append({"case_id": cid, "calls": calls, "tokens": tokens})

    results = {}
    for window, items in windows.items():
        calls = [i["calls"] for i in items]
        tokens = [i["tokens"] for i in items]
        results[window] = {
            "avg_calls": round(sum(calls) / max(1, len(calls)), 2),
            "avg_tokens": round(sum(tokens) / max(1, len(tokens)), 0),
            "total_calls": sum(calls),
        }
    return results


def run_f3_dedup_effectiveness(traces):
    """F3: Candidate dedup — how many duplicates rejected?"""
    total_rejections = 0
    total_attempts = 0
    rejection_cases = []

    for cid, t in traces.items():
        case_rejections = 0
        for a in t.get("attempts", []):
            if a.get("patch_attempt") is not None:
                total_attempts += 1
            if a.get("candidate_rejected"):
                total_rejections += 1
                case_rejections += 1
            if a.get("non_patch_candidate_rejection"):
                total_rejections += 1
                case_rejections += 1
        if case_rejections:
            rejection_cases.append({"case_id": cid, "rejections": case_rejections})

    return {
        "total_patch_attempts": total_attempts,
        "total_duplicate_rejections": total_rejections,
        "rejection_rate": round(total_rejections / max(1, total_attempts + total_rejections), 4),
        "cases_with_rejections": rejection_cases,
    }


def main():
    print("Loading traces...")
    traces = load_traces()
    print(f"Loaded {len(traces)} traces\n")

    results = {}

    print("=" * 60)
    print("E1: Regression Catch Rate")
    print("=" * 60)
    e1 = run_e1_regression_catch(traces)
    results["E1_regression_catch"] = e1
    for k, v in e1.items():
        print(f"  {k}: {v}")
    print()

    print("=" * 60)
    print("E2: Post-Regression Recovery Rate")
    print("=" * 60)
    e2 = run_e2_post_regression_recovery(traces)
    results["E2_post_regression_recovery"] = e2
    for k, v in e2.items():
        if k != "catch_case_ids":
            print(f"  {k}: {v}")
    print()

    print("=" * 60)
    print("E3: Examiner Scope Escalation")
    print("=" * 60)
    e3 = run_e3_scope_escalation(traces)
    results["E3_scope_escalation"] = e3
    for k, v in e3.items():
        print(f"  {k}: {v}")
    print()

    print("=" * 60)
    print("E4: Solver Strategy Diversity")
    print("=" * 60)
    e4 = run_e4_strategy_diversity(traces)
    results["E4_strategy_diversity"] = e4
    for window, data in e4.items():
        print(f"  {window}:")
        for k, v in data.items():
            print(f"    {k}: {v}")
    print()

    print("=" * 60)
    print("B2: Memory Growth Trajectory")
    print("=" * 60)
    b2 = run_b2_memory_growth(traces)
    results["B2_memory_growth"] = b2
    s = b2["summary"]
    print(f"  Features: {s['first_bug_features']} → {s['last_bug_features']}")
    print(f"  Reflections: {s['first_bug_reflections']} → {s['last_bug_reflections']}")
    print(f"  Strategies: {s['first_bug_strategies']} → {s['last_bug_strategies']}")
    print()

    print("=" * 60)
    print("B3: Transfer Benefit Correlation")
    print("=" * 60)
    b3 = run_b3_transfer_benefit(traces)
    results["B3_transfer_benefit"] = b3
    s = b3["summary"]
    print(f"  Solved: {s['solved_count']} bugs, avg richness={s['avg_retrieval_richness_solved']}, avg overlap={s['avg_feature_overlap_solved']}")
    print(f"  Unsolved: {s['unsolved_count']} bugs, avg richness={s['avg_retrieval_richness_unsolved']}, avg overlap={s['avg_feature_overlap_unsolved']}")
    print()

    print("=" * 60)
    print("F1: LLM Call Efficiency")
    print("=" * 60)
    f1 = run_f1_call_efficiency(traces)
    results["F1_call_efficiency"] = f1
    for window, data in f1.items():
        print(f"  {window}: avg_calls={data['avg_calls']}, avg_tokens={data['avg_tokens']}")
    print()

    print("=" * 60)
    print("F3: Candidate Dedup Effectiveness")
    print("=" * 60)
    f3 = run_f3_dedup_effectiveness(traces)
    results["F3_dedup"] = f3
    for k, v in f3.items():
        if k != "cases_with_rejections":
            print(f"  {k}: {v}")
    print()

    # Save full results
    out_path = OUT_DIR / "trace_analysis_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nFull results saved to {out_path}")

    # Generate markdown report
    md_lines = [
        "# Trace-Based Experiment Results",
        "",
        f"Generated: {datetime.now().isoformat()}",
        f"Source: `artifacts/runs/d4j30-self-evolved-real-v16/`",
        f"Traces analyzed: {len(traces)}",
        "",
        "---",
        "",
        "## E1: Regression Catch Rate (Examiner Effectiveness)",
        "",
        f"| Metric | Value |",
        f"|---|---:|",
        f"| Total visible-pass attempts | {e1['total_visible_pass_attempts']} |",
        f"| Regression overfits caught | **{e1['regression_caught_overfit']}** |",
        f"| Regression passed (true solves) | {e1['regression_passed']} |",
        f"| **Catch rate** | **{e1['catch_rate']:.1%}** |",
        "",
        f"> {e1['interpretation']}",
        "",
        "---",
        "",
        "## E2: Post-Regression Recovery (Solver Resilience)",
        "",
        f"| Metric | Value |",
        f"|---|---:|",
        f"| Regression catch events | {e2['regression_catch_events']} |",
        f"| Eventually solved | **{e2['eventually_solved']}** |",
        f"| Never recovered | {e2['never_recovered']} |",
        f"| **Post-regression solve rate** | **{e2['post_regression_solve_rate']:.1%}** |",
        f"| Avg additional attempts to solve | {e2['avg_additional_attempts_to_solve']} |",
        "",
        "---",
        "",
        "## E3: Examiner Scope Escalation",
        "",
        "| Run Third | Scope Distribution |",
        "|---|---|",
    ]
    for third, dist in e3["scope_distribution_by_run_third"].items():
        md_lines.append(f"| {third} | {dist} |")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## E4: Solver Strategy Diversity")
    md_lines.append("")
    md_lines.append("| Window | Unique Styles | Total Patches | Entropy (bits) | Top 3 |")
    md_lines.append("|---|---:|---:|---:|---|")
    for window, data in e4.items():
        top3 = ", ".join(f"{s}({n})" for s, n in data["top_3"])
        md_lines.append(f"| {window} | {data['unique_styles']} | {data['total_patches']} | {data['entropy_bits']} | {top3} |")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## B2: Memory Growth Trajectory")
    md_lines.append("")
    md_lines.append("| Metric | First Bug | Last Bug | Growth |")
    md_lines.append("|---|---:|---:|---:|")
    b2s = b2["summary"]
    md_lines.append(f"| Patch ranking features | {b2s['first_bug_features']} | {b2s['last_bug_features']} | +{b2s['last_bug_features'] - b2s['first_bug_features']} |")
    md_lines.append(f"| Failure reflections | {b2s['first_bug_reflections']} | {b2s['last_bug_reflections']} | +{b2s['last_bug_reflections'] - b2s['first_bug_reflections']} |")
    md_lines.append(f"| Success strategies | {b2s['first_bug_strategies']} | {b2s['last_bug_strategies']} | +{b2s['last_bug_strategies'] - b2s['first_bug_strategies']} |")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## B3: Transfer Benefit Correlation")
    md_lines.append("")
    b3s = b3["summary"]
    md_lines.append(f"| Group | Count | Avg Retrieval Richness | Avg Feature Overlap |")
    md_lines.append(f"|---|---:|---:|---:|")
    md_lines.append(f"| **Solved** | {b3s['solved_count']} | {b3s['avg_retrieval_richness_solved']} | {b3s['avg_feature_overlap_solved']} |")
    md_lines.append(f"| Unsolved | {b3s['unsolved_count']} | {b3s['avg_retrieval_richness_unsolved']} | {b3s['avg_feature_overlap_unsolved']} |")
    md_lines.append("")
    if b3s["avg_retrieval_richness_solved"] > b3s["avg_retrieval_richness_unsolved"]:
        md_lines.append("> Solved bugs had **higher memory retrieval richness** — evidence of transfer benefit.")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## F1: LLM Call Efficiency Over Time")
    md_lines.append("")
    md_lines.append("| Window | Avg Calls/Bug | Avg Tokens/Bug | Total Calls |")
    md_lines.append("|---|---:|---:|---:|")
    for window, data in f1.items():
        md_lines.append(f"| {window} | {data['avg_calls']} | {data['avg_tokens']:,} | {data['total_calls']} |")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## F3: Candidate Dedup Effectiveness")
    md_lines.append("")
    md_lines.append(f"| Metric | Value |")
    md_lines.append(f"|---|---:|")
    md_lines.append(f"| Total patch attempts | {f3['total_patch_attempts']} |")
    md_lines.append(f"| Duplicate rejections | **{f3['total_duplicate_rejections']}** |")
    md_lines.append(f"| Rejection rate | {f3['rejection_rate']:.1%} |")
    md_lines.append(f"| Cases with rejections | {len(f3['cases_with_rejections'])} |")
    md_lines.append("")

    md_path = OUT_DIR / "trace_analysis_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Report saved to {md_path}")


if __name__ == "__main__":
    main()
