# Experiment Design: Validating the Examiner-Solver Adversarial Self-Improving System

## Overview

Seven experiment groups, 23 experiments total, designed to answer one question: **does the adversarial self-improving architecture actually produce better repair than alternatives, and can we prove why?**

### Current Validation Gaps

| What exists | What's missing |
|---|---|
| Synthetic 4-variant ablation (deterministic) | Same ablation on **real D4J bugs** |
| Single 30-bug run (v16) | **Learning curve** — does solve rate increase as memory accumulates? |
| No ordering analysis | **Robustness** — does bug order change results? |
| No per-store decomposition | Which of the **7 memory stores** contribute most? |
| No transfer isolation | Does **cross-project transfer** actually work or is it noise? |
| No adversarial dynamic measurement | How often does **Examiner catch Solver overfit**? Does Solver recover? |
| No efficiency analysis | Does memory **reduce cost** (calls, tokens, attempts) over time? |
| No negative transfer check | Can learning Bug A **hurt** Bug B? |

---

## Group A: Architecture Ablation (Does the Adversarial Structure Matter?)

### A1: Full 4-Variant Ablation on Real D4J Bugs

**Hypothesis**: The full two-dimensional system outperforms all three partial variants on real bugs, not just synthetic cases.

**Method**: Run the existing `run_d4j_memory_ablation_experiments.sh` script on the full 30-case config (not just failed-8). Compare four memory modes:
- `none` (feedback only, no cross-task memory)
- `check_only` (Examiner only)
- `repair_only` (Solver only)
- `full` (both dimensions)

**Metrics**: Pass@1, Pass@3, regression pass rate, duplicate rejection count, avg attempts, avg LLM calls.

**Expected**: `full` > `repair_only` > `check_only` ≈ `none` on solve rate. `check_only` should have lowest regression overfit rate. `repair_only` should have lowest duplicate rate.

**Config needed**: `configs/defects4j_30.json` with `memory_mode` override per run.

---

### A2: Per-Memory-Store Ablation (Which Stores Matter?)

**Hypothesis**: Different memory stores contribute differently; removing the most impactful store causes the largest drop.

**Method**: Create 7 ablation variants, each disabling exactly ONE of the 7 sub-memories:

| Variant | Disabled store | What breaks |
|---|---|---|
| `no_patch_ranking` | `patch_ranking` | Solver can't rank patch styles |
| `no_test_selection` | `test_selection` | Examiner can't learn test scope |
| `no_repair_skill` | `repair_skill_memory` | No skill ranking in prompt |
| `no_test_skill` | `test_skill_memory` | No test skill guidance |
| `no_regression` | `regression_outcomes` | No regression warnings |
| `no_reflections` | `failure_reflections` | No cross-task failure lessons |
| `no_strategies` | `success_strategies` | No success pattern reuse |

**Metrics**: Δ solve rate vs full system per variant. Rank stores by impact.

**Expected**: `no_patch_ranking` and `no_test_selection` cause the largest drops (they're the primary retrieval paths). `no_reflections` and `no_strategies` cause moderate drops (they enrich prompts but don't control core decisions).

**Implementation**: Add `disabled_stores` parameter to `BenchmarkMemory` that zeroes out specific tables during retrieval.

---

### A3: Feature Weighting Ablation

**Hypothesis**: The 2× weight for specific features (class/exception/trigger) vs 1× for broad (project) produces better transfer than uniform weighting.

**Method**: Run the 30-case benchmark with three `_feature_weight` configurations:
- `specific_2x` (current: class/exception/trigger = 2.0, project = 1.0)
- `uniform_1x` (all features = 1.0)
- `specific_4x` (class/exception/trigger = 4.0, project = 1.0)
- `project_only` (only project features, weight = 1.0 — effectively no specific transfer)

**Metrics**: Solve rate, cross-project transfer rate (solve rate on projects NOT seen early in the run).

**Expected**: `specific_2x` > `uniform_1x` > `project_only`. `specific_4x` may overfit to specific features and transfer worse.

---

### A4: Delta Function Sensitivity

**Hypothesis**: The asymmetric delta design (overfit = -1.0 outcome but +0.75 skill) is intentional and optimal.

**Method**: Vary the `_outcome_delta` and `_skill_delta` values:

| Variant | Change | Tests |
|---|---|---|
| `symmetric_overfit` | Overfit skill delta = -1.0 (same as outcome) | Does Solver stop learning from overfit? |
| `weak_negative` | All negative deltas halved (-0.5, -0.25, -0.125) | Does weaker punishment hurt? |
| `strong_positive` | Solved delta = +3.0 | Does stronger reward help or cause recency bias? |
| `no_asymmetry` | outcome_delta == skill_delta everywhere | Does removing the design insight hurt? |

**Metrics**: Solve rate, overfit catch rate, strategy diversity.

**Expected**: `no_asymmetry` underperforms because the system can't distinguish "wrong patch" from "wrong skill." `symmetric_overfit` underperforms because Solver stops learning from caught overfits.

---

## Group B: Cumulative Learning (Does It Actually Self-Evolve?)

### B1: Learning Curve Analysis

**Hypothesis**: Solve rate increases as the system processes more bugs — proving true self-evolution, not just per-bug repair.

**Method**: Run the 30-case benchmark with checkpoints every 5 bugs. At each checkpoint:
- Save memory snapshot
- Run the REMAINING bugs with frozen memory (no further updates)
- Measure solve rate on the remaining set

```
Checkpoint at bug 5:  freeze memory → solve rate on bugs 6-30
Checkpoint at bug 10: freeze memory → solve rate on bugs 11-30
Checkpoint at bug 15: freeze memory → solve rate on bugs 16-30
Checkpoint at bug 20: freeze memory → solve rate on bugs 21-30
Checkpoint at bug 25: freeze memory → solve rate on bugs 26-30
```

**Metrics**: Solve rate vs bugs processed (should be monotonically increasing if self-evolution works).

**Expected**: Upward trend. Early checkpoints (5 bugs) solve fewer remaining bugs than late checkpoints (25 bugs).

**Implementation**: Add `--freeze-memory-after N` flag to the runner. At bug N, save memory and disable updates for remaining bugs.

---

### B2: Memory Growth Trajectory

**Hypothesis**: Memory growth is non-uniform — it accelerates on novel feature combinations and plateaus on familiar ones.

**Method**: Track per-bug memory delta during the 30-case run:
- New features added per bug
- New reflections per bug
- New strategies per bug
- Score changes (total absolute delta) per bug

**Metrics**: Memory growth rate (new entries per bug) over the run timeline.

**Expected**: Growth is front-loaded (bugs 1-10 add many features), then plateaus as later bugs share features with earlier ones. This proves the feature-keyed transfer is working — later bugs "recognize" earlier experience.

**Implementation**: Instrument `BenchmarkMemory.update()` to log a snapshot of table sizes before/after each call.

---

### B3: Transfer Benefit Quantification

**Hypothesis**: Bugs processed later in the run benefit from earlier bugs' memory — the marginal solve probability increases with feature overlap to prior bugs.

**Method**: For each bug N in the run, compute:
- **Feature overlap score**: How many of bug N's features were seen in bugs 1..N-1
- **Memory retrieval richness**: How many non-empty guidance items (preferences, skills, warnings, strategies, reflections) were returned for bug N
- **Solve outcome**: Did bug N get solved?

**Metrics**: Correlation between (feature overlap, retrieval richness) and solve probability.

**Expected**: Positive correlation — bugs with high feature overlap to prior bugs solve more often.

---

## Group C: Transfer Experiments (Does Cross-Task Knowledge Actually Transfer?)

### C1: Cross-Project Transfer

**Hypothesis**: Memory learned on one project (Chart) transfers to a different project (Lang or Math) when they share class/exception/trigger features.

**Method**:
1. Run on Chart 1-10 only → save memory
2. Run on Lang 1,3-11 with Chart-only memory (frozen) → measure solve rate
3. Run on Math 1-10 with Chart-only memory (frozen) → measure solve rate
4. Compare against Lang/Math with empty memory (baseline)

**Metrics**: Δ solve rate with vs without Chart memory.

**Expected**: Small but positive transfer if shared features (e.g., `exception:assertionerror` appears across projects). Near-zero transfer for project-specific features.

---

### C2: Leave-One-Out Transfer

**Hypothesis**: Removing one bug from training and testing on it specifically measures transfer quality.

**Method**: For each of the 30 bugs:
1. Run on the OTHER 29 bugs → save memory
2. Run on the held-out bug with that memory (frozen)
3. Compare against running the held-out bug with empty memory

**Metrics**: Δ solve rate (with 29-bug memory vs empty memory) per bug.

**Expected**: Most bugs benefit from memory. Bugs with unique features (no overlap with other 29) show no benefit. Bugs with common features show significant benefit.

**Note**: This requires 30 × 2 = 60 D4J runs (expensive but definitive).

---

### C3: Negative Transfer Detection

**Hypothesis**: In some cases, learning Bug A HURTS Bug B (negative transfer) — the system should be robust to this.

**Method**: Analyze the leave-one-out data from C2. Identify bugs where solve rate DECREASED with memory (negative transfer). For each negative transfer case:
- Examine which memory entries were retrieved
- Check if conflicting scores (positive for one patch style, negative for another on same features) caused confusion

**Metrics**: Count of negative transfer cases. Magnitude of regression.

**Expected**: Rare (< 15% of cases). If common, the feature weighting or dedup mechanism needs adjustment.

---

## Group D: Robustness and Stability

### D1: Bug Ordering Sensitivity

**Hypothesis**: The system is robust to bug ordering — different orderings produce similar solve rates.

**Method**: Run the 30-case benchmark with 5 different orderings:
- `original` (Chart 1-10, Lang 1/3-11, Math 1-10)
- `reverse` (Math 10-1, Lang 11-3/1, Chart 10-1)
- `random_1`, `random_2`, `random_3` (shuffled)

**Metrics**: Mean solve rate ± standard deviation across orderings. Coefficient of variation.

**Expected**: CV < 10% (low sensitivity to ordering). If CV > 20%, ordering effects dominate and the system isn't robust.

**Implementation**: Add `--shuffle-seed N` flag to the runner.

---

### D2: Memory Decay Sensitivity

**Hypothesis**: Moderate decay (0.9-0.95) improves recent-bug focus without losing long-term patterns.

**Method**: Run the 30-case benchmark with `apply_decay()` called between bugs, varying the factor:
- `1.0` (no decay — current default)
- `0.95` (mild decay)
- `0.90` (moderate decay)
- `0.80` (aggressive decay)

**Metrics**: Solve rate, memory size over time, feature retention rate.

**Expected**: 0.90-0.95 may slightly improve or maintain solve rate while keeping memory compact. 0.80 likely hurts because useful long-term patterns fade too fast.

---

### D3: Memory Contamination Recovery

**Hypothesis**: The system can recover from incorrect memory entries (e.g., a false success strategy).

**Method**:
1. Run on 15 bugs normally → save memory
2. Inject 10 adversarial entries: high positive scores for WRONG patch styles on common features
3. Run on remaining 15 bugs with contaminated memory
4. Measure: does the system self-correct? (Do the wrong scores get overridden by real outcomes?)

**Metrics**: Solve rate on bugs 16-30. Tracking of injected scores over time (do they converge to correct values?).

**Expected**: Solve rate drops initially but recovers as real outcomes override false entries within 3-5 bugs. Proves the system is self-correcting.

---

## Group E: Adversarial Dynamic Analysis (Examiner vs Solver)

### E1: Regression Catch Rate

**Hypothesis**: The Examiner dimension catches a significant fraction of Solver overfits that would otherwise pass as "solved."

**Method**: During the 30-case run, log every instance where:
- Visible tests pass BUT regression tests fail
- The Examiner's scope choice (trigger/relevant/all) made the difference (i.e., trigger-only would have missed it)

**Metrics**:
- **Regression catch count**: How many overfits were caught by broader scope
- **Examiner scope accuracy**: Did the learned scope (from `preferred_test_scope`) catch more overfits than default `trigger`?

**Expected**: The Examiner catches 15-30% of would-be "false solves." Learned scope catches more than default.

---

### E2: Post-Regression Recovery Rate

**Hypothesis**: After the Examiner catches an overfit, the Solver's next attempt is more likely to succeed (it learned from the reflection).

**Method**: For every regression-caught attempt, track the next attempt:
- Did the Solver change patch style?
- Did the next attempt pass regression?
- How many attempts until final solve?

**Metrics**:
- **Post-regression solve rate**: % of cases where the bug was eventually solved after a regression catch
- **Attempts to recover**: Average additional attempts needed after regression catch

**Expected**: Post-regression solve rate > 60%. Average recovery in 1-2 additional attempts.

---

### E3: Examiner Scope Escalation Pattern

**Hypothesis**: The Examiner learns to use broader test scopes (relevant/all) for bug classes where trigger-only testing previously missed overfits.

**Method**: Track `preferred_test_scope()` evolution over the 30-bug run:
- Bug 1-10: What scope was preferred?
- Bug 11-20: Did preferences shift?
- Bug 21-30: Are broader scopes preferred for classes with known overfit history?

**Metrics**: Scope distribution over time. Correlation between regression outcomes and subsequent scope preference changes.

**Expected**: Scope preferences shift from trigger → relevant for classes where overfitting was caught.

---

### E4: Solver Strategy Diversity

**Hypothesis**: The Solver explores diverse strategies early (high entropy) and converges to proven patterns later (low entropy) — a healthy explore/exploit balance.

**Method**: Track the distribution of `patch_style` values used across the run:
- Calculate Shannon entropy of patch styles per 5-bug window
- Track unique patch styles used per window

**Metrics**: Strategy entropy over time. Unique strategies per window.

**Expected**: Entropy decreases over time as the Solver converges to proven strategies. But it should NOT reach zero (complete convergence = no exploration = stuck).

---

## Group F: Efficiency and Cost

### F1: LLM Call Efficiency Over Time

**Hypothesis**: Memory reduces LLM calls per bug as the run progresses — later bugs need fewer attempts.

**Method**: Track DeepSeek calls per bug during the 30-case run.

**Metrics**: Average calls per bug for bugs 1-10 vs 11-20 vs 21-30.

**Expected**: Calls decrease over time (e.g., 5.0 → 3.5 → 3.0 avg calls per bug).

---

### F2: Token Efficiency

**Hypothesis**: Memory guidance makes prompts more focused, reducing total token consumption per solve.

**Method**: Track prompt_tokens and completion_tokens per bug. Compare bugs with rich memory guidance vs bugs with empty memory.

**Metrics**: Tokens per solve (successful bugs only). Correlation between retrieval richness and token count.

**Expected**: Rich memory → fewer tokens per solve (more targeted prompts = less trial-and-error).

---

### F3: Candidate Dedup Effectiveness

**Hypothesis**: The candidate dedup mechanism prevents significant wasted attempts.

**Method**: Track:
- How many patch candidates were rejected as duplicates
- What would have happened without dedup (estimated wasted attempts)

**Metrics**: Dedup rejection count per bug. Estimated attempt savings.

**Expected**: 10-30% of candidates rejected as duplicates, saving 0.5-1.5 attempts per bug on average.

---

## Group G: Baseline Comparisons

### G1: vs Random Strategy Selection

**Hypothesis**: Memory-guided strategy selection outperforms random selection.

**Method**: Run 30-case benchmark with `memory_mode=none` but randomize the patch style guidance in the prompt (instead of memory-ranked).

**Metrics**: Solve rate, attempts, overfit rate.

---

### G2: vs Fixed Strategy Ordering

**Hypothesis**: Adaptive memory-ranked ordering outperforms a fixed ordering.

**Method**: Run with a fixed patch style ordering (alphabetical or reverse-alphabetical) in the prompt, ignoring memory preferences.

**Metrics**: Solve rate, first-attempt success rate.

---

### G3: vs Prompt-Only Reflection (No Persistent Memory)

**Hypothesis**: Persistent cross-task memory outperforms single-turn reflection (only using the current bug's failure feedback, no cross-task transfer).

**Method**: Run with `feedback` system (multi-attempt, visible test feedback, rollback) but no persistent memory updates. This is the existing `feedback` baseline.

**Metrics**: Solve rate, duplicate attempt rate (should be higher without dedup memory).

---

## Implementation Priority

| Priority | Experiment | Why | Cost |
|---|---|---|---|
| **P0** | A1 (real D4J ablation) | Core claim — must prove on real bugs | 4 × 30-case runs |
| **P0** | B1 (learning curve) | Proves self-evolution | 5 × partial runs |
| **P0** | E1+E2 (regression catch + recovery) | Proves adversarial dynamic | Analysis of existing v16 traces |
| **P1** | C2 (leave-one-out transfer) | Definitive transfer proof | 60 runs (expensive) |
| **P1** | D1 (ordering robustness) | Statistical credibility | 5 × 30-case runs |
| **P1** | F1 (call efficiency) | Cost argument | Analysis of existing v16 traces |
| **P2** | A2 (per-store ablation) | Decomposition insight | 7 × 30-case runs |
| **P2** | B3 (transfer benefit correlation) | Transfer quantification | Analysis of existing v16 traces |
| **P2** | E3+E4 (examiner/solver dynamics) | Adversarial depth | Analysis of existing v16 traces |
| **P3** | A3, A4, D2, D3, C1, C3, F2, F3, G1-G3 | Refinement and robustness | Various |

---

## Metrics Summary

### Primary (reported in all experiments)
- **Solve rate** (regression pass on first successful attempt or within budget)
- **Pass@1** (solved on first patch attempt)
- **Pass@3** (solved within 3 attempts)

### Secondary
- **Regression pass rate** (of all visible-pass cases, how many also pass regression)
- **Duplicate rejection rate** (fraction of candidates rejected as duplicates)
- **Avg LLM calls per bug**
- **Avg attempts per bug**
- **Unsafe edit rate** (should always be 0)

### Analytical (for deep-dive experiments)
- **Feature overlap score** (B3)
- **Memory retrieval richness** (B3)
- **Regression catch count** (E1)
- **Post-regression solve rate** (E2)
- **Strategy entropy** (E4)
- **Token efficiency** (F2)

---

## Reproducibility Requirements

Each experiment must produce:
1. `summary.json` — aggregate metrics
2. `metrics.csv` — per-case metrics
3. `memory_before.json` / `memory_after.json` — memory snapshots
4. `traces/*.json` — per-case attempt traces
5. `analysis.md` — automated analysis with the specific metrics for that experiment group

All experiments use the same Dockerized Defects4J environment, same DeepSeek model (`deepseek-v4-pro`), same 30-case config (unless otherwise specified).
