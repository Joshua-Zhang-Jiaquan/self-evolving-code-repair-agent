# Agent System Overview

## What It Is

A code repair agent that fixes real Java bugs (Defects4J) using DeepSeek — without fine-tuning model weights. It improves through an **Examiner-Solver adversarial loop**: two dimensions of persistent memory that push against each other, each making the other better.

---

## Method: Adversarial Self-Evolution

| Agent | Dimension | Controls | Learns |
|---|---|---|---|
| **Examiner** | What-to-check | Which tests to run, regression scope, validation depth | Which test scopes catch overfitting, when to escalate trigger → relevant → all |
| **Solver** | How-to-repair | Patch style ranking, repair skills, failure reflections | Which patch patterns succeed for similar bugs, which are dead ends |

The Examiner actively tries to **break** the Solver's patches by running broader test scopes. When it catches a regression (visible pass, regression fail), the Solver's patch style gets penalized (−1.0) but its retry skill gets rewarded (+0.75) — the patch was wrong, but the skill of retrying is right.

---

## Structure: 7 Memory Stores

**Examiner (3 stores)**:
- `test_selection` — learns which test scope (trigger/relevant/all) is most informative
- `test_skill_memory` — learns testing skills (e.g., `regression-relevant-caught-overfit`)
- `regression_outcomes` — records which patch_style + test_scope combos cause regression failures

**Solver (4 stores)**:
- `patch_ranking` — learns which patch styles succeed for similar features
- `repair_skill_memory` — learns repair skills (e.g., `retry-after-feedback`)
- `failure_reflections` — sanitized failure lessons for cross-task transfer
- `success_strategies` — proven repair patterns for reuse

Memory is keyed by **features** (`project:Lang`, `class:numberutils`, `exception:assertionerror`), not case IDs. Specific features carry **2× weight**. A new bug reuses experience from all past bugs sharing its features.

---

## Workflow: Per-Bug Repair Loop

```
1. Checkout bug → compile → run trigger tests → capture failure output
2. Extract features from project / test output / metadata
3. Query memory:
   • Solver → patch preferences, repair skills, success strategies, failure reflections
   • Examiner → preferred test scope, test skills, regression warnings
4. Read focused source snippets (failure needles, trigger methods, assertion context)
5. Build structured JSON repair prompt injecting ALL memory guidance
6. Call DeepSeek → parse repair plan (patch_hunks)
7. Apply patch safely (blocks test edits, path traversal, brace imbalance)
8. Compile → run trigger tests
9. If visible pass → Examiner runs regression tests (scope chosen by memory)
10. Update memory:
    • Success (+1.5) → archive success strategy
    • Overfit caught (−1.0) → record regression warning + failure reflection
    • Compile fail (−0.75) / visible fail (−0.5) → penalize patch style
11. If failed → rollback, re-checkout, retry with enriched guidance + dedup
```

**Robustness**: candidate dedup (rejects identical failed patches), non-patch round budget (read/parse errors don't consume attempts), adaptive attempt bonus (+2 when memory says retries help), prompt budget scaling (shrinks context on repeated failures).

---

## Result

| Metric | Value |
|---|---:|
| Defects4J solved | **22/30** |
| Ablation proof (full vs partial) | **4/4** vs 0/4, 0/4, 1/4 |
| Unit tests | **168 passed** |
| Unsafe edits | **0** |
| Memory features learned (during 30-bug run) | **5 → 109** |
