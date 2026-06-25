# Self-Improving Code-Repair Agent: An Examiner-Solver Adversarial Self-Evolving System

## Core Architecture

The system is not a monolithic repair bot. It is a **dual-agent adversarial loop** — two complementary intelligences locked in productive tension, each driving the other to improve:

| Role | Dimension | What it controls | What it learns |
|---|---|---|---|
| **The Examiner** | What-to-check | Test selection, regression scope, file inspection strategy, validation depth | Which tests catch overfitting, which scopes reveal hidden failures, when to escalate from trigger → relevant → full regression |
| **The Solver** | How-to-repair | Patch style selection, repair skill choice, failure reflection, success strategy reuse | Which patch patterns succeed for similar bugs, which strategies are dead ends, what worked before that can transfer |

Neither is sufficient alone. The Examiner without the Solver knows *where to look* but not *how to fix*. The Solver without the Examiner knows *how to fix* but overfits to surface signals. **The adversarial tension between them is the engine of self-improvement.**

---

## The Adversarial Dynamic

This is not cooperation — it is **productive conflict**:

```
Solver produces patch → Examiner tests it
  ├─ If visible tests pass BUT regression fails:
  │    Examiner WINS this round → records regression outcome (-1.0)
  │    Solver learns: "this patch style overfits" (patch_ranking drops)
  │    BUT Solver also learns: "retry-after-feedback is a valuable skill" (+0.75 skill delta)
  │    → The asymmetry is intentional: the PATCH was wrong, but the SKILL of retrying is right
  │
  ├─ If visible tests fail:
  │    Examiner catches the error → records visible failure (-0.5)
  │    Solver learns: "this strategy doesn't work for these features"
  │
  └─ If all tests pass (visible + regression):
       Solver WINS → both dimensions record success (+1.5)
       Solver's success strategy is archived for future bugs with similar features
       Examiner's test scope choice is reinforced as effective
```

The key insight: **the Examiner is adversarial by design**. It actively tries to *break* the Solver's patches by running progressively broader test scopes (trigger → relevant → all). When it finds a regression, it doesn't just reject — it records the specific `patch_style + test_scope` combination that failed, creating a **regression warning** that future rounds must avoid.

This is analogous to **red-team/blue-team** security dynamics or **generator/discriminator** in GANs — but instead of optimizing neural weights, the system optimizes an external memory of strategies, skills, and reflections.

---

## Iterative Memory Updates (Both Dimensions)

After every repair attempt, the system updates **seven memory stores** across both dimensions:

### Examiner's Memory (What-to-Check)

| Store | Updated When | Signal |
|---|---|---|
| `test_selection` | Every outcome | Which test scope was informative (+1.5 solved / -1.0 overfit caught) |
| `test_skill_memory` | Every outcome | Which testing skills work (e.g., `regression-relevant-caught-overfit` gets +0.75 when it catches a regression) |
| `regression_outcomes` | Regression checked | Records `style:X\|scope:Y` combos that failed — generates future warnings |

### Solver's Memory (How-to-Repair)

| Store | Updated When | Signal |
|---|---|---|
| `patch_ranking` | Every outcome | Which patch styles succeed for these features (+1.5 / -0.75 compile fail / -0.5 visible fail) |
| `repair_skill_memory` | Every outcome | Which skills transfer (e.g., `retry-after-feedback`, `repair-after-regression`) |
| `failure_reflections` | On failure | Sanitized lessons — case-specific details stripped, transferable patterns retained |
| `success_strategies` | On success | What worked, archived for reuse on similar future bugs |

---

## Skill and Tool Evolution

The system doesn't just update scores — it evolves its **active skill set** and **tool selection**:

### Skill Evolution

The Examiner and Solver each maintain a ranked skill list. Skills are promoted or demoted based on outcome deltas:

- **Examiner skills** like `regression-relevant-caught-overfit`, `trigger-first`, `run-relevant` move up when they catch real problems
- **Solver skills** like `retry-after-feedback`, `repair-after-regression`, `exact-old-text-grounding` move up when they lead to solutions
- Failed skills (e.g., a patch style that repeatedly overfits) sink to the bottom and are effectively abandoned

These skills are **injected into the LLM prompt** — the system literally tells DeepSeek "these skills have worked before, use them" and "these strategies have failed, avoid them."

### Tool Selection Evolution

The Examiner controls **which tools are used and when**:

- **Test scope selection**: `preferred_test_scope()` returns the empirically best scope (trigger/relevant/all) based on past outcomes for similar features
- **Adaptive attempt budget**: `attempt_bonus()` grants extra repair attempts when history shows retries are effective for this class of bugs
- **Candidate dedup**: The system tracks patch strategy signatures and **rejects duplicates** — the Solver is prevented from wasting attempts on strategies already proven ineffective
- **Regression escalation**: When visible tests pass, the Examiner decides whether to run trigger-only, relevant, or full regression — and it learns from past overfitting which scope to choose

---

## Cross-Task Transfer: The Multiplier

Memory is keyed by **features** (`project:Lang`, `class:numberutils`, `exception:assertionerror`, `trigger:testprecision`), not case IDs. This means:

1. Bug A (Lang, `class:numberutils`) is repaired → Solver learns `numeric-conversion` style works, Examiner learns `relevant` scope catches overfit
2. Bug B (also Lang, `class:numberutils`, different bug entirely) arrives → **both dimensions already have relevant experience**
3. The Solver gets `numeric-conversion` as a top-ranked patch style
4. The Examiner gets `relevant` as preferred test scope + a regression warning about past overfit patterns
5. Bug B benefits from Bug A's experience **without any retraining**

Specific features (class, exception, trigger) carry **2× weight** over broad features (project). This ensures transfer is driven by *meaningful similarity*, not project co-occurrence.

During the 30-bug Defects4J run, memory grew from **5 features to 109 features** — each new bug enriched the shared knowledge base for subsequent bugs.

---

## The Proof: Adversarial Necessity

The deterministic ablation experiment proves that **the adversarial structure is necessary, not optional**:

| Configuration | Solved | What Happens |
|---|---|---|
| **No memory** (feedback only) | 0/4 | Solver tries random strategies, Examiner always uses trigger scope → 4 duplicate risks, 3 overfit risks |
| **Examiner only** (check memory) | 0/4 | Examiner correctly selects test scope (100% correct), but Solver has no patch guidance → picks wrong style every time |
| **Solver only** (repair memory) | 1/4 | Solver correctly selects patch style (100% correct), but Examiner defaults to trigger scope → 3 regression overfit risks |
| **Both** (full adversarial) | **4/4** | Examiner prevents overfit (0 risks), Solver prevents duplication (0 risks), **complementary coverage = complete** |

The system only works when both adversarial agents are active and learning. Remove either, and the other's intelligence becomes useless — like a security team with only red teamers or only blue teamers.

---

## What Makes This "Self-Evolving"

| Traditional RL/ML | This System |
|---|---|
| Updates neural network weights | Updates external memory tables |
| Requires gradient computation | Requires only test execution |
| Learns within a single task | Learns **across** tasks via feature transfer |
| Examiner (reward function) is fixed | Examiner **evolves** its testing strategy |
| Solver (policy network) is the only learner | **Both** Examiner and Solver learn simultaneously |
| Forgetting is a problem (catastrophic) | Forgetting is controlled (decay + dedup + caps) |
| Improvement requires retraining | Improvement is **immediate** — next bug benefits from last bug's experience |

The system is self-evolving because **every repair attempt makes both agents smarter for the next one** — without any parameter updates, without any retraining, without any human intervention. The adversarial loop between Examiner and Solver is the engine; the feature-keyed memory is the transmission; and the accumulated reflections, strategies, and regression warnings are the fuel.
