# Artifact Index

This directory stores experiment outputs. Large work directories from Defects4J checkouts may be omitted from a submission bundle, but result summaries, metrics, traces, memory snapshots, and patches are retained for audit.

## Best Complete Run

```text
artifacts/runs/d4j30-self-evolved-real-v16
```

Important files:

| File | Meaning |
| --- | --- |
| `summary.json` | Aggregate metrics for the run |
| `metrics.csv` | Per-case metrics |
| `failure_analysis.md` | Failure modes and unresolved cases |
| `memory_before.json` | Persistent memory before the run |
| `memory_after.json` | Persistent memory after the run |
| `memory_snapshots/*.json` | Memory after each task |
| `traces/*.json` | Full per-case tool, prompt, patch, test, and memory trace |
| `patches/*.diff` | Generated candidate patches |

Best result:

| Metric | Value |
| --- | ---: |
| Cases | 30 |
| Solved | 22 |
| Pass@1 | 17/30 |
| Pass@3 | 21/30 |
| Visible pass | 23/30 |
| Regression pass | 22/30 |
| Compile success | 30/30 |
| DeepSeek calls | 121 |

Unsolved cases:

```text
Lang-3, Lang-10, Math-1, Math-2, Math-3, Math-4, Math-6, Math-7
```

## Focused Mechanism Runs

The following runs are partial/focused experiments after v16. They are useful for mechanism analysis but are not counted as full benchmark results:

```text
d4j-failed8-real-v17
d4j-failed8-real-v18-compact
d4j-failed8-real-v19-empty-recovery
d4j-failed8-real-v20-compact-recovery
d4j-failed8-real-v21-compact-first
d4j-lang3-real-v22-feedback-guidance
d4j-lang3-real-v23-critical-feedback
d4j-lang3-real-v24-critical-feedback-180s
d4j-lang3-real-v25-dedup
d4j-failed8-real-v26-dedup
d4j-failed8-real-v27-dedup-limit
d4j-failed8-real-v28-early-dedup
d4j-lang10-real-v29-regex-needles
d4j-lang10-real-v30-root-constraints
d4j-lang10-real-v31-nonpatch-dedup
```

Key lessons:

- DeepSeek empty content and transport timeouts must be treated as non-patch failures.
- Candidate dedup reduces repeated failed strategies and wasted test runs.
- Root-failure retention improves grounding after patch apply failures.
- Regression-aware memory is necessary when visible tests pass but relevant tests fail.

## Two-Dimensional Memory Proof

Fast local proof artifacts:

```text
artifacts/proof_experiments/two_dimensional_memory-v4/
  summary.json
  metrics.csv
  analysis.md
  *_memory_after_train.json
```

Summary:

| Variant | Solved | Check Correct | Repair Correct | Duplicate Risk | Regression Overfit Risk |
| --- | ---: | ---: | ---: | ---: | ---: |
| feedback-only | 0/4 | 0.25 | 0.00 | 4 | 3 |
| check-memory-only | 0/4 | 1.00 | 0.00 | 4 | 0 |
| repair-memory-only | 1/4 | 0.25 | 1.00 | 0 | 3 |
| two-dimensional memory | 4/4 | 1.00 | 1.00 | 0 | 0 |

This proof uses the same `BenchmarkMemory` update and retrieval code as the real Defects4J runner. It demonstrates why the system maintains both what-to-check memory and how-to-repair memory.

## Cost Interpretation

The run records DeepSeek calls and token counts. `estimated_cost_usd` is computed from optional environment variables:

```text
DEEPSEEK_INPUT_USD_PER_MTOK
DEEPSEEK_OUTPUT_USD_PER_MTOK
```

If these rates are not set, the artifact cost field is `0.0`; that should be interpreted as "not priced in this run", not as actual free usage.
