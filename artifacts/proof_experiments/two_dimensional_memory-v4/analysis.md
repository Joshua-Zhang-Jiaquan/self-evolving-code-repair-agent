# Two-Dimensional Memory Proof Experiment

This deterministic experiment uses the real BenchmarkMemory update/retrieval code.
It isolates four variants: feedback-only, check-memory-only, repair-memory-only, and full two-dimensional memory.

| Variant | Solved | Check Correct | Repair Correct | Duplicate Risk | Regression Overfit Risk | Avg Tool Calls | Avg Test Runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `feedback_only` | 0/4 | 0.25 | 0.0 | 4 | 3 | 4.5 | 1.0 |
| `check_memory_only` | 0/4 | 1.0 | 0.0 | 4 | 0 | 3.0 | 1.75 |
| `repair_memory_only` | 1/4 | 0.25 | 1.0 | 0 | 3 | 4.5 | 1.0 |
| `two_dimensional_memory` | 4/4 | 1.0 | 1.0 | 0 | 0 | 3.0 | 1.75 |

Interpretation: check memory improves what evidence/tests are selected; repair memory improves patch-style selection; the two-dimensional variant combines both benefits.
