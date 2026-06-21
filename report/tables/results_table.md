# Results table

## SWE-bench Lite official harness

The official SWE-bench Lite harness was **never executed** in this workspace (Docker blocked by `unshare: operation not permitted`, `swebench` package unavailable). The table below records only that each planned run was `blocked`, not any resolved rate.

| Run ID | Type | Predictions | Official Harness Status |
|---|---|---:|---|
| ablation_no_feedback_features | feedback | 40 | blocked |
| ablation_no_process_reward | learning | 40 | blocked |
| ablation_reduced_test_budget | learning | 40 | blocked |
| baseline_main | baseline | 40 | blocked |
| feedback_main | feedback | 40 | blocked |
| learning_main | learning | 40 | blocked |
| baseline_smoke | baseline | 1 | blocked |
| feedback_smoke | feedback | 1 | blocked |
| gold_patch_smoke | gold_smoke | 2 | blocked |
| learning_smoke | learning | 1 | blocked |
| strict_bridge_smoke | baseline | 2 | blocked |

No official SWE-bench resolved rate is claimed.

## Defects4J fallback (executed)

| Run ID | Instances | Patch source | Resolved | Status |
|---|---|---:|---:|---|
| d4j_gold_smoke | Lang_1 | Buggy→fixed source diff | 1/1 | completed |
| d4j_baseline_smoke | Lang_1, Math_5 | Rule-based Java agent | 0/2 | completed |
| d4j_empty_langmath_eval | Lang_1,3-6; Math_1-5 | Empty-patch throughput/caching test | 0/10 | fallback |

The 1/1 gold-patch result validates the evaluator. The 0/2 and 0/10 results show the current rule-based agent does not yet produce valid Java patches.
