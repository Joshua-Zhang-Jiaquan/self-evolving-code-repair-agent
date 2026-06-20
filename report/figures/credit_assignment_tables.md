# Long-Horizon Credit Assignment Diagnostics

These tables are diagnostic/correlational summaries of logged rewards, not causal proof.

## Action types

| action | steps | mean reward-to-go | leave-one-out Δ | partial progress |
|---|---:|---:|---:|---:|
| search | 136 | 3.894118 | 5.288889 | 0.000000 |
| read_file | 13 | 6.303846 | -0.036111 | 0.000000 |
| inspect_test | 49 | -0.284694 | -0.136111 | 0.000000 |

## Positions

| step index | steps | mean reward-to-go | success rate | partial progress | top action |
|---:|---:|---:|---:|---:|---|
| 0 | 18 | 5.116667 | 0.000000 | 0.000000 | search |
| 1 | 18 | 4.416667 | 0.000000 | 0.000000 | read_file |
| 2 | 18 | 4.466667 | 0.000000 | 0.000000 | search |
| 3 | 18 | 3.975000 | 0.000000 | 0.000000 | search |
| 4 | 18 | 3.483333 | 0.000000 | 0.000000 | search |
| 5 | 18 | 2.991667 | 0.000000 | 0.000000 | search |
| 6 | 15 | 3.000000 | 0.000000 | 0.000000 | search |
| 7 | 15 | 2.500000 | 0.000000 | 0.000000 | search |
| 8 | 15 | 2.000000 | 0.000000 | 0.000000 | search |
| 9 | 15 | 1.500000 | 0.000000 | 0.000000 | search |
| 10 | 15 | 1.000000 | 0.000000 | 0.000000 | search |
| 11 | 15 | 0.500000 | 0.000000 | 0.000000 | search |

## Test/reward components

| component | nonzero count | total weighted value | mean weighted value |
|---|---:|---:|---:|
| pass | 0 | 0.000000 | 0.000000 |
| visible_test_pass | 0 | 0.000000 | 0.000000 |
| visible_test_failure | 0 | 0.000000 | 0.000000 |
| hidden_regression_ready | 0 | 0.000000 | 0.000000 |
| partial_progress | 0 | 0.000000 | 0.000000 |
| test_runs | 0 | 0.000000 | 0.000000 |
| timeout | 0 | 0.000000 | 0.000000 |
