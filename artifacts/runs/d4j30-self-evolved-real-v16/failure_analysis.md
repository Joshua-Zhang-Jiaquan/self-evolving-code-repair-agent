# Failure Analysis

- `self_evolved` `Lang-3`: agent failure, status=failed
- `self_evolved` `Lang-10`: agent failure, status=failed
- `self_evolved` `Math-1`: agent failure, status=failed
- `self_evolved` `Math-2`: agent failure, status=failed
- `self_evolved` `Math-3`: agent failure, status=failed
- `self_evolved` `Math-4`: agent failure, status=failed
- `self_evolved` `Math-6`: agent failure, status=failed
- `self_evolved` `Math-7`: agent failure, status=failed

## Failure Mode Counts

- `llm_error`: 7 cases
- `old_text_not_found`: 5 cases
- `parse_error`: 5 cases
- `visible_failed_after_apply`: 4 cases
- `compile_failed_after_apply`: 2 cases
- `no_op_patch`: 1 cases
- `regression_failed_after_visible`: 1 cases

## Per-Case Labels

- `self_evolved` `Lang-3`: old_text_not_found=6, visible_failed_after_apply=2, no_op_patch=1, parse_error=1, llm_error=1
- `self_evolved` `Lang-10`: visible_failed_after_apply=6, llm_error=4, old_text_not_found=3, parse_error=2, compile_failed_after_apply=1
- `self_evolved` `Math-1`: llm_error=10, old_text_not_found=9, parse_error=1
- `self_evolved` `Math-2`: llm_error=12
- `self_evolved` `Math-3`: llm_error=12
- `self_evolved` `Math-4`: llm_error=12
- `self_evolved` `Math-6`: visible_failed_after_apply=6, old_text_not_found=4, llm_error=2, parse_error=1
- `self_evolved` `Math-7`: old_text_not_found=5, parse_error=3, visible_failed_after_apply=2, regression_failed_after_visible=2, compile_failed_after_apply=1
