# Self-Evolving Code Repair Agent

面向代码修复 Agent 的自进化后训练项目。仓库实现了一个可运行、可复现、可记录轨迹的代码修复 Agent, 并在 Dockerized Defects4J 环境中使用 DeepSeek 作为补丁生成器进行真实 benchmark 测试。

当前最佳完整 Defects4J-30 结果来自 `artifacts/runs/d4j30-self-evolved-real-v16`:

| Metric | Value |
| --- | ---: |
| Cases | 30 |
| Solved | 22 |
| Pass@1 | 17/30 |
| Pass@3 | 21/30 |
| Visible trigger-test pass | 23/30 |
| Regression pass | 22/30 |
| Compile success | 30/30 |
| Unsafe edit rate | 0.0 |
| DeepSeek calls | 121 |
| Prompt tokens | 1,776,537 |
| Completion tokens | 753,279 |

本仓库不声称已经 30/30 全量通过。v17-v31 的后续实验改进了最新架构中的 DeepSeek 空响应恢复、timeout 非补丁计数、candidate dedup、root failure retention、test-skill memory 和 regression-aware memory, 但尚未完成新的 30-case 全量 run 来超过 v16。

## Design Highlight: Two-Dimensional Self-Improvement Memory

系统的核心设计不是把失败总结简单追加到 prompt, 而是维护两个互补的长期记忆维度:

- `what-to-check memory`: 学习下一轮应该检查什么, 包括读哪些文件和 snippets、关注哪些 assertion/failure signals、运行 trigger/relevant/all 哪类测试、何时提高 regression validation 优先级。
- `how-to-repair memory`: 学习下一轮应该如何修复, 包括 patch style ranking、repair-skill ranking、failure reflection、success strategy、duplicate failed strategy rejection。

这两个维度分别改变 Agent 的信息获取策略和补丁生成策略。报告中的实验设计建议用四组消融证明有效性: feedback-only、check-memory-only、repair-memory-only、two-dimensional memory。若二维记忆在 held-out tasks 上同时降低工具/测试成本、减少重复失败策略、降低 visible-only overfit, 或提升 Pass@1/Pass@3, 则能说明提升来自跨任务记忆迁移, 而不是单次反思 prompt。

Fast local proof experiment:

```bash
bash scripts/run_memory_ablation_proof.sh
```

Current deterministic proof result:

| Variant | Solved | Check Correct | Repair Correct | Duplicate Risk | Regression Overfit Risk |
| --- | ---: | ---: | ---: | ---: | ---: |
| feedback-only | 0/4 | 0.25 | 0.00 | 4 | 3 |
| check-memory-only | 0/4 | 1.00 | 0.00 | 4 | 0 |
| repair-memory-only | 1/4 | 0.25 | 1.00 | 0 | 3 |
| two-dimensional memory | 4/4 | 1.00 | 1.00 | 0 | 0 |

Real D4J proof runner, requiring DeepSeek:

```bash
export DEEPSEEK_API_KEY="<set-in-shell-only>"
export DEEPSEEK_MODEL=deepseek-v4-pro
SKIP_DOCKER_BUILD=1 bash scripts/run_d4j_memory_ablation_experiments.sh
```

## Repository Layout

```text
code_repair_agent/
  agent.py                # toy benchmark agent and policy-memory baseline
  environment.py          # POMDP-style local repair environment
  tasks.py                # toy visible/hidden test tasks
  evolution.py            # trajectory-level self-improvement logic
  evaluate.py             # toy baseline/feedback/evolution evaluation
  defects4j.py            # Defects4J CLI wrapper
  d4j_benchmark.py        # Defects4J case config parsing
  d4j_memory.py           # persistent self-improvement memory
  deepseek_repair.py      # structured JSON repair prompt/parser
  llm.py                  # DeepSeek chat client, env-key only
  real_benchmark.py       # Dockerized Defects4J benchmark loop
  safe_patch.py           # safe source-only patch application
configs/
  defects4j_30.json       # Chart 1-10, Lang 1 and 3-11, Math 1-10
  defects4j_failed8_v16.json
  reward.json
scripts/
  run_defects4j_benchmark.sh
  run_defects4j_test_sweep.sh
  make_submission_bundle.sh
tests/
  test_environment.py
  test_evaluation.py
  test_real_benchmark_components.py
reports/
  final_submission_report.md
  defects4j_self_evolving_agent_report.md
artifacts/
  runs/d4j30-self-evolved-real-v16/
```

## Install and Local Tests

Python unit tests do not require Defects4J or DeepSeek:

```bash
python3 -m pip install -e .
PYTHONPYCACHEPREFIX=/tmp/rl_pycache python3 -m pytest -q
```

Current local result:

```text
94 passed
```

## Dockerized Defects4J Benchmark

The real benchmark uses Defects4J official CLI behavior: `checkout`, `compile`, `test`, `export`, and metadata queried from the Defects4J checkout.

Prerequisites:

- Docker daemon running.
- Network access to build the image and call the model API.
- DeepSeek API key set only in the shell environment.

Do not write secrets into repository files, prompts, traces, or reports.

```bash
export DEEPSEEK_API_KEY="<set-in-shell-only>"
export DEEPSEEK_MODEL=deepseek-v4-pro
RUN_ID=d4j30-self-evolved-new bash scripts/run_defects4j_benchmark.sh
```

If Docker Hub is temporarily unavailable but the image already exists locally, reuse the local image:

```bash
SKIP_DOCKER_BUILD=1 RUN_ID=d4j30-self-evolved-new bash scripts/run_defects4j_benchmark.sh
```

The script builds `Dockerfile.defects4j`, mounts this repository at `/workspace`, and writes outputs to:

```text
artifacts/runs/<run_id>/
  summary.json
  metrics.csv
  memory_before.json
  memory_after.json
  failure_analysis.md
  traces/*.json
  patches/*.diff
```

## Systems Compared

- `baseline`: one-shot patch generation, no persistent memory.
- `feedback`: multi-attempt repair using visible test feedback and rollback, no cross-task memory.
- `self_evolved`: multi-attempt repair with persistent patch ranking memory, test-skill memory, regression-aware memory, failure reflection memory, success strategies, candidate dedup, and adaptive attempt budget.

The strongest complete run currently available is `self_evolved` only on the 30-case benchmark. The report also documents earlier toy `baseline/+feedback/+learning` ablations and focused v17-v31 harness ablations.

## Reports and Submission Files

- [SUBMISSION.md](SUBMISSION.md): grading-oriented checklist and file index.
- [reports/final_submission_report.md](reports/final_submission_report.md): final technical report for submission.
- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md): exact commands and artifact verification.
- [artifacts/README.md](artifacts/README.md): artifact inventory and result interpretation.

## Academic Integrity Notes

- The benchmark patches are generated by the Agent and recorded as traces.
- Tests, build files, secrets, and paths outside checkout are blocked by `SafePatchApplier`.
- Regression tests are used for validation and memory update after an episode, not leaked into the online prompt before candidate selection.
- The submitted result is not manually edited to claim benchmark success.
