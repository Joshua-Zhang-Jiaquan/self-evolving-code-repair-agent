# 面向代码修复 Agent 的自进化: 从环境反馈到策略改进

日期: 2026-06-23  
项目目录: `/Users/rccn/Documents/rl`  
数据集: Defects4J  
模型后端: DeepSeek `deepseek-v4-pro`  
最佳完整运行: `artifacts/runs/d4j30-self-evolved-real-v16`

## 摘要

本项目实现了一个面向代码修复任务的 self-improving Agent。系统将代码修复建模为部分可观测多轮决策问题, 在 Dockerized Defects4J 环境中完成 checkout、compile、metadata export、visible trigger tests、candidate patch generation、safe patch application、regression validation 和 trajectory logging。Agent 使用 DeepSeek 生成结构化 JSON 补丁, 通过长期外部记忆在任务和轮次之间迁移经验。

最佳完整 Defects4J-30 运行是 `d4j30-self-evolved-real-v16`: 30 个任务中修复 22 个, Pass@1 为 17/30, Pass@3 为 21/30, visible trigger-test pass 为 23/30, regression pass 为 22/30。该结果不等价于全量通过; 未解任务为 `Lang-3`, `Lang-10`, `Math-1`, `Math-2`, `Math-3`, `Math-4`, `Math-6`, `Math-7`。后续 v17-v31 实验改进了最新架构中的候选去重、DeepSeek 恢复、root failure retention 和 regression-aware memory, 但尚未完成新的 30-case 全量运行来证明超过 v16。

## 1. 问题定义

每个代码修复任务实例包含:

- `issue`: bug 描述、失败现象或需求说明。
- `repository`: buggy checkout。
- `test_command`: visible trigger tests 和 regression tests。
- `optional_hints`: Defects4J metadata、失败测试名、错误栈、历史 memory。

目标是在不修改测试、不硬编码答案、不访问 checkout 外路径的约束下生成补丁, 使 visible tests 和 regression tests 均通过。Agent 必须记录多轮工具调用、prompt、patch、测试结果、memory 更新、token 使用和运行时间。

## 2. POMDP 建模

代码修复可形式化为 POMDP:

- 隐状态 `s_t`: 完整仓库、真实 bug 位置、隐藏/回归测试、当前补丁、外部执行环境和预算。
- 观测 `o_t`: issue、Defects4J metadata、已读取 source snippets、read-only visible test snippets、当前 diff、最近测试输出、历史工具轨迹和 memory。
- 动作 `a_t`: `checkout/export`, `search`, `read_file`, `inspect_test`, `edit_file`, `run_tests`, `rollback`, `final_answer`。
- 转移 `T`: 读文件只改变观测; patch application 改变仓库; test action 产生反馈并消耗成本; rollback/fresh checkout 重置状态。
- 奖励 `R`: 结果奖励、过程奖励、成本惩罚和安全惩罚的组合。
- 终止: visible+regression 通过、达到 patch attempt 上限、达到 non-patch round 上限、重复候选早停或 infrastructure failure。

奖励形式:

```text
R = 1.0 * pass_visible_tests
  + 2.0 * pass_regression_tests
  + 0.2 * relevant_file_found
  - 0.05 * num_tool_calls
  - 0.10 * num_test_runs
  - 1.0 * unsafe_edit
  - 1.0 * test_deletion
  - 1.0 * visible_only_overfit
```

`pass_regression_tests` 权重高于 visible tests, 用于抑制只拟合触发测试的补丁。工具和测试成本惩罚限制无效搜索。`unsafe_edit` 和 `test_deletion` 防止删除测试、改 build 文件、越权路径写入等 reward hacking。`visible_only_overfit` 由 regression-aware memory 记录, 用于在后续相似任务中降低风险补丁风格的优先级。

## 3. Agent 系统实现

系统由以下模块组成:

- `code_repair_agent/real_benchmark.py`: Defects4J benchmark loop, 包括 checkout、compile、test、patch attempt、trace logging 和 metrics aggregation。
- `code_repair_agent/defects4j.py`: Defects4J CLI wrapper。
- `code_repair_agent/deepseek_repair.py`: DeepSeek prompt builder 和 JSON response parser。
- `code_repair_agent/llm.py`: DeepSeek Chat Completions client, 默认模型 `deepseek-v4-pro`, 只从环境变量读取 API key。
- `code_repair_agent/safe_patch.py`: source-only safe patch applier。
- `code_repair_agent/d4j_memory.py`: persistent self-improvement memory。
- `configs/defects4j_30.json`: 30-case benchmark 配置。
- `Dockerfile.defects4j`: Java 11 + Defects4J + Python 的可复现运行环境。

工具动作覆盖 assignment 要求中的 7 类:

| Tool | Implementation |
| --- | --- |
| `search` | snippet selection 使用 trigger methods、failure needles、source call needles |
| `read_file` | `_read_snippet_context` 读取 modified classes、requested files、related source 和 read-only tests |
| `inspect_test` | 读取 `failing_tests`, assertion summary, trigger output |
| `edit_file` | SafePatchApplier 应用 JSON patch hunks |
| `run_tests` | Defects4J `test -t`, `test -r`, `test` |
| `rollback` | 每次失败后 fresh checkout, 清理污染 diff |
| `final_answer` | 输出 `summary.json`, `metrics.csv`, `failure_analysis.md`, traces 和 patches |

安全补丁应用约束:

- 禁止修改 test dirs。
- 禁止修改 build files、generated files、secrets 和 checkout 外路径。
- `old` 文本必须能在当前源码中唯一定位。
- ambiguous replacement、no-op patch 和 duplicate replacement 会被拒绝。

## 4. 自进化方法: 二维记忆设计

本项目采用外部策略/记忆层, 不更新 LLM 参数。Self-evolving loop 是:

```text
attempt -> observe feedback -> reflect -> update memory -> retrieve memory for later tasks
```

本系统的特殊设计是把经验记忆拆成两个正交维度:

1. `what-to-check memory`: 学习下一轮应该检查什么, 包括读哪些文件、关注哪些失败信号、运行 trigger/relevant/all 中哪类测试、是否需要优先验证回归风险。
2. `how-to-repair memory`: 学习下一轮应该怎样修, 包括 patch style、repair skill、失败策略禁用、成功策略复用和补丁候选排序。

这种二维拆分避免了把所有经验混成一句 reflection。代码修复的失败常常不是单纯“不会写补丁”, 而是检查维度和修复维度交叉出错: 有时模型没有看对文件或测试, 有时看对了证据但采用了错误补丁风格。二维记忆让 controller 可以分别改进“信息获取策略”和“补丁生成策略”, 再在 prompt 中组合使用。

### 4.1 What-to-check memory

检查维度由 `test_selection`, `test_skill_memory`, `regression_outcomes` 和 source-needle 规则组成。它学习:

- 先运行 trigger tests、relevant tests 还是 all tests。
- 哪些失败类型需要优先读取 read-only visible test。
- 哪些 modified classes、requested files、failure stack frames 和 assertion needles 应进入 source snippets。
- 哪些 patch style 曾经 visible pass 但 regression fail, 因而下一轮必须更早做 regression validation。
- 当 LLM timeout、empty response 或 old text not found 时, 是否应缩小 prompt、保留 root failure、重新读取源码片段, 而不是消耗 patch attempt。

这个维度回答的是: “下一步应该看什么证据, 用什么 verifier 检查补丁是否可靠?”

### 4.2 How-to-repair memory

修复维度由 `patch_ranking`, `repair_skill_memory`, `failure_reflections`, `success_strategies` 和 candidate dedup 组成。它学习:

- 相似 feature 下哪些 patch style 成功过, 例如 guard、boundary、numeric conversion、API contract、regex parsing。
- 哪些 repair skill 在 compile failure、visible failure、regression failure 后更有效。
- 哪些补丁策略已经失败, 后续候选必须改变 root-cause hypothesis。
- 哪些成功轨迹可以复用为 compact strategy。
- 哪些 diff signature 或 semantic signature 是重复失败候选, 应该在运行测试前拒绝。

这个维度回答的是: “基于当前证据, 应该怎样生成更可能通过的补丁?”

### 4.3 二维记忆的耦合方式

两个维度不是独立输出最终答案, 而是在每一轮 attempt 中共同影响 prompt 和 controller:

```text
features = project + trigger tokens + exception/assertion + modified classes

what_to_check = retrieve(test_selection, test_skill_memory, regression_outcomes, source_needles)
how_to_repair = retrieve(patch_ranking, repair_skill_memory, failure_reflections, success_strategies)

prompt = issue + root failure + selected snippets + what_to_check warnings + how_to_repair strategies
controller = safe patch + duplicate rejection + chosen test scope + memory update
```

例如 Lang-10 的 regex failure 中, what-to-check memory/logic 保留 root failure 并把 `\s*+`, `Expected FDF failure`, `escapeRegex`, `CopyQuotedStrategy` 放入 snippets; how-to-repair memory/logic 则禁止重复的错误 patch strategy, 要求候选改变修复假设。Lang-3 的 numeric failure 中, candidate dedup 把重复的 `numeric-type:add-float-before-double` 识别为已失败修复方向, 从而避免浪费后续测试预算。

### 4.4 Regression-aware memory

`regression_outcomes` 记录 `patch_style + test_scope` 是否导致 visible-only overfit。相似任务会收到 warning, 避免重复风险策略。这是对“只优化公开测试”的显式约束, 也连接了两个维度: 检查维度提高 relevant/regression test 优先级, 修复维度降低导致回归失败的 patch style。

### 4.5 Failure reflection memory

失败轨迹被压缩为 reflection:

- `old_text_not_found`: 下一轮使用更短 exact old block 和 line anchor。
- `compile_failure`: 下一轮避免语法不平衡和缺失 import。
- `visible_failure`: 下一轮改变 root-cause hypothesis。
- `regression_failure`: 下一轮避免 visible-only patch。
- `llm_error` / timeout / empty content: 作为 non-patch failure 恢复, 不消耗 patch attempt。
- `duplicate_failed_patch_strategy`: 重复候选直接拒绝, 不运行测试。

v17-v31 的架构迭代重点是把失败反馈从单轮 prompt 文本提升为可迁移的二维机制: DeepSeek compact recovery、timeout non-patch counting、candidate dedup、root failure retention、regex/source constraints、test-skill memory 和 regression-aware warnings。

## 5. 实验设置

完整 benchmark 使用 `configs/defects4j_30.json`:

| Project | Bug IDs | Count |
| --- | --- | ---: |
| Chart | 1-10 | 10 |
| Lang | 1, 3-11 | 10 |
| Math | 1-10 | 10 |

Visible feedback 使用 trigger tests。候选补丁通过 visible tests 后, 再运行 relevant/regression tests。Regression results 不在候选选择前泄漏给 Agent; 它们只用于候选验证和 episode 后 memory 更新。

运行环境:

- Docker image: `Dockerfile.defects4j`
- Java: OpenJDK 11
- Defects4J: official GitHub repository initialized in image
- Model: DeepSeek `deepseek-v4-pro`
- API key: shell environment only

## 6. 主结果

最佳完整运行 `d4j30-self-evolved-real-v16`:

| Metric | Value |
| --- | ---: |
| Cases | 30 |
| Solved | 22 |
| Pass@1 | 0.5667 = 17/30 |
| Pass@3 | 0.7000 = 21/30 |
| Visible pass rate | 0.7667 = 23/30 |
| Regression pass rate | 0.7333 = 22/30 |
| Compile success rate | 1.0000 = 30/30 |
| Avg tool calls | 6.1667 |
| Avg test runs | 11.5667 |
| Avg patch size | 11.8667 |
| Unsafe edit rate | 0.0000 |
| Infrastructure failures | 0 |
| Agent failures | 8 |
| DeepSeek calls | 121 |
| Prompt tokens | 1,776,537 |
| Completion tokens | 753,279 |
| Wall time | 9,271.7438 s |

Solved cases:

```text
Chart-1, Chart-2, Chart-3, Chart-4, Chart-5,
Chart-6, Chart-7, Chart-8, Chart-9, Chart-10,
Lang-1, Lang-4, Lang-5, Lang-6, Lang-7,
Lang-8, Lang-9, Lang-11,
Math-5, Math-8, Math-9, Math-10
```

Failed cases:

```text
Lang-3, Lang-10, Math-1, Math-2, Math-3, Math-4, Math-6, Math-7
```

Compared with an earlier complete run:

| Run | Solved | Pass@1 | Pass@3 | Regression | Compile |
| --- | ---: | ---: | ---: | ---: | ---: |
| `d4j30-self-evolved-real-v1` | 12/30 | 11/30 | 12/30 | 12/30 | 29/30 |
| `d4j30-self-evolved-real-v16` | 22/30 | 17/30 | 21/30 | 22/30 | 30/30 |

This improvement shows that harness-level repair, grounding, feedback, memory, and safety mechanisms materially changed benchmark performance.

## 7. Ablations and Focused Mechanism Evidence

The following focused experiments are not full benchmark results; they isolate mechanisms after v16.

| Experiment | Evidence |
| --- | --- |
| Remove compact recovery | v18 showed repeated empty DeepSeek responses on Lang-3 |
| Add empty/timeout recovery | v19-v20 treated retryable LLM failures as non-patch failures |
| Add critical feedback only | v22-v24 still repeated Lang-3 wrong numeric strategy |
| Add candidate dedup | v25 rejected repeated `numeric-type:add-float-before-double` without wasting tests |
| Early dedup on failed8 | v28 reduced Lang-3 DeepSeek calls from v16's 14 to 3 in focused failure exit |
| Add regex source constraints | v29-v30 injected Lang-10 root failure constraints and exact source needles |
| Non-patch dedup | v31 partial trace recorded two applied Lang-10 candidates, both visible failed |

Interpretation: feedback alone is not enough when the model repeats the same semantic edit. A self-improving Agent needs memory and controller-level mechanisms that reject repeated failed strategies, preserve root failures, and decide whether an event should consume patch budget.

### 7.1 Proposed experiment: proving two-dimensional memory usefulness

To prove that self-improvement is caused by persistent two-dimensional memory rather than lucky retries or prompt wording, run the same train/eval protocol with four variants. The training split updates memory; the evaluation split is run afterward with the same model, bug list, attempt cap, timeout, and random/order settings.

| Variant | What-to-check memory | How-to-repair memory | Expected Observation |
| --- | --- | --- | --- |
| A. Feedback only | Disabled | Disabled | Uses visible feedback but repeats more failed checks and patch styles |
| B. Check-memory only | Enabled | Disabled | Runs better tests and reads better snippets, but may still repeat bad patch strategies |
| C. Repair-memory only | Disabled | Enabled | Ranks better patch styles, but may miss regression risks or wrong files |
| D. Two-dimensional memory | Enabled | Enabled | Best tradeoff: higher Pass@1/Pass@3 or lower tool/test cost at equal success |

Concrete protocol:

1. Use the same Defects4J case order and fixed attempt caps, for example `configs/defects4j_30.json` or the failed8 split from v16.
2. Round 0 starts with empty memory and records `memory_before.json`.
3. Round 1 runs A/B/C/D on the training split and writes separate memory files:
   - `memory_feedback_only.json`
   - `memory_check_only.json`
   - `memory_repair_only.json`
   - `memory_2d.json`
4. Round 2 evaluates each memory on held-out cases without updating memory online, so the comparison measures transfer rather than within-task reflection.
5. Report Pass@1, Pass@3, visible pass, regression pass, avg tool calls, avg test runs, duplicate rejection count, old-text-not-found count, regression-failure-after-visible count, DeepSeek calls, and tokens.

The key proof signals are not only more solved cases. A useful self-improving system can also show:

- lower average test runs because what-to-check memory chooses a better validation scope;
- fewer old-text-not-found errors because snippet/file selection improves;
- fewer duplicate failed candidates because how-to-repair memory rejects repeated patch styles;
- fewer visible-only overfits because regression-aware memory changes both test choice and patch ranking;
- improved Pass@1/Pass@3 on held-out cases compared with feedback-only.

This experiment directly tests the two dimensions. If B improves tool/test efficiency but not patch success, C improves candidate quality but misses regression risks, and D combines both benefits, then the result supports the claim that self-improvement comes from structured memory transfer rather than one-off prompt reflection.

### 7.2 Executed local proof experiment

In addition to the proposed D4J protocol, the repository includes a fast deterministic proof experiment that uses the same `BenchmarkMemory` update and retrieval code:

```bash
bash scripts/run_memory_ablation_proof.sh
```

Latest output:

```text
artifacts/proof_experiments/two_dimensional_memory-v4/
```

| Variant | Solved | Check Correct | Repair Correct | Duplicate Risk | Regression Overfit Risk | Avg Tool Calls | Avg Test Runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `feedback_only` | 0/4 | 0.25 | 0.00 | 4 | 3 | 4.5 | 1.0 |
| `check_memory_only` | 0/4 | 1.00 | 0.00 | 4 | 0 | 3.0 | 1.75 |
| `repair_memory_only` | 1/4 | 0.25 | 1.00 | 0 | 3 | 4.5 | 1.0 |
| `two_dimensional_memory` | 4/4 | 1.00 | 1.00 | 0 | 0 | 3.0 | 1.75 |

The result shows the intended separation: check memory alone learns the right validation/evidence strategy but cannot choose the right patch style; repair memory alone chooses the right patch style but misses regression-sensitive checks; only two-dimensional memory combines both and solves all held-out synthetic transfer cases.

The proof also exposed one useful design issue: broad `project:*` memory can overpower more specific `class:*` memory. The current implementation therefore uses specificity-weighted memory aggregation, giving class/exception/trigger features higher weight than project-level features. This is a general self-improving design improvement rather than a benchmark-specific patch.

## 8. Failure Analysis

v16 failure modes:

| Failure Mode | Count |
| --- | ---: |
| `llm_error` | 7 |
| `old_text_not_found` | 5 |
| `parse_error` | 5 |
| `visible_failed_after_apply` | 4 |
| `compile_failed_after_apply` | 2 |
| `no_op_patch` | 1 |
| `regression_failed_after_visible` | 1 |

Lang-3: the model repeatedly produced a wrong numeric type ordering strategy. Candidate dedup reduced waste but did not synthesize the correct Java numeric conversion fix.

Lang-10: root failure involved regex whitespace parsing. Source grounding improved after regex constraints, but generated patches still failed visible tests or had patch grounding problems.

Math-1 to Math-4: many failures were transport/LLM errors in v16; later retry mechanisms address this class but need a new complete run for confirmation.

Math-6 and Math-7: visible/regression mismatch shows why regression-aware memory is necessary. Trigger tests alone are not a reliable success signal.

## 9. Training Paradigm Boundaries

SFT is useful for learning high-quality repair trajectories and formatting, but it cannot directly optimize test pass rates.

RLVR is useful when tests serve as verifier, but it can induce reward hacking unless unsafe edits and hidden/regression tests are enforced.

Agentic RL matches multi-step tool use and long-horizon credit assignment, but it is expensive and hard to stabilize.

OPD can distill high-cost successful trajectories into a cheaper policy, but teacher quality determines whether it preserves or amplifies bias.

Self-distillation can reuse the Agent's own successful traces, but needs held-out tasks and failure analysis to avoid feedback loops.

This project implements a lightweight external-memory approximation of Agentic RL/OPD: it converts environment feedback into patch ranking, test selection, skill memory, regression warnings, and failure reflections without changing model parameters.

## 10. Safety, Cost, and Limitations

Safety:

- API keys are environment-only.
- SafePatch blocks test deletion, build-file edits, secret paths, and checkout-outside edits.
- Regression tests are not leaked into online prompt before candidate choice.
- Trace redaction removes the environment API key if present.

Cost:

- v16 recorded 121 DeepSeek calls.
- Token usage was 1,776,537 prompt tokens and 753,279 completion tokens.
- `estimated_cost_usd` is 0.0 because pricing rates were not configured in the artifact; it should be interpreted as not priced, not as actually free.

Limitations:

- Current best verified full result is 22/30, not 30/30.
- v17-v31 focused runs are partial mechanism evidence.
- Memory is external JSON, not model parameter learning.
- Repeated interaction with the same 30 tasks can overfit benchmark patterns; cross-project validation is needed.

## 11. Reproducibility

Unit tests:

```bash
python3 -m pip install -e .
PYTHONPYCACHEPREFIX=/tmp/rl_pycache python3 -m pytest -q
```

Full benchmark:

```bash
export DEEPSEEK_API_KEY="<set-in-shell-only>"
export DEEPSEEK_MODEL=deepseek-v4-pro
RUN_ID=d4j30-self-evolved-repro bash scripts/run_defects4j_benchmark.sh
```

Verify best submitted run:

```bash
python3 - <<'PY'
import csv, json, pathlib
run = pathlib.Path("artifacts/runs/d4j30-self-evolved-real-v16")
summary = json.loads((run / "summary.json").read_text())
systems = (summary.get("summary") or summary)["systems"]
print(json.dumps(systems["self_evolved"], indent=2))
rows = list(csv.DictReader((run / "metrics.csv").open()))
print("cases", len(rows))
print("failed", [row["case_id"] for row in rows if row["status"] != "solved"])
PY
```

## 12. Conclusion

The project delivers a runnable self-evolving code repair Agent with real Defects4J integration, structured DeepSeek patch generation, safe patch application, persistent memory, full trajectory logging, and reproducible reports. The best complete benchmark currently solves 22/30 Defects4J tasks. The most important finding is that self-improvement should not be treated as prompt reflection alone: useful improvement came from controller and memory mechanisms that change future search behavior, test selection, patch ranking, and failure avoidance across tasks.
