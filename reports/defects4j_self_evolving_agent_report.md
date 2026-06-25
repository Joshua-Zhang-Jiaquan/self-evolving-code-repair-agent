# 面向代码修复 Agent 的自进化: Defects4J 真实基准技术报告

生成日期: 2026-06-23  
代码目录: `/Users/rccn/Documents/rl`  
基准: Defects4J, 30 个 active bugs  
模型: DeepSeek, 默认模型名 `deepseek-v4-pro`  
核心结论: 当前最好完整 benchmark 版本是 `d4j30-self-evolved-real-v16`, 在 30 个任务中修复 22 个。v17-v31 之后的改动改进了框架稳定性、候选去重和失败反馈，但尚未完成新的 30-task 全量运行来证明超过 v16。

## 1. 摘要

本项目实现了一个 Docker 化 Defects4J 代码修复 Agent。Agent 接收 bug 实例, 多轮读取环境、生成补丁、应用补丁、运行触发测试和回归测试, 并将轨迹、补丁、测试结果、DeepSeek 调用和 memory 更新写入 `artifacts/runs/<run_id>/`。系统包含三个版本: `baseline`, `feedback`, `self_evolved`。本报告重点分析当前已验证的最佳版本与后续 v31 前后的架构。

截至当前 artifact, 可比的完整 30-task run 中最佳结果为:

| Run ID | 系统 | Cases | Solved | Pass@1 | Pass@3 | Visible Pass | Regression Pass | Compile Success | DeepSeek Calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `d4j30-self-evolved-real-v16` | `self_evolved` | 30 | 22 | 17/30 | 21/30 | 23/30 | 22/30 | 30/30 | 121 |
| `d4j30-self-evolved-real-v1` | `self_evolved` | 30 | 12 | 11/30 | 12/30 | 13/30 | 12/30 | 29/30 | 67 |

v16 相比 v1 的提升主要来自更强的安全补丁应用、失败反馈、snippet 定位、可持久化 memory 和多轮尝试机制。v17-v31 主要围绕 v16 的 8 个未解任务继续迭代, 加入了 DeepSeek 空响应恢复、timeout 非补丁尝试处理、root failure 保留、candidate dedup、regex failure constraints 和 test-skill/regression-aware memory。由于这些 focused runs 没有完成新的 30-case 全量 benchmark, 它们不能作为新的总成绩, 但可以作为架构改进和消融证据。

本报告不声称已经全量通过。当前已验证最好结果是 22/30。要证明 30/30, 必须在同一 bug list、同一 attempt budget、无人工修复和无 benchmark 答案硬编码的条件下重新跑完整 30-task benchmark, 并产出新的 `summary.json`, `metrics.csv`, traces 和 patches。

## 2. 问题定义与 POMDP 建模

代码修复任务可以建模为一个部分可观测多轮决策问题。

状态 `s_t` 包含:

- bug 元信息: project, bug id, Defects4J metadata, trigger tests, relevant tests, modified classes。
- 当前 checkout 快照: 源码、测试文件、当前 git diff。
- 已观察上下文: source snippets, read-only visible test snippets, line-numbered snippets。
- 工具轨迹: checkout, compile, test, patch apply, rollback/fresh checkout, LLM call。
- 最近测试反馈: trigger failures, compile failures, regression failures, `failing_tests` 内容。
- 自进化 memory: patch ranking, test selection, repair skill, test skill, regression outcome, failure reflection, success strategy。
- 预算: patch attempt 上限、non-patch round 上限、snippet budget、DeepSeek timeout 和 max tokens。

动作 `a_t` 包含:

- `checkout/export`: 从 Defects4J 初始化 buggy checkout, 导出 `dir.src.classes`, `dir.src.tests`, `classes.modified`, `tests.trigger`, `tests.relevant`。
- `search/read_file`: 通过 modified classes、trigger tests、failure needles、requested files 读取源码和测试片段。
- `inspect_test`: 读取 `failing_tests`、断言、错误栈和 visible test 上下文。
- `edit_file`: 接收 DeepSeek JSON patch hunks, 通过 `SafePatchApplier` 只修改源码目录。
- `run_tests`: 运行 compile、trigger tests、relevant 或 all tests。
- `rollback`: 每次失败后 fresh checkout, 避免失败补丁污染下一轮。
- `final_answer`: 写出 patch、trace、metrics、memory snapshots 和 failure report。

观测 `o_t` 是部分可观测的。Agent 能看到 visible trigger tests 和 selected source snippets, 但不能把 hidden/regression 结果泄漏进最终补丁选择前的 prompt。回归测试只用于候选补丁验证和 episode 后 memory 更新。

转移由工具和代码库状态决定。补丁应用成功会改变源码和 diff, 测试动作会产生新的输出, rollback/fresh checkout 会重置工作区。LLM 输出是不确定的, 可能产生 parse error、empty content、old text not found 或编译失败。

终止条件:

- `solved`: visible trigger tests 通过且 regression scope 通过。
- `failed`: 达到 patch attempt 上限。
- `early_stop`: 达到 duplicate strategy rejection 上限或 non-patch round 上限。
- `infrastructure_error`: checkout、compile、Defects4J CLI 或 Docker 环境失败。

奖励设计:

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

动机:

- visible tests 给出在线反馈, 但权重低于 regression, 防止只过触发测试。
- relevant file/process reward 鼓励定位到正确文件, 缓解长程信用分配。
- tool/test 成本惩罚约束无限搜索和重复测试。
- unsafe/test deletion penalty 防止奖励黑客。
- visible-only overfit penalty 由 regression-aware memory 记录, 防止后续相似任务重复同类补丁。

## 3. Agent 架构

### 3.1 Dockerized Defects4J harness

Runner 使用 Defects4J 官方 CLI 行为作为基准来源, 包括:

- `defects4j checkout`
- `defects4j compile`
- `defects4j export`
- `defects4j test -t <trigger>`
- `defects4j test -r` 或 `defects4j test`

配置文件 `configs/defects4j_30.json` 定义 30 个任务:

- Chart 1-10
- Lang 1, 3-11, 排除 deprecated Lang 2
- Math 1-10

每个 run 输出:

- `artifacts/runs/<run_id>/summary.json`
- `artifacts/runs/<run_id>/metrics.csv`
- `artifacts/runs/<run_id>/failure_analysis.md`
- `artifacts/runs/<run_id>/memory_before.json`
- `artifacts/runs/<run_id>/memory_after.json`
- `artifacts/runs/<run_id>/traces/*.json`
- `artifacts/runs/<run_id>/patches/*.diff`

### 3.2 DeepSeek patch generator

DeepSeek 客户端位于 `code_repair_agent/llm.py`。默认模型为 `deepseek-v4-pro`, API key 只从环境变量 `DEEPSEEK_API_KEY` 读取。Trace redaction 会移除环境中的 key, 报告和 artifacts 不应包含明文 secret。

Prompt 构造位于 `code_repair_agent/deepseek_repair.py`, 要求模型返回 JSON:

```json
{
  "diagnosis": "short root-cause hypothesis",
  "files_to_read": ["optional source files"],
  "patch_hunks": [
    {
      "file": "relative/source/File.java",
      "old": "exact source text",
      "new": "replacement source text",
      "line_start": 1,
      "line_end": 1
    }
  ],
  "tests_to_run_next": ["trigger or relevant tests"],
  "confidence": 0.0,
  "final_explanation": "short patch explanation",
  "patch_style": "guard|boundary|api-contract|..."
}
```

安全约束:

- 不允许修改 tests、build files、generated files、secrets 或 checkout 外路径。
- `old` 必须来自当前 source snippets, 不能使用失败补丁残留文本。
- visible tests 标记为 read-only, 只能用于诊断。
- hidden/regression 测试结果不能在在线 prompt 中泄漏。

### 3.3 Safe patch application

`SafePatchApplier` 负责应用补丁。关键约束:

- path 必须在 checkout 内。
- patch 文件必须在 `dir.src.classes` 下。
- test dirs 和危险路径被拒绝。
- 支持 exact replacement 和唯一 whitespace-normalized replacement。
- 拒绝 ambiguous old text。
- 拒绝 no-op patch 和明显 duplicate replacement。

这使得 Agent 即使生成错误补丁, 也不会通过删除测试或改配置来获得虚假奖励。

### 3.4 三个系统版本

`baseline`:

- 单次生成补丁。
- 不使用 cross-task memory。
- 失败后不进行多轮反馈修复。

`feedback`:

- 多轮 attempt。
- 使用 visible test、compile failure、old text not found 等反馈修复。
- 不使用长期 self-evolution memory。

`self_evolved`:

- 多轮 attempt。
- 使用 persistent memory。
- 根据相似 feature 注入 patch style、repair skill、test skill、regression warning、failure reflection 和 success strategy。
- 根据 memory 增加 bounded attempt bonus。
- 记录每一步 memory 更新, 支持跨任务和跨轮次迁移。

## 4. 自进化机制

### 4.1 Patch ranking memory

`BenchmarkMemory.patch_ranking` 使用 feature -> patch_style -> score 的表。Feature 来自:

- `project:<name>`
- trigger test tokens
- exception/assertion 类型
- modified classes

成功补丁增加分数, 编译失败、visible failure、regression failure 降低分数。后续相似任务会把高分 patch style 作为 prompt preference。

### 4.2 Test-skill 与 regression-aware memory

当前 memory 不只记录 test scope, 还区分:

- `test_selection`: trigger/relevant/all 的历史效果。
- `test_skill_memory`: 例如 no-test-run、test-after-compile、repair-after-regression。
- `regression_outcomes`: 记录某类 patch style + test scope 是否曾经 visible pass 但 regression fail。

当相似 feature 命中负向 regression outcome, prompt 会注入 warning:

```text
style:<patch_style>|scope:<scope> has failed regression before; avoid visible-only overfitting
```

这使 memory 不只优化 visible pass, 也对过拟合触发测试的策略进行惩罚。

### 4.3 Failure reflection memory

失败后写入紧凑 reflection, 包括:

- `old_text_not_found`
- `compile_failure`
- `visible_failure`
- `regression_failure`
- `llm_error`
- `parse_error`
- `duplicate_failed_patch_strategy`

后续相似任务只注入相关 reflections, 避免无限累积噪音。v31 前的实现还将 retryable LLM timeout 和 empty response 标记为 non-patch failure, 不浪费 patch attempt budget。

### 4.4 Candidate dedup and root-failure retention

v25-v31 期间加入了两个关键机制:

1. Candidate dedup
   - 为已失败补丁提取 semantic signature。
   - 如果模型再次生成相同策略, 不运行 compile/test。
   - duplicate rejection 作为 non-patch candidate rejection, 不消耗 patch attempt。
   - 对于 `replacement text already exists` 的 apply failure, 也视为重复候选。

2. Root failure retention
   - patch apply failure 或 grounding failure 不再覆盖初始 trigger failure。
   - Prompt 始终保留 root failing output, 同时加入最新反馈。
   - 对 Lang-10 一类 regex whitespace failure, 从 root failure 提取 `\s*+`, `Expected FDF failure`, `escapeRegex`, `CopyQuotedStrategy` 等 source needles 和约束。

这些改动的目标是提高 Pass@1/Pass@3 的有效尝试质量, 而不是靠增加尝试次数 brute force。

## 5. 实验设置

### 5.1 数据集

完整 benchmark 使用 `configs/defects4j_30.json`:

| Project | Bug IDs | Count |
| --- | --- | ---: |
| Chart | 1-10 | 10 |
| Lang | 1, 3-11 | 10 |
| Math | 1-10 | 10 |

v16 后 focused rerun 使用 `configs/defects4j_failed8_v16.json`:

```text
Lang-3, Lang-10, Math-1, Math-2, Math-3, Math-4, Math-6, Math-7
```

focused rerun 只用于诊断和框架改进, 不作为 30-case 总成绩。

### 5.2 评测指标

报告指标:

- Pass@1
- Pass@3
- Visible trigger-test pass rate
- Regression pass rate
- Compile success rate
- Average tool calls
- Average test runs
- Patch size
- Unsafe edit rate
- Wall time
- DeepSeek calls
- Prompt/completion tokens
- Infrastructure failures vs agent failures

### 5.3 复现命令

单元测试:

```bash
PYTHONPYCACHEPREFIX=/tmp/rl_pycache python3 -m pytest -q
```

完整 30-case benchmark 模板:

```bash
export DEEPSEEK_API_KEY=<set-in-shell-only>
export DEEPSEEK_MODEL=deepseek-v4-pro
python3 -m code_repair_agent.real_benchmark \
  --config configs/defects4j_30.json \
  --run-id d4j30-self-evolved-real-<new-id> \
  --systems self_evolved \
  --require-model
```

未解 8-case focused rerun 模板:

```bash
export DEEPSEEK_API_KEY=<set-in-shell-only>
export DEEPSEEK_MODEL=deepseek-v4-pro
python3 -m code_repair_agent.real_benchmark \
  --config configs/defects4j_failed8_v16.json \
  --run-id d4j-failed8-real-<new-id> \
  --systems self_evolved \
  --require-model
```

注意: key 不写入 config、README、trace 或报告。

## 6. 主要结果

### 6.1 最佳完整 benchmark: v16

`artifacts/runs/d4j30-self-evolved-real-v16/summary.json`:

| Metric | Value |
| --- | ---: |
| Cases | 30 |
| Solved | 22 |
| Pass@1 | 0.5667, 即 17/30 |
| Pass@3 | 0.7000, 即 21/30 |
| Visible pass rate | 0.7667, 即 23/30 |
| Regression pass rate | 0.7333, 即 22/30 |
| Compile success rate | 1.0000, 即 30/30 |
| Avg tool calls | 6.1667 |
| Avg test runs | 11.5667 |
| Avg patch size | 11.8667 |
| Unsafe edit rate | 0.0000 |
| DeepSeek calls | 121 |
| Prompt tokens | 1,776,537 |
| Completion tokens | 753,279 |
| Wall time | 9,271.7438 s |

v16 solved cases:

```text
Chart-1, Chart-2, Chart-3, Chart-4, Chart-5,
Chart-6, Chart-7, Chart-8, Chart-9, Chart-10,
Lang-1, Lang-4, Lang-5, Lang-6, Lang-7,
Lang-8, Lang-9, Lang-11,
Math-5, Math-8, Math-9, Math-10
```

v16 failed cases:

```text
Lang-3, Lang-10, Math-1, Math-2, Math-3, Math-4, Math-6, Math-7
```

v16 failure mode counts:

| Failure Mode | Count |
| --- | ---: |
| `llm_error` | 7 |
| `old_text_not_found` | 5 |
| `parse_error` | 5 |
| `visible_failed_after_apply` | 4 |
| `compile_failed_after_apply` | 2 |
| `no_op_patch` | 1 |
| `regression_failed_after_visible` | 1 |

### 6.2 v1 到 v16 的提升

| Run | Solved | Pass@1 | Pass@3 | Visible | Regression | Compile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 12/30 | 11/30 | 12/30 | 13/30 | 12/30 | 29/30 |
| v16 | 22/30 | 17/30 | 21/30 | 23/30 | 22/30 | 30/30 |

主要提升来自:

- 更稳的 Defects4J metadata export 和 trigger/relevant test routing。
- 更强的 source snippet selection, 使用 modified classes、trigger tests、failure stack 和 assertion needles。
- SafePatch grounding, 对 exact old/new、line anchor、ambiguous replacement 做约束。
- 多轮 visible feedback 与 fresh checkout rollback。
- self-evolved memory 对 patch style、test scope、regression warning 的跨任务复用。

### 6.3 v17-v31 focused improvement evidence

v17-v31 没有形成新的完整 30-case summary, 因此不能替代 v16 总成绩。但它们提供了以下机制证据:

| Run | 目标 | 观察 |
| --- | --- | --- |
| v18 | DeepSeek compact prompt | Lang-3 重复 empty message, 暴露 json mode/长 prompt 不稳定 |
| v19-v20 | empty recovery 和 timeout | timeout 开始被识别为 retryable/non-patch, 但早期 deadline 仍需修正 |
| v22-v24 | critical feedback | Lang-3 能注入失败方向指导, 但模型仍重复错误 float/double 策略 |
| v25 | candidate dedup | Lang-3 第二次重复 `numeric-type:add-float-before-double` 被拒绝, 未再浪费测试 |
| v28 | early dedup | Lang-3 从 v16 的 14 DeepSeek calls 降到 3 calls 后失败退出, 提高成本效率 |
| v29-v30 | regex constraints | Lang-10 prompt 含 `\s*+`, `Expected FDF failure`, `escapeRegex`, `CopyQuotedStrategy`, source grounding 改善 |
| v31 | nonpatch dedup | 运行被停止在 partial trace, Lang-10 已记录两次可应用但 visible failed 的候选 |

v31 只有 partial trace:

```text
artifacts/runs/d4j-lang10-real-v31-nonpatch-dedup/traces/self_evolved-Lang-10.partial.json
```

该 trace 显示 Lang-10 已完成 2 次 patch attempts, 两次 patch 均可应用且编译后 visible tests 失败。由于没有完成 summary, v31 不计入完整 benchmark 成绩。

## 7. 消融与机制分析

### 7.1 无 compact recovery

v18 暴露 DeepSeek 在长 prompt + JSON mode 下可能返回空 content。没有 recovery 时, Agent 会把这类失败记成普通 attempt, 浪费 patch budget。

修复后:

- 空响应触发 compact recovery prompt。
- 可降低 prompt budget scale。
- 可选择 fallback no-json-mode。
- 该类 retryable LLM error 不消耗 patch attempt。

### 7.2 无 non-patch timeout 处理

早期 focused run 中, timeout 被算作 patch attempt, 导致还没生成候选补丁就耗尽预算。

修复后:

- `DeepSeek subprocess exceeded ... deadline` 被识别为 retryable LLM error。
- feedback/self_evolved 中标记为 `non_patch_model_failure`。
- patch attempt index 不增加。
- prompt 自动缩小, 便于下一次恢复。

### 7.3 无 candidate dedup

v22-v24 中, Lang-3 多次重复同类错误策略。单纯把失败测试反馈塞回 prompt 并不能保证模型换策略。

修复后:

- 对失败 diff 生成 strategy signature。
- 重复策略直接拒绝, 不运行 compile/test。
- memory 写入 `duplicate_failed_patch_strategy` reflection。
- 对 Pass@3 重要, 因为三次机会应覆盖不同语义候选。

### 7.4 无 root failure retention

Lang-10 中如果下一轮只看 old text not found 或 patch apply failure, 模型会丢失初始 regex 语义错误。

修复后:

- 初始 trigger failure 一直保留。
- 新 feedback 追加到 root failure 后。
- 从 root failure 提取 regex/source needles。
- Prompt 中明确 `\s*+` 不应被错误放宽为 literal space flexible。

结果是 source grounding 变好, 但还未解决 Lang-10 语义补丁生成问题。

## 8. 成功案例与失败模式

### 8.1 成功案例

Chart 1-10 在 v16 中全部解决, 说明当前框架在以下任务上有效:

- modified class 明确。
- trigger failure 与源码路径对应关系直接。
- 小补丁即可修复。
- visible trigger tests 和 relevant regression tests 一致。

Lang-1, Lang-4, Lang-5, Lang-6, Lang-7, Lang-8, Lang-9, Lang-11 也通过, 说明对 Apache Commons Lang 的常见 API contract、boundary、formatting 类 bug 有一定泛化。

Math-5, Math-8, Math-9, Math-10 通过, 说明框架不是只对 Chart/Lang 有效。

### 8.2 Lang-3

主要失败:

- 模型反复生成 `Float` before `Double` 或相似 numeric type 策略。
- visible failure 说明方向不够, 但模型未生成正确的更细粒度类型处理。
- v25/v28 dedup 能避免重复测试, 但不能自动创造新的正确策略。

下一步应提升:

- 类型层级和 overload resolution 的诊断 prompt。
- 对 Java numeric conversion 的局部静态分析。
- 将 failed patch signature 映射到 forbidden semantic edit, 而不只是 candidate 去重。

### 8.3 Lang-10

主要失败:

- 初始补丁尝试把 `escapeRegex(regex, formatField, true)` 改成 `false`, 编译通过但 visible 失败。
- 后续候选出现 old text not found 或 brace imbalance。
- v29-v31 已能注入 regex-specific constraints, 但模型仍难以生成小而正确的 balanced hunk。

下一步应提升:

- 结构化 Java AST patch application。
- 对 regex builder 方法的局部执行/差分测试。
- 对可疑 source method 生成更短 exact old block, 减少 brace imbalance。

### 8.4 Math failures

v16 的 Math-1 到 Math-4 有大量 `llm_error`, 部分来自 transport/timeout 或 empty response。不应把这些全部归因于修复能力不足。v31 前已有 transport retry 和 non-patch timeout, 但尚未在完整 failed8 上验证。

Math-6/Math-7 包含 visible pass 后 regression fail 或多次 visible fail, 说明只靠 trigger feedback 容易过拟合。regression-aware memory 是针对这类问题的核心机制, 但还需要更多成功和失败 episode 来稳定学习。

## 9. 安全、成本与泛化

安全:

- API key 只允许在 shell 环境中设置。
- Trace redaction 会替换环境中的 key。
- SafePatch 禁止测试删除、build 文件修改、checkout 外路径和 unsafe path。
- Hidden/regression tests 不进入在线 prompt。
- 每次失败后 fresh checkout, 避免在污染状态上继续搜索。

成本:

- v16 使用 121 次 DeepSeek calls。
- 记录 prompt tokens 1,776,537 和 completion tokens 753,279。
- `estimated_cost_usd` 由环境中的费率变量计算, 当前 artifact 中费率为 0, 因此报告值是 0.0, 不是实际商业成本。
- v25-v31 的 candidate dedup 主要改善成本效率, 例如 Lang-3 focused run 能减少重复模型调用和测试运行。

泛化:

- Memory feature 使用 project、exception、trigger token、modified class, 不直接用 bug id 学答案。
- Prompt constraints 来自 visible failure、source snippets 和 memory reflections, 不是人工写入 benchmark patch。
- 仍有过拟合风险: 如果 memory 只在 30 个任务上反复更新, 可能学到 dataset-specific patterns。需要跨项目 split 或新 bug list 做外部验证。

## 10. 当前最佳版本选择

如果标准是“完整 benchmark 已验证成绩”, 最佳版本是:

```text
artifacts/runs/d4j30-self-evolved-real-v16
```

理由:

- 这是当前最高的完整 30-case 可比 run。
- solved 22/30, 高于 v1 的 12/30。
- 具有完整 `summary.json`, `metrics.csv`, `failure_analysis.md`, traces 和 patches。
- 没有 infrastructure failure, compile success rate 30/30。

如果标准是“下一轮继续实验的推荐架构”, 应使用当前 v31 前后的代码架构:

- 保留 v16 的多轮 self_evolved framework。
- 加入 DeepSeek empty/timeout recovery。
- 加入 candidate dedup 和 duplicate non-patch rejection。
- 加入 root failure retention。
- 加入 regex/source-needle constraints。
- 加入 test-skill 与 regression-aware memory。

严格报告中应写作:

```text
v16 是当前最佳已完成成绩; v31 架构是下一轮候选系统, 尚需完成 30-case rerun 后才能声称超过 v16。
```

## 11. 如何证明全量通过

不能用 focused run、人工修复或单个 bug 成功来证明全量通过。必须满足:

1. 使用同一 `configs/defects4j_30.json`。
2. 每个任务从 fresh buggy checkout 开始。
3. Agent 输入中只包含 issue metadata、visible trigger failures、source snippets 和历史 memory, 不泄漏 hidden/regression tests。
4. 补丁只能由 Agent 生成, 不能人工按 benchmark 答案改源码。
5. SafePatch 不能允许测试删除、build 修改或 checkout 外路径。
6. 每个 case 都输出 trace、patch、测试结果和 memory snapshots。
7. `summary.json` 显示:
   - `cases = 30`
   - `regression_pass_rate = 1.0`
   - `agent_failures = 0`
   - `infrastructure_failures = 0`
   - `unsafe_edit_rate = 0.0`
8. `metrics.csv` 每行 case 的 `status = solved`, `visible_pass = True`, `regression_pass = True`。

当前 artifact 不满足第 7 条, 因此不能声称全量通过。

## 12. 后续优化方向

1. AST-aware patching
   - 对 Java 方法、if block、return statement 做结构化替换, 减少 brace imbalance 和 old text not found。

2. Semantic candidate diversity
   - 将 failed patch signature 升级为语义约束, 要求后续候选改变 root cause hypothesis, 而不只是 diff hash 不同。

3. Local dynamic probes
   - 对 regex parser、numeric conversion、math edge cases 生成小型临时 probe, 作为 visible feedback 的补充。
   - Probe 只能在 scratch path 运行, 不能修改 benchmark tests。

4. Better test selection policy
   - 学习什么时候先 trigger, 什么时候直接 relevant。
   - 对历史 regression failure feature 提高 relevant/all test 优先级。

5. Cross-task validation
   - 将 memory 在 Chart/Lang 上训练, 在 Math 或另一个 Defects4J project 上测试。
   - 或使用 SWE-bench Lite 子集验证是否跨数据集泛化。

6. Cost-aware controller
   - 当重复 LLM error 或 duplicate strategy 过多时早停。
   - 将 DeepSeek max tokens、snippet budget、attempt bonus 作为可学习策略。

## 13. 结论

本项目已经实现一个真实 Defects4J self-evolving code repair Agent, 具备 Dockerized benchmark harness、DeepSeek patch generation、safe patch application、完整 trace logging、persistent memory 和多轮反馈修复。当前最好的完整 30-task run 是 `d4j30-self-evolved-real-v16`, 修复 22/30, Pass@1 为 17/30, Pass@3 为 21/30。

v31 前后的架构改进显示 self-improving 机制确实能减少无效重复尝试, 改善 source grounding 和失败反馈利用, 但还没有把完整 benchmark 成绩提升到 30/30。下一步应基于当前架构重跑完整 30-case benchmark, 并重点解决 Lang-3 的 numeric semantic repair、Lang-10 的 regex builder patching 和 Math 系列的 LLM transport/semantic failure。
