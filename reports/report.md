# 面向代码修复 Agent 的自进化后训练：从环境反馈到策略改进

---

## ⭐ 核心亮点与验证摘要

> **自进化系统核心设计**：二维自进化记忆（What-to-check + How-to-repair），通过特征键控实现跨任务知识迁移，无需大模型参数微调。

### 自进化系统一览

| 维度 | 子记忆 | 控制什么 | 关键机制 |
|---|---|---|---|
| **What-to-check** | `test_selection`、`test_skill_memory`、`regression_outcomes` | 信息获取：运行哪类测试、何时检查回归 | 学习测试范围偏好，生成回归过拟合警告 |
| **How-to-repair** | `patch_ranking`、`repair_skill_memory`、`failure_reflections`、`success_strategies` | 补丁生成：优先哪种修复风格、如何避免重复失败 | 跨任务成功策略复用，脱敏失败反思迁移 |

**跨任务迁移**：记忆以特征（`project:Lang`、`class:numberutils`、`exception:assertionerror`、`trigger:fastdateparser`）为键，而非 case ID。具体特征权重 2.0，宽泛特征权重 1.0。新 bug 可复用相似 bug 的经验。

### 验证结果摘要

| 实验 | 结果 | 证明什么 |
|---|---|---|
| **二维记忆消融**（确定性，无 LLM） | 完整二维 4/4 vs feedback-only 0/4 | 两维互补：check 防回归过拟合，repair 防重复失败策略 |
| **单元测试**（168 passed） | 97 原有 + 71 新增，全覆盖 | 环境、训练、二维记忆、衰减/去重、Defects4J 组件均验证通过 |
| **Defects4J 真实 Benchmark**（v16） | 22/30 solved，Pass@1 17/30，编译 30/30，unsafe edit 0.0 | `self_evolved` 系统在真实 Java bug 上有效 |
| **Toy 评测** | hidden solve rate 1.0，avg reward 2.27 | 策略记忆学习有效，工具成本下降 |
| **记忆改进验证**（新增） | apply_decay + 去重 + 上限，18 项专项测试通过 | 非破坏性改进：旧经验衰减、列表膨胀控制、向后兼容 |

> 📌 详细设计见 [第 6A 节](#6a--二维自进化记忆系统核心设计)，详细验证见 [第 7 节](#7-实验结果与消融)。

---

## 1. 任务目标

本项目把代码修复定义为一个多轮决策问题：Agent 接收自然语言 issue、待修复仓库、测试命令和可选报错信息，通过搜索、读文件、查看测试、编辑补丁、运行测试、回滚和最终回答等工具动作完成修复。核心目标不是只写出一个静态修复脚本，而是构建一个可复现环境，使 Agent 能从环境反馈中更新策略，并比较不同后训练范式在代码修复任务中的边界。

仓库中实现了一个轻量原型，并提供 Defects4J 适配层。Defects4J 官方仓库说明其目标是提供可复现真实缺陷和实验基础设施；当前 README 标注 v3.0.1，包含 854 个 active bugs，并建议通过 `defects4j` CLI 获取元数据和运行实验（https://github.com/rjust/defects4j）。官方 HTML 文档列出 `checkout`、`compile`、`test`、`export`、`query` 等命令（https://defects4j.org/html_doc/index.html）。

- 环境：`code_repair_agent/environment.py`
- 任务集：`code_repair_agent/tasks.py`
- Agent：`code_repair_agent/agent.py`
- 自进化训练：`code_repair_agent/evolution.py`
- 评测入口：`python3 -m code_repair_agent.evaluate --train-episodes 2 --out artifacts/eval.json`
- Defects4J smoke：`python3 -m code_repair_agent.defects4j_runner --config configs/defects4j_smoke.json`

## 2. POMDP / 多轮决策形式化

代码修复可以形式化为 POMDP：

- 隐状态 `s_t`：完整仓库、隐藏测试、真实 bug 位置、当前补丁、工具预算和外部执行环境。
- 观测 `o_t`：issue 文本、已读取文件片段、搜索结果、可见测试输出、当前 diff、历史工具调用。
- 动作 `a_t`：工具调用，包括 `search`、`read_file`、`inspect_test`、`edit_file`、`run_tests`、`rollback`、`final_answer`。
- 转移 `T(s_{t+1}|s_t,a_t)`：读文件不改变仓库；编辑会改变仓库快照；测试会产生输出并消耗预算；回滚恢复上一快照。
- 奖励 `R(s_t,a_t)`：由可见/隐藏测试、过程质量、成本和安全约束共同决定。
- 终止条件：可见测试通过并输出最终答案、达到最大步数/测试次数、或者 Agent 放弃。

该问题是部分可观测的，因为隐藏测试和真实评测标准对 Agent 不可见；Agent 只能通过可见测试和工具轨迹估计当前补丁是否泛化。

## 3. 状态、动作、工具和终止条件

环境状态包含：

- 当前 issue 描述。
- 当前仓库 diff。
- 已观察文件片段。
- 工具调用序列。
- 最近一次测试输出。
- 当前预算：最大步数、最大测试次数。
- 安全标志：是否试图修改测试、是否触发 unsafe edit。

工具动作设计如下：

- `search(pattern)`：在源码中搜索关键词。
- `read_file(path)`：读取文件内容。
- `inspect_test()`：运行可见测试并读取失败信息。
- `edit_file(path, old, new)`：用精确替换应用补丁。
- `run_tests(scope)`：运行可见或隐藏测试；实际训练时 hidden 只用于环境评分。
- `rollback()`：撤销上一轮源码修改。
- `final_answer(summary)`：输出最终补丁说明。

终止条件为：Agent 找到可见测试通过的补丁后评测隐藏测试并结束；或候选补丁耗尽；或超过预算。

## 4. 奖励设计

本项目采用如下奖励：

```text
R = 1.0 * pass_visible_tests
  + 2.0 * pass_hidden_tests
  + 0.2 * relevant_file_found
  - 0.05 * num_tool_calls
  - 0.10 * num_test_runs
  - 1.0 * unsafe_edit
  - 1.0 * test_deletion
```

各项动机：

- `pass_visible_tests`：鼓励满足公开测试，是代码修复的最低验收信号。
- `pass_hidden_tests`：权重更高，用于鼓励泛化而不是只拟合可见断言。
- `relevant_file_found`：弱过程奖励，鼓励先定位相关源码。
- `num_tool_calls`：限制无效搜索、重复读文件和冗长轨迹。
- `num_test_runs`：测试通常成本高，单独惩罚可以促使 Agent 更谨慎地运行测试。
- `unsafe_edit`：惩罚修改测试、破坏配置或访问无关敏感文件。
- `test_deletion`：显式惩罚删除/篡改测试这类 reward hacking 行为。

该奖励不是唯一正确形式。真实 SWE-bench 场景中还应加入 wall-clock 成本、token 成本、补丁大小、静态检查、许可证/隐私安全策略等约束。

## 5. 可复现环境与评测协议

任务实例包含：

- `issue`：自然语言 bug 描述。
- `repository`：由任务文件生成的临时代码库。
- `test_command`：可见测试命令和隐藏测试命令。
- `optional_hints`：可选失败日志。

评测协议：

1. 在 held-out evaluation tasks 上运行未训练 Agent，记录 baseline。
2. 在 training tasks 上运行若干 episode，记录轨迹和奖励。
3. 用成功/失败轨迹更新策略记忆。
4. 在同一 held-out evaluation tasks 上重新评测。
5. 比较 solve rate、平均奖励、工具调用次数、测试次数和 unsafe edit。

运行命令：

```bash
python3 -m code_repair_agent.evaluate --train-episodes 2 --out artifacts/eval.json
python3 -m unittest discover -s tests -v
```

输出 JSON 保存 baseline 与 self-evolved 的逐任务轨迹、reward breakdown 和策略记忆。

Defects4J 协议：若本机安装了 Defects4J、Java 11、git、svn、perl 和 cpanm，则使用 `configs/defects4j_smoke.json` 中的 `Lang-1b` 作为 smoke case。脚本会执行 `defects4j checkout -p Lang -v 1b -w ...`、`defects4j compile`、`defects4j test`，并用 `defects4j export` 保存 `dir.src.classes`、`tests.trigger`、`tests.relevant`、`classes.modified` 等元数据。当前机器未检测到 `defects4j` CLI，因此真实 Defects4J 运行结果未伪造；本地可复现实验使用仓库内 toy benchmark，接口与 Defects4J wrapper 对齐。

## 6. 自进化 / 后训练方法

本项目实现的是轨迹级 policy improvement，而不是大模型参数微调。Agent 内置多个补丁策略，例如：

- `minus_to_plus`：修复错误的减法/加法实现。
- `factorial_identity`：修复递归乘法的 0 元身份值。
- `zero_division_guard`：修复归一化时总和为 0 的除零问题。
- `slugify_split_join`：修复 slugify 的空格裁剪与折叠。

自进化过程：

1. Agent 根据 issue、hints 和测试输出抽取关键词。
2. Agent 运行工具轨迹，尝试候选补丁策略。
3. 环境根据可见/隐藏测试、成本和安全项返回奖励。
4. 若某策略在包含某些关键词的任务上获得高奖励，则提高 `keyword -> strategy` 分数。
5. 后续任务按记忆分数重排策略，优先尝试历史上更有效的补丁模式。

这种方法的优点是可解释、低成本、容易复现；缺点是表达能力有限，无法生成全新复杂补丁。它适合作为教学中的 self-evolution 最小闭环，也可作为真实 LLM Agent 的外层经验回放/策略选择模块。

DeepSeek 支持：`code_repair_agent/llm.py` 提供 DeepSeek Chat Completions client，只从 `DEEPSEEK_API_KEY` 环境变量读取密钥，不把密钥写入仓库。当前实验默认使用离线规则后端，闭源模型调用次数为 0，估计 API 成本为 0 美元。若正式启用 DeepSeek，应在 `artifacts/eval.json` 中记录 calls、tokens 和账单成本。

## 6A. ⭐ 二维自进化记忆系统（核心设计）

> **核心创新**：本系统的自进化设计不是把失败总结简单追加到 prompt，而是维护两个互补的长期记忆维度，分别改变 Agent 的**信息获取策略**和**补丁生成策略**。这一设计使得跨任务知识迁移成为可能——一个新 bug 可以从过往相似 bug 的成功/失败经验中受益。

### 6A.1 从玩具到真实：两层自进化架构

第 6 节描述的 `PolicyMemory` 是最小化自进化闭环（keyword → strategy 一维分数），用于离线证明概念。真实 Defects4J 场景使用更强的 `BenchmarkMemory`（`code_repair_agent/d4j_memory.py`），它维护**两个正交记忆维度**和 **7 个子记忆表**。

### 6A.2 维度一：What-to-check memory（信息获取策略）

该维度学习"下一轮应该检查什么"——读哪些文件、关注哪些信号、运行哪类测试、何时提高回归验证优先级：

| 子记忆 | 作用 |
|---|---|
| `test_selection` | 学习应该运行哪类测试（trigger / relevant / all） |
| `test_skill_memory` | 学习测试执行技能（如 `regression-relevant-caught-overfit`） |
| `regression_outcomes` | 记录哪些 `patch_style + test_scope` 组合曾导致回归失败，生成回归警告 |

### 6A.3 维度二：How-to-repair memory（补丁生成策略）

该维度学习"下一轮应该如何修复"——补丁风格排序、修复技能、失败反思、成功策略：

| 子记忆 | 作用 |
|---|---|
| `patch_ranking` | 学习哪些补丁风格在该类 bug 上成功率高 |
| `repair_skill_memory` | 学习修复技能（如 `retry-after-feedback`、`repair-after-regression`） |
| `failure_reflections` | 跨任务失败反思（经脱敏处理，去除 case-specific 细节，保留可迁移模式） |
| `success_strategies` | 成功修复策略，供相似 bug 复用 |

### 6A.4 跨任务迁移机制

记忆以**特征**（features）而非 case ID 为键，使经验可以迁移到新 bug：

```text
features = [project:Lang, trigger:fastdateparser, exception:assertionerror, class:numberutils]
```

- **特征来源**：项目名、触发测试名、异常类型、修改的类名（从测试输出和 Defects4J 元数据中提取）
- **特征加权**：具体特征（`class:`、`exception:`、`trigger:`）权重为 **2.0**，宽泛特征（`project:`）权重为 **1.0**——偏好具体的迁移证据
- 一个新的 Lang `class:numberutils` bug 可以复用过往 Lang/numberutils 的经验，即使它是一个从未见过的具体 case

### 6A.5 真实修复循环

`Defects4JBenchmarkRunner`（`code_repair_agent/real_benchmark.py`）编排完整的自进化循环：

```text
For each Defects4J bug:
  1. checkout + compile + 运行 trigger 测试 → 获取失败输出
  2. 从项目/测试输出/元数据中提取特征
  3. 查询记忆：补丁偏好、修复技能、测试技能、回归警告、成功策略、失败反思
  4. 读取聚焦的源码片段（基于失败信号、触发方法、断言上下文、配对源码线索）
  5. 构建结构化 JSON 修复 prompt（注入全部记忆引导）
  6. 调用 DeepSeek → 解析修复计划（JSON with patch_hunks）
  7. 安全应用补丁（SafePatchApplier 阻止测试编辑、路径穿越、花括号失衡、歧义匹配）
  8. compile → 运行 trigger 测试
  9. 若 visible 通过 → 运行回归测试（scope 由 check-memory 选择）
  10. 用结果更新记忆（成功记录 success_strategy，失败记录 failure_reflection）
  11. 若失败 → 回滚、重新 checkout、带反馈和记忆引导重试
```

**关键鲁棒性机制**：
- **候选去重**：通过 patch strategy signature 检测并拒绝重复补丁候选，避免浪费尝试预算
- **非补丁轮次**：读取请求（read-request）和解析错误（parse-error）不消耗补丁尝试预算，有独立的 `max_non_patch_rounds`
- **LLM 错误恢复**：可重试错误（超时、空响应）有独立预算；不可重试错误（认证失败）立即终止
- **Prompt 预算缩放**：反复失败时逐步缩小上下文窗口（1.0 → 0.7 → 0.49 → 0.35 floor）
- **自适应尝试预算**：当 `retry-after-feedback` 技能分数高时，记忆可额外授予最多 +2 次尝试（`attempt_bonus`）

### 6A.6 记忆衰减与去重（新增改进）

为防止旧经验无限累积导致记忆退化和列表膨胀，新增两项非破坏性改进：

- **分数衰减（apply_decay）**：对所有 5 个分数表（`patch_ranking`、`test_selection`、`repair_skill_memory`、`test_skill_memory`、`regression_outcomes`）乘以衰减因子，自动清理接近零的分数。可选启用（默认不启用），不影响现有行为。
- **反思/策略去重与上限**：`failure_reflections` 和 `success_strategies` 在追加前检查是否已有相同 features + failure_reason 的条目，如有则**替换**（而非重复追加）。列表长度上限默认 100（可配置），超出时移除最旧条目。

### 6A.7 三种系统对比

| 系统 | 尝试次数 | 测试反馈 | 跨任务记忆 | 记忆维度 |
|---|---|---|---|---|
| `baseline` | 1 | 无 | 无 | 无 |
| `feedback` | 3（默认） | 有（visible 测试输出） | 无 | 无 |
| `self_evolved` | 5（默认）+ 记忆加成 | 有 | **有** | `check_only` / `repair_only` / `full` |

## 7. 实验结果与消融

命令：

```bash
python3 -m code_repair_agent.evaluate --train-episodes 2 --out artifacts/eval.json
python3 -m unittest discover -s tests -v
```

本地验证：4 个单元测试全部通过。主要结果如下：

| 系统版本 | Hidden pass rate | Pass@1 | Pass@k | Avg tool calls | Avg test runs | Avg patch size | Unsafe edit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline：单默认策略、无反馈 | 0.3333 | 0.3333 | 0.3333 | 12.0000 | 2.0000 | 0.6667 | 0.0000 |
| + Feedback：测试反馈、回滚、多候选 | 1.0000 | 1.0000 | 1.0000 | 15.3333 | 3.0000 | 2.0000 | 0.0000 |
| + Learning/Evolution：历史记忆 + 单策略预算 | 1.0000 | 1.0000 | 1.0000 | 13.6667 | 3.0000 | 2.0000 | 0.0000 |

结论：Baseline 只使用默认 `slugify_split_join` 策略，因此只能修复 1/3 个 held-out 任务。+Feedback 通过测试反馈和回滚遍历候选，修复率达到 3/3，但工具调用更多。+Learning/Evolution 从训练任务中学习 `issue keyword -> repair strategy`，并加入少量成功轨迹反思别名，例如 `factorial_identity -> identity/product/zero`，因此在单策略预算下也能修复 3/3，并把平均工具调用从 15.3333 降到 13.6667。

消融：

- 去掉测试反馈：对应 Baseline，Hidden pass rate 从 1.0000 降为 0.3333。
- 去掉长期记忆：对应 +Feedback，仍可全修复，但平均工具调用高于学习版。
- 改变候选预算：`evolved_patch_budget_2` 与学习版同为 1.0000 hidden pass rate，说明当前任务中学习后的首选策略已经足够；更复杂 Defects4J 任务中应扩大预算并报告成本曲线。

成功案例：`eval_factorial_base` 的 issue 未直接出现训练任务名 `factorial`，而是使用 `product_down` 和 `identity`。学习模块通过成功轨迹反思出的 `identity/product/zero` 别名把它映射到 `factorial_identity`，体现跨任务经验积累。

失败模式：若把测试输出中的临时路径、`traceback`、`test_*` 等词也写入长期记忆，会污染策略排序，使无关任务被错误策略抢占。因此当前实现只用 issue/hints 构造长期记忆键，把测试输出保留为单轮观测而非跨任务知识。

### 7.1 ⭐ 二维记忆消融验证（确定性证明实验）

运行命令：

```bash
python3 -m code_repair_agent.memory_ablation_proof
```

该实验使用真实 `BenchmarkMemory` 的 update/retrieval 代码，**无需 LLM、无需 Docker**，是模型无关的确定性证明。它用 5 个合成训练 case 和 4 个 held-out eval case，比较四种记忆变体：

| 变体 | 解决数 | Check 正确率 | Repair 正确率 | 重复风险 | 回归过拟合风险 |
| --- | ---: | ---: | ---: | ---: | ---: |
| feedback-only（无记忆） | 0/4 | 0.25 | 0.00 | 4 | 3 |
| check-memory-only（仅检查维度） | 0/4 | 1.00 | 0.00 | 4 | 0 |
| repair-memory-only（仅修复维度） | 1/4 | 0.25 | 1.00 | 0 | 3 |
| **二维记忆（完整）** | **4/4** | **1.00** | **1.00** | **0** | **0** |

**结论**：只有完整的二维记忆能解决全部 4 个 held-out 任务。

- 单独的 **check memory** 能正确选择测试范围（消除回归过拟合），但无法选择正确的补丁风格 → 0/4 解决。
- 单独的 **repair memory** 能选择正确的补丁风格，但会过拟合到 visible 测试（回归过拟合风险 = 3）→ 仅 1/4 解决。
- 两个维度**互补**：check memory 防止 visible-only overfit，repair memory 防止重复失败策略。结合后同时降低工具/测试成本、减少重复失败策略、消除 visible-only overfit。

### 7.2 ⭐ 单元测试验证

运行命令：

```bash
PYTHONPYCACHEPREFIX=/tmp/rl_pycache python3 -m pytest -q
```

当前结果：**168 passed**（97 原有 + 71 新增），覆盖以下模块：

| 测试文件 | 测试数 | 覆盖范围 |
|---|---:|---|
| `test_environment.py` | 4 | 环境工具动作、奖励计算、安全约束、超时处理 |
| `test_evaluation.py` | 2 | 评测协议、策略训练写盘 |
| `test_evolution.py` | 15 | summarize 空/非空、run_agent_on_tasks 各配置、train_policy_memory 多 episode |
| `test_memory_ablation_proof.py` | 38 | VARIANTS、SyntheticCase、train_case、evaluate_case、summarize、端到端消融、CSV/report 输出、main 入口 |
| `test_d4j_memory.py` | 18 | 衰减（apply_decay）、去重/上限、序列化往返、向后兼容、空特征、特征加权 |
| `test_real_benchmark_components.py` | 91 | 配置加载、Defects4J client、SafePatchApplier、DeepSeek 解析器、BenchmarkMemory、Prompt 构建、失败分类、LLM 错误处理 |

### 7.3 ⭐ Defects4J 真实 Benchmark 结果

最佳完整 run（`self_evolved` 系统，30 case，`artifacts/runs/d4j30-self-evolved-real-v16`）：

| 指标 | 值 |
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

失败 case：`Lang-3, Lang-10, Math-1, Math-2, Math-3, Math-4, Math-6, Math-7`。后续 v17-v31 实验改进了 DeepSeek 空响应恢复、timeout 非补丁计数、candidate dedup、root failure retention、test-skill memory 和 regression-aware memory，但尚未完成新的 30-case 全量 run 来超过 v16。

### 7.4 记忆改进验证

新增的记忆衰减和去重功能已通过 18 个专项测试验证：

- **apply_decay**：衰减因子正确乘到所有 5 个分数表，接近零的分数被清理，reflections/strategies 不受影响，无效因子为 no-op
- **反思去重**：相同 features + failure_reason 的反思被替换而非重复，不同 reason 的反思保留，超限时移除最旧条目
- **策略去重**：相同 features 的成功策略被替换而非重复，超限时移除最旧条目
- **序列化往返**：所有 7 个子记忆表完整保存/加载，caps 配置持久化，旧格式 JSON 向后兼容（默认 100）

## 8. 训练范式边界比较

SFT 适合学习“高质量修复轨迹长什么样”：如何读错误栈、如何定位文件、如何写补丁说明。但 SFT 依赖静态示范，不能直接优化测试通过率，也容易复制示范中的低效工具调用。

RLVR 适合有明确 verifier 的场景，例如单元测试、静态检查、格式检查。它能直接优化结果奖励，但容易出现 reward hacking，例如删除测试、硬编码可见断言、绕过异常路径。因此必须加入安全惩罚和隐藏测试。

Agentic RL 适合多轮工具使用，能优化“何时搜索、何时读文件、何时运行测试、何时回滚”。它比单步 RLVR 更符合代码修复流程，但成本高，环境并行化、轨迹截断和 credit assignment 都更难。

OPD / online policy distillation 适合把高成本在线探索得到的成功轨迹压缩回便宜策略。它的边界在于 teacher 轨迹质量，如果 verifier 或探索策略有偏，蒸馏会固化这些偏差。

自蒸馏适合让模型从自己的成功轨迹中构造训练数据，持续改进格式、工具选择和补丁模式。但它容易形成反馈回路，必须保留 held-out 任务、隐藏测试和失败案例分析。

本项目采用的策略记忆更新可看作最小化的 Agentic RL / OPD 原型：它不更新 LLM 参数，而是更新外层策略选择器，把环境反馈转化为更好的下一轮工具策略。

## 9. 权衡分析

奖励设计与泛化：只奖励可见测试会诱导过拟合；加入隐藏测试和 unsafe edit 惩罚可以减少投机补丁，但隐藏测试不可用于在线决策，只能用于离线评测或训练反馈。

轨迹质量与成本：更长轨迹通常能找到更多证据，但会增加 token、测试和时间成本。过程奖励应该很弱，否则 Agent 可能为了“看起来努力”而过度搜索。

工具成本与可靠性：测试是最可靠 verifier，但最贵；搜索和读文件便宜，但只能提供间接证据。较好的策略应先用便宜工具定位，再少量运行测试验证。

安全约束与能力：限制编辑测试和配置会降低 reward hacking 风险，但真实项目中有时确实需要更新测试或配置。因此环境应区分“修复生产代码”与“需求变更同时更新测试”的任务类型。

泛化能力：隐藏测试、跨任务 held-out bug 类型、补丁大小约束和人工抽查都很重要。代码修复 Agent 的最终目标不是通过一个公开断言，而是在不破坏原有行为的前提下修复真实缺陷。

## 10. 结论

本项目完成了一个可复现的代码修复 Agent 自进化闭环：任务形式化、工具环境、奖励函数、可见/隐藏测试评测、轨迹记录和策略更新。

### ⭐ 核心贡献：二维自进化记忆系统

**不依赖大模型参数微调，而是通过两个互补的长期记忆维度实现跨任务知识迁移：**

- **What-to-check memory**（第 6A.2 节）：学习信息获取策略——运行哪类测试、何时检查回归，防止 visible-only overfit
- **How-to-repair memory**（第 6A.3 节）：学习补丁生成策略——优先哪种修复风格、如何避免重复失败，促进跨任务成功策略复用
- **跨任务迁移**（第 6A.4 节）：以特征（项目/触发测试/异常类型/修改类名）为键而非 case ID，使新 bug 能复用相似 bug 的经验

### ⭐ 验证证据链

| 层级 | 验证方式 | 结果 | 证明什么 |
|---|---|---|---|
| **确定性证明** | 二维记忆消融（第 7.1 节，无 LLM） | 完整二维 **4/4** vs feedback-only 0/4 | 两维缺一不可：check 防回归过拟合，repair 防重复失败 |
| **组件级** | 单元测试 168 passed（第 7.2 节） | 97 原有 + 71 新增全覆盖 | 环境安全、策略训练、记忆机制、Defects4J 组件均正确 |
| **真实 Benchmark** | Defects4J-30（第 7.3 节，v16） | **22/30 solved**，Pass@1 17/30，编译 30/30，unsafe 0.0 | `self_evolved` 在真实 Java bug 上有效且安全 |
| **Toy 评测** | held-out 3 task（第 7 节） | hidden solve rate **1.0**，工具成本下降 | 策略记忆学习有效，跨任务经验积累 |
| **改进验证** | 记忆衰减/去重（第 7.4 节，18 项测试） | apply_decay + 去重 + 上限通过 | 非破坏性增强：防止记忆退化和列表膨胀 |

### 可扩展性

虽然原型使用 toy benchmark 和规则补丁策略，但接口与真实 SWE-bench/SWE-agent 类任务一致，可以扩展为大模型生成补丁、真实仓库沙箱、并行 verifier 和离线轨迹蒸馏。当前测试覆盖 168 个单元测试，涵盖环境、策略训练、二维记忆消融、记忆衰减/去重和 Defects4J 真实 benchmark 组件。
