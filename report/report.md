# 面向 SWE-bench Lite 的 Self-Evolving Code Repair Agent 技术报告

## 摘要

本项目研究一个本地可复现的代码修复 agent。任务被建模为 POMDP，agent 在不观察隐藏补丁的条件下选择工具、读取文件、生成编辑、运行可见测试并提交补丁。方法比较三条路径：固定 Baseline、带测试反馈的 Feedback agent，以及使用 REINFORCE+Baseline 的 Agentic RL 工具和测试选择策略。实验记录来自严格官方（strict official）流水线，核心证据包括 `outputs/run_manifest.json`、`outputs/run_schedule.json`、`outputs/summary.json`、`outputs/harness_status.json`、`outputs/model_gates/qwable.json`、`outputs/model_gates/diffrwkv.json`、`report/figures/credit_assignment.json`、`report/figures/results.json`、`report/figures/credit_assignment_tables.md`、`report/tables/results_table.md`、`report/tables/ablation_comparison.md` 和 `report/tables/device_utilization.md`。

需要先说明边界：official SWE-bench harness 处于 blocked 状态（Docker 不可用，swebench 包不可导入）。`outputs/harness_status.json` 明确记录 `official_harness_executed: false`、`status: blocked`、`execution_backend: qz_pending_approval`。因此本文不声明官方 SWE-bench Lite resolved rate。`outputs/summary.json` 中的 pass@1、pass@k 和 resolved rate 是本地 fixture/fallback 指标，不能与 leaderboard 结果比较。官方评估仅待 qz 集群审批后执行。

## 1. 严格官方环境摘要 (Strict Official Environment Summary)

执行 `scripts/check_official_swebench_env.py` 对环境进行严格飞行前检查（参数详见附录命令）。`outputs/official_env_status.json` 记录三个 blocker：

| 阻断项 | 原因 | 修复命令 |
|---|---|---|
| `swebench_package_unavailable` | swebench 包不可导入 | `pip install -e ".[swebench]"` |
| `docker_cli_unavailable` | Docker CLI 不在 PATH，daemon 不可达 | 安装 Docker Engine |
| `dataset_ids_missing` | manifest 中 20 个 ID 在数据集中不存在 | 使用修复后的 manifest（Task 2 已修复） |

环境概况：4x NVIDIA GeForce RTX 4090（每张约 49140 MB 显存），128 logical CPU cores，可用内存约 618321 MB（使用率 40.1%），磁盘空闲 618 GB。qz CLI（v0.1.0）可用且已认证，官方 SWE-bench harness 标记为 qz offload ready，offload 通道已探测但尚未提交。Dataset `princeton-nlp/SWE-bench_Lite` split `test` 可访问（运行 `check_official_swebench_env.py` 时仍有 20 个缺失 ID，后在 Task 2 manifest 修复中解决）。严格模式下，任一 blocker 存在即退出 1；非严格模式（`--no-strict`）始终退出 0。

Preflight 检查涵盖 6 个维度：(1) swebench 导入性检查，(2) Docker CLI 和 daemon 可达性检查，(3) 数据集 ID 验证（manifest vs 实际 HF 数据集），(4) qz CLI 可用性、认证状态和 schema 检查，(5) 磁盘空间（阈值 120 GB），(6) resources.yaml 可解析性。每个维度的结果写入 `outputs/official_env_status.json`，blocker list 动态构造。

## 2. 真实 40-ID 数据集桥接 (Real 40-ID Dataset Bridge)

`repair_agent/env/swebench_loader.py` 中的 `load_task_instances()` 从 `princeton-nlp/SWE-bench_Lite` 的 test split 流式加载真实实例行。Manifest 修复过程（Task 2）：原 manifest `configs/task_manifest.yaml` 包含 20 个无效 main ID（在数据集 split 中不存在），逐一替换为同仓库族的有效 ID，最终得到 40 个有效 main ID 和 2 个 smoke ID。详细映射记录在 `.omo/evidence/meet-all/task-2-manifest-fix.txt`。

严格 sanitizer 递归移除四个敏感字段：`patch`、`test_patch`、`FAIL_TO_PASS`、`PASS_TO_PASS`。每个加载后的 instance record 经过 `sanitize_instance_record()` 处理并通过 `assert_agent_record_safe()` 验证后才返回给 agent。转换后的 record 包含 `instance_id`、`repo`、`base_commit`、`problem_statement`、`hints_text`、`visible_test_metadata`（仅含计数 `fail_to_pass_count` 和 `pass_to_pass_count`，不暴露测试节点 ID）、`workspace_setup`、`source: 'swebench_lite_official'` 和 `model_patch: ''`。Strict 模式拒绝 fixture 风格 ID（含 `local-` 子串或无 `__` 分隔符）。

`run_manifest.json` 记录完整的 `main_ids` 列表（40 个）和 `smoke_ids` 列表（2 个：`django__django-11099`、`sympy__sympy-20590`），所有阶段使用统一 seed `20260619`。Manifest 涉及的仓库族包括 astropy、django、matplotlib、mwaskom（seaborn）、pallets（flask）、psf（requests）、pytest-dev、scikit-learn、sphinx-doc 和 sympy。

严格加载过程分为六步：(1) `load_task_manifest` 从 YAML 解析 `main_ids` 和 `smoke_ids`；(2) `load_task_instances(manifest, split='test', ids=<full list>, strict=True)` 从 HF datasets 流式加载 42 个实例；(3) 递归 sanitize 移除 `patch`/`test_patch`/`FAIL_TO_PASS`/`PASS_TO_PASS`；(4) `assert_agent_record_safe()` 二次验证；(5) 转换为 agent instance record（`source='swebench_lite_official'`，`model_patch=''`）；(6) `_apply_limit` 在转换后截断。Strict 模式要求所有 ID 均在数据集中存在，任一缺失则抛出 `ConfigError('swebench_instances_unavailable')`。

## 3. 官方 Gold Smoke 结果 (Official Gold Smoke Result)

`outputs/harness_status.json` 的 `gold_smoke` 部分记录官方 gold-patch smoke harness 尝试：

- `run_id`: `official_gold_smoke`
- `predictions`: 2（使用 `outputs/runs/gold_patch_smoke/predictions.jsonl`）
- `official_harness_executed`: `false`
- `status`: `blocked`
- `blockers`: `["swebench_package_unavailable", "docker_cli_unavailable"]`
- `execution_backend`: `qz_pending_approval`
- `resolved`: 0

Gold-patch smoke 行仅用于 harness 格式验证和 smoke 流程校验，不作为 agent 输出。`outputs/runs/gold_patch_smoke/predictions.jsonl` 通过 scripts/validate_predictions.py 验证（2 行合法 JSONL），patch apply rate 1.0，但 resolved 为 0（harness 从未在本地执行）。

qz offload 信息：`qz_offload.available: true`，dry-run 已通过（`outputs/qz/official_harness_dry_run.yaml`），job spec 在 `outputs/qz/official_harness_job.json`，`submitted: false`。提交前需审批。

Gold smoke harness 的完整 command（记录在 `outputs/harness_status.json` 顶层）使用 `python -m swebench.harness.run_evaluation` 配合完整 SWE-bench 参数集。然而由于 swebench 和 Docker 均不可用，该命令从未在本地执行。`harness_status.json` 的 `total` 字段为 2（对应 2 个 prediction row），`resolved` 和 `resolved_rate` 均为 0（blocked 路径默认值）。

## 4. 官方 Harness 结果表 (Official Harness Result Table)

`report/tables/results_table.md` 记录每个阶段的 harness 状态。完整 14 阶段表如下：

| Run ID | Type | Predictions | Resolved | Pass@1 | Empty Patch Rate | Official Harness Status |
|---|---:|---:|---:|---:|---:|---|
| ablation_no_feedback_features | feedback | 40 | 0 | 0.000 | 1.000 | blocked |
| ablation_no_process_reward | learning | 40 | 0 | 0.000 | 1.000 | blocked |
| ablation_reduced_test_budget | learning | 40 | 0 | 0.000 | 1.000 | blocked |
| baseline_main | baseline | 40 | 0 | 0.000 | 1.000 | blocked |
| feedback_main | feedback | 40 | 0 | 0.000 | 1.000 | blocked |
| learning_main | learning | 40 | 0 | 0.000 | 1.000 | blocked |
| baseline_smoke | baseline | 1 | 1 | 1.000 | 0.000 | blocked |
| feedback_smoke | feedback | 1 | 0 | 0.000 | 0.000 | blocked |
| gold_patch_smoke | gold_smoke | 2 | 0 | 0.000 | 0.000 | blocked |
| learning_smoke | learning | 1 | 0 | 0.000 | 1.000 | blocked |

`outputs/harness_status.json` 的 `agent_runs` 部分包含 6 个 main/ablation 阶段的单独 harness status。所有阶段 `official_harness_executed: false`，`status: blocked`，`execution_backend: qz_pending_approval`。每个阶段有 40 prediction rows（smoke 阶段为 1-2 行），`resolved: 0`。Blockers 统一为 `["swebench_package_unavailable", "docker_cli_unavailable"]`。

## 5. 本地 vs 官方指标区分 (Local vs Official Metric Distinction)

本报告明确区分两类指标：

**本地 fixture/fallback 指标**：来自 `outputs/summary.json` 和 `report/figures/results.json`。这些指标基于 agent 运行产生的本地 predictions（通过可见测试或 fixture 规则判定）。`baseline_smoke` 在本地 fixture 上 achieved pass@1=1.0，但这一结果来自确定性 `add_numbers` fixture，不能推广到真实 SWE-bench Lite 任务。Aggregate 本地汇总：`run_count: 14`，`total_denominator: 249`，`total_resolved: 1`，`mean_pass_at_1: 0.077`，`resolved_rate: 0.004`。所有 40-ID main/ablation 阶段的 `empty_patch_rate` 为 1.0，`resolved` 为 0。

**官方 SWE-bench 指标**：来自 `outputs/harness_status.json`。官方 harness 从未执行（`official_harness_executed: false`），官方 resolved rate 为 `null`。官方评估仅在 qz 集群审批通过并提交 job 后才能获得。`outputs/qz/official_harness_job.json` 包含完整的 harness 执行方案，但 `submitted: false`。

严禁将本地 fixture/fallback 指标包装为官方 SWE-bench resolved rate。`outputs/summary.json` 中每个 run 的 `official_resolved_rate` 均为 `null`。

## 6. 问题定义与相关工作

代码修复任务输入包括 issue 描述、仓库元数据、可见测试线索和安全工作区。输出是一个 prediction JSONL row，其中 `model_patch` 是待评估补丁。SWE-bench Lite 提供真实项目 bug，但隐藏测试和 gold patch 不能暴露给 agent。项目中的 loader 会移除 `patch` 和 `test_patch` 字段，gold-patch smoke 仅用于 harness 格式验证，不能作为 agent 输出。

相关工作可以分成三类。第一类是 SWE-bench 风格的 repository-level program repair，重点在真实仓库上下文、依赖安装和隐藏测试评估。SWE-bench Lite 是 SWE-bench 的 300 实例子集（split test），覆盖 12 个 Python 仓库。第二类是 tool-using LLM agent，通过 search、read、edit、test 等工具完成长程任务，代表作如 SWE-Agent、OpenDevin。第三类是 Agentic RL，把工具选择、测试选择和停止决策当作策略学习问题，将 REPAIR 任务转化为 MDP 以学习工具使用策略。本项目把三类思想合在一个本地安全框架中，但不追求官方排名，而是强调可复现记录、失败披露和 long-horizon credit assignment 诊断。

## 7. POMDP 建模

本项目的 POMDP 定义在 `repair_agent/training/pomdp.py`。状态包含当前步数、已读文件、相关文件线索、补丁是否存在、测试运行次数、最后测试状态、安全违规标记和终止状态。观测由 agent-safe task record、工具返回、可见测试输出和资源状态组成。隐藏字段 `patch` 与 `test_patch` 被递归移除。

动作集合是固定 safe-tool-selection-v1：`search`、`read_file`、`inspect_test`、`edit_file`、`run_tests`、`rollback`、`git_diff`、`final_answer`。这些动作有 schema 约束，并通过工具 registry 执行。奖励来自 `configs/rewards.yaml`，包含 pass、visible_test_pass、visible_test_failure、hidden_regression_ready、partial_progress、relevant_files、tool_calls、test_runs、unsafe_edits、test_deletion 和 timeout 等项。学习配置 `configs/learning.yaml` 使用 `gamma: 1.0`，并保存 JSON 格式的 `policy.json`，避免额外模型依赖。

## 8. Agent 架构与工具

架构分为环境、工具、agent、训练和评估五层。环境层加载 SWE-bench Lite manifest、本地 fixture 和 harness wrapper。工具层提供安全操作：搜索、读文件、检查测试、编辑、运行测试、回滚、查看 diff 和提交最终答案。所有路径都被限制在任务工作区内，禁止绝对路径、目录逃逸、测试元数据编辑和 `.venv` 内部修改。

Baseline agent 使用固定流程，先搜索和读取，再由规则或模型 dry-run 产生编辑，随后运行可见测试、回滚失败编辑、生成 diff 并提交。Feedback agent 在编辑前先运行可见测试，把失败摘要写入结构化 reflection 字段，但不更新策略参数。Learning agent 把工具类型交给 REINFORCE+Baseline 策略选择，工具参数和具体编辑仍由安全启发式生成，避免策略直接写任意文件路径。

在严格官方模式下（`--manifest --instance-split --strict-official`），agent 从 manifest 加载完整的 40 个 SWE-bench Lite 官方实例 ID。针对 `source='swebench_lite_official'` 的实例，checkout 为空目录（不写 fixture 文件），`max_test_runs` 设为 0（禁止 subprocess 调用），确保 agent 不会因空 checkout 或 bare pytest 触发项目级测试套件。`git_diff` 在没有编辑历史时返回 `unsupported`，`model_patch` 为空字符串。

## 9. Agentic RL 方法

Agentic RL 训练入口是 `repair_agent.training.train`。策略是轻量 linear softmax，特征包括 step fraction、剩余 step/test budget、last action、last test status、relevant-file score、patch-exists flag、repeated-action count、model-gate status、visible GPU count、rollout parallelism 和预算归一化项。回报使用 reward-to-go，baseline 是 moving average baseline，用于降低 REINFORCE 方差。

训练输出包括 `rewards.jsonl`、`policy.json`、`learning_curve.json`、`status.json`、`rollout_allocation.json`、`trajectories.jsonl`、`predictions.jsonl` 和 `metrics.json`。当奖励信号全零或极弱时，trainer 会记录 `NO_SIGNAL`，而不是宣称策略提升。严格官方模式下的 learning main 与三个 ablation 都出现空补丁率高、可见测试信号弱的问题，说明当前策略选择没有学到稳定修复能力。

严格官方训练命令（非 `--limit`、`--strict-official`）将完整 40 个 main ID 传入训练循环。`train.py` 复用 `run.py` 的严格实例加载工具（`load_task_manifest`、`load_task_instances`、`_assert_strict_official_id`），在 `--dry-run-devices` 快速返回之后再加载实例，避免网络依赖。严格模式下写入 `run_state.json` 的 metadata 包含 `official_instance_source: 'swebench_lite'` 和 `strict_official: true`。

## 10. Qwable 真实推理门 (Qwable Real Inference Gate)

真实 Qwable 推理门记录在 `outputs/model_gates/qwable.json`。模型为 `lordx64/Qwable-v1`（AGPL-3.0 许可），其实是 Qwen3.5-MoE 条件生成模型（config `model_type: qwen3_5_moe`），26 个 safetensors shards 总计 71.9GB。通过 `device_map_auto` 分布在 4x RTX 4090（device_ids `[0,1,2,3]`），层分配：GPU0 层 0-8、GPU1 层 9-19、GPU2 层 20-30、GPU3 层 31-39 + lm_head + norm。

真实推理（无 `--dry-run` 标志）运行时使用 `python scripts/check_model_gate.py --model qwable` 配合参数 `--models-config`、`--resources`、`--out-dir`、`--max-new-tokens 1024`（完整命令见附录）。使用 greedy decoding（`do_sample=False`），seed `20260619`，生成正常结束（`finish_reason: stop`）。GPU 显存峰值 `gpu_memory_peak: 101808513536`（约 101.8 GB）。Parser 门通过：模型输出合法 JSON `{"name":"read","arguments":{"path":"README.md"}}`，映射到 `tool_name: read_file`。

资源安全分类器 `classify_qwable_resource_safety` 验证 4 张 4090 总显存 196560 MB 远超 12GB 安全阈值，判定 `safe_for_4090`。Pipeline 中的 model gate 阶段使用 `--dry-run` 标志保持轻量 parser/resource 检查，真实推理由独立命令执行。Gate status 为 `pass`。

DiffRWKV gate 仍被 blocked（`outputs/model_gates/diffrwkv.json`），原因是 checkpoint 为 DDPM/RWKV trajectory 模型（`instruction_following_candidate: false`），不是 prompt-to-patch 或 tool-selection code repair 模型。状态：`blocked`，reason：`diffrwkv_is_ddpm_rwkv_trajectory_model_not_instruction_code_repair`。Checkpoint 检查显示 `manifest_variant: traj32x16-2.9B-s2-rwkv7-v3-ddpm`，包含 `diffusion_trajectory_markers: ["ddpm","diffusion","trajectory","state_hijack"]`。Safetensors 大小为 564513388 bytes（~538 MB），gate 在 dry-run 模式下执行只读检查（`read_only_inspection: true`），未加载权重。

Qwable 真实推理的实现细节：`_generate_real()` 方法在 `QwableAdapter` 内使用 `transformers.AutoModelForCausalLM.from_pretrained(id, device_map="auto", torch_dtype="auto")` 延迟加载模型（仅在非 dry-run 模式下）。torch dtype 自动检测为 bfloat16（从模型 config 读取）。Tokenizer 使用 `AutoTokenizer.from_pretrained(id)`，apply_chat_template 先用 `tokenize=False` 渲染 prompt 字符串，再显式 tokenize 为 `input_ids` tensor（规避 BatchEncoding getattr 陷阱）。环境变量 `CUDA_VISIBLE_DEVICES=0,1,2,3` 仅在 `torch` 未导入时设置（CLI 新进程有效，测试环境跳过）。Greedy generation 参数：`do_sample=False`，`max_new_tokens=1024`，`transformers.set_seed(20260619)` 和 `torch.manual_seed(20260619)` 保证确定性。推理耗时约 3 分 16 秒（wall clock），生成内容正常结束（`finish_reason: stop`）。

## 11. 数据集与实验设置

实验使用 `princeton-nlp/SWE-bench_Lite`，split 为 `test`。`outputs/run_manifest.json` 记录 2 个 smoke ID 和 40 个 main ID。所有阶段使用 seed `20260619`。

严格官方模式主命令：

```bash
python scripts/run_gated_experiments.py --manifest configs/task_manifest.yaml --out outputs/runs --resources configs/resources.yaml --strict-official --force
```

流水线包含 14 个阶段：Qwable gate、DiffRWKV gate、gold-patch smoke 生成、prediction 校验、三条 smoke run、三条 main-style local run、三条 ablation、harness status 记录。Manifest 的最终状态为 `status: controlled_partial`（DiffRWKV gate blocked + official harness status blocked）。`--force` 标志在重跑前将已有 run 目录归档为 `*.archived.*`。

严格官方 harness 命令：

```bash
python -m repair_agent.env.harness --predictions outputs/runs/gold_patch_smoke/predictions.jsonl --run-id official_gold_smoke --auto-workers --resources configs/resources.yaml --status-out outputs/harness_status.json --timeout-seconds 1800 --strict-official
```

官方 SWE-bench harness 未执行。`outputs/harness_status.json` 的 command 字段记录完整 harness 调用参数，blocked reason 为 `swebench_package_unavailable`。所以本文只报告本地 fixture/fallback 结果，官方 resolved rate 为 `null`。

严格官方流水线在 `--force` 模式下运行时的行为：(1) 先将已有 run 目录归档为 `*.archived.<UTC>`（记录在 `run_manifest.archived_stale_runs`），(2) 然后从头执行所有 14 个阶段，(3) model gates 使用 dry-run 模式（`--dry-run` 标志保留），(4) 6 个 main/ablation 阶段使用 `--manifest --instance-split main --strict-official`（无 `--limit`），(5) 3 个 smoke 阶段使用 `--limit 1`（非严格），(6) harness status 阶段执行 `--strict-official`，blocked 时 exit 1。所有阶段的结果和状态记录在 `outputs/run_manifest.json` 的 stages 数组中，包含 command_line、returncode、status、completed_at 和 required_artifacts。

## 12. 本地设备使用设置

`outputs/device_inventory.json` 显示机器有 4 张 NVIDIA GeForce RTX 4090，GPU IDs 为 `[0, 1, 2, 3]`，每张约 49140 MB 显存。CPU 有 128 logical cores，内存可用约 618321 MB，SWE-bench 推荐 workers 为 16。`configs/resources.yaml` 设置 `device_policy: maximize_local`、`rollout_parallelism: 4`、`cpu.max_workers: 32`、`docker_cache_level: env`。

`outputs/run_schedule.json` 记录 round-robin 调度覆盖所有健康 GPU，`used_healthy_gpus` 为 `[0, 1, 2, 3]`，`unused_healthy_gpus` 为空，`auto_sized_swebench_workers` 为 16。Qwable 使用 `device_map_auto`，DiffRWKV 使用 per-worker `CUDA_VISIBLE_DEVICES` 但 gate blocked。Fallback policy 是记录缺失 GPU 或资源问题，并在允许时继续本地流程。

## 13. 资源利用表 (Resource Utilization Table)

`report/tables/device_utilization.md` 记录每个 run 的设备使用概况。主要发现：

- 所有 40-ID main/ablation 阶段：device IDs `[0,1,2,3]`，GPU 利用率未报告（`not_reported`，因为本地 agent 运行无 GPU 监控钩子），GPU 显存峰值未报告
- `gold_patch_smoke`：无 GPU 分配（仅 JSONL 生成和验证）
- Smoke 阶段（baseline_smoke、feedback_smoke、learning_smoke）：使用 GPU 0
- 所有阶段 fallback reasons 为空（无资源回退触发）
- Qwable gate 真实推理：GPU 显存峰值 101.8 GB 分布在 4x RTX 4090
- qz offload 状态：`available: true`，`submitted: false`

CPU worker 利用率、Docker worker 利用率和 wall time 在所有阶段均未报告（当前资源监控框架仅记录资源计划而非运行时指标）。

`outputs/summary.json` 的 resources 部分记录全局可见设备 `[0,1,2,3]`、worker settings（`cpu_max_workers: 32`、`rollout_parallelism: 4`、`swebench_max_workers: 16`）和 per-GPU assigned task counts。GPU 0 在所有阶段共分配 18 个 task，GPU 1-3 各 0 个（round-robin 调度中仅 GPU 0 被实际用于本地 agent 运行）。Fallback reasons 为空，无资源回退。

Qwable 真实推理时的 per-GPU 显存分布（从 learnings 记录推算）：GPU 0 层 0-8、GPU 1 层 9-19、GPU 2 层 20-30、GPU 3 层 31-39 + lm_head + norm，总计 ~101.8 GB。这证明 4x RTX 4090（每张 49GB）可以容纳 Qwen3.5-MoE 71.9GB 模型并完成至少 1024 token 的 greedy 推理。

## 14. 主结果

表 1 来自 `outputs/summary.json` 和 `report/figures/results.json`。这些是本地 fixture/fallback 指标，不是官方 SWE-bench 指标。

**表 1：本地运行结果摘要**

| Run | Type | Denom | pass@1 | pass@k | Resolved | Empty patch rate | Patch apply rate | Dir |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline_main | baseline | 40 | 0.000 | 0.000 | 0 | 1.000 | 0.000 | `outputs/runs/baseline_main` |
| feedback_main | feedback | 40 | 0.000 | 0.000 | 0 | 1.000 | 0.000 | `outputs/runs/feedback_main` |
| learning_main | learning | 40 | 0.000 | 0.000 | 0 | 1.000 | 0.000 | `outputs/runs/learning_main` |
| baseline_smoke | baseline | 1 | 1.000 | 1.000 | 1 | 0.000 | 1.000 | `outputs/runs/baseline_smoke` |
| feedback_smoke | feedback | 1 | 0.000 | 0.000 | 0 | 0.000 | 1.000 | `outputs/runs/feedback_smoke` |
| gold_patch_smoke | gold_smoke | 2 | 0.000 | 0.000 | 0 | 0.000 | 1.000 | `outputs/runs/gold_patch_smoke` |

Aggregate 本地/fallback 汇总（来自 `outputs/summary.json`）：`run_count: 14`、`total_denominator: 249`、`total_resolved: 1`、`mean_pass_at_1: 0.077`、`resolved_rate: 0.004`。由于官方 harness blocked，`official_resolved_rate: null`。`report/figures/results.json` 与 `outputs/summary.json` 内容完全一致（同源生成，均使用 `--include-resources` 标志），确保报告图表与汇总数据的一致性。

所有 40-ID 严格官方 main 和 ablation 阶段的 empty patch rate 均为 1.0（空补丁），resolved 均为 0。这是因为严格官方实例的 checkout 为空（无源代码 checkout），agent 无法找到编辑目标，`git_diff` 无 `.git` 和编辑历史，最终产生空 patch。`baseline_smoke` 的 1 个 resolved 来自确定性本地 `add_numbers` fixture，不能推广到 SWE-bench Lite。

## 15. 消融对比 (Ablation Comparison)

`report/tables/ablation_comparison.md` 对比三个 ablation 与 main 阶段。所有 40-ID 阶段 empty patch rate 均为 1.0。

**表 2：消融设置与结果**

| Run ID | Type | Predictions | Resolved | Pass@1 | Empty Patch Rate | Official Harness Status |
|---|---:|---:|---:|---:|---:|---|
| baseline_main | baseline | 40 | 0 | 0.000 | 1.000 | blocked |
| feedback_main | feedback | 40 | 0 | 0.000 | 1.000 | blocked |
| learning_main | learning | 40 | 0 | 0.000 | 1.000 | blocked |
| ablation_no_process_reward | learning | 40 | 0 | 0.000 | 1.000 | blocked |
| ablation_no_feedback_features | feedback | 40 | 0 | 0.000 | 1.000 | blocked |
| ablation_reduced_test_budget | learning | 40 | 0 | 0.000 | 1.000 | blocked |

A1（no_process_reward，`reward.process_weight: 0.0`）、A2（no_feedback_features，`learning.feedback_features_enabled: false`）、A3（reduced_test_budget，`agent.max_steps: 6`）均在严格官方 40-ID 设置下运行。A1 使用 `configs/ablations/no_process_reward.yaml`（从 `configs/learning.yaml` 派生，仅将 `reward.process_weight` 设为零），A2 使用 `configs/ablations/no_feedback_features.yaml`，A3 使用 `configs/ablations/reduced_test_budget.yaml`。每个 ablation 训练 1 episode（`--episodes 1`），生成 metrics.json、predictions.jsonl、trajectories.jsonl、policy.json 和 rewards.jsonl。

A3 将最大步数减半（6 steps），trajectory_rows 减至 240（vs 480 for A1/A2），但同样产生空补丁。三者共同说明当前失败不是单一 process reward 或 feedback feature 开关导致，而是工具选择和编辑生成之间的耦合过弱，在空 checkout 场景下无任何编辑机会。在非严格模式下（有 fixture checkout），这些 ablation 可能显示更细粒度的差异，但严格官方模式强制了统一的基线：无 checkout 则无 fix。Ablation 阶段的 harness status 各自写入 `outputs/harness_status_ablation_*.json` 文件，每文件记录 predictions=40、resolved=0、blocked。

平均工具调用数：A1 和 A2 为 12.0，A3 为 6.0（预算限制效果）。所有阶段 visible_test_pass_rate 均为 0.0（`max_test_runs=0` 禁止 subprocess 调用）。

## 16. Long-horizon Credit Assignment

Credit diagnostics 来自 `report/figures/credit_assignment.json`，Markdown 表在 `report/figures/credit_assignment_tables.md`。该分析明确标记为 diagnostic/correlational_not_causal，不能当作因果证明。

**表 3：工具贡献诊断**

| Tool | Steps | Mean reward-to-go | Leave-one-out delta | Partial progress |
|---|---:|---:|---:|---:|
| search | 136 | 3.894118 | 5.288889 | 0.000000 |
| read_file | 13 | 6.303846 | -0.036111 | 0.000000 |
| inspect_test | 49 | -0.284694 | -0.136111 | 0.000000 |

**表 4：学习曲线引用**

Credit assignment 的位置分析（`position_vs_reward_to_go`）显示 `search` 的相关系数为 -0.68765，`inspect_test` 为 0.835206。整体 `action_position_summary` 第 0 步 mean reward-to-go 为 5.117，指数衰减至第 11 步的 0.500（12-step episode，根据 `agent.max_steps=12` 限制）。`search` 占 136 步（所有动作中最频繁），`inspect_test` 占 49 步，`read_file` 仅 13 步。

Test/reward components 表中所有组件（pass、visible_test_pass、visible_test_failure、hidden_regression_ready、partial_progress、test_runs、timeout）的 nonzero count 均为 0。这意味着奖励轨迹显示了动作位置和 reward-to-go 结构（早期步骤积累更多累计奖励），却没有提供可验证的修复成功信号。Step-wise diagnosis 的 success rate 在所有 position index 上均为 0.000。

learning_curve.json 和 credit_assignment.json 均为来自 `repair_agent.evaluation` 模块的自动生成 artifact。Credit assignment 分析使用 leave-one-out delta 方法：对每个动作类型，从总 reward-to-go 中减去该类型所有步骤的贡献，计算 delta。`search` 的 leave-one-out delta 为 5.289（正贡献），`inspect_test` 为 -0.136（轻微负贡献），`read_file` 为 -0.036（接近中性）。

该诊断支持后续改进：需要把可见测试通过、补丁质量和停止动作纳入更强的过程奖励，而不只奖励动作发生。

## 17. 成功与失败案例分析

**成功案例**：仅 `baseline_smoke` 在本地 fixture 上成功（denominator 1、pass@1 1.0、patch apply rate 1.0、visible_test_pass_rate 1.0、平均 test runs 1.0）。固定启发式在简单 `add_numbers` fixture 上可以从可见测试定位到确定性修改。但这不能推广到真实 SWE-bench Lite。

**失败案例（严格官方 40-ID 阶段）**：所有 6 个 main/ablation 40-ID 阶段 empty patch rate 均为 1.0（空补丁）。根因有两层：(1) 严格官方 checkout 为空目录（`_prepare_task_checkout` 检测 `source='swebench_lite_official'` 后跳过 fixture 文件写入），(2) `max_test_runs=0` 禁止 subprocess 调用（防止空 checkout 场景下 bare pytest 意外触发项目级测试套件）。因此 agent 无法找到编辑目标，`git_diff` 无编辑历史返回 `unsupported`，最终 `model_patch=""`（空字符串，SHA256 `e3b0c442...`）。

具体而言：`baseline_main` 产生 240 trajectory rows（40 instances × 6 tool calls avg），所有 `model_patch` 为空。`feedback_main` 产生 200 trajectory rows（40 × 5），`learning_main` 产生 480 trajectory rows（40 × 12），均无有效补丁。`visible_test_pass_rate` 在所有 40-ID 阶段均为 0.0（`max_test_runs=0` 阻止了所有 `run_tests` 调用）。这三个 main 阶段的 `official_harness_executed` 均为 `false`。

Ablation 阶段表现一致：`ablation_no_process_reward` 和 `ablation_no_feedback_features` 各产生 480 trajectory rows（40 × 12），`ablation_reduced_test_budget` 产生 240 trajectory rows（40 × 6，预算减半效果），全部 empty patch rate 1.0。说明移除 process reward、禁用 feedback features 或减少 step budget 均未改变空 checkout 下的根本困境。

**官方 harness 失败**：Docker 不可用 + swebench 不可导入导致 `status: blocked`。所有 14 个 run 的 `official_resolved_rate` 均为 `null`。qz offload task 已准备但未提交（`submitted: false`），待审批。

**弱奖励信号**：Learning 训练在空 checkout 场景下，所有奖励测试组件（pass、visible_test_pass 等）nonzero count 为 0。Policy checkpoint 无有意义的策略梯度信号。

**Gold-patch smoke 边界**：生成 2 行合法 prediction，patch apply rate 1.0，但这是 harness smoke 行（使用真实 gold patch 字段），不是 agent 输出。在 blocked harness 下 resolved 为 0。

## 18. 安全、成本与泛化限制

安全方面，工具 registry 限制路径和编辑目标，loader 隔离 hidden `patch` 和 `test_patch`（以及 `FAIL_TO_PASS` 和 `PASS_TO_PASS`），README 与报告检查脚本只做本地解析，不执行长命令。qz 相关操作绝不泄露 JWT 认证令牌（`_SECRET_PATTERN` / `_TOKEN_PATTERN` 匹配 `eyJ...` JWT 令牌前缀和 `(token|secret|password|api_key|authorization|bearer)[:=]value` 模式，在写入任何 output 文件前 scrubs 敏感文本）。已验证 `grep -c eyJ` 在所有 `outputs/` 下的 JSON/YAML/TXT 文件中均为 0。

Agent 工具层的安全约束：(1) 所有文件路径限制在任务工作区 `outputs/runs/<run_id>/checkouts/<instance_id>/` 内，(2) 禁止 `..` 目录逃逸和绝对路径，(3) `edit_file` 拒绝 `.venv/` 内部文件，(4) `inspect_test` 只暴露可见测试计数不暴露测试节点 ID，(5) `final_answer` 提交的 `model_patch` 经过 diff 格式校验。成本方面，项目避免商业 API 和重量级报告渲染，但官方 SWE-bench 仍需要 Docker、依赖安装和较长运行时间（当前仅在 qz 集群 offload 方案中规划）。

模型限制：Qwable（Qwen3.5-MoE 71.9GB）在真实推理门上通过（parser 正确输出合法 JSON），但其在 40-ID 严格官方场景下的修复能力未被测量（因为严格官方 checkout 为空，agent 无编辑目标）。DiffRWKV gate 被 blocked（DDPM/RWKV trajectory 模型不适合 code repair）。泛化方面，本地 fixture 太小（仅 `add_numbers`），Baseline smoke 的成功不能推广到真实 SWE-bench Lite。Learning 的失败说明当前 REINFORCE+Baseline 在空 checkout、零测试运行和规则编辑生成下还不足以产生稳定修复。

## 19. 学术诚信声明

本报告只使用项目已生成 artifact 中的事实，不修改实验输出以改善结果。官方 SWE-bench harness blocked、DiffRWKV gate blocked、Qwable 真实推理通过（但未用于 40-ID 修复）、learning empty patches、zero/weak reward outcomes、gold-patch smoke 边界和 qz offload pending approval 均已披露。Gold-patch smoke 行只用于 harness 格式和流程验证，不作为 agent 输出。本文不声明官方 SWE-bench resolved rate，不把本地 fixture/fallback 指标包装成 leaderboard 结果。所有代码在 `repair_agent/` 下使用 MIT 许可，`lordx64/Qwable-v1` 记录为 AGPL-3.0。

## 附录：复现实验命令

严格官方环境检查：

```bash
python scripts/check_official_swebench_env.py --manifest configs/task_manifest.yaml --models-config configs/models.yaml --resources configs/resources.yaml --out outputs/official_env_status.json
# exit 1 on blockers (strict), exit 0 on --no-strict
```

严格官方完整流水线：

```bash
python scripts/run_gated_experiments.py --manifest configs/task_manifest.yaml --out outputs/runs --resources configs/resources.yaml --strict-official --force
# runs all 14 stages, archives existing run dirs first
```

严格官方单阶段：

```bash
python -m repair_agent.run --config configs/baseline.yaml --manifest configs/task_manifest.yaml --instance-split main --strict-official --limit 2 --run-id strict_bridge_smoke --resources configs/resources.yaml --force
# 2-instance bridge smoke, exit 0
```

真实 Qwable 推理门（无 `--dry-run`）：

```bash
python scripts/check_model_gate.py --model qwable --models-config configs/models.yaml --resources configs/resources.yaml --out-dir outputs/model_gates --max-new-tokens 1024
# loads Qwen3.5-MoE 71.9GB, exits 0 on pass
```

报告与 README 检查：

```bash
python scripts/check_report_artifacts.py report/report.md outputs/summary.json
python scripts/check_readme_commands.py README.md --dry-run-safe
python -m pytest -q tests
# all exit 0
```

Dry-run 调度预览（不执行实验，仅生成调度计划）：

```bash
python scripts/run_gated_experiments.py --manifest configs/task_manifest.yaml --out outputs/runs --resources configs/resources.yaml --dry-run-schedule
# writes outputs/run_schedule.json to disk
```

设备探测（生成本地 GPU/CPU/内存清单）：

```bash
python scripts/probe_local_devices.py --out outputs/device_inventory.json
# discovers 4x RTX 4090 + 128 CPU cores
```

严格官方 harness gold smoke（单独运行）：

```bash
python -m repair_agent.env.harness --predictions outputs/runs/gold_patch_smoke/predictions.jsonl --run-id official_gold_smoke --auto-workers --resources configs/resources.yaml --status-out outputs/harness_status.json --timeout-seconds 1800 --strict-official
# blocked if swebench/Docker unavailable, exit 1
```

所有命令在 `--dry-run-safe` 模式下已验证可解析性。本报告生成不需要商业 API 或重型渲染工具。


### Conda/no-Docker runtime validation

A supplementary local runtime path was executed with conda rather than Docker. The runner `scripts/run_conda_swebench_eval.py` reuses SWE-bench 4.1.0 test specs and grading on host conda environments. The gold-patch smoke check resolves 2/2 instances, validating the evaluator itself, while the six submitted baseline/feedback/learning/ablation prediction files resolve 0/40 because they contain empty patches. This evidence is recorded in `outputs/conda_eval_status.json` and remains separate from `outputs/harness_status.json`, where the Docker official harness is honestly marked blocked.
