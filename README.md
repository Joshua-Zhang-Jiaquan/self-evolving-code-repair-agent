# Self-Evolving Code Repair Agent with Agentic RL

> 中文概述：本项目实现一个面向 SWE-bench Lite 代码修复任务的本地可复现实验框架。核心比较对象是 Baseline、Feedback、REINFORCE+Baseline 三种工具和测试选择策略，并记录 long-horizon credit assignment 诊断。当前结果来自本地 fixture/fallback 流水线，不是官方 SWE-bench resolved rate。

This repository contains a local-first code repair agent for studying Agentic RL on SWE-bench Lite style tasks. It includes safe repair tools, deterministic local fixtures, model gates, GPU scheduling records, run summaries, ablation configs, and a Chinese-first technical report.

## What is included

| Area | Files |
|---|---|
| Agent and tools | `repair_agent/agent`, `repair_agent/tools`, `repair_agent/run.py` |
| RL method | `repair_agent/training/pomdp.py`, `repair_agent/training/train.py`, `configs/rewards.yaml` |
| SWE-bench wrapper | `repair_agent/env`, `configs/task_manifest.yaml`, `scripts/make_gold_smoke.py` |
| Model gates | `configs/models.yaml`, `scripts/check_model_gate.py`, `outputs/model_gates/*.json` |
| Experiment records | `outputs/run_manifest.json`, `outputs/run_schedule.json`, `outputs/summary.json` |
| Report | `report/report.md`, `report/figures/credit_assignment.json`, `report/figures/credit_assignment_tables.md` |

## Important status disclosure

The official SWE-bench harness is blocked in strict official mode when Docker image building is unavailable. `outputs/harness_status.json` records `official_harness_executed: false`, `status: blocked`, `execution_backend: qz_pending_approval`, and `max_workers: 16`. A later local strict smoke attempt can pass the package/Docker preflight and reach the official harness, but local Docker image building is still blocked in this workspace by `unshare: operation not permitted`; strict mode records this as `status: fallback` and exits nonzero.

To remove the Docker dependency, this project now includes a **Defects4J fallback evaluator** in `repair_agent/env/defects4j_harness.py`. When the Docker-based SWE-bench harness is blocked or fails, the harness wrapper automatically falls back to local Defects4J evaluation if:

1. Defects4J is installed (`DEFECTS4J_HOME` is set or the framework is found at `/tmp/opencode/defects4j`, `~/defects4j`, etc.), and
2. The prediction file contains Defects4J-formatted instance ids (`Project_BugId`, e.g. `Lang_1`, `Math_5`).

This project does not claim an official SWE-bench Lite resolved rate. Local fixture, fallback, and Defects4J metrics in `outputs/summary.json` are separate. Official SWE-bench evaluation is still pending a Docker-capable environment or qz cluster approval.

Model gates are also explicit:

| Model | Gate status | Meaning |
|---|---|---|
| `lordx64/Qwable-v1` | `pass` | Real inference gate passed (Qwen3.5-MoE 71.9GB, 4x RTX 4090, GPU peak ~101.8GB). Dry-run parser and resource gate also pass. License note: AGPL-3.0. |
| DiffRWKV local checkpoint | `blocked` | The checkpoint is a DDPM/RWKV trajectory model, not an instruction-following code repair model. It is not counted as a repair baseline. |

## Install

Use Python 3.12. The commands below are local-only and don't require commercial APIs.

```bash
# safe: setup
conda env create -f environment.yml
conda activate repair_agent
```

or:

```bash
# safe: setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

## Model setup and gates

Model metadata lives in `configs/models.yaml`. The Qwable gate checks parser and resources in dry-run mode. The real inference gate loads the full model and generates a test completion. The DiffRWKV gate inspects checkpoint metadata without loading the external checkpoint.

```bash
# safe: real Qwable inference gate (loads full model, ~101.8 GB GPU memory)
python scripts/check_model_gate.py --model qwable --models-config configs/models.yaml --resources configs/resources.yaml --out-dir outputs/model_gates --max-new-tokens 1024
```

```bash
# safe: dry-run gate
python scripts/check_model_gate.py --model qwable --dry-run --models-config configs/models.yaml --resources configs/resources.yaml --out-dir outputs/model_gates
```

```bash
# safe: dry-run metadata inspection
python scripts/check_model_gate.py --model diffrwkv --dry-run --models-config configs/models.yaml --resources configs/resources.yaml --out-dir outputs/model_gates
```

## SWE-bench Lite setup

The fixed manifest is `configs/task_manifest.yaml`. Gold-patch smoke rows are generated only from actual dataset `patch` fields or an explicit local source. Synthetic gold patches are rejected.

```bash
# safe: strict official preflight check (detects blockers: swebench, docker, dataset IDs)
python scripts/check_official_swebench_env.py --manifest configs/task_manifest.yaml --models-config configs/models.yaml --resources configs/resources.yaml --out outputs/official_env_status.json
```

```bash
# prereq: datasets cache or network access to princeton-nlp/SWE-bench_Lite
python scripts/make_gold_smoke.py --manifest configs/task_manifest.yaml --out outputs/runs/gold_patch_smoke/predictions.jsonl
```

```bash
# safe: local JSONL validation
python scripts/validate_predictions.py outputs/runs/gold_patch_smoke/predictions.jsonl
```

Official harness command shape is wrapped by `repair_agent.env.harness`. In strict official mode it detects Docker/swebench blockers, constrains evaluation to prediction instance IDs, forces local image builds, and exits nonzero on official harness fallback rather than claiming an official score.

```bash
# prereq: official SWE-bench harness, Docker, dataset cache; long-running
python -m repair_agent.env.harness --predictions outputs/runs/gold_patch_smoke/predictions.jsonl --run-id official_gold_smoke --auto-workers --resources configs/resources.yaml --status-out outputs/harness_status.json --timeout-seconds 1800 --strict-official
```

## Defects4J fallback evaluation

When the Docker-based SWE-bench harness is blocked, the harness wrapper can fall back to a local Defects4J evaluator. Defects4J does not require Docker and uses checked-out Java project repositories.

Install Defects4J once (Java 11+ and Perl are required):

```bash
# safe: clone Defects4J (requires network)
git clone --depth 1 https://github.com/rjust/defects4j.git /tmp/opencode/defects4j

# safe: install Perl dependencies
curl -L https://cpanmin.us | perl - App::cpanminus
cpanm --notest --installdeps /tmp/opencode/defects4j

# long-running: download project repos and external tools
# prereq: cpanm and Defects4J Perl deps already installed
cd /tmp/opencode/defects4j && ./init.sh
```

Predictions use Defects4J instance ids (`Project_BugId`):

```jsonl
{"instance_id": "Lang_1", "model_name_or_path": "unit", "model_patch": "--- a/...\n+++ b/...\n..."}
{"instance_id": "Math_5", "model_name_or_path": "unit", "model_patch": "--- a/...\n+++ b/...\n..."}
```

Run the fallback through the same harness wrapper:

```bash
# safe: falls back to Defects4J when Docker is unavailable
python -m repair_agent.env.harness --predictions outputs/runs/d4j_predictions.jsonl --run-id d4j_fallback --max-workers 2 --strict-official --status-out outputs/harness_status_d4j.json
```

Flags:

| Flag | Purpose |
|---|---|
| `--defects4j-home PATH` | Use a specific Defects4J installation. |
| `--skip-defects4j-fallback` | Keep the original behavior and record `blocked` when Docker fails. |

The status JSON records `defects4j_harness_executed: true` and `defects4j_instances` when the fallback runs. In strict mode the wrapper exits 0 if Defects4J evaluation completes (even though the official Docker harness was not used).

## Local device policy

`configs/resources.yaml` is the source of local resource decisions. The recorded inventory in `outputs/device_inventory.json` shows 4 healthy RTX 4090 GPUs with IDs `[0, 1, 2, 3]`, 128 logical CPUs, and recommended SWE-bench workers `16`.

```bash
# safe: local probe
python scripts/probe_local_devices.py --out outputs/device_inventory.json
```

```bash
# safe: deterministic schedule preview
python scripts/run_gated_experiments.py --manifest configs/task_manifest.yaml --out outputs/runs --resources configs/resources.yaml --dry-run-schedule
```

The dry-run schedule command writes `outputs/run_schedule.json` by default.

Scheduling policy:

| Resource | Recorded policy |
|---|---|
| GPU coverage | Use healthy IDs `[0, 1, 2, 3]` with round-robin stage assignment. |
| Qwable | `device_map_auto` intent over visible GPUs. |
| DiffRWKV | Per-worker `CUDA_VISIBLE_DEVICES`, but gate is blocked before repair use. |
| Rollouts | `rollout_parallelism: 4`, one worker per healthy GPU when possible. |
| CPU | `cpu.max_workers: 32`, leaving headroom on 128 logical cores. |
| SWE-bench | Auto-sized `16` workers from CPU and RAM. |
| Fallback | Missing GPUs or saturation are recorded in artifact JSON and local execution continues when policy allows. |

## Smoke run

Use these commands to run deterministic local fixtures or strict official smoke tests. They don't produce official SWE-bench scores.

```bash
# safe: strict official 40-ID bridge smoke (2 instances, manifest-driven)
python -m repair_agent.run --config configs/baseline.yaml --manifest configs/task_manifest.yaml --instance-split main --strict-official --limit 2 --run-id strict_bridge_smoke --resources configs/resources.yaml --force
```

```bash
# safe: local fixture
python -m repair_agent.run --config configs/baseline.yaml --resources configs/resources.yaml --run-id baseline_smoke --limit 1
```

```bash
# safe: local fixture
python -m repair_agent.run --config configs/feedback.yaml --resources configs/resources.yaml --run-id feedback_smoke --limit 1
```

```bash
# safe: local fixture
python -m repair_agent.training.train --config configs/learning.yaml --resources configs/resources.yaml --episodes 1 --run-id learning_smoke --limit 1
```

## Full local run

This is the strict official pipeline. It runs all 14 stages including model gates, gold smoke, 3 smoke runs, 6 main/ablation runs (40 IDs each), and harness status. Use `--force` to re-run from scratch (archives existing run dirs).

```bash
# prereq: local dataset cache for gold smoke; may take several minutes
python scripts/run_gated_experiments.py --manifest configs/task_manifest.yaml --out outputs/runs --resources configs/resources.yaml --strict-official --force
```

The manifest records exact stage commands, task IDs, seeds, statuses, required artifacts, and model gates in `outputs/run_manifest.json`. Blocked stages (DiffRWKV gate, official harness) are recorded with their blocker reasons and qz offload status.

## Evaluation summary

**Official SWE-bench was never tested.** The Docker-based SWE-bench harness was blocked in this workspace (`unshare: operation not permitted`, `swebench` unavailable), so no official SWE-bench Lite resolved rate exists. Any previous SWE-bench score table should be treated as invalid.

The numbers still present in `outputs/summary.json` under `aggregate` come from local fixture smoke tests, not from a benchmark:
- `run_count: 14`, `total_denominator: 249`, `total_resolved: 1`
- `mean_pass_at_1 = 0.077`, `resolved_rate = 0.004`

`baseline_smoke` achieved pass@1=1.0 only on the deterministic `add_numbers` fixture. The 40-ID strict official runs produced empty patches and were never evaluated by the official harness.

The **only** benchmark-grade evaluations that actually executed are the Defects4J non-Docker fallback runs below:

| Run | Instances | Patch source | Resolved | Report |
|---|---|---:|---:|---|
| d4j_gold_smoke | Lang_1 | Buggy→fixed source diff | 1/1 | `logs/run_evaluation/d4j_gold_smoke/report.json` |
| d4j_baseline_smoke | Lang_1, Math_5 | Rule-based Java agent | 0/2 | `logs/run_evaluation/d4j_baseline_smoke_eval/report.json` |
| d4j_empty_langmath_eval | Lang_1,3-6; Math_1-5 | Empty-patch throughput test | 0/10 | `logs/run_evaluation/d4j_empty_langmath_eval/report.json` |

The gold-patch smoke validates the evaluator plumbing. The baseline and empty-patch runs show that the current rule-based agent does not yet generate valid Java repairs, but the checkout→compile→test loop is exercised and accelerated.

## Java / Defects4J repair

The agent now supports Java/Defects4J tasks via `--defects4j`:

```bash
# prereq: Defects4J installed and project repos initialized
python -m repair_agent.run --config configs/defects4j.yaml --defects4j --defects4j-home /tmp/opencode/defects4j --defects4j-manifest configs/defects4j_manifest.yaml --instance-split smoke --run-id d4j_baseline_smoke --resources configs/resources.yaml --force
```

```bash
# prereq: predictions file exists; long-running (compile+test)
python -m repair_agent.env.harness --predictions outputs/runs/d4j_baseline_smoke/predictions.jsonl --run-id d4j_baseline_smoke_eval --max-workers 4 --strict-official --defects4j-home /tmp/opencode/defects4j --status-out outputs/harness_status_d4j_baseline_smoke.json
```

Defects4J checkout, compile, and test are cached under `outputs/.d4j_cache` and reused across evaluations. `max-workers` controls parallel evaluation. The current manifest pins a small locally-testable subset; the full benchmark is queued for qz (see below).

## Defects4J fallback results

Defects4J was installed and initialized successfully at `/tmp/opencode/defects4j` (status: `/tmp/opencode/d4j_setup_status.json`). The non-Docker fallback evaluator was validated end-to-end on the classic `Lang_1` bug:

| Run | Instances | Patch source | Resolved | Report |
|---|---|---:|---:|---|
| d4j_gold_smoke | Lang_1 | Buggy→fixed source diff | 1/1 | `logs/run_evaluation/d4j_gold_smoke/report.json` |
| d4j_baseline_smoke | Lang_1, Math_5 | Rule-based Java agent | 0/2 | `logs/run_evaluation/d4j_baseline_smoke_eval/report.json` |
| d4j_empty_langmath_eval | Lang_1,3-6; Math_1-5 | Empty-patch throughput/caching test | 0/10 | `logs/run_evaluation/d4j_empty_langmath_eval/report.json` |

Status JSONs:
- `outputs/harness_status_d4j_gold_smoke.json`
- `outputs/harness_status_d4j_baseline_smoke.json`
- `outputs/harness_status_d4j_empty_langmath.json`

Key fields:

- `official_harness_executed: false` — the Docker-based SWE-bench harness was not used.
- `defects4j_harness_executed: true` — the fallback evaluator ran locally.
- `resolved: 1`, `total: 1`, `resolved_rate: 1.0` for the gold patch; the other runs produced no valid patches.
- Evaluations use a checkout cache (`outputs/.d4j_cache`) and a ThreadPool to accelerate repeated compile/test cycles.

Note: the rule-based baseline does not yet generate real Java fixes; the 0/2 and 0/10 results are honest baseline scores, not evaluator errors.

## qz 4×H200 job specs

GPU inference and CPU evaluation job specs for the full Defects4J benchmark are generated under `outputs/qz/`:

```bash
# safe: generate qz job specs (does not submit)
python scripts/make_defects4j_qz_jobs.py
```

This emits:
- `outputs/qz/defects4j_infer_job.json` + `defects4j_infer_dry_run.yaml` — agent prediction generation on 4×H200.
- `outputs/qz/defects4j_eval_job.json` + `defects4j_eval_dry_run.yaml` — CPU-bound Defects4J evaluation.
- `configs/resources.h200.yaml` — H200 resource profile.

Before submitting, resolve the `RESOLVE_BEFORE_SUBMISSION` placeholders (`workspace_id`, `logic_compute_group_id`, `spec_id`, image) and ensure Defects4J project repos are initialized on the cluster (run `./init.sh` or mount a pre-initialized `DEFECTS4J_HOME`). No job has been submitted yet.

## Report generation and checks

The report is plain Markdown, so no heavyweight renderer is needed.

```bash
# safe: verify report references
python scripts/check_report_artifacts.py report/report.md outputs/summary.json
```

```bash
# safe: parse README shell blocks without executing commands
python scripts/check_readme_commands.py README.md --dry-run-safe
```

```bash
# safe: full project tests
python -m pytest -q tests
```

## Academic integrity

The project separates agent-safe records from hidden SWE-bench `patch` and `test_patch` fields. Manual fixes and gold-patch smoke rows are not presented as agent outputs. Failed gates, blocked harness status, empty patches, weak reward signals, and fallback metrics are documented rather than removed.

## License

Project code is MIT. `lordx64/Qwable-v1` is recorded as AGPL-3.0 in the model gate artifacts.
