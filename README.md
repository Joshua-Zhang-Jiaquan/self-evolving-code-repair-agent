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

The official SWE-bench harness is blocked in strict official mode. `outputs/harness_status.json` records `official_harness_executed: false`, `status: blocked`, `execution_backend: qz_pending_approval`, and `max_workers: 16`. This project does not claim an official SWE-bench Lite resolved rate. Local fixture and fallback metrics in `outputs/summary.json` are separate. Official evaluation is pending qz cluster approval.

Model gates are also explicit:

| Model | Gate status | Meaning |
|---|---|---|
| `lordx64/Qwable-v1` | `pass` | Real inference gate passed (Qwen3.5-MoE 71.9GB, 4x RTX 4090, GPU peak ~101.8GB). Dry-run parser and resource gate also pass. License note: AGPL-3.0. |
| DiffRWKV local checkpoint | `blocked` | The checkpoint is a DDPM/RWKV trajectory model, not an instruction-following code repair model. It is not counted as a repair baseline. |


## Conda no-Docker evaluation evidence

In addition to the blocked Docker official-harness status above, this repository includes a local conda evaluation path in `scripts/run_conda_swebench_eval.py`. It uses SWE-bench 4.1.0 test specs and grading without Docker. The consolidated artifact is `outputs/conda_eval_status.json`: Docker was not used, the gold-patch smoke validation resolves 2/2 instances, and all six submitted agent/ablation runs resolve 0/40 because their patches are empty. These conda results are reproducibility evidence only; they do not change the Docker official harness disclosure in `outputs/harness_status.json`.

```bash
# prereq: swebench/datasets available; long-running conda evaluation, no Docker
python scripts/run_conda_swebench_eval.py --predictions outputs/runs/gold_patch_smoke/predictions.jsonl --run-id gold_conda --out outputs/conda_eval_gold.json --timeout 3000
```

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

Official harness command shape is wrapped by `repair_agent.env.harness`. In strict official mode it detects Docker/swebench blockers and records blocked status rather than claiming an official score.

```bash
# prereq: official SWE-bench harness, Docker, dataset cache; long-running
python -m repair_agent.env.harness --predictions outputs/runs/gold_patch_smoke/predictions.jsonl --run-id official_gold_smoke --auto-workers --resources configs/resources.yaml --status-out outputs/harness_status.json --timeout-seconds 1800 --strict-official
```

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

`outputs/summary.json` currently reports 14 local/fallback runs and 249 submitted local rows. Aggregate local fixture/fallback values are `mean_pass_at_1 = 0.077`, `resolved_rate = 0.004`, and `total_resolved = 1`. These numbers are not official SWE-bench metrics. Official harness is blocked (Docker/swebench unavailable), `official_resolved_rate = null`.

Main rows to inspect (strict official 40-ID results, all empty patches):

| Run | Type | Local pass@1 | Local resolved | Empty patch rate | Run dir |
|---|---:|---:|---:|---:|---|
| Baseline main | baseline | 0.000 | 0/40 | 1.000 | `outputs/runs/baseline_main` |
| Feedback main | feedback | 0.000 | 0/40 | 1.000 | `outputs/runs/feedback_main` |
| Learning main | learning | 0.000 | 0/40 | 1.000 | `outputs/runs/learning_main` |
| A1 no process reward | learning | 0.000 | 0/40 | 1.000 | `outputs/runs/ablation_no_process_reward` |
| A2 no feedback features | feedback | 0.000 | 0/40 | 1.000 | `outputs/runs/ablation_no_feedback_features` |
| A3 reduced test budget | learning | 0.000 | 0/40 | 1.000 | `outputs/runs/ablation_reduced_test_budget` |
| Baseline smoke | baseline | 1.000 | 1/1 | 0.000 | `outputs/runs/baseline_smoke` |

Gold-patch smoke rows use actual SWE-bench Lite dataset patches only for harness smoke generation. They are not agent output.

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
