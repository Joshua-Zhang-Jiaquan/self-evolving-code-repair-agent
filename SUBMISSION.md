# Submission Checklist

Project: Self-Evolving Code Repair Agent  
Dataset: Defects4J  
Main model backend: DeepSeek `deepseek-v4-pro`  
Best complete run: `artifacts/runs/d4j30-self-evolved-real-v16`

## Required Files

| Requirement | File or Directory | Status |
| --- | --- | --- |
| README with install, run, reproduction instructions | `README.md` | Included |
| Agent implementation | `code_repair_agent/agent.py`, `code_repair_agent/real_benchmark.py`, `code_repair_agent/deepseek_repair.py`, `code_repair_agent/llm.py` | Included |
| Environment, task loading, test running, state management | `code_repair_agent/environment.py`, `code_repair_agent/defects4j.py`, `code_repair_agent/d4j_benchmark.py` | Included |
| Training/self-evolution module | `code_repair_agent/evolution.py`, `code_repair_agent/d4j_memory.py` | Included |
| Evaluation scripts and metrics | `code_repair_agent/evaluate.py`, `code_repair_agent/real_benchmark.py`, `code_repair_agent/d4j_test_sweep.py` | Included |
| Configs | `configs/` | Included |
| Docker environment | `Dockerfile.defects4j`, `scripts/run_defects4j_benchmark.sh` | Included |
| Unit tests | `tests/` | Included |
| Technical report | `reports/final_submission_report.md` | Included |
| Key trajectories and patches | `artifacts/runs/d4j30-self-evolved-real-v16/` | Included |
| Artifact index | `artifacts/README.md` | Included |

## Best Verified Result

The best complete, comparable 30-case Defects4J run is:

```text
artifacts/runs/d4j30-self-evolved-real-v16
```

Summary:

| Metric | Value |
| --- | ---: |
| Cases | 30 |
| Solved | 22 |
| Pass@1 | 17/30 |
| Pass@3 | 21/30 |
| Visible trigger-test pass | 23/30 |
| Regression pass | 22/30 |
| Compile success | 30/30 |
| DeepSeek calls | 121 |
| Prompt tokens | 1,776,537 |
| Completion tokens | 753,279 |

Failed cases:

```text
Lang-3, Lang-10, Math-1, Math-2, Math-3, Math-4, Math-6, Math-7
```

## How to Reproduce

Local unit tests:

```bash
python3 -m pip install -e .
PYTHONPYCACHEPREFIX=/tmp/rl_pycache python3 -m pytest -q
```

Full Dockerized benchmark:

```bash
export DEEPSEEK_API_KEY="<set-in-shell-only>"
export DEEPSEEK_MODEL=deepseek-v4-pro
RUN_ID=d4j30-self-evolved-repro bash scripts/run_defects4j_benchmark.sh
```

If Docker Hub times out but `code-repair-defects4j:latest` already exists locally:

```bash
SKIP_DOCKER_BUILD=1 RUN_ID=d4j30-self-evolved-repro bash scripts/run_defects4j_benchmark.sh
```

Create a submission archive:

```bash
bash scripts/make_submission_bundle.sh
```

Run the fast four-way memory proof:

```bash
bash scripts/run_memory_ablation_proof.sh
```

Run the real D4J four-way memory proof when DeepSeek is available:

```bash
export DEEPSEEK_API_KEY="<set-in-shell-only>"
export DEEPSEEK_MODEL=deepseek-v4-pro
SKIP_DOCKER_BUILD=1 bash scripts/run_d4j_memory_ablation_experiments.sh
```

The bundle intentionally excludes large Defects4J work directories and Python caches, but includes source, configs, reports, tests, scripts, and selected benchmark artifacts.

## Claims and Non-Claims

Claims:

- The repository implements a runnable multi-round code repair Agent.
- The Agent records tool calls, prompts, patches, tests, metrics, and memory snapshots.
- The self-evolving system uses persistent two-dimensional external memory rather than only one-off prompt reflection:
  `what-to-check memory` learns file/snippet/test/regression-check choices, while `how-to-repair memory` learns patch style, repair skill, failure reflection, success strategy, and duplicate-strategy rejection.
- A real Dockerized Defects4J run on 30 active bugs has been recorded and solved 22/30.

Non-claims:

- The current artifacts do not prove 30/30 full benchmark pass.
- Focused v17-v31 runs are mechanism-improvement evidence, not a replacement for full 30-case metrics.
- No pasted API key is included in submission files.
