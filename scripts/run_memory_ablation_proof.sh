#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-artifacts/proof_experiments/two_dimensional_memory}"
python3 -m code_repair_agent.memory_ablation_proof --out-dir "${OUT_DIR}"
