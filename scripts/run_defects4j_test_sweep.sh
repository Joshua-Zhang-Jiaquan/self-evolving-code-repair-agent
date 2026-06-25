#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-code-repair-defects4j:latest}"
RUN_ID="${RUN_ID:-d4j-test-sweep-$(date -u +%Y%m%dT%H%M%SZ)}"
JOBS="${JOBS:-4}"
DEFECTS4J_TIMEOUT="${DEFECTS4J_TIMEOUT:-3600}"
SKIP_DOCKER_BUILD="${SKIP_DOCKER_BUILD:-0}"

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not running. Start Docker Desktop, then rerun this script." >&2
  exit 2
fi

if [[ "${SKIP_DOCKER_BUILD}" == "1" ]]; then
  if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    echo "SKIP_DOCKER_BUILD=1 but image ${IMAGE_NAME} does not exist." >&2
    exit 2
  fi
else
  if ! docker build -f Dockerfile.defects4j -t "${IMAGE_NAME}" .; then
    if docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
      echo "Docker build failed; reusing existing local image ${IMAGE_NAME}." >&2
    else
      exit 1
    fi
  fi
fi
docker run --rm \
  -e DEFECTS4J_TIMEOUT="${DEFECTS4J_TIMEOUT}" \
  -v "$PWD":/workspace \
  -w /workspace \
  "${IMAGE_NAME}" \
  python3 -m code_repair_agent.d4j_test_sweep \
    --config configs/defects4j_30.json \
    --run-id "${RUN_ID}" \
    --out-dir artifacts/d4j_test_sweeps \
    --jobs "${JOBS}" \
    --timeout "${DEFECTS4J_TIMEOUT}" \
    --scopes trigger,relevant,all
