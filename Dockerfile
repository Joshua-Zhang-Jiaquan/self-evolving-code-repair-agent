# Self-Evolving Code Repair Agent Runtime
# Build: docker build -t repair_agent .
# Run:   docker run --gpus all -v $(pwd):/workspace -it repair_agent

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# Install core Python dependencies
COPY pyproject.toml README.md ./
COPY repair_agent/ ./repair_agent/

# Install the package in editable mode with all extras
RUN pip install -e ".[all]"

# For SWE-bench: Docker-in-Docker access (bind mount /var/run/docker.sock at runtime)
# Run with: docker run --gpus all -v /var/run/docker.sock:/var/run/docker.sock -v $(pwd):/workspace repair_agent

ENTRYPOINT ["python", "-m", "repair_agent"]
