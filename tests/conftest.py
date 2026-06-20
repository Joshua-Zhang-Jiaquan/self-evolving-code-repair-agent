"""Shared pytest fixtures for repair_agent test suite."""
from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory (where pyproject.toml lives)."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def temp_output_dir() -> Iterator[Path]:
    """Create and return a temporary outputs directory; cleaned up after test."""
    with tempfile.TemporaryDirectory(prefix="repair_agent_test_") as tmp:
        yield Path(tmp)


@pytest.fixture
def temp_json_file(temp_output_dir: Path) -> Path:
    """Create a path for a temporary JSON output file."""
    return temp_output_dir / "test_output.json"
