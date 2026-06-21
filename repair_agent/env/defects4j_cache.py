from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from repair_agent.env.defects4j_harness import Defects4JInstance


def cache_path(workdir_root: Path, instance: Defects4JInstance, version: str) -> Path:
    """Return a stable cache directory under *workdir_root* for *instance* and *version*."""
    return workdir_root / ".d4j_cache" / f"{instance.project}_{instance.bug_id}{version}"


def ensure_cached(
    instance: Defects4JInstance,
    version: str,
    d4j_home: str | None = None,
    *,
    workdir_root: Path | None = None,
) -> Path:
    """Checkout a Defects4J bug *version* once and cache the result.

    If ``.defects4j.config`` already exists in the cache directory the checkout
    is skipped.  *d4j_home* is accepted for forward compatibility but the
    checkout delegates to :func:`repair_agent.env.defects4j_harness.run_defects4j_command`
    which auto-detects the installation.
    """
    _ = d4j_home  # reserved; run_defects4j_command auto-detects the home.
    if workdir_root is None:
        workdir_root = Path("outputs") / ".d4j_cache"
    cached = cache_path(workdir_root, instance, version)

    if (cached / ".defects4j.config").exists():
        return cached

    cached.parent.mkdir(parents=True, exist_ok=True)
    if cached.exists():
        shutil.rmtree(cached)

    # Lazy import to avoid circular dependency at module level.
    from repair_agent.env.defects4j_harness import run_defects4j_command  # noqa: PLC0415

    result = run_defects4j_command(
        [
            "defects4j",
            "checkout",
            "-p",
            instance.project,
            "-v",
            f"{instance.bug_id}{version}",
            "-w",
            str(cached),
        ],
        timeout=300,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-500:]
        raise RuntimeError(
            f"defects4j checkout (cache) failed for {instance.instance_id}: {detail}"
        )
    cached.mkdir(parents=True, exist_ok=True)
    (cached / ".defects4j.config").touch()
    return cached


def materialize_workdir(cache_path: Path, workdir: Path) -> Path:
    """Copy *cache_path* into *workdir*.

    The function tries (in order) ``cp -al`` (hardlink clone),
    ``git clone --local --shared``, and finally :func:`shutil.copytree`.
    """
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.parent.mkdir(parents=True, exist_ok=True)

    # 1) hardlink clone ------------------------------
    try:
        subprocess.run(
            ["cp", "-al", str(cache_path), str(workdir)],
            check=True,
            capture_output=True,
            timeout=60,
        )
        return workdir
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    # 2) git clone -----------------------------------
    if (cache_path / ".git").is_dir():
        try:
            subprocess.run(
                ["git", "clone", "--local", "--shared", "-q", str(cache_path), str(workdir)],
                check=True,
                capture_output=True,
                timeout=120,
            )
            return workdir
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            pass

    # 3) fallback ------------------------------------
    if workdir.exists():
        shutil.rmtree(workdir)
    shutil.copytree(cache_path, workdir, symlinks=True)
    return workdir


def reset_workdir(workdir: Path) -> bool:
    """Restore *workdir* to the last committed state via ``git checkout`` and ``git clean``."""
    try:
        subprocess.run(
            ["git", "checkout", "--", "."],
            cwd=workdir,
            check=True,
            capture_output=True,
            timeout=60,
        )
        subprocess.run(
            ["git", "clean", "-fdq"],
            cwd=workdir,
            check=True,
            capture_output=True,
            timeout=60,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False
    return True


def prewarm(
    instances: Iterable[Defects4JInstance],
    d4j_home: str | None = None,
    *,
    workdir_root: Path | None = None,
) -> dict[str, Path]:
    """Serially ensure a cached checkout for every unique (project, bug_id, ``"b"``) tuple.

    Returns a mapping ``{project}_{bug_id}b -> cache_path``.
    """
    seen: dict[tuple[str, int, str], Path] = {}
    for instance in instances:
        key = (instance.project, instance.bug_id, "b")
        if key not in seen:
            seen[key] = ensure_cached(
                instance, "b", d4j_home, workdir_root=workdir_root
            )
    return {f"{p}_{b}{v}": path for (p, b, v), path in seen.items()}
