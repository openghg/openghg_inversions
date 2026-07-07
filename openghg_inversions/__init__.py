"""OpenGHG inversions package."""

from __future__ import annotations

import getpass
import os
import tempfile
from pathlib import Path


def _set_default_hpc_cache_dir() -> None:
    """Use a per-task local cache on batch systems unless the user configured one."""
    if os.environ.get("XDG_CACHE_HOME"):
        return

    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID")
    if job_id is None:
        return

    task_id = os.environ.get("SLURM_PROCID") or os.environ.get("SLURM_ARRAY_TASK_ID") or "0"
    cache_dir = Path(tempfile.gettempdir()) / f"openghg-inversions-cache-{getpass.getuser()}-{job_id}-{task_id}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)


_set_default_hpc_cache_dir()
