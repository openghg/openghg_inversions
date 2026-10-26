"""Read descriptive sample envelopes for shared posterior diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_stage_manifest(path: str | Path, *, stage: str) -> tuple[Path, dict[str, Any]]:
    """Load and validate one OGI stage manifest envelope."""
    manifest_path = Path(path).resolve()
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Stage manifest {manifest_path} must contain one JSON object.")
    version = 3 if loaded.get("schema") in {
        "openghg_inversions.standard.sample", "openghg_inversions.multisector.sample",
    } else 1
    if type(loaded.get("schema_version")) is not int or loaded["schema_version"] != version:
        raise ValueError(
            f"Stage manifest {manifest_path} has an unsupported schema_version; expected {version}. "
            "Use the software version that wrote it or prepare and sample again."
        )
    expected = {
        "producer": "openghg_inversions",
        "stage": stage,
    }
    mismatched = {
        name: (loaded.get(name), value) for name, value in expected.items() if loaded.get(name) != value
    }
    if mismatched:
        raise ValueError(f"Stage manifest {manifest_path} has an invalid envelope: {mismatched!r}.")
    return manifest_path, loaded
