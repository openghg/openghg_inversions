"""Load saved stage records for trusted, matched run files.

Family workflows own configuration identities and choose their replay policy.
This boundary checks record formats and required output-contract contents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from openghg_inversions.inference.sampling import RhimeSampler


def _load_stage_manifest(path: str | Path, *, stage: str) -> tuple[Path, dict[str, Any]]:
    """Load and validate one OGI stage manifest envelope."""
    manifest_path = Path(path).resolve()
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Stage manifest {manifest_path} must contain one JSON object.")
    supported_versions = (1, 2) if stage == "sample" else (1,)
    if type(loaded.get("schema_version")) is not int or loaded["schema_version"] not in supported_versions:
        raise ValueError(f"Stage manifest {manifest_path} has an unsupported schema_version.")
    if stage == "sample" and loaded["schema_version"] == 1:
        entries = loaded.get("artifacts")
        if isinstance(entries, Mapping) and "output_binding" in entries:
            raise ValueError("A sample manifest with an output binding requires schema_version=2.")
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


def _sampler_metadata_from_sample_manifest(manifest: Mapping[str, Any]) -> dict[str, int | str]:
    """Describe recorded settings; unavailable descriptions do not prevent replay.

    Use the sampler's existing normalization and output description, retaining
    only supplied fields. Unknown runtime options are not needed for products.
    If known settings cannot be parsed, leave their description unavailable;
    the original record still retains the supplied provenance.
    """
    effective = manifest.get("effective_configuration")
    sampler = effective.get("sampler") if isinstance(effective, Mapping) else None
    if not isinstance(sampler, Mapping):
        return {}
    options = {
        name: sampler[name]
        for name in (*RhimeSampler.INTEGER_OPTION_NAMES, "nuts_sampler")
        if name in sampler
    }
    try:
        description = RhimeSampler(**options).describe()
    except (TypeError, ValueError, OverflowError):
        return {}
    return {name: value for name, value in description.items() if name in options}
