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
from openghg_inversions.postprocessing.contracts import OutputContract


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


def _load_output_binding(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> OutputContract:
    """Load the referenced graph-free output contract."""
    artifacts = manifest.get("artifacts")
    relative_path = artifacts.get("output_binding") if isinstance(artifacts, Mapping) else None
    if not isinstance(relative_path, str) or not relative_path or Path(relative_path).is_absolute():
        raise ValueError("Sample manifest requires a relative output_binding artifact path.")
    binding_path = (manifest_path.parent / relative_path).resolve()
    if not binding_path.is_relative_to(manifest_path.parent):
        raise ValueError("Output binding must be beneath its sample manifest directory.")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    fields = {"schema", "schema_version", "output_contract"}
    if not isinstance(binding, dict) or set(binding) != fields:
        raise ValueError("Output binding has missing or unexpected fields.")
    if (
        binding["schema"] != "openghg_inversions.output_binding"
        or type(binding["schema_version"]) is not int
        or binding["schema_version"] != 1
    ):
        raise ValueError("Output binding has an unsupported schema or schema_version.")
    return OutputContract.from_dict(binding["output_contract"])


def _load_sample_manifest(
    path: str | Path,
    *,
    configuration_identity: str | None = None,
) -> dict[str, Any]:
    """Load a sample record and check its configured replay restriction."""
    _, manifest = _load_stage_manifest(path, stage="sample")
    if configuration_identity is not None:
        expected = configuration_identity
        if manifest.get("configuration_identity") != expected:
            raise ValueError(
                "Posterior does not match the effective RHIME configuration: "
                f"manifest has {manifest.get('configuration_identity')!r}, current configuration has {expected!r}."
            )
    return manifest


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
