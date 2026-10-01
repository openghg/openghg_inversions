"""Authenticate saved stage manifests, numerical artifacts and output bindings.

Family workflows own configuration identities and choose their replay policy.
This boundary verifies the saved envelope and content digests before use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from openghg_inversions.postprocessing.contracts import OutputContract

from ._stage_artifacts import file_identity as _file_identity
from .sampling import RhimeSampler


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
        for field in ("artifacts", "artifact_identities"):
            entries = loaded.get(field)
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


def _verify_manifest_artifact(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    artifact_name: str,
    artifact_path: str | Path,
) -> str:
    """Verify a supplied artifact against its stage-manifest content digest."""
    identities = manifest.get("artifact_identities")
    recorded = identities.get(artifact_name) if isinstance(identities, Mapping) else None
    if not isinstance(recorded, str):
        raise ValueError(
            f"Stage manifest {manifest_path} does not contain a content identity for {artifact_name!r}."
        )
    actual = _file_identity(Path(artifact_path).resolve())
    if actual != recorded:
        raise ValueError(
            f"{artifact_name.replace('_', '-').capitalize()} content does not match {manifest_path}: "
            f"manifest has {recorded!r}, supplied artifact has {actual!r}."
        )
    return actual


def _load_output_binding(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> OutputContract:
    """Verify the graph-free output contract bound to these exact saved arrays."""
    artifacts = manifest.get("artifacts")
    relative_path = artifacts.get("output_binding") if isinstance(artifacts, Mapping) else None
    if not isinstance(relative_path, str) or not relative_path or Path(relative_path).is_absolute():
        raise ValueError("Sample manifest requires a relative output_binding artifact path.")
    binding_path = (manifest_path.parent / relative_path).resolve()
    if not binding_path.is_relative_to(manifest_path.parent):
        raise ValueError("Output binding must be beneath its sample manifest directory.")
    _verify_manifest_artifact(
        manifest,
        manifest_path=manifest_path,
        artifact_name="output_binding",
        artifact_path=binding_path,
    )
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    fields = {"schema", "schema_version", "artifact_identities", "output_contract"}
    if not isinstance(binding, dict) or set(binding) != fields:
        raise ValueError("Output binding has missing or unexpected fields.")
    if (
        binding["schema"] != "openghg_inversions.output_binding"
        or type(binding["schema_version"]) is not int
        or binding["schema_version"] != 1
    ):
        raise ValueError("Output binding has an unsupported schema or schema_version.")
    identities = manifest["artifact_identities"]
    expected = {name: identities[name] for name in ("prepared_inputs", "posterior")}
    if binding["artifact_identities"] != expected:
        raise ValueError("Output binding does not match the prepared-input and posterior identities.")
    return OutputContract.from_dict(binding["output_contract"])


def _verify_sample_manifest(
    path: str | Path,
    *,
    posterior: str | Path,
    configuration_identity: str | None = None,
    prepared_inputs: str | Path | None = None,
) -> dict[str, Any]:
    """Authenticate a posterior handoff and its optional scientific context."""
    manifest_path, manifest = _load_stage_manifest(path, stage="sample")
    _verify_manifest_artifact(
        manifest,
        manifest_path=manifest_path,
        artifact_name="posterior",
        artifact_path=posterior,
    )
    if prepared_inputs is not None:
        _verify_manifest_artifact(
            manifest,
            manifest_path=manifest_path,
            artifact_name="prepared_inputs",
            artifact_path=prepared_inputs,
        )
    if configuration_identity is not None:
        expected = configuration_identity
        if manifest.get("configuration_identity") != expected:
            raise ValueError(
                "Posterior does not match the effective RHIME configuration: "
                f"manifest has {manifest.get('configuration_identity')!r}, current configuration has {expected!r}."
            )
    return manifest


def _sampler_from_sample_manifest(manifest: Mapping[str, Any], *, path: str | Path) -> RhimeSampler:
    """Restore the sampler configuration recorded by the sampling stage."""
    effective = manifest.get("effective_configuration")
    sampler = effective.get("sampler") if isinstance(effective, Mapping) else None
    if not isinstance(sampler, Mapping):
        raise ValueError(f"Sample manifest {Path(path).resolve()} has no effective sampler configuration.")
    try:
        return RhimeSampler(**dict(sampler))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Sample manifest {Path(path).resolve()} has an invalid effective sampler configuration: {exc}"
        ) from exc
