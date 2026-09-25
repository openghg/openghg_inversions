"""Bind durable affine flux ingredients to one coherent CO2 preparation artifact."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from openghg.util import cf_ureg  # pyright: ignore[reportPrivateImportUsage]

from openghg_inversions.array_ops import require_unique_index, same_index
from openghg_inversions.basis.affine_flux_map import AffineFluxMap
from openghg_inversions.basis.affine_flux_map_io import AffineFluxMapArtifact, load
from openghg_inversions.basis.operators import BucketBasisOperator, MultiSourceBucketBasisOperator

from .co2_preparation import Co2PreparedInputs


def prepared_inputs_content_id(path: str | Path) -> str:
    """Hash the exact saved prepared artifact, including all Zarr store files."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    digest = sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]
    for file in files:
        if root.is_dir():
            name = file.relative_to(root).as_posix().encode("utf-8")
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _same_axis(array: xr.DataArray, dim: str, expected: xr.DataArray, *, name: str) -> None:
    actual_index = require_unique_index(array, dim, name=name)
    expected_index = require_unique_index(expected, dim, name="prepared inputs")
    if not same_index(actual_index, expected_index):
        raise ValueError(f"{name} {dim!r} labels differ from prepared inputs.")


def _same_flux_units(actual: xr.DataArray, expected: xr.DataArray) -> None:
    actual_units = actual.attrs.get("units")
    expected_units = expected.attrs.get("units")
    try:
        factor = cf_ureg.Quantity(1, actual_units).to(expected_units).magnitude
    except Exception as exc:
        raise ValueError("Affine flux units are incompatible with prepared basis flux units.") from exc
    if not np.isclose(factor, 1, rtol=1e-12, atol=0):
        raise ValueError("Affine flux units have a different numeric scale from prepared basis flux units.")


@dataclass(frozen=True, slots=True, eq=False)
class BoundCo2AffineFluxMap:
    """Affine reconstruction with the prepared retained mean as its sole reference."""

    artifact: AffineFluxMapArtifact
    reference_state: xr.DataArray

    @property
    def affine_map(self) -> AffineFluxMap:
        """Expose the factorized native mean, signed flux, and prolongation."""
        return self.artifact.affine_map

    def state_to_native(self, state: xr.DataArray) -> xr.DataArray:
        """Evaluate the retained-state-conditional native scaling mean."""
        return self.affine_map.state_to_native(state, reference_state=self.reference_state)

    def state_to_flux(self, state: xr.DataArray) -> xr.DataArray:
        """Evaluate the retained-state-conditional signed flux mean."""
        return self.affine_map.state_to_flux(state, reference_state=self.reference_state)


def _bind_affine_flux_map(
    artifact: AffineFluxMapArtifact,
    prepared: Co2PreparedInputs,
    *,
    prepared_inputs_id: str,
) -> BoundCo2AffineFluxMap:
    """Validate a pair using an identity established by its artifact caller.

    In-memory callers must supply an identity for the same prepared object.
    Public saved-artifact binding uses :func:`load_and_bind_affine_flux_map`
    to derive that identity from the actual prepared artifact bytes.
    """
    if artifact.prepared_inputs_id != prepared_inputs_id:
        raise ValueError("Affine flux prepared-input content identity does not match.")
    projection = dict(artifact.projection_provenance)
    strategy = prepared.provenance.get("projection_strategy")
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("Prepared inputs lack projection_strategy provenance.")
    if projection != {"projection_strategy": strategy}:
        raise ValueError("Affine flux projection provenance differs from prepared inputs.")

    affine_map = artifact.affine_map
    reference = prepared.inv_inputs["alpha_prior_mean"]
    if reference.dims != (affine_map.state_dim,):
        raise ValueError("Prepared alpha_prior_mean uses a different retained-state dimension.")
    retained = (
        affine_map.prolongation
        if isinstance(affine_map.prolongation, xr.DataArray)
        else affine_map.prolongation.basis_matrix
    )
    _same_axis(retained, affine_map.state_dim, reference, name="Affine prolongation")
    if reference.attrs.get("units") != "1":
        raise ValueError("Prepared alpha_prior_mean must be dimensionless in units '1'.")

    basis = prepared.basis_functions
    if not isinstance(basis.operator, (BucketBasisOperator, MultiSourceBucketBasisOperator)):
        raise ValueError("Prepared basis requires a bucket operator for CO2 affine binding.")
    basis_flux = basis.flux
    if isinstance(basis.operator, MultiSourceBucketBasisOperator):
        native_source_dim = affine_map.native_dims[0]
        if basis.operator.source_dim in basis_flux.dims:
            basis_flux = basis_flux.rename({basis.operator.source_dim: native_source_dim})
    if affine_map.flux.dims != basis_flux.dims:
        raise ValueError("Affine flux dimensions differ from prepared basis flux dimensions.")
    for dim in basis_flux.dims:
        _same_axis(affine_map.flux, dim, basis_flux, name="Affine flux")
    for dim in basis.operator.meta.grid_dims:
        _same_axis(affine_map.native_mean, dim, basis.operator.basis_matrix, name="Affine native_mean")
    if isinstance(basis.operator, MultiSourceBucketBasisOperator):
        source_dim = affine_map.native_dims[0]
        if affine_map.native_dims[1:] != basis.operator.meta.grid_dims:
            raise ValueError("Affine native dimensions differ from prepared multisource grid.")
        if tuple(map(str, affine_map.native_mean.indexes[source_dim])) != basis.operator.source_labels:
            raise ValueError("Affine native source labels differ from prepared basis.")
    elif affine_map.native_dims != basis.operator.meta.grid_dims:
        raise ValueError("Affine native dimensions differ from prepared basis grid.")
    _same_flux_units(affine_map.flux, basis_flux)
    if isinstance(affine_map.prolongation, MultiSourceBucketBasisOperator):
        if not isinstance(basis.operator, MultiSourceBucketBasisOperator):
            raise ValueError("Affine bucket source representation differs from prepared basis.")
        if affine_map.prolongation.source_labels != basis.operator.source_labels:
            raise ValueError("Affine bucket source labels differ from prepared basis.")
    elif isinstance(affine_map.prolongation, BucketBasisOperator):
        if not isinstance(basis.operator, BucketBasisOperator):
            raise ValueError("Affine bucket representation differs from prepared basis.")
    return BoundCo2AffineFluxMap(artifact, reference)


def load_and_bind_affine_flux_map(
    reconstruction_path: str | Path,
    prepared_inputs_path: str | Path,
) -> BoundCo2AffineFluxMap:
    """Load both saved artifacts and verify the prepared artifact's content ID."""
    prepared_inputs_id = prepared_inputs_content_id(prepared_inputs_path)
    prepared = Co2PreparedInputs.load(prepared_inputs_path)
    artifact = load(reconstruction_path)
    bound = _bind_affine_flux_map(artifact, prepared, prepared_inputs_id=prepared_inputs_id)
    operator = artifact.affine_map.prolongation
    prepared_operator = prepared.basis_functions.operator
    if isinstance(operator, MultiSourceBucketBasisOperator):
        for source in operator.source_labels:
            try:
                xr.testing.assert_equal(operator.basis_flat[source], prepared_operator.basis_flat[source])
            except AssertionError as exc:
                raise ValueError(
                    f"Affine bucket assignments differ from prepared source {source!r}."
                ) from exc
    elif isinstance(operator, BucketBasisOperator):
        try:
            xr.testing.assert_equal(operator.basis_flat, prepared_operator.basis_flat)
        except AssertionError as exc:
            raise ValueError("Affine bucket assignments differ from prepared basis.") from exc
    return bound


def _artifact(
    prepared: Co2PreparedInputs,
    affine_map: AffineFluxMap,
    *,
    prepared_inputs_id: str,
    projection_provenance: Mapping[str, Any] | None,
    reconstruction_provenance: Mapping[str, Any] | None,
    source_provenance: Mapping[str, Any] | None,
) -> AffineFluxMapArtifact:
    projection = dict(projection_provenance or {})
    projection.setdefault("projection_strategy", prepared.provenance.get("projection_strategy"))
    artifact = AffineFluxMapArtifact(
        affine_map=affine_map,
        prepared_inputs_id=prepared_inputs_id,
        projection_provenance=projection,
        reconstruction_provenance=dict(reconstruction_provenance or {}),
        source_provenance=dict(source_provenance or {}),
    )
    _bind_affine_flux_map(artifact, prepared, prepared_inputs_id=prepared_inputs_id)
    return artifact


def produce_bucket_affine_flux_map(
    prepared: Co2PreparedInputs,
    native_mean: xr.DataArray,
    *,
    prepared_inputs_id: str,
    projection_provenance: Mapping[str, Any] | None = None,
    reconstruction_provenance: Mapping[str, Any] | None = None,
    source_provenance: Mapping[str, Any] | None = None,
) -> AffineFluxMapArtifact:
    """Reuse the prepared bucket operator and signed flux without flattening FU*."""
    basis = prepared.basis_functions
    if not isinstance(basis.operator, (BucketBasisOperator, MultiSourceBucketBasisOperator)):
        raise ValueError("Bucket affine production requires a bucket BasisFunctions operator.")
    flux = basis.flux
    if isinstance(basis.operator, MultiSourceBucketBasisOperator):
        native_source_dim = native_mean.dims[0]
        if basis.operator.source_dim in flux.dims:
            flux = flux.rename({basis.operator.source_dim: native_source_dim})
    affine_map = AffineFluxMap(native_mean, flux, basis.operator, basis.operator.meta.state_dim)
    return _artifact(
        prepared,
        affine_map,
        prepared_inputs_id=prepared_inputs_id,
        projection_provenance=projection_provenance,
        reconstruction_provenance=reconstruction_provenance,
        source_provenance=source_provenance,
    )


def import_explicit_affine_flux_map(
    prepared: Co2PreparedInputs,
    native_mean: xr.DataArray,
    flux: xr.DataArray,
    prolongation: xr.DataArray,
    *,
    prepared_inputs_id: str,
    projection_provenance: Mapping[str, Any] | None = None,
    reconstruction_provenance: Mapping[str, Any] | None = None,
    source_provenance: Mapping[str, Any] | None = None,
    reference_state: xr.DataArray | None = None,
) -> AffineFluxMapArtifact:
    """Import exact supplied-restriction U* without constructing its projection.

    An externally supplied reference state is checked against prepared
    ``alpha_prior_mean`` and then discarded.
    """
    authoritative = prepared.inv_inputs["alpha_prior_mean"]
    if reference_state is not None:
        try:
            xr.testing.assert_identical(reference_state.rename(authoritative.name), authoritative)
        except AssertionError as exc:
            raise ValueError("Imported reference_state differs from prepared alpha_prior_mean.") from exc
    operator = prepared.basis_functions.operator
    if isinstance(operator, MultiSourceBucketBasisOperator) and operator.source_dim in flux.dims:
        flux = flux.rename({operator.source_dim: native_mean.dims[0]})
    state_dim = authoritative.dims[0]
    if not isinstance(state_dim, str):
        raise ValueError("Prepared retained-state dimension must be a string.")
    affine_map = AffineFluxMap(native_mean, flux, prolongation, state_dim)
    return _artifact(
        prepared,
        affine_map,
        prepared_inputs_id=prepared_inputs_id,
        projection_provenance=projection_provenance,
        reconstruction_provenance=reconstruction_provenance,
        source_provenance=source_provenance,
    )


__all__ = [
    "BoundCo2AffineFluxMap",
    "import_explicit_affine_flux_map",
    "load_and_bind_affine_flux_map",
    "prepared_inputs_content_id",
    "produce_bucket_affine_flux_map",
]
