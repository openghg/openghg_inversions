"""Versioned persistence for separate affine native-flux ingredients."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Set
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import xarray as xr

from openghg_inversions.array_ops import to_dense
from openghg_inversions.serialization import (
    MULTIINDEX_DIMS_ATTR,
    encode_multiindexes_for_storage,
    open_datatree_loaded,
    restore_declared_multiindexes,
    save_datatree,
)

from .affine_flux_map import AffineFluxMap, RETAINED_STATE_CONDITIONAL
from .operators import BasisOperator


SCHEMA = "openghg_inversions.affine_flux_map"
SCHEMA_VERSION = 1
AFFINE_CONVENTION = "native_mean_plus_prolongation_times_state_minus_reference"
_ROOT_ATTRS = {
    "schema",
    "schema_version",
    "affine_convention",
    "representation",
    "state_dim",
    "native_dims",
    "uncertainty_scope",
    "prepared_inputs_id",
    "projection_provenance",
    "reconstruction_provenance",
    "source_provenance",
}
_STATE_METADATA_VARS = {"basis_group", "basis_partition", "region_in_partition"}
_PROHIBITED_NAMES = {
    "fp_x_flux",
    "pi",
    "projection_matrix",
    "native_b",
    "native_covariance",
    "native_by_native_covariance",
    "fu",
    "flux_times_prolongation",
    "state_to_flux",
    "country_map",
    "country_by_state",
    "derived_quantity_map",
    "quantity_residual_covariance",
    "residual_block",
    "reporting_sector_mapping",
    "source_to_sector",
    "sector_mapping",
}


def _json_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Freeze a JSON-safe provenance object without coercing unsupported values."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys.")
    try:
        encoded = json.dumps(dict(value), sort_keys=True, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON-serializable.") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must be a JSON object.")

    def check_keys(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key.lower() in _PROHIBITED_NAMES:
                    raise ValueError(f"Prohibited {name} element {key!r}.")
                check_keys(child)
        elif isinstance(item, list):
            for child in item:
                check_keys(child)

    check_keys(decoded)
    return MappingProxyType(decoded)


@dataclass(frozen=True, slots=True, eq=False)
class AffineFluxMapArtifact:
    """A reconstruction map and the metadata needed to bind prepared inputs."""

    affine_map: AffineFluxMap
    prepared_inputs_id: str
    projection_provenance: Mapping[str, Any]
    reconstruction_provenance: Mapping[str, Any]
    source_provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.affine_map, AffineFluxMap):
            raise TypeError("affine_map must be an AffineFluxMap.")
        if not isinstance(self.prepared_inputs_id, str) or not self.prepared_inputs_id.strip():
            raise ValueError("prepared_inputs_id must be a non-empty string.")
        for name in ("projection_provenance", "reconstruction_provenance", "source_provenance"):
            object.__setattr__(self, name, _json_mapping(getattr(self, name), name))


def _array_node(array: xr.DataArray, name: str) -> xr.DataTree:
    # Ancillary input attrs are not reconstruction ingredients. Keep units,
    # while provenance belongs in the explicit versioned root metadata.
    units = array.attrs["units"]
    array = to_dense(array) if name == "prolongation" else array
    array = array.copy(deep=False)
    array.attrs = {"units": units}
    return xr.DataTree(encode_multiindexes_for_storage(array.rename(name).to_dataset()))


def _operator_dataset_for_storage(dataset: xr.Dataset) -> xr.Dataset:
    """Keep only the bucket representation's declared variable attributes."""
    result = dataset.copy(deep=False)
    for name in result.data_vars:
        attrs = result[name].attrs
        result[name].attrs = {"units": attrs["units"]} if "units" in attrs else {}
    return encode_multiindexes_for_storage(result)


def to_datatree(artifact: AffineFluxMapArtifact) -> xr.DataTree:
    """Encode one map without composing flux and prolongation."""
    value = artifact.affine_map
    if value.representation == "bucket":
        prolongation = value.prolongation.to_datatree().map_over_datasets(  # type: ignore[union-attr]
            _operator_dataset_for_storage
        )
    else:
        prolongation = _array_node(value.prolongation, "prolongation")  # type: ignore[arg-type]
    tree = xr.DataTree.from_dict(
        {
            "native_mean": _array_node(value.native_mean, "native_mean"),
            "flux": _array_node(value.flux, "flux"),
            "prolongation": prolongation,
        }
    )
    tree.attrs.update(
        schema=SCHEMA,
        schema_version=SCHEMA_VERSION,
        affine_convention=AFFINE_CONVENTION,
        representation=value.representation,
        state_dim=value.state_dim,
        native_dims=json.dumps(value.native_dims),
        uncertainty_scope=value.uncertainty_scope,
        prepared_inputs_id=artifact.prepared_inputs_id,
        projection_provenance=json.dumps(dict(artifact.projection_provenance), sort_keys=True),
        reconstruction_provenance=json.dumps(dict(artifact.reconstruction_provenance), sort_keys=True),
        source_provenance=json.dumps(dict(artifact.source_provenance), sort_keys=True),
    )
    return tree


def _require_keys(actual: Set[Hashable], expected: Set[str], context: str) -> None:
    extra = actual - expected
    missing = expected - actual
    if extra:
        raise ValueError(f"Unexpected {context} element {min(extra, key=repr)!r}.")
    if missing:
        raise ValueError(f"Missing {context} element {sorted(missing)[0]!r}.")


def _validate_array_node(node: xr.DataTree, name: str) -> xr.DataArray:
    _require_keys(set(node.children), set(), f"{name} child")
    dataset = node.to_dataset()
    _require_keys(set(dataset.data_vars), {name}, f"{name} variable")
    if set(dataset.attrs) - {MULTIINDEX_DIMS_ATTR}:
        raise ValueError(f"Unexpected {name} group attribute.")
    _validate_coordinates(dataset, name)
    array = restore_declared_multiindexes(dataset, strict=MULTIINDEX_DIMS_ATTR in dataset.attrs)[name]
    _require_keys(set(array.attrs), {"units"}, f"{name} attribute")
    return array


def _validate_coordinates(dataset: xr.Dataset, context: str) -> None:
    """Reject coordinate-shaped extra payloads, including prohibited fields."""
    extra = set(dataset.coords) - set(dataset.dims)
    if MULTIINDEX_DIMS_ATTR in dataset.attrs:
        try:
            declaration = json.loads(dataset.attrs[MULTIINDEX_DIMS_ATTR])
            levels = {level for record in declaration["dims"] for level in record["levels"]}
        except (TypeError, KeyError, ValueError) as exc:
            raise ValueError(f"Malformed MultiIndex metadata in {context}.") from exc
        extra -= levels
    if extra:
        raise ValueError(f"Unexpected {context} coordinate {min(extra, key=repr)!r}.")
    for name, coordinate in dataset.coords.items():
        unexpected_attrs = set(coordinate.attrs) - {
            "units",
            "long_name",
            "standard_name",
            "axis",
            "calendar",
            "compress",
        }
        if unexpected_attrs:
            raise ValueError(
                f"Unexpected {context} coordinate {name!r} attribute {sorted(unexpected_attrs)[0]!r}."
            )


def _validate_operator_node(node: xr.DataTree) -> xr.DataTree:
    kind = node.attrs.get("kind")
    if kind not in ("bucket", "multisource_bucket"):
        raise ValueError(f"Unsupported bucket prolongation kind {kind!r}.")
    expected_attrs = {"schema", "schema_version", "kind", "grid_dims", "state_dim"}
    if kind == "bucket":
        expected_attrs.add("region_labels")
    else:
        expected_attrs.update(("source_dim", "region_in_source_dim"))
    _require_keys(set(node.attrs) - {MULTIINDEX_DIMS_ATTR}, expected_attrs, "bucket attribute")
    _require_keys(set(node.to_dataset().data_vars), {"basis_flat"}, "bucket variable")
    _validate_coordinates(node.to_dataset(), "bucket")
    _require_keys(
        set(node.children), {"state_metadata"} if "state_metadata" in node.children else set(), "bucket child"
    )
    if "state_metadata" in node.children:
        metadata = node["state_metadata"]
        _require_keys(set(metadata.children), set(), "state_metadata child")
        _require_keys(set(metadata.to_dataset().data_vars), _STATE_METADATA_VARS, "state_metadata variable")
        if set(metadata.attrs) - {MULTIINDEX_DIMS_ATTR}:
            raise ValueError("Unexpected state_metadata group attribute.")
        _validate_coordinates(metadata.to_dataset(), "state_metadata")
    for path, child in node.subtree_with_keys:
        for variable in child.to_dataset().data_vars.values():
            if set(variable.attrs) - {"units"}:
                raise ValueError(f"Unexpected bucket variable attribute in {path!r}.")
    return node.map_over_datasets(
        lambda ds: restore_declared_multiindexes(ds, strict=MULTIINDEX_DIMS_ATTR in ds.attrs)
    )


def _decode_json_attr(attrs: Mapping[Hashable, Any], name: str) -> Mapping[str, Any]:
    raw = attrs[name]
    if not isinstance(raw, str):
        raise ValueError(f"{name} must be a JSON object string.")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} contains invalid JSON.") from exc
    return _json_mapping(value, name)


def from_datatree(tree: xr.DataTree) -> AffineFluxMapArtifact:
    """Validate the complete payload before reconstructing any map object."""
    _require_keys(set(tree.attrs), _ROOT_ATTRS, "root attribute")
    _require_keys(set(tree.children), {"native_mean", "flux", "prolongation"}, "root child")
    _require_keys(set(tree.to_dataset().data_vars), set(), "root variable")
    _require_keys(set(tree.to_dataset().coords), set(), "root coordinate")
    if tree.attrs["schema"] != SCHEMA or tree.attrs["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported AffineFluxMap schema or schema_version.")
    if tree.attrs["affine_convention"] != AFFINE_CONVENTION:
        raise ValueError("Unsupported affine_convention.")
    if tree.attrs["uncertainty_scope"] != RETAINED_STATE_CONDITIONAL:
        raise ValueError("Unsupported uncertainty_scope.")
    representation = tree.attrs["representation"]
    if representation not in ("bucket", "explicit"):
        raise ValueError(f"Unsupported prolongation representation {representation!r}.")
    state_dim = tree.attrs["state_dim"]
    if not isinstance(state_dim, str) or not state_dim:
        raise ValueError("state_dim must be a non-empty string.")
    try:
        native_dims = json.loads(tree.attrs["native_dims"])
    except (TypeError, ValueError) as exc:
        raise ValueError("native_dims must be a JSON list.") from exc
    if (
        not isinstance(native_dims, list)
        or not native_dims
        or not all(isinstance(dim, str) and dim for dim in native_dims)
        or len(set(native_dims)) != len(native_dims)
    ):
        raise ValueError("native_dims must contain unique dimension names.")
    prepared_inputs_id = tree.attrs["prepared_inputs_id"]
    if not isinstance(prepared_inputs_id, str) or not prepared_inputs_id.strip():
        raise ValueError("prepared_inputs_id must be a non-empty string.")
    projection = _decode_json_attr(tree.attrs, "projection_provenance")
    reconstruction = _decode_json_attr(tree.attrs, "reconstruction_provenance")
    source = _decode_json_attr(tree.attrs, "source_provenance")
    native_mean = _validate_array_node(tree.children["native_mean"], "native_mean")
    flux = _validate_array_node(tree.children["flux"], "flux")
    if tuple(native_dims) != native_mean.dims:
        raise ValueError("native_dims do not match native_mean dimensions.")
    if representation == "explicit":
        prolongation: xr.DataArray | BasisOperator = _validate_array_node(
            tree.children["prolongation"], "prolongation"
        )
    else:
        prolongation = BasisOperator.decode_datatree(_validate_operator_node(tree.children["prolongation"]))
    return AffineFluxMapArtifact(
        affine_map=AffineFluxMap(native_mean, flux, prolongation, state_dim=state_dim),
        prepared_inputs_id=prepared_inputs_id,
        projection_provenance=projection,
        reconstruction_provenance=reconstruction,
        source_provenance=source,
    )


def save(artifact: AffineFluxMapArtifact, path: str | Path) -> None:
    """Save a NetCDF or Zarr affine reconstruction artifact."""
    save_datatree(to_datatree(artifact), path)


def load(path: str | Path) -> AffineFluxMapArtifact:
    """Load and validate a NetCDF or Zarr affine reconstruction artifact."""
    return from_datatree(open_datatree_loaded(path))


__all__ = ["AffineFluxMapArtifact", "to_datatree", "from_datatree", "save", "load"]
