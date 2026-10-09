"""Versioned dataset-only acquisition artifacts; no OpenGHG wrapper codec."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import xarray as xr
import zarr

from openghg_inversions.array_ops import to_dense

from ._provenance import InputProvenance, MergedDataProvenance
from ._site_options import SiteOptions
from .serialise import OutputFormat, _make_merged_data_name, _split_suffix

if TYPE_CHECKING:
    from .acquisition import RhimeMergedData

SCHEMA = "openghg-inversions-rhime-acquisition"
SCHEMA_VERSION = 1


def _encode_provenance(provenance: MergedDataProvenance) -> dict:
    """Encode selected identities using the version-one file schema."""
    inputs = {
        f"{kind}:{label}": asdict(identity)
        for kind in ("observations", "footprints", "flux")
        for label, identity in getattr(provenance, kind).items()
    }
    if provenance.boundary is not None:
        inputs["boundary"] = asdict(provenance.boundary)
    return {
        "openghg": {"version": provenance.openghg_version, "commit": provenance.openghg_commit},
        "inputs": inputs,
    }


def _decode_provenance(value: Any, *, expected_inputs: set[str] | None = None) -> MergedDataProvenance:
    """Validate version-one metadata without filling missing persisted facts."""
    if not isinstance(value, dict) or set(value) != {"openghg", "inputs"}:
        raise ValueError("Artifact provenance requires OpenGHG and input identities.")
    software = value["openghg"]
    if not isinstance(software, dict) or set(software) != {"version", "commit"}:
        raise ValueError("OpenGHG provenance requires version and commit strings (or 'unknown').")
    if not isinstance(value["inputs"], dict):
        raise ValueError("Artifact input provenance must be a mapping.")
    if expected_inputs is not None and set(value["inputs"]) != expected_inputs:
        raise ValueError("Artifact provenance must identify each retained input.")
    groups: dict[str, dict[str, InputProvenance]] = {"observations": {}, "footprints": {}, "flux": {}}
    boundary = None
    for label, identity in value["inputs"].items():
        if not isinstance(identity, dict) or set(identity) != {"store", "uuid", "dataversion"}:
            raise ValueError("Input provenance requires store, uuid and dataversion identifiers.")
        selected = InputProvenance(**{
            name: tuple(item) if isinstance(item, list) else item for name, item in identity.items()
        })
        if label == "boundary":
            boundary = selected
        elif isinstance(label, str):
            kind, separator, name = label.partition(":")
            if kind not in groups or not separator:
                raise ValueError("Artifact provenance contains an unsupported input label.")
            groups[kind][name] = selected
        else:
            raise ValueError("Artifact provenance input labels must be strings.")
    return MergedDataProvenance(
        openghg_version=software["version"], openghg_commit=software["commit"],
        observations=groups["observations"], footprints=groups["footprints"],
        flux=groups["flux"], boundary=boundary,
    )


def artifact_path(
    directory: str | Path,
    species: str | None,
    start_date: str | None,
    output_name: str | None,
    name: str | None,
    output_format: OutputFormat | None,
) -> tuple[Path, OutputFormat]:
    """Resolve one exact artifact path, without probing alternate formats."""
    if directory is None:
        raise ValueError("Provide a merged-data directory.")
    if name is None:
        if species is None or start_date is None or output_name is None:
            raise ValueError("Provide merged_data_name or species, start_date and output_name.")
        name = _make_merged_data_name(species, start_date, output_name)
    name, suffix = _split_suffix(name)
    output_format = suffix or output_format or "zarr.zip"
    if output_format not in {"netcdf", "zarr", "zarr.zip"}:
        raise ValueError(f"Unsupported merged-data format {output_format!r}.")
    return Path(directory) / (
        name + (".nc" if output_format == "netcdf" else "." + output_format)
    ), output_format


def _json_default(value: Any) -> Any:
    # Scientific attributes may contain NumPy scalars/arrays, never arbitrary objects.
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported scientific attribute type: {type(value).__name__}")


def _encode_dataset(dataset: xr.Dataset) -> xr.Dataset:
    result = dataset.copy(deep=False)
    attributes = {
        "dataset": dict(dataset.attrs),
        "variables": {name: dict(var.attrs) for name, var in dataset.variables.items()},
    }
    result.attrs = {"scientific_attributes": json.dumps(attributes, default=_json_default)}
    result.encoding = {}
    for name in result.variables:
        if name in result.coords:
            result = result.assign_coords({name: to_dense(result[name])})
        else:
            result[name] = to_dense(result[name])
        result[name].attrs = {}
        result[name].encoding = {}
    return result


def _decode_dataset(dataset: xr.Dataset) -> xr.Dataset:
    attributes = json.loads(dataset.attrs["scientific_attributes"])
    dataset.attrs = attributes["dataset"]
    if set(attributes["variables"]) != set(dataset.variables):
        raise ValueError("Artifact scientific attribute inventory does not match its variables.")
    for name, attrs in attributes["variables"].items():
        dataset[name].attrs = attrs
    return dataset


def save_artifact(merged: RhimeMergedData, path: Path, output_format: OutputFormat) -> None:
    """Materialize related acquired arrays together through one DataTree write."""
    if merged.acquisition.get("stage") != "acquired":
        raise ValueError(
            "Modern merged artifacts require acquired, unfiltered data; for legacy input confirm acquisition_stage='acquired' only when known."
        )
    selectors = {}
    for field in fields(SiteOptions):
        selectors[field.name] = [
            {"slice": [value.start, value.stop, value.step]} if isinstance(value, slice) else value
            for value in getattr(merged.site_options, field.name)
        ]
    manifest = {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "stage": "acquired",
        "site_options": selectors,
        "sources": list(merged.flux_data),
        "boundary": merged.boundary_data is not None,
        "split_by_sectors": merged.split_by_sectors,
        "provenance": _encode_provenance(merged.provenance),
        "acquisition": merged.acquisition,
    }
    nodes = {"/": xr.Dataset(attrs={"manifest": json.dumps(manifest, default=_json_default)})}
    nodes.update(
        {f"site_{i}": _encode_dataset(merged.site_data[site]) for i, site in enumerate(merged.sites)}
    )
    nodes.update({f"flux_{i}": _encode_dataset(data) for i, data in enumerate(merged.flux_data.values())})
    if merged.boundary_data is not None:
        nodes["boundary"] = _encode_dataset(merged.boundary_data)
    tree = xr.DataTree.from_dict(nodes)
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "netcdf":
        tree.to_netcdf(path)
    else:
        # Rechunking is explicit at this serialization boundary, not on the handoff.
        tree = cast(xr.DataTree, tree.map_over_datasets(lambda ds: xr.unify_chunks(ds)[0]))
        if output_format == "zarr.zip":
            with zarr.ZipStore(path, mode="w") as store:
                tree.to_zarr(store, mode="w", consolidated=False)
                zarr.consolidate_metadata(store)
        else:
            tree.to_zarr(path, mode="w-")


def load_artifact(cls: type[RhimeMergedData], path: Path, output_format: OutputFormat) -> RhimeMergedData:
    """Open an exact modern artifact lazily and validate its complete inventory."""
    store = zarr.ZipStore(path, mode="r") if output_format == "zarr.zip" else None
    tree = None
    try:
        tree = xr.open_datatree(
            store if store is not None else path,
            engine="zarr" if output_format != "netcdf" else None,
            chunks={},
        )
        if "manifest" not in tree.attrs:
            raise ValueError(
                "Not a modern merged-data artifact; use the explicit load_legacy importer for old caches."
            )
        manifest = json.loads(tree.attrs["manifest"])
        if (manifest["schema"], manifest["version"], manifest["stage"]) != (
            SCHEMA,
            SCHEMA_VERSION,
            "acquired",
        ):
            raise ValueError("Unsupported modern merged-data schema, version or acquisition stage.")
        selectors = manifest["site_options"]
        if not isinstance(selectors, dict) or set(selectors) != {field.name for field in fields(SiteOptions)}:
            raise ValueError("Artifact must contain every SiteOptions selector.")
        if any(not isinstance(values, list) for values in selectors.values()):
            raise ValueError("Artifact SiteOptions fields must be explicit JSON lists.")
        options = SiteOptions(
            **{
                name: tuple(
                    slice(*value["slice"]) if isinstance(value, dict) and set(value) == {"slice"} else value
                    for value in values
                )
                for name, values in selectors.items()
            }
        )
        # Reuse the owning input validator for types as well as aligned lengths.
        if (
            SiteOptions.from_inputs(
                **{field.name: getattr(options, field.name) for field in fields(SiteOptions)}
            )
            != options
        ):
            raise ValueError("Artifact site labels/selectors are not canonical.")
        sources = manifest["sources"]
        if (
            not isinstance(sources, list)
            or any(not isinstance(source, str) for source in sources)
            or len(sources) != len(set(sources))
        ):
            raise ValueError("Artifact source labels must be unique strings.")
        if type(manifest["boundary"]) is not bool or type(manifest["split_by_sectors"]) is not bool:
            raise ValueError("Artifact layout flags must be booleans.")
        expected = {f"site_{i}" for i in range(len(options.sites))} | {
            f"flux_{i}" for i in range(len(sources))
        }
        if manifest["boundary"]:
            expected.add("boundary")
        if set(tree.children) != expected:
            raise ValueError("Artifact dataset inventory does not match its manifest.")
        if manifest["acquisition"].get("stage") != "acquired":
            raise ValueError("Modern artifact acquisition facts disagree with its stage.")
        expected_inputs = {
            f"{kind}:{site}" for kind in ("observations", "footprints") for site in options.sites
        } | {f"flux:{source}" for source in sources}
        if manifest["boundary"]:
            expected_inputs.add("boundary")
        result = cls(
            site_data={
                site: _decode_dataset(tree[f"site_{i}"].to_dataset()) for i, site in enumerate(options.sites)
            },
            flux_data={
                source: _decode_dataset(tree[f"flux_{i}"].to_dataset()) for i, source in enumerate(sources)
            },
            boundary_data=_decode_dataset(tree["boundary"].to_dataset()) if manifest["boundary"] else None,
            site_options=options,
            split_by_sectors=manifest["split_by_sectors"],
            provenance=_decode_provenance(manifest["provenance"], expected_inputs=expected_inputs),
            acquisition=manifest["acquisition"],
        )
        result._artifact = tree
        result._zip_store = store
        return result
    except Exception:
        if tree is not None:
            tree.close()
        if store is not None:
            store.close()
        raise
