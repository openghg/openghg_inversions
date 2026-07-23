"""Build the frozen native-grid input for the Gamma--Beta PARIS HPC test.

The source is the audited May 2014 reconstruction snapshot produced by
``paris_rjmcmc_reconstruction.py``.  This script performs no OpenGHG search
and does not discover scientific inputs.  It checks the complete source-file
digest, preserves the exact 1,382-row ordering, maps the rectangular InTEM
label-6 domain to a 183-by-128 native grid, and writes one immutable NetCDF
plus a JSON provenance sidecar.

The Gamma--Beta nominal base measure is normalized spherical grid-cell area.
It is calculated directly from the labelled latitude/longitude cell centres;
no flux-derived epsilon floor or other support repair is used.
"""

from __future__ import annotations

import argparse
import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import NDArray

PARIS_OBSERVATIONS = 1_382
PARIS_GRID_SHAPE = (183, 128)
PARIS_OUTER_REGIONS = 6
EXPECTED_SOURCE_SHA256 = "45d49c37ecd967f503151f931940eb883693ca2841d882cab3a8464d01649b56"
INPUT_SCHEMA_ID = "paris-may-2014-gamma-beta-native-v1"
SIDECAR_SCHEMA_ID = "paris-gamma-beta-frozen-input-sidecar-v1"
WEIGHT_POLICY_ID = "spherical-grid-cell-area-v1"
OUTER_LABELS = tuple(f"intem_label_{index}" for index in range(PARIS_OUTER_REGIONS))
COUNTRIES = ("GBR", "IRL")
EARTH_RADIUS_M = 6_371_000.0

FloatArray = NDArray[np.float64]


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    """Hash one array's dtype, shape, and contiguous value bytes."""
    contiguous = np.ascontiguousarray(array)
    digest = sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape, separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _decode_metadata(encoded: np.ndarray) -> dict[str, Any]:
    """Decode and validate the source snapshot's no-pickle JSON metadata."""
    values = np.asarray(encoded)
    if values.ndim != 1 or values.dtype != np.dtype(np.uint8):
        raise ValueError("Source metadata must be a one-dimensional uint8 JSON byte array.")
    try:
        metadata = json.loads(values.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Source metadata is not valid UTF-8 JSON.") from error
    if not isinstance(metadata, dict):
        raise TypeError("Source metadata JSON must be an object.")
    return metadata


def _strictly_increasing(values: np.ndarray, *, name: str) -> FloatArray:
    """Return finite float64 cell centres after enforcing strict ordering."""
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size < 2:
        raise ValueError(f"{name} must be a one-dimensional coordinate with at least two cells.")
    if not np.all(np.isfinite(result)) or not np.all(np.diff(result) > 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing.")
    return result


def _cell_edges(centres: FloatArray) -> FloatArray:
    """Construct cell edges by midpoint interpolation and endpoint extrapolation."""
    midpoints = 0.5 * (centres[:-1] + centres[1:])
    return np.concatenate(
        (
            np.asarray([centres[0] - 0.5 * (centres[1] - centres[0])]),
            midpoints,
            np.asarray([centres[-1] + 0.5 * (centres[-1] - centres[-2])]),
        )
    )


def spherical_grid_cell_area(
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    radius_m: float = EARTH_RADIUS_M,
) -> FloatArray:
    """Calculate positive spherical cell areas from ordered centre coordinates."""
    latitudes = _strictly_increasing(latitude, name="latitude")
    longitudes = _strictly_increasing(longitude, name="longitude")
    if not np.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("radius_m must be finite and strictly positive.")
    latitude_edges = _cell_edges(latitudes)
    longitude_edges = _cell_edges(longitudes)
    if latitude_edges[0] < -90.0 or latitude_edges[-1] > 90.0:
        raise ValueError("Latitude cell edges extend outside [-90, 90] degrees.")
    if np.any(np.diff(longitude_edges) > 360.0):
        raise ValueError("Longitude cell width cannot exceed 360 degrees.")
    latitude_factor = np.abs(
        np.sin(np.deg2rad(latitude_edges[1:]))
        - np.sin(np.deg2rad(latitude_edges[:-1]))
    )
    longitude_width = np.abs(np.deg2rad(np.diff(longitude_edges)))
    area = np.asarray(radius_m**2 * latitude_factor[:, None] * longitude_width[None, :])
    if not np.all(np.isfinite(area)) or np.any(area <= 0.0):
        raise ValueError("Spherical grid-cell area calculation did not produce positive support.")
    return area


def _inner_rectangle(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exact row/column indices for the filled native inner rectangle."""
    values = np.asarray(mask)
    if values.ndim != 2 or values.dtype != np.dtype(np.bool_):
        raise ValueError("inner_mask must be a two-dimensional Boolean array.")
    rows = np.flatnonzero(values.any(axis=1))
    columns = np.flatnonzero(values.any(axis=0))
    if rows.size != PARIS_GRID_SHAPE[0] or columns.size != PARIS_GRID_SHAPE[1]:
        raise ValueError(
            "inner_mask must select exactly "
            f"{PARIS_GRID_SHAPE[0]} latitude by {PARIS_GRID_SHAPE[1]} longitude cells."
        )
    if not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
        raise ValueError("inner_mask latitude indices must be contiguous.")
    if not np.array_equal(columns, np.arange(columns[0], columns[-1] + 1)):
        raise ValueError("inner_mask longitude indices must be contiguous.")
    expected = np.zeros_like(values)
    expected[np.ix_(rows, columns)] = True
    if not np.array_equal(values, expected):
        raise ValueError("inner_mask must be a completely filled rectangular slice.")
    return rows, columns


def _measurement_labels(site: np.ndarray, time_ns: np.ndarray) -> np.ndarray:
    """Build unique, stable ``site|time`` labels in source row order."""
    sites = np.asarray(site)
    times = np.asarray(time_ns)
    if sites.shape != (PARIS_OBSERVATIONS,) or times.shape != (PARIS_OBSERVATIONS,):
        raise ValueError("site and time must each contain exactly one value per observation.")
    site_labels = np.asarray([str(value).strip() for value in sites.tolist()], dtype=np.str_)
    if np.any(site_labels == ""):
        raise ValueError("site labels must be nonempty.")
    if times.dtype.kind not in "iu":
        raise ValueError("time must contain integer nanoseconds since the Unix epoch.")
    times_i64 = times.astype(np.int64, copy=False)
    if np.any(times_i64 == np.iinfo(np.int64).min):
        raise ValueError("time cannot contain NaT.")
    time_labels = np.datetime_as_string(times_i64.astype("datetime64[ns]"), unit="ns")
    labels = np.asarray(
        [f"{site_name}|{time_name}" for site_name, time_name in zip(site_labels, time_labels, strict=True)]
    )
    if np.unique(labels).size != PARIS_OBSERVATIONS:
        raise ValueError("Source site|time observation labels must be unique.")
    return labels


def _required_array(
    snapshot: np.lib.npyio.NpzFile,
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: np.dtype[Any] | None = None,
    positive: bool = False,
) -> np.ndarray:
    """Load one required finite source array with an exact shape."""
    if name not in snapshot.files:
        raise ValueError(f"Source snapshot is missing required array {name!r}.")
    values = np.asarray(snapshot[name])
    if values.shape != shape:
        raise ValueError(f"Source array {name!r} must have shape {shape}; found {values.shape}.")
    if dtype is not None and values.dtype != dtype:
        raise ValueError(f"Source array {name!r} must have dtype {dtype}; found {values.dtype}.")
    if values.dtype.kind in "iuf" and not np.all(np.isfinite(values)):
        raise ValueError(f"Source array {name!r} contains non-finite values.")
    if positive and np.any(values <= 0.0):
        raise ValueError(f"Source array {name!r} must be strictly positive.")
    return values


def _country_fractions(
    metadata: dict[str, Any],
    *,
    latitude: FloatArray,
    longitude: FloatArray,
    rows: np.ndarray,
    columns: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and verify the archived GBR/IRL country fractions on the inner grid."""
    archive_value = metadata.get("archived_flux")
    archive_digest = metadata.get("archived_flux_sha256")
    if not isinstance(archive_value, str) or not isinstance(archive_digest, str):
        raise TypeError("Source metadata does not identify the archived flux and its SHA-256.")
    archive_path = Path(archive_value)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archived flux comparator is not a file: {archive_path}")
    actual_digest = _sha256_file(archive_path)
    if actual_digest != archive_digest:
        raise ValueError("Archived flux comparator SHA-256 does not match source metadata.")
    with xr.open_dataset(archive_path, engine="h5netcdf") as opened:
        if "country_fraction" not in opened:
            raise ValueError("Archived flux comparator has no country_fraction variable.")
        archive_latitude = np.asarray(opened["latitude"].values, dtype=np.float64)
        archive_longitude = np.asarray(opened["longitude"].values, dtype=np.float64)
        if not np.allclose(archive_latitude, latitude, rtol=0.0, atol=2.0e-5):
            raise ValueError("Archived country-fraction latitude does not align with the source grid.")
        if not np.allclose(archive_longitude, longitude, rtol=0.0, atol=2.0e-5):
            raise ValueError("Archived country-fraction longitude does not align with the source grid.")
        selected = opened["country_fraction"].sel(country=list(COUNTRIES))
        fractions = np.asarray(
            selected.transpose("country", "latitude", "longitude").values,
            dtype=np.float64,
        )[:, rows][:, :, columns]
    if not np.all(np.isfinite(fractions)) or np.any(fractions < 0.0) or np.any(fractions > 1.0):
        raise ValueError("Archived GBR/IRL country fractions must be finite and within [0, 1].")
    return fractions, {
        "path": str(archive_path.resolve()),
        "sha256": actual_digest,
        "countries": list(COUNTRIES),
        "coordinate_tolerance_degrees": 2.0e-5,
    }


def _build_dataset(
    snapshot: np.lib.npyio.NpzFile,
    *,
    source_path: Path,
    source_digest: str,
    include_country_fractions: bool,
) -> tuple[xr.Dataset, dict[str, Any]]:
    """Validate the source snapshot and construct the frozen native dataset."""
    observations = _required_array(snapshot, "observations", (PARIS_OBSERVATIONS,))
    observation_sd = _required_array(
        snapshot,
        "observation_sd",
        (PARIS_OBSERVATIONS,),
        positive=True,
    )
    sensitivity = _required_array(
        snapshot,
        "sensitivity",
        (PARIS_OBSERVATIONS, PARIS_GRID_SHAPE[0] * PARIS_GRID_SHAPE[1]),
    )
    outer_design = _required_array(
        snapshot,
        "outer_design",
        (PARIS_OBSERVATIONS, PARIS_OUTER_REGIONS),
    )
    fixed_offset = _required_array(snapshot, "fixed_offset", (PARIS_OBSERVATIONS,))
    site = _required_array(snapshot, "site", (PARIS_OBSERVATIONS,))
    time_ns = _required_array(snapshot, "time", (PARIS_OBSERVATIONS,))
    latitude_full = _strictly_increasing(snapshot["latitude"], name="source latitude")
    longitude_full = _strictly_increasing(snapshot["longitude"], name="source longitude")
    full_shape = (latitude_full.size, longitude_full.size)
    inner_mask = _required_array(
        snapshot,
        "inner_mask",
        full_shape,
        dtype=np.dtype(np.bool_),
    )
    rows, columns = _inner_rectangle(inner_mask)
    latitude = latitude_full[rows]
    longitude = longitude_full[columns]
    labels = _measurement_labels(site, time_ns)
    metadata = _decode_metadata(snapshot["metadata"])
    expected_metadata = {
        "schema": "paris-rjmcmc-native-snapshot-v1",
        "n_observations": PARIS_OBSERVATIONS,
        "n_inner_cells": PARIS_GRID_SHAPE[0] * PARIS_GRID_SHAPE[1],
        "n_outer_regions": PARIS_OUTER_REGIONS,
    }
    for name, expected in expected_metadata.items():
        if metadata.get(name) != expected:
            raise ValueError(
                f"Source metadata {name!r} must equal {expected!r}; found {metadata.get(name)!r}."
            )

    area = spherical_grid_cell_area(latitude, longitude)
    area_total = float(area.sum(dtype=np.float64))
    nominal_weight = area / area_total
    if np.any(nominal_weight <= 0.0):
        raise RuntimeError("Area normalization unexpectedly removed positive support.")
    prior_flux_full = _required_array(snapshot, "current_prior_flux", full_shape)
    prior_flux = np.asarray(prior_flux_full[np.ix_(rows, columns)], dtype=np.float64)
    if np.any(prior_flux < 0.0):
        raise ValueError("current_prior_flux cannot contain negative values.")

    variables: dict[str, Any] = {
        "fp_x_flux": (
            ("nmeasure", "lat", "lon"),
            np.ascontiguousarray(sensitivity.reshape((PARIS_OBSERVATIONS, *PARIS_GRID_SHAPE))),
            {
                "long_name": "native-grid response to unit grid-cell flux scaling",
                "units": metadata.get("units", {}).get("sensitivity", "ppb at unit flux scale"),
                "flattening_order": "C order; longitude varies fastest",
            },
        ),
        "mf": (
            ("nmeasure",),
            np.asarray(observations, dtype=np.float64),
            {"long_name": "observed methane mole fraction", "units": "ppb"},
        ),
        "mf_error": (
            ("nmeasure",),
            np.asarray(observation_sd, dtype=np.float64),
            {
                "long_name": "fixed independent methane observation error",
                "units": "ppb",
                "error_model": metadata.get("error_model", ""),
            },
        ),
        "nominal_weight": (
            ("lat", "lon"),
            nominal_weight,
            {
                "long_name": "normalized spherical grid-cell area base measure",
                "units": "1",
                "policy_id": WEIGHT_POLICY_ID,
                "normalization": "grid_cell_area / sum(grid_cell_area)",
                "support_policy": "strictly positive analytic area; no flux epsilon floor",
            },
        ),
        "grid_cell_area": (
            ("lat", "lon"),
            area,
            {
                "long_name": "spherical grid-cell area",
                "units": "m2",
                "earth_radius_m": EARTH_RADIUS_M,
                "edge_construction": "centre midpoints with endpoint half-spacing extrapolation",
            },
        ),
        "outer_design": (
            ("nmeasure", "outer_region"),
            np.asarray(outer_design),
            {
                "long_name": "fixed-geometry inferred outer-region response",
                "units": "ppb at unit flux scale",
                "source_order": "numerical InTEM region labels 0 through 5",
            },
        ),
        "YaprioriBC": (
            ("nmeasure",),
            np.asarray(fixed_offset, dtype=np.float64),
            {
                "long_name": "fixed archived prior boundary-condition contribution",
                "units": "ppb",
                "treatment": metadata.get("boundary_treatment", ""),
            },
        ),
        "prior_flux": (
            ("lat", "lon"),
            prior_flux,
            {
                "long_name": "current OpenGHG prior methane flux used by the reconstruction",
                "units": "mol m-2 s-1",
                "role": "auxiliary scientific postprocessing; not a Gamma-Beta driver input",
            },
        ),
        "site": (
            ("nmeasure",),
            np.asarray([str(value) for value in site.tolist()], dtype=np.str_),
            {"long_name": "observation site code"},
        ),
        "time": (
            ("nmeasure",),
            np.asarray(time_ns, dtype=np.int64).astype("datetime64[ns]"),
            {"long_name": "observation interval left-edge time"},
        ),
    }
    for source_name, output_name, long_name in (
        ("archived_prior", "archived_prior_prediction", "archived RHIME prior prediction"),
        ("archived_posterior", "archived_posterior_prediction", "archived RHIME posterior prediction"),
        ("archived_uncertainty", "archived_posterior_uncertainty", "archived RHIME uncertainty"),
    ):
        if source_name in snapshot.files:
            values = _required_array(snapshot, source_name, (PARIS_OBSERVATIONS,))
            variables[output_name] = (
                ("nmeasure",),
                np.asarray(values, dtype=np.float64),
                {
                    "long_name": long_name,
                    "units": "ppb",
                    "role": "auxiliary comparator; not a Gamma-Beta driver input or flux truth",
                },
            )

    archive_provenance: dict[str, Any] | None = None
    if include_country_fractions:
        fractions, archive_provenance = _country_fractions(
            metadata,
            latitude=latitude_full,
            longitude=longitude_full,
            rows=rows,
            columns=columns,
        )
        variables["country_fraction"] = (
            ("country", "lat", "lon"),
            fractions,
            {
                "long_name": "archived country overlap fraction",
                "units": "1",
                "role": "auxiliary GBR/IRL scientific postprocessing",
            },
        )

    coordinates: dict[str, Any] = {
        "nmeasure": (
            "nmeasure",
            labels,
            {"long_name": "unique source-order site|time observation identifier"},
        ),
        "lat": ("lat", latitude, {"standard_name": "latitude", "units": "degrees_north"}),
        "lon": ("lon", longitude, {"standard_name": "longitude", "units": "degrees_east"}),
        "outer_region": (
            "outer_region",
            np.asarray(OUTER_LABELS, dtype=np.str_),
            {"long_name": "reviewed InTEM outer-region label in numerical source order"},
        ),
    }
    if include_country_fractions:
        coordinates["country"] = (
            "country",
            np.asarray(COUNTRIES, dtype=np.str_),
            {"long_name": "ISO 3166-1 alpha-3 country code"},
        )

    source_metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    dataset = xr.Dataset(
        variables,
        coords=coordinates,
        attrs={
            "schema_id": INPUT_SCHEMA_ID,
            "title": "Frozen May 2014 native PARIS input for Gamma-Beta RJMCMC",
            "source_snapshot": str(source_path.resolve()),
            "source_snapshot_sha256": source_digest,
            "source_snapshot_schema": metadata["schema"],
            "source_metadata_json": source_metadata_json,
            "inner_mapping": (
                f"source rows {int(rows[0])}:{int(rows[-1]) + 1}, "
                f"columns {int(columns[0])}:{int(columns[-1]) + 1}; C order"
            ),
            "nominal_weight_policy": WEIGHT_POLICY_ID,
            "nominal_weight_support": "strictly positive spherical cell area; no epsilon floor",
            "outer_region_order": ",".join(OUTER_LABELS),
            "construction_script": str(Path(__file__).resolve()),
        },
    )
    construction = {
        "source_metadata": metadata,
        "mapping": {
            "observation_count": PARIS_OBSERVATIONS,
            "native_grid_shape": list(PARIS_GRID_SHAPE),
            "source_grid_shape": list(full_shape),
            "source_latitude_index_slice": [int(rows[0]), int(rows[-1]) + 1],
            "source_longitude_index_slice": [int(columns[0]), int(columns[-1]) + 1],
            "source_sensitivity_order": "Boolean inner_mask flattened in C order",
            "output_sensitivity_order": "nmeasure,lat,lon; longitude varies fastest",
            "measurement_label": "site|numpy.datetime_as_string(time_ns, unit='ns')",
            "measurement_labels_unique": True,
            "outer_region_labels": list(OUTER_LABELS),
            "outer_region_source_order": list(range(PARIS_OUTER_REGIONS)),
        },
        "nominal_weight": {
            "policy_id": WEIGHT_POLICY_ID,
            "earth_radius_m": EARTH_RADIUS_M,
            "cell_edge_construction": "centre midpoints and endpoint half-spacing extrapolation",
            "normalization": "area divided by float64 sum over the 183x128 grid",
            "normalization_area_m2": area_total,
            "minimum_area_m2": float(area.min()),
            "maximum_area_m2": float(area.max()),
            "minimum_normalized_weight": float(nominal_weight.min()),
            "maximum_normalized_weight": float(nominal_weight.max()),
            "normalized_weight_sum": float(nominal_weight.sum(dtype=np.float64)),
            "epsilon_floor": None,
        },
        "auxiliary_inputs": {
            "prior_flux": {
                "source_array": "current_prior_flux",
                "units": "mol m-2 s-1",
                "used_by_sampler": False,
            },
            "country_fraction": archive_provenance,
        },
    }
    return dataset, construction


def _netcdf_encoding(dataset: xr.Dataset) -> dict[str, dict[str, Any]]:
    """Return deterministic lossless compression settings for numeric variables."""
    encoding: dict[str, dict[str, Any]] = {}
    for name, variable in dataset.data_vars.items():
        if variable.dtype.kind in "biuf" and variable.ndim > 0:
            encoding[name] = {"zlib": True, "complevel": 4, "shuffle": True}
    return encoding


def _atomic_write_netcdf(dataset: xr.Dataset, path: Path) -> None:
    """Write one HDF5-backed NetCDF atomically."""
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        dataset.to_netcdf(
            temporary,
            engine="h5netcdf",
            encoding=_netcdf_encoding(dataset),
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(document: dict[str, Any], path: Path) -> None:
    """Write strict canonical JSON atomically."""
    temporary = path.with_name(f".{path.name}.tmp")
    text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Audited native snapshot NPZ.")
    parser.add_argument("--output", type=Path, required=True, help="New frozen NetCDF path.")
    parser.add_argument(
        "--expected-source-sha256",
        default=EXPECTED_SOURCE_SHA256,
        help="Required whole-file source SHA-256.",
    )
    parser.add_argument(
        "--include-country-fractions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include verified archived GBR/IRL country fractions.",
    )
    return parser


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    """Build and persist the frozen input and auditable sidecar."""
    source = arguments.source.resolve()
    output = arguments.output.resolve()
    sidecar = output.with_suffix(".json")
    if not source.is_file():
        raise FileNotFoundError(f"Source snapshot is not a file: {source}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output parent directory does not exist: {output.parent}")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"Frozen output or sidecar already exists: {output}, {sidecar}")
    expected_digest = arguments.expected_source_sha256.lower()
    if len(expected_digest) != 64 or any(value not in "0123456789abcdef" for value in expected_digest):
        raise ValueError("--expected-source-sha256 must be exactly 64 hexadecimal characters.")
    source_digest = _sha256_file(source)
    if source_digest != expected_digest:
        raise ValueError("Source snapshot SHA-256 does not match --expected-source-sha256.")

    with np.load(source, allow_pickle=False) as snapshot:
        dataset, construction = _build_dataset(
            snapshot,
            source_path=source,
            source_digest=source_digest,
            include_country_fractions=arguments.include_country_fractions,
        )
    semantic_hashes = {
        name: _sha256_array(np.asarray(dataset[name].values))
        for name in sorted(str(value) for value in dataset.variables)
    }
    _atomic_write_netcdf(dataset, output)
    output_digest = _sha256_file(output)
    script_path = Path(__file__).resolve()
    document = {
        "schema_id": SIDECAR_SCHEMA_ID,
        "input_schema_id": INPUT_SCHEMA_ID,
        "source": {
            "path": str(source),
            "sha256": source_digest,
            "expected_sha256": expected_digest,
        },
        "output": {
            "path": str(output),
            "sha256": output_digest,
            "netcdf_engine": "h5netcdf",
            "sizes": {name: int(size) for name, size in dataset.sizes.items()},
            "variable_semantic_sha256": semantic_hashes,
        },
        "construction": construction,
        "construction_script": {
            "path": str(script_path),
            "sha256": _sha256_file(script_path),
        },
        "command": {
            "source": str(source),
            "output": str(output),
            "expected_source_sha256": expected_digest,
            "include_country_fractions": bool(arguments.include_country_fractions),
        },
    }
    _atomic_write_json(document, sidecar)
    return document


def main(argv: list[str] | None = None) -> int:
    """Run the command-line builder."""
    document = run(_parser().parse_args(argv))
    print(json.dumps(document, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
