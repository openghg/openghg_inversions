"""Concrete metadata schemas for the acquired, per-site RHIME handoff."""

from collections.abc import Sequence

import numpy as np
import xarray as xr
from xarray_validate import (
    AttrSchema, CoordsSchema, DataArraySchema, DatasetSchema, DimsSchema,
    SchemaError, ValidationContext,
)
from xarray_validate.units import set_registry

from ._units import cf_ureg, mole_fraction_unit_scale

set_registry(cf_ureg)

_REAL_DTYPES = (np.integer, np.floating)
_UNCERTAINTIES = ("mf_error", "mf_repeatability", "mf_variability")


def _same_unit_scale(actual: object, expected: str, *, name: str) -> None:
    """Accept equal-scale expressions without converting borrowed values."""
    if not isinstance(actual, str) or not actual.strip():
        raise ValueError(f"{name}.attrs.units must be a nonempty string.")
    try:
        AttrSchema(units=expected).validate(actual)
        return
    except (SchemaError, ValueError):
        # Native Unit equality misses equal-scale quantities such as umol/mol.
        pass
    try:
        quantity = cf_ureg.parse_expression(actual)
        reference = cf_ureg.parse_expression(expected)
        quantity = quantity if hasattr(quantity, "to") else cf_ureg.Quantity(quantity)
        reference = reference if hasattr(reference, "to") else cf_ureg.Quantity(reference)
        scale = float(quantity.to(reference.units).magnitude / reference.magnitude)
    except Exception as error:
        raise ValueError(f"{name}.attrs.units is incompatible with {expected!r}.") from error
    if not np.isclose(scale, 1.0, rtol=1e-12, atol=0):
        raise ValueError(f"{name}.attrs.units must have the same numeric scale as {expected!r}.")


def _indexed_coordinates(dataset: xr.Dataset, names: Sequence[str]) -> None:
    """Check small indexed coordinates without reading auxiliary coordinates."""
    for name in names:
        if name not in dataset.coords:
            continue  # The declarative coordinate schema reports this omission.
        if name not in dataset.indexes:
            raise ValueError(f"coords.{name} requires an indexed dimension coordinate.")
        index = dataset.indexes[name]
        if not index.is_unique or index.hasnans:
            raise ValueError(f"coords.{name} requires unique labels without missing values.")


def merged_site_schema(
    *, split_by_sectors: bool = False, require_footprints: bool = False,
    use_bc: bool = False, grid_dims: tuple[str, str] = ("lat", "lon"),
    concentration_units: str | None = None, flux_sources: Sequence[str] | None = None,
) -> DatasetSchema:
    """Describe one acquired site before errors, filtering and basis projection.

    Named dimension permutations and extra variables/attributes are accepted.
    Missing error, response and directional-BC unit attributes inherit the mf
    numerical convention; explicit attributes must have the same scale. This
    checks metadata only, permits empty sites and missing payload values, and
    never supplies attributes or converts values. Grid order is latitude then
    longitude for the ordinary two-dimensional footprint grid.

    Custom callbacks express uncertainty alternatives and unit/source relations;
    those checks are not included by xarray-validate's schema export.

    Args:
        split_by_sectors: Select a source-resolved response instead of a total response.
        require_footprints: Require raw footprints for generated basis weights.
        use_bc: Require all four directional boundary contributions.
        grid_dims: Native spatial axis names, latitude then longitude.
        concentration_units: Optional reference scale for mf. None checks the
            site's units without comparing them to another site's scale.
        flux_sources: Optional authoritative set of sector response labels.
            None allows any valid source labels, retaining their order.

    Returns:
        Metadata schema accepting extra variables, attributes and coordinates.
    """
    # Truth-testing a malformed lazy scalar here would execute its graph.
    for name, flag in (("split_by_sectors", split_by_sectors), ("require_footprints", require_footprints), ("use_bc", use_bc)):
        if type(flag) is not bool:
            raise ValueError(f"{name} must be a Python bool.")
    response = "fp_x_flux_sectoral" if split_by_sectors else "fp_x_flux"
    response_dims = (*grid_dims, "time", *(("source",) if split_by_sectors else ()))
    data_vars = {
        "mf": DataArraySchema(dtype=_REAL_DTYPES, dims=("time",), attrs={"units": AttrSchema(type=str)}),
        # Regex keys describe optional fields without requiring every component.
        "{mf_error|mf_repeatability|mf_variability}": DataArraySchema(dtype=_REAL_DTYPES, dims=("time",)),
        response: DataArraySchema(dtype=_REAL_DTYPES, dims=DimsSchema(response_dims, ordered=False)),
    }
    coordinate_schemas = {
        "time": DataArraySchema(dtype=np.datetime64, dims=("time",)),
        **{dim: DataArraySchema(dtype=_REAL_DTYPES, dims=(dim,)) for dim in grid_dims},
    }
    if require_footprints:
        data_vars["fp"] = DataArraySchema(dtype=_REAL_DTYPES, dims=DimsSchema((*grid_dims, "time"), ordered=False))
    if split_by_sectors:
        coordinate_schemas["source"] = DataArraySchema(dims=("source",))
    if use_bc:
        for direction in "nesw":
            edge = grid_dims[1] if direction in "ns" else grid_dims[0]
            data_vars[f"bc_{direction}"] = DataArraySchema(
                dtype=_REAL_DTYPES, dims=DimsSchema((edge, "height", "time"), ordered=False),
            )
        coordinate_schemas["height"] = DataArraySchema(dtype=_REAL_DTYPES, dims=("height",))

    def relationships(dataset: xr.Dataset) -> None:
        if not any(name in dataset for name in _UNCERTAINTIES):
            raise ValueError("data_vars requires mf_error or at least one of mf_repeatability and mf_variability.")
        _indexed_coordinates(dataset, tuple(coordinate_schemas))
        if "mf" in dataset:
            units = dataset.mf.attrs.get("units")
            if not isinstance(units, str) or not units.strip():
                raise ValueError("data_vars.mf.attrs.units must be a nonempty concentration unit string.")
            scale = mole_fraction_unit_scale(units, context="merged data_vars.mf.attrs.units")
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("data_vars.mf.attrs.units must have a finite positive mole-fraction scale.")
            if concentration_units is not None:
                _same_unit_scale(units, concentration_units, name="data_vars.mf")
            for name in (*_UNCERTAINTIES, response, *((f"bc_{d}" for d in "nesw") if use_bc else ())):
                if name in dataset and "units" in dataset[name].attrs:
                    _same_unit_scale(dataset[name].attrs["units"], units, name=f"data_vars.{name}")
        if split_by_sectors and "source" in dataset.indexes:
            labels = tuple(dataset.indexes["source"])
            if any(not isinstance(label, str) or not label for label in labels):
                raise ValueError("coords.source requires nonempty string labels.")
            if flux_sources is not None and set(labels) != set(flux_sources):
                raise ValueError(f"coords.source must identify exactly the requested flux sources {tuple(flux_sources)!r}.")

    return DatasetSchema(data_vars=data_vars, coords=CoordsSchema(coordinate_schemas), checks=[relationships])


def merged_flux_schema(*, timed: bool, grid_dims: tuple[str, str] = ("lat", "lon")) -> DatasetSchema:
    """Describe native inventory flux, independent of observation times."""
    dims = (*grid_dims, *(("time",) if timed else ()))
    coords = {dim: DataArraySchema(dtype=_REAL_DTYPES, dims=(dim,)) for dim in grid_dims}
    if timed:
        coords["time"] = DataArraySchema(dtype=np.datetime64, dims=("time",))

    def units(dataset: xr.Dataset) -> None:
        _indexed_coordinates(dataset, tuple(coords))
        if "flux" in dataset:
            _same_unit_scale(dataset.flux.attrs.get("units"), "mol m-2 s-1", name="data_vars.flux")

    return DatasetSchema(
        data_vars={"flux": DataArraySchema(dtype=_REAL_DTYPES, dims=DimsSchema(dims, ordered=False))},
        coords=CoordsSchema(coords), checks=[units],
    )


def validate_merged_dataset(schema: DatasetSchema, dataset: xr.Dataset, *, path: str) -> None:
    """Include site/source and variable paths in structural and policy failures."""
    context = ValidationContext(mode="lazy")  # Error collection, never Dask execution.
    try:
        schema.validate(dataset, context=context)
    except ValueError as error:
        summary = f"{context.result.get_error_summary()}\n" if context.has_errors else ""
        raise ValueError(f"{path}: {summary}{error}") from error
    if context.has_errors:
        raise ValueError(f"{path}: {context.result.get_error_summary()}")
