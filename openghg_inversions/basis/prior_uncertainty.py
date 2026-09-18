"""Basis-aware projection and calibration of scale-factor prior uncertainty.

The public helpers in this module operate on retained :class:`BasisFunctions`
objects.  They treat grid-cell scale perturbations as independent and project
their standard deviations onto the operator's labelled state coordinate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

import numpy as np
import xarray as xr

from openghg_inversions.array_ops import concat_gather_data_arrays, force_align
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.basis.operators import (
    BasisOperator,
    MultiSourceBucketBasisOperator,
)

_STATE_VARIABLES = (
    "state_total",
    "state_prior_stdev_numerator",
    "state_status",
    "state_is_active",
    "x_prior_stdev",
    "target_state_total",
)
_STATUS_DTYPE = "<U18"
MEDIAN_RELATIVE_TARGET_STATISTIC = "median-relative"
MEAN_TOTAL_TARGET_STATISTIC = "mean-total"
TargetStatistic = Literal["median-relative", "mean-total"]


def _as_data_array(value: xr.DataArray | float, *, name: str) -> xr.DataArray:
    """Return a named DataArray without discarding caller-provided labels."""
    if isinstance(value, xr.DataArray):
        return value
    return xr.DataArray(float(value), name=name)


def _align_grid(
    array: xr.DataArray,
    reference: xr.DataArray,
    *,
    grid_dims: tuple[str, ...],
    name: str,
    require_grid: bool,
) -> xr.DataArray:
    """Validate grid dimensions and force numerically equivalent coordinates."""
    present = tuple(dim for dim in grid_dims if dim in array.dims)
    if require_grid and present != grid_dims:
        missing = [dim for dim in grid_dims if dim not in array.dims]
        raise ValueError(f"{name} is missing operator grid dimension(s) {missing!r}.")
    if present and present != grid_dims:
        missing = [dim for dim in grid_dims if dim not in array.dims]
        raise ValueError(f"{name} contains only part of the operator grid; missing dimension(s) {missing!r}.")
    if not present:
        return array
    return force_align(array, reference, dims=grid_dims)


def _validate_non_grid_dims(
    array: xr.DataArray,
    *,
    grid_dims: tuple[str, ...],
    name: str,
) -> None:
    """Reject unresolved axes that would create multiple prior projections."""
    extra_dims = [dim for dim in array.dims if dim not in grid_dims]
    if extra_dims:
        raise ValueError(
            f"{name} has unsupported non-grid dimension(s) {extra_dims!r}; "
            "select a single field before projecting prior uncertainty."
        )


def _scalar_bool(value: xr.DataArray) -> bool:
    """Compute one labelled Boolean reduction."""
    return bool(value.compute().item())


def _validate_finite(array: xr.DataArray, *, name: str) -> None:
    """Reject non-finite numeric inputs before moment projection."""
    if _scalar_bool((~np.isfinite(array)).any()):
        raise ValueError(f"{name} must contain only finite values.")


def _validate_nonnegative(array: xr.DataArray, *, name: str) -> None:
    """Reject negative or non-finite scale inputs."""
    _validate_finite(array, name=name)
    if _scalar_bool((array < 0).any()):
        raise ValueError(f"{name} must be non-negative.")


def _stable_weight_scale(weights: xr.DataArray) -> float:
    """Return a finite scale that prevents overflow in squared weights."""
    _validate_finite(weights, name="flux * area_grid")
    scale = float(np.abs(weights).max().compute().item())
    return scale if scale > 0 else 1.0


def _select_source(array: xr.DataArray, source_dim: str, source: str, *, name: str) -> xr.DataArray:
    """Select a labelled source when present, otherwise return a shared field."""
    if source_dim not in array.dims:
        return array
    try:
        return array.sel({source_dim: source}, drop=True)
    except KeyError as exc:
        raise ValueError(f"{name} does not define source {source!r} on {source_dim!r}.") from exc


def _source_contexts(
    basis_functions: BasisFunctions,
    flux: xr.DataArray,
) -> tuple[str | None, list[tuple[str | None, BasisOperator]], bool]:
    """Return source labels/operators and whether their state axis is ragged."""
    operator = basis_functions.operator
    if isinstance(operator, MultiSourceBucketBasisOperator):
        source_dim = operator.source_dim
        source_coord = operator.basis_matrix.coords[source_dim]
        labels = list(dict.fromkeys(str(value) for value in source_coord.values.tolist()))
        contexts = [
            (label, operator.operator_for_source(label, state_dim=operator.region_in_source_dim))
            for label in labels
        ]
        return source_dim, contexts, True

    if "source" in flux.dims:
        labels = [str(value) for value in flux.coords["source"].values.tolist()]
        return "source", [(label, operator) for label in labels], False

    return None, [(None, operator)], False


def _project_one_source(
    operator: BasisOperator,
    *,
    flux: xr.DataArray,
    area_grid: xr.DataArray,
    grid_cell_prior_stdev: xr.DataArray,
) -> xr.Dataset:
    """Project one source while retaining operator state coordinates."""
    grid_dims = operator.meta.grid_dims
    basis_matrix = operator.basis_matrix

    flux = _align_grid(flux, basis_matrix, grid_dims=grid_dims, name="flux", require_grid=True)
    area_grid = _align_grid(
        area_grid,
        basis_matrix,
        grid_dims=grid_dims,
        name="area_grid",
        require_grid=True,
    )
    grid_cell_prior_stdev = _align_grid(
        grid_cell_prior_stdev,
        basis_matrix,
        grid_dims=grid_dims,
        name="grid_cell_prior_stdev",
        require_grid=False,
    )
    _validate_non_grid_dims(flux, grid_dims=grid_dims, name="flux")
    _validate_non_grid_dims(area_grid, grid_dims=grid_dims, name="area_grid")
    _validate_non_grid_dims(
        grid_cell_prior_stdev,
        grid_dims=grid_dims,
        name="grid_cell_prior_stdev",
    )
    _validate_finite(flux, name="flux")
    _validate_nonnegative(area_grid, name="area_grid")
    _validate_nonnegative(grid_cell_prior_stdev, name="grid_cell_prior_stdev")

    weights = flux * area_grid
    weight_scale = _stable_weight_scale(weights)
    scaled_weights = weights / weight_scale
    scaled_state_total = xr.dot(scaled_weights, basis_matrix, dim=list(grid_dims))
    scaled_state_variance_numerator = xr.dot(
        (scaled_weights * grid_cell_prior_stdev) ** 2,
        basis_matrix,
        dim=list(grid_dims),
    )
    scaled_state_variance_numerator = xr.where(
        scaled_state_variance_numerator >= 0,
        scaled_state_variance_numerator,
        0.0,
    )
    scaled_state_prior_stdev_numerator = np.sqrt(scaled_state_variance_numerator)
    state_total = (scaled_state_total * weight_scale).rename("state_total")
    state_prior_stdev_numerator = (scaled_state_prior_stdev_numerator * weight_scale).rename(
        "state_prior_stdev_numerator"
    )

    nonzero_total = np.abs(scaled_state_total) > 0
    safe_total = xr.where(nonzero_total, np.abs(scaled_state_total), 1.0)
    x_prior_stdev = xr.where(
        nonzero_total,
        scaled_state_prior_stdev_numerator / safe_total,
        xr.where(scaled_state_prior_stdev_numerator == 0, 0.0, np.nan),
    ).rename("x_prior_stdev")
    state_status = xr.where(
        ~np.isfinite(state_total) | ~np.isfinite(state_prior_stdev_numerator),
        np.asarray("nonfinite", dtype=_STATUS_DTYPE),
        xr.where(
            nonzero_total,
            "ok",
            xr.where(scaled_state_prior_stdev_numerator == 0, "zero", "cancellation"),
        ),
    ).rename("state_status")

    return xr.Dataset(
        {
            "state_total": state_total,
            "state_prior_stdev_numerator": state_prior_stdev_numerator,
            "state_status": state_status,
            "x_prior_stdev": x_prior_stdev,
        }
    )


def _apply_state_activity(
    projection: xr.Dataset,
    operator: BasisOperator,
    state_is_active: xr.DataArray | None,
) -> xr.Dataset:
    """Mask inactive state widths so diagnostics match the sampled model."""
    state_dim = operator.meta.state_dim
    valid_width = np.isfinite(projection["x_prior_stdev"]) & (projection["x_prior_stdev"] > 0)
    if state_is_active is None:
        active = valid_width.rename("state_is_active")
    else:
        if state_is_active.dtype != np.dtype(bool):
            raise ValueError("state_is_active must be Boolean.")
        if state_is_active.dims != (state_dim,):
            raise ValueError(f"state_is_active must have exactly the {state_dim!r} dimension.")
        expected_index = projection["x_prior_stdev"].get_index(state_dim)
        provided_index = state_is_active.get_index(state_dim)
        if len(expected_index) != len(provided_index) or not expected_index.isin(provided_index).all():
            raise ValueError("state_is_active labels do not match the basis state labels.")
        active = state_is_active.sel({state_dim: projection["x_prior_stdev"].coords[state_dim]})
        if _scalar_bool((active & ~valid_width).any()):
            raise ValueError(
                "state_is_active selects a state whose projected prior width is zero or non-finite."
            )
        active = active.rename("state_is_active")

    return projection.assign(
        x_prior_stdev=xr.where(active, projection["x_prior_stdev"], 0.0),
        state_prior_stdev_numerator=xr.where(
            active,
            projection["state_prior_stdev_numerator"],
            0.0,
        ),
        state_status=xr.where(
            active,
            projection["state_status"],
            xr.where(valid_width, "inactive", projection["state_status"]),
        ),
        state_is_active=active,
    )


def _select_ragged_state_activity(
    state_is_active: xr.DataArray,
    *,
    operator: MultiSourceBucketBasisOperator,
    source_operator: BasisOperator,
    source: str,
) -> xr.DataArray:
    """Select one source from a canonical gathered-state activity mask."""
    state_dim = operator.meta.state_dim
    source_dim = operator.source_dim
    source_state_dim = source_operator.meta.state_dim
    if state_is_active.dims != (state_dim,):
        return state_is_active
    if source_dim not in state_is_active.coords or source_state_dim not in state_is_active.coords:
        raise ValueError(
            "A gathered ragged state_is_active mask must retain its source and "
            f"{source_state_dim!r} MultiIndex level coordinates."
        )

    selected = state_is_active.where(state_is_active.coords[source_dim] == source, drop=True)
    labels = np.asarray(selected.coords[source_state_dim].values)
    return xr.DataArray(
        selected.data.astype(bool),
        dims=(source_state_dim,),
        coords={source_state_dim: labels},
        name="state_is_active",
    )


def _add_target_diagnostics(
    projection: xr.Dataset,
    operator: BasisOperator,
    *,
    flux: xr.DataArray,
    area_grid: xr.DataArray,
    target_matrix: xr.DataArray,
) -> xr.Dataset:
    """Add target/state totals and achieved uncertainty diagnostics."""
    grid_dims = operator.meta.grid_dims
    state_dim = operator.meta.state_dim
    basis_matrix = operator.basis_matrix
    if state_dim in target_matrix.dims:
        raise ValueError(
            f"target_matrix dimension {state_dim!r} aliases the basis state dimension; "
            "rename the target dimension before calibration."
        )
    flux = _align_grid(flux, basis_matrix, grid_dims=grid_dims, name="flux", require_grid=True)
    area_grid = _align_grid(
        area_grid,
        basis_matrix,
        grid_dims=grid_dims,
        name="area_grid",
        require_grid=True,
    )
    target_matrix = _align_grid(
        target_matrix,
        basis_matrix,
        grid_dims=grid_dims,
        name="target_matrix",
        require_grid=True,
    )

    weights = flux * area_grid
    target_state_total = xr.dot(
        target_matrix * weights,
        basis_matrix,
        dim=list(grid_dims),
    ).rename("target_state_total")
    target_state_absolute_total = xr.dot(
        np.abs(target_matrix * weights),
        basis_matrix,
        dim=list(grid_dims),
    )
    target_total = target_state_total.sum(dim=state_dim).rename("target_total")
    target_absolute_total = target_state_absolute_total.sum(dim=state_dim).rename("target_absolute_total")

    contributions = xr.where(
        target_state_total == 0,
        0.0,
        target_state_total * projection["x_prior_stdev"],
    )
    achieved_target_stdev = np.sqrt((contributions**2).sum(dim=state_dim)).rename("achieved_target_stdev")
    nonzero_target_total = np.abs(target_total) > 0
    safe_target_total = xr.where(nonzero_target_total, np.abs(target_total), 1.0)
    achieved_target_relative_stdev = xr.where(
        nonzero_target_total,
        achieved_target_stdev / safe_target_total,
        np.nan,
    ).rename("achieved_target_relative_stdev")

    target_status = xr.where(
        ~np.isfinite(target_total) | ~np.isfinite(target_absolute_total),
        np.asarray("nonfinite", dtype=_STATUS_DTYPE),
        xr.where(
            target_absolute_total == 0,
            "zero",
            xr.where(
                target_total == 0,
                "cancellation",
                xr.where(
                    ~np.isfinite(achieved_target_stdev),
                    "state_cancellation",
                    xr.where(achieved_target_stdev == 0, "zero_stdev", "ok"),
                ),
            ),
        ),
    ).rename("target_status")

    return projection.assign(
        target_state_total=target_state_total,
        target_total=target_total,
        target_absolute_total=target_absolute_total,
        achieved_target_stdev=achieved_target_stdev,
        achieved_target_relative_stdev=achieved_target_relative_stdev,
        target_status=target_status,
    )


def _gather_ragged_state_arrays(
    arrays: Mapping[str, xr.DataArray],
    *,
    operator: MultiSourceBucketBasisOperator,
) -> xr.DataArray:
    """Gather per-source arrays into the operator's canonical state order."""
    gathered = concat_gather_data_arrays(
        arrays,
        key_dim=operator.source_dim,
        ragged_dim=operator.region_in_source_dim,
        stack_dim=operator.meta.state_dim,
    )
    canonical_index = operator.basis_matrix.coords[operator.meta.state_dim].to_index()
    if not gathered.coords[operator.meta.state_dim].to_index().equals(canonical_index):
        raise ValueError("Projected ragged state order does not match the retained basis operator.")
    return gathered


def _project_sources(
    basis_functions: BasisFunctions,
    *,
    flux: xr.DataArray,
    area_grid: xr.DataArray,
    grid_cell_prior_stdev: xr.DataArray,
) -> xr.DataArray:
    """Project all source contexts and combine their labelled state axes."""
    source_dim, contexts, ragged = _source_contexts(basis_functions, flux)
    pieces: dict[str, xr.DataArray] = {}
    for source, operator in contexts:
        source_flux = flux
        source_area = area_grid
        source_stdev = grid_cell_prior_stdev
        if source is not None and source_dim is not None:
            source_flux = _select_source(flux, source_dim, source, name="flux")
            source_area = _select_source(area_grid, source_dim, source, name="area_grid")
            source_stdev = _select_source(
                grid_cell_prior_stdev,
                source_dim,
                source,
                name="grid_cell_prior_stdev",
            )
        projected = _project_one_source(
            operator,
            flux=source_flux,
            area_grid=source_area,
            grid_cell_prior_stdev=source_stdev,
        )["x_prior_stdev"]
        if source is None:
            return projected
        pieces[source] = projected

    if source_dim is None:
        raise RuntimeError("Source-labelled projections were produced without a source dimension.")
    if ragged:
        operator = cast(MultiSourceBucketBasisOperator, basis_functions.operator)
        return _gather_ragged_state_arrays(pieces, operator=operator).rename("x_prior_stdev")

    source_coord = xr.IndexVariable(source_dim, list(pieces))
    return xr.concat(list(pieces.values()), dim=source_coord).rename("x_prior_stdev")


def project_basis_prior_stdev(
    basis_functions: BasisFunctions,
    *,
    area_grid: xr.DataArray,
    grid_cell_prior_stdev: xr.DataArray | float,
    flux: xr.DataArray | None = None,
) -> xr.DataArray:
    """Project independent grid-cell scale uncertainty onto basis states.

    For membership matrix ``A`` and cell-total weights ``w = flux * area``,
    the projected standard deviation is

    ``sqrt(sum(A * (w * s)**2)) / abs(sum(A * w))``.

    ``s`` may be scalar, source-labelled, or gridded.  A state with no weighted
    flux receives zero standard deviation.  A state whose signed total cancels
    to zero while its numerator is nonzero receives ``NaN`` because its
    multiplicative scale uncertainty is undefined.

    Args:
        basis_functions: Retained basis artifact and operator.
        area_grid: Grid-cell areas on the operator grid.
        grid_cell_prior_stdev: Independent grid-cell scale-factor standard
            deviation.
        flux: Optional replacement flux.  The retained flux is used by default.

    Returns:
        Labelled state standard deviations named ``x_prior_stdev``.
    """
    selected_flux = basis_functions.flux if flux is None else flux
    stdev = _as_data_array(grid_cell_prior_stdev, name="grid_cell_prior_stdev")
    result = _project_sources(
        basis_functions,
        flux=selected_flux,
        area_grid=area_grid,
        grid_cell_prior_stdev=stdev,
    )
    result.attrs.update(
        {
            "description": "Basis-state scale-factor prior standard deviation.",
            "projection": "independent grid-cell Gaussian moments",
        }
    )
    return result


def _calibration_status(unit_projection: xr.Dataset) -> xr.DataArray:
    """Return a scalar calibration status for one source."""
    relative = unit_projection["achieved_target_relative_stdev"]
    target_status = unit_projection["target_status"]
    valid = (target_status == "ok") & np.isfinite(relative) & (relative > 0)
    return xr.where(
        valid.any(),
        np.asarray("ok", dtype=_STATUS_DTYPE),
        xr.where(
            (target_status == "cancellation").any(),
            "cancellation",
            xr.where(
                (target_status == "state_cancellation").any(),
                "state_cancellation",
                xr.where(
                    (target_status == "zero").all(),
                    "zero",
                    xr.where((target_status == "zero_stdev").any(), "zero_stdev", "nonfinite"),
                ),
            ),
        ),
    ).rename("calibration_status")


def _valid_values(values: xr.DataArray, valid: xr.DataArray) -> np.ndarray:
    """Compute the small target-level reduction input, including dask arrays."""
    selected = np.asarray(values.where(valid).compute().values, dtype=np.float64).reshape(-1)
    return selected[np.isfinite(selected)]


def _unit_calibration_factor(
    unit_projection: xr.Dataset,
    *,
    requested_relative_stdev: float,
    target_statistic: TargetStatistic,
) -> float:
    """Return the linear scale matching the requested aggregate statistic."""
    target_status = unit_projection["target_status"]
    if target_statistic == MEDIAN_RELATIVE_TARGET_STATISTIC:
        relative = unit_projection["achieved_target_relative_stdev"]
        valid = (target_status == "ok") & np.isfinite(relative) & (relative > 0)
        values = _valid_values(relative, valid)
        if not values.size:
            return np.nan
        return requested_relative_stdev / float(np.median(values))

    target_total = np.abs(unit_projection["target_total"])
    unit_stdev = unit_projection["achieved_target_stdev"]
    valid = (
        (target_status == "ok")
        & np.isfinite(target_total)
        & np.isfinite(unit_stdev)
        & (target_total > 0)
        & (unit_stdev > 0)
    )
    totals = _valid_values(target_total, valid)
    stdevs = _valid_values(unit_stdev, valid)
    if not totals.size or not stdevs.size:
        return np.nan
    return requested_relative_stdev * float(np.mean(totals)) / float(np.mean(stdevs))


def _calibrate_one_source(
    operator: BasisOperator,
    *,
    flux: xr.DataArray,
    area_grid: xr.DataArray,
    target_matrix: xr.DataArray,
    target_relative_stdev: xr.DataArray,
    target_statistic: TargetStatistic,
    state_is_active: xr.DataArray | None,
) -> xr.Dataset:
    """Calibrate and diagnose one source."""
    if target_relative_stdev.ndim != 0:
        raise ValueError(
            "target_relative_stdev must be scalar after source selection; "
            "supply one requested relative standard deviation per source."
        )
    requested = float(target_relative_stdev)
    if not np.isfinite(requested) or requested <= 0:
        raise ValueError("target_relative_stdev must be finite and strictly positive.")

    unit = _project_one_source(
        operator,
        flux=flux,
        area_grid=area_grid,
        grid_cell_prior_stdev=xr.DataArray(1.0),
    )
    unit = _apply_state_activity(unit, operator, state_is_active)
    unit = _add_target_diagnostics(
        unit,
        operator,
        flux=flux,
        area_grid=area_grid,
        target_matrix=target_matrix,
    )
    status = _calibration_status(unit)
    calibration_factor = _unit_calibration_factor(
        unit,
        requested_relative_stdev=requested,
        target_statistic=target_statistic,
    )
    grid_stdev = xr.where(status == "ok", calibration_factor, np.nan).rename("grid_cell_prior_stdev")

    calibrated = xr.Dataset(
        {
            "state_total": unit["state_total"],
            "state_prior_stdev_numerator": xr.where(
                unit["state_is_active"],
                unit["state_prior_stdev_numerator"] * grid_stdev,
                0.0,
            ),
            "state_status": unit["state_status"],
            "x_prior_stdev": xr.where(
                unit["state_is_active"],
                unit["x_prior_stdev"] * grid_stdev,
                0.0,
            ),
            "state_is_active": unit["state_is_active"],
        }
    )
    calibrated = _add_target_diagnostics(
        calibrated,
        operator,
        flux=flux,
        area_grid=area_grid,
        target_matrix=target_matrix,
    )
    return calibrated.assign(
        grid_cell_prior_stdev=grid_stdev,
        requested_target_relative_stdev=xr.DataArray(
            requested,
            name="requested_target_relative_stdev",
        ),
        calibration_status=status,
        target_statistic=xr.DataArray(target_statistic, name="target_statistic"),
    )


def _combine_ragged_calibrations(
    pieces: Mapping[str, xr.Dataset],
    *,
    operator: MultiSourceBucketBasisOperator,
) -> xr.Dataset:
    """Combine ragged state variables without conflicting MultiIndex levels."""
    state_arrays = {
        name: _gather_ragged_state_arrays(
            {source: dataset[name] for source, dataset in pieces.items()},
            operator=operator,
        ).rename(name)
        for name in _STATE_VARIABLES
    }
    source_variable_names = [
        name for name in next(iter(pieces.values())).data_vars if name not in _STATE_VARIABLES
    ]
    calibration_source_dim = f"calibration_{operator.source_dim}"
    source_coord = xr.IndexVariable(calibration_source_dim, list(pieces))
    source_datasets = [dataset[source_variable_names] for dataset in pieces.values()]
    source_diagnostics = xr.concat(source_datasets, dim=source_coord)
    result = xr.Dataset(state_arrays).merge(source_diagnostics)
    result.attrs["calibration_source_dimension"] = calibration_source_dim
    result.attrs["state_source_coordinate"] = operator.source_dim
    return result


def calibrate_basis_prior_stdev(
    basis_functions: BasisFunctions,
    *,
    area_grid: xr.DataArray,
    target_matrix: xr.DataArray,
    target_relative_stdev: xr.DataArray | float,
    target_statistic: TargetStatistic = MEDIAN_RELATIVE_TARGET_STATISTIC,
    state_is_active: xr.DataArray | None = None,
    flux: xr.DataArray | None = None,
) -> xr.Dataset:
    """Calibrate grid-cell and basis-state prior widths to aggregate targets.

    Calibration first projects a unit grid-cell standard deviation. For each
    source, linearity then gives the cell standard deviation required to match
    either the median target-relative SD or the ratio of mean target SD to mean
    absolute target total. A scalar target request is shared; a source-labelled
    request is selected by label. With one target per source, either statistic
    matches the requested value exactly.

    ``target_matrix`` may contain any caller-defined target dimensions in
    addition to the operator grid.  No countries, masks, or target percentages
    are built into this API.

    The result includes ``grid_cell_prior_stdev``, ``x_prior_stdev``, state and
    target totals, achieved target standard deviations, achieved relative
    standard deviations, and explicit state/target/calibration status strings.
    Status ``zero`` means the target contains no absolute weighted flux;
    ``cancellation`` means nonzero signed weights sum to a zero target total;
    ``state_cancellation`` means a target depends on a basis state whose own
    signed total cancels to zero.

    Args:
        basis_functions: Retained basis artifact and operator.
        area_grid: Grid-cell areas on the operator grid.
        target_matrix: Caller-defined target masks or weights.
        target_relative_stdev: Requested relative standard deviation, scalar or
            source-labelled.
        target_statistic: Aggregate calibration statistic, either
            ``"median-relative"`` or ``"mean-total"``.
        state_is_active: Optional labelled Boolean state mask. Inactive state
            widths are set to zero and omitted from target uncertainty, matching
            active-state model sampling.
        flux: Optional replacement flux.  The retained flux is used by default.

    Returns:
        Dataset containing calibrated widths and projection diagnostics.
    """
    selected_flux = basis_functions.flux if flux is None else flux
    canonical_state_dim = basis_functions.operator.meta.state_dim
    if canonical_state_dim in target_matrix.dims:
        raise ValueError(
            f"target_matrix dimension {canonical_state_dim!r} aliases the basis state "
            "dimension; rename the target dimension before calibration."
        )
    if target_statistic not in {
        MEDIAN_RELATIVE_TARGET_STATISTIC,
        MEAN_TOTAL_TARGET_STATISTIC,
    }:
        raise ValueError("target_statistic must be 'median-relative' or 'mean-total'.")
    requested = _as_data_array(target_relative_stdev, name="target_relative_stdev")
    source_dim, contexts, ragged = _source_contexts(basis_functions, selected_flux)
    pieces: dict[str, xr.Dataset] = {}

    for source, operator in contexts:
        source_flux = selected_flux
        source_area = area_grid
        source_target = target_matrix
        source_requested = requested
        source_active = state_is_active
        if source is not None and source_dim is not None:
            source_flux = _select_source(selected_flux, source_dim, source, name="flux")
            source_area = _select_source(area_grid, source_dim, source, name="area_grid")
            source_target = _select_source(
                target_matrix,
                source_dim,
                source,
                name="target_matrix",
            )
            source_requested = _select_source(
                requested,
                source_dim,
                source,
                name="target_relative_stdev",
            )
            if state_is_active is not None and source_dim in state_is_active.dims:
                source_active = _select_source(
                    state_is_active,
                    source_dim,
                    source,
                    name="state_is_active",
                )
            elif state_is_active is not None and ragged:
                source_active = _select_ragged_state_activity(
                    state_is_active,
                    operator=cast(
                        MultiSourceBucketBasisOperator,
                        basis_functions.operator,
                    ),
                    source_operator=operator,
                    source=source,
                )
        calibrated = _calibrate_one_source(
            operator,
            flux=source_flux,
            area_grid=source_area,
            target_matrix=source_target,
            target_relative_stdev=source_requested,
            target_statistic=target_statistic,
            state_is_active=source_active,
        )
        if source is None:
            return calibrated
        pieces[source] = calibrated

    if source_dim is None:
        raise RuntimeError("Source-labelled calibrations were produced without a source dimension.")
    if ragged:
        operator = cast(MultiSourceBucketBasisOperator, basis_functions.operator)
        return _combine_ragged_calibrations(pieces, operator=operator)

    source_coord = xr.IndexVariable(source_dim, list(pieces))
    return cast(xr.Dataset, xr.concat(list(pieces.values()), dim=source_coord))
