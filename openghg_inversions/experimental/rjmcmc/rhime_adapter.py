"""Adapt filtered RHIME fine-grid inputs to the numerical RJMCMC core.

The public adapter deliberately retains fine-grid sensitivity rather than a
fixed-basis design matrix. It transposes sensitivity to
``(nmeasure, lat, lon)`` and flattens the spatial axes in C order, with
longitude varying fastest. Optional always-active design columns and fixed
offsets are consumed only when their data-variable names are supplied
explicitly; the adapter never discovers boundary-condition fields implicitly.
Inferred model error and multi-sector inputs remain outside this integration
seam.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
import xarray as xr

from openghg_inversions.experimental.rjmcmc.core import (
    FixedDesignBlock,
    TransDimensionalProblem,
    uniform_log_k_prior,
)

_SENSITIVITY_DIMS = ("nmeasure", "lat", "lon")


def _require_variable(dataset: xr.Dataset, name: str) -> xr.DataArray:
    """Return a required data variable or raise a targeted validation error."""
    if name not in dataset.data_vars:
        raise ValueError(f"RHIME inversion inputs must contain data variable {name!r}.")
    return dataset[name]


def _require_exact_dims(array: xr.DataArray, name: str, expected: tuple[str, ...]) -> None:
    """Require a data array to contain exactly the declared dimensions."""
    actual = tuple(str(dimension) for dimension in array.dims)
    if len(actual) == len(expected) and set(actual) == set(expected):
        return

    missing = tuple(dimension for dimension in expected if dimension not in actual)
    extra = tuple(dimension for dimension in actual if dimension not in expected)
    raise ValueError(
        f"{name!r} must have exactly dimensions {expected} in any order; "
        f"received {actual}, missing={missing}, extra={extra}."
    )


def _numeric_values(array: xr.DataArray, name: str) -> NDArray[np.float64]:
    """Convert one input array to float64 with a variable-specific error."""
    try:
        return np.asarray(array.values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name!r} must contain numeric values.") from error


def _grid_axis_values(sensitivity: xr.DataArray, dimension: str) -> NDArray[np.float64]:
    """Return a finite, unique one-dimensional coordinate for a grid axis."""
    if dimension not in sensitivity.coords:
        raise ValueError(f"Fine-grid sensitivity must define a {dimension!r} dimension coordinate.")
    coordinate = sensitivity.coords[dimension]
    if coordinate.dims != (dimension,):
        raise ValueError(f"The {dimension!r} coordinate must be one-dimensional along {dimension!r}.")
    values = _numeric_values(coordinate, dimension)
    if values.shape != (sensitivity.sizes[dimension],):
        raise ValueError(f"The {dimension!r} coordinate length must match its dimension size.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"The {dimension!r} coordinate must contain only finite values.")
    if np.unique(values).size != values.size:
        raise ValueError(f"The {dimension!r} coordinate values must be unique.")
    return values


def _fixed_design_dimension(design: xr.DataArray, name: str) -> str:
    """Return the coefficient dimension of a two-dimensional fixed design."""
    actual = tuple(str(dimension) for dimension in design.dims)
    if design.ndim != 2 or actual.count("nmeasure") != 1:
        raise ValueError(
            f"{name!r} must have exactly two dimensions including 'nmeasure'; received {actual}."
        )
    coefficient_dimension = actual[0] if actual[1] == "nmeasure" else actual[1]
    if design.sizes[coefficient_dimension] < 1:
        raise ValueError(f"{name!r} must contain at least one fixed-design column.")
    return coefficient_dimension


def _align_nmeasure_exact(
    reference: xr.DataArray,
    candidate: xr.DataArray,
    candidate_name: str,
) -> xr.DataArray:
    """Align a candidate exactly to the reference measurement index."""
    excluded_dimensions = (set(reference.dims) | set(candidate.dims)) - {"nmeasure"}
    try:
        _, aligned = xr.align(
            reference,
            candidate,
            join="exact",
            copy=False,
            exclude=excluded_dimensions,
        )
    except ValueError as error:
        raise ValueError(
            f"{candidate_name!r} must align exactly with fine-grid sensitivity along 'nmeasure'."
        ) from error
    return aligned


def _fixed_prior_moments(
    values: ArrayLike,
    *,
    name: str,
    n_columns: int,
) -> NDArray[np.float64]:
    """Normalize scalar or per-column fixed-coefficient prior moments."""
    try:
        moments = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values.") from error
    if moments.ndim == 0:
        moments = np.full(n_columns, float(moments), dtype=np.float64)
    elif moments.shape == (n_columns,):
        moments = np.array(moments, dtype=np.float64, copy=True)
    else:
        raise ValueError(f"{name} must be a scalar or have one value per fixed-design column.")
    if not np.all(np.isfinite(moments)) or np.any(moments <= 0.0):
        raise ValueError(f"{name} must contain only finite, strictly positive values.")
    return moments


def problem_from_rhime_inputs(
    inv_inputs: xr.Dataset,
    *,
    k_min: int,
    k_max: int,
    coefficient_prior_mean: float,
    coefficient_prior_sd: float,
    log_k_prior: ArrayLike | None = None,
    sensitivity_name: str = "fp_x_flux",
    observation_name: str = "mf",
    observation_sd_name: str = "mf_error",
    fixed_design_name: str | None = None,
    fixed_offset_name: str | None = None,
    fixed_coefficient_prior_mean: ArrayLike | None = None,
    fixed_coefficient_prior_sd: ArrayLike | None = None,
) -> TransDimensionalProblem:
    """Build a single-sector RJMCMC problem from filtered RHIME inputs.

    The adapter deliberately consumes fine-grid sensitivity rather than a
    pre-aggregated basis design. Sensitivity is transposed to
    ``(nmeasure, lat, lon)`` and flattened in C order, so longitude varies
    fastest within each latitude row. Observation, error, and any explicitly
    selected optional arrays must already describe the same filtered
    ``nmeasure`` axis. A selected fixed design is transposed to
    ``(nmeasure, n_fixed)``. Boundary-condition variables are never selected
    automatically. Model-error parameters and multi-sector sensitivities are
    outside this integration seam.

    Args:
        inv_inputs: Filtered prepared RHIME dataset containing fine-grid
            sensitivity, observations, and fixed observation errors.
        k_min: Minimum number of active Voronoi regions.
        k_max: Maximum number of active Voronoi regions.
        coefficient_prior_mean: Arithmetic mean of the lognormal coefficient
            prior.
        coefficient_prior_sd: Arithmetic standard deviation of the lognormal
            coefficient prior.
        log_k_prior: Optional normalized log probabilities for every supported
            active-region count. A discrete-uniform prior is used when omitted.
        sensitivity_name: Fine-grid sensitivity data-variable name.
        observation_name: Observation data-variable name.
        observation_sd_name: Fixed observation-error data-variable name.
        fixed_design_name: Optional name of an explicitly selected always-active
            two-dimensional design with one ``nmeasure`` axis.
        fixed_offset_name: Optional name of an explicitly selected additive
            prediction offset along ``nmeasure``.
        fixed_coefficient_prior_mean: Positive arithmetic means for fixed
            lognormal coefficient priors. Required with ``fixed_design_name``;
            a scalar is broadcast and a vector must have one value per column.
        fixed_coefficient_prior_sd: Positive arithmetic standard deviations for
            fixed lognormal coefficient priors, with the same required scalar
            or per-column semantics as the means.

    Returns:
        An immutable numerical problem with flattened fine-grid sensitivity and
        matching ``(latitude, longitude)`` grid coordinates.

    Raises:
        TypeError: If ``inv_inputs`` is not an xarray dataset.
        ValueError: If required variables, dimensions, coordinates, alignment,
            or numerical values are malformed.
    """
    if not isinstance(inv_inputs, xr.Dataset):
        raise TypeError("inv_inputs must be an xarray.Dataset.")

    sensitivity = _require_variable(inv_inputs, sensitivity_name)
    observations = _require_variable(inv_inputs, observation_name)
    observation_sd = _require_variable(inv_inputs, observation_sd_name)
    _require_exact_dims(sensitivity, sensitivity_name, _SENSITIVITY_DIMS)
    _require_exact_dims(observations, observation_name, ("nmeasure",))
    _require_exact_dims(observation_sd, observation_sd_name, ("nmeasure",))

    sensitivity = sensitivity.transpose(*_SENSITIVITY_DIMS)
    try:
        sensitivity, observations, observation_sd = xr.align(
            sensitivity,
            observations,
            observation_sd,
            join="exact",
            copy=False,
        )
    except ValueError as error:
        raise ValueError(
            f"{observation_name!r} and {observation_sd_name!r} must align exactly "
            f"with {sensitivity_name!r} along 'nmeasure'."
        ) from error

    fixed_block = None
    if fixed_design_name is None:
        if fixed_coefficient_prior_mean is not None or fixed_coefficient_prior_sd is not None:
            raise ValueError("fixed coefficient prior moments require an explicit fixed_design_name.")
    else:
        if fixed_coefficient_prior_mean is None or fixed_coefficient_prior_sd is None:
            raise ValueError(
                "fixed_design_name requires both fixed_coefficient_prior_mean and fixed_coefficient_prior_sd."
            )
        fixed_design = _require_variable(inv_inputs, fixed_design_name)
        coefficient_dimension = _fixed_design_dimension(fixed_design, fixed_design_name)
        fixed_design = _align_nmeasure_exact(sensitivity, fixed_design, fixed_design_name)
        fixed_design = fixed_design.transpose("nmeasure", coefficient_dimension)
        fixed_design_values = _numeric_values(fixed_design, fixed_design_name)
        if not np.all(np.isfinite(fixed_design_values)):
            raise ValueError(f"{fixed_design_name!r} must contain only finite values.")
        n_fixed = fixed_design.sizes[coefficient_dimension]
        fixed_block = FixedDesignBlock(
            design=fixed_design_values,
            coefficient_prior_mean=_fixed_prior_moments(
                fixed_coefficient_prior_mean,
                name="fixed_coefficient_prior_mean",
                n_columns=n_fixed,
            ),
            coefficient_prior_sd=_fixed_prior_moments(
                fixed_coefficient_prior_sd,
                name="fixed_coefficient_prior_sd",
                n_columns=n_fixed,
            ),
        )

    fixed_offset = None
    if fixed_offset_name is not None:
        fixed_offset_array = _require_variable(inv_inputs, fixed_offset_name)
        _require_exact_dims(fixed_offset_array, fixed_offset_name, ("nmeasure",))
        fixed_offset_array = _align_nmeasure_exact(
            sensitivity,
            fixed_offset_array,
            fixed_offset_name,
        )
        fixed_offset = _numeric_values(fixed_offset_array, fixed_offset_name)
        if not np.all(np.isfinite(fixed_offset)):
            raise ValueError(f"{fixed_offset_name!r} must contain only finite values.")

    latitudes = _grid_axis_values(sensitivity, "lat")
    longitudes = _grid_axis_values(sensitivity, "lon")
    latitude_grid, longitude_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    grid_coordinates = np.column_stack(
        (latitude_grid.reshape(-1, order="C"), longitude_grid.reshape(-1, order="C"))
    )
    sensitivity_values = _numeric_values(sensitivity, sensitivity_name)
    fine_grid_design = sensitivity_values.reshape(sensitivity.sizes["nmeasure"], -1, order="C")
    selected_log_k_prior = (
        uniform_log_k_prior(k_min, k_max)
        if log_k_prior is None
        else np.asarray(log_k_prior, dtype=np.float64)
    )

    return TransDimensionalProblem(
        observations=_numeric_values(observations, observation_name),
        observation_sd=_numeric_values(observation_sd, observation_sd_name),
        sensitivities=fine_grid_design,
        grid_coordinates=grid_coordinates,
        k_min=k_min,
        k_max=k_max,
        log_k_prior=selected_log_k_prior,
        coefficient_prior_mean=coefficient_prior_mean,
        coefficient_prior_sd=coefficient_prior_sd,
        fixed_offset=fixed_offset,
        fixed_block=fixed_block,
    )
