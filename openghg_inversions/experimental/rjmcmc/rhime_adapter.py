"""Adapt filtered RHIME fine-grid inputs to the numerical TD-MCMC core.

The public adapter deliberately retains fine-grid sensitivity rather than a
fixed-basis design matrix. It transposes sensitivity to
``(nmeasure, lat, lon)`` and flattens the spatial axes in C order, with
longitude varying fastest. Boundary-condition terms, inferred model error,
and multi-sector inputs remain outside this initial integration seam.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
import xarray as xr

from openghg_inversions.experimental.rjmcmc.core import TransDimensionalProblem, uniform_log_k_prior

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
) -> TransDimensionalProblem:
    """Build a single-sector TD-MCMC problem from filtered RHIME inputs.

    The adapter deliberately consumes fine-grid sensitivity rather than a
    pre-aggregated basis design. Sensitivity is transposed to
    ``(nmeasure, lat, lon)`` and flattened in C order, so longitude varies
    fastest within each latitude row. Observation and error arrays must already
    describe the same filtered ``nmeasure`` axis. Boundary-condition terms,
    model-error parameters, and multi-sector sensitivities are outside this
    initial integration seam.

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
    )
