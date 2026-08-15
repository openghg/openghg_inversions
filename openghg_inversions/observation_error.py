"""Backend-neutral aggregation-error covariance contracts.

Aggregation error is fixed input data, separate from measurement error and
the inferred RHIME model-error term.  Prepared inversion inputs may represent
it exactly as a dense covariance, efficiently as a low-rank-plus-diagonal
covariance, or diagnostically as independent standard deviations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import xarray as xr

AggregationErrorMode: TypeAlias = Literal["auto", "none", "dense", "low_rank", "diagonal"]

AGGREGATION_ERROR_SD = "aggregation_error_sd"
AGGREGATION_ERROR_COVARIANCE = "aggregation_error_covariance"
LOW_RANK_FACTOR = "low_rank_factor"
DIAGONAL_RESIDUAL_VARIANCE = "diagonal_residual_variance"


@dataclass(frozen=True)
class AggregationError:
    """Validated aggregation-error representation selected for a likelihood."""

    mode: Literal["none", "dense", "low_rank", "diagonal"]
    marginal_variance: np.ndarray
    covariance: xr.DataArray | None = None
    factor: xr.DataArray | None = None
    diagonal_variance: xr.DataArray | None = None


def _numeric_finite(name: str, array: xr.DataArray) -> np.ndarray:
    values = np.asarray(array.values)
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"Aggregation-error input {name!r} must be numeric.")
    if not np.isfinite(values).all():
        raise ValueError(f"Aggregation-error input {name!r} must contain only finite values.")
    return values


def _validate_vector(
    data: xr.Dataset,
    name: str,
    *,
    output_dim: str,
    nonnegative: bool = True,
) -> tuple[xr.DataArray, np.ndarray]:
    array = data[name]
    if array.dims != (output_dim,):
        raise ValueError(
            f"Aggregation-error input {name!r} must have dims ({output_dim!r},); "
            f"got {array.dims!r}."
        )
    if array.sizes[output_dim] != data.sizes[output_dim]:
        raise ValueError(f"Aggregation-error input {name!r} is not observation-aligned.")
    values = _numeric_finite(name, array)
    if nonnegative and (values < 0).any():
        raise ValueError(f"Aggregation-error input {name!r} must contain only non-negative values.")
    return array, values


def validate_observation_error_inputs(
    data: xr.Dataset,
    *,
    output_dim: str = "nmeasure",
) -> None:
    """Validate observations and independent diagonal error inputs."""
    if "mf" not in data:
        raise ValueError("Canonical inversion inputs must contain 'mf'.")
    if data["mf"].dims != (output_dim,):
        raise ValueError(
            f"Canonical inversion input 'mf' must have dims ({output_dim!r},); got {data['mf'].dims!r}."
        )
    missing = [name for name in ("mf_error", "min_error") if name not in data]
    if missing:
        raise ValueError(f"Canonical inversion inputs are missing error component(s): {missing!r}.")
    for name in ("mf_error", "min_error"):
        _validate_vector(data, name, output_dim=output_dim)


def _select_aggregation_error_mode(
    data: xr.Dataset, requested: AggregationErrorMode
) -> Literal["none", "dense", "low_rank", "diagonal"]:
    """Select the requested aggregation-error representation without materializing it."""
    if requested not in ("auto", "none", "dense", "low_rank", "diagonal"):
        raise ValueError(
            "`aggregation_error_mode` must be one of 'auto', 'none', 'dense', "
            f"'low_rank', or 'diagonal'; got {requested!r}."
        )
    if requested != "auto":
        return requested

    dense = AGGREGATION_ERROR_COVARIANCE in data
    low_rank = LOW_RANK_FACTOR in data or DIAGONAL_RESIDUAL_VARIANCE in data
    if dense and low_rank:
        raise ValueError(
            "Prepared inputs contain both dense and low-rank aggregation-error covariance; "
            "set `aggregation_error_mode` explicitly."
        )
    if dense:
        return "dense"
    if low_rank:
        return "low_rank"
    if AGGREGATION_ERROR_SD in data:
        return "diagonal"
    return "none"


def resolve_aggregation_error(
    data: xr.Dataset,
    mode: AggregationErrorMode = "auto",
    *,
    output_dim: str = "nmeasure",
    covariance_dim: str = "nmeasure_cov",
) -> AggregationError:
    """Validate and select an aggregation-error covariance representation.

    In ``"auto"`` mode, a structured representation takes precedence over
    ``aggregation_error_sd`` because that vector is commonly retained as a
    marginal diagnostic beside the exact covariance.  Supplying both dense and
    low-rank forms is ambiguous and therefore requires an explicit selection.
    """
    if output_dim not in data.dims:
        raise ValueError(f"Prepared inputs have no observation dimension {output_dim!r}.")
    selected = _select_aggregation_error_mode(data, mode)
    nmeasure = data.sizes[output_dim]

    if selected == "none":
        return AggregationError(mode="none", marginal_variance=np.zeros(nmeasure))

    if selected == "diagonal":
        if AGGREGATION_ERROR_SD not in data:
            raise ValueError(
                f"Diagonal aggregation error requires {AGGREGATION_ERROR_SD!r} in prepared inputs."
            )
        standard_deviation, values = _validate_vector(
            data, AGGREGATION_ERROR_SD, output_dim=output_dim
        )
        return AggregationError(
            mode="diagonal",
            marginal_variance=values**2,
            diagonal_variance=standard_deviation**2,
        )

    if selected == "dense":
        if AGGREGATION_ERROR_COVARIANCE not in data:
            raise ValueError(
                f"Dense aggregation error requires {AGGREGATION_ERROR_COVARIANCE!r} in prepared inputs."
            )
        covariance = data[AGGREGATION_ERROR_COVARIANCE]
        if covariance.dims != (output_dim, covariance_dim):
            raise ValueError(
                f"Aggregation-error input {AGGREGATION_ERROR_COVARIANCE!r} must have dims "
                f"({output_dim!r}, {covariance_dim!r}); got {covariance.dims!r}."
            )
        if covariance.shape != (nmeasure, nmeasure):
            raise ValueError(
                f"Aggregation-error input {AGGREGATION_ERROR_COVARIANCE!r} must be square and "
                f"match {output_dim!r}; got shape {covariance.shape!r}."
            )
        values = _numeric_finite(AGGREGATION_ERROR_COVARIANCE, covariance)
        scale = max(float(np.max(np.abs(values))), 1.0)
        tolerance = 1e-10 * scale
        if not np.allclose(values, values.T, rtol=1e-10, atol=tolerance):
            raise ValueError(f"Aggregation-error input {AGGREGATION_ERROR_COVARIANCE!r} must be symmetric.")
        if float(np.linalg.eigvalsh(values).min()) < -tolerance:
            raise ValueError(
                f"Aggregation-error input {AGGREGATION_ERROR_COVARIANCE!r} must be positive semidefinite."
            )
        marginal_variance = np.diag(values).copy()
        _validate_marginal_sd(data, marginal_variance, output_dim=output_dim)
        return AggregationError(
            mode="dense",
            marginal_variance=marginal_variance,
            covariance=covariance,
        )

    missing = [
        name for name in (LOW_RANK_FACTOR, DIAGONAL_RESIDUAL_VARIANCE) if name not in data
    ]
    if missing:
        raise ValueError(f"Low-rank aggregation error is missing input(s): {missing!r}.")
    factor = data[LOW_RANK_FACTOR]
    if factor.ndim != 2 or factor.dims[0] != output_dim:
        raise ValueError(
            f"Aggregation-error input {LOW_RANK_FACTOR!r} must be a two-dimensional array "
            f"whose first dimension is {output_dim!r}; got {factor.dims!r}."
        )
    if factor.sizes[output_dim] != nmeasure:
        raise ValueError(f"Aggregation-error input {LOW_RANK_FACTOR!r} is not observation-aligned.")
    if factor.shape[1] < 1:
        raise ValueError(f"Aggregation-error input {LOW_RANK_FACTOR!r} must contain at least one rank column.")
    factor_values = _numeric_finite(LOW_RANK_FACTOR, factor)
    diagonal, diagonal_values = _validate_vector(
        data, DIAGONAL_RESIDUAL_VARIANCE, output_dim=output_dim
    )
    marginal_variance = np.sum(factor_values**2, axis=1) + diagonal_values
    _validate_marginal_sd(data, marginal_variance, output_dim=output_dim)
    return AggregationError(
        mode="low_rank",
        marginal_variance=marginal_variance,
        factor=factor,
        diagonal_variance=diagonal,
    )


def _validate_marginal_sd(
    data: xr.Dataset,
    marginal_variance: np.ndarray,
    *,
    output_dim: str,
) -> None:
    """Validate an optional marginal-SD diagnostic beside structured input."""
    if AGGREGATION_ERROR_SD not in data:
        return
    _, values = _validate_vector(data, AGGREGATION_ERROR_SD, output_dim=output_dim)
    expected = np.sqrt(marginal_variance)
    if not np.allclose(values, expected, rtol=1e-6, atol=1e-12):
        raise ValueError(
            f"Aggregation-error diagnostic {AGGREGATION_ERROR_SD!r} must equal the square root "
            "of the selected covariance diagonal."
        )
