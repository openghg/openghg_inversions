"""Backend-neutral aggregation-error covariance contracts.

Aggregation error is fixed input data, separate from measurement error and
the inferred RHIME model-error term.  Prepared inversion inputs may represent
it exactly as a dense covariance, efficiently as a low-rank-plus-diagonal
covariance, or diagnostically as independent standard deviations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from dask import compute as dask_compute
import numpy as np
import xarray as xr

AggregationErrorMode: TypeAlias = Literal["auto", "none", "dense", "low_rank", "diagonal"]

AGGREGATION_ERROR_SD = "aggregation_error_sd"
AGGREGATION_ERROR_COVARIANCE = "aggregation_error_covariance"
LOW_RANK_FACTOR = "low_rank_factor"
DIAGONAL_RESIDUAL_VARIANCE = "diagonal_residual_variance"
OBSERVATION_ERROR_INPUT_NAMES = ("mf", "mf_error", "min_error")


@dataclass(frozen=True)
class AggregationError:
    """Validated aggregation-error representation selected for a likelihood.

    Args:
        mode: Concrete covariance representation.
        marginal_variance: Observation-aligned covariance diagonal.
        covariance: Optional dense covariance matrix.
        factor: Optional low-rank covariance factor.
        diagonal_variance: Optional independent residual variance for a
            low-rank representation.
    """

    mode: Literal["none", "dense", "low_rank", "diagonal"]
    marginal_variance: np.ndarray
    covariance: xr.DataArray | None = None
    factor: xr.DataArray | None = None
    diagonal_variance: xr.DataArray | None = None


def _numeric_finite(
    name: str,
    array: xr.DataArray,
    *,
    owner: str = "Aggregation-error input",
) -> np.ndarray:
    """Return numeric finite array values or raise a labelled error.

    Args:
        name: Scientific input name used in diagnostics.
        array: Labelled values to materialize and validate.
        owner: Component label used in diagnostics.

    Returns:
        Materialized NumPy values.

    Raises:
        ValueError: If values are non-numeric or non-finite.
    """
    values = np.asarray(array.values)
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{owner} {name!r} must be numeric.")
    if not np.isfinite(values).all():
        raise ValueError(f"{owner} {name!r} must contain only finite values.")
    return values


def _materialize_together(*arrays: xr.DataArray) -> tuple[xr.DataArray, ...]:
    """Return shallow labelled copies whose related payloads are eager."""
    computed = dask_compute(*(array.data for array in arrays))
    return tuple(
        array.copy(deep=False, data=values)
        for array, values in zip(arrays, computed, strict=True)
    )


def _require_eager_payload(array: xr.DataArray, name: str, *, owner: str) -> None:
    """Reject a lazy public value before validation can compute it repeatedly."""
    if array.chunks is not None:
        raise ValueError(
            f"{owner} input {name!r} must be eager; construct it with "
            "`resolve_aggregation_error()` before building the likelihood."
        )


def _validate_vector(
    data: xr.Dataset,
    name: str,
    *,
    output_dim: str,
    nonnegative: bool = True,
) -> tuple[xr.DataArray, np.ndarray]:
    """Validate one observation-aligned aggregation-error vector.

    Args:
        data: Dataset owning the vector and observation dimension.
        name: Vector variable name.
        output_dim: Required observation dimension.
        nonnegative: Whether negative values are invalid.

    Returns:
        Original labelled vector and its materialized numeric values.

    Raises:
        ValueError: If the vector has invalid dimensions, alignment, values,
            or sign.
    """
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
    """Validate observations and independent diagonal error inputs.

    Args:
        data: Prepared inversion inputs containing ``mf``, ``mf_error``, and
            ``min_error``.
        output_dim: Observation dimension required on each validated input.

    Raises:
        ValueError: If a required input is absent, not an observation-aligned
            vector, non-numeric, non-finite, or negative where an error must
            be non-negative.
    """
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


def validate_observation_error_arrays(
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    minimum_error: xr.DataArray,
    *,
    owner: str,
    output_dim: str = "nmeasure",
) -> None:
    """Validate the named scientific arrays consumed by an error component.

    Args:
        observations: Observed mole fractions.
        observation_error: Reported observation-error standard deviations.
        minimum_error: Minimum total-error standard deviations.
        owner: Name of the likelihood/error component consuming the arrays.
        output_dim: Required observation dimension.

    Raises:
        ValueError: If an input is not an aligned observation vector or an
            error array is non-numeric, non-finite, or negative.
    """
    if observations.dims != (output_dim,):
        raise ValueError(
            f"{owner} input 'observations' must have dims "
            f"({output_dim!r},); got {observations.dims!r}."
        )
    _numeric_finite("observations", observations, owner=f"{owner} input")
    nmeasure = observations.sizes[output_dim]
    for name, array in (
        ("observation_error", observation_error),
        ("minimum_error", minimum_error),
    ):
        if array.dims != (output_dim,):
            raise ValueError(
                f"{owner} input {name!r} must have dims "
                f"({output_dim!r},); got {array.dims!r}."
            )
        if array.sizes[output_dim] != nmeasure:
            raise ValueError(f"{owner} input {name!r} is not observation-aligned.")
        values = _numeric_finite(name, array, owner=f"{owner} input")
        if (values < 0).any():
            raise ValueError(f"{owner} input {name!r} must contain only non-negative values.")


def validate_observation_alignment(
    observations: xr.DataArray,
    array: xr.DataArray,
    *,
    input_name: str,
    owner: str,
    output_dim: str = "nmeasure",
) -> None:
    """Require an input to carry the observations' exact indexed ordering.

    The indexed dimension coordinate is eager structural data. Comparing its
    values does not materialize either scientific payload, and deliberately
    ignores coordinate attributes and unrelated auxiliary coordinates.

    Args:
        observations: Reference observation vector.
        array: Scientific input containing the observation dimension.
        input_name: Input name used in diagnostics.
        owner: Component name used in diagnostics.
        output_dim: Shared observation dimension.

    Raises:
        ValueError: If the input omits the observation dimension or its
            observation labels differ from ``observations``.
    """
    if output_dim not in array.dims:
        raise ValueError(f"{owner} input {input_name!r} has no {output_dim!r} dimension.")
    reference_index = observations.get_index(output_dim)
    candidate_index = array.get_index(output_dim)
    if not reference_index.equals(candidate_index):
        raise ValueError(
            f"{owner} input {input_name!r} has incompatible observation coordinate {output_dim!r}."
        )


def validate_aggregation_error_alignment(
    observations: xr.DataArray,
    aggregation_error: AggregationError,
    *,
    owner: str,
    output_dim: str = "nmeasure",
    covariance_dim: str = "nmeasure_cov",
) -> None:
    """Validate one coherent aggregation-error value against observations.

    This is the cheap consumer-boundary check for the public
    :class:`AggregationError` value. Full numerical covariance validation is
    owned by :func:`resolve_aggregation_error` for general prepared inputs.

    Args:
        observations: Reference observation vector.
        aggregation_error: Selected aggregation-error representation.
        owner: Likelihood or component name used in diagnostics.
        output_dim: Observation dimension.
        covariance_dim: Dense covariance's second observation dimension.

    Raises:
        ValueError: If mode and payload disagree, the marginal variance is
            invalid or inconsistent with the payload, or labelled observation
            axes do not match the observations.
    """
    marginal_variance = np.asarray(aggregation_error.marginal_variance)
    nmeasure = observations.sizes[output_dim]
    if (
        marginal_variance.shape != (nmeasure,)
        or not np.issubdtype(marginal_variance.dtype, np.number)
        or not np.isfinite(marginal_variance).all()
        or (marginal_variance < 0).any()
    ):
        raise ValueError(
            f"{owner} aggregation-error marginal variance must be a finite, "
            f"non-negative vector of length {nmeasure}."
        )

    payloads = (
        aggregation_error.covariance is not None,
        aggregation_error.factor is not None,
        aggregation_error.diagonal_variance is not None,
    )
    expected_payloads = {
        "none": (False, False, False),
        "dense": (True, False, False),
        "low_rank": (False, True, True),
        "diagonal": (False, False, True),
    }
    expected_payload = expected_payloads.get(aggregation_error.mode)
    if expected_payload is None:
        raise ValueError(
            f"{owner} aggregation-error mode {aggregation_error.mode!r} is unsupported."
        )
    if payloads != expected_payload:
        raise ValueError(
            f"{owner} aggregation-error mode {aggregation_error.mode!r} has "
            "inconsistent covariance payloads."
        )

    if aggregation_error.mode == "none":
        expected_marginal = np.zeros(nmeasure)
    elif aggregation_error.mode == "dense":
        covariance = aggregation_error.covariance
        assert covariance is not None
        if covariance.dims != (output_dim, covariance_dim) or covariance.shape != (
            nmeasure,
            nmeasure,
        ):
            raise ValueError(
                f"{owner} aggregation-error dense covariance must have dims "
                f"({output_dim!r}, {covariance_dim!r}) and shape "
                f"({nmeasure}, {nmeasure})."
            )
        if covariance_dim not in covariance.coords:
            raise ValueError(
                f"{owner} input {AGGREGATION_ERROR_COVARIANCE!r} must carry "
                f"the observation coordinate {covariance_dim!r}."
            )
        column_labels = np.asarray(covariance.coords[covariance_dim].values)
        observation_labels = np.asarray(observations.get_index(output_dim).values)
        if not np.array_equal(column_labels, observation_labels):
            raise ValueError(
                f"{owner} input {AGGREGATION_ERROR_COVARIANCE!r} has "
                f"incompatible observation coordinate {covariance_dim!r}."
            )
        _require_eager_payload(
            covariance,
            AGGREGATION_ERROR_COVARIANCE,
            owner=owner,
        )
        expected_marginal = np.diag(
            _numeric_finite(
                AGGREGATION_ERROR_COVARIANCE,
                covariance,
                owner=f"{owner} input",
            )
        )
    elif aggregation_error.mode == "low_rank":
        factor = aggregation_error.factor
        diagonal_variance = aggregation_error.diagonal_variance
        assert factor is not None and diagonal_variance is not None
        if (
            factor.ndim != 2
            or factor.dims[0] != output_dim
            or factor.shape[0] != nmeasure
            or factor.shape[1] < 1
        ):
            raise ValueError(
                f"{owner} input {LOW_RANK_FACTOR!r} must have {nmeasure} observation rows "
                "and at least one rank column."
            )
        if (
            diagonal_variance.dims != (output_dim,)
            or diagonal_variance.shape != (nmeasure,)
        ):
            raise ValueError(
                f"{owner} input {DIAGONAL_RESIDUAL_VARIANCE!r} must be a "
                f"non-negative vector of length {nmeasure}."
            )
        _require_eager_payload(factor, LOW_RANK_FACTOR, owner=owner)
        _require_eager_payload(
            diagonal_variance,
            DIAGONAL_RESIDUAL_VARIANCE,
            owner=owner,
        )
        factor_values = _numeric_finite(LOW_RANK_FACTOR, factor, owner=f"{owner} input")
        diagonal_values = _numeric_finite(
            DIAGONAL_RESIDUAL_VARIANCE,
            diagonal_variance,
            owner=f"{owner} input",
        )
        if (diagonal_values < 0).any():
            raise ValueError(
                f"{owner} input {DIAGONAL_RESIDUAL_VARIANCE!r} must be a "
                f"non-negative vector of length {nmeasure}."
            )
        expected_marginal = np.sum(factor_values**2, axis=1) + diagonal_values
    else:
        diagonal_variance = aggregation_error.diagonal_variance
        assert diagonal_variance is not None
        if diagonal_variance.dims != (output_dim,) or diagonal_variance.shape != (
            nmeasure,
        ):
            raise ValueError(
                f"{owner} aggregation-error diagonal variance must be a "
                f"non-negative vector of length {nmeasure}."
            )
        _require_eager_payload(
            diagonal_variance,
            DIAGONAL_RESIDUAL_VARIANCE,
            owner=owner,
        )
        expected_marginal = _numeric_finite(
            DIAGONAL_RESIDUAL_VARIANCE,
            diagonal_variance,
            owner=f"{owner} input",
        )
        if (
            expected_marginal.shape != (nmeasure,) or (expected_marginal < 0).any()
        ):
            raise ValueError(
                f"{owner} aggregation-error diagonal variance must be a "
                f"non-negative vector of length {nmeasure}."
            )

    if not np.allclose(
        marginal_variance,
        expected_marginal,
        rtol=1e-6,
        atol=1e-12,
    ):
        raise ValueError(
            f"{owner} aggregation-error marginal variance is inconsistent with "
            f"its {aggregation_error.mode!r} covariance payload."
        )


def select_aggregation_error_mode(
    data: xr.Dataset, requested: AggregationErrorMode
) -> Literal["none", "dense", "low_rank", "diagonal"]:
    """Select an aggregation-error representation without materializing it.

    Args:
        data: Prepared inversion inputs containing any available aggregation-error
            representations.
        requested: Requested representation, or ``"auto"`` to infer one from
            the available inputs.

    Returns:
        The selected concrete aggregation-error representation.

    Raises:
        ValueError: If ``requested`` is invalid or ``"auto"`` finds both dense
            and low-rank representations.
    """
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


def aggregation_error_input_names(
    data: xr.Dataset,
    requested: AggregationErrorMode,
) -> tuple[str, ...]:
    """Return labelled arrays required by the selected error component.

    Args:
        data: Prepared inputs containing available aggregation-error products.
        requested: Requested representation, or ``"auto"``.

    Returns:
        Variable names required to materialize the selected representation.

    Raises:
        ValueError: If the requested mode is invalid or automatic selection is
            ambiguous.
    """
    selected = select_aggregation_error_mode(data, requested)
    if selected == "dense":
        names = [AGGREGATION_ERROR_COVARIANCE]
        if AGGREGATION_ERROR_SD in data:
            names.append(AGGREGATION_ERROR_SD)
        return tuple(names)
    if selected == "low_rank":
        names = [LOW_RANK_FACTOR, DIAGONAL_RESIDUAL_VARIANCE]
        if AGGREGATION_ERROR_SD in data:
            names.append(AGGREGATION_ERROR_SD)
        return tuple(names)
    if selected == "diagonal":
        return (AGGREGATION_ERROR_SD,)
    return ()


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

    Args:
        data: Prepared inversion inputs containing the requested aggregation-
            error representation.
        mode: Representation to use, or ``"auto"`` to select from available
            inputs.
        output_dim: Observation dimension used by error vectors and the first
            covariance dimension.
        covariance_dim: Second dimension required for a dense covariance.

    Returns:
        Validated aggregation-error arrays and their marginal variance.

    Raises:
        ValueError: If the selected inputs are absent, malformed, inconsistent,
            non-finite, or not a valid covariance representation.
    """
    if output_dim not in data.dims:
        raise ValueError(f"Prepared inputs have no observation dimension {output_dim!r}.")
    selected = select_aggregation_error_mode(data, mode)
    nmeasure = data.sizes[output_dim]

    if selected == "none":
        return AggregationError(mode="none", marginal_variance=np.zeros(nmeasure))

    if selected == "diagonal":
        if AGGREGATION_ERROR_SD not in data:
            raise ValueError(
                f"Diagonal aggregation error requires {AGGREGATION_ERROR_SD!r} in prepared inputs."
            )
        standard_deviation = data[AGGREGATION_ERROR_SD]
        if standard_deviation.dims != (output_dim,):
            raise ValueError(
                f"Aggregation-error input {AGGREGATION_ERROR_SD!r} must have dims "
                f"({output_dim!r},); got {standard_deviation.dims!r}."
            )
        if standard_deviation.sizes[output_dim] != nmeasure:
            raise ValueError(
                f"Aggregation-error input {AGGREGATION_ERROR_SD!r} is not observation-aligned."
            )
        (standard_deviation,) = _materialize_together(standard_deviation)
        values = _numeric_finite(AGGREGATION_ERROR_SD, standard_deviation)
        if (values < 0).any():
            raise ValueError(
                f"Aggregation-error input {AGGREGATION_ERROR_SD!r} must contain only "
                "non-negative values."
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
        observation_labels = np.asarray(data.get_index(output_dim).values)
        if covariance_dim in covariance.coords:
            column_labels = np.asarray(covariance.coords[covariance_dim].values)
            if not np.array_equal(column_labels, observation_labels):
                raise ValueError(
                    f"Aggregation-error input {AGGREGATION_ERROR_COVARIANCE!r} "
                    f"has incompatible observation coordinate {covariance_dim!r}."
                )
        else:
            covariance = covariance.assign_coords({covariance_dim: observation_labels})
        (covariance,) = _materialize_together(covariance)
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
    diagonal = data[DIAGONAL_RESIDUAL_VARIANCE]
    if diagonal.dims != (output_dim,):
        raise ValueError(
            f"Aggregation-error input {DIAGONAL_RESIDUAL_VARIANCE!r} must have dims "
            f"({output_dim!r},); got {diagonal.dims!r}."
        )
    if diagonal.sizes[output_dim] != nmeasure:
        raise ValueError(
            f"Aggregation-error input {DIAGONAL_RESIDUAL_VARIANCE!r} is not observation-aligned."
        )

    factor, diagonal = _materialize_together(factor, diagonal)
    factor_values = _numeric_finite(LOW_RANK_FACTOR, factor)
    diagonal_values = _numeric_finite(DIAGONAL_RESIDUAL_VARIANCE, diagonal)
    if (diagonal_values < 0).any():
        raise ValueError(
            f"Aggregation-error input {DIAGONAL_RESIDUAL_VARIANCE!r} must contain only "
            "non-negative values."
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
    """Validate an optional marginal-SD diagnostic beside structured input.

    Args:
        data: Prepared inputs that may contain ``aggregation_error_sd``.
        marginal_variance: Diagonal of the selected structured covariance.
        output_dim: Observation dimension required on the diagnostic.

    Raises:
        ValueError: If the diagnostic is invalid or disagrees with the selected
            covariance diagonal.
    """
    if AGGREGATION_ERROR_SD not in data:
        return
    _, values = _validate_vector(data, AGGREGATION_ERROR_SD, output_dim=output_dim)
    expected = np.sqrt(marginal_variance)
    if not np.allclose(values, expected, rtol=1e-6, atol=1e-12):
        raise ValueError(
            f"Aggregation-error diagnostic {AGGREGATION_ERROR_SD!r} must equal the square root "
            "of the selected covariance diagonal."
        )
