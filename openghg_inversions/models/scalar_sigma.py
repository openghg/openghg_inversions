"""Global scalar-sigma likelihood using a reusable labelled eigenbasis.

For a fixed positive-semidefinite base covariance ``C0``, this component
evaluates ``C0 + sigma_global**2 I`` without repeating a dense factorization at
every sampler step. Preparation owns alignment, eager materialization, and the
one dense eigendecomposition. Model construction consumes the resulting
labelled eigenbasis as a trusted numerical value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from dask import compute as dask_compute
from dask.array import Array as DaskArray
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable
import xarray as xr

from openghg_inversions.array_ops import to_dense, validate_covariance_coordinates
from openghg_inversions.models.components import add_model_data
from openghg_inversions.models.priors import parse_prior, positive_prior_args
from openghg_inversions.observation_error import AggregationError
from openghg_inversions.serialization import decode_cf_multiindexes, encode_cf_multiindexes


SCALAR_SIGMA_CACHE_SCHEMA = "openghg_inversions.scalar_sigma_eigenbasis"
SCALAR_SIGMA_CACHE_VERSION = 1
SCALAR_SIGMA_MODE_DIM = "scalar_sigma_mode"
EIGENVECTORS = "eigenvectors"
EIGENVALUES = "eigenvalues"


def _align_observation_array(
    observations: xr.DataArray,
    array: xr.DataArray,
    *,
    name: str,
    output_dim: str,
) -> xr.DataArray:
    """Return an array with the exact observation coordinate or raise."""
    if output_dim not in observations.indexes:
        raise ValueError(f"Scalar-sigma observations require an indexed {output_dim!r} coordinate.")
    if output_dim not in array.dims:
        raise ValueError(f"Scalar-sigma input {name!r} must include dimension {output_dim!r}.")
    try:
        _, aligned = xr.align(observations, array, join="exact", copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Scalar-sigma input {name!r} must use the exact ordered observation coordinate."
        ) from error
    return aligned


def _materialize_together(*arrays: xr.DataArray) -> tuple[xr.DataArray, ...]:
    """Materialize related dense payloads in one shared Dask operation."""
    dense_arrays = tuple(to_dense(array) for array in arrays)
    if all(not isinstance(array.data, DaskArray) for array in dense_arrays):
        return dense_arrays
    computed = dask_compute(*(array.data for array in dense_arrays))
    return tuple(
        array.copy(deep=False, data=data) for array, data in zip(dense_arrays, computed, strict=True)
    )


def _finite_values(name: str, array: xr.DataArray) -> NDArray[np.float64]:
    """Return finite floating-point values from one eager labelled array."""
    try:
        values = np.asarray(array.values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Scalar-sigma input {name!r} must be numeric.") from error
    if not np.isfinite(values).all():
        raise ValueError(f"Scalar-sigma input {name!r} must contain only finite values.")
    return values


def _scale_relative_tolerance(values: NDArray[np.float64]) -> float:
    """Return a unit-invariant tolerance relative to the numerical scale."""
    scale = max(
        float(np.max(np.abs(values), initial=0.0)),
        np.finfo(values.dtype).tiny,
    )
    return 1.0e-10 * scale


def scalar_sigma_base_covariance(
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    aggregation_error: AggregationError,
    *,
    output_dim: str = "nmeasure",
    covariance_dim: str = "nmeasure_cov",
) -> NDArray[np.float64]:
    """Align and materialize the fixed scalar-sigma covariance once.

    The observation error and selected aggregation-error payload retain their
    labels through exact alignment. Their dense payloads are then materialized
    together before forming ``A + D_obs``.

    Args:
        observations: One-dimensional observations carrying the canonical
            ordered observation coordinate and a non-empty ``units`` attribute.
        observation_error: Non-negative reported standard deviations in the
            same units and observation order as ``observations``.
        aggregation_error: Validated aggregation-error representation for the
            same observations.
        output_dim: Observation dimension name.
        covariance_dim: Column dimension used by a dense covariance.

    Returns:
        Dense symmetric ``A + D_obs`` covariance. Preparation validates
        positive semidefiniteness during its single eigendecomposition.

    Raises:
        ValueError: If dimensions, labels, units, values, or covariance
            properties are invalid.
    """
    if observations.dims != (output_dim,):
        raise ValueError(
            f"Scalar-sigma observations must have dims ({output_dim!r},); got {observations.dims!r}."
        )
    observation_units = str(observations.attrs.get("units", "")).strip()
    error_units = str(observation_error.attrs.get("units", "")).strip()
    if not observation_units or observation_units != error_units:
        raise ValueError("Scalar-sigma observations and reported errors require matching explicit units.")
    error = _align_observation_array(
        observations,
        observation_error,
        name="observation_error",
        output_dim=output_dim,
    )
    if error.dims != (output_dim,):
        raise ValueError(
            f"Scalar-sigma observation_error must have dims ({output_dim!r},); got {error.dims!r}."
        )

    payloads: list[tuple[str, xr.DataArray]] = []
    if aggregation_error.mode == "dense":
        if aggregation_error.covariance is None:
            raise ValueError("Dense aggregation error requires a covariance array.")
        covariance = aggregation_error.covariance
        validate_covariance_coordinates(
            covariance,
            dim=output_dim,
            covariance_dim=covariance_dim,
        )
        covariance = _align_observation_array(
            observations,
            covariance,
            name="aggregation_error_covariance",
            output_dim=output_dim,
        )
        payloads.append(("aggregation_error_covariance", covariance))
    elif aggregation_error.mode == "low_rank":
        if aggregation_error.factor is None or aggregation_error.diagonal_variance is None:
            raise ValueError("Low-rank aggregation error requires factor and diagonal arrays.")
        factor = _align_observation_array(
            observations,
            aggregation_error.factor,
            name="low_rank_factor",
            output_dim=output_dim,
        )
        if factor.ndim != 2 or factor.dims[0] != output_dim:
            raise ValueError(
                "Scalar-sigma low_rank_factor must be two-dimensional with the "
                f"first dimension {output_dim!r}."
            )
        diagonal = _align_observation_array(
            observations,
            aggregation_error.diagonal_variance,
            name="diagonal_residual_variance",
            output_dim=output_dim,
        )
        if diagonal.dims != (output_dim,):
            raise ValueError(
                f"Scalar-sigma diagonal_residual_variance must be one-dimensional on {output_dim!r}."
            )
        payloads.extend([("low_rank_factor", factor), ("diagonal_residual_variance", diagonal)])
    elif aggregation_error.mode == "diagonal":
        if aggregation_error.diagonal_variance is None:
            raise ValueError("Diagonal aggregation error requires a variance array.")
        diagonal = _align_observation_array(
            observations,
            aggregation_error.diagonal_variance,
            name="diagonal_variance",
            output_dim=output_dim,
        )
        if diagonal.dims != (output_dim,):
            raise ValueError(f"Scalar-sigma diagonal_variance must have dims ({output_dim!r},).")
        payloads.append(("diagonal_variance", diagonal))
    elif aggregation_error.mode != "none":
        raise ValueError(f"Unsupported aggregation-error mode {aggregation_error.mode!r}.")

    materialized = _materialize_together(error, *(array for _, array in payloads))
    error_values = _finite_values("observation_error", materialized[0])
    if (error_values < 0.0).any():
        raise ValueError("Scalar-sigma observation_error must be non-negative.")
    n_observation = observations.sizes[output_dim]
    covariance_values = np.diag(np.square(error_values))
    eager_payloads = {name: array for (name, _), array in zip(payloads, materialized[1:], strict=True)}
    if aggregation_error.mode == "dense":
        values = _finite_values(
            "aggregation_error_covariance",
            eager_payloads["aggregation_error_covariance"],
        )
        if values.shape != (n_observation, n_observation):
            raise ValueError("Scalar-sigma aggregation covariance must be square.")
        covariance_values += values
    elif aggregation_error.mode == "low_rank":
        factor_values = _finite_values("low_rank_factor", eager_payloads["low_rank_factor"])
        diagonal_values = _finite_values(
            "diagonal_residual_variance",
            eager_payloads["diagonal_residual_variance"],
        )
        if (diagonal_values < 0.0).any():
            raise ValueError("Scalar-sigma diagonal_residual_variance must be non-negative.")
        covariance_values += factor_values @ factor_values.T
        covariance_values.flat[:: n_observation + 1] += diagonal_values
    elif aggregation_error.mode == "diagonal":
        diagonal_values = _finite_values("diagonal_variance", eager_payloads["diagonal_variance"])
        if (diagonal_values < 0.0).any():
            raise ValueError("Scalar-sigma diagonal_variance must be non-negative.")
        covariance_values.flat[:: n_observation + 1] += diagonal_values

    tolerance = _scale_relative_tolerance(covariance_values)
    if not np.allclose(
        covariance_values,
        covariance_values.T,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("Scalar-sigma base covariance must be symmetric.")
    covariance_values = (covariance_values + covariance_values.T) * 0.5
    return cast(NDArray[np.float64], covariance_values)


@dataclass(frozen=True)
class ScalarSigmaEigenbasis:
    """Trusted labelled eigenbasis for ``C0 + sigma_global**2 I``.

    Args:
        eigenvectors: Square labelled matrix with dimensions ``(output_dim,
            scalar_sigma_mode)``.
        eigenvalues: Non-negative values on ``scalar_sigma_mode``.
        concentration_units: Physical concentration units shared by the
            observations, reported errors, and scalar mismatch amplitude.
        output_dim: Observation dimension represented by the rows.

    Raises:
        ValueError: If dimensions, coordinates, units, or numerical values do
            not define a finite labelled eigenbasis.
    """

    eigenvectors: xr.DataArray
    eigenvalues: xr.DataArray
    concentration_units: str
    output_dim: str = "nmeasure"

    def __post_init__(self) -> None:
        """Validate dimensions and own eager numerical payloads."""
        expected_vector_dims = (self.output_dim, SCALAR_SIGMA_MODE_DIM)
        if self.eigenvectors.dims != expected_vector_dims:
            raise ValueError(
                f"eigenvectors must have dims {expected_vector_dims!r}; got {self.eigenvectors.dims!r}."
            )
        if self.eigenvalues.dims != (SCALAR_SIGMA_MODE_DIM,):
            raise ValueError(f"eigenvalues must have dims ({SCALAR_SIGMA_MODE_DIM!r},).")
        n_observation = self.eigenvectors.sizes[self.output_dim]
        if n_observation == 0 or self.eigenvectors.shape != (
            n_observation,
            n_observation,
        ):
            raise ValueError("Scalar-sigma eigenvectors must be non-empty and square.")
        if self.eigenvalues.sizes[SCALAR_SIGMA_MODE_DIM] != n_observation:
            raise ValueError("Scalar-sigma eigenvalues must match the eigenvector size.")
        if self.output_dim not in self.eigenvectors.indexes:
            raise ValueError("Scalar-sigma eigenvectors require an indexed observation coordinate.")
        vector_modes = self.eigenvectors.indexes.get(SCALAR_SIGMA_MODE_DIM)
        value_modes = self.eigenvalues.indexes.get(SCALAR_SIGMA_MODE_DIM)
        if vector_modes is None or value_modes is None or not vector_modes.equals(value_modes):
            raise ValueError("Scalar-sigma eigenvector and eigenvalue modes must match exactly.")
        units = str(self.concentration_units).strip()
        if not units:
            raise ValueError("Scalar-sigma eigenbasis requires concentration units.")
        source_vectors = np.asarray(self.eigenvectors.values)
        source_dtype = (
            source_vectors.dtype if np.issubdtype(source_vectors.dtype, np.floating) else np.dtype(np.float64)
        )
        vectors = np.array(source_vectors, dtype=np.float64, copy=True)
        values = np.array(self.eigenvalues.values, dtype=np.float64, copy=True)
        if not np.isfinite(vectors).all() or not np.isfinite(values).all():
            raise ValueError("Scalar-sigma eigenbasis values must be finite.")
        if (values < 0.0).any():
            raise ValueError("Scalar-sigma eigenvalues must be non-negative.")
        orthogonality_tolerance = 100.0 * np.finfo(source_dtype).eps * np.sqrt(n_observation)
        if not np.allclose(
            vectors.T @ vectors,
            np.eye(n_observation),
            rtol=orthogonality_tolerance,
            atol=orthogonality_tolerance,
        ):
            raise ValueError("Scalar-sigma eigenvectors must be orthonormal.")
        vectors.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(
            self,
            "eigenvectors",
            self.eigenvectors.copy(deep=False, data=vectors),
        )
        object.__setattr__(
            self,
            "eigenvalues",
            self.eigenvalues.copy(deep=False, data=values),
        )
        object.__setattr__(self, "concentration_units", units)

    @property
    def n_observation(self) -> int:
        """Return the number of represented observations."""
        return self.eigenvectors.sizes[self.output_dim]

    def to_dataset(self) -> xr.Dataset:
        """Return the versioned xarray cache representation.

        Returns:
            Dataset containing the labelled eigenvectors, eigenvalues, schema,
            units, and observation-dimension name.
        """
        return xr.Dataset(
            {EIGENVECTORS: self.eigenvectors, EIGENVALUES: self.eigenvalues},
            attrs={
                "schema": SCALAR_SIGMA_CACHE_SCHEMA,
                "schema_version": SCALAR_SIGMA_CACHE_VERSION,
                "output_dim": self.output_dim,
                "concentration_units": self.concentration_units,
            },
        )

    def logp(
        self,
        value: TensorVariable,
        mean: TensorVariable,
        sigma: TensorVariable,
    ) -> TensorVariable:
        """Return the normalized differentiable Gaussian log density.

        Args:
            value: Observed concentration vector.
            mean: Modelled concentration vector.
            sigma: Global IID mismatch standard deviation.

        Returns:
            Scalar symbolic log density.
        """
        vectors = pt.as_tensor_variable(pm.floatX(np.asarray(self.eigenvectors.values)))
        values = pt.as_tensor_variable(pm.floatX(np.asarray(self.eigenvalues.values)))
        residual = pt.dot(vectors.T, value - mean)
        variance = values + pt.square(sigma)
        valid = pt.all(pt.gt(variance, 0.0))
        safe_variance = pt.where(valid, variance, pt.ones_like(variance))
        density = -pm.floatX(0.5) * (
            pm.floatX(self.n_observation * np.log(2.0 * np.pi))
            + pt.sum(pt.log(safe_variance))
            + pt.sum(pt.square(residual) / safe_variance)
        )
        return cast(TensorVariable, pt.where(valid, density, pm.floatX(-np.inf)))

    def logpdf(self, residual: NDArray[np.float64], *, sigma: float) -> float:
        """Evaluate the represented Gaussian log density with NumPy.

        Args:
            residual: Observation-minus-model residual vector.
            sigma: Non-negative global mismatch standard deviation.

        Returns:
            Normalized Gaussian log density.

        Raises:
            ValueError: If the residual shape or sigma value is invalid.
        """
        residual_value = np.asarray(residual, dtype=np.float64)
        if residual_value.shape != (self.n_observation,):
            raise ValueError(f"residual has shape {residual_value.shape}, expected {(self.n_observation,)}.")
        if not np.isfinite(residual_value).all():
            return -np.inf
        if not np.isfinite(sigma) or sigma < 0.0:
            raise ValueError("sigma must be finite and non-negative.")
        variance = np.asarray(self.eigenvalues.values) + sigma**2
        if (variance <= 0.0).any():
            return -np.inf
        transformed = np.asarray(self.eigenvectors.values).T @ residual_value
        return float(
            -0.5
            * (
                self.n_observation * np.log(2.0 * np.pi)
                + np.log(variance).sum()
                + np.sum(np.square(transformed) / variance)
            )
        )

    def random(
        self,
        mean: np.ndarray,
        sigma: np.ndarray,
        rng: np.random.Generator | None = None,
        size: int | Sequence[int] | None = None,
    ) -> np.ndarray:
        """Draw random values for the PyMC custom distribution.

        Args:
            mean: Modelled concentration vector.
            sigma: Global mismatch standard deviation.
            rng: Optional NumPy random generator supplied by PyMC.
            size: Optional leading sample shape.

        Returns:
            Random concentration vectors with the requested leading shape.
        """
        generator = np.random.default_rng() if rng is None else rng
        sample_shape = () if size is None else (size,) if isinstance(size, int) else tuple(size)
        variance = np.asarray(self.eigenvalues.values) + np.square(np.asarray(sigma))
        standard = generator.normal(size=(*sample_shape, self.n_observation))
        noise = (standard * np.sqrt(variance)) @ np.asarray(self.eigenvectors.values).T
        mean_values = np.asarray(mean)
        return (mean_values + noise).astype(mean_values.dtype, copy=False)


def prepare_scalar_sigma_eigenbasis(
    *,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    aggregation_error: AggregationError,
    output_dim: str = "nmeasure",
) -> ScalarSigmaEigenbasis:
    """Prepare the labelled eigenbasis for one fixed CO2 covariance.

    Args:
        observations: CO2 observations carrying an indexed observation
            coordinate and explicit concentration units.
        observation_error: Reported standard deviations in the same units and
            exact observation order.
        aggregation_error: Validated fixed aggregation-error representation.
        output_dim: Observation dimension name.

    Returns:
        Eager labelled eigenbasis ready for serialization or model building.

    Raises:
        ValueError: If inputs cannot form a finite positive-semidefinite base
            covariance.
    """
    covariance = scalar_sigma_base_covariance(
        observations,
        observation_error,
        aggregation_error,
        output_dim=output_dim,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = _scale_relative_tolerance(eigenvalues)
    if float(eigenvalues.min(initial=0.0)) < -tolerance:
        raise ValueError("Scalar-sigma base covariance must be positive semidefinite.")
    mode = np.arange(observations.sizes[output_dim])
    return ScalarSigmaEigenbasis(
        eigenvectors=xr.DataArray(
            eigenvectors,
            dims=(output_dim, SCALAR_SIGMA_MODE_DIM),
            coords={
                output_dim: observations.coords[output_dim],
                SCALAR_SIGMA_MODE_DIM: mode,
            },
            name=EIGENVECTORS,
        ),
        eigenvalues=xr.DataArray(
            np.maximum(eigenvalues, 0.0),
            dims=SCALAR_SIGMA_MODE_DIM,
            coords={SCALAR_SIGMA_MODE_DIM: mode},
            name=EIGENVALUES,
        ),
        concentration_units=str(observations.attrs["units"]),
        output_dim=output_dim,
    )


def save_scalar_sigma_eigenbasis(
    path: str | Path,
    eigenbasis: ScalarSigmaEigenbasis,
) -> Path:
    """Save one labelled scalar-sigma cache as NetCDF.

    Args:
        path: Destination ending in ``.nc``.
        eigenbasis: Prepared labelled eigenbasis.

    Returns:
        Resolved destination path.

    Raises:
        ValueError: If ``path`` does not end in ``.nc``.
        OSError: If the destination cannot be written.
    """
    destination = Path(path).resolve()
    if destination.suffix != ".nc":
        raise ValueError("Scalar-sigma eigenbasis caches must use a '.nc' suffix.")
    dataset = eigenbasis.to_dataset()
    if isinstance(dataset.indexes.get(eigenbasis.output_dim), pd.MultiIndex):
        dataset = encode_cf_multiindexes(dataset, eigenbasis.output_dim)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(destination)
    return destination


def load_scalar_sigma_eigenbasis(
    path: str | Path,
    *,
    observations: xr.DataArray,
) -> ScalarSigmaEigenbasis:
    """Load and align a versioned scalar-sigma eigenbasis cache.

    Loading is the cache trust boundary. It validates the small xarray schema,
    exact dimensions, finite values, units, and ordered observation labels.
    It does not reconstruct or hash the dense covariance.

    Args:
        path: NetCDF cache created by :func:`save_scalar_sigma_eigenbasis`.
        observations: Observations that will consume the cache.

    Returns:
        Eager trusted eigenbasis aligned to ``observations``.

    Raises:
        ValueError: If schema, dimensions, labels, units, or values disagree.
        OSError: If the cache cannot be opened.
    """
    source = Path(path).resolve()
    with xr.open_dataset(source) as opened:
        dataset = opened.load()
    if dataset.attrs.get("schema") != SCALAR_SIGMA_CACHE_SCHEMA:
        raise ValueError("Scalar-sigma cache has an unsupported schema.")
    if dataset.attrs.get("schema_version") != SCALAR_SIGMA_CACHE_VERSION:
        raise ValueError("Scalar-sigma cache has an unsupported schema version.")
    output_dim = dataset.attrs.get("output_dim")
    if not isinstance(output_dim, str) or not output_dim:
        raise ValueError("Scalar-sigma cache has no valid output dimension.")
    if output_dim in dataset.coords and "compress" in dataset[output_dim].attrs:
        dataset = decode_cf_multiindexes(dataset, output_dim)
    missing = [name for name in (EIGENVECTORS, EIGENVALUES) if name not in dataset]
    if missing:
        raise ValueError(f"Scalar-sigma cache is missing variable(s): {missing!r}.")
    if observations.dims != (output_dim,):
        raise ValueError(
            f"Scalar-sigma cache expects observations on {output_dim!r}; got {observations.dims!r}."
        )
    try:
        _, vectors = xr.align(
            observations,
            dataset[EIGENVECTORS],
            join="exact",
            copy=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Scalar-sigma cache must use the exact ordered observation coordinate.") from error
    units = str(dataset.attrs.get("concentration_units", "")).strip()
    if units != str(observations.attrs.get("units", "")).strip():
        raise ValueError("Scalar-sigma cache and observations require matching units.")
    return ScalarSigmaEigenbasis(
        eigenvectors=vectors,
        eigenvalues=dataset[EIGENVALUES],
        concentration_units=units,
        output_dim=output_dim,
    )


def add_scalar_sigma_eigen_likelihood(
    *,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    aggregation_error: AggregationError,
    mean: TensorVariable,
    eigenbasis: ScalarSigmaEigenbasis,
    sigma_prior: Mapping[str, Any],
    output_dim: str = "nmeasure",
) -> TensorVariable:
    """Add a global scalar-sigma likelihood from a trusted eigenbasis.

    Args:
        observations: CO2 observations represented by ``eigenbasis``.
        observation_error: Reported observation standard deviations. These are
            registered as model data and contribute to ``epsilon``.
        aggregation_error: Validated fixed aggregation error. Its marginal
            variance contributes to ``epsilon``; its full covariance is already
            represented by ``eigenbasis``.
        mean: Completed modelled concentration on ``output_dim``.
        eigenbasis: Prepared and aligned eigenbasis loaded before entering the
            registered PyMC model context.
        sigma_prior: Explicit positive-support prior in the observations'
            concentration units.
        output_dim: Observation dimension name.

    Returns:
        Observed custom Gaussian variable named ``y``.

    Raises:
        ValueError: If ``sigma_prior`` does not have non-negative support.
    """
    observed = add_model_data(observations.transpose(output_dim), "Y")
    reported_error = add_model_data(observation_error.transpose(output_dim), "error")
    sigma = parse_prior("sigma_global", positive_prior_args(sigma_prior))
    pm.Deterministic(
        "epsilon",
        pt.sqrt(
            reported_error**2 + pm.floatX(np.asarray(aggregation_error.marginal_variance)) + pt.square(sigma)
        ),
        dims=output_dim,
    )
    return cast(
        TensorVariable,
        pm.CustomDist(
            "y",
            mean,
            sigma,
            logp=eigenbasis.logp,
            random=eigenbasis.random,
            signature="(n),()->(n)",
            observed=observed,
            dims=output_dim,
        ),
    )


__all__ = [
    "SCALAR_SIGMA_CACHE_SCHEMA",
    "SCALAR_SIGMA_CACHE_VERSION",
    "ScalarSigmaEigenbasis",
    "add_scalar_sigma_eigen_likelihood",
    "load_scalar_sigma_eigenbasis",
    "prepare_scalar_sigma_eigenbasis",
    "save_scalar_sigma_eigenbasis",
    "scalar_sigma_base_covariance",
]
