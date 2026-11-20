"""Labelled arithmetic-moment contracts for correlated positive states.

This backend-neutral module validates an already-reduced state coordinate and
an explicitly dense arithmetic covariance, then derives the latent Gaussian
moments required to represent those moments with a multivariate LogNormal
distribution. PyMC graph construction lives in
``openghg_inversions.models.components``.

The covariance accepted here is for the reduced inversion state, not a native
grid. The implementation materializes dense covariance, latent-covariance, and
Cholesky arrays, with quadratic memory use and cubic factorization cost. Native
covariances with tens or hundreds of thousands of grid cells must therefore
remain structured and be projected into the reduced state before constructing
this contract.

This module does not perform that native-to-reduced transformation or remove
state components. The coherent covariance, transformed-forward-model, and
aggregation-error identities are exact only for a jointly Gaussian state.
Reusing the resulting first two moments while representing the retained state
as LogNormal and the unresolved contribution as Gaussian is a moment-matched
closure, not exact marginalization of a LogNormal state. Fixing a state at a
known value is instead handled by ``models.StateActivity``.

Model builders take a ``CorrelatedPrior``: arithmetic moments with a marginal
family per state, joined by a Gaussian copula. Its only marginal is lognormal,
built by ``CorrelatedLognormalPrior``. That class's constructor eagerly
computes xarray inputs and owns independent copies; ``to_dataset`` and
``from_dataset`` provide persistence boundaries. Construction warns before
dense covariance materialization when the reduced state exceeds 1,000
components.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


_SERIALIZED_VARIABLE_NAMES = frozenset(
    {
        "arithmetic_mean",
        "arithmetic_covariance",
        "latent_mean",
        "latent_covariance",
        "latent_cholesky",
    }
)

_DENSE_STATE_WARNING_THRESHOLD = 1000


def _materialize_numeric(array: xr.DataArray, *, name: str) -> np.ndarray:
    """Materialize an xarray object as owned finite real float64 values.

    Args:
        array: Labelled array to compute and copy into memory.
        name: Human-readable input name used in validation errors.

    Returns:
        An owned NumPy array with ``float64`` dtype.

    Raises:
        ValueError: If the values are non-numeric, complex, or non-finite.

    Notes:
        Calling ``compute`` can trigger lazy computation or input/output before
        the returned values are copied into memory.
    """
    values = np.asarray(array.compute().values)
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.complexfloating):
        raise ValueError(f"{name} must contain real numeric values.")
    values = np.array(values, dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _copy_state_auxiliary_coords(
    target: xr.DataArray,
    source: xr.DataArray,
    *,
    state_dim: str,
) -> xr.DataArray:
    """Copy state-aligned auxiliary coordinates into a canonical array.

    Args:
        target: Canonical array that already owns its primary coordinates.
        source: Array from which auxiliary state metadata is copied.
        state_dim: Name of the sole permitted dimension for copied coordinates.

    Returns:
        A deep copy of ``target`` with compatible auxiliary coordinates copied
        from ``source``.

    Notes:
        Scalar and ``state_dim``-only coordinates are compatible. Coordinates
        already present on ``target`` or involving any other dimension are
        skipped.
    """
    result = target
    for name, coord in source.coords.items():
        if name in result.coords:
            continue
        if set(coord.dims).issubset({state_dim}):
            result = result.assign_coords({name: coord.copy(deep=True)})
    return result.copy(deep=True)


def _require_state_index(array: xr.DataArray, state_dim: str, *, name: str) -> pd.Index:
    """Validate and return the labelled index owned by a state dimension.

    Args:
        array: Array whose state index is required.
        state_dim: Name of the state dimension and coordinate.
        name: Human-readable input name used in validation errors.

    Returns:
        The unique pandas index attached to ``state_dim``.

    Raises:
        ValueError: If the coordinate is absent or non-unique, or if a
            MultiIndex has missing or duplicate level names.
    """
    if state_dim not in array.coords or state_dim not in array.indexes:
        raise ValueError(f"{name} must have a labelled {state_dim!r} coordinate.")
    index = array.indexes[state_dim]
    if not index.is_unique:
        raise ValueError(f"{name} {state_dim!r} coordinate must contain unique values.")
    if isinstance(index, pd.MultiIndex):
        names = tuple(index.names)
        if any(not isinstance(level, str) or not level for level in names):
            raise ValueError(f"{name} {state_dim!r} MultiIndex levels must have non-empty names.")
        if len(set(names)) != len(names):
            raise ValueError(f"{name} {state_dim!r} MultiIndex level names must be unique.")
    return index


def _coord_values(coord: xr.DataArray) -> list[Any]:
    """Return coordinate labels without promoting tuple objects to a MultiIndex."""
    index = coord.to_index()
    return list(index.tolist())


def _label_token(value: Any) -> str:
    """Return a stable, serializable identity token for one state label."""
    return json.dumps(value, default=str, ensure_ascii=False, separators=(",", ":"))


def _covariance_array(
    covariance: xr.DataArray | np.ndarray,
    *,
    mean: xr.DataArray,
    state_dim: str,
    covariance_dim: str,
) -> xr.DataArray:
    """Normalize covariance to labelled rows and a same-order column axis.

    Args:
        covariance: Dense covariance values, optionally with xarray labels.
        mean: Validated one-dimensional arithmetic mean.
        state_dim: Name of the labelled covariance row dimension.
        covariance_dim: Distinct name for the covariance column dimension.

    Returns:
        An owned ``float64`` covariance with canonical dimensions and a
        serialized column-label identity coordinate.

    Raises:
        ValueError: If the shape, dimensions, labels, values, or serialized
            column identity do not match the arithmetic mean.
    """
    state_coord = mean.coords[state_dim]
    mean_index = _require_state_index(mean, state_dim, name="Arithmetic mean")
    state_size = mean.sizes[state_dim]
    expected_label_tokens = [_label_token(label) for label in mean_index.tolist()]
    label_coord = f"{covariance_dim}_label"
    if isinstance(covariance, xr.DataArray):
        covariance_attrs = dict(covariance.attrs)
        if covariance.dims != (state_dim, covariance_dim):
            raise ValueError(
                "Arithmetic covariance must have dimensions "
                f"({state_dim!r}, {covariance_dim!r}); got {covariance.dims!r}."
            )
        if covariance.shape != (state_size, state_size):
            raise ValueError(
                "Arithmetic covariance must be square and match the state length; "
                f"got shape {covariance.shape!r} for {state_size} states."
            )
        covariance_index = _require_state_index(
            covariance,
            state_dim,
            name="Arithmetic covariance",
        )
        if not covariance_index.equals(mean_index):
            raise ValueError(
                "Arithmetic covariance row labels must exactly match the arithmetic mean labels "
                "in the same order."
            )
        if covariance_dim in covariance.coords:
            column_labels = _coord_values(covariance.coords[covariance_dim])
            if column_labels != list(mean_index.tolist()):
                raise ValueError(
                    "Arithmetic covariance column labels must exactly match the arithmetic mean "
                    "labels in the same order."
                )
        if label_coord in covariance.coords:
            stored_label_coord = covariance.coords[label_coord]
            if stored_label_coord.dims != (covariance_dim,):
                raise ValueError(
                    f"Arithmetic covariance label coordinate {label_coord!r} must have only "
                    f"the {covariance_dim!r} dimension."
                )
            if list(np.asarray(stored_label_coord.values, dtype=str)) != expected_label_tokens:
                raise ValueError(
                    "Arithmetic covariance serialized column identity must exactly match the "
                    "arithmetic mean labels in the same order."
                )
        values = _materialize_numeric(covariance, name="Arithmetic covariance")
    else:
        covariance_attrs = {}
        values = np.asarray(covariance)
        if values.shape != (state_size, state_size):
            raise ValueError(
                "Arithmetic covariance must be square and match the state length; "
                f"got shape {values.shape!r} for {state_size} states."
            )
        if not np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.complexfloating):
            raise ValueError("Arithmetic covariance must contain real numeric values.")
        values = np.array(values, dtype=np.float64, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("Arithmetic covariance must contain only finite values.")

    result = xr.DataArray(
        values,
        dims=(state_dim, covariance_dim),
        coords={
            state_dim: state_coord,
            label_coord: (covariance_dim, np.asarray(expected_label_tokens, dtype=str)),
        },
        name="arithmetic_covariance",
        attrs=covariance_attrs,
    )
    return _copy_state_auxiliary_coords(result, mean, state_dim=state_dim)


def _matrix_tolerance(values: np.ndarray) -> float:
    """Return a scale-aware numerical validation tolerance."""
    return 1e-10 * max(float(np.max(np.abs(values))), 1.0)


def _warn_large_dense_state(state_size: int, *, stacklevel: int = 3) -> None:
    """Warn when a dense reduced state exceeds the operational size threshold.

    Args:
        state_size: Number of components in the reduced inversion state.

    Warns:
        UserWarning: If ``state_size`` exceeds the operational threshold of
            1000 components. The threshold is not a mathematical limit.
    """
    if state_size > _DENSE_STATE_WARNING_THRESHOLD:
        warnings.warn(
            "CorrelatedLognormalPrior expects an already-reduced dense covariance; "
            f"state size {state_size} exceeds the operational threshold of "
            f"{_DENSE_STATE_WARNING_THRESHOLD} (not a mathematical limit). This "
            "representation is unsuitable for a native-grid covariance.",
            UserWarning,
            stacklevel=stacklevel,
        )


def _validate_arithmetic_moments(
    mean: xr.DataArray,
    arithmetic_covariance: xr.DataArray | np.ndarray,
    *,
    covariance_dim: str | None,
    stacklevel: int = 4,
) -> tuple[str, str, np.ndarray, xr.DataArray]:
    """Validate a positive arithmetic mean and a symmetric PSD covariance.

    Args:
        mean: One-dimensional arithmetic mean with a unique labelled state
            coordinate.
        arithmetic_covariance: Dense covariance, as accepted by
            :class:`CorrelatedLognormalPrior`.
        covariance_dim: Optional covariance column dimension name.
        stacklevel: Warning stack level for the dense-state warning.

    Returns:
        State dimension, covariance dimension, owned mean values and the
        canonical symmetrized covariance.

    Raises:
        TypeError: If ``mean`` is not an xarray ``DataArray``.
        ValueError: If dimensions, labels or values are invalid, the mean is
            not positive, or the covariance is not symmetric PSD.
    """
    if not isinstance(mean, xr.DataArray):
        raise TypeError("Arithmetic mean must be an xarray DataArray.")
    if mean.ndim != 1:
        raise ValueError(f"Arithmetic mean must be one-dimensional; got dimensions {mean.dims!r}.")
    state_dim = str(mean.dims[0])
    covariance_dim = covariance_dim or f"{state_dim}_covariance"
    if covariance_dim == state_dim:
        raise ValueError("Covariance matrix axes must use distinct dimension names.")
    reserved_coord_names = _SERIALIZED_VARIABLE_NAMES | {f"{covariance_dim}_label"}
    auxiliary_names = {name for name in mean.coords if isinstance(name, str) and name != state_dim}
    reserved_auxiliary_names = auxiliary_names & reserved_coord_names
    if reserved_auxiliary_names:
        raise ValueError(
            "State auxiliary coordinate names must not use reserved correlated-state "
            f"dataset names; got {sorted(reserved_auxiliary_names)!r}."
        )
    if covariance_dim in mean.coords:
        raise ValueError(
            "Covariance matrix column dimension must not collide with a state or auxiliary "
            f"coordinate name; got {covariance_dim!r}."
        )
    if f"{covariance_dim}_label" in mean.coords:
        raise ValueError(
            "Covariance matrix label coordinate must not collide with a state auxiliary "
            f"coordinate name; got {covariance_dim + '_label'!r}."
        )
    _require_state_index(mean, state_dim, name="Arithmetic mean")
    mean_values = _materialize_numeric(mean, name="Arithmetic mean")
    if (mean_values <= 0).any():
        raise ValueError("Arithmetic mean must contain only positive values.")
    _warn_large_dense_state(mean.sizes[state_dim], stacklevel=stacklevel)

    covariance = _covariance_array(
        arithmetic_covariance,
        mean=mean,
        state_dim=state_dim,
        covariance_dim=covariance_dim,
    )
    covariance_values = np.asarray(covariance.values, dtype=np.float64)
    tolerance = _matrix_tolerance(covariance_values)
    if not np.allclose(
        covariance_values,
        covariance_values.T,
        rtol=1e-10,
        atol=tolerance,
    ):
        raise ValueError("Arithmetic covariance must be symmetric.")
    covariance_values = 0.5 * (covariance_values + covariance_values.T)
    covariance = covariance.copy(data=covariance_values)
    if float(np.linalg.eigvalsh(covariance_values).min()) < -tolerance:
        raise ValueError("Arithmetic covariance must be positive semidefinite.")
    return state_dim, covariance_dim, mean_values, covariance


class CorrelatedLognormalPrior:
    """Validated labelled moments for one correlated positive state vector.

    The arithmetic covariance is an already-reduced dense matrix. It uses a
    distinct second dimension whose entries follow the primary state coordinate
    in the same order. Only the primary axis owns the rich scientific
    coordinate, avoiding duplicate MultiIndex level coordinates while retaining
    an explicit labelled-row contract.

    Attributes:
        mean: Arithmetic mean with dimension ``(state_dim,)``. Access returns
            an independent deep copy.
        arithmetic_covariance: Dense arithmetic covariance with dimensions
            ``(state_dim, covariance_dim)``. Access returns an independent deep
            copy.
        latent_mean: Derived Gaussian mean with dimension ``(state_dim,)``.
            Access returns an independent deep copy.
        latent_covariance: Derived Gaussian covariance with dimensions
            ``(state_dim, covariance_dim)``. Access returns an independent deep
            copy.
        latent_cholesky: Lower Cholesky factor with dimensions
            ``(state_dim, covariance_dim)``. Access returns an independent deep
            copy.
        state_dim: Name of the scientific state dimension.
        covariance_dim: Name of the same-order covariance column dimension.

    Notes:
        Construction owns all supplied values. Array properties return deep
        copies so callers cannot mutate the cached validated moments through
        the public API.
    """

    _mean: xr.DataArray
    _arithmetic_covariance: xr.DataArray
    _latent_mean: xr.DataArray
    _latent_covariance: xr.DataArray
    _latent_cholesky: xr.DataArray
    state_dim: str
    covariance_dim: str

    @property
    def mean(self) -> xr.DataArray:
        """Return the validated arithmetic mean.

        Returns:
            An independent deep copy of the labelled arithmetic mean.
        """
        return self._mean.copy(deep=True)

    @property
    def arithmetic_covariance(self) -> xr.DataArray:
        """Return the validated arithmetic covariance.

        Returns:
            An independent deep copy of the dense labelled covariance.
        """
        return self._arithmetic_covariance.copy(deep=True)

    @property
    def latent_mean(self) -> xr.DataArray:
        """Return the derived latent Gaussian mean.

        Returns:
            An independent deep copy of the labelled latent mean.
        """
        return self._latent_mean.copy(deep=True)

    @property
    def latent_covariance(self) -> xr.DataArray:
        """Return the derived latent Gaussian covariance.

        Returns:
            An independent deep copy of the dense latent covariance.
        """
        return self._latent_covariance.copy(deep=True)

    @property
    def latent_cholesky(self) -> xr.DataArray:
        """Return the Cholesky factor of the latent covariance.

        Returns:
            An independent deep copy of the dense lower-triangular factor.
        """
        return self._latent_cholesky.copy(deep=True)

    def __init__(
        self,
        mean: xr.DataArray,
        arithmetic_covariance: xr.DataArray | np.ndarray,
        *,
        covariance_dim: str | None = None,
    ) -> None:
        """Validate arithmetic moments and derive cached latent moments.

        Args:
            mean: Positive one-dimensional arithmetic mean with a unique
                labelled state coordinate.
            arithmetic_covariance: Already-reduced dense arithmetic covariance.
                Its shape must be ``(p, p)`` for a length-``p`` mean. Xarray
                input must use the state dimension for rows and
                ``covariance_dim`` for columns, with matching label order;
                NumPy input is interpreted positionally in mean-label order.
            covariance_dim: Optional name for the covariance column dimension.
                Defaults to ``"{state_dim}_covariance"``.

        Raises:
            TypeError: If ``mean`` is not an xarray ``DataArray``.
            ValueError: If dimensions, labels, values, or arithmetic moments
                do not define a finite positive-definite latent covariance.

        Warns:
            UserWarning: If the reduced state contains more than 1000
                components. This is an operational threshold, not a
                mathematical limit.

        Notes:
            Inputs are eagerly computed, canonicalized, and copied. Mean and
            covariance units must be consistent (covariance has squared mean
            units); unit metadata is retained where possible but is neither
            converted nor validated.

            For arithmetic mean ``m`` and covariance ``C``, each latent
            covariance entry is
            ``Sigma[i,j] = log(1 + C[i,j] / (m[i] * m[j]))`` and
            ``mu[i] = log(m[i]) - 0.5 * Sigma[i,i]``. For a mean-one state this
            becomes ``Sigma[i,j] = log(1 + C[i,j])`` element by element; the
            implementation uses ``numpy.log1p`` for that scalar operation and
            does not take a matrix logarithm.
        """
        state_dim, covariance_dim, mean_values, covariance = _validate_arithmetic_moments(
            mean, arithmetic_covariance, covariance_dim=covariance_dim
        )
        covariance_values = np.asarray(covariance.values, dtype=np.float64)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            relative_covariance = covariance_values / np.outer(mean_values, mean_values)
        if not np.isfinite(relative_covariance).all():
            raise ValueError("Arithmetic moments must produce only finite relative covariance values.")
        if (1.0 + relative_covariance <= 0).any():
            raise ValueError("Arithmetic moments do not define finite LogNormal latent covariance entries.")
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            latent_covariance_values = np.log1p(relative_covariance)
        if not np.isfinite(latent_covariance_values).all():
            raise ValueError("Derived latent covariance must contain only finite values.")
        latent_tolerance = _matrix_tolerance(latent_covariance_values)
        if not np.allclose(
            latent_covariance_values,
            latent_covariance_values.T,
            rtol=1e-10,
            atol=latent_tolerance,
        ):
            raise ValueError("Derived latent covariance must be symmetric.")
        try:
            latent_cholesky_values = np.linalg.cholesky(latent_covariance_values)
        except np.linalg.LinAlgError as exc:
            minimum = float(np.linalg.eigvalsh(latent_covariance_values).min())
            raise ValueError(
                f"Derived latent covariance must be positive definite; minimum eigenvalue is {minimum:.6g}."
            ) from exc

        latent_mean_values = np.log(mean_values) - 0.5 * np.diag(latent_covariance_values)
        if not np.isfinite(latent_mean_values).all():
            raise ValueError("Derived latent mean must contain only finite values.")
        state_coord = mean.coords[state_dim]
        canonical_mean = _copy_state_auxiliary_coords(
            xr.DataArray(
                mean_values,
                dims=(state_dim,),
                coords={state_dim: state_coord},
                name="arithmetic_mean",
                attrs=dict(mean.attrs),
            ),
            mean,
            state_dim=state_dim,
        )
        matrix_coords = {name: coord for name, coord in canonical_mean.coords.items()}
        canonical_latent_mean = _copy_state_auxiliary_coords(
            xr.DataArray(
                latent_mean_values,
                dims=(state_dim,),
                coords={state_dim: state_coord},
                name="latent_mean",
            ),
            canonical_mean,
            state_dim=state_dim,
        )
        self._mean = canonical_mean.copy(deep=True)
        self._arithmetic_covariance = covariance.copy(deep=True)
        self._latent_mean = canonical_latent_mean.copy(deep=True)
        self._latent_covariance = xr.DataArray(
            latent_covariance_values,
            dims=(state_dim, covariance_dim),
            coords=matrix_coords,
            name="latent_covariance",
        ).copy(deep=True)
        self._latent_cholesky = xr.DataArray(
            latent_cholesky_values,
            dims=(state_dim, covariance_dim),
            coords=matrix_coords,
            name="latent_cholesky",
        ).copy(deep=True)
        self.state_dim = state_dim
        self.covariance_dim = covariance_dim

    def to_dataset(self) -> xr.Dataset:
        """Serialize the validated arithmetic and latent moments.

        Returns:
            A new labelled dataset containing deep copies of both moment
            parameterizations and the correlated-state schema metadata.
        """
        return xr.Dataset(
            {
                self._mean.name: self._mean.copy(deep=True),
                self._arithmetic_covariance.name: self._arithmetic_covariance.copy(deep=True),
                self._latent_mean.name: self._latent_mean.copy(deep=True),
                self._latent_covariance.name: self._latent_covariance.copy(deep=True),
                self._latent_cholesky.name: self._latent_cholesky.copy(deep=True),
            },
            attrs={
                "correlated_state_schema": "openghg_inversions.correlated_lognormal_prior/v1",
                "state_dim": self.state_dim,
                "covariance_dim": self.covariance_dim,
                "moment_parameterization": "arithmetic_mean_and_covariance",
                "latent_parameterization": "whitened_gaussian",
            },
        )

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> CorrelatedLognormalPrior:
        """Reload and revalidate a serialized correlated-state contract.

        Args:
            dataset: Dataset produced by :meth:`to_dataset`.

        Returns:
            A newly validated prior derived from the stored arithmetic moments.

        Raises:
            ValueError: If schema metadata, variables, dimensions, labels, or
                cached latent moments are missing or inconsistent.

        Warns:
            UserWarning: If the stored reduced state contains more than 1000
                components. This is an operational threshold, not a
                mathematical limit.

        Notes:
            Stored latent values are checked against moments recomputed from
            the eagerly materialized arithmetic inputs; they are never trusted
            as an alternative construction path.
        """
        expected_schema = "openghg_inversions.correlated_lognormal_prior/v1"
        if dataset.attrs.get("correlated_state_schema") != expected_schema:
            raise ValueError("Correlated-state dataset has an unsupported or missing schema declaration.")
        state_dim = dataset.attrs.get("state_dim")
        covariance_dim = dataset.attrs.get("covariance_dim")
        if not isinstance(state_dim, str) or not isinstance(covariance_dim, str):
            raise ValueError("Correlated-state dataset dimension metadata must contain strings.")
        missing = [
            name
            for name in (
                "arithmetic_mean",
                "arithmetic_covariance",
                "latent_mean",
                "latent_covariance",
                "latent_cholesky",
            )
            if name not in dataset
        ]
        if missing:
            raise ValueError(f"Correlated-state dataset is missing variable(s): {missing!r}.")
        if dataset["arithmetic_mean"].dims != (state_dim,):
            raise ValueError("Stored arithmetic mean dimensions do not match the schema metadata.")

        expected_dims = {
            "latent_mean": (state_dim,),
            "latent_covariance": (state_dim, covariance_dim),
            "latent_cholesky": (state_dim, covariance_dim),
        }
        arithmetic_index = _require_state_index(
            dataset["arithmetic_mean"],
            state_dim,
            name="Stored arithmetic mean",
        )
        for name, dims in expected_dims.items():
            if dataset[name].dims != dims:
                raise ValueError(f"Stored {name!r} dimensions must be {dims!r}; got {dataset[name].dims!r}.")
            derived_index = _require_state_index(
                dataset[name],
                state_dim,
                name=f"Stored {name}",
            )
            if not derived_index.equals(arithmetic_index):
                raise ValueError(f"Stored {name!r} state labels must match the arithmetic mean labels.")

        prior = cls(
            dataset["arithmetic_mean"],
            dataset["arithmetic_covariance"],
            covariance_dim=covariance_dim,
        )
        for name in ("latent_mean", "latent_covariance", "latent_cholesky"):
            stored = np.asarray(dataset[name].values)
            recomputed = np.asarray(getattr(prior, name).values)
            if stored.shape != recomputed.shape or not np.allclose(
                stored,
                recomputed,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError(f"Stored {name!r} is inconsistent with the declared arithmetic moments.")
        return prior


STATE_MARGINALS = ("lognormal", "normal", "weibull", "gamma")
"""Marginal families a :class:`CorrelatedPrior` state may use."""

_HERMITE_TERMS = 60
"""Hermite terms in the latent-correlation series for Weibull and gamma states."""

_HERMITE_NODES, _HERMITE_WEIGHTS = np.polynomial.hermite_e.hermegauss(300)
_HERMITE_WEIGHTS = _HERMITE_WEIGHTS / np.sqrt(2.0 * np.pi)

_SERIES_VARIANCE_TOLERANCE = 1e-6
"""Largest relative variance the truncated series may miss for one state."""

_WEIBULL_SHAPE_BOUNDS = (0.02, 1000.0)

GAMMA_LATENT_CAP = 8.0
"""Gamma states are held at their quantile for latent ``u = 8`` above it.

TensorFlow Probability's JAX inverse incomplete gamma overflows for upper-tail
probabilities below about 1e-22 (``u`` near 10). ``Phi(-8)`` is 6e-16, so the
cap changes the moments negligibly and keeps every backend identical.
"""


def _weibull_shape(cv: np.ndarray) -> np.ndarray:
    """Solve ``Gamma(1 + 2/k) / Gamma(1 + 1/k)**2 - 1 = cv**2`` for each shape ``k``."""
    from scipy import optimize, special

    def excess(log_shape: float, target: float) -> float:
        inverse = np.exp(-log_shape)
        return float(np.expm1(special.gammaln(1 + 2 * inverse) - 2 * special.gammaln(1 + inverse)) - target)

    low, high = np.log(_WEIBULL_SHAPE_BOUNDS[0]), np.log(_WEIBULL_SHAPE_BOUNDS[1])
    shapes = np.empty(cv.shape)
    for index, value in enumerate(cv):
        target = float(value) ** 2
        if not excess(high, target) < 0 < excess(low, target):
            raise ValueError(
                f"Weibull marginal cannot reach coefficient of variation {float(value):.6g}; "
                f"its shape would lie outside {_WEIBULL_SHAPE_BOUNDS!r}."
            )
        shapes[index] = np.exp(optimize.brentq(excess, low, high, args=(target,), xtol=1e-14, rtol=1e-14))
    return shapes


def _marginal_parameters(family: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each state's ``(shape, scale)`` for its family, mean and SD.

    Lognormal: log-scale SD and median. Normal: no shape (NaN) and the SD.
    Gamma and Weibull: the usual shape and scale.
    """
    from scipy import special

    cv = sd / mean
    shape = np.full(mean.shape, np.nan)
    scale = np.full(mean.shape, np.nan)
    lognormal = family == "lognormal"
    shape[lognormal] = np.sqrt(np.log1p(np.square(cv[lognormal])))
    scale[lognormal] = mean[lognormal] * np.exp(-0.5 * np.square(shape[lognormal]))
    normal = family == "normal"
    scale[normal] = sd[normal]
    gamma = family == "gamma"
    shape[gamma] = 1.0 / np.square(cv[gamma])
    scale[gamma] = mean[gamma] * np.square(cv[gamma])
    weibull = family == "weibull"
    if weibull.any():
        shape[weibull] = _weibull_shape(cv[weibull])
        scale[weibull] = mean[weibull] / special.gamma(1.0 + 1.0 / shape[weibull])
    return shape, scale


def _marginal_quantile(
    family: np.ndarray, mean: np.ndarray, shape: np.ndarray, scale: np.ndarray, latent: np.ndarray
) -> np.ndarray:
    """Map standard-normal latent values ``u`` (last axis = state) to ``F^-1(Phi(u))``."""
    from scipy import special

    latent = np.asarray(latent, dtype=np.float64)
    result = np.empty(np.broadcast_shapes(latent.shape, mean.shape))
    for name in np.unique(family):
        index = np.flatnonzero(family == name)
        u = latent[..., index]
        k, theta = shape[index], scale[index]
        if name == "lognormal":
            values = theta * np.exp(k * u)
        elif name == "normal":
            values = mean[index] + theta * u
        elif name == "gamma":
            u = np.minimum(u, GAMMA_LATENT_CAP)
            with np.errstate(over="ignore", under="ignore"):
                values = theta * np.where(
                    u > 0,
                    special.gammainccinv(k, special.ndtr(-np.maximum(u, 0))),
                    special.gammaincinv(k, special.ndtr(np.minimum(u, 0))),
                )
        else:
            with np.errstate(over="ignore", under="ignore", divide="ignore"):
                values = theta * np.power(-special.log_ndtr(-u), 1.0 / k)
        result[..., index] = values
    return result


def _hermite_coefficients(
    family: np.ndarray, mean: np.ndarray, sd: np.ndarray, shape: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Return normalized Hermite coefficients ``E[F^-1(Phi(U)) He_k(U)] / sqrt(k!)``, k >= 1.

    Lognormal and normal coefficients are exact; Weibull and gamma use
    Gauss-Hermite quadrature.
    """
    from scipy import special

    order = np.arange(1, _HERMITE_TERMS + 1)
    coefficients = np.zeros((mean.size, _HERMITE_TERMS))
    lognormal = family == "lognormal"
    if lognormal.any():
        coefficients[lognormal] = mean[lognormal, None] * np.exp(
            order * np.log(shape[lognormal, None]) - 0.5 * special.gammaln(order + 1.0)
        )
    coefficients[family == "normal", 0] = sd[family == "normal"]
    series = np.isin(family, ("weibull", "gamma"))
    if series.any():
        values = _marginal_quantile(
            family[series], mean[series], shape[series], scale[series],
            np.broadcast_to(_HERMITE_NODES[:, None], (_HERMITE_NODES.size, int(series.sum()))),
        ).T
        weighted = values * _HERMITE_WEIGHTS
        previous, current = np.ones_like(_HERMITE_NODES), _HERMITE_NODES.copy()
        for k in order:
            coefficients[series, k - 1] = weighted @ current
            previous, current = current, (_HERMITE_NODES * current - np.sqrt(k) * previous) / np.sqrt(k + 1.0)
        captured = np.square(coefficients[series]).sum(axis=1) / np.square(sd[series])
        worst = int(np.argmax(np.abs(captured - 1.0)))
        if abs(captured[worst] - 1.0) > _SERIES_VARIANCE_TOLERANCE:
            raise ValueError(
                "A Weibull or gamma state is too skewed for the copula series: coefficient of "
                f"variation {float((sd / mean)[series][worst]):.6g} keeps only "
                f"{float(captured[worst]):.6g} of its variance in {_HERMITE_TERMS} terms."
            )
    return coefficients


def _solve_latent_correlation(
    family: np.ndarray,
    mean: np.ndarray,
    shape: np.ndarray,
    coefficients: np.ndarray,
    covariance: np.ndarray,
    labels: list[Any],
) -> np.ndarray:
    """Return the latent correlation that reproduces ``covariance`` through the marginals.

    Pairs with a normal state, or two lognormal states, have closed forms.
    Other pairs solve ``sum_k c_ik c_jk r**k = C_ij`` by bisection; the series
    is nondecreasing in ``r`` because every marginal map is increasing.

    Raises:
        ValueError: If a pair's covariance is outside what its marginals can
            reach, or the latent correlation is not positive definite.
    """
    size = mean.size
    rows, cols = np.triu_indices(size, 1)
    target = covariance[rows, cols]
    first, second = family[rows], family[cols]
    correlation = np.zeros(target.shape)
    infeasible = np.zeros(target.shape, dtype=bool)
    nonzero = target != 0

    linear = nonzero & ((first == "normal") | (second == "normal"))
    correlation[linear] = target[linear] / (coefficients[rows[linear], 0] * coefficients[cols[linear], 0])

    both = nonzero & (first == "lognormal") & (second == "lognormal")
    relative = target[both] / (mean[rows[both]] * mean[cols[both]])
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation[both] = np.log1p(relative) / (shape[rows[both]] * shape[cols[both]])
    infeasible[both] = relative <= -1

    series = nonzero & ~linear & ~both
    if series.any():
        # Highest order first, one row per order, for Horner evaluation.
        product = np.ascontiguousarray((coefficients[rows[series]] * coefficients[cols[series]]).T[::-1])

        def pair_covariance(r: np.ndarray) -> np.ndarray:
            value = product[0].copy()
            for row in product[1:]:
                value *= r
                value += row
            return value * r

        wanted = target[series]
        low, high = -np.ones(wanted.shape), np.ones(wanted.shape)
        infeasible[series] = (wanted < pair_covariance(low)) | (wanted > pair_covariance(high))
        for _ in range(64):
            middle = 0.5 * (low + high)
            above = pair_covariance(middle) < wanted
            low, high = np.where(above, middle, low), np.where(above, high, middle)
        correlation[series] = 0.5 * (low + high)

    infeasible |= ~np.isfinite(correlation) | (np.abs(correlation) > 1 + 1e-12)
    if infeasible.any():
        pairs = [
            f"({labels[i]!r}, {labels[j]!r})"
            for i, j in zip(rows[infeasible][:5], cols[infeasible][:5], strict=True)
        ]
        raise ValueError(
            f"No latent correlation reproduces the arithmetic covariance of {int(infeasible.sum())} "
            f"state pair(s) with their marginals, e.g. {', '.join(pairs)}."
        )
    result = np.eye(size)
    result[rows, cols] = result[cols, rows] = np.clip(correlation, -1.0, 1.0)
    return result


class CorrelatedPrior:
    """Correlated state prior given by arithmetic moments and per-state marginals.

    The prior is a Gaussian copula. A whitened standard-normal vector ``z`` is
    correlated as ``u = L z`` and each state is ``x_i = F_i^-1(Phi(u_i))``,
    where ``F_i`` is the state's marginal with its arithmetic mean and SD. The
    latent correlations ``L L^T`` are chosen pair by pair so that the
    arithmetic covariance is matched exactly (the NORTA construction).
    Changing a marginal changes only the shape of that state's distribution.

    Marginals (:data:`STATE_MARGINALS`):

    - ``"lognormal"`` (default). With it on every state, the copula is the
      :class:`CorrelatedLognormalPrior` construction, available as
      ``lognormal``, and models are built exactly as before.
    - ``"normal"``: a Gaussian state, which may be negative.
    - ``"weibull"`` and ``"gamma"``: positive and less right-skewed than the
      lognormal at the same mean and SD, so a posterior constrained only in
      a sum of states is pulled less towards small values.

    Pairs with a normal state, and lognormal pairs, have closed-form latent
    correlations. Weibull and gamma pairs solve a Hermite series. A pair whose
    covariance no latent correlation can reach, or a latent correlation that
    is not positive definite, is rejected without repair.

    Attributes:
        lognormal: Validated lognormal copula when every marginal is
            lognormal, otherwise ``None``.
        marginal: Marginal family name on the state dimension, in mean-label
            order. Treat it as read-only.
    """

    lognormal: CorrelatedLognormalPrior | None
    marginal: xr.DataArray

    def __init__(
        self,
        mean: xr.DataArray,
        arithmetic_covariance: xr.DataArray | np.ndarray,
        *,
        marginal: str | xr.DataArray = "lognormal",
        covariance_dim: str | None = None,
    ) -> None:
        """Validate the arithmetic moments and marginals and solve the copula.

        Args:
            mean: Positive one-dimensional arithmetic mean with a unique
                labelled state coordinate.
            arithmetic_covariance: Dense arithmetic covariance, as accepted by
                :class:`CorrelatedLognormalPrior`.
            marginal: One family name for every state, or names on the state
                dimension with the mean's labels.
            covariance_dim: Optional name for the covariance column dimension.

        Raises:
            TypeError: If ``mean`` is not an xarray ``DataArray``.
            ValueError: If a marginal is not in :data:`STATE_MARGINALS`, its
                labels differ from the mean's, a state has zero variance
                under a non-lognormal marginal, or no positive-definite latent
                correlation reproduces the arithmetic covariance.
        """
        names = marginal if isinstance(marginal, str) else np.asarray(marginal.values).astype(str)
        if np.all(np.asarray(names) == "lognormal"):
            self._set(CorrelatedLognormalPrior(mean, arithmetic_covariance, covariance_dim=covariance_dim), marginal)
            return
        state_dim, covariance_dim, mean_values, covariance = _validate_arithmetic_moments(
            mean, arithmetic_covariance, covariance_dim=covariance_dim, stacklevel=5
        )
        canonical_mean = _copy_state_auxiliary_coords(
            xr.DataArray(
                mean_values,
                dims=(state_dim,),
                coords={state_dim: mean.coords[state_dim]},
                name="arithmetic_mean",
                attrs=dict(mean.attrs),
            ),
            mean,
            state_dim=state_dim,
        )
        self.lognormal = None
        self._mean = canonical_mean
        self._arithmetic_covariance = covariance
        self._state_dim = state_dim
        self._covariance_dim = covariance_dim
        self._set_marginal(marginal)
        family = np.asarray(self.marginal.values).astype(str)
        covariance_values = np.asarray(covariance.values, dtype=np.float64)
        sd = np.sqrt(np.clip(np.diag(covariance_values), 0.0, None))
        if (sd <= 0).any():
            raise ValueError("Every state of a copula prior must have positive arithmetic variance.")
        shape, scale = _marginal_parameters(family, mean_values, sd)
        coefficients = _hermite_coefficients(family, mean_values, sd, shape, scale)
        correlation = _solve_latent_correlation(
            family, mean_values, shape, coefficients, covariance_values,
            list(canonical_mean.indexes[state_dim]),
        )
        try:
            cholesky = np.linalg.cholesky(correlation)
        except np.linalg.LinAlgError as exc:
            minimum = float(np.linalg.eigvalsh(correlation).min())
            raise ValueError(
                "The latent correlation that reproduces the arithmetic covariance with these marginals "
                f"is not positive definite; minimum eigenvalue is {minimum:.6g}."
            ) from exc
        self._shape, self._scale = shape, scale
        self._latent_correlation, self._latent_correlation_cholesky = correlation, cholesky

    @classmethod
    def from_lognormal(cls, prior: CorrelatedLognormalPrior) -> CorrelatedPrior:
        """Use an already validated lognormal prior on every state.

        Args:
            prior: Correlated lognormal moments, reused without recomputation.

        Returns:
            The same moments with the lognormal marginal on every state.
        """
        result = cls.__new__(cls)
        result._set(prior, "lognormal")
        return result

    def _set(self, lognormal: CorrelatedLognormalPrior, marginal: str | xr.DataArray) -> None:
        """Store an all-lognormal copula and its marginal names after checking them."""
        self.lognormal = lognormal
        self._mean = lognormal._mean
        self._arithmetic_covariance = lognormal._arithmetic_covariance
        self._state_dim = lognormal.state_dim
        self._covariance_dim = lognormal.covariance_dim
        self._set_marginal(marginal)
        latent_sd = np.sqrt(np.diag(lognormal._latent_covariance.values))
        self._shape = latent_sd
        self._scale = np.exp(lognormal._latent_mean.values)
        self._latent_correlation = lognormal._latent_covariance.values / np.outer(latent_sd, latent_sd)
        self._latent_correlation_cholesky = lognormal._latent_cholesky.values / latent_sd[:, None]

    def _set_marginal(self, marginal: str | xr.DataArray) -> None:
        """Store the marginal names on the mean's state labels after checking them."""
        mean = self._mean
        if isinstance(marginal, str):
            marginal = mean.copy(data=np.full(mean.size, marginal, dtype=object))
        elif marginal.dims != (self._state_dim,) or not marginal.indexes[self._state_dim].equals(
            mean.indexes[self._state_dim]
        ):
            raise ValueError("Marginal families must use the prior mean's state labels in the same order.")
        unsupported = sorted(set(map(str, marginal.values)) - set(STATE_MARGINALS))
        if unsupported:
            raise ValueError(f"Unsupported state marginal(s) {unsupported!r}; available: {STATE_MARGINALS!r}.")
        self.marginal = mean.copy(data=np.asarray(marginal.values).astype(str).astype(object)).rename("marginal")
        self.marginal.attrs = {}

    @property
    def mean(self) -> xr.DataArray:
        """Return the arithmetic mean.

        Returns:
            An independent deep copy of the labelled arithmetic mean.
        """
        return self._mean.copy(deep=True)

    @property
    def arithmetic_covariance(self) -> xr.DataArray:
        """Return the arithmetic covariance.

        Returns:
            An independent deep copy of the dense labelled covariance.
        """
        return self._arithmetic_covariance.copy(deep=True)

    @property
    def state_dim(self) -> str:
        """Return the name of the state dimension."""
        return self._state_dim

    @property
    def covariance_dim(self) -> str:
        """Return the name of the same-order covariance column dimension."""
        return self._covariance_dim

    @property
    def is_lognormal(self) -> bool:
        """Whether every state has the lognormal marginal."""
        return self.lognormal is not None

    def _state_matrix(self, values: np.ndarray, name: str) -> xr.DataArray:
        coords = {key: coord for key, coord in self._mean.coords.items()}
        return xr.DataArray(values.copy(), dims=(self._state_dim, self._covariance_dim), coords=coords, name=name)

    @property
    def latent_correlation(self) -> xr.DataArray:
        """Return the copula's latent correlation ``L L^T``.

        Returns:
            A dense labelled copy on ``(state_dim, covariance_dim)``.
        """
        return self._state_matrix(self._latent_correlation, "latent_correlation")

    @property
    def latent_correlation_cholesky(self) -> xr.DataArray:
        """Return the lower Cholesky factor ``L`` of the latent correlation.

        Returns:
            A dense labelled copy on ``(state_dim, covariance_dim)``.
        """
        return self._state_matrix(self._latent_correlation_cholesky, "latent_correlation_cholesky")

    @property
    def marginal_shape(self) -> xr.DataArray:
        """Return each state's marginal shape.

        Lognormal: SD of the log state. Weibull and gamma: the shape ``k``.
        Normal: NaN.
        """
        return self._mean.copy(data=self._shape.copy()).rename("marginal_shape").assign_attrs(units="1")

    @property
    def marginal_scale(self) -> xr.DataArray:
        """Return each state's marginal scale, in the mean's units.

        Lognormal: the median. Normal: the SD. Weibull and gamma: the scale.
        """
        return self._mean.copy(data=self._scale.copy()).rename("marginal_scale")

    def transform_latent(self, latent: np.ndarray) -> np.ndarray:
        """Map correlated standard-normal values ``u`` to states ``F^-1(Phi(u))``.

        Args:
            latent: Array whose last axis is the state, in mean-label order,
                holding ``u = L z``.

        Returns:
            The states, with the shape of ``latent``.
        """
        family = np.asarray(self.marginal.values).astype(str)
        return _marginal_quantile(family, self._mean.values, self._shape, self._scale, latent)

    def sample(self, size: int, *, rng: np.random.Generator | int | None = None) -> np.ndarray:
        """Draw states from the prior with NumPy and SciPy.

        Args:
            size: Number of draws.
            rng: Generator or seed.

        Returns:
            Draws with shape ``(size, n_state)``, in mean-label order.
        """
        generator = np.random.default_rng(rng)
        whitened = generator.standard_normal((size, self._mean.size))
        return self.transform_latent(whitened @ self._latent_correlation_cholesky.T)
