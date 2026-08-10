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

The main entry point is ``CorrelatedLognormalPrior``. Its constructor and
``from_moments`` eagerly compute xarray inputs and own independent copies;
``to_dataset`` and ``from_dataset`` provide persistence boundaries. Construction
warns before dense covariance materialization when the reduced state exceeds
1,000 components.
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


def _warn_large_dense_state(state_size: int) -> None:
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
            stacklevel=3,
        )


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
        _warn_large_dense_state(mean.sizes[state_dim])

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

    @classmethod
    def from_moments(
        cls,
        mean: xr.DataArray,
        arithmetic_covariance: xr.DataArray | np.ndarray,
        *,
        covariance_dim: str | None = None,
    ) -> CorrelatedLognormalPrior:
        """Construct a prior from labelled arithmetic moments.

        Args:
            mean: Positive one-dimensional arithmetic mean with a unique
                labelled state coordinate.
            arithmetic_covariance: Already-reduced dense ``(p, p)`` arithmetic
                covariance. Xarray labels must match ``mean`` exactly; NumPy
                values are interpreted in mean-label order.
            covariance_dim: Optional name for the covariance column dimension;
                see the constructor for the complete coordinate contract.

        Returns:
            A validated contract containing owned arithmetic moments and the
            derived latent Gaussian moments.

        Raises:
            TypeError: If ``mean`` is not an xarray ``DataArray``.
            ValueError: If dimensions, labels, values, or arithmetic moments
                fail validation.

        Warns:
            UserWarning: If the reduced state contains more than 1000
                components. This is an operational threshold, not a
                mathematical limit.
        """
        return cls(
            mean,
            arithmetic_covariance,
            covariance_dim=covariance_dim,
        )

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

        prior = cls.from_moments(
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
