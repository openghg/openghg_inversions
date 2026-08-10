"""Labelled arithmetic-moment contracts for correlated positive states.

This module is backend-neutral.  It validates the scientific state coordinate
and converts requested arithmetic LogNormal moments into the latent Gaussian
moments needed by a model backend.  PyMC graph construction lives in
``openghg_inversions.models.components``.

Correlated marginalization is deliberately separate from
``models.StateActivity``.  ``StateActivity`` conditions inactive states on
configured fixed values; selecting a principal marginal from a joint prior is
a different statistical operation and must not silently restore omitted states
as constants.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
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
        "retained_state",
    }
)


def _materialize_numeric(array: xr.DataArray, *, name: str) -> np.ndarray:
    """Return an owned array of finite real float64 values."""
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
    """Copy ordinary state-aligned metadata not recreated by the state index."""
    result = target
    for name, coord in source.coords.items():
        if name in result.coords:
            continue
        if set(coord.dims).issubset({state_dim}):
            result = result.assign_coords({name: coord.copy(deep=True)})
    return result.copy(deep=True)


def _require_state_index(array: xr.DataArray, state_dim: str, *, name: str) -> pd.Index:
    """Return the unique labelled index owned by ``state_dim``."""
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
    """Normalize covariance to one labelled and one same-order matrix axis."""
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


@dataclass(frozen=True, init=False)
class CorrelatedLognormalPrior:
    """Validated labelled moments for one correlated positive state vector.

    ``arithmetic_covariance`` uses a distinct second matrix dimension whose
    entries are declared to follow the primary state coordinate in the same
    order.  Only the primary axis owns the rich scientific coordinate.  This
    avoids duplicating MultiIndex level coordinates while retaining an explicit
    labelled row contract.
    """

    _mean: xr.DataArray
    _arithmetic_covariance: xr.DataArray
    _latent_mean: xr.DataArray
    _latent_covariance: xr.DataArray
    _latent_cholesky: xr.DataArray
    state_dim: str
    covariance_dim: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        """Require construction through ``from_moments`` validation."""
        raise TypeError("Use CorrelatedLognormalPrior.from_moments(...) to construct a prior.")

    @classmethod
    def _from_validated(
        cls,
        *,
        mean: xr.DataArray,
        arithmetic_covariance: xr.DataArray,
        latent_mean: xr.DataArray,
        latent_covariance: xr.DataArray,
        latent_cholesky: xr.DataArray,
        state_dim: str,
        covariance_dim: str,
    ) -> CorrelatedLognormalPrior:
        """Construct an instance after the public validation boundary."""
        instance = object.__new__(cls)
        object.__setattr__(instance, "_mean", mean.copy(deep=True))
        object.__setattr__(instance, "_arithmetic_covariance", arithmetic_covariance.copy(deep=True))
        object.__setattr__(instance, "_latent_mean", latent_mean.copy(deep=True))
        object.__setattr__(instance, "_latent_covariance", latent_covariance.copy(deep=True))
        object.__setattr__(instance, "_latent_cholesky", latent_cholesky.copy(deep=True))
        object.__setattr__(instance, "state_dim", state_dim)
        object.__setattr__(instance, "covariance_dim", covariance_dim)
        return instance

    @property
    def mean(self) -> xr.DataArray:
        """Return an independent copy of the validated arithmetic mean."""
        return self._mean.copy(deep=True)

    @property
    def arithmetic_covariance(self) -> xr.DataArray:
        """Return an independent copy of the validated arithmetic covariance."""
        return self._arithmetic_covariance.copy(deep=True)

    @property
    def latent_mean(self) -> xr.DataArray:
        """Return an independent copy of the derived latent mean."""
        return self._latent_mean.copy(deep=True)

    @property
    def latent_covariance(self) -> xr.DataArray:
        """Return an independent copy of the derived latent covariance."""
        return self._latent_covariance.copy(deep=True)

    @property
    def latent_cholesky(self) -> xr.DataArray:
        """Return an independent copy of the derived latent Cholesky factor."""
        return self._latent_cholesky.copy(deep=True)

    @classmethod
    def from_moments(
        cls,
        mean: xr.DataArray,
        arithmetic_covariance: xr.DataArray | np.ndarray,
        *,
        covariance_dim: str | None = None,
    ) -> CorrelatedLognormalPrior:
        """Validate arithmetic moments and derive latent Gaussian moments.

        For arithmetic mean ``m`` and covariance ``C``, the latent moments are

        ``Sigma[i,j] = log(1 + C[i,j] / (m[i] * m[j]))`` and
        ``mu[i] = log(m[i]) - 0.5 * Sigma[i,i]``.

        The familiar mean-one contract is the special case
        ``Sigma = log1p(C)``.
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
        return cls._from_validated(
            mean=canonical_mean,
            arithmetic_covariance=covariance,
            latent_mean=canonical_latent_mean,
            latent_covariance=xr.DataArray(
                latent_covariance_values,
                dims=(state_dim, covariance_dim),
                coords=matrix_coords,
                name="latent_covariance",
            ),
            latent_cholesky=xr.DataArray(
                latent_cholesky_values,
                dims=(state_dim, covariance_dim),
                coords=matrix_coords,
                name="latent_cholesky",
            ),
            state_dim=state_dim,
            covariance_dim=covariance_dim,
        )

    def select_marginal(self, retained: xr.DataArray) -> MarginalCorrelatedLognormalPrior:
        """Select a labelled principal marginal without fixing omitted states.

        This operation only marginalizes the prior. It does not reduce a
        forward operator or construct the unresolved aggregation covariance.
        Callers using coherent reduction must supply the matching retained
        design and unresolved covariance from the same preparation ledger.
        """
        if retained.dims != (self.state_dim,):
            raise ValueError(
                f"Retained-state mask must have only the {self.state_dim!r} dimension; got {retained.dims!r}."
            )
        retained_index = _require_state_index(retained, self.state_dim, name="Retained-state mask")
        state_index = self._mean.indexes[self.state_dim]
        if not retained_index.equals(state_index):
            raise ValueError(
                "Retained-state mask labels must exactly match the prior state labels in the same order."
            )
        retained_values = np.asarray(retained.compute().values)
        if retained_values.dtype != np.dtype(bool):
            raise ValueError("Retained-state mask must contain only boolean values.")
        positions = np.flatnonzero(retained_values)
        if positions.size == 0:
            raise ValueError("A correlated LogNormal marginal must retain at least one state.")

        marginal_mean = self._mean.isel({self.state_dim: positions})
        marginal_covariance = self._arithmetic_covariance.isel(
            {self.state_dim: positions, self.covariance_dim: positions}
        )
        prior = CorrelatedLognormalPrior.from_moments(
            marginal_mean,
            marginal_covariance,
            covariance_dim=self.covariance_dim,
        )
        canonical_mask = xr.DataArray(
            retained_values,
            dims=(self.state_dim,),
            coords={self.state_dim: self._mean.coords[self.state_dim]},
            name="retained_state",
        )
        canonical_mask = _copy_state_auxiliary_coords(
            canonical_mask,
            self._mean,
            state_dim=self.state_dim,
        )
        return MarginalCorrelatedLognormalPrior._from_validated(
            full_prior=self,
            retained=canonical_mask,
            prior=prior,
        )

    def to_dataset(self) -> xr.Dataset:
        """Return a serializable labelled dataset containing both moment spaces."""
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
        """Reload and revalidate a dataset produced by ``to_dataset``."""
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


@dataclass(frozen=True, init=False)
class MarginalCorrelatedLognormalPrior:
    """Explicit result of marginalizing a labelled joint prior.

    This object intentionally has no fixed-value or full-state reconstruction
    field.  Omitted states are integrated out, not conditioned on constants.
    """

    full_prior: CorrelatedLognormalPrior
    _retained: xr.DataArray
    prior: CorrelatedLognormalPrior

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        """Require construction through validated marginal selection."""
        raise TypeError("Use CorrelatedLognormalPrior.select_marginal(...) to construct a marginal prior.")

    @classmethod
    def _from_validated(
        cls,
        *,
        full_prior: CorrelatedLognormalPrior,
        retained: xr.DataArray,
        prior: CorrelatedLognormalPrior,
    ) -> MarginalCorrelatedLognormalPrior:
        """Construct the wrapper after recomputing its principal marginal."""
        instance = object.__new__(cls)
        object.__setattr__(instance, "full_prior", full_prior)
        object.__setattr__(instance, "_retained", retained.copy(deep=True))
        object.__setattr__(instance, "prior", prior)
        return instance

    @property
    def retained(self) -> xr.DataArray:
        """Return an independent copy of the validated retained-state mask."""
        return self._retained.copy(deep=True)

    @property
    def omitted(self) -> xr.DataArray:
        """Return the labelled omitted-state mask in full-state order."""
        return (~self._retained).rename("marginalized_state").copy(deep=True)

    def to_dataset(self) -> xr.Dataset:
        """Serialize the full prior and retained-state marginalization ledger."""
        dataset = self.full_prior.to_dataset()
        dataset["retained_state"] = (
            self.full_prior.state_dim,
            np.array(self._retained.values, dtype=bool, copy=True),
        )
        dataset.attrs = dict(dataset.attrs)
        dataset.attrs["inactive_state_semantics"] = "coherent_prior_marginalization"
        return dataset

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> MarginalCorrelatedLognormalPrior:
        """Reload and reapply an explicit coherent-marginalization ledger."""
        if dataset.attrs.get("inactive_state_semantics") != "coherent_prior_marginalization":
            raise ValueError("Marginal correlated-state dataset must declare coherent prior marginalization.")
        if "retained_state" not in dataset:
            raise ValueError("Marginal correlated-state dataset is missing 'retained_state'.")
        full_prior = CorrelatedLognormalPrior.from_dataset(dataset)
        return full_prior.select_marginal(dataset["retained_state"])
