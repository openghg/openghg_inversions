from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from openghg_inversions.hbmcmc.inversion_pymc import parse_prior, PriorArgs


class ModelComponent(ABC):
    @abstractmethod
    def build(self) -> pm.Model:
        """Construct a sub-model to add variables within a main model."""
        pass


class FluxForwardModel(ModelComponent):
    """Flux component of forward model."""

    def __init__(
        self,
        name: str,
        h_matrix: xr.DataArray | np.ndarray,
        prior_args: PriorArgs,
        flux: xr.DataArray | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.name = name

        self.h_matrix = h_matrix
        self.h_matrix_values = h_matrix if isinstance(h_matrix, np.ndarray) else h_matrix.values

        self.prior_args = prior_args
        self.flux = flux  # if you didn't want to use scaling factors...

    def build(self) -> pm.Model:
        self.model = pm.Model(name=self.name)  # name used to distinguish variables created by this component

        with self.model:
            x = parse_prior("x", self.prior_args, shape=self.h_matrix_values.shape[1])
            hx = pm.Data("hx", self.h_matrix)
            pm.Deterministic("mu", pt.dot(hx, x))

        return self.model


class BCForwardModel(ModelComponent):
    """Baseline (boundary conditions) component of forward model."""

    def __init__(
        self,
        name: str,
        h_matrix: xr.DataArray | np.ndarray,
        prior_args: PriorArgs,
        bc: xr.DataArray | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.name = name

        self.h_matrix = h_matrix
        self.h_matrix_values = h_matrix if isinstance(h_matrix, np.ndarray) else h_matrix.values

        self.prior_args = prior_args
        self.bc = bc

    def build(self) -> pm.Model:
        self.model = pm.Model(name=self.name)  # name used to distinguish variables created by this component

        with self.model:
            bc = parse_prior("bc", self.prior_args, shape=self.h_matrix_values.shape[1])
            hbc = pm.Data("hbc", self.h_matrix_values)
            pm.Deterministic("mu_bc", pt.dot(hbc, bc))

        return self.model


class RHIMELikelihood(ModelComponent):
    """Likelihood for RHIME model."""

    def __init__(
        self,
        name: str,
        y_obs: np.ndarray,
        error: np.ndarray,
        sigma_prior: PriorArgs,
        min_error: np.ndarray | None = None,
        pollution_events_from_obs: bool = True,
        no_model_error: bool = False,
    ) -> None:
        super().__init__()
