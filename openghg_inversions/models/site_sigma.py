r"""Labelled run-level site mismatch scales for Gaussian observations.

This component ports the site-sigma likelihood used by Verification Games in
``src/verification_games/rhime_calibration/site_sigma.py`` at commit
``41d061aea153ddc56130694bfa18b7e801fcd9df`` (original model commit
``88f8d4cb21c7eb84b601c26fa51e806ff0bb3ed7``). For fixed aggregation
covariance :math:`A`, reported
observation-error covariance :math:`D_{obs}`, and one mismatch amplitude per
site, it constructs

.. math::

   R = A + D_{obs} + \operatorname{diag}(\sigma_{site(i)}^2).

Site labels retain their stable first-occurrence order. Amplitudes use the same
units as the observations and reported errors; they are not dimensionless
multipliers. Verification Games uses an explicit ``HalfNormal(0.75)`` prior in
ppm for its benchmark, but that scientific choice is deliberately not a
universal OpenGHG Inversions default. Callers must provide either a fixed
mapping covering the observed sites or an explicit positive-support prior.

The concrete component owns only the site-mismatch term. Fixed aggregation
covariance representation and PyMC likelihood mechanics remain owned by
:mod:`openghg_inversions.models._gaussian_observation`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.array_ops import expand_mapping
from openghg_inversions.models._gaussian_observation import (
    add_aggregation_error_data,
    add_gaussian_observation_likelihood,
)
from openghg_inversions.models.components import add_model_data
from openghg_inversions.models.coords import add_coords
from openghg_inversions.models.priors import parse_prior, positive_prior_args
from openghg_inversions.observation_error import (
    AggregationError,
    validate_observation_error_arrays,
)
from openghg_inversions.sigma import SigmaAlignment


SITE_SIGMA_DIM = "sigma_site_dim"
SITE_SIGMA = "sigma_site"
SITE_SIGMA_INDEX = "sigma_site_index"
SIGMA_OBSERVATION = "sigma_observation"


class SiteSigmaGaussianLikelihood:
    """Callable site-sigma component and its static output contract."""

    rhime_metadata = {
        "mismatch_component": "iid_site_sigma",
        "residual_covariance": "A + D_obs + diag(sigma_site[site(i)]^2)",
        "site_order": "stable_first_occurrence",
        "verification_games_source": (
            "src/verification_games/rhime_calibration/site_sigma.py@41d061aea153ddc56130694bfa18b7e801fcd9df"
        ),
        "verification_games_original_model_commit": "88f8d4cb21c7eb84b601c26fa51e806ff0bb3ed7",
        "variable_roles": {
            "site_iid_mismatch_standard_deviation": SITE_SIGMA,
            "observation_to_site_sigma_index": SITE_SIGMA_INDEX,
            "observation_aligned_site_iid_mismatch_standard_deviation": SIGMA_OBSERVATION,
            "total_marginal_observation_standard_deviation": "epsilon",
            "observed_concentration": "y",
        },
        "variable_units": {SITE_SIGMA_INDEX: "1"},
    }

    def __call__(
        self,
        *,
        observations: xr.DataArray,
        observation_error: xr.DataArray,
        aggregation_error: AggregationError,
        mean: TensorVariable,
        fixed_site_amplitudes: Mapping[str, float] | None = None,
        site_amplitude_prior: Mapping[str, Any] | None = None,
        output_dim: str = "nmeasure",
        observation_error_name: str = "error",
    ) -> TensorVariable:
        """Add a Gaussian likelihood with one additive mismatch scale per site.

        Exactly one amplitude mode is required. Fixed amplitudes are supplied as a
        mapping containing the observation sites. Inferred amplitudes
        require an explicit positive-support prior from the Verification Games set:
        HalfNormal, HalfStudentT, Exponential, Gamma, LogNormal, or a Uniform with a
        non-negative lower bound.

        Args:
            observations: One-dimensional observed mole fractions whose sole
                dimension is ``output_dim`` and which carry a one-dimensional
                ``site`` coordinate on that dimension.
            observation_error: One-dimensional, non-negative reported
                observation-error standard deviations in the same units and
                observation order as ``observations``.
            aggregation_error: Validated fixed aggregation-error representation
                for the same observation order. Its diagonal, dense, or low-rank
                covariance is included exactly once.
            mean: Completed one-dimensional forward-model concentration tensor
                aligned with ``output_dim`` and ``observations``.
            fixed_site_amplitudes: Fixed amplitudes keyed by site label, in the same
                units as ``observations``. Entries for sites absent after filtering
                are ignored.
            site_amplitude_prior: Explicit positive-support prior for inferred site
                amplitudes, in the same units as ``observations``.
            output_dim: Observation dimension used by named PyMC variables.
            observation_error_name: PyMC data name for reported error.

        Returns:
            The observed Gaussian variable named ``y``. The graph also records the
            labelled ``sigma_site`` vector, observation lookup, aligned amplitudes,
            and total marginal scale ``epsilon``.

        Raises:
            ValueError: If the observations, site labels, amplitude mode, fixed
                mapping, prior support, or complete fixed covariance is invalid.

        Notes:
            Call this component inside
            :func:`openghg_inversions.models.registered_model` so scientific site
            labels are retained in the coordinate registry. The observation values
            are eagerly materialized at the named PyMC graph boundary; the input
            xarray objects are borrowed and are not mutated.
        """
        if (fixed_site_amplitudes is None) == (site_amplitude_prior is None):
            raise ValueError("Pass exactly one of `fixed_site_amplitudes` or `site_amplitude_prior`.")
        validate_observation_error_arrays(
            observations,
            observation_error,
            None,
            owner="Site-sigma likelihood",
            output_dim=output_dim,
        )
        alignment = SigmaAlignment.from_observations(observations)
        site_index = alignment.site_index
        site_coord = alignment.site_labels.rename({"nsigma_site": SITE_SIGMA_DIM}).rename(
            SITE_SIGMA_DIM
        )

        if fixed_site_amplitudes is not None:
            if any(isinstance(value, (bool, np.bool_)) for value in fixed_site_amplitudes.values()):
                raise ValueError("Fixed site amplitudes must be numeric, not boolean.")
            fixed_sigma = expand_mapping(
                fixed_site_amplitudes,
                site_coord,
                name=SITE_SIGMA,
            )
            try:
                fixed_sigma = fixed_sigma.astype(np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError("Fixed site amplitudes must be numeric.") from error
            amplitude_values = np.asarray(fixed_sigma.values)
            if not np.isfinite(amplitude_values).all() or (amplitude_values < 0.0).any():
                raise ValueError(
                    "`fixed_site_amplitudes` must contain only finite, non-negative values."
                )
            aligned_variance = np.square(amplitude_values[np.asarray(site_index.values)])
            fixed_independent_variance = (
                np.square(np.asarray(observation_error.values, dtype=np.float64)) + aligned_variance
            )
            if aggregation_error.mode == "low_rank":
                assert aggregation_error.diagonal_variance is not None
                low_rank_diagonal = fixed_independent_variance + np.asarray(
                    aggregation_error.diagonal_variance.values,
                    dtype=np.float64,
                )
                if (low_rank_diagonal <= 0.0).any():
                    raise ValueError(
                        "The stock low-rank Gaussian likelihood requires strictly "
                        "positive independent-plus-residual diagonal variance; a "
                        "low-rank factor cannot rescue a zero diagonal in this backend."
                    )
        reported_error = add_model_data(observation_error, observation_error_name)
        index_data = add_model_data(site_index)
        registered_aggregation_error = add_aggregation_error_data(
            aggregation_error,
            observations,
            output_dim=output_dim,
        )

        if fixed_site_amplitudes is not None:
            sigma_site = add_model_data(fixed_sigma)
        else:
            assert site_amplitude_prior is not None
            add_coords({SITE_SIGMA_DIM: site_coord})
            sigma_site = parse_prior(
                SITE_SIGMA,
                positive_prior_args(site_amplitude_prior),
                dims=SITE_SIGMA_DIM,
            )

        sigma_observation = pm.Deterministic(
            SIGMA_OBSERVATION,
            sigma_site[index_data],
            dims=output_dim,
        )
        independent_variance = reported_error**2 + pt.square(sigma_observation)
        pm.Deterministic(
            "epsilon",
            pt.sqrt(independent_variance + registered_aggregation_error.marginal_variance),
            dims=output_dim,
        )
        return add_gaussian_observation_likelihood(
            observed=pm.floatX(observations.compute().values),
            mean=mean,
            independent_variance=independent_variance,
            aggregation_error=registered_aggregation_error,
            output_dim=output_dim,
        )


add_site_sigma_gaussian_likelihood = SiteSigmaGaussianLikelihood()


__all__ = [
    "SIGMA_OBSERVATION",
    "SITE_SIGMA",
    "SITE_SIGMA_DIM",
    "SITE_SIGMA_INDEX",
    "add_site_sigma_gaussian_likelihood",
]
