"""Condition persisted CO2 country functionals on a saved fit's draws.

Staged CO2 outputs report complete native country totals when the prepared
artifact carries :attr:`~openghg_inversions.rhime.co2.co2_preparation.Co2PreparedInputs.country_functionals`.
This module rebuilds what
:func:`~openghg_inversions.postprocessing.unresolved_functionals.complete_functional_draws`
needs from the saved trace and prepared artifact alone: each posterior draw's
observation residual and the fit's own ``(A + R)^-1`` at that draw's noise
parameters.

``A`` is the aggregation-error representation the fit used, as stored in the
prepared artifact. With the default rank-512 LRPD artifact it is that
approximation, not the exact unresolved covariance of the products; prepare
with ``aggregation_error_rank=None`` to condition on the exact ``A``. The
conditional-covariance diagnostic recorded on the outputs flags a large
mismatch.
"""

from __future__ import annotations

from collections.abc import Hashable
import json

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
import xarray as xr

from openghg_inversions.coherent_reduction import NativeFunctionalProducts
from openghg_inversions.models.fixed_ou import prepare_fixed_ou_low_rank
from openghg_inversions.observation_error import (
    AggregationError,
    aggregation_error_as_low_rank,
    resolve_aggregation_error,
)
from openghg_inversions.postprocessing.co2_flux_outputs import co2_country_functionals
from openghg_inversions.postprocessing.countries import Countries
from openghg_inversions.postprocessing.inversion_output import trace_group
from openghg_inversions.postprocessing.unresolved_functionals import (
    ObservationCovarianceSolve,
    UnresolvedFunctionalConditioning,
    fixed_ou_solve,
)

from .co2_affine_output import BoundCo2AffineFluxMap
from .co2_preparation import COUNTRY_FUNCTIONAL_DIM, Co2PreparedInputs

__all__ = ["co2_unresolved_conditioning"]

FloatArray = NDArray[np.float64]
SAMPLE_DIMS = ("chain", "draw")
OU_SITE_AMPLITUDE = "ou_site_amplitude"
OU_SITE_INDEX = "ou_site_index"
OU_TAU_HOURS = "ou_tau_hours"
OU_SITE_DIM = "ou_site"
# Likelihoods whose observation covariance is the aggregation error plus one
# diagonal independent variance per draw, recorded through ``epsilon``.
_DIAGONAL_LIKELIHOODS = {
    ("openghg_inversions.models.site_sigma", "add_site_sigma_gaussian_likelihood"),
}
# Recorded independent variances may round below zero by this fraction of
# the total marginal variance before the record is treated as inconsistent.
_VARIANCE_ROUNDOFF = 1e-8


def co2_unresolved_conditioning(
    trace: xr.DataTree,
    prepared: Co2PreparedInputs,
    *,
    bound: BoundCo2AffineFluxMap,
    countries: Countries,
    seed: int = 0,
) -> UnresolvedFunctionalConditioning:
    """Return per-draw conditioning for the prepared country functionals.

    Supported likelihoods are the default additive-sigma and the site-sigma
    likelihoods, whose covariance is ``A + diag(v)`` with ``v`` recovered per
    draw as ``epsilon**2 - diag(A)``, and both fixed-OU likelihoods (fixed or
    inferred amplitudes, including the cached-sigma recipe), whose
    covariance is rebuilt from the recorded observation error, OU site
    index, time scales and each draw's site amplitudes. Other recorded
    likelihoods raise ``ValueError`` rather than claiming complete outputs.

    Products are restricted to the requested countries. States the
    preparation marginalised into ``A`` (``state_information_marginalised``)
    are declared on the result.

    Args:
        trace: Annotated CO2 trace generated from ``prepared``.
        prepared: Prepared artifact with ``country_functionals``.
        bound: Affine reconstruction bound to ``prepared``.
        countries: Requested reporting countries on the map's grid.
        seed: Seed for conditional-covariance draws.

    Returns:
        Conditioning for
        :func:`~openghg_inversions.postprocessing.co2_flux_outputs.co2_country_flux_outputs`.

    Raises:
        ValueError: If the artifact has no country functionals, a requested
            country or source was not prepared, the trace lacks the
            residual or noise records, or the likelihood is unsupported.
    """
    products = prepared.country_functionals
    if products is None:
        raise ValueError("Prepared CO2 inputs carry no country functional products.")
    source_dim = bound.affine_map.native_dims[0] if len(bound.affine_map.native_dims) == 3 else None
    requested = co2_country_functionals(bound, countries, source_dim=source_dim).indexes[COUNTRY_FUNCTIONAL_DIM]
    products = _select_functionals(products, requested)

    inputs = prepared.inv_inputs
    observations = inputs["mf"]
    roles = json.loads(trace.attrs.get("rhime_variable_roles", "{}"))
    posterior = trace_group(trace, "posterior")
    model_mean = roles.get("model_mean")
    if model_mean not in posterior:
        raise ValueError("Complete CO2 country outputs require posterior model_mean draws for residuals.")
    modelled, _ = xr.align(posterior[model_mean], observations, join="exact", copy=False)
    residual = (observations - modelled).transpose(*SAMPLE_DIMS, "nmeasure")
    residual = residual.rename("residual").assign_attrs(units=observations.attrs["units"])

    marginalised = inputs.get("state_information_marginalised")
    return UnresolvedFunctionalConditioning(
        products=products,
        residual=residual,
        solve=_observation_solve(trace, prepared, roles),
        marginalised=None if marginalised is None else marginalised.astype(bool),
        seed=seed,
    )


def _select_functionals(products: NativeFunctionalProducts, requested: pd.Index) -> NativeFunctionalProducts:
    """Return the products' rows for ``requested`` functionals, in that order.

    Rows of ``Q`` are independent functionals, so any subset of rows keeps
    exact products: their entries of ``Q m``, ``D`` and ``Q B_perp H.T``, and
    the matching block of ``Q B_perp Q.T``.
    """
    prepared_index = products.functional_mean.indexes[COUNTRY_FUNCTIONAL_DIM]
    if prepared_index.equals(requested) and list(prepared_index.names) == list(requested.names):
        return products
    if list(prepared_index.names) != list(requested.names):
        raise ValueError(
            "Requested country functionals use different levels from the prepared ones "
            f"({list(requested.names)!r} != {list(prepared_index.names)!r})."
        )
    positions = prepared_index.get_indexer(requested)
    if (positions < 0).any():
        missing = [label for label, position in zip(requested, positions, strict=True) if position < 0]
        raise ValueError(
            f"Country functionals were not prepared for {missing[:5]!r}; prepare again with these countries "
            "or report retained-only country totals."
        )
    column_dim = str(products.unresolved_covariance.dims[1])
    rows = {COUNTRY_FUNCTIONAL_DIM: positions}
    return NativeFunctionalProducts(
        functional_mean=products.functional_mean.isel(rows),
        retained_response=products.retained_response.isel(rows),
        unresolved_observation_covariance=products.unresolved_observation_covariance.isel(rows),
        unresolved_covariance=products.unresolved_covariance.isel({**rows, column_dim: positions}),
        retained_covariance=products.retained_covariance,
        effective_observation_operator=products.effective_observation_operator,
        marginalised=products.marginalised,
    )


def _observation_solve(
    trace: xr.DataTree,
    prepared: Co2PreparedInputs,
    roles: dict[str, str],
) -> ObservationCovarianceSolve:
    """Select the fit's per-draw ``(A + R)^-1`` from its recorded likelihood."""
    aggregation = resolve_aggregation_error(prepared.inv_inputs, prepared.aggregation_error_mode)
    constant = trace_group(trace, "constant_data") if "constant_data" in trace.children else xr.Dataset()
    if OU_TAU_HOURS in constant:
        return _fixed_ou_solve(trace, prepared, aggregation, constant)
    builder = trace.attrs.get("rhime_likelihood_builder")
    if builder is not None:
        identity = json.loads(builder)
        if (identity.get("module"), identity.get("qualname")) not in _DIAGONAL_LIKELIHOODS:
            raise ValueError(
                f"Complete CO2 country outputs do not support the recorded likelihood {identity!r}; "
                "supported likelihoods are additive_sigma, site_sigma and fixed_ou."
            )
    return _diagonal_solve(trace, prepared, aggregation, roles)


def _fixed_ou_solve(
    trace: xr.DataTree,
    prepared: Co2PreparedInputs,
    aggregation: AggregationError,
    constant: xr.Dataset,
) -> ObservationCovarianceSolve:
    """Rebuild the fit's fixed-OU covariance and use each draw's site amplitudes."""
    observations = prepared.inv_inputs["mf"]
    for name in ("error", OU_SITE_INDEX):
        if name not in constant:
            raise ValueError(f"Complete fixed-OU CO2 outputs require constant_data {name!r}.")
    error, site_index = xr.align(constant["error"], constant[OU_SITE_INDEX], observations, join="exact", copy=False)[:2]
    tau = constant[OU_TAU_HOURS]
    if tau.dims != (OU_SITE_DIM,):
        raise ValueError(f"Recorded {OU_TAU_HOURS!r} must have dimensions ({OU_SITE_DIM!r},).")
    labels = [str(label) for label in tau[OU_SITE_DIM].values]
    factor, aggregation_diagonal = aggregation_error_as_low_rank(aggregation)
    covariance = prepare_fixed_ou_low_rank(
        factor,
        np.square(np.asarray(error.values, dtype=np.float64)) + aggregation_diagonal,
        np.asarray(observations["time"].values),
        np.asarray(site_index.values, dtype=np.int64),
        np.asarray(tau.values, dtype=np.float64),
        site_labels=labels,
    )
    posterior = trace_group(trace, "posterior")
    if OU_SITE_AMPLITUDE in posterior:
        amplitude = posterior[OU_SITE_AMPLITUDE]
        description = "prepared A + recorded observation error + fixed-tau OU at each draw's site amplitudes"
    elif OU_SITE_AMPLITUDE in constant:
        state = posterior[json.loads(trace.attrs["rhime_variable_roles"])["flux_scale"]]
        amplitude = constant[OU_SITE_AMPLITUDE].expand_dims(
            {dim: state[dim].values for dim in SAMPLE_DIMS}
        )
        description = "prepared A + recorded observation error + fixed-tau OU at fixed site amplitudes"
    else:
        raise ValueError(f"Complete fixed-OU CO2 outputs require {OU_SITE_AMPLITUDE!r} draws or constants.")
    amplitude = amplitude.assign_coords({OU_SITE_DIM: [str(label) for label in amplitude[OU_SITE_DIM].values]})
    return fixed_ou_solve(
        covariance,
        amplitude,
        site_dim=OU_SITE_DIM,
        observation_index=observations.indexes["nmeasure"],
        description=f"{description} ({prepared.aggregation_error_mode} A)",
    )


def _diagonal_solve(
    trace: xr.DataTree,
    prepared: Co2PreparedInputs,
    aggregation: AggregationError,
    roles: dict[str, str],
) -> ObservationCovarianceSolve:
    """Return ``(A + diag(v))^-1`` with ``v = epsilon**2 - diag(A)`` for each draw."""
    observations = prepared.inv_inputs["mf"]
    posterior = trace_group(trace, "posterior")
    name = roles.get("model_error")
    if name not in posterior:
        raise ValueError("Complete CO2 country outputs require posterior model_error (epsilon) draws.")
    epsilon, _ = xr.align(posterior[name], observations, join="exact", copy=False)
    epsilon = epsilon.transpose(*SAMPLE_DIMS, "nmeasure")
    total = np.square(np.asarray(epsilon.values, dtype=np.float64))
    independent = total - aggregation.marginal_variance[None, None, :]
    if np.any(independent < -_VARIANCE_ROUNDOFF * total):
        raise ValueError(
            "Recorded epsilon is smaller than the prepared aggregation error; the trace was not generated "
            "from this preparation's A."
        )
    independent = np.clip(independent, 0.0, None)
    chains = {label: position for position, label in enumerate(epsilon["chain"].values.tolist())}
    draws = {label: position for position, label in enumerate(epsilon["draw"].values.tolist())}
    index = observations.indexes["nmeasure"]
    varies = not np.all(independent == independent[:1, :1])
    description = f"prepared {prepared.aggregation_error_mode} A + recorded diagonal independent variance per draw"
    if not varies:
        factorized = _diagonal_update(aggregation, independent[0, 0])
        return ObservationCovarianceSolve(
            index, lambda rhs, chain, draw: factorized(rhs), varies_by_draw=False, description=description
        )

    def solve(rhs: FloatArray, chain: Hashable, draw: Hashable) -> FloatArray:
        """Solve with this draw's independent variance."""
        try:
            variance = independent[chains[chain], draws[draw]]
        except KeyError as error:
            raise ValueError(f"No epsilon draw for chain {chain!r}, draw {draw!r}.") from error
        return _diagonal_update(aggregation, variance)(rhs)

    return ObservationCovarianceSolve(index, solve, varies_by_draw=True, description=description)


def _diagonal_update(aggregation: AggregationError, variance: FloatArray):
    """Factorize ``A + diag(variance)`` once and return its solve.

    Dense ``A`` uses a Cholesky factor. Low-rank ``A = F F.T + diag(d)`` uses
    the Woodbury identity with ``D = d + variance``, at ``O(n k^2)`` cost.
    """
    if aggregation.mode == "dense":
        assert aggregation.covariance is not None
        covariance = np.asarray(aggregation.covariance.values, dtype=np.float64) + np.diag(variance)
        factor = cho_factor(covariance, lower=True)
        return lambda rhs: np.asarray(cho_solve(factor, rhs), dtype=np.float64)
    if aggregation.mode == "low_rank":
        assert aggregation.factor is not None and aggregation.diagonal_variance is not None
        low_rank = np.asarray(aggregation.factor.values, dtype=np.float64)
        diagonal = np.asarray(aggregation.diagonal_variance.values, dtype=np.float64) + variance
    else:
        low_rank = np.empty((variance.size, 0))
        diagonal = aggregation.marginal_variance + variance
    if np.any(diagonal <= 0.0):
        raise ValueError("The diagonal part of A + R must be positive to condition country functionals.")
    weight = 1.0 / diagonal
    weighted = low_rank * weight[:, None]
    capacitance = cho_factor(np.eye(low_rank.shape[1]) + low_rank.T @ weighted, lower=True)

    def solve(rhs: FloatArray) -> FloatArray:
        """Apply the Woodbury inverse."""
        scaled = rhs * weight[:, None] if rhs.ndim == 2 else rhs * weight
        if not low_rank.shape[1]:
            return scaled
        return scaled - weighted @ cho_solve(capacitance, low_rank.T @ scaled)

    return solve
