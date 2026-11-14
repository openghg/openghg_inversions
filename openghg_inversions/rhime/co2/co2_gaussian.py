r"""Exact fixed-noise Gaussian estimation for the single-tracer CO2 recipe.

Selected with ``[sampling] method = "gaussian"``, this estimator replaces NUTS
for a linear-Gaussian version of the ordinary CO2 graph:

- the retained flux state has a Gaussian prior with the prepared arithmetic
  moments ``alpha_prior_mean`` and ``alpha_prior_covariance`` (the same
  moments the default correlated LogNormal prior matches), built as a
  whitened state from a PSD factor of the covariance;
- boundary and offset states keep their configured priors, which must be
  Normal;
- the likelihood is the additive-Gaussian likelihood with fixed error,
  :math:`W = A + R` with :math:`R = \mathrm{diag}(s_y^2 + s_{fixed}^2)` and
  the prepared aggregation covariance :math:`A`.

The graph is built by the same CO2 builder as for NUTS. Its likelihood mean
is read as an affine map of the free Normal variables and checked to be
affine, so the intercept, boundary representation and offsets enter exactly
as in the sampled model. :func:`~openghg_inversions.inference.gls.paige_gls`
solves the posterior from a factor :math:`F_W` of :math:`W` without inverting
or whitening it: :math:`[L_A \mid \mathrm{diag}(\sqrt{R + D})]` for a low-rank
plus diagonal :math:`A`, and the Cholesky factor of :math:`A + R` for a dense
:math:`A` (clipped eigenvalues of :math:`A` only when that fails; no positive
mode is dropped). Independent exact draws then pass through the same graph, so the
trace has the NUTS layout, roles and units, and existing replay and products
run unchanged. A ``gaussian_posterior`` group records the exact moments.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import logging
import secrets
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytensor
import xarray as xr

from openghg_inversions.biosphere_prior import BiospherePrior
from openghg_inversions.inference.exact_gaussian import (
    affine_maps, exact_gaussian_trace, normal_variables, solve_linear_gaussian_model,
)
from openghg_inversions.inference.gls import psd_factor
from openghg_inversions.inference.sampling import RhimeSampler
from openghg_inversions.models.boundary import AdditiveChoice, AnomalyScalingChoice, MultiplicativeChoice
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.builders import RhimeModelBuildResult

from .co2_model import GaussianStatePrior
from .co2_preparation import Co2PreparedInputs
from .co2_runner import annotate_co2_trace, build_co2_from_materialized, materialize_co2_inputs
from .model_mismatch import EmpiricalBayesMismatch, empirical_bayes_mismatch, observation_sites, per_site_mismatch

logger = logging.getLogger(__name__)

STATE_PRIOR_SUBSTITUTION = "lognormal→normal (moment-matched)"
"""Recorded when the default correlated LogNormal flux prior is replaced by its moment-matched Normal."""

GAUSSIAN_ALGORITHM = "gaussian_paige_gls"
"""``rhime_actual_inference`` algorithm name of the exact Gaussian estimator."""

GAUSSIAN_POSTERIOR_GROUP = "gaussian_posterior"
"""Trace group holding the exact Gaussian moments."""

_MODE_DIM = "flux_scaling_mode"
_ALTERNATIVE = 'or use method = "nuts"'


def _require_normal(prior: Mapping[str, Any] | None, label: str) -> None:
    """Refuse a configured non-Normal prior before any array is read."""
    if prior is None:
        return
    pdf = str(prior.get("pdf", "")).casefold().replace("-", "").replace("_", "")
    if pdf != "normal":
        raise ValueError(
            f'method = "gaussian" requires a Normal {label} prior, not {prior.get("pdf")!r}; '
            f'set its pdf to "normal" {_ALTERNATIVE}.'
        )


def require_gaussian_precision() -> None:
    """Require a float64 PyTensor process, as the graph supplies the affine map.

    ``pm.Data`` casts registered arrays to the global ``floatX``; a float32
    graph would round the design and intercept to about seven digits.

    Raises:
        ValueError: If ``floatX`` is not float64.
    """
    if pytensor.config.floatX != "float64":
        raise ValueError(
            'method = "gaussian" requires PyTensor floatX=float64; set PYTENSOR_FLAGS=floatX=float64 '
            "before starting Python."
        )


def check_co2_nuts_mismatch(model: Mapping[str, Any]) -> None:
    """Refuse per-site and empirical-Bayes mismatch tables outside the Gaussian estimator.

    Args:
        model: CO2 model choices or resolved model arguments.

    Raises:
        ValueError: If ``fixed_model_mismatch`` is a per-site or empirical-Bayes table.
    """
    mismatch = model.get("fixed_model_mismatch")
    if isinstance(mismatch, Mapping | EmpiricalBayesMismatch):
        raise ValueError(
            "Per-site and empirical-Bayes likelihood.fixed_model_mismatch tables require "
            'sampling.method = "gaussian"; NUTS takes a scalar fixed_model_mismatch.'
        )


def check_co2_gaussian_likelihood(model: Mapping[str, Any]) -> None:
    """Refuse a likelihood other than additive-Gaussian with fixed error.

    Args:
        model: Mapping with the selected ``likelihood_builder``, as from
            :meth:`~openghg_inversions.rhime.co2.configuration.Co2ModelConfig.sampler_arguments`.

    Raises:
        ValueError: For a fixed-OU or other replacement likelihood.
    """
    builder = model.get("likelihood_builder")
    if builder is add_fixed_ou_gaussian_likelihood:
        raise ValueError(
            'method = "gaussian" does not yet support fixed-OU correlation (likelihood.kind = "fixed_ou"); '
            f'use likelihood.kind = "additive_sigma" with a fixed model mismatch {_ALTERNATIVE}.'
        )
    if builder is not None:
        raise ValueError(
            'method = "gaussian" supports only likelihood.kind = "additive_sigma" with fixed error; '
            f"the selected likelihood infers its error or is not Gaussian {_ALTERNATIVE}."
        )


def check_co2_gaussian_model(model: Mapping[str, Any]) -> None:
    """Refuse model choices that the fixed-noise Gaussian estimator cannot solve exactly.

    Configuration-only checks run before any preparation is opened; the
    biosphere family is checked once the calibrated prior is loaded, and the
    built graph is checked again for Normal priors and an affine mean.

    Args:
        model: CO2 model choices, as fields of
            :class:`~openghg_inversions.rhime.co2.configuration.Co2ModelConfig`
            or the resolved arguments from its ``model_arguments``.

    Raises:
        ValueError: For a fixed-OU or other non-additive likelihood, an
            inferred model-error scale, a non-Normal boundary or offset prior,
            or a calibrated biosphere prior that is not Gaussian on every state.
    """
    check_co2_gaussian_likelihood(model)
    if not model.get("no_model_error", False):
        raise ValueError(
            'method = "gaussian" needs a fixed likelihood error, but this model samples the model-error '
            "scale sigma. Set likelihood.no_model_error = true and a fixed model mismatch "
            "(likelihood.fixed_model_mismatch = ..., e.g. the median R from a previous NUTS run), "
            f"{_ALTERNATIVE}."
        )
    _require_normal(model.get("offset_prior"), "offset")
    boundary = model.get("boundary")
    if isinstance(boundary, MultiplicativeChoice):
        _require_normal(boundary.prior, "boundary")
    elif isinstance(boundary, AdditiveChoice | AnomalyScalingChoice):
        _require_normal(boundary.mean_prior, "boundary mean-shift")
    biosphere = model.get("biosphere_prior")
    if biosphere is not None:
        family = biosphere.data.attrs.get("family")
        if family != "gaussian" or not bool(biosphere.data.biosphere.all()):
            raise ValueError(
                f'method = "gaussian" cannot use a {family} biosphere prior: its protected states are '
                f"LogNormal. Remove model.biosphere to use the prepared Gaussian moments {_ALTERNATIVE}."
            )


def _gaussian_state_prior(
    inputs: xr.Dataset, biosphere: BiospherePrior | None,
) -> tuple[GaussianStatePrior, dict[str, Any]]:
    """Return the Gaussian retained-state moments and their PSD factor record."""
    mean = inputs["alpha_prior_mean"]
    covariance = inputs["alpha_prior_covariance"]
    description = "correlated arithmetic-moment gaussian"
    if biosphere is not None:
        baseline_mean, _ = xr.align(biosphere.data.baseline_mean, mean, join="exact")
        if not (np.array_equal(baseline_mean.values, mean.values)
                and np.array_equal(biosphere.data.baseline_covariance.values, covariance.values)):
            raise ValueError("Biosphere calibration must use this preparation's baseline arithmetic moments.")
        covariance = biosphere.data.covariance
        description = str(biosphere.data.attrs["family"])
    state_dim = str(mean.dims[0])
    factor = psd_factor(np.asarray(covariance.values, dtype=np.float64))
    labelled = xr.DataArray(
        factor.factor, dims=(state_dim, _MODE_DIM),
        coords={**mean.coords, _MODE_DIM: np.arange(factor.factor.shape[1])},
    )
    return GaussianStatePrior(mean=mean, factor=labelled, description=description), factor.diagnostics


def _co2_noise_factor(
    inputs: xr.Dataset, prepared: Co2PreparedInputs, fixed_model_mismatch: Any,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Return a factor ``F_W`` of the fixed CO2 likelihood covariance ``A + R``.

    ``R = diag(mf_error**2 + fixed_model_mismatch**2)``. The mismatch follows
    the CO2 builder: an explicit labelled value is installed in ``inputs``;
    otherwise an explicit scalar wins over a prepared ``fixed_model_mismatch``
    field. Low-rank and diagonal aggregation errors give the exact
    ``[low_rank_factor | diag(sqrt(R + diagonal_residual_variance))]``.

    A dense ``A`` gives the Cholesky factor of ``A + R`` formed directly, which
    is backward stable for that covariance; Paige's method then never inverts
    it. When that matrix is not numerically positive definite, for example
    with zero ``R``, the factor is ``[L_A | diag(sqrt(R))]`` with ``L_A`` from
    an eigendecomposition of ``A`` that clips only negative eigenvalues. No
    positive eigenvalue is dropped: a mode below any relative tolerance can
    still exceed ``R``.
    """
    error = np.asarray(inputs["mf_error"].transpose("nmeasure").values, dtype=np.float64)
    if fixed_model_mismatch is not None and not isinstance(fixed_model_mismatch, xr.DataArray):
        mismatch = np.full_like(error, float(fixed_model_mismatch))
    elif "fixed_model_mismatch" in inputs:
        mismatch = np.asarray(inputs["fixed_model_mismatch"].transpose("nmeasure").values, dtype=np.float64)
    else:
        mismatch = np.zeros_like(error)
    variance = error**2 + mismatch**2
    aggregation = resolve_aggregation_error(inputs, prepared.aggregation_error_mode)
    record: dict[str, Any] = {"aggregation_error_mode": aggregation.mode,
                              "independent_variance": "mf_error**2 + fixed_model_mismatch**2"}
    if aggregation.mode in ("low_rank", "diagonal"):
        assert aggregation.diagonal_variance is not None
        variance = variance + np.asarray(aggregation.diagonal_variance.values, dtype=np.float64)
    if aggregation.mode == "dense":
        assert aggregation.covariance is not None
        covariance = np.asarray(aggregation.covariance.values, dtype=np.float64)
        covariance = 0.5 * (covariance + covariance.T)
        total = covariance.copy()
        total[np.diag_indices_from(total)] += variance
        try:
            factor = np.asarray(np.linalg.cholesky(total), dtype=np.float64)
        except np.linalg.LinAlgError:
            pass
        else:
            record["noise_factor_method"] = "cholesky(A + R)"
            return factor, record
        del total
        clipped = psd_factor(covariance)
        record["noise_factor_method"] = "[eigh_clip(A) | diag(sqrt(R))]"
        record["aggregation_psd_factor"] = clipped.diagnostics
        return np.hstack([clipped.factor, np.diag(np.sqrt(variance))]), record
    columns = [np.diag(np.sqrt(variance))]
    if aggregation.mode == "low_rank":
        assert aggregation.factor is not None
        columns.insert(0, np.asarray(aggregation.factor.values, dtype=np.float64))
    record["noise_factor_method"] = {
        "low_rank": "[low_rank_factor | diag(sqrt(R + D))]", "diagonal": "diag(sqrt(R + D))", "none": "diag(sqrt(R))",
    }[aggregation.mode]
    return np.hstack(columns), record


def _check_noise_against_graph(built: RhimeModelBuildResult, noise_factor: NDArray[np.float64]) -> None:
    """Require the factor's marginal variances to match the graph's ``epsilon``."""
    epsilon = np.asarray(pytensor.function([], built.model["epsilon"])(), dtype=np.float64)
    marginal = np.einsum("ij,ij->i", noise_factor, noise_factor)
    if not np.allclose(marginal, epsilon**2, rtol=1e-8, atol=1e-12 * float(np.max(epsilon**2, initial=0.0))):
        raise RuntimeError("The Gaussian noise factor does not reproduce the graph's marginal error.")


def _state_labels(trace: xr.DataTree, name: str) -> list[str]:
    """Return readable labels for the flattened elements of one posterior variable."""
    variable = trace["posterior"].dataset[name].isel(chain=0, draw=0, drop=True)
    if variable.ndim == 0:
        return [name]
    index = variable.to_series().index
    return [f"{name}[{', '.join(map(str, label)) if isinstance(label, tuple) else label}]" for label in index]


def _gaussian_posterior_group(
    built: RhimeModelBuildResult, variables: Any, posterior: Any, trace: xr.DataTree,
    *, rng: np.random.Generator, attrs: Mapping[str, Any],
) -> xr.Dataset:
    """Return exact physical moments of the flux scaling and other sampled states."""
    model = built.model
    reported = ["flux_scaling"] + [name for name in variables.names if not name.startswith("flux_scaling")]
    maps = affine_maps(model, [model[name] for name in reported], variables, rng=rng)
    offset = np.concatenate([item.offset for item in maps])
    matrix = np.vstack([item.matrix for item in maps])
    posterior_factor = matrix @ posterior.cov_factor
    prior_factor = matrix * variables.prior_sd
    labels = [label for name in reported for label in _state_labels(trace, name)]
    names = [name for name, item in zip(reported, maps) for _ in range(item.offset.size)]
    elements = [index for item in maps for index in range(item.offset.size)]
    coords = {
        "state": np.arange(offset.size), "state_cov": np.arange(offset.size),
        "state_variable": ("state", names), "state_element": ("state", elements),
        "state_label": ("state", labels),
    }
    return xr.Dataset(
        {
            "posterior_mean": ("state", offset + matrix @ posterior.mean),
            "posterior_covariance": (("state", "state_cov"), posterior_factor @ posterior_factor.T),
            "prior_mean": ("state", offset + matrix @ variables.prior_mean),
            "prior_covariance": (("state", "state_cov"), prior_factor @ prior_factor.T),
        },
        coords=coords,
        attrs=dict(attrs),
    )


def _likelihood_terms(built: RhimeModelBuildResult) -> tuple[Any, NDArray[np.float64]]:
    """Return the likelihood mean expression and the observed values it is compared with."""
    y = built.model["y"]
    observations = np.asarray(built.model.rvs_to_values[y].data, dtype=np.float64)
    return y.owner.op.dist_params(y.owner)[0], observations


def _sigma_by_site(sites: NDArray[np.str_], sigma: NDArray[np.float64]) -> dict[str, float] | None:
    """Return one standard deviation per site when it is constant within every site."""
    result = {}
    for site in dict.fromkeys(sites.tolist()):
        values = np.unique(sigma[sites == site])
        if values.size != 1:
            return None
        result[site] = float(values[0])
    return result


def _empirical_bayes(
    choice: EmpiricalBayesMismatch, inputs: xr.Dataset, prepared: Co2PreparedInputs,
    build: Any, rng: np.random.Generator,
) -> tuple[xr.DataArray, dict[str, Any]]:
    """Estimate the per-site empirical-Bayes mismatch from the prior residual of the Gaussian graph.

    The graph is built with zero mismatch only to read its affine likelihood
    mean and state priors: the residual is taken at the prior mean of every
    state, and the prior-predictive state variance is the diagonal of
    ``J diag(prior_sd**2) J^T`` (retained flux, boundary and offset states).
    """
    zero = xr.zeros_like(inputs["mf"], dtype=np.float64).rename("fixed_model_mismatch")
    provisional = build(inputs.assign(fixed_model_mismatch=zero), zero)
    mean, observations = _likelihood_terms(provisional)
    variables = normal_variables(provisional.model)
    (mean_map,) = affine_maps(provisional.model, [mean], variables, rng=rng)
    residual = observations - (mean_map.offset + mean_map.matrix @ variables.prior_mean)
    prior_state_variance = np.einsum("ij,j->i", mean_map.matrix**2, variables.prior_sd**2)
    aggregation = resolve_aggregation_error(inputs, prepared.aggregation_error_mode)
    sites = observation_sites(inputs["mf"])
    sigma, record = empirical_bayes_mismatch(
        choice, sites=sites, residual=residual, prior_state_variance=prior_state_variance,
        aggregation_variance=np.asarray(aggregation.marginal_variance, dtype=np.float64),
        error_variance=np.asarray(inputs["mf_error"].transpose("nmeasure").values, dtype=np.float64) ** 2,
    )
    return zero.copy(data=sigma), record


def sample_co2_gaussian(
    prepared: Co2PreparedInputs, model: Mapping[str, Any], sampler: RhimeSampler,
) -> tuple[xr.DataTree, dict[str, Any]]:
    """Solve the exact fixed-noise Gaussian posterior and return a NUTS-layout trace.

    The fixed mismatch may be a scalar, a per-site mapping, a per-measurement
    ``DataArray`` or an :class:`~openghg_inversions.rhime.co2.model_mismatch.EmpiricalBayesMismatch`
    (see :mod:`~openghg_inversions.rhime.co2.model_mismatch`); an explicit
    value wins over a prepared ``fixed_model_mismatch`` field, which wins over
    none. Replacing the default LogNormal flux prior by its moment-matched
    Normal, and an empirical-Bayes mismatch, each log one warning.

    Args:
        prepared: CO2 preparation.
        model: Resolved model arguments; see :func:`check_co2_gaussian_model`.
        sampler: Supplies ``draws`` (independent posterior draws in one
            chain), the ``random_seed`` in ``sample_kwargs``, and the prior and
            posterior predictive choices. A fresh seed is drawn and recorded
            when none is configured.

    Returns:
        The annotated trace, including a ``gaussian_posterior`` group, and the
        recorded inference description.

    Raises:
        ValueError: If the model is not linear-Gaussian with fixed noise, or a
            per-site mismatch does not name exactly the observed sites.
    """
    require_gaussian_precision()
    check_co2_gaussian_model(model)
    mismatch = model.get("fixed_model_mismatch")
    direct = mismatch if mismatch is None or isinstance(mismatch, int | float | xr.DataArray) else None
    inputs, priors, boundary = materialize_co2_inputs(
        prepared_inputs=prepared,
        fixed_model_mismatch=direct,
        no_model_error=True,
        boundary=model.get("boundary"),
        offset_prior=model.get("offset_prior"),
        offset_args=model.get("offset_args"),
    )
    biosphere = model.get("biosphere_prior")
    gaussian_prior, prior_record = _gaussian_state_prior(inputs, biosphere)
    if biosphere is None:
        logger.warning(
            'method = "gaussian" replaces the default correlated LogNormal flux prior with the Normal prior of '
            "the same arithmetic moments (alpha_prior_mean, alpha_prior_covariance): %s.",
            STATE_PRIOR_SUBSTITUTION,
        )

    def build(model_inputs: xr.Dataset, fixed: Any) -> RhimeModelBuildResult:
        return build_co2_from_materialized(
            model_inputs, prepared_inputs=prepared, priors=priors, boundary=boundary,
            fixed_model_mismatch=fixed, no_model_error=True,
            offset_prior=model.get("offset_prior"), offset_args=model.get("offset_args"),
            gaussian_prior=gaussian_prior,
        )

    configured = (sampler.sample_kwargs or {}).get("random_seed")
    seed = int(configured) if configured is not None else secrets.randbits(63)
    check_rng, trace_rng, report_rng, mismatch_rng = np.random.default_rng(seed).spawn(4)
    mismatch_record: dict[str, Any]
    fixed: float | xr.DataArray | None
    if isinstance(mismatch, EmpiricalBayesMismatch):
        fixed, eb_record = _empirical_bayes(mismatch, inputs, prepared, build, mismatch_rng)
        inputs = inputs.assign(fixed_model_mismatch=fixed)
        mismatch_record = {"source": "empirical Bayes", "empirical_bayes": eb_record}
    elif isinstance(mismatch, Mapping):
        fixed = per_site_mismatch(mismatch, inputs["mf"])
        inputs = inputs.assign(fixed_model_mismatch=fixed)
        mismatch_record = {"source": "specified per site"}
    else:
        fixed = mismatch
        mismatch_record = {"source": (
            "specified scalar" if isinstance(mismatch, int | float)
            else "specified per measurement" if isinstance(mismatch, xr.DataArray)
            else "prepared fixed_model_mismatch" if "fixed_model_mismatch" in inputs else "none"
        )}
    built = build(inputs, fixed)
    noise_factor, noise_record = _co2_noise_factor(inputs, prepared, fixed)
    _check_noise_against_graph(built, noise_factor)
    if "fixed_model_mismatch" in built.model.named_vars:
        sigma = np.asarray(pytensor.function([], built.model["fixed_model_mismatch"])(), dtype=np.float64)
        try:
            by_site = _sigma_by_site(observation_sites(inputs["mf"]), sigma)
        except ValueError:
            by_site = None
        mismatch_record["sigma_by_site"] = by_site
    mismatch_record["units"] = prepared.inv_inputs["mf"].attrs.get("units")
    if isinstance(mismatch, EmpiricalBayesMismatch):
        logger.warning(
            "Empirical Bayes model mismatch (%s, floor %g): sigma by site %s %s. The Gaussian posterior treats "
            "these as known, so it does not include sigma uncertainty, and the observations are used twice "
            "(to set sigma and to fit).",
            mismatch.mode, mismatch.floor,
            json.dumps({site: round(value, 6) for site, value in mismatch_record["sigma_by_site"].items()}),
            mismatch_record["units"],
        )
    likelihood_mean, observations = _likelihood_terms(built)
    variables, posterior = solve_linear_gaussian_model(
        built.model, likelihood_mean=likelihood_mean, observations=observations,
        noise_factor=noise_factor, rng=check_rng,
    )
    del noise_factor
    trace = exact_gaussian_trace(
        built.model, variables, posterior, sampler=sampler, rng=trace_rng,
        variable_roles=built.output_contract.variable_roles,
    )
    trace = annotate_co2_trace(
        trace, built, concentration_units=prepared.inv_inputs["mf"].attrs.get("units"),
        biosphere_prior=biosphere,
    )
    diagnostics = posterior.diagnostics.to_dict()
    prior_draws = (
        0 if not sampler.sample_prior_predictive
        else sampler.draws if sampler.sample_prior_predictive is True else int(sampler.sample_prior_predictive)
    )
    mismatch_summary = {key: value for key, value in mismatch_record.items() if key != "empirical_bayes"}
    if isinstance(mismatch, EmpiricalBayesMismatch):
        mismatch_summary.update(empirical_bayes_mode=mismatch.mode, empirical_bayes_floor=mismatch.floor)
    actual: dict[str, Any] = {
        "algorithm": GAUSSIAN_ALGORITHM, "draws": sampler.draws, "chains": 1, "seed": seed,
        "prior_draws": prior_draws, "noise_rank": diagnostics["noise_rank"],
        "prior_rank": diagnostics["prior_rank"], "constraint_rank": diagnostics["constraint_rank"],
        "model_mismatch": mismatch_summary,
    }
    attrs: dict[str, Any] = {
        "algorithm": GAUSSIAN_ALGORITHM, "draws": sampler.draws, "seed": seed,
        "state_prior": "gaussian with the prepared arithmetic moments" if biosphere is None
        else "gaussian calibrated biosphere prior",
        "gls_diagnostics": json.dumps(diagnostics, sort_keys=True),
        "flux_prior_psd_factor": json.dumps(prior_record, sort_keys=True),
        "noise_factor": json.dumps(noise_record, sort_keys=True),
        "model_mismatch": json.dumps(mismatch_record, sort_keys=True),
        "sampled_variables": json.dumps(list(variables.names)),
    }
    if biosphere is None:
        actual["state_prior_substitution"] = attrs["state_prior_substitution"] = STATE_PRIOR_SUBSTITUTION
    trace[GAUSSIAN_POSTERIOR_GROUP] = _gaussian_posterior_group(
        built, variables, posterior, trace, rng=report_rng, attrs=attrs,
    )
    trace.attrs["rhime_actual_inference"] = json.dumps(actual)
    return trace, actual


__all__ = [
    "GAUSSIAN_ALGORITHM", "GAUSSIAN_POSTERIOR_GROUP", "STATE_PRIOR_SUBSTITUTION", "check_co2_nuts_mismatch", "check_co2_gaussian_likelihood", "check_co2_gaussian_model",
    "require_gaussian_precision", "sample_co2_gaussian",
]
