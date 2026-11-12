"""Diagnostics to run on a prior and its prior predictive before sampling.

These functions answer three questions about an inversion before any posterior
sampling is spent on it:

* Is the noise model wide enough for the observed model-data mismatch?
  :func:`noise_calibration` compares the prior predictive noise with the
  de-meaned mismatch between the observations and the prior predictive mean,
  per group of observations (for example sites) and for the whole network.
* Which retained states can the observations see at all?
  :func:`state_signal` measures each state's prior signal in observation
  units, ``sqrt(mean_o (H[o, j] s_j)^2)``, where ``s_j`` is the prior standard
  deviation of state ``j``.
* How much of a country's native prior variance do the retained states carry,
  and how much does the reduction push into the unresolved observation
  covariance ``A``? :func:`country_prior_budget` evaluates the exact coherent
  reduction with country totals as the "observations".

:func:`prior_predictive_component_sd` summarises how the prior predictive
variance divides between named additive components, such as baseline,
surface flux and noise.

The ``assess_*`` functions grade those results as ``"pass"``, ``"warn"`` or
``"fail"`` against :class:`PresampleThresholds`. The defaults for unitless
quantities come from the 2021 CO2/O2 verification-game inversions and are a
starting point, not a calibrated standard. Signal thresholds are in
observation units, so they have no defaults.

All functions are eager and read-only: inputs are borrowed and never mutated.
Plotting belongs in ``fluxie``; :func:`state_values_on_grid` maps per-state
results onto the native grid so standard flux-map plots can show them.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr

from openghg_inversions.basis.covariance_products import RetainedProjectionStrategy
from openghg_inversions.coherent_reduction import reduce_native_gaussian
from openghg_inversions.native_covariance import InvertibleNativeCovarianceAction

__all__ = [
    "PresampleThresholds",
    "assess_country_budget",
    "assess_noise_calibration",
    "assess_state_signal",
    "country_prior_budget",
    "noise_calibration",
    "prior_predictive_component_sd",
    "state_signal",
    "state_values_on_grid",
]

PASS, WARN, FAIL = "pass", "warn", "fail"


@dataclass(frozen=True)
class PresampleThresholds:
    """Grading thresholds for the pre-sampling diagnostics.

    Attributes:
        ratio_fail: Noise SD / de-meaned mismatch SD below this fails.
        ratio_warn: Ratio below this (and at least ``ratio_fail``) warns.
        ratio_wide: Ratio above this warns that the noise is much wider than
            the mismatch, which hides the flux signal.
        coverage_fail: Central noise-interval coverage of the de-meaned
            mismatch below this fails.
        coverage_warn: Coverage below this (and at least ``coverage_fail``)
            warns. Compare with the nominal ``coverage`` passed to
            :func:`noise_calibration`.
        network_group_fail_fraction: The network result is at least ``"warn"``
            when this fraction of groups fails, even if the pooled statistic
            passes.
        resolved_warn: Country resolved variance fraction below this warns.
        resolved_fail: Country resolved variance fraction below this fails.
        heavy_tail_scale_sd: Prior scale SD at or above which a weak state
            counts as heavy-tailed.
        weak_heavy_tail_warn: Number of weak heavy-tailed states that warns.
        weak_heavy_tail_fail: Number of weak heavy-tailed states that fails.
        negligible_share_warn: Share of states below the negligible-signal
            threshold that warns.
        negligible_share_fail: Share of states below the negligible-signal
            threshold that fails.
    """

    ratio_fail: float = 0.6
    ratio_warn: float = 0.8
    ratio_wide: float = 2.5
    coverage_fail: float = 0.8
    coverage_warn: float = 0.9
    network_group_fail_fraction: float = 0.25
    resolved_warn: float = 0.8
    resolved_fail: float = 0.5
    heavy_tail_scale_sd: float = 2.0
    weak_heavy_tail_warn: int = 1
    weak_heavy_tail_fail: int = 3
    negligible_share_warn: float = 0.25
    negligible_share_fail: float = 0.5


def _labels(groups: xr.DataArray, observation_dim: Hashable) -> xr.DataArray:
    """Return the group labels as a named 1-D array on the observation axis."""
    if groups.dims != (observation_dim,):
        raise ValueError(f"groups must be 1-D on {observation_dim!r}; got dims {groups.dims}.")
    return groups.rename("group").reset_coords(drop=True)


def _sample_dims(draws: xr.DataArray, sample_dims: Sequence[Hashable]) -> list[Hashable]:
    """Return the sample dimensions present in ``draws``, requiring at least one."""
    present = [dim for dim in sample_dims if dim in draws.dims]
    if not present:
        raise ValueError(f"Draws have none of the sample dimensions {tuple(sample_dims)}.")
    return present


def _by_group(
    values: xr.DataArray,
    labels: xr.DataArray,
    reduce: str,
    network_label: str,
    **kwargs: object,
) -> xr.DataArray:
    """Reduce over each group and over all rows, with the network appended last."""
    grouped = getattr(values.groupby(labels), reduce)(**kwargs)
    network = getattr(values, reduce)(**kwargs).expand_dims(group=[network_label])
    return xr.concat([grouped, network], dim="group")


def noise_calibration(
    observed: xr.DataArray,
    predicted_mean: xr.DataArray,
    noise_draws: xr.DataArray,
    *,
    observation_dim: Hashable,
    groups: xr.DataArray,
    sample_dims: Sequence[Hashable] = ("chain", "draw"),
    coverage: float = 0.95,
    network_label: str = "network",
) -> xr.Dataset:
    """Compare the prior predictive noise with the observed model-data mismatch.

    The mismatch is ``observed - predicted_mean``, de-meaned within each group
    so that a constant offset (which the baseline terms are meant to absorb)
    does not count against the noise model. The noise is the part of the prior
    predictive that is not the modelled mean, ``y - mu`` for each draw, which
    includes every error term the likelihood carries (for example unresolved
    aggregation error, model error and measurement error).

    Call this once per species when observation rows of several species share
    an axis.

    Args:
        observed: Observations on ``observation_dim``.
        predicted_mean: Prior predictive mean of the modelled (noise-free)
            value on ``observation_dim``.
        noise_draws: Prior predictive noise draws, ``y - mu``, with
            ``observation_dim`` and one or more ``sample_dims``.
        observation_dim: Observation dimension shared by the inputs.
        groups: Group label for each observation, for example the site.
        sample_dims: Draw dimensions to reduce over; absent ones are ignored.
        coverage: Nominal probability of the central noise interval.
        network_label: Label of the all-observations entry on ``group``.

    Returns:
        Dataset on ``group`` (each group, then the network) with
        ``n_obs``; ``mismatch_mean`` (mean of ``observed - predicted_mean``);
        ``mismatch_sd`` (SD of the de-meaned mismatch); ``noise_sd`` (RMS
        over rows of the per-row noise SD); ``noise_over_mismatch``; and
        ``coverage``, the fraction of de-meaned mismatches inside the per-row
        central noise interval. Mismatch and noise values are in the units of
        ``observed``.

    Raises:
        ValueError: If ``coverage`` is not in (0, 1), the draws have no
            sample dimension, or ``groups`` is not 1-D on ``observation_dim``.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must be in (0, 1); got {coverage!r}.")
    labels = _labels(groups, observation_dim)
    dims = _sample_dims(noise_draws, sample_dims)
    observed, predicted_mean, noise_draws, labels = xr.align(
        observed, predicted_mean, noise_draws, labels, join="exact"
    )
    mismatch = (observed - predicted_mean).reset_coords(drop=True)
    demeaned = (mismatch.groupby(labels) - mismatch.groupby(labels).mean()).drop_vars(
        "group", errors="ignore"
    )
    noise_var = noise_draws.var(dims, ddof=1).reset_coords(drop=True)
    tail = (1.0 - coverage) / 2.0
    bounds = noise_draws.quantile([tail, 1.0 - tail], dim=dims).reset_coords(drop=True)
    inside = ((demeaned >= bounds.isel(quantile=0)) & (demeaned <= bounds.isel(quantile=1))).astype(float)

    result = xr.Dataset(
        {
            "n_obs": _by_group(xr.ones_like(mismatch), labels, "sum", network_label),
            "mismatch_mean": _by_group(mismatch, labels, "mean", network_label),
            "mismatch_sd": _by_group(demeaned, labels, "std", network_label, ddof=1),
            "noise_sd": np.sqrt(_by_group(noise_var, labels, "mean", network_label)),
            "coverage": _by_group(inside, labels, "mean", network_label),
        }
    )
    result["noise_over_mismatch"] = result.noise_sd / result.mismatch_sd
    result["n_obs"] = result.n_obs.astype(int)
    result.attrs.update(nominal_coverage=coverage, network_label=network_label)
    return result


def prior_predictive_component_sd(
    components: Mapping[str, xr.DataArray],
    *,
    observation_dim: Hashable,
    groups: xr.DataArray,
    sample_dims: Sequence[Hashable] = ("chain", "draw"),
    network_label: str = "network",
) -> xr.Dataset:
    """Summarise the prior predictive spread of named additive components.

    For each component, the per-row SD over draws is pooled as an RMS within
    each group and over the network. ``variance_share`` divides each pooled
    variance by their sum, so it is a decomposition only when the supplied
    components are additive and close to independent (for example baseline,
    surface flux and noise).

    Args:
        components: Draws of each component, keyed by name, each with
            ``observation_dim`` and ``sample_dims``.
        observation_dim: Observation dimension shared by the components.
        groups: Group label for each observation, for example the site.
        sample_dims: Draw dimensions to reduce over; absent ones are ignored.
        network_label: Label of the all-observations entry on ``group``.

    Returns:
        Dataset on ``group`` and ``component`` with ``sd`` (in the units of the
        components) and ``variance_share``.

    Raises:
        ValueError: If no components are given.
    """
    if not components:
        raise ValueError("Pass at least one component.")
    labels = _labels(groups, observation_dim)
    sds = []
    for name, draws in components.items():
        row_var = draws.var(_sample_dims(draws, sample_dims), ddof=1).reset_coords(drop=True)
        row_var, aligned = xr.align(row_var, labels, join="exact")
        sds.append(np.sqrt(_by_group(row_var, aligned, "mean", network_label)).expand_dims(component=[name]))
    sd = xr.concat(sds, dim="component")
    return xr.Dataset({"sd": sd, "variance_share": sd**2 / (sd**2).sum("component")})


def state_signal(
    sensitivity: xr.DataArray,
    state_sd: xr.DataArray,
    *,
    observation_dim: Hashable,
    state_dim: Hashable,
    observation_groups: xr.DataArray | None = None,
) -> xr.Dataset:
    """Measure each retained state's prior signal in observation units.

    The signal of state ``j`` is ``sqrt(mean_o (H[o, j] s_j)^2)``: the RMS over
    observations of the change a one-SD prior perturbation of that state makes
    to the modelled value. A state whose signal is far below the observation
    noise cannot be constrained and keeps its prior.

    With ``s_j`` the prior SD and ``sigma_o`` a common noise SD, the
    information ``sum_o (H[o, j] s_j)^2 / sigma_o^2`` used to select states
    for marginalisation equals ``n_obs * signal_rms^2 / sigma_o^2``.

    Args:
        sensitivity: Effective sensitivity ``H`` of the modelled value to each
            retained state, on ``observation_dim`` and ``state_dim``, in
            observation units per unit of the state.
        state_sd: Prior SD of each state on ``state_dim``. For scaling-factor
            states this is the SD of the scaling factor (for a lognormal
            state, of the factor itself, not its logarithm).
        observation_dim: Observation dimension of ``sensitivity``.
        state_dim: State dimension shared by both inputs.
        observation_groups: Optional label per observation (for example the
            species); the signal is then measured within each group.

    Returns:
        Dataset on ``state_dim`` (and ``group`` when ``observation_groups`` is
        given) with ``signal_rms`` and ``signal_max`` (largest absolute
        response), in the units of ``sensitivity``.
    """
    sensitivity, state_sd = xr.align(sensitivity, state_sd, join="exact")
    response = abs(sensitivity * state_sd).reset_coords(drop=True)
    if observation_groups is None:
        return xr.Dataset(
            {
                "signal_rms": np.sqrt((response**2).mean(observation_dim)),
                "signal_max": response.max(observation_dim),
            }
        )
    labels = _labels(observation_groups, observation_dim)
    response, labels = xr.align(response, labels, join="exact")
    grouped = response.groupby(labels)
    return xr.Dataset(
        {
            "signal_rms": np.sqrt((response**2).groupby(labels).mean(observation_dim)),
            "signal_max": grouped.max(observation_dim),
        }
    )


def country_prior_budget(
    *,
    covariance: InvertibleNativeCovarianceAction,
    basis_prolongation: xr.DataArray,
    state_dim: str,
    native_flux: xr.DataArray,
    country_weights: xr.DataArray,
    country_dim: str = "country",
    native_mean: xr.DataArray | None = None,
    strategy: RetainedProjectionStrategy | None = None,
    observation_batch_size: int = 64,
) -> xr.Dataset:
    """Split each country's native prior variance into retained and unresolved parts.

    Country totals are linear functionals ``f_c = sum_g w[c, g] F[g] x[g]`` of
    the native scaling field ``x ~ N(m, B)``, with native flux ``F`` and
    country weights ``w`` (for example country fraction times cell area). The
    exact coherent reduction with these functionals in place of observations
    gives

    * ``variance_resolved = D C_alpha D^T``, what the retained states can
      express and what a posterior built on them reports, and
    * ``variance_unresolved = A``, the native remainder that the reduction
      moves into the unresolved covariance and that never reaches
      retained-state country uncertainties.

    Users choose the countries through ``country_weights``. From a
    :class:`~openghg_inversions.postprocessing.countries.Countries` object
    built with ``country_selections`` and ``country_regions`` on the native
    grid, use ``countries.matrix * countries.area_grid``.

    Args:
        covariance: Native covariance action ``B`` of the scaling field.
        basis_prolongation: Basis prolongation on the native dimensions and
            ``state_dim``, as for
            :func:`~openghg_inversions.coherent_reduction.reduce_native_gaussian`.
        state_dim: Retained-state dimension.
        native_flux: Native prior flux on the covariance's native dimensions.
        country_weights: Weights on ``country_dim`` and the native dimensions.
            The units of ``native_flux * country_weights`` summed over the grid
            are the units of the returned totals.
        country_dim: Country dimension of ``country_weights``.
        native_mean: Native scaling mean ``m``; defaults to one everywhere.
        strategy: Retained restriction strategy, which must match the one used
            to build the inversion's retained states.
        observation_batch_size: Covariance right-hand-side batch size.

    Returns:
        Dataset on ``country_dim`` with ``prior_total`` (``F m`` summed with
        the weights), ``variance_native``, ``variance_resolved``,
        ``variance_unresolved`` and ``resolved_fraction``.
    """
    native_dims = tuple(covariance.native_dims)
    mean = xr.ones_like(native_flux, dtype=float) if native_mean is None else native_mean
    functional = (native_flux * country_weights).transpose(country_dim, *native_dims)
    reduction = reduce_native_gaussian(
        covariance=covariance,
        basis_prolongation=basis_prolongation,
        state_dim=state_dim,
        native_mean=mean,
        native_sensitivity=functional,
        observation_dim=country_dim,
        observation_batch_size=observation_batch_size,
        strategy=strategy,
    )
    operator = reduction.effective_observation_operator.transpose(country_dim, state_dim).values
    retained = reduction.retained_covariance.values
    resolved = np.einsum("cs,st,ct->c", operator, retained, operator)
    unresolved = np.diag(reduction.unresolved_observation_covariance.values).copy()
    prior = reduction.native_observation_mean
    result = xr.Dataset(
        {
            "prior_total": prior,
            "variance_resolved": prior.copy(data=resolved),
            "variance_unresolved": prior.copy(data=unresolved),
        }
    )
    result["variance_native"] = result.variance_resolved + result.variance_unresolved
    result["resolved_fraction"] = result.variance_resolved / result.variance_native
    result.attrs["projection_strategy"] = reduction.projection_strategy
    return result


def state_values_on_grid(
    values: xr.DataArray | xr.Dataset,
    basis_prolongation: xr.DataArray,
    state_dim: Hashable,
) -> xr.DataArray | xr.Dataset:
    """Map per-state values onto the native grid for plotting.

    Each native cell takes the value of the state its basis prolongation
    assigns it to, so a bucket basis gives piecewise-constant maps. Cells that
    no state covers are NaN.

    Args:
        values: Values on ``state_dim``, such as the output of
            :func:`state_signal`.
        basis_prolongation: Prolongation on the native dimensions and
            ``state_dim``.
        state_dim: State dimension shared by both inputs.

    Returns:
        ``values`` on the native dimensions in place of ``state_dim``.
    """
    covered = basis_prolongation.sum(state_dim)
    mapped = xr.dot(basis_prolongation, values, dim=state_dim) / covered
    return mapped.where(covered > 0)


def _grade(
    value: xr.DataArray,
    *,
    fail: float,
    warn: float,
    higher_is_better: bool,
) -> xr.DataArray:
    """Return pass, warn or fail for each element of ``value``."""
    bad, poor = (value < fail, value < warn) if higher_is_better else (value > fail, value > warn)
    return xr.where(bad, FAIL, xr.where(poor, WARN, PASS))


def _worst(*statuses: xr.DataArray) -> xr.DataArray:
    """Return the most severe status at each position."""
    rank = {PASS: 0, WARN: 1, FAIL: 2}
    order = np.array([PASS, WARN, FAIL])
    worst = np.maximum.reduce([np.vectorize(rank.__getitem__)(np.asarray(s)) for s in statuses])
    return statuses[0].copy(data=order[worst])


def assess_noise_calibration(
    calibration: xr.Dataset,
    thresholds: PresampleThresholds | None = None,
) -> xr.Dataset:
    """Grade the output of :func:`noise_calibration`.

    A group fails when the noise is too narrow (``noise_over_mismatch`` below
    ``ratio_fail`` or coverage below ``coverage_fail``), warns when it is
    narrow, has low coverage, or is wider than ``ratio_wide`` times the
    mismatch, and passes otherwise. The network entry is also at least
    ``"warn"`` when at least ``network_group_fail_fraction`` of the groups
    fail.

    Args:
        calibration: Output of :func:`noise_calibration`.
        thresholds: Grading thresholds; defaults to :class:`PresampleThresholds`.

    Returns:
        ``calibration`` with a ``status`` variable on ``group``.
    """
    t = thresholds or PresampleThresholds()
    ratio = calibration.noise_over_mismatch
    status = _worst(
        _grade(ratio, fail=t.ratio_fail, warn=t.ratio_warn, higher_is_better=True),
        _grade(calibration.coverage, fail=t.coverage_fail, warn=t.coverage_warn, higher_is_better=True),
        xr.where(ratio > t.ratio_wide, WARN, PASS),
    )
    network = calibration.attrs.get("network_label", "network")
    groups = status.drop_sel(group=network)
    if groups.size and float((groups == FAIL).mean()) >= t.network_group_fail_fraction:
        floor = xr.where(status.group == network, WARN, PASS)
        status = _worst(status, floor)
    return calibration.assign(status=status)


def assess_country_budget(
    budget: xr.Dataset,
    thresholds: PresampleThresholds | None = None,
) -> xr.Dataset:
    """Grade the resolved fraction from :func:`country_prior_budget`.

    Args:
        budget: Output of :func:`country_prior_budget`.
        thresholds: Grading thresholds; defaults to :class:`PresampleThresholds`.

    Returns:
        ``budget`` with a ``status`` variable on the country dimension.
    """
    t = thresholds or PresampleThresholds()
    status = _grade(
        budget.resolved_fraction, fail=t.resolved_fail, warn=t.resolved_warn, higher_is_better=True
    )
    return budget.assign(status=status)


def assess_state_signal(
    signal_rms: xr.DataArray,
    scale_sd: xr.DataArray,
    *,
    negligible: float,
    dead: float,
    state_dim: Hashable,
    thresholds: PresampleThresholds | None = None,
) -> xr.Dataset:
    """Classify states by prior signal and grade the weak, heavy-tailed ones.

    States with ``signal_rms < dead`` are effectively unobserved; states with
    ``dead <= signal_rms < negligible`` are weak. A weak state with a large
    prior scale SD is the risky case: the data cannot hold it, so it can wander
    over a heavy prior tail during sampling and slow mixing.

    The signal thresholds are in the units of ``signal_rms``. A useful
    ``negligible`` is about a tenth of the smallest observation noise SD.
    Pass a subset of states (for example only land sources) to grade the
    negligible share for that subset.

    Args:
        signal_rms: State signal from :func:`state_signal`, on ``state_dim``.
        scale_sd: Prior SD of each state on ``state_dim``.
        negligible: Signal below which a state is negligible.
        dead: Signal below which a state is effectively unobserved.
        state_dim: State dimension.
        thresholds: Grading thresholds; defaults to :class:`PresampleThresholds`.

    Returns:
        Dataset with per-state ``signal_class`` (``"dead"``, ``"weak"`` or
        ``"active"``) and ``weak_heavy_tail``, plus scalar
        ``n_weak_heavy_tail``, ``negligible_share`` and their statuses
        ``weak_heavy_tail_status`` and ``negligible_share_status``.

    Raises:
        ValueError: If ``dead`` is not below ``negligible``.
    """
    if not dead < negligible:
        raise ValueError(f"dead ({dead}) must be below negligible ({negligible}).")
    t = thresholds or PresampleThresholds()
    signal_rms, scale_sd = xr.align(signal_rms, scale_sd, join="exact")
    signal_class = xr.where(signal_rms < dead, "dead", xr.where(signal_rms < negligible, "weak", "active"))
    weak_heavy_tail = (signal_class == "weak") & (scale_sd >= t.heavy_tail_scale_sd)
    count = weak_heavy_tail.sum(state_dim)
    share = (signal_rms < negligible).mean(state_dim)
    return xr.Dataset(
        {
            "signal_class": signal_class,
            "weak_heavy_tail": weak_heavy_tail,
            "n_weak_heavy_tail": count,
            "negligible_share": share,
            "weak_heavy_tail_status": _grade(
                count,
                fail=t.weak_heavy_tail_fail - 0.5,
                warn=t.weak_heavy_tail_warn - 0.5,
                higher_is_better=False,
            ),
            "negligible_share_status": _grade(
                share, fail=t.negligible_share_fail, warn=t.negligible_share_warn, higher_is_better=False
            ),
        }
    )
