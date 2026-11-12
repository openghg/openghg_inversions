"""Score posterior country and sector-total uncertainties against a known truth.

:func:`functional_coverage` is a per-run building block for synthetic (OSSE)
inversions, where the true value of each reported functional is known. It checks
that the reported posterior spread is calibrated: that the z-scores have unit
RMS and that the central intervals contain the truth about as often as their
nominal probability says.

Grouped functionals, such as a country group or a sum over sectors, are formed
by summing **joint draws** before any statistic is calculated. Means and SDs
of the members must never be combined, because that ignores their posterior
correlation.

Three things are flagged because they are known to make a score misleading:

* ``sd_below_floor``: the posterior SD is below an absolute floor or below a
  fraction of the prior SD. A tiny country with a near-zero SD gives an
  enormous z-score for a negligible absolute error.
* ``retained_only``: the draws only carry the uncertainty of the retained
  states, so the SD leaves out the unresolved part. The scope is read from the
  ``uncertainty_scope`` attribute of ``draws`` (or an ``uncertainty_scope``
  coordinate on the functional dimension).
* A non-finite z-score, for example from a zero SD or a missing truth.

The function is eager and read-only: inputs are borrowed and never modified,
and Dask-backed draws are loaded.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

import numpy as np
import xarray as xr
from scipy.stats import norm

__all__ = ["RETAINED_ONLY_SCOPES", "functional_coverage", "wilson_interval"]

#: ``uncertainty_scope`` values that mark draws as missing the unresolved variance.
RETAINED_ONLY_SCOPES = ("retained_only", "retained_state_conditional")


def wilson_interval(successes: xr.DataArray, trials: xr.DataArray, confidence: float = 0.95) -> xr.Dataset:
    """Wilson score interval for a binomial proportion.

    Args:
        successes: Number of successes.
        trials: Number of trials, broadcast against ``successes``.
        confidence: Probability content of the interval, in (0, 1).

    Returns:
        Dataset with ``proportion``, ``lower`` and ``upper``, all NaN where
        ``trials`` is zero.

    Raises:
        ValueError: If ``confidence`` is not in (0, 1).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence!r}.")
    z = norm.ppf(0.5 + confidence / 2.0)
    n = trials.where(trials > 0)
    p = successes / n
    denominator = 1.0 + z**2 / n
    centre = (p + z**2 / (2.0 * n)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2)) / denominator
    return xr.Dataset({"proportion": p, "lower": centre - half, "upper": centre + half})


def _sample_dims(draws: xr.DataArray, sample_dims: Sequence[Hashable]) -> list[Hashable]:
    """Return the sample dimensions present in ``draws``, requiring at least one."""
    present = [dim for dim in sample_dims if dim in draws.dims]
    if not present:
        raise ValueError(f"Draws have none of the sample dimensions {tuple(sample_dims)}.")
    return present


def _sum_per_draw(draws: xr.DataArray, dims: Sequence[Hashable]) -> xr.DataArray:
    """Sum over ``dims`` that are present, propagating NaN rather than skipping it."""
    present = [dim for dim in dims if dim in draws.dims]
    return draws.sum(present, skipna=False) if present else draws


def _group_sums(draws: xr.DataArray, dim: Hashable, groups: Mapping[str, Sequence[Hashable]]) -> xr.DataArray:
    """Sum the members of each group per draw, giving ``dim`` the group names."""
    available = set(draws[dim].values.tolist())
    pieces = []
    for name, members in groups.items():
        members = list(members)
        if not members:
            raise ValueError(f"Group {name!r} has no members.")
        if len(set(members)) != len(members):
            raise ValueError(f"Group {name!r} lists a member more than once.")
        missing = [member for member in members if member not in available]
        if missing:
            raise ValueError(f"Group {name!r} has members not in {dim!r}: {missing}.")
        total = draws.sel({dim: members}).sum(dim, skipna=False)
        pieces.append(total.expand_dims({dim: [name]}))
    if not pieces:
        raise ValueError("groups must contain at least one group.")
    return xr.concat(pieces, dim=dim)


def _retained_only(
    draws: xr.DataArray,
    dim: Hashable,
    groups: Mapping[str, Sequence[Hashable]] | None,
    scopes: Sequence[str],
) -> xr.DataArray:
    """Flag functionals whose uncertainty scope is retained-state only.

    A group is flagged if any member is, because the group SD then also leaves
    out unresolved variance.
    """
    if "uncertainty_scope" in draws.coords and dim in draws["uncertainty_scope"].dims:
        flags = xr.DataArray(
            np.isin(draws["uncertainty_scope"].values, list(scopes)),
            dims=(dim,),
            coords={dim: draws[dim].values},
        )
    else:
        flags = xr.DataArray(
            np.full(draws.sizes[dim], draws.attrs.get("uncertainty_scope") in scopes),
            dims=(dim,),
            coords={dim: draws[dim].values},
        )
    if groups is None:
        return flags
    return xr.DataArray(
        [bool(flags.sel({dim: list(members)}).any()) for members in groups.values()],
        dims=(dim,),
        coords={dim: list(groups)},
    )


def _prepare(
    draws: xr.DataArray,
    dim: Hashable,
    sum_dims: Sequence[Hashable],
    groups: Mapping[str, Sequence[Hashable]] | None,
) -> xr.DataArray:
    """Sum over ``sum_dims`` and then over each group, per draw."""
    draws = _sum_per_draw(draws, sum_dims).compute()
    return draws if groups is None else _group_sums(draws, dim, groups)


def _central_interval_membership(
    draws: xr.DataArray, truth: xr.DataArray, intervals: Sequence[float], sample_dims: list[Hashable]
) -> xr.DataArray:
    """Whether the truth lies in each central interval of the draws."""
    members = []
    for probability in intervals:
        tail = (1.0 - probability) / 2.0
        bounds = draws.quantile([tail, 1.0 - tail], dim=sample_dims).reset_coords(drop=True)
        inside = (truth >= bounds.isel(quantile=0, drop=True)) & (truth <= bounds.isel(quantile=1, drop=True))
        members.append(inside.expand_dims(interval=[probability]))
    return xr.concat(members, dim="interval")


def functional_coverage(
    draws: xr.DataArray,
    truth: xr.DataArray,
    groups: Mapping[str, Sequence[Hashable]] | None = None,
    intervals: Sequence[float] = (0.68, 0.95),
    min_sd: float | None = None,
    min_sd_fraction_of_prior: float | None = None,
    prior_draws: xr.DataArray | None = None,
    *,
    functional_dim: Hashable = "country",
    sum_dims: Sequence[Hashable] = (),
    sample_dims: Sequence[Hashable] = ("chain", "draw"),
    confidence: float = 0.95,
    retained_only_scopes: Sequence[str] = RETAINED_ONLY_SCOPES,
) -> xr.Dataset:
    """Score the calibration of posterior functionals against a known truth.

    Order of operations: sum over ``sum_dims`` per draw, sum each group's
    members per draw, then calculate every statistic from the resulting draws.
    The truth is summed in the same way.

    Args:
        draws: Posterior draws of the functionals, with the ``sample_dims``
            present and ``functional_dim``; for example country totals, with a
            ``sector`` dimension to sum over.
        truth: True values with the dimensions of ``draws`` apart from the
            sample dimensions, in the same units. Dimensions in ``sum_dims``
            are summed. Must align exactly with ``draws``.
        groups: Optional mapping from a group name to the members of
            ``functional_dim`` it sums, such as ``{"EU": ["FRA", "DEU"]}``.
            Only the named groups are scored, and groups may overlap. Without
            it every member of ``functional_dim`` is scored on its own.
        intervals: Probabilities of the central posterior intervals to check.
        min_sd: Flag functionals whose posterior SD is below this absolute
            value, in the units of ``draws``.
        min_sd_fraction_of_prior: Flag functionals whose posterior SD is below
            this fraction of the prior SD. Requires ``prior_draws``.
        prior_draws: Prior draws of the same functionals as ``draws``; they are
            summed and grouped in the same way.
        functional_dim: Dimension of the functionals, or of the members that
            ``groups`` combine.
        sum_dims: Dimensions, such as ``sector``, to sum per draw before
            grouping. Those absent from an input are ignored.
        sample_dims: Draw dimensions; those absent from ``draws`` are ignored.
        confidence: Probability content of the Wilson intervals on coverage.
        retained_only_scopes: ``uncertainty_scope`` values that count as
            ``retained_only``. The scope is read from ``draws.attrs`` (all
            functionals) or from an ``uncertainty_scope`` coordinate along
            ``functional_dim`` (per functional).

    Returns:
        Dataset with, per functional (on ``functional_dim`` and any other
        dimension left in ``draws``): ``mean``, ``sd`` (ddof 1), ``truth``,
        ``z`` = ``(mean - truth) / sd``, ``pit`` (fraction of draws below the
        truth, counting ties as half), ``in_interval`` (on ``interval``),
        ``sd_below_floor``, ``retained_only`` and ``excluded``, true where the
        functional is flagged or has a non-finite ``z``. ``prior_sd`` is
        included when ``prior_draws`` is given.

        Aggregates over all functionals, for ``subset`` ``"all"`` and
        ``"unflagged"`` (excluding ``excluded``): ``n``, ``coverage``,
        ``coverage_lower`` and ``coverage_upper`` (Wilson, per ``interval``),
        and ``rms_z`` over the finite z-scores. Functionals with a non-finite
        truth are left out of the coverage counts. The nominal and Wilson
        confidence levels are in ``attrs``.

    Raises:
        ValueError: If the draws have no sample dimension or fewer than two
            draws, ``intervals`` or ``confidence`` are outside (0, 1),
            ``min_sd_fraction_of_prior`` is given without ``prior_draws``, a
            group is empty or has unknown members, or the truth does not
            align with the draws.
    """
    if any(not 0.0 < probability < 1.0 for probability in intervals):
        raise ValueError(f"intervals must be in (0, 1); got {tuple(intervals)!r}.")
    if min_sd_fraction_of_prior is not None and prior_draws is None:
        raise ValueError("min_sd_fraction_of_prior requires prior_draws.")
    if functional_dim not in draws.dims:
        raise ValueError(f"draws have no {functional_dim!r} dimension; got {draws.dims}.")
    sdims = _sample_dims(draws, sample_dims)
    if np.prod([draws.sizes[dim] for dim in sdims]) < 2:
        raise ValueError("At least two draws are needed to estimate a posterior SD.")

    grouped = _prepare(draws, functional_dim, sum_dims, groups)
    retained = _retained_only(draws, functional_dim, groups, retained_only_scopes)
    truth = _prepare(truth, functional_dim, sum_dims, groups)
    grouped, truth = xr.align(grouped, truth.reset_coords(drop=True), join="exact")
    grouped = grouped.reset_coords(drop=True)

    mean = grouped.mean(sdims)
    sd = grouped.std(sdims, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (mean - truth) / sd
    z = z.where(sd > 0)
    n_draws = float(np.prod([grouped.sizes[dim] for dim in sdims]))
    pit = ((grouped < truth).sum(sdims) + 0.5 * (grouped == truth).sum(sdims)) / n_draws
    pit = pit.where(truth.notnull())
    in_interval = _central_interval_membership(grouped, truth, intervals, sdims)

    below_floor = xr.zeros_like(sd, dtype=bool)
    result = {"mean": mean, "sd": sd, "truth": truth, "z": z, "pit": pit}
    if prior_draws is not None:
        prior_sd = _prepare(prior_draws, functional_dim, sum_dims, groups).reset_coords(drop=True)
        prior_sd = prior_sd.std(_sample_dims(prior_draws, sample_dims), ddof=1)
        prior_sd, sd = xr.align(prior_sd, sd, join="exact")
        result["prior_sd"] = prior_sd
        if min_sd_fraction_of_prior is not None:
            below_floor = below_floor | (sd < min_sd_fraction_of_prior * prior_sd)
    if min_sd is not None:
        below_floor = below_floor | (sd < min_sd)
    retained = retained.broadcast_like(sd)

    excluded = below_floor | retained | ~np.isfinite(z)
    result.update(
        in_interval=in_interval,
        sd_below_floor=below_floor,
        retained_only=retained,
        excluded=excluded,
    )
    dataset = xr.Dataset(result)
    dataset = xr.merge([dataset, _aggregate(dataset, confidence)])
    dataset.attrs.update(
        intervals=[float(probability) for probability in intervals],
        wilson_confidence=confidence,
        n_draws=int(n_draws),
    )
    return dataset


def _aggregate(scores: xr.Dataset, confidence: float) -> xr.Dataset:
    """Coverage with Wilson intervals and RMS z over all and unflagged functionals."""
    dims = list(scores["sd"].dims)
    pieces = []
    for subset, keep in (("all", xr.ones_like(scores["excluded"])), ("unflagged", ~scores["excluded"])):
        has_truth = scores["truth"].notnull() & keep
        trials = has_truth.sum(dims)
        hits = (scores["in_interval"] & has_truth).sum(dims)
        wilson = wilson_interval(hits, trials, confidence)
        kept_z = scores["z"].where(keep)
        piece = xr.Dataset(
            {
                "n": keep.sum(dims),
                "coverage": wilson["proportion"],
                "coverage_lower": wilson["lower"],
                "coverage_upper": wilson["upper"],
                "rms_z": np.sqrt((kept_z**2).mean(dims)),
            }
        )
        pieces.append(piece.expand_dims(subset=[subset]))
    return xr.concat(pieces, dim="subset")
