"""RHIME sampling helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
import pymc as pm
import xarray as xr

from openghg_inversions._timing import log_timing, timer_seconds, timer_start
from openghg_inversions._sampling import _reset_retained_draws as _shared_reset_retained_draws
from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords

NutsSampler = Literal["pymc", "nutpie", "numpyro", "blackjax"]


def _finite_values(data: xr.DataArray) -> np.ndarray:
    """Return finite numeric values from a sample-stats variable."""
    values = np.asarray(data.values)
    if values.dtype == bool:
        return values.astype(float).reshape(-1)
    values = values.astype(float, copy=False).reshape(-1)
    return values[np.isfinite(values)]


def _sample_stat_mean(sample_stats: xr.Dataset, name: str) -> float | None:
    """Return a finite mean for a sample-stats variable, if available."""
    if name not in sample_stats:
        return None
    values = _finite_values(sample_stats[name])
    if values.size == 0:
        return None
    return float(values.mean())


def _sample_stat_max(sample_stats: xr.Dataset, name: str) -> float | None:
    """Return a finite maximum for a sample-stats variable, if available."""
    if name not in sample_stats:
        return None
    values = _finite_values(sample_stats[name])
    if values.size == 0:
        return None
    return float(values.max())


def _sample_stat_sum(sample_stats: xr.Dataset, name: str) -> int | None:
    """Return an integer sum for a sample-stats variable, if available."""
    if name not in sample_stats:
        return None
    values = _finite_values(sample_stats[name])
    if values.size == 0:
        return None
    return int(values.sum())


def _log_sample_stats(trace: xr.DataTree, *, label: str) -> None:
    """Log compact sampler diagnostics from an inference DataTree."""
    if "sample_stats" not in trace.children:
        return
    sample_stats = trace["sample_stats"].to_dataset()

    fields: dict[str, float | int | None] = {
        "n_steps_mean": _sample_stat_mean(sample_stats, "n_steps"),
        "n_steps_max": _sample_stat_max(sample_stats, "n_steps"),
        "tree_depth_mean": _sample_stat_mean(sample_stats, "tree_depth"),
        "tree_depth_max": _sample_stat_max(sample_stats, "tree_depth"),
        "step_size_mean": _sample_stat_mean(sample_stats, "step_size"),
        "acceptance_rate_mean": _sample_stat_mean(sample_stats, "acceptance_rate"),
        "divergences": _sample_stat_sum(sample_stats, "diverging"),
    }
    if any(value is not None for value in fields.values()):
        log_timing(label, 0.0, **fields)


def _reset_retained_draws(trace: xr.DataTree, *, burn: int) -> xr.DataTree:
    """Relabel retained draws and preserve the discarded burn-in count.

    Args:
        trace: Inference data whose draw-bearing groups are relabelled in place.
        burn: Number of discarded burn-in draws to record in metadata.

    Returns:
        The mutated inference data, with each draw coordinate reset to
        consecutive zero-based integers and ``burn`` stored on the trace and
        draw-bearing groups.
    """
    return _shared_reset_retained_draws(trace, burn=burn)


class RhimeSampler:
    """PyMC sampler configuration and execution for RHIME models.

    Args:
        draws: Number of post-tuning draws requested from PyMC.
        burn: Number of draws to discard from each chain after sampling.
        tune: Number of PyMC tuning draws.
        chains: Number of MCMC chains.
        nuts_sampler: PyMC NUTS backend name.
        progressbar: Whether PyMC progress output should be shown.
        sample_kwargs: Extra keyword arguments forwarded to ``pm.sample``.
        sample_prior_predictive: Whether to append prior predictive draws.
        sample_posterior_predictive: Whether to append posterior predictive
            draws, or variable names to sample.
        posterior_predictive_kwargs: Extra keyword arguments forwarded to
            ``pm.sample_posterior_predictive``.
    """

    __slots__ = (
        "draws",
        "burn",
        "tune",
        "chains",
        "nuts_sampler",
        "progressbar",
        "sample_kwargs",
        "sample_prior_predictive",
        "sample_posterior_predictive",
        "posterior_predictive_kwargs",
    )

    draws: int
    burn: int
    tune: int
    chains: int
    nuts_sampler: NutsSampler
    progressbar: bool
    sample_kwargs: dict[str, Any] | None
    sample_prior_predictive: bool | int
    sample_posterior_predictive: bool | tuple[str, ...]
    posterior_predictive_kwargs: dict[str, Any] | None

    def __init__(
        self,
        *,
        draws: int = 1000,
        burn: int = 0,
        tune: int = 1000,
        chains: int = 4,
        nuts_sampler: NutsSampler | str = "pymc",
        progressbar: bool = False,
        sample_kwargs: dict[str, Any] | None = None,
        sample_prior_predictive: bool | int = True,
        sample_posterior_predictive: bool | Sequence[str] = ("y",),
        posterior_predictive_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.draws = int(draws)
        self.burn = int(burn)
        self.tune = int(tune)
        self.chains = int(chains)
        self.nuts_sampler = self._normalise_nuts_sampler(nuts_sampler)
        self.progressbar = bool(progressbar)
        self.sample_kwargs = None if sample_kwargs is None else dict(sample_kwargs)
        self.sample_prior_predictive = sample_prior_predictive
        self.sample_posterior_predictive = self._normalise_posterior_predictive(sample_posterior_predictive)
        self.posterior_predictive_kwargs = (
            None if posterior_predictive_kwargs is None else dict(posterior_predictive_kwargs)
        )

    @staticmethod
    def _normalise_nuts_sampler(value: NutsSampler | str) -> NutsSampler:
        """Validate and normalize the PyMC NUTS backend name."""
        if value not in ("pymc", "nutpie", "numpyro", "blackjax"):
            raise ValueError("`nuts_sampler` must be one of 'pymc', 'nutpie', 'numpyro', or 'blackjax'.")
        return cast(NutsSampler, value)

    @staticmethod
    def _normalise_posterior_predictive(
        value: bool | Sequence[str],
    ) -> bool | tuple[str, ...]:
        """Normalize posterior predictive settings while preserving booleans."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return (value,)
        return tuple(str(item) for item in value)

    def __eq__(self, other: object) -> bool:
        """Compare sampler settings."""
        if not isinstance(other, RhimeSampler):
            return NotImplemented
        return all(getattr(self, name) == getattr(other, name) for name in self.__slots__)

    def __repr__(self) -> str:
        """Return a concise representation for tests and debugging."""
        args = ", ".join(f"{name}={getattr(self, name)!r}" for name in self.__slots__)
        return f"{type(self).__name__}({args})"

    def sample(self, model: pm.Model) -> xr.DataTree:
        """Sample a built RHIME model and append requested predictive groups."""
        sample_kwargs = dict(self.sample_kwargs or {})
        sample_kwargs.pop("return_inferencedata", None)
        idata_kwargs = dict(sample_kwargs.pop("idata_kwargs", {}))
        idata_kwargs["log_likelihood"] = True
        sample_kwargs.setdefault("progressbar", self.progressbar)
        sample_kwargs.setdefault("cores", self.chains)

        timing_start = timer_start()
        with model:
            raw_trace = cast(
                xr.DataTree,
                pm.sample(
                    draws=self.draws,
                    tune=self.tune,
                    chains=self.chains,
                    return_inferencedata=True,
                    idata_kwargs=idata_kwargs,
                    nuts_sampler=self.nuts_sampler,
                    **sample_kwargs,
                ),
            )
        log_timing(
            "rhime.sampler.pm_sample",
            timer_seconds(timing_start),
            draws=self.draws,
            tune=self.tune,
            chains=self.chains,
            nuts_sampler=self.nuts_sampler,
        )
        _log_sample_stats(raw_trace, label="rhime.sampler.sample_stats")

        timing_start = timer_start()
        trace = cast(xr.DataTree, raw_trace.isel(draw=slice(self.burn, None)))
        trace = _reset_retained_draws(trace, burn=self.burn)
        log_timing("rhime.sampler.burn_slicing", timer_seconds(timing_start), burn=self.burn)

        trace = self._extend_predictive(trace, model=model)
        timing_start = timer_start()
        registry = get_coord_registry(model)
        if registry is not None:
            trace = restore_inferencedata_coords(trace, registry)
        log_timing(
            "rhime.sampler.coord_restore",
            timer_seconds(timing_start),
            restored=registry is not None,
        )
        return trace

    def _extend_predictive(self, trace: xr.DataTree, *, model: pm.Model) -> xr.DataTree:
        """Extend sampled trace with configured predictive groups."""
        if self.sample_prior_predictive:
            prior_draws = (
                trace["posterior"].sizes["draw"]
                if self.sample_prior_predictive is True
                else int(self.sample_prior_predictive)
            )
            timing_start = timer_start()
            with model:
                trace.update(pm.sample_prior_predictive(prior_draws, model))
            log_timing(
                "rhime.sampler.prior_predictive",
                timer_seconds(timing_start),
                draws=prior_draws,
            )

        if self.sample_posterior_predictive:
            posterior_var_names = (
                None if self.sample_posterior_predictive is True else list(self.sample_posterior_predictive)
            )
            posterior_predictive_kwargs = dict(self.posterior_predictive_kwargs or {})
            posterior_predictive_kwargs.setdefault("model", model)
            if posterior_var_names is not None:
                posterior_predictive_kwargs.setdefault("var_names", posterior_var_names)
            timing_start = timer_start()
            with model:
                trace.update(pm.sample_posterior_predictive(trace, **posterior_predictive_kwargs))
            log_timing(
                "rhime.sampler.posterior_predictive",
                timer_seconds(timing_start),
                var_names=posterior_var_names,
            )

        return trace
