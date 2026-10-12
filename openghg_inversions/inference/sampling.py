"""PyMC inference and predictive draws from an explicit model and variable roles.

This layer consumes a model; it does not acquire data, select a scientific
recipe, construct reconstruction bindings, or write products.
"""

from __future__ import annotations

# ruff: noqa: E402

from openghg_inversions._pymc_config import configure_pytensor

configure_pytensor()

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal, cast

import numpy as np
import pymc as pm
import xarray as xr
from pymc.util import get_random_generator  # type: ignore[import-untyped]

from openghg_inversions._timing import log_timing, timer_seconds, timer_start
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
    """Log compact sampler diagnostics from a trace tree."""
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
        trace: Trace tree whose draw-bearing groups are relabelled in place.
        burn: Number of discarded burn-in draws to record in metadata.

    Returns:
        The mutated trace tree, with each draw coordinate reset to
        consecutive zero-based integers and ``burn`` stored on the trace and
        draw-bearing groups.
    """
    trace.attrs["burn"] = burn
    for group_name, node in trace.children.items():
        group = node.to_dataset()
        if "draw" not in group.dims:
            continue
        group = group.assign_coords(draw=np.arange(group.sizes["draw"]))
        group.attrs["burn"] = burn
        trace[group_name] = group
    return trace


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
            Its ``random_seed`` also seeds independent prior and posterior
            predictive streams without consuming a caller-owned generator.
        sample_prior_predictive: Whether to append prior predictive draws.
        sample_posterior_predictive: Whether to append posterior predictive
            draws, or variable names to sample.
        posterior_predictive_kwargs: Extra keyword arguments forwarded to
            ``pm.sample_posterior_predictive``. Explicit ``random_seed`` and
            ``var_names`` override their sampler defaults; ``random_seed=None``
            requests unseeded posterior prediction.
    """

    # The established configured frontend exposes this subset. Predictive
    # selection remains available on the independent sampler constructor.
    CONFIG_OPTION_NAMES: ClassVar[tuple[str, ...]] = (
        "draws", "burn", "tune", "chains", "nuts_sampler", "progressbar",
        "sample_kwargs", "posterior_predictive_kwargs",
    )
    INTEGER_OPTION_NAMES: ClassVar[tuple[str, ...]] = ("draws", "burn", "tune", "chains")
    MAPPING_OPTION_NAMES: ClassVar[tuple[str, ...]] = ("sample_kwargs", "posterior_predictive_kwargs")

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

    def sample(
        self,
        model: pm.Model,
        *,
        variable_roles: Mapping[str, str] | None = None,
    ) -> xr.DataTree:
        """Sample a built RHIME model and append requested predictive groups.

        Args:
            model: Concrete PyMC model to sample.
            variable_roles: Optional semantic-role manifest from a
                ``RhimeModelBuildResult``. Posterior-predictive entries may be
                role names. The historical default ``"y"`` resolves to the
                explicit ``concentration`` role when a custom model has no
                variable named ``y``.
        """
        sample_kwargs = dict(self.sample_kwargs or {})
        sample_kwargs.pop("return_inferencedata", None)
        idata_kwargs = dict(sample_kwargs.pop("idata_kwargs", {}))
        idata_kwargs.setdefault("log_likelihood", True)
        sample_kwargs.setdefault("progressbar", self.progressbar)
        sample_kwargs.setdefault("cores", self.chains)
        if not isinstance(self.sample_prior_predictive, bool | int | np.integer) or self.sample_prior_predictive < 0:
            raise ValueError("sample_prior_predictive must be a boolean or non-negative integer.")
        posterior_kwargs = self._posterior_predictive_options(model, variable_roles)
        # Reserve prediction siblings after PyMC's per-chain streams.
        root_rng = get_random_generator(sample_kwargs.get("random_seed"), copy=True)
        prior_rng, posterior_rng = root_rng.spawn(self.chains + 2)[-2:]
        posterior_kwargs.setdefault("random_seed", posterior_rng)
        posterior_seed = posterior_kwargs["random_seed"]
        if self.sample_posterior_predictive and posterior_seed is not None:
            normalized_seed = get_random_generator(posterior_seed, copy=True)
            if not isinstance(posterior_seed, int | np.integer):
                posterior_kwargs["random_seed"] = normalized_seed

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
        trace = raw_trace.isel(draw=slice(self.burn, None), missing_dims="ignore")
        trace = _reset_retained_draws(trace, burn=self.burn)
        log_timing("rhime.sampler.burn_slicing", timer_seconds(timing_start), burn=self.burn)

        trace = self._extend_predictive(
            trace, model=model, prior_random_seed=prior_rng,
            posterior_predictive_kwargs=posterior_kwargs,
        )
        for group in trace.children.values():
            if "draw" in group.dims:
                group.attrs.setdefault("burn", self.burn)
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

    def _posterior_predictive_options(
        self,
        model: pm.Model,
        variable_roles: Mapping[str, str] | None,
    ) -> dict[str, Any]:
        """Resolve prediction names before MCMC, scoping roles to their model.

        Alternate models use concrete names. For the sampled model, a missing
        default ``y`` may resolve through an explicit concentration role.
        """
        kwargs = dict(self.posterior_predictive_kwargs or {})
        kwargs.setdefault("model", model)
        if not self.sample_posterior_predictive:
            return kwargs
        if not isinstance(self.sample_posterior_predictive, bool):
            kwargs.setdefault("var_names", list(self.sample_posterior_predictive))
        names = kwargs.get("var_names")
        if names is not None:
            if isinstance(names, str):
                names = [names]
            prediction_model = kwargs.get("model") or model
            roles = dict(variable_roles or {}) if prediction_model is model else {}
            if "y" not in prediction_model.named_vars and "concentration" in roles:
                roles.setdefault("y", roles["concentration"])
            resolved = [roles.get(name, name) for name in names]
            missing = sorted(set(resolved) - set(prediction_model.named_vars))
            if missing:
                raise ValueError(
                    "RhimeSampler posterior-predictive variables are absent from the model: "
                    f"{missing!r}. Use concrete model names or semantic keys from `variable_roles`."
                )
            kwargs["var_names"] = list(dict.fromkeys(resolved))
        return kwargs

    def _extend_predictive(
        self,
        trace: xr.DataTree,
        *,
        model: pm.Model,
        prior_random_seed: np.random.Generator,
        posterior_predictive_kwargs: dict[str, Any],
    ) -> xr.DataTree:
        """Extend sampled trace with configured predictive groups."""
        if self.sample_prior_predictive:
            prior_draws = (
                trace["posterior"].sizes["draw"]
                if self.sample_prior_predictive is True
                else int(self.sample_prior_predictive)
            )
            timing_start = timer_start()
            with model:
                trace.update(pm.sample_prior_predictive(prior_draws, model, random_seed=prior_random_seed))
            log_timing(
                "rhime.sampler.prior_predictive",
                timer_seconds(timing_start),
                draws=prior_draws,
            )

        if self.sample_posterior_predictive:
            timing_start = timer_start()
            with model:
                trace.update(pm.sample_posterior_predictive(trace, **posterior_predictive_kwargs))
            log_timing(
                "rhime.sampler.posterior_predictive",
                timer_seconds(timing_start),
                var_names=posterior_predictive_kwargs.get("var_names"),
            )

        return trace
