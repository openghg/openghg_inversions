"""PyMC inference and predictive draws from an explicit model and variable roles.

This layer consumes a model; it does not acquire data, select a scientific
recipe, construct reconstruction bindings, or write products.
"""

from __future__ import annotations

# ruff: noqa: E402

from openghg_inversions._pymc_config import configure_pytensor

configure_pytensor()

from collections.abc import Mapping, Sequence
from copy import copy
import json
from typing import Any, ClassVar, Literal, cast

import numpy as np
import pymc as pm
import xarray as xr
from pymc.util import get_random_generator  # type: ignore[import-untyped]

from openghg_inversions._timing import log_timing, timer_seconds, timer_start
from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords
from openghg_inversions.models.priors import prior_records
from openghg_inversions.postprocessing.contracts import resolve_predictive_names

NutsSampler = Literal["pymc", "nutpie", "numpyro", "blackjax"]


def _copy_sampler_options(value: Any) -> Any:
    """Own keyword containers and NumPy options, borrowing opaque runtime objects."""
    if isinstance(value, dict):
        return {key: _copy_sampler_options(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_sampler_options(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_sampler_options(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _validate_idata_options(kwargs: Mapping[str, Any] | None, *, label: str) -> None:
    """Keep posterior storage and inference coordinates owned by the model."""
    if label == "sample_kwargs" and "var_names" in (kwargs or {}):
        raise ValueError("sampler.sample_kwargs must not contain var_names; retain the complete posterior.")
    idata_kwargs = (kwargs or {}).get("idata_kwargs", {})
    if idata_kwargs is None:
        return
    forbidden = {"coords", "dims"}.intersection(idata_kwargs)
    if forbidden:
        raise ValueError(
            f"sampler.{label}['idata_kwargs'] must not contain "
            f"{sorted(forbidden)!r}; register coordinates and dimensions on the model instead."
        )


def _sampler_options_equal(left: Any, right: Any) -> bool:
    """Compare owned option values without materializing borrowed arrays."""
    if left is right:
        return True
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _sampler_options_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _sampler_options_equal(a, b) for a, b in zip(left, right)
        )
    equal = left == right
    return isinstance(equal, (bool, np.bool_)) and bool(equal)


def resolve_sampler(
    sampler: RhimeSampler | Literal["default"],
    **default_options: Any,
) -> RhimeSampler:
    """Resolve a live request into an invocation-owned sampler.

    Args:
        sampler: Configured sampler or ``"default"`` for fresh recipe defaults.
        **default_options: Recipe-specific defaults used only for ``"default"``.

    Returns:
        A fresh sampler with independent option containers. Models, steps, and
        borrowed xarray or Dask values retain their identity.

    Raises:
        TypeError: If the request is neither a sampler nor ``"default"``.
        ValueError: If default options override posterior storage or model coordinates.
    """
    if isinstance(sampler, str) and sampler == "default":
        return RhimeSampler(**default_options)
    if not isinstance(sampler, RhimeSampler):
        raise TypeError(
            "sampler must be a RhimeSampler or 'default'; None no longer selects defaults. "
            "Omit the optional sampler argument or pass sampler='default'."
        )
    resolved = copy(sampler)
    resolved.sample_kwargs = _copy_sampler_options(sampler.sample_kwargs)
    resolved.posterior_predictive_kwargs = _copy_sampler_options(sampler.posterior_predictive_kwargs)
    return resolved


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
            Nested containers and NumPy option arrays are copied; opaque
            runtime objects and xarray/Dask values are borrowed.
            Posterior ``var_names`` overrides are invalid because sampling
            retains the complete posterior. ``idata_kwargs.coords`` and ``dims``
            are invalid because model registration owns coordinates and dimensions.
            Its ``random_seed`` also seeds independent prior and posterior
            predictive streams without consuming a caller-owned generator.
        sample_prior_predictive: Whether to append prior predictive draws.
        sample_posterior_predictive: Whether to append posterior predictive
            draws, or variable names to sample.
        posterior_predictive_kwargs: Extra keyword arguments forwarded to
            ``pm.sample_posterior_predictive``, with the same ownership rules
            as ``sample_kwargs``. Explicit ``random_seed`` and ``var_names``
            override their sampler defaults; ``random_seed=None`` requests
            unseeded posterior prediction.

    Raises:
        ValueError: If the NUTS backend is unknown, posterior variable selection
            is overridden, or inference coordinates override model registration.
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
        _validate_idata_options(sample_kwargs, label="sample_kwargs")
        _validate_idata_options(posterior_predictive_kwargs, label="posterior_predictive_kwargs")
        self.sample_kwargs = _copy_sampler_options(sample_kwargs)
        self.sample_prior_predictive = sample_prior_predictive
        self.sample_posterior_predictive = self._normalise_posterior_predictive(sample_posterior_predictive)
        self.posterior_predictive_kwargs = _copy_sampler_options(posterior_predictive_kwargs)

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

    def describe(self) -> dict[str, int | str]:
        """Snapshot the sampling settings used by output metadata."""
        return {
            "draws": self.draws,
            "burn": self.burn,
            "tune": self.tune,
            "chains": self.chains,
            "nuts_sampler": self.nuts_sampler,
        }

    def __eq__(self, other: object) -> bool:
        """Compare sampler settings."""
        if not isinstance(other, RhimeSampler):
            return NotImplemented
        return all(
            _sampler_options_equal(getattr(self, name), getattr(other, name)) for name in self.__slots__
        )

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
        _validate_idata_options(self.sample_kwargs, label="sample_kwargs")
        sample_kwargs = _copy_sampler_options(self.sample_kwargs or {})
        sample_kwargs.pop("return_inferencedata", None)
        idata_kwargs = dict(sample_kwargs.pop("idata_kwargs", {}))
        idata_kwargs.setdefault("log_likelihood", True)
        sample_kwargs.setdefault("progressbar", self.progressbar)
        sample_kwargs.setdefault("cores", self.chains)
        if not isinstance(self.sample_prior_predictive, bool | int | np.integer) or self.sample_prior_predictive < 0:
            raise ValueError("sample_prior_predictive must be a boolean or non-negative integer.")
        posterior_kwargs = self._posterior_predictive_options(model, variable_roles)
        # PyMC copies random_seed and spawns one child per chain. Copy here too
        # so spawning prediction streams cannot advance a caller-owned RNG.
        root_rng = get_random_generator(sample_kwargs.get("random_seed"), copy=True)
        # Skip those chain children; the next two are prediction siblings.
        # For two chains: [chain0, chain1, prior, posterior_predictive]. Spawn
        # both even if prior prediction is disabled, keeping the latter fixed.
        prior_rng, posterior_rng = root_rng.spawn(self.chains + 2)[-2:]
        # An explicit posterior seed wins, including None (unseeded prediction).
        posterior_kwargs.setdefault("random_seed", posterior_rng)
        posterior_seed = posterior_kwargs["random_seed"]
        if self.sample_posterior_predictive and posterior_seed is not None:
            # Validate before MCMC and copy mutable RNGs. Normalize sequences
            # to one RNG: posterior prediction otherwise treats them as a list
            # of per-chain seeds. Keep integers unchanged for native semantics.
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
        trace.attrs["rhime_priors"] = json.dumps(prior_records(model))
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
        kwargs = _copy_sampler_options(self.posterior_predictive_kwargs or {})
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
            kwargs["var_names"] = resolve_predictive_names(
                list(names), available=set(prediction_model.named_vars), variable_roles=roles,
            )
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
