"""PyMC inference and predictive draws from an explicit model and variable roles.

This layer consumes a model; it does not acquire data, select a scientific
recipe, construct reconstruction bindings, or write products.
"""

from __future__ import annotations

# ruff: noqa: E402

from openghg_inversions._pymc_config import configure_pytensor

configure_pytensor()

import json
import logging
import os
from collections.abc import Mapping, Sequence
from copy import copy
from typing import Any, ClassVar, Literal, cast

import numpy as np
import pymc as pm
import xarray as xr
from pymc.util import get_random_generator  # type: ignore[import-untyped]
from pytensor.compile.mode import Mode
from pytensor.graph.traversal import ancestors

from openghg_inversions._timing import log_timing, timer_seconds, timer_start
from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords
from openghg_inversions.postprocessing.contracts import resolve_predictive_names

from .convergence import posterior_convergence_check

NutsSampler = Literal["pymc", "nutpie", "numpyro", "blackjax"]
_JAX_NUTS_SAMPLERS = frozenset({"numpyro", "blackjax"})

logger = logging.getLogger(__name__)


def _jax_sampling_precision(model: pm.Model, nuts_sampler: NutsSampler) -> dict[str, bool | str]:
    """Check and record process-start precision for a JAX sampler.

    Args:
        model: Built model whose sampler value variables must retain their dtype.
        nuts_sampler: Resolved NUTS backend.

    Returns:
        PyTensor's floatX and the effective JAX x64 flag, or an empty mapping
        for a backend that does not use JAX.

    Raises:
        ValueError: If the current JAX runtime disagrees with PyTensor floatX
            or cannot represent the model's value variables.
    """
    if nuts_sampler not in _JAX_NUTS_SAMPLERS:
        return {}

    import jax
    import pytensor
    import pytensor.link.jax.dispatch  # noqa: F401  # First import sets JAX precision from floatX.

    floatx = str(pytensor.config.floatX)
    x64 = bool(jax.config.x64_enabled)
    if x64 != (floatx == "float64") or (
        not x64 and any(variable.dtype == "float64" for variable in model.value_vars)
    ):
        raise ValueError(
            "JAX precision does not match PyTensor's floatX or the model's value variables. "
            "Start a fresh Python process with PYTENSOR_FLAGS=floatX=float32 or floatX=float64 "
            "before building and sampling the model."
        )
    if not x64 and os.environ.get("JAX_ENABLE_X64", "").lower() in {"y", "yes", "t", "true", "on", "1"}:
        logger.warning(
            "PyTensor floatX selects float32 for this JAX sampler. Set "
            "PYTENSOR_FLAGS=floatX=float64 before starting Python to sample in float64."
        )

    return {"floatX": floatx, "jax_enable_x64": x64}


def describe_recorded_sampler(recorded: object) -> dict[str, int | str]:
    """Normalize available inference metadata without constructing a sampler.

    Only the fields returned by ``RhimeSampler.describe`` are retained.
    Missing fields stay absent; an invalid known value makes the description
    unavailable without preventing scientific replay.
    """
    if not isinstance(recorded, Mapping):
        return {}
    try:
        description: dict[str, int | str] = {
            name: int(recorded[name]) for name in RhimeSampler.INTEGER_OPTION_NAMES if name in recorded
        }
        if "nuts_sampler" in recorded:
            description["nuts_sampler"] = RhimeSampler._normalise_nuts_sampler(recorded["nuts_sampler"])
    except (TypeError, ValueError, OverflowError):
        return {}
    return description


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


def _burn_retained_draws(trace: xr.DataTree, *, burn: int) -> xr.DataTree:
    """Discard and relabel post-tuning draws, preserving independent groups.

    Args:
        trace: Owned trace tree whose posterior-aligned groups are sliced
            in place. Other groups, such as a ``posterior_predictive`` group
            supplied through ``idata_kwargs``, are sliced too when their chain
            and draw indexes match the original posterior's; warmup and prior
            groups keep their own axes.
        burn: Number of post-tuning draws to discard and record in metadata.

    Returns:
        The mutated trace tree, with retained draw coordinates reset to
        consecutive zero-based integers and ``burn`` stored on the trace and
        posterior-aligned groups.
    """
    trace.attrs["burn"] = burn
    # PyMC computes both log-density groups from the same posterior states, and
    # ZarrTrace(include_transformed=True) records those states' transformed values.
    aligned = ["posterior", "unconstrained_posterior", "sample_stats", "log_likelihood", "log_prior"]
    if "posterior" in trace.children:
        posterior = trace["posterior"].to_dataset()
        indexes = {name: posterior.indexes[name] for name in ("chain", "draw") if name in posterior.indexes}
        for name, child in trace.children.items():
            group = child.to_dataset()
            if (
                name not in aligned
                and not name.startswith(("warmup_", "prior"))
                and len(indexes) == 2
                and all(dim in group.indexes and group.indexes[dim].equals(index) for dim, index in indexes.items())
            ):
                aligned.append(name)

    for group_name in aligned:
        if group_name not in trace.children:
            continue
        group = trace[group_name].to_dataset()
        if "draw" not in group.dims:
            continue
        group = group.isel(draw=slice(burn, None))
        group = group.assign_coords(draw=np.arange(group.sizes["draw"]))
        group.attrs["burn"] = burn
        trace[group_name] = group
    return trace


def _extend_log_likelihood(
    trace: xr.DataTree,
    model: pm.Model,
    selection: Literal[True] | Sequence[str],
    *,
    compile_kwargs: dict[str, Any],
    progressbar: bool = False,
) -> xr.DataTree:
    """Attach posterior log likelihood with the selected model compiler."""
    with model:
        return cast(
            xr.DataTree,
            pm.compute_log_likelihood(
                trace,
                model=model,
                var_names=None if selection is True else list(selection),
                extend_inferencedata=True,
                progressbar=progressbar,
                compile_kwargs=compile_kwargs,
            ),
        )


def _restore_deterministics(
    posterior: xr.Dataset,
    model: pm.Model,
    *,
    compile_kwargs: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Evaluate the unchanged graph after pm.sample returns.

    PyMC allocates the final arrays once and evaluates one state at a time.
    Merging borrows those arrays.

    Args:
        posterior: Labelled posterior or warmup samples of the free states.
        model: Model that produced the samples.
        compile_kwargs: Compiler options resolved for the fitted model.

    Returns:
        The supplied samples with all deterministic values on the same axes.
    """
    if not model.deterministics:
        return posterior

    # Match ordinary recording's observed-value and transform conventions.
    options = dict(compile_kwargs or {})
    givens = dict(options.get("givens") or ())
    for rv in model.observed_RVs:
        value = model.rvs_to_values[rv]
        transform = model.rvs_to_transforms.get(rv)
        if transform is not None:
            value = transform.backward(value, *rv.owner.inputs)
            value = rv.type.filter_variable(value, allow_convert=True)
        givens[rv] = value
    options["givens"] = givens

    derived = cast(
        xr.Dataset,
        pm.compute_deterministics(posterior, model=model, progressbar=False, compile_kwargs=options),
    )
    derived = derived.assign_coords(chain=posterior.chain, draw=posterior.draw)
    return xr.merge([posterior, derived], join="exact", compat="override")


def _numpy_callback_compile_kwargs(model: pm.Model) -> dict[str, Any]:
    """Return compile options for a model whose density calls a NumPy/SciPy Op holding prepared arrays.

    Some likelihood Ops (fixed OU, cached sigma) evaluate prepared NumPy/SciPy
    targets in ``perform``. PyTensor's default Numba linker has no native
    implementation for them, so it runs them through an object-mode callback
    that serializes the Op, including every array it holds, into each compiled
    function. That costs several times the arrays' memory per compiled
    function (logp, gradient, log likelihood, predictive draws), copies the
    payload on each call, and fails outright once the serialized Op exceeds
    2 GiB. The C/VM linker calls ``perform`` directly.

    Args:
        model: Model whose joint log density is inspected.

    Returns:
        ``{"mode": Mode(linker="cvm", optimizer="fast_run")}`` when the log
        density contains an Op marked ``holds_numpy_target = True``, otherwise
        an empty dict, which keeps PyTensor's default mode.
    """
    logp = getattr(model, "logp", None)
    if logp is None:
        return {}

    for variable in ancestors([logp()]):
        if variable.owner is not None and getattr(variable.owner.op, "holds_numpy_target", False):
            return {"mode": Mode(linker="cvm", optimizer="fast_run")}
    return {}


def sample_prior_predictive(
    model: pm.Model,
    *,
    draws: int,
    random_seed: int | np.random.Generator | None = None,
    compile_kwargs: dict[str, Any] | None = None,
) -> xr.DataTree:
    """Draw prior predictions using the model's safe PyTensor linker.

    Args:
        model: Concrete PyMC model to predict from.
        draws: Number of prior-predictive draws.
        random_seed: Seed or generator for the prediction stream.
        compile_kwargs: Already resolved PyMC compile options, if available.
            Omit to select options from the model. An empty mapping keeps the
            PyMC defaults.

    Returns:
        Prior and prior-predictive draws in PyMC's trace format.
    """
    if compile_kwargs is None:
        compile_kwargs = _numpy_callback_compile_kwargs(model)
    if compile_kwargs:
        return pm.sample_prior_predictive(draws, model, random_seed=random_seed, compile_kwargs=compile_kwargs)
    return pm.sample_prior_predictive(draws, model, random_seed=random_seed)


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
            ``compute_convergence_checks`` defaults to ``False`` because the
            sampler assesses retained draws after burn slicing. Set it to
            ``True`` to also run PyMC's convergence checks.
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
        defer_deterministics: bool = False,
    ) -> xr.DataTree:
        """Sample a built RHIME model and append requested predictive groups.

        Args:
            model: Concrete PyMC model to sample.
            variable_roles: Optional semantic-role manifest from a
                ``RhimeModelBuildResult``. Posterior-predictive entries may be
                role names. The historical default ``"y"`` resolves to the
                explicit ``concentration`` role when a custom model has no
                variable named ``y``.
            defer_deterministics: For PyMC, record only free random variables
                during sampling, then evaluate deterministics for the retained
                posterior and any saved warmup draws. The likelihood still
                evaluates terms it needs during sampling. The default records
                deterministics during sampling. Other backends, explicit
                traces or callbacks, a separate conversion model in
                ``idata_kwargs``, transformed-variable requests, and empty
                retained groups also use ordinary recording.

        Returns:
            Sampled trace with requested predictive groups, after burn slicing
            and registered-coordinate restoration. Its ``sampler_convergence``
            attribute contains a JSON convergence report from
            :func:`~openghg_inversions.inference.convergence.posterior_convergence_check`
            using ArviZ's default thresholds. The report assesses the model's free
            random variables, whose names are saved as JSON in
            ``sampler_convergence_variables``. Assessment is eager; a failed or
            incomplete convergence report does not prevent returning the trace.
            For the ``numpyro`` and ``blackjax`` backends, ``sampler_precision``
            records the PyTensor floatX and effective JAX x64 flag as JSON.

        Raises:
            ValueError: If the sampler or predictive options are invalid, or
                the JAX runtime is incompatible with the model's precision.
        """
        _validate_idata_options(self.sample_kwargs, label="sample_kwargs")
        sample_kwargs = _copy_sampler_options(self.sample_kwargs or {})
        sample_kwargs.pop("return_inferencedata", None)
        idata_kwargs = dict(sample_kwargs.pop("idata_kwargs", {}))
        idata_kwargs.setdefault("log_likelihood", True)

        compile_kwargs = _numpy_callback_compile_kwargs(model) if self.nuts_sampler == "pymc" else {}
        if compile_kwargs:
            compile_kwargs.update(sample_kwargs.get("compile_kwargs") or {})
            if sample_kwargs.get("backend") is None:
                sample_kwargs["compile_kwargs"] = compile_kwargs

        compact = (
            defer_deterministics
            and self.nuts_sampler == "pymc"
            and not idata_kwargs.get("include_transformed", False)
            and sample_kwargs.get("trace") is None
            and sample_kwargs.get("callback") is None
            and self.burn < self.draws
            # A different conversion model owns observed data and labels.
            and idata_kwargs.get("model", model) is model
        )
        deferred_log_likelihood: bool | Sequence[str] = False
        if compact:
            sample_kwargs["var_names"] = [rv.name for rv in model.free_RVs]
            # Compute likelihood using only retained posterior states.
            deferred_log_likelihood = idata_kwargs["log_likelihood"]
            idata_kwargs["log_likelihood"] = False
        elif compile_kwargs:
            # PyMC's automatic log-likelihood calculation ignores compile options.
            deferred_log_likelihood = idata_kwargs.pop("log_likelihood")

        # Assess the retained samples after burn slicing instead of repeating PyMC's pre-burn checks.
        sample_kwargs.setdefault("compute_convergence_checks", False)
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

        precision = _jax_sampling_precision(model, self.nuts_sampler)

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

        if deferred_log_likelihood and not compact:
            raw_trace = _extend_log_likelihood(
                raw_trace, model, deferred_log_likelihood, compile_kwargs=compile_kwargs,
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
        trace = raw_trace.copy(deep=False)
        trace = _burn_retained_draws(trace, burn=self.burn)
        log_timing("rhime.sampler.burn_slicing", timer_seconds(timing_start), burn=self.burn)

        if compact:
            evaluation_compile_kwargs = compile_kwargs or sample_kwargs.get("compile_kwargs") or {}
            if deferred_log_likelihood:
                timing_start = timer_start()
                trace = _extend_log_likelihood(
                    trace, model, deferred_log_likelihood, compile_kwargs=evaluation_compile_kwargs,
                    progressbar=self.progressbar,
                )
                log_timing("rhime.sampler.log_likelihood", timer_seconds(timing_start))

            timing_start = timer_start()
            for group_name in ("posterior", "warmup_posterior"):
                if group_name in trace.children:
                    trace[group_name] = _restore_deterministics(
                        trace[group_name].to_dataset(), model, compile_kwargs=evaluation_compile_kwargs,
                    )
            log_timing("rhime.sampler.deterministics", timer_seconds(timing_start))

        trace = self._extend_predictive(
            trace,
            model=model,
            prior_random_seed=prior_rng,
            posterior_predictive_kwargs=posterior_kwargs,
            compile_kwargs=compile_kwargs,
        )
        for name, group in trace.children.items():
            if not name.startswith("warmup_") and "draw" in group.dims:
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

        if precision:
            trace.attrs["sampler_precision"] = json.dumps(precision, sort_keys=True)
        variable_names = [variable.name for variable in model.free_RVs]
        trace.attrs["sampler_convergence_variables"] = json.dumps(variable_names)
        timing_start = timer_start()
        _, convergence = posterior_convergence_check(trace, variable_names=variable_names)
        trace.attrs["sampler_convergence"] = json.dumps(convergence, sort_keys=True)
        log_timing(
            "rhime.sampler.convergence",
            timer_seconds(timing_start),
            status=convergence["status"],
            message=convergence["message"],
            **convergence["measured_values"],
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
        compile_kwargs: dict[str, Any],
    ) -> xr.DataTree:
        """Extend sampled trace with configured predictive groups."""
        if self.sample_prior_predictive:
            prior_draws = (
                trace["posterior"].sizes["draw"]
                if self.sample_prior_predictive is True
                else int(self.sample_prior_predictive)
            )
            timing_start = timer_start()
            trace.update(
                sample_prior_predictive(
                    model, draws=prior_draws, random_seed=prior_random_seed, compile_kwargs=compile_kwargs
                )
            )
            log_timing(
                "rhime.sampler.prior_predictive",
                timer_seconds(timing_start),
                draws=prior_draws,
            )

        if self.sample_posterior_predictive:
            if (
                "compile_kwargs" not in posterior_predictive_kwargs
                and posterior_predictive_kwargs.get("backend") is None
            ):
                prediction_model = posterior_predictive_kwargs.get("model") or model
                prediction_options = (
                    compile_kwargs if prediction_model is model else _numpy_callback_compile_kwargs(prediction_model)
                )
                if prediction_options:
                    posterior_predictive_kwargs["compile_kwargs"] = prediction_options

            timing_start = timer_start()
            with model:
                trace.update(pm.sample_posterior_predictive(trace, **posterior_predictive_kwargs))
            log_timing(
                "rhime.sampler.posterior_predictive",
                timer_seconds(timing_start),
                var_names=posterior_predictive_kwargs.get("var_names"),
            )

        return trace
