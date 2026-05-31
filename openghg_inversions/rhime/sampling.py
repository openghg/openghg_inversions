"""RHIME sampling helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

import arviz as az
import pymc as pm

NutsSampler = Literal["pymc", "nutpie", "numpyro", "blackjax"]


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

    def sample(self, model: pm.Model) -> az.InferenceData:
        """Sample a built RHIME model and append requested predictive groups."""
        sample_kwargs = dict(self.sample_kwargs or {})
        sample_kwargs.pop("return_inferencedata", None)
        idata_kwargs = dict(sample_kwargs.pop("idata_kwargs", {}))
        idata_kwargs["log_likelihood"] = True
        sample_kwargs.setdefault("progressbar", self.progressbar)
        sample_kwargs.setdefault("cores", self.chains)

        with model:
            raw_trace = cast(
                az.InferenceData,
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

        trace = cast(az.InferenceData, raw_trace.isel(draw=slice(self.burn, None)))
        return self._extend_predictive(trace, model=model)

    def _extend_predictive(self, trace: az.InferenceData, *, model: pm.Model) -> az.InferenceData:
        """Extend sampled trace with configured predictive groups."""
        if self.sample_prior_predictive:
            prior_draws = (
                cast(Any, trace).posterior.sizes["draw"]
                if self.sample_prior_predictive is True
                else int(self.sample_prior_predictive)
            )
            with model:
                trace.extend(pm.sample_prior_predictive(prior_draws, model))

        if self.sample_posterior_predictive:
            posterior_var_names = (
                None if self.sample_posterior_predictive is True else list(self.sample_posterior_predictive)
            )
            posterior_predictive_kwargs = dict(self.posterior_predictive_kwargs or {})
            posterior_predictive_kwargs.setdefault("model", model)
            if posterior_var_names is not None:
                posterior_predictive_kwargs.setdefault("var_names", posterior_var_names)
            with model:
                trace.extend(pm.sample_posterior_predictive(trace, **posterior_predictive_kwargs))

        return trace
