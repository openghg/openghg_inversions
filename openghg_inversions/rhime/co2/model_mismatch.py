r"""Fixed model-mismatch choices for the exact Gaussian CO2 estimator.

``likelihood.fixed_model_mismatch`` is a known concentration standard
deviation :math:`\sigma` added in quadrature to ``mf_error``. With
``sampling.method = "gaussian"`` it may be given as:

- a scalar, applied to every observation;
- a per-site mapping ``{site = ppm}``, mapped onto observations through their
  ``site`` coordinate; every retained site needs a value and unknown sites are
  refused;
- a per-measurement ``xarray.DataArray`` on ``nmeasure`` (Python API);
- an **empirical Bayes** choice ``{empirical_bayes = "prior_residual_moment",
  floor = 0.1}``, estimated per site from the prior residuals.

Precedence is unchanged: an explicit value (in any of these forms) wins over a
prepared ``fixed_model_mismatch`` field, which wins over no mismatch.

Empirical Bayes uses the prior residual :math:`r = y - \bar y`, where
:math:`\bar y` is the affine likelihood mean at the prior mean of every state
(intercept, retained flux, boundary and offset prior means). Two modes exist:

- ``"prior_residual_moment"`` (default): per site,
  :math:`\sigma_s^2 = \max(0, \overline{r^2}_s - \overline{v}_s
  - \overline{A}_s - \overline{e^2}_s)`, where :math:`v` is the diagonal of
  the prior-predictive state covariance (:math:`H B H^T` plus the boundary and
  offset prior variance), :math:`A` the aggregation-error variance and
  :math:`e` the reported ``mf_error``. :math:`\sigma_s` is then floored.
- ``"prior_residual_rms"``: :math:`\sigma_s = \sqrt{\overline{r^2}_s}`, floored.
  This counts the prior flux spread, the boundary and offset spread, the
  aggregation error and the reported error as model mismatch too, so it
  overstates :math:`\sigma`.

Both are point estimates: the posterior treats :math:`\sigma` as known, so
it carries no :math:`\sigma` uncertainty, and the observations are used twice,
to set :math:`\sigma` and to fit. When the prior equals the truth (for example
the UOB control), the prior residual is pure observation noise plus aggregation
error, so an empirical-Bayes :math:`\sigma` there says nothing about real model
error.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any, Literal, get_args

import numpy as np
from numpy.typing import NDArray
import xarray as xr

EmpiricalBayesMode = Literal["prior_residual_moment", "prior_residual_rms"]
EMPIRICAL_BAYES_MODES: tuple[str, ...] = get_args(EmpiricalBayesMode)


@dataclass(frozen=True)
class EmpiricalBayesMismatch:
    """Per-site empirical-Bayes model mismatch, estimated from prior residuals.

    Attributes:
        mode: ``"prior_residual_moment"`` (default) or ``"prior_residual_rms"``;
            see the module documentation.
        floor: Minimum per-site standard deviation in observation units.
    """

    mode: EmpiricalBayesMode = "prior_residual_moment"
    floor: float = 0.0

    def __post_init__(self) -> None:
        """Validate the mode and the floor."""
        if self.mode not in EMPIRICAL_BAYES_MODES:
            raise ValueError(f"empirical_bayes must be one of {list(EMPIRICAL_BAYES_MODES)!r}, not {self.mode!r}.")
        if isinstance(self.floor, bool) or not isfinite(float(self.floor)) or float(self.floor) < 0:
            raise ValueError("The empirical-Bayes floor must be finite and non-negative.")


FixedMismatch = float | Mapping[str, float] | EmpiricalBayesMismatch | xr.DataArray


def parse_fixed_model_mismatch(value: object, path: str) -> float | Mapping[str, float] | EmpiricalBayesMismatch:
    """Parse a configured fixed mismatch: a scalar, a site table or an empirical-Bayes table.

    Args:
        value: Configured value.
        path: Configuration path used in messages.

    Returns:
        A non-negative scalar, a read-only site mapping or the empirical-Bayes
        choice.

    Raises:
        TypeError: If the value or an entry has the wrong type.
        ValueError: If a value is negative or non-finite, a table is empty, or
            an empirical-Bayes table has unknown options.
    """

    def number(item: object, item_path: str) -> float:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise TypeError(f"{item_path} must be numeric.")
        result = float(item)
        if not isfinite(result) or result < 0:
            raise ValueError(f"{item_path} must be finite and non-negative.")
        return result

    if not isinstance(value, Mapping):
        return number(value, path)
    table = dict(value)
    if "empirical_bayes" in table:
        mode = table.pop("empirical_bayes")
        if not isinstance(mode, str):
            raise TypeError(f"{path}.empirical_bayes must be a mode name.")
        floor = number(table.pop("floor"), f"{path}.floor") if "floor" in table else 0.0
        if table:
            raise ValueError(f"Unknown empirical-Bayes option(s): {', '.join(f'{path}.{name}' for name in table)}.")
        return EmpiricalBayesMismatch(mode=mode, floor=floor)  # type: ignore[arg-type]
    if not table:
        raise ValueError(f"{path} must not be an empty site table.")
    if any(not isinstance(site, str) or not site.strip() for site in table):
        raise TypeError(f"{path} site names must be non-empty strings.")
    return MappingProxyType({site.strip(): number(item, f"{path}.{site}") for site, item in table.items()})


def observation_sites(observations: xr.DataArray) -> NDArray[np.str_]:
    """Return the per-observation site labels.

    Raises:
        ValueError: If ``observations`` has no observation-aligned ``site`` coordinate.
    """
    site = observations.coords.get("site")
    if site is None or site.dims != observations.dims or observations.ndim != 1:
        raise ValueError("Per-site model mismatch requires an observation-aligned 'site' coordinate.")
    return np.asarray(site.values).astype(str)


def per_site_mismatch(sigma_by_site: Mapping[str, float], observations: xr.DataArray) -> xr.DataArray:
    """Map a per-site standard deviation onto the observations.

    Args:
        sigma_by_site: Standard deviation for every retained site.
        observations: Observations with a ``site`` coordinate.

    Returns:
        Labelled per-observation standard deviations named ``fixed_model_mismatch``.

    Raises:
        ValueError: If a configured site is not observed or an observed site
            has no value.
    """
    sites = observation_sites(observations)
    observed = list(dict.fromkeys(sites.tolist()))
    unknown = sorted(set(sigma_by_site) - set(observed))
    missing = [site for site in observed if site not in sigma_by_site]
    if unknown or missing:
        raise ValueError(
            "Per-site fixed_model_mismatch must name exactly the observed sites; "
            f"unknown: {unknown!r}, missing: {missing!r}."
        )
    values = np.array([sigma_by_site[site] for site in sites], dtype=np.float64)
    return xr.zeros_like(observations, dtype=np.float64).copy(data=values).rename("fixed_model_mismatch")


def empirical_bayes_mismatch(
    choice: EmpiricalBayesMismatch,
    *,
    sites: NDArray[np.str_],
    residual: NDArray[np.float64],
    prior_state_variance: NDArray[np.float64],
    aggregation_variance: NDArray[np.float64],
    error_variance: NDArray[np.float64],
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Estimate a per-site empirical-Bayes mismatch from prior residuals.

    Args:
        choice: Mode and floor.
        sites: Per-observation site labels.
        residual: ``y`` minus the affine likelihood mean at the prior state.
        prior_state_variance: Diagonal of the prior-predictive state
            covariance, including boundary and offset prior variance.
        aggregation_variance: Diagonal of the aggregation-error covariance.
        error_variance: ``mf_error ** 2``.

    Returns:
        Per-observation standard deviations and a JSON-safe record with, per
        site, the observation count, every averaged term, the raw estimate,
        the final value and whether the floor applied.
    """
    sigma = np.empty_like(residual, dtype=np.float64)
    by_site: dict[str, dict[str, Any]] = {}
    for site in dict.fromkeys(sites.tolist()):
        rows = sites == site
        terms = {
            "n_observations": int(rows.sum()),
            "mean_residual_squared": float(np.mean(residual[rows] ** 2)),
            "mean_prior_state_variance": float(np.mean(prior_state_variance[rows])),
            "mean_aggregation_variance": float(np.mean(aggregation_variance[rows])),
            "mean_mf_error_variance": float(np.mean(error_variance[rows])),
        }
        if choice.mode == "prior_residual_rms":
            raw_variance = terms["mean_residual_squared"]
        else:
            raw_variance = (terms["mean_residual_squared"] - terms["mean_prior_state_variance"]
                            - terms["mean_aggregation_variance"] - terms["mean_mf_error_variance"])
        raw = float(np.sqrt(max(raw_variance, 0.0)))
        value = max(raw, choice.floor)
        sigma[rows] = value
        by_site[site] = {**terms, "raw_variance": float(raw_variance), "raw_sigma": raw, "sigma": value,
                         "floor_applied": bool(choice.floor > raw)}
    return sigma, {"label": "empirical Bayes", **asdict(choice), "by_site": by_site}


__all__ = [
    "EMPIRICAL_BAYES_MODES", "EmpiricalBayesMismatch", "FixedMismatch", "empirical_bayes_mismatch",
    "observation_sites", "parse_fixed_model_mismatch", "per_site_mismatch",
]
