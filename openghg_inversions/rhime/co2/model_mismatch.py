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
  floor = 0.1}``, estimated per site from the prior residuals. ``floor`` is
  required; a declared mismatch or the median sigma of a previous NUTS run is a
  sensible value.

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
to set :math:`\sigma` and to fit. Known limitations of the moment mode:

- When the prior equals the truth (for example the UOB control), the prior
  residual is pure observation noise plus aggregation error. Its expectation
  lacks the whole prior state spread that the moment mode subtracts, so the
  estimate is biased low by the mean of :math:`\mathrm{diag}(H B H^T)` (and
  the boundary and offset spread), not merely noisy, and usually returns the
  floor. An empirical-Bayes :math:`\sigma` there is no evidence about model
  error.
- When the prior-predictive variance sits in a few modes shared across
  observations, a per-site mean of :math:`r^2` has only a handful of effective
  degrees of freedom, so the subtraction is dominated by sampling error. In
  the E-r07 configuration the four shared boundary scalings contribute about
  271 ppm² of the average subtracted variance and the retained flux about
  4.7 ppm², about 98 % from the boundary.

A per-site type-II maximum-likelihood estimate, maximising
:math:`\log N(y; \bar y, K + \mathrm{diag}(e^2 + \sigma_s^2))` with the full
correlated prior-predictive covariance :math:`K`, is the proposed remedy; it is
not implemented yet.

The floor is required. When every site sits at its floor and the resulting
mean :math:`e^2 + \sigma^2` is below ``1e-6`` times the mean aggregation
variance, the estimate is refused unless ``allow_zero = true``: such a
mismatch would make the posterior overconfident.
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


@dataclass(frozen=True, kw_only=True)
class EmpiricalBayesMismatch:
    """Per-site empirical-Bayes model mismatch, estimated from prior residuals.

    Attributes:
        floor: Required minimum per-site standard deviation in observation
            units, for example a declared mismatch or a NUTS median.
        mode: ``"prior_residual_moment"`` (default) or ``"prior_residual_rms"``;
            see the module documentation.
        allow_zero: Accept a result where every site is at its floor and the
            total independent variance is negligible against the aggregation
            variance.
    """

    floor: float
    mode: EmpiricalBayesMode = "prior_residual_moment"
    allow_zero: bool = False

    def __post_init__(self) -> None:
        """Validate the mode and the floor."""
        if self.mode not in EMPIRICAL_BAYES_MODES:
            raise ValueError(f"empirical_bayes must be one of {list(EMPIRICAL_BAYES_MODES)!r}, not {self.mode!r}.")
        if isinstance(self.floor, bool) or not isfinite(float(self.floor)) or float(self.floor) < 0:
            raise ValueError("The empirical-Bayes floor must be finite and non-negative.")
        if not isinstance(self.allow_zero, bool):
            raise TypeError("allow_zero must be a boolean.")


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
        if "floor" not in table:
            raise ValueError(
                f"{path}.floor is required with empirical_bayes: set a minimum sigma in observation units, "
                "for example the declared mismatch or the median sigma of a previous NUTS run."
            )
        floor = number(table.pop("floor"), f"{path}.floor")
        allow_zero = table.pop("allow_zero", False)
        if not isinstance(allow_zero, bool):
            raise TypeError(f"{path}.allow_zero must be a boolean.")
        if table:
            raise ValueError(f"Unknown empirical-Bayes option(s): {', '.join(f'{path}.{name}' for name in table)}.")
        return EmpiricalBayesMismatch(mode=mode, floor=floor, allow_zero=allow_zero)  # type: ignore[arg-type]
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

    Raises:
        ValueError: If every site is at its floor and the mean of
            ``error_variance + sigma**2`` is below ``1e-6`` times the mean
            ``aggregation_variance``, unless ``choice.allow_zero``.
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
                         "floor_applied": bool(choice.floor > raw), "at_floor": bool(raw <= choice.floor)}
    independent = float(np.mean(error_variance + sigma**2))
    negligible = independent < 1e-6 * float(np.mean(aggregation_variance))
    if negligible and all(item["at_floor"] for item in by_site.values()) and not choice.allow_zero:
        raise ValueError(
            f"Empirical-Bayes sigma ({choice.mode}) is at the floor {choice.floor:g} at every site and the mean "
            f"independent variance {independent:.3g} is negligible against the aggregation variance, which "
            "would make the posterior overconfident. Set a larger floor (for example the declared mismatch or a "
            "NUTS median sigma), or allow_zero = true to accept it."
        )
    return sigma, {"label": "empirical Bayes", **asdict(choice), "by_site": by_site,
                   "mean_independent_variance": independent}


__all__ = [
    "EMPIRICAL_BAYES_MODES", "EmpiricalBayesMismatch", "FixedMismatch", "empirical_bayes_mismatch",
    "observation_sites", "parse_fixed_model_mismatch", "per_site_mismatch",
]
