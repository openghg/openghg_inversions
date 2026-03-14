"""Reusable PyMC model graph helpers.

These helpers operate on the active PyMC model context and are designed to be
xarray-first. They return PyTensor/PyMC tensors and should not implement their
own coordinate sanitization policy; coordinate handling lives in
``openghg_inversions.models.coords``.

Naming conventions:
- ``data_name``: name for registered ``pm.Data``
- ``var_name``: name for the latent random variable
- ``output_name``: name for the aligned deterministic output
- plain ``name`` is reserved for helpers that truly create only one semantic
  object or where a base name is the clearest API

Frequency indicators may be supplied explicitly or derived from observation
coordinates using shared helper logic based on
``openghg_inversions.inversion_inputs.make_freq_indicator``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor import TensorVariable

from openghg_inversions.inversion_inputs import make_freq_indicator
from openghg_inversions.models.coords import add_coords
from openghg_inversions.models.priors import parse_prior


def _extract_time_coord(data: xr.DataArray, output_dim: str) -> xr.DataArray | None:
    if "time" in data.coords:
        coord = data.coords["time"]
        if output_dim in coord.dims:
            return coord

    if output_dim in data.indexes:
        index = data.indexes[output_dim]
        if isinstance(index, pd.MultiIndex) and "time" in index.names:
            return xr.DataArray(
                index.get_level_values("time").to_numpy(),
                dims=(output_dim,),
                coords={output_dim: data.coords[output_dim]},
                name="time",
            )

    return None


def _resolve_freq_indicator(
    *,
    explicit_indicator: xr.DataArray | np.ndarray | None,
    freq: str | None,
    data: xr.DataArray,
    output_dim: str,
    fallback_name: str,
) -> xr.DataArray | None:
    if explicit_indicator is not None:
        if isinstance(explicit_indicator, xr.DataArray):
            return explicit_indicator.rename(explicit_indicator.name or fallback_name)
        return xr.DataArray(
            np.asarray(explicit_indicator, dtype=int),
            dims=(output_dim,),
            coords={output_dim: data.coords[output_dim]},
            name=fallback_name,
        )

    if freq is None:
        return None

    time_coord = _extract_time_coord(data, output_dim=output_dim)
    if time_coord is None:
        raise ValueError(f"Cannot derive frequency indicator for {fallback_name!r}: no time coordinate found.")

    # TODO: once component-side derivation is relied on consistently, some
    # explicit frequency-indicator plumbing in make_inv_inputs(...) may be removable.
    return make_freq_indicator(time_coord, freq).rename(fallback_name)


def add_model_data(data: xr.DataArray, name: str | None = None) -> TensorVariable:
    """Add labelled xarray data to the active PyMC model."""
    name = name or data.name
    if name is None:
        raise ValueError("Data must have a name if a name is not provided.")

    model = pm.modelcontext(None)
    if name in model:
        return model[name]

    dim_coords = {dim: data.coords[dim] for dim in data.dims if dim in data.coords}
    add_coords(dim_coords)
    return pm.Data(name, data.values, dims=data.dims)


def add_linear_component(
    data: xr.DataArray,
    /,
    data_name: str,
    prior_args: dict,
    var_name: str,
    output_name: str,
    output_dim: str = "nmeasure",
    compute_deterministic: bool = True,
) -> TensorVariable:
    """Add a linear latent component and its aligned forward-model contribution."""
    data = data.transpose(output_dim, ...)
    h = add_model_data(data, data_name)
    input_dims = tuple(dim for dim in data.dims if dim != output_dim)
    latent = parse_prior(var_name, prior_args, dims=input_dims)
    output = pt.dot(h, latent)
    if compute_deterministic:
        return pm.Deterministic(output_name, output, dims=output_dim)
    return output


def add_sigma_component(
    site_indicator: xr.DataArray,
    /,
    prior_args: dict,
    sigma_freq_index: xr.DataArray | None = None,
    sigma_freq: str | None = None,
    var_name: str = "sigma",
    output_name: str | None = None,
    per_site: bool = True,
    output_dim: str = "nmeasure",
    compute_deterministic: bool = False,
) -> TensorVariable:
    """Add inferpymc-compatible sigma terms and align them to observations."""
    site_indicator = site_indicator.rename("site_indicator").transpose(output_dim)
    freq_index = _resolve_freq_indicator(
        explicit_indicator=sigma_freq_index,
        freq=sigma_freq,
        data=site_indicator,
        output_dim=output_dim,
        fallback_name="sigma_freq_index" if var_name == "sigma" else f"{var_name}_freq_indicator",
    )
    if freq_index is None:
        raise ValueError("Sigma frequency information must be provided via `sigma_freq_index` or `sigma_freq`.")

    site_data = site_indicator if per_site else xr.zeros_like(site_indicator)
    add_model_data(site_data, "site_indicator")
    freq_data = add_model_data(freq_index.transpose(output_dim), freq_index.name)

    nsigma_site = int(site_data.max().item()) + 1 if per_site else 1
    nsigma_time = int(freq_index.max().item()) + 1 if freq_index.size else 0
    add_coords(
        {
            "nsigma_site": np.arange(nsigma_site),
            "nsigma_time": np.arange(nsigma_time),
        }
    )

    sigma = parse_prior(var_name, prior_args, dims=("nsigma_site", "nsigma_time"))
    aligned = sigma[site_data.values.astype(int), freq_data]
    if compute_deterministic:
        deterministic_name = output_name or f"{var_name}_aligned"
        return pm.Deterministic(deterministic_name, aligned, dims=output_dim)
    return aligned


def add_offset_component(
    site_indicator: xr.DataArray,
    /,
    prior_args: dict,
    offset_freq_indicator: xr.DataArray | np.ndarray | None = None,
    offset_freq: str | None = None,
    var_name: str = "offset_latent",
    output_name: str = "offset",
    output_dim: str = "nmeasure",
    drop_first: bool = False,
) -> TensorVariable:
    """Add a site-only or site-by-period offset component."""
    site_indicator = site_indicator.rename("site_indicator").transpose(output_dim)
    indicator = _resolve_freq_indicator(
        explicit_indicator=offset_freq_indicator,
        freq=offset_freq,
        data=site_indicator,
        output_dim=output_dim,
        fallback_name="offset_freq_indicator",
    )

    site_codes = np.asarray(site_indicator.values, dtype=int)
    site_matrix = pd.get_dummies(site_codes, drop_first=drop_first, dtype=int).values

    if indicator is not None:
        period_codes = np.asarray(indicator.values, dtype=int)
        period_matrix = pd.get_dummies(period_codes, dtype=int).values
        design_matrix = (
            site_matrix[:, :, None] * period_matrix[:, None, :]
        ).reshape(site_matrix.shape[0], -1)
    else:
        design_matrix = site_matrix

    latent = parse_prior(var_name, prior_args, shape=design_matrix.shape[1])
    return pm.Deterministic(output_name, pt.dot(np.asarray(design_matrix), latent), dims=output_dim)


def add_inferpymc_likelihood_component(
    data: xr.Dataset,
    /,
    mu: TensorVariable,
    mu_bc: TensorVariable | None,
    sigprior: dict,
    offset: TensorVariable | None = None,
    power: dict | float = 1.99,
    pollution_events_from_obs: bool = False,
    no_model_error: bool = False,
    sigma_per_site: bool = True,
    output_dim: str = "nmeasure",
) -> TensorVariable:
    """Add the inferpymc observation model.

    ``mu`` is the non-baseline forward-model contribution. ``mu_bc`` is the
    baseline contribution, usually ``H_bc @ bc``, plus offset if applicable.
    """
    y_data = add_model_data(data["mf"].transpose(output_dim), "Y")
    error_data = add_model_data(data["mf_error"].transpose(output_dim), "error")
    min_error_data = add_model_data(data["min_error"].transpose(output_dim), "min_error")

    sigma = add_sigma_component(
        data["site_indicator"].transpose(output_dim),
        prior_args=sigprior,
        sigma_freq_index=data["sigma_freq_index"].transpose(output_dim),
        var_name="sigma",
        per_site=sigma_per_site,
        output_dim=output_dim,
    )

    if pollution_events_from_obs is True:
        if mu_bc is not None:
            pollution_event = pt.abs(y_data - mu_bc)
        else:
            pollution_event = pt.abs(y_data) + 1e-6 * pt.mean(y_data)
    else:
        pollution_event = pt.abs(mu)

    pollution_event_scaled_error = pollution_event * sigma

    if no_model_error is True:
        mean_obs = np.nanmean(data["mf"].values)
        small_amount = 1e-12 * mean_obs
        eps = pt.maximum(pt.abs(error_data), small_amount)
    else:
        power0 = parse_prior("power", power) if isinstance(power, dict) else power
        eps = pt.maximum(
            pt.sqrt(error_data**2 + pt.pow(pollution_event_scaled_error, power0)),
            min_error_data,
        )

    total_mu = mu
    if mu_bc is not None:
        total_mu = total_mu + mu_bc
    if offset is not None:
        total_mu = total_mu + offset

    epsilon = pm.Deterministic("epsilon", eps, dims=output_dim)
    pm.Normal("y", mu=total_mu, sigma=epsilon, observed=y_data, dims=output_dim)
    return epsilon
