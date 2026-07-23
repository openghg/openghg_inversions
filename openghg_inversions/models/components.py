"""Reusable PyMC model graph helpers.

These helpers operate on the active PyMC model context and are designed to be
xarray-first. They return PyTensor/PyMC tensors and should not implement their
own coordinate sanitization policy; coordinate handling lives in
``openghg_inversions.models.coords``.

All component helpers operate inside an active PyMC model context.

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

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.component_data import SigmaComponentData, prepare_sigma_component_data
from openghg_inversions.inversion_inputs import make_freq_indicator
from openghg_inversions.models.coords import add_coords
from openghg_inversions.models.priors import parse_prior


@dataclass
class LinearComponentResult:
    """Objects created by ``add_linear_component``."""

    data: TensorVariable
    latent: TensorVariable
    output: TensorVariable


def get_model_latent(variable: TensorVariable, base_name: str) -> TensorVariable:
    """Return the effective latent variable for a named model component.

    Args:
        variable: User-facing variable returned by ``parse_prior``.
        base_name: Base model variable name used to look up a reparameterized
            latent variable.

    Returns:
        The reparameterized latent variable ``{base_name}_latent`` when it is
        present on the active model, otherwise ``variable``.
    """
    model = pm.modelcontext(None)
    resolved = resolve_model_variable(model, base_name)
    if resolved is not None:
        return resolved
    return variable


def resolve_model_variable(model: pm.Model, base_name: str) -> TensorVariable | None:
    """Return a named model variable, preferring the reparameterised latent form.

    Args:
        model: PyMC model to inspect.
        base_name: Base variable name to resolve.

    Returns:
        The reparameterised latent variable ``{base_name}_latent`` when it is
        present on ``model``, otherwise the user-facing variable named
        ``base_name``. Returns ``None`` if neither variable exists.
    """
    latent_name = f"{base_name}_latent"
    if latent_name in model.named_vars:
        return cast(TensorVariable, model.named_vars[latent_name])
    if base_name in model.named_vars:
        return cast(TensorVariable, model.named_vars[base_name])
    return None


def _extract_time_coord(data: xr.DataArray, output_dim: str) -> xr.DataArray | None:
    """Extract a time coordinate aligned to the observation dimension."""
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
    """Return an explicit or derived observation-aligned frequency indicator."""
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
        raise ValueError(
            f"Cannot derive frequency indicator for {fallback_name!r}: no time coordinate found."
        )

    # TODO: thread sigma_freq/sigma_per_site explicitly through inferpymc model
    # building so sigma components can derive their own indicator, then remove
    # sigma_freq_index from make_inv_inputs(...).
    return make_freq_indicator(time_coord, freq).rename(fallback_name)


def add_model_data(data: xr.DataArray, name: str | None = None) -> TensorVariable:
    """Add labelled xarray data to the active PyMC model.

    Args:
        data: Xarray data to register as ``pm.Data``.
        name: Optional PyMC variable name. If omitted, ``data.name`` is used.

    Returns:
        The registered ``pm.Data`` tensor for ``data``.

    Raises:
        ValueError: If no name can be determined for the data variable.
    """
    name = name or (str(data.name) if data.name is not None else None)
    if name is None:
        raise ValueError("Data must have a name if a name is not provided.")

    model = pm.modelcontext(None)
    if name in model:
        return model[name]

    dims = tuple(str(dim) for dim in data.dims)
    add_coords(data.coords, model_dims=dims)
    return cast(TensorVariable, pm.Data(name, data.values, dims=dims))


def add_linear_component(
    data: xr.DataArray,
    /,
    data_name: str,
    prior_args: dict,
    var_name: str,
    output_name: str,
    output_dim: str = "nmeasure",
    compute_deterministic: bool = True,
) -> LinearComponentResult:
    """Add a linear latent component and its aligned forward-model contribution.

    Args:
        data: Sensitivity matrix or other linear data term.
        data_name: Name used when registering the data as ``pm.Data``.
        prior_args: Prior specification for the latent random variable.
        var_name: Name for the latent random variable.
        output_name: Name for the aligned deterministic output.
        output_dim: Observation/output dimension name.
        compute_deterministic: Whether to wrap the aligned output in
            ``pm.Deterministic``.

    Returns:
        A ``LinearComponentResult`` containing the registered data tensor, the
        effective latent variable, and the aligned output tensor.
    """
    output_dim = str(output_dim)
    data = data.transpose(output_dim, ...)
    h = add_model_data(data, data_name)
    input_dims = tuple(str(dim) for dim in data.dims if dim != output_dim)
    user_facing = parse_prior(var_name, prior_args, dims=input_dims)
    latent = get_model_latent(user_facing, var_name)
    output = pt.dot(h, user_facing)
    if compute_deterministic:
        output = pm.Deterministic(output_name, output, dims=output_dim)
    return LinearComponentResult(data=h, latent=latent, output=output)


def add_sigma_component(
    site_indicator: xr.DataArray | SigmaComponentData,
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
    """Add inferpymc-compatible sigma terms and align them to observations.

    Args:
        site_indicator: Observation-aligned site indicator or sigma data
            prepared by :func:`prepare_sigma_component_data`.
        prior_args: Prior specification for the sigma random variable.
        sigma_freq_index: Optional explicit observation-aligned frequency
            indicator.
        sigma_freq: Optional frequency string used to derive an indicator when
            ``sigma_freq_index`` is not provided.
        var_name: Name for the latent sigma random variable.
        output_name: Optional name for an observation-aligned deterministic
            output.
        per_site: Whether sigma varies by site.
        output_dim: Observation/output dimension name.
        compute_deterministic: Whether to register the aligned sigma term as a
            deterministic variable.

    Returns:
        The observation-aligned sigma tensor or deterministic variable.

    Raises:
        ValueError: If the supplied indexes are invalid or frequency
            information cannot be derived.
    """
    output_dim = str(output_dim)
    if isinstance(site_indicator, SigmaComponentData):
        component_data = site_indicator
        if component_data.output_dim != output_dim:
            raise ValueError(
                f"Prepared sigma data use output dimension {component_data.output_dim!r}, not {output_dim!r}."
            )
    else:
        component_data = prepare_sigma_component_data(
            site_indicator,
            sigma_freq_index=sigma_freq_index,
            sigma_freq=sigma_freq,
            per_site=per_site,
            output_dim=output_dim,
            var_name=var_name,
        )

    site_data_var = add_model_data(component_data.site_index, component_data.site_index_name)
    freq_data = add_model_data(component_data.freq_index, component_data.freq_index_name)

    add_coords(
        {
            "nsigma_site": np.arange(component_data.nsigma_site),
            "nsigma_time": np.arange(component_data.nsigma_time),
        }
    )

    sigma = parse_prior(var_name, prior_args, dims=("nsigma_site", "nsigma_time"))
    aligned = sigma[site_data_var, freq_data]
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
    """Add a site-only or site-by-period offset component.

    Args:
        site_indicator: Observation-aligned site indicator.
        prior_args: Prior specification for the offset latent variable.
        offset_freq_indicator: Optional explicit observation-aligned offset
            frequency indicator.
        offset_freq: Optional frequency string used to derive an indicator when
            ``offset_freq_indicator`` is not provided.
        var_name: Name for the latent offset variable.
        output_name: Name for the aligned deterministic offset output.
        output_dim: Observation/output dimension name.
        drop_first: Whether to omit the first site indicator column.

    Returns:
        The aligned offset deterministic variable.
    """
    output_dim = str(output_dim)
    site_indicator = site_indicator.rename("site_indicator").transpose(output_dim)
    add_model_data(site_indicator, "site_indicator")
    indicator = _resolve_freq_indicator(
        explicit_indicator=offset_freq_indicator,
        freq=offset_freq,
        data=site_indicator,
        output_dim=output_dim,
        fallback_name="offset_freq_indicator",
    )
    if indicator is not None:
        add_model_data(indicator.transpose(output_dim), str(indicator.name))

    site_codes = np.asarray(site_indicator.values, dtype=int)
    site_matrix = pd.get_dummies(site_codes, drop_first=drop_first, dtype=int).values

    if indicator is not None:
        period_codes = np.asarray(indicator.values, dtype=int)
        period_matrix = pd.get_dummies(period_codes, dtype=int).values
        design_matrix = (site_matrix[:, :, None] * period_matrix[:, None, :]).reshape(
            site_matrix.shape[0], -1
        )
    else:
        design_matrix = site_matrix

    design_name = f"{output_name}_design"
    design_data = add_model_data(
        xr.DataArray(
            design_matrix,
            dims=(output_dim, "noffset_term"),
            coords={
                output_dim: site_indicator.coords[output_dim],
                "noffset_term": np.arange(design_matrix.shape[1]),
            },
            name=design_name,
        ),
        design_name,
    )
    latent = parse_prior(var_name, prior_args, shape=design_matrix.shape[1])
    return pm.Deterministic(output_name, pt.dot(design_data, latent), dims=output_dim)


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
    sigma_freq: str | None = None,
    output_dim: str = "nmeasure",
) -> TensorVariable:
    """Add the inferpymc observation model.

    ``mu`` is the non-baseline forward-model contribution. ``mu_bc`` is the
    baseline contribution, usually ``H_bc @ bc``, plus offset if applicable.

    Args:
        data: Canonical inferpymc input dataset.
        mu: Non-baseline forward-model contribution.
        mu_bc: Baseline contribution, if present.
        sigprior: Prior specification for sigma.
        offset: Optional aligned offset term.
        power: Scalar or prior specification controlling pollution-event
            scaling.
        pollution_events_from_obs: Whether to derive pollution events from the
            observations instead of ``mu``.
        no_model_error: Whether to bypass the model-error term.
        sigma_per_site: Whether sigma varies by site.
        sigma_freq: Optional frequency used to derive sigma-period indexes when
            they are not already present in ``data``.
        output_dim: Observation/output dimension name.

    Returns:
        The ``epsilon`` deterministic variable used by the observation model.
    """
    y_data = add_model_data(data["mf"].transpose(output_dim), "Y")
    error_data = add_model_data(data["mf_error"].transpose(output_dim), "error")
    min_error_data = add_model_data(data["min_error"].transpose(output_dim), "min_error")

    sigma_component_data = prepare_sigma_component_data(
        data["site_indicator"].transpose(output_dim),
        sigma_freq_index=(
            data["sigma_freq_index"].transpose(output_dim) if "sigma_freq_index" in data else None
        ),
        sigma_freq=sigma_freq,
        per_site=sigma_per_site,
        output_dim=output_dim,
        var_name="sigma",
    )
    sigma = add_sigma_component(
        sigma_component_data,
        prior_args=sigprior,
        var_name="sigma",
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
        small_amount = pm.floatX(1e-12 * mean_obs)
        eps = pt.maximum(pt.abs(error_data), small_amount)
    else:
        power0 = parse_prior("power", power) if isinstance(power, dict) else power
        eps = pt.maximum(
            pt.sqrt(error_data**2 + pt.pow(pollution_event_scaled_error, power0)),
            min_error_data,
        )

    # TODO: this calculation should probably happen separately
    # e.g. using a add_linear_component_sum function.
    total_mu = mu
    if mu_bc is not None:
        total_mu = total_mu + mu_bc
    if offset is not None:
        total_mu = total_mu + offset

    epsilon = pm.Deterministic("epsilon", eps, dims=output_dim)
    pm.Normal("y", mu=total_mu, sigma=epsilon, observed=y_data, dims=output_dim)
    return epsilon
