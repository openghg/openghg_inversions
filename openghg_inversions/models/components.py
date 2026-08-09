"""Reusable PyMC model graph helpers.

These helpers operate on the active PyMC model context and are designed to be
xarray-first. They return PyTensor/PyMC tensors and should not implement their
own coordinate sanitization policy; coordinate handling lives in
``openghg_inversions.models.coords``.

All component helpers operate inside an active PyMC model context.
``add_state_vector`` consumes an already resolved activity contract;
``add_state_linear_component`` performs design inspection and policy resolution
before constructing that graph.

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
from typing import Any, cast

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.inversion_inputs import make_freq_indicator
from openghg_inversions.models.coords import add_coords
from openghg_inversions.models.priors import parse_prior
from openghg_inversions.models.state_activity import (
    ResolvedStateActivity,
    StateActivity,
    active_prior_args,
    detect_zero_sensitivity,
    resolve_state_activity,
)
from openghg_inversions.sigma import SigmaAlignment


@dataclass
class LinearComponentResult:
    """Objects created by ``add_linear_component``."""

    data: TensorVariable
    latent: TensorVariable
    output: TensorVariable


@dataclass
class StateVectorResult:
    """Objects created by ``add_state_vector``.

    Attributes:
        latent: Effective sampled latent variable, or ``None`` when every
            state is fixed.
        state: Full ordered state vector, including fixed values.
        activity: Resolved state-activity contract in canonical state order.
    """

    latent: TensorVariable | None
    state: TensorVariable
    activity: ResolvedStateActivity


@dataclass
class StateLinearComponentResult:
    """Objects created by ``add_state_linear_component``.

    Attributes:
        data: Full sensitivity matrix registered with PyMC.
        latent: Effective active-state latent variable, or ``None`` when no
            states are active.
        state: Full ordered state vector. This is the ordinary user-facing
            prior when all states are active and a deterministic otherwise.
        output: Forward-model contribution from active and fixed states.
        activity: Resolved state-activity contract in canonical state order.
    """

    data: TensorVariable
    latent: TensorVariable | None
    state: TensorVariable
    output: TensorVariable
    activity: ResolvedStateActivity


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


def _with_legacy_all_active_coord(
    data: xr.DataArray,
    state_activity: StateActivity | None,
    *,
    output_dim: str,
) -> xr.DataArray:
    """Supply a positional state coordinate for the legacy all-active policy.

    Args:
        data: Candidate sensitivity matrix.
        state_activity: Optional activity policy.
        output_dim: Observation/output dimension name.

    Returns:
        ``data`` unchanged, or with positional labels for its sole state axis.
    """
    state_dims = [str(dim) for dim in data.dims if dim != output_dim]
    legacy_all_active = (
        state_activity is not None
        and not state_activity.prune_zero
        and not state_activity.fixed_groups
        and isinstance(state_activity.active, (bool, np.bool_))
        and bool(state_activity.active)
    )
    if legacy_all_active and len(state_dims) == 1 and state_dims[0] not in data.coords:
        state_dim = state_dims[0]
        return data.assign_coords({state_dim: np.arange(data.sizes[state_dim])})
    return data


def add_state_vector(
    activity: ResolvedStateActivity,
    /,
    prior_args: dict[str, Any],
    var_name: str,
) -> StateVectorResult:
    """Construct an active/fixed state graph from a resolved activity contract.

    When every state is active, this creates the same base prior graph as
    ``add_linear_component``. Partial activity creates an active-only prior and
    restores it into a full deterministic state vector. An all-fixed policy
    creates no random variable and exposes the fixed values as the full
    deterministic state.

    Args:
        activity: Resolved activity and state-coordinate contract. Linear
            design inspection must be completed before calling this helper.
        prior_args: Prior specification. Distribution parameters may be scalar,
            full-state arrays, or labelled state ``DataArray`` objects.
        var_name: Name of the full user-facing state vector.

    Returns:
        The effective latent, full state vector, and supplied activity.

    Raises:
        KeyError: If the prior specification omits a required parameter.
        TypeError: If the prior specification contains an unsupported value.
        ValueError: If state-valued prior parameters are invalid.

    Notes:
        This helper registers state variables and state coordinates, but it
        does not inspect or register a linear design and does not construct a
        forward-model output. The registered activity mask is immutable
        build-time metadata in semantic terms; changing it with ``pm.set_data``
        would not rebuild the latent state layout. Call this helper inside an
        active ``pm.Model`` context.
    """
    state_dim = activity.state_dim
    state_coord = activity.zero_sensitivity.coords[state_dim]
    add_coords(activity.zero_sensitivity.coords, model_dims=(state_dim,))
    parsed_prior_args = active_prior_args(prior_args, activity)

    if activity.n_active == activity.n_state:
        state = parse_prior(var_name, parsed_prior_args, dims=state_dim)
        return StateVectorResult(
            latent=get_model_latent(state, var_name),
            state=state,
            activity=activity,
        )

    add_model_data(
        activity.active.rename(f"{var_name}_is_active"),
        f"{var_name}_is_active",
    )
    fixed_value = add_model_data(
        activity.fixed_value.rename(f"{var_name}_fixed_value"),
        f"{var_name}_fixed_value",
    )
    active_indices = activity.active_indices
    latent: TensorVariable | None = None
    active_state: TensorVariable | None = None
    if activity.n_active:
        active_dim = f"{state_dim}_{var_name}_active"
        active_index = state_coord.to_index()[active_indices]
        if isinstance(active_index, pd.MultiIndex):
            # Keep tuple labels without re-registering the MultiIndex level
            # names already owned by the full state dimension.
            active_coord = np.empty(activity.n_active, dtype=object)
            active_coord[:] = active_index.tolist()
        else:
            active_coord = active_index.to_numpy()
        add_coords({active_dim: active_coord})
        active_state = parse_prior(
            f"{var_name}_active",
            parsed_prior_args,
            dims=active_dim,
        )
        latent = get_model_latent(active_state, f"{var_name}_active")

    full_state = fixed_value
    if active_state is not None:
        full_state = pt.set_subtensor(full_state[active_indices], active_state)
    state = pm.Deterministic(var_name, full_state, dims=state_dim)
    return StateVectorResult(latent=latent, state=state, activity=activity)


def add_state_linear_component(
    data: xr.DataArray,
    /,
    data_name: str,
    prior_args: dict,
    var_name: str,
    output_name: str,
    state_activity: StateActivity | None = None,
    output_dim: str = "nmeasure",
    compute_deterministic: bool = True,
) -> StateLinearComponentResult:
    """Add a linear component that samples only active labelled states.

    The public ``var_name`` is always a full ordered vector. When every state
    is active, it retains the ordinary prior graph produced by
    ``add_linear_component``. Otherwise, active states are sampled in
    ``{var_name}_active`` and restored with inactive fixed values into a full
    deterministic state. The forward contribution is always ``H @ state``.

    Args:
        data: Finite sensitivity matrix containing ``output_dim`` and exactly
            one other dimension with a unique labelled state coordinate. The
            explicit legacy all-active policy may synthesize positional labels.
        data_name: Name used when registering the full matrix as ``pm.Data``.
        prior_args: Prior specification. Distribution parameters may be scalar,
            full-state arrays, or labelled state ``DataArray`` objects.
        var_name: Name of the full deterministic state vector.
        output_name: Name for the aligned forward-model contribution.
        state_activity: Optional active/fixed policy. The default prunes only
            exactly-zero sensitivity columns and fixes them to one.
        output_dim: Observation/output dimension name.
        compute_deterministic: Whether to wrap the output in a named
            ``pm.Deterministic``.

    Returns:
        Registered data, optional active latent, full state, output, and the
        resolved activity contract.

    Raises:
        ValueError: If the sensitivity layout, state policy, labels, fixed
            values, or state-valued prior parameters are invalid.

    Notes:
        This helper mutates the active ``pm.Model`` by registering coordinates,
        the full design, state variables, and optionally the named output.
    """
    output_dim = str(output_dim)
    data = _with_legacy_all_active_coord(
        data.transpose(output_dim, ...),
        state_activity,
        output_dim=output_dim,
    )
    activity = resolve_state_activity(
        detect_zero_sensitivity(data, output_dim=output_dim),
        state_activity,
    )
    h_full = add_model_data(data, data_name)
    vector = add_state_vector(
        activity,
        prior_args=prior_args,
        var_name=var_name,
    )

    output = pt.dot(h_full, vector.state)
    if compute_deterministic:
        output = pm.Deterministic(output_name, output, dims=output_dim)

    return StateLinearComponentResult(
        data=h_full,
        latent=vector.latent,
        state=vector.state,
        output=cast(TensorVariable, output),
        activity=vector.activity,
    )


def add_sigma_component(
    alignment: SigmaAlignment,
    /,
    prior_args: dict,
    compute_deterministic: bool = False,
) -> TensorVariable:
    """Register a latent sigma component and align it to observations.

    Args:
        alignment: Backend-neutral site and period alignment for the component.
        prior_args: Prior specification for the sigma random variable.
        compute_deterministic: Whether to register the aligned sigma term as a
            deterministic variable.

    Returns:
        The observation-aligned sigma tensor or deterministic variable.
    """
    site_data_var = add_model_data(alignment.site_index)
    period_data_var = add_model_data(alignment.period_index)

    add_coords(
        {
            "nsigma_site": np.arange(alignment.nsite),
            "nsigma_time": np.arange(alignment.nperiod),
        }
    )

    sigma = parse_prior("sigma", prior_args, dims=("nsigma_site", "nsigma_time"))
    aligned = sigma[site_data_var, period_data_var]
    if compute_deterministic:
        return pm.Deterministic("sigma_aligned", aligned, dims="nmeasure")
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
    per_site: bool = True,
) -> TensorVariable:
    """Add a global, site-only, or site-by-period offset component.

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
        per_site: Whether to create site-specific terms. If false, create one
            global scalar latent offset and broadcast it over observations.

    Returns:
        The aligned offset deterministic variable.
    """
    output_dim = str(output_dim)
    site_indicator = site_indicator.rename("site_indicator").transpose(output_dim)
    add_model_data(site_indicator, "site_indicator")
    if not per_site:
        if offset_freq_indicator is not None or offset_freq is not None:
            raise ValueError("Global offsets do not accept an offset frequency.")
        if drop_first:
            raise ValueError("Global offsets do not support `drop_first=True`.")
        latent = parse_prior(var_name, prior_args)
        aligned = pt.broadcast_to(latent, (site_indicator.sizes[output_dim],))
        return pm.Deterministic(output_name, aligned, dims=output_dim)

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
    sigma_alignment: SigmaAlignment,
    offset: TensorVariable | None = None,
    power: dict | float = 1.99,
    pollution_events_from_obs: bool = False,
    no_model_error: bool = False,
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
        sigma_alignment: Backend-neutral site and period alignment for sigma.
        offset: Optional aligned offset term.
        power: Scalar or prior specification controlling pollution-event
            scaling.
        pollution_events_from_obs: Whether to derive pollution events from the
            observations instead of ``mu``.
        no_model_error: Whether to bypass the model-error term.
        output_dim: Observation/output dimension name.

    Returns:
        The ``epsilon`` deterministic variable used by the observation model.
    """
    y_data = add_model_data(data["mf"].transpose(output_dim), "Y")
    error_data = add_model_data(data["mf_error"].transpose(output_dim), "error")
    min_error_data = add_model_data(data["min_error"].transpose(output_dim), "min_error")

    sigma = add_sigma_component(
        sigma_alignment,
        prior_args=sigprior,
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
        eps = cast(Any, pt.maximum)(pt.abs(error_data), small_amount)
    else:
        power0 = parse_prior("power", power) if isinstance(power, dict) else power
        eps = cast(Any, pt.maximum)(
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
