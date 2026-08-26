"""Reusable PyMC model graph helpers.

These helpers operate on the active PyMC model context and are designed to be
xarray-first. They return explicit component results or PyTensor/PyMC tensors
and should not implement their own coordinate sanitization policy; coordinate
handling lives in ``openghg_inversions.models.coords``.

All component helpers operate inside an active PyMC model context.
``add_state_vector`` consumes an already resolved activity contract;
``add_linear_component`` consumes a sensitivity matrix inspected by
``prepare_linear_sensitivity`` before constructing that graph.

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

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.inversion_inputs import make_freq_indicator, make_site_indicator
from openghg_inversions.models.coords import add_coords
from openghg_inversions.models.priors import parse_prior
from openghg_inversions.models.state_activity import (
    PreparedLinearSensitivity,
    ResolvedStateActivity,
    StateActivity,
    active_prior_args,
    resolve_state_activity,
)
from openghg_inversions.sigma import SigmaAlignment


@dataclass
class LinearComponentResult:
    """Objects created by :func:`add_linear_component`."""

    data: TensorVariable
    latent: TensorVariable | None
    state: TensorVariable
    output: TensorVariable
    activity: ResolvedStateActivity


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
class CorrelatedStateResult:
    """Objects created by ``add_correlated_lognormal_state``.

    Attributes:
        latent: Standard-normal whitened state used by the sampler.
        state: Positive user-facing state with the requested arithmetic
            LogNormal moments.
        prior: Validated backend-neutral moment contract used to build the
            graph.
    """

    latent: TensorVariable
    state: TensorVariable
    prior: CorrelatedLognormalPrior


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

    materialized = data.compute()
    dims = tuple(str(dim) for dim in materialized.dims)
    add_coords(materialized.coords, model_dims=dims)
    return cast(TensorVariable, pm.Data(name, materialized.values, dims=dims))


def add_linear_component(
    prepared: PreparedLinearSensitivity,
    /,
    data_name: str,
    prior_args: dict,
    var_name: str,
    output_name: str,
    state_activity: StateActivity | None = None,
    output_dim: str = "nmeasure",
    compute_deterministic: bool = True,
) -> LinearComponentResult:
    """Add one independent labelled linear component.

    Args:
        prepared: Retained sensitivity and full-state mapping produced by
            :func:`prepare_linear_sensitivity`.
        data_name: Name used when registering the data as ``pm.Data``.
        prior_args: Prior specification for the latent random variable.
        var_name: Name for the latent random variable.
        output_name: Name for the aligned deterministic output.
        output_dim: Observation/output dimension name.
        compute_deterministic: Whether to wrap the aligned output in
            ``pm.Deterministic``.

        state_activity: Optional active/fixed policy over the full scientific
            state. ``None`` samples every retained column.

    Returns:
        The registered sensitivity, effective latent, full state, aligned forward
        contribution, and resolved activity.
    """
    output_dim = str(output_dim)
    if output_dim != prepared.output_dim:
        raise ValueError(
            f"Prepared linear sensitivity owns output dimension {prepared.output_dim!r}, "
            f"not {output_dim!r}."
        )
    activity = resolve_state_activity(prepared.removed, state_activity)
    vector = add_state_vector(activity, prior_args=prior_args, var_name=var_name)
    output = apply_linear_sensitivity(
        prepared,
        vector.state,
        data_name=data_name,
        output_name=output_name,
        compute_deterministic=compute_deterministic,
    )
    data = cast(TensorVariable, pm.modelcontext(None)[data_name])
    return LinearComponentResult(
        data=data,
        latent=vector.latent,
        state=vector.state,
        output=output,
        activity=vector.activity,
    )


def apply_linear_sensitivity(
    prepared: PreparedLinearSensitivity,
    state: TensorVariable,
    /,
    *,
    data_name: str,
    output_name: str,
    compute_deterministic: bool = True,
) -> TensorVariable:
    """Apply a prepared sensitivity to an already-built full state vector."""
    h = add_model_data(prepared.sensitivity, data_name)
    output = pt.dot(h, state[prepared.retained_indices])
    if compute_deterministic:
        output = pm.Deterministic(output_name, output, dims=prepared.output_dim)
    return cast(TensorVariable, output)


def add_linked_linear_component(
    prepared: PreparedLinearSensitivity,
    linked_state: TensorVariable,
    /,
    *,
    data_name: str,
    output_name: str,
) -> TensorVariable:
    """Apply a prepared sensitivity to an already constructed linked state.

    Args:
        prepared: Sensitivity prepared by :func:`prepare_linear_sensitivity`.
            It owns retained-state selection and the output dimension.
        linked_state: Existing full state expression in the state order owned
            by ``prepared``. The caller owns any ratio or other transformation
            used to construct this expression.
        data_name: Name used to register the prepared sensitivity as
            ``pm.Data``.
        output_name: Name for the output-dimension-aligned deterministic.

    Returns:
        The linked linear signal registered as ``output_name``.

    Notes:
        This helper creates neither a state nor a multiplier. It only registers
        the prepared sensitivity and owns the resulting deterministic output.
    """
    return apply_linear_sensitivity(
        prepared,
        linked_state,
        data_name=data_name,
        output_name=output_name,
    )


def add_coherent_affine_component(
    fixed_contribution: xr.DataArray,
    linear_signal: TensorVariable,
    /,
    *,
    output_name: str,
) -> TensorVariable:
    """Add a labelled fixed contribution to an already composed linear signal.

    Args:
        fixed_contribution: Labelled affine intercept
            ``mu_prior - H_alpha @ m_alpha``, not the full prior-forward mean.
            Its name owns the registered ``pm.Data`` name and its coordinates
            own the output axis labels.
        linear_signal: Existing linear signal composed by the calling model
            recipe.
        output_name: Name for the affine deterministic output.

    Returns:
        The deterministic sum of the registered fixed contribution and linear
        signal.

    Notes:
        This component uses the equivalent coherent-reduction identity
        ``mu = mu_prior + H_alpha @ (x - m_alpha) =``
        ``(mu_prior - H_alpha @ m_alpha) + H_alpha @ x``.

        This helper does not construct states, sensitivities, ratios, or
        channel signals. The calling recipe owns those scientific choices.
    """
    fixed = add_model_data(fixed_contribution)
    return pm.Deterministic(output_name, fixed + linear_signal, dims=fixed_contribution.dims)


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
        does not inspect or register a sensitivity matrix and does not construct a
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


def add_correlated_lognormal_state(
    prior: CorrelatedLognormalPrior,
    /,
    *,
    var_name: str,
) -> CorrelatedStateResult:
    """Add a whitened correlated LogNormal state to the active PyMC model.

    Args:
        prior: Validated labelled arithmetic and latent moment contract.
        var_name: Name of the positive user-facing state. The whitened standard
            normal is named ``{var_name}_latent``.

    Returns:
        The whitened latent, positive state, and supplied prior contract.

    Raises:
        ValueError: If the arithmetic mean, latent moments, Cholesky diagonal,
            or exponentiated central state is not finite and positive where
            required in PyMC's configured floating-point dtype.

    Notes:
        This function must run in an active ``pm.Model`` context. After backend
        dtype validation completes, it mutates that model by registering the
        length-``p`` state coordinate, ``{var_name}_latent`` random variable,
        and length-``p`` ``{var_name}`` deterministic state.

        ``prior`` should contain reduced arithmetic moments produced together
        with the matching forward operator and Gaussian unresolved-error term.
        The coherent covariance, transformed-forward-model, and
        aggregation-error identities are exact only for a jointly Gaussian
        state. Reusing those first two moments with a LogNormal retained state
        and Gaussian unresolved contribution is a moment-matched closure, not
        exact LogNormal marginalization. Known-exact state fixing is handled
        separately by ``StateActivity`` in the state-linear component builders.
    """
    state_dim = prior.state_dim
    mean = prior.mean
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        backend_mean = np.asarray(pm.floatX(np.asarray(mean.values)))
        backend_latent_mean = np.asarray(pm.floatX(np.asarray(prior.latent_mean.values)))
        backend_cholesky = np.asarray(pm.floatX(np.asarray(prior.latent_cholesky.values)))
    if not np.isfinite(backend_mean).all() or (backend_mean <= 0).any():
        raise ValueError(
            "Correlated LogNormal arithmetic means must remain finite and positive in the model float dtype."
        )
    if not np.isfinite(backend_latent_mean).all() or not np.isfinite(backend_cholesky).all():
        raise ValueError("Correlated LogNormal moments must remain finite in the model float dtype.")
    if (np.diag(backend_cholesky) <= 0).any():
        raise ValueError(
            "Correlated LogNormal Cholesky diagonal must remain positive in the model float dtype."
        )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        backend_central_state = np.exp(backend_latent_mean)
    if not np.isfinite(backend_central_state).all() or (backend_central_state <= 0).any():
        raise ValueError(
            "Correlated LogNormal central states must remain finite and positive "
            "after exponentiation in the model float dtype."
        )

    # All predictable dtype validation must finish before the active model is
    # changed.  In particular, failed validation must not leave a latent RV
    # that makes a corrected retry impossible.
    add_coords(mean.coords, model_dims=(state_dim,))
    latent = pm.Normal(f"{var_name}_latent", 0.0, 1.0, dims=state_dim)
    latent_mean = pt.as_tensor_variable(backend_latent_mean)
    cholesky = pt.as_tensor_variable(backend_cholesky)
    state = pm.Deterministic(
        var_name,
        pt.exp(latent_mean + pt.dot(cholesky, latent)),
        dims=state_dim,
    )
    return CorrelatedStateResult(latent=latent, state=state, prior=prior)


def add_correlated_lognormal_state_with_activity(
    activity: ResolvedStateActivity,
    prior: CorrelatedLognormalPrior,
    /,
    *,
    var_name: str,
) -> StateVectorResult:
    """Construct a correlated LogNormal state with exact active/fixed values.

    The arithmetic-moment prior is subset to sampled states before its
    LogNormal transformation. Inactive states keep their exact fixed values in
    the full public vector. This is the correlated counterpart of
    :func:`add_state_vector`.

    Args:
        activity: Resolved activity in canonical full-state order.
        prior: Validated labelled arithmetic-moment LogNormal prior for the
            full state.
        var_name: Name of the full user-facing state vector.

    Returns:
        Effective whitened latent, full state vector, and supplied activity.
    """
    state_dim = activity.state_dim
    if prior.state_dim != state_dim:
        raise ValueError(
            "Correlated LogNormal prior and state activity must use the same "
            f"state dimension; found {prior.state_dim!r} and {state_dim!r}."
        )
    mean = prior.mean
    activity_index = activity.zero_sensitivity.coords[state_dim].to_index()
    if not mean.coords[state_dim].to_index().equals(activity_index):
        raise ValueError(
            "Correlated LogNormal prior labels must exactly match state-activity labels in the same order."
        )
    covariance = prior.arithmetic_covariance

    # All contract checks above deliberately precede model mutation.
    add_coords(activity.zero_sensitivity.coords, model_dims=(state_dim,))

    if activity.n_active == activity.n_state:
        result = add_correlated_lognormal_state(
            prior,
            var_name=var_name,
        )
        return StateVectorResult(
            latent=result.latent,
            state=result.state,
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
        active_index = mean.coords[state_dim].to_index()[active_indices]
        if isinstance(active_index, pd.MultiIndex):
            active_index = active_index.set_names(
                [f"{name}_{var_name}_active" for name in active_index.names]
            )
            active_coords = xr.Coordinates.from_pandas_multiindex(active_index, active_dim)
        else:
            active_coords = {active_dim: active_index.to_numpy()}
        active_mean = xr.DataArray(
            mean.isel({state_dim: active_indices}).values,
            dims=(active_dim,),
            coords=active_coords,
            name=mean.name,
            attrs=mean.attrs,
        )
        active_covariance = covariance.isel(
            {state_dim: active_indices, prior.covariance_dim: active_indices}
        ).values
        result = add_correlated_lognormal_state(
            CorrelatedLognormalPrior(active_mean, active_covariance),
            var_name=f"{var_name}_active",
        )
        latent = result.latent
        active_state = result.state

    full_state = fixed_value
    if active_state is not None:
        full_state = pt.set_subtensor(full_state[active_indices], active_state)
    state = pm.Deterministic(var_name, full_state, dims=state_dim)
    return StateVectorResult(latent=latent, state=state, activity=activity)


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
    observations: xr.DataArray,
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
        observations: Observation data carrying an observation-aligned
            ``site`` coordinate.
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

    Raises:
        ValueError: If ``observations`` does not have an observation-aligned
            ``site`` coordinate, or if global-offset options conflict.
    """
    output_dim = str(output_dim)
    if "site" not in observations.coords or observations.coords["site"].dims != (output_dim,):
        raise ValueError(
            "Offset observations must have an observation-aligned `site` coordinate."
        )
    site_indicator = make_site_indicator(observations.coords["site"])
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
