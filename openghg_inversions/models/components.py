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

Model components derive frequency indicators from observation coordinates with
``openghg_inversions.inversion_inputs.make_freq_indicator``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.array_ops import select_gathered_data_array
from openghg_inversions.inversion_inputs import make_freq_indicator, make_site_indicator
from openghg_inversions.models.coords import add_coords, add_model_data as add_model_data, component_coords
from openghg_inversions.models._flux import _namespace_sector_state_coords
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
class OffsetComponentResult:
    """Graph objects and labelled design created for one offset component."""

    design: xr.DataArray
    latent: TensorVariable
    coefficients: TensorVariable
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
    coordinate_owner: str | None = None,
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
        coordinate_owner: Explicit namespace for registered state coordinates;
            scientific inputs and activity policies retain their local names.

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
    vector = add_state_vector(
        activity, prior_args=prior_args, var_name=var_name, coordinate_owner=coordinate_owner,
    )
    output = apply_linear_sensitivity(
        prepared,
        vector.state,
        data_name=data_name,
        output_name=output_name,
        compute_deterministic=compute_deterministic,
        coordinate_owner=coordinate_owner,
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
    coordinate_owner: str | None = None,
) -> TensorVariable:
    """Apply a prepared sensitivity using its state's explicit coordinate owner."""
    h = add_model_data(component_coords(
        prepared.sensitivity, coordinate_owner, observation_dim=prepared.output_dim,
    ), data_name)
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


def _active_state_coords(
    coordinate: xr.DataArray,
    active_indices: np.ndarray,
    namespace: str,
) -> tuple[str, xr.Coordinates | dict[str, Any]]:
    """Select active labels and namespace their dimension and MultiIndex levels."""
    active_dim = f"{coordinate.dims[0]}_{namespace}"
    active_index = coordinate.to_index()[active_indices]
    if isinstance(active_index, pd.MultiIndex):
        active_index = active_index.set_names([f"{name}_{namespace}" for name in active_index.names])
        return active_dim, xr.Coordinates.from_pandas_multiindex(active_index, active_dim)
    return active_dim, {active_dim: active_index.to_numpy()}


def add_state_vector(
    activity: ResolvedStateActivity,
    /,
    prior_args: dict[str, Any],
    var_name: str,
    coordinate_owner: str | None = None,
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
        coordinate_owner: Optional namespace for registered state coordinates.

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
    owned_zero = component_coords(activity.zero_sensitivity, coordinate_owner)
    state_dim = str(owned_zero.dims[0])
    state_coord = owned_zero.coords[state_dim]
    add_coords(owned_zero.coords, model_dims=(state_dim,))
    parameters = {
        name: component_coords(value, coordinate_owner) if isinstance(value, xr.DataArray) else value
        for name, value in active_prior_args(prior_args, activity, select_active=False).items()
    }

    if activity.n_active == activity.n_state:
        state = parse_prior(var_name, parameters, dims=state_dim)
        return StateVectorResult(
            latent=get_model_latent(state, var_name),
            state=state,
            activity=activity,
        )

    add_model_data(
        component_coords(activity.active, coordinate_owner).rename(f"{var_name}_is_active"),
        f"{var_name}_is_active",
    )
    fixed_value = add_model_data(
        component_coords(activity.fixed_value, coordinate_owner).rename(f"{var_name}_fixed_value"),
        f"{var_name}_fixed_value",
    )
    active_indices = activity.active_indices
    latent: TensorVariable | None = None
    active_state: TensorVariable | None = None
    if activity.n_active:
        active_dim, active_coords = _active_state_coords(state_coord, active_indices, f"{var_name}_active")
        add_coords(active_coords, model_dims=(active_dim,))
        active_state = parse_prior(
            f"{var_name}_active",
            parameters,
            dims=active_dim,
            active_indices=active_indices,
            record_name=var_name,
        )
        latent = get_model_latent(active_state, f"{var_name}_active")

    full_state = fixed_value
    if active_state is not None:
        full_state = pt.set_subtensor(full_state[active_indices], active_state)
    state = pm.Deterministic(var_name, full_state, dims=state_dim)
    return StateVectorResult(latent=latent, state=state, activity=activity)


def _add_multisector_state_vector(
    zero_sensitivity: xr.DataArray,
    prior_blocks: Mapping[str, tuple[str, dict[str, Any], StateActivity | None]],
    *,
    local_state_dim: str,
    var_name: str,
    coordinate_owner: str | None = None,
) -> TensorVariable:
    """Insert independent sector-prior draws into one ordered full state.

    Each block owns its prior family, sector-local labelled parameters and
    activity policy. Private prior dimensions may differ in size; the physical
    state and zero-sensitivity mask use the complete gathered coordinate.
    Block keys cover the sector layout, as validated by the owning recipe.
    """
    state_dim = str(zero_sensitivity.dims[0])
    state_index = zero_sensitivity.get_index(state_dim)
    region_level = next(str(name) for name in state_index.names if name != "sector")
    owned_zero = component_coords(zero_sensitivity, coordinate_owner)
    owned_dim = str(owned_zero.dims[0])
    add_coords(owned_zero.coords, model_dims=(owned_dim,))
    active = np.zeros(len(state_index), dtype=bool)
    fixed_values = np.empty(len(state_index), dtype=float)
    draws = []
    for sector, (record_name, prior, policy) in prior_blocks.items():
        positions = np.flatnonzero(state_index.get_level_values("sector") == sector)
        local_zero = select_gathered_data_array(
            zero_sensitivity, key=sector, key_dim="sector",
            ragged_dim=region_level, stack_dim=state_dim,
        )
        if state_dim != local_state_dim:
            local_zero = local_zero.rename({state_dim: local_state_dim})
        activity = resolve_state_activity(local_zero, policy)
        aligned_prior = active_prior_args(dict(prior), activity, select_active=False)
        active[positions] = activity.active.to_numpy()
        fixed_values[positions] = activity.fixed_value.to_numpy()
        if not activity.n_active:
            continue

        prior_namespace = f"prior_{record_name}"
        block_zero = _namespace_sector_state_coords(
            local_zero, variable_suffix=prior_namespace, namespace_state_dim=True,
        )
        block_dim = str(block_zero.dims[0])
        add_coords(block_zero.coords, model_dims=(block_dim,))
        parameters = {
            name: _namespace_sector_state_coords(
                value, variable_suffix=prior_namespace, namespace_state_dim=True,
            ) if isinstance(value, xr.DataArray) and value.ndim else value
            for name, value in aligned_prior.items()
        }
        active_indices = activity.active_indices
        active_dim, active_coords = _active_state_coords(
            local_zero.coords[local_state_dim], active_indices, f"active_{record_name}",
        )
        add_coords(active_coords, model_dims=(active_dim,))
        draw = parse_prior(
            f"{record_name}_active", parameters, dims=active_dim,
            active_indices=active_indices, record_name=record_name,
        )
        draws.append((positions[active_indices], draw))

    if active.all():
        state = pt.zeros(len(state_index))
    else:
        add_model_data(owned_zero.copy(data=active).rename(f"{var_name}_is_active"))
        state = add_model_data(owned_zero.copy(data=fixed_values).rename(f"{var_name}_fixed_value"))
    for positions, draw in draws:
        state = pt.set_subtensor(state[positions], draw)
    return cast(TensorVariable, pm.Deterministic(var_name, state, dims=owned_dim))


def add_correlated_lognormal_state(
    prior: CorrelatedLognormalPrior,
    /,
    *,
    var_name: str,
    coordinate_owner: str | None = None,
) -> CorrelatedStateResult:
    """Add a whitened correlated LogNormal state to the active PyMC model.

    Args:
        prior: Validated labelled arithmetic and latent moment contract.
        var_name: Name of the positive user-facing state. The whitened standard
            normal is named ``{var_name}_latent``.
        coordinate_owner: Optional component namespace for registered state
            coordinates; the supplied scientific prior remains unchanged.

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
    mean = component_coords(prior.mean, coordinate_owner)
    state_dim = str(mean.dims[0])
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
    coordinate_owner: str | None = None,
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
        coordinate_owner: Optional component namespace for registered state
            coordinates; scientific prior and activity labels stay unchanged.

    Returns:
        Effective whitened latent, full state vector, and supplied activity.
    """
    active_prior = prepare_active_correlated_lognormal_prior(
        activity,
        prior,
        var_name=var_name,
    )
    return _add_prepared_correlated_lognormal_state_with_activity(
        activity,
        active_prior,
        var_name=var_name,
        coordinate_owner=coordinate_owner,
    )


def _add_prepared_correlated_lognormal_state_with_activity(
    activity: ResolvedStateActivity,
    active_prior: CorrelatedLognormalPrior | None,
    /,
    *,
    var_name: str,
    coordinate_owner: str | None = None,
) -> StateVectorResult:
    """Construct a state from the package-prepared active prior."""
    registered_zero = component_coords(activity.zero_sensitivity, coordinate_owner)
    state_dim = str(registered_zero.dims[0])
    add_coords(registered_zero.coords, model_dims=(state_dim,))

    if activity.n_active == activity.n_state:
        assert active_prior is not None
        result = add_correlated_lognormal_state(
            active_prior,
            var_name=var_name,
            coordinate_owner=coordinate_owner,
        )
        return StateVectorResult(
            latent=result.latent,
            state=result.state,
            activity=activity,
        )

    add_model_data(
        component_coords(activity.active, coordinate_owner),
        f"{var_name}_is_active",
    )
    fixed_value = add_model_data(
        component_coords(activity.fixed_value, coordinate_owner),
        f"{var_name}_fixed_value",
    )
    active_indices = activity.active_indices
    latent: TensorVariable | None = None
    active_state: TensorVariable | None = None
    if activity.n_active:
        assert active_prior is not None
        result = add_correlated_lognormal_state(
            active_prior,
            var_name=f"{var_name}_active",
            coordinate_owner=coordinate_owner,
        )
        latent = result.latent
        active_state = result.state

    full_state = fixed_value
    if active_state is not None:
        full_state = pt.set_subtensor(full_state[active_indices], active_state)
    state = pm.Deterministic(var_name, full_state, dims=state_dim)
    return StateVectorResult(latent=latent, state=state, activity=activity)


def prepare_active_correlated_lognormal_prior(
    activity: ResolvedStateActivity,
    prior: CorrelatedLognormalPrior,
    /,
    *,
    var_name: str,
) -> CorrelatedLognormalPrior | None:
    """Align and subset one correlated prior for active-state construction."""
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
    xr.merge(
        [mean.coords.to_dataset(), activity.zero_sensitivity.coords.to_dataset()],
        join="exact",
        compat="equals",
    )
    if activity.n_active == 0:
        return None
    if activity.n_active == activity.n_state:
        return prior

    active_indices = activity.active_indices
    active_dim, active_coords = _active_state_coords(mean.coords[state_dim], active_indices, f"{var_name}_active")
    active_mean = xr.DataArray(
        mean.isel({state_dim: active_indices}).values,
        dims=(active_dim,),
        coords=active_coords,
        name=mean.name,
        attrs=mean.attrs,
    )
    active_covariance = prior.arithmetic_covariance.isel(
        {state_dim: active_indices, prior.covariance_dim: active_indices}
    ).values
    return CorrelatedLognormalPrior(active_mean, active_covariance)


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
            "nsigma_site": np.asarray(alignment.site_labels),
            "nsigma_time": np.arange(alignment.nperiod),
        }
    )

    sigma = parse_prior("sigma", prior_args, dims=("nsigma_site", "nsigma_time"))
    aligned = sigma[site_data_var, period_data_var]
    if compute_deterministic:
        return pm.Deterministic("sigma_aligned", aligned, dims="nmeasure")
    return aligned


def _add_offset_component_result(
    observations: xr.DataArray,
    /,
    prior_args: dict,
    offset_freq: str | None = None,
    var_name: str = "offset_latent",
    output_name: str = "offset",
    output_dim: str = "nmeasure",
    drop_first: bool = False,
    per_site: bool = True,
    namespace: str = "",
) -> OffsetComponentResult:
    """Build one offset component and return its labelled design and graph terms."""
    output_dim = str(output_dim)
    output_coord = observations.coords[output_dim]
    if namespace:
        output_index = observations.indexes.get(output_dim)
        if isinstance(output_index, pd.MultiIndex):
            output_coord = xr.Coordinates.from_pandas_multiindex(
                output_index.set_names([f"{namespace}{name}" for name in output_index.names]), output_dim
            )[output_dim]
        add_coords({output_dim: output_coord})
    term_dim = f"{namespace}offset_term"
    if not per_site:
        if offset_freq is not None:
            raise ValueError("Global offsets do not accept an offset frequency.")
        if drop_first:
            raise ValueError("Global offsets do not support `drop_first=True`.")
        design = xr.DataArray(
            np.ones((observations.sizes[output_dim], 1), dtype=np.float64),
            dims=(output_dim, term_dim),
            coords={output_dim: output_coord, term_dim: ["global"]},
            name="offset_design",
        )
        coefficient = parse_prior(var_name, prior_args)
        coefficients = pt.atleast_1d(coefficient)
        output = pm.Deterministic(
            output_name,
            pt.broadcast_to(coefficient, (observations.sizes[output_dim],)),
            dims=output_dim,
        )
        return OffsetComponentResult(
            design=design,
            latent=get_model_latent(coefficient, var_name),
            coefficients=coefficients,
            output=output,
        )

    if "site" not in observations.coords or observations.coords["site"].dims != (output_dim,):
        raise ValueError(
            "Offset observations must have an observation-aligned `site` coordinate."
        )
    if bool(pd.isna(observations.coords["site"].values).any()):
        raise ValueError("Offset observations must have non-missing site labels.")
    site_indicator = make_site_indicator(observations.coords["site"])
    site_indicator = site_indicator.rename(f"{namespace}site_indicator").transpose(output_dim)
    indicator = None
    if offset_freq is not None:
        time_coord = observations.coords.get("time")
        if time_coord is None or time_coord.dims != (output_dim,):
            raise ValueError(
                "Cannot derive offset frequency indicator: no observation-aligned "
                "time coordinate found."
            )
        if bool(pd.isna(time_coord.values).any()):
            raise ValueError("Offset frequencies require complete observation timestamps.")
        indicator = make_freq_indicator(time_coord, offset_freq).rename(
            f"{namespace}offset_freq_indicator"
        )

    site_codes = np.asarray(site_indicator.values, dtype=int)
    site_labels = pd.unique(np.asarray(observations.coords["site"].values))
    selected_sites = np.arange(int(drop_first), site_labels.size)
    if selected_sites.size == 0:
        raise ValueError("drop_first removes the only available offset site.")
    site_matrix = (site_codes[:, None] == selected_sites[None, :]).astype(int)
    term_coords: xr.Coordinates | dict[str, Any]
    if indicator is not None:
        if bool(pd.isna(indicator.values).any()):
            raise ValueError("Offset frequency indicators must have non-missing labels.")
        period_dummies = pd.get_dummies(indicator.values, dtype=int)
        period_matrix = period_dummies.values
        period_labels = period_dummies.columns.to_numpy()
        design_matrix = (site_matrix[:, :, None] * period_matrix[:, None, :]).reshape(
            site_matrix.shape[0], -1
        )
        term_index = pd.MultiIndex.from_product(
            [site_labels[selected_sites], period_labels],
            names=(f"{namespace}offset_site", f"{namespace}offset_period"),
        )
        term_coords = xr.Coordinates.from_pandas_multiindex(
            term_index,
            term_dim,
        )
    else:
        design_matrix = site_matrix
        selected_labels = site_labels[selected_sites]
        term_coords = {
            term_dim: selected_labels,
            f"{namespace}offset_site": (term_dim, selected_labels),
        }

    design = xr.DataArray(
        design_matrix,
        dims=(output_dim, term_dim),
        coords={
            output_dim: output_coord,
            **term_coords,
        },
        name="offset_design",
    )
    if namespace:
        if isinstance(observations.indexes.get(output_dim), pd.MultiIndex):
            site_indicator = site_indicator.reset_index(output_dim, drop=True).assign_coords(
                {output_dim: output_coord}
            )
            if indicator is not None:
                indicator = indicator.reset_index(output_dim, drop=True).assign_coords({output_dim: output_coord})
        else:
            site_indicator = site_indicator.rename({
                name: f"{namespace}{name}" for name in site_indicator.coords if name != output_dim
            })
            if indicator is not None:
                indicator = indicator.rename({
                    name: f"{namespace}{name}" for name in indicator.coords if name != output_dim
                })
    add_model_data(site_indicator, str(site_indicator.name))
    if indicator is not None:
        add_model_data(indicator.transpose(output_dim), str(indicator.name))
    design_data = add_model_data(design, f"{output_name}_design")
    coefficient = parse_prior(var_name, prior_args, dims=term_dim)
    coefficients = pt.atleast_1d(coefficient)
    aligned = pt.dot(design_data, coefficients)
    output = pm.Deterministic(
        output_name,
        aligned,
        dims=output_dim,
    )
    return OffsetComponentResult(
        design=design,
        latent=get_model_latent(coefficient, var_name),
        coefficients=coefficients,
        output=output,
    )


def add_offset_component(
    observations: xr.DataArray,
    /,
    prior_args: dict,
    offset_freq: str | None = None,
    var_name: str = "offset_latent",
    output_name: str = "offset",
    output_dim: str = "nmeasure",
    drop_first: bool = False,
    per_site: bool = True,
    namespace: str = "",
) -> TensorVariable:
    """Add a global, site-only, or site-by-period offset component.

    Args:
        observations: Observation data carrying an observation-aligned
            ``site`` coordinate.
        prior_args: Prior specification for the offset latent variable.
        offset_freq: Optional frequency string used to derive periods from the
            observation time coordinate.
        var_name: Name for the latent offset variable.
        output_name: Name for the aligned deterministic offset output.
        output_dim: Observation/output dimension name.
        drop_first: Whether to omit the first site indicator column.
        per_site: Whether to create site-specific terms. If false, create one
            global scalar latent offset and broadcast it over observations.
        namespace: Optional prefix for component coordinates and indicator data
            when several independent offset components share one model.

    Returns:
        The aligned offset deterministic variable.

    Raises:
        ValueError: If ``observations`` lacks required site or time coordinates,
            or the selected global or drop-first options cannot form a component.
    """
    return _add_offset_component_result(
        observations,
        prior_args=prior_args,
        offset_freq=offset_freq,
        var_name=var_name,
        output_name=output_name,
        output_dim=output_dim,
        drop_first=drop_first,
        per_site=per_site,
        namespace=namespace,
    ).output
