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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self, cast
from warnings import warn

from dask import compute
import numpy as np
import pandas as pd
import pymc as pm
from pymc.pytensorf import floatX
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.correlated_state import GAMMA_LATENT_CAP, CorrelatedLognormalPrior, CorrelatedPrior
from openghg_inversions.array_ops import select_gathered_data_array
from openghg_inversions.inversion_inputs import make_freq_indicator, make_site_indicator
from openghg_inversions.models.coords import add_component_data, add_coords, add_model_data as add_model_data, component_coords
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
    """Objects created by ``add_correlated_lognormal_state`` or ``add_correlated_copula_state``.

    Attributes:
        latent: Standard-normal whitened state used by the sampler.
        state: User-facing state with the requested arithmetic moments.
        prior: Validated backend-neutral moment contract used to build the
            graph.
    """

    latent: TensorVariable
    state: TensorVariable
    prior: CorrelatedLognormalPrior | CorrelatedPrior


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
    h = add_component_data(prepared.sensitivity, data_name, coordinate_owner, observation_dim=prepared.output_dim)
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


@dataclass(frozen=True)
class ConcentrationTerm:
    """One concentration contribution in its absolute and likelihood forms.

    Most terms enter the likelihood mean as their absolute ``output``. A
    centered term instead contributes a small ``likelihood_contribution``
    against observations translated by its fixed ``observation_shift``;
    adding the shift back gives ``output``. The translation changes no
    covariance, so both forms describe the same probability model. The shift
    is independent of inferred parameters: sampled boundary means and site
    biases remain in the likelihood contribution. This record does not
    require the contribution to be affine in its parameters.

    Attributes:
        output: Absolute concentration contribution.
        likelihood_contribution: This term's contribution to the likelihood mean.
        observation_shift: Labelled concentration subtracted from the
            observations, or ``None`` when the term enters absolutely.
    """

    output: TensorVariable
    likelihood_contribution: TensorVariable
    observation_shift: xr.DataArray | None = None

    @classmethod
    def absolute(cls, output: TensorVariable, **fields: Any) -> Self:
        """Describe a term whose likelihood contribution is its absolute output.

        Subclasses pass their additional keyword-only fields through ``fields``.

        Args:
            output: Absolute concentration contribution, also used as the
                likelihood contribution.
            **fields: Additional fields of the subclass.

        Returns:
            The term, with no observation shift.
        """
        return cls(output, output, **fields)


@dataclass(frozen=True)
class AffineBlock:
    """One labelled two-dimensional design and its physical coefficient vector.

    Rows follow the observation labels. Columns follow the matching one-
    dimensional physical coefficient expression in exactly that order. Each
    block retains its own design/coefficient units; blocks with different
    physical units remain separate until the numerical consumer lowers them.

    Attributes:
        design: Labelled observation-by-coefficient matrix.
        coefficients: One-dimensional physical coefficient expression, ordered
            like the design's columns.
    """

    design: xr.DataArray
    coefficients: TensorVariable


@dataclass(frozen=True, kw_only=True)
class AffineConcentrationTerm(ConcentrationTerm):
    """Concentration term whose likelihood contribution is affine in its coefficients.

    Fixed contribution plus design/coefficient products equals the term's
    likelihood contribution. The fixed contribution is a labelled concentration
    vector in observation-row order. Physical coefficients may be transformed sampled
    variables; ``sampled_rvs`` contains the underlying sampled variables,
    distinct from those physical coefficient expressions. Numerical consumers
    concatenate blocks at their own boundary; every other consumer can treat
    the term as an ordinary :class:`ConcentrationTerm`.

    Attributes:
        fixed_contribution: Labelled concentration vector, in observation-row
            order, that does not depend on the coefficients.
        blocks: Design/coefficient pairs whose products add to the fixed
            contribution.
        sampled_rvs: Underlying sampled variables of the coefficients.
    """

    fixed_contribution: xr.DataArray
    blocks: tuple[AffineBlock, ...]
    sampled_rvs: tuple[TensorVariable, ...]


def warn_constant_offset_confounding(
    terms: Sequence[ConcentrationTerm],
    *,
    drop_first: bool,
    rows: np.ndarray | None = None,
    channel: str | None = None,
) -> None:
    """Warn when an affine coefficient has the same constant response as offsets.

    Call after materialization, before adding unrestricted observation offsets.
    Any affine design column equal to one on every selected row (absolute
    tolerance 1e-12) is exactly confounded with an unrestricted constant
    offset: the likelihood identifies only their sum. Anchoring the first
    site's offset (``drop_first``) removes that constant mode. Overlap with
    non-constant responses is not assessed.

    Args:
        terms: Terms already added to the model; non-affine terms are skipped.
        drop_first: Whether the offsets exclude the first site's constant.
        rows: Optional boolean row selection, for one channel of a joint design.
        channel: Optional channel name for the warning message.
    """
    if drop_first:
        return
    for term in terms:
        if not isinstance(term, AffineConcentrationTerm):
            continue
        for block in term.blocks:
            design = np.asarray(block.design.values)
            if rows is not None:
                design = design[rows]
            if design.size and np.all(np.abs(design - 1.0) <= 1e-12, axis=0).any():
                warn(
                    (f"{channel}: " if channel is not None else "")
                    + "A model coefficient and unrestricted observation offsets share a constant "
                    "observation response; the likelihood identifies only their combined constant. "
                    "Separating boundary-product bias from observation bias requires prior or external "
                    "calibration information.",
                    UserWarning,
                    stacklevel=2,
                )
                return


def likelihood_representation(
    terms: Sequence[ConcentrationTerm],
    observations: xr.DataArray,
    modelled_mean: TensorVariable,
) -> tuple[TensorVariable, xr.DataArray]:
    """Choose the likelihood mean and observations for concentration terms.

    Without observation shifts this returns ``modelled_mean`` and the
    observations unchanged. Otherwise it sums the likelihood contributions
    directly, rather than subtracting a large shift from ``modelled_mean``,
    which would lose small centered anomalies, and registers the total shift
    as model data ``boundary_reference_shift`` so replay can restore absolute
    predictions. Call it inside the active model.

    Args:
        terms: Complete ordered decomposition of ``modelled_mean``. Tensor
            contributions must follow the observations' row order and use
            their concentration units. Shifts must have exactly the
            observations' dimensions; those and indexed row labels are
            checked here.
        observations: Labelled observations in the contributions' common units.
        modelled_mean: Prebuilt absolute mean from those same term outputs,
            retaining the recipe's summation grouping and deterministic name.

    Returns:
        The likelihood mean and the translated observations, which keep the
        observations' attributes.

    Raises:
        ValueError: If any shift's dimensions differ from the observations',
            or indexed observation labels differ between observations and any
            shift, or between shifts.
    """
    shifts = [term.observation_shift for term in terms if term.observation_shift is not None]
    if not shifts:
        return modelled_mean, observations
    # ``xr.align`` compares only shared dimensions, so differently named axes would broadcast.
    for shift in shifts:
        if shift.dims != observations.dims:
            raise ValueError(
                f"Observation shifts must have the observations' dimensions {observations.dims}, not {shift.dims}."
            )
    observations, *shifts = xr.align(observations, *shifts, join="exact", copy=False)
    shift = sum(shifts[1:], start=shifts[0])
    add_model_data(shift.rename("boundary_reference_shift"), "boundary_reference_shift")
    mean = pt.add(*(term.likelihood_contribution for term in terms))
    return mean, (observations - shift).assign_attrs(observations.attrs)


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
    *,
    transform: pm.distributions.transforms.Transform | None = None,
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
        transform: Optional replacement for the prior's default PyMC transform.

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

    transform_options = {} if transform is None else {"transform": transform, "default_transform": None}
    if activity.n_active == activity.n_state:
        state = parse_prior(var_name, parameters, dims=state_dim, **transform_options)
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
            **transform_options,
        )
        latent = get_model_latent(active_state, f"{var_name}_active")

    full_state = fixed_value
    if active_state is not None:
        full_state = pt.set_subtensor(full_state[active_indices], active_state)
    state = pm.Deterministic(var_name, full_state, dims=state_dim)
    return StateVectorResult(latent=latent, state=state, activity=activity)


def _add_multisource_state_vector(
    zero_sensitivity: xr.DataArray,
    prior_blocks: Mapping[str, tuple[str, dict[str, Any], StateActivity | None]],
    *,
    local_state_dim: str,
    var_name: str,
) -> TensorVariable:
    """Insert independent source-prior draws into one ordered full state.

    Each block owns its prior family, source-local labelled parameters and
    activity policy. Private prior dimensions may differ in size; the physical
    state and zero-sensitivity mask use the complete gathered coordinate.
    Block keys cover the source layout, as validated by the owning recipe.
    """
    state_dim = str(zero_sensitivity.dims[0])
    state_index = zero_sensitivity.get_index(state_dim)
    region_level = next(str(name) for name in state_index.names if name != "source")
    add_coords(zero_sensitivity.coords, model_dims=(state_dim,))
    active = np.zeros(len(state_index), dtype=bool)
    fixed_values = np.empty(len(state_index), dtype=float)
    draws = []
    for source, (record_name, prior, policy) in prior_blocks.items():
        positions = np.flatnonzero(state_index.get_level_values("source") == source)
        local_zero = select_gathered_data_array(
            zero_sensitivity, key=source, key_dim="source",
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
        add_model_data(zero_sensitivity.copy(data=active).rename(f"{var_name}_is_active"))
        state = add_model_data(zero_sensitivity.copy(data=fixed_values).rename(f"{var_name}_fixed_value"))
    for positions, draw in draws:
        state = pt.set_subtensor(state[positions], draw)
    return cast(TensorVariable, pm.Deterministic(var_name, state, dims=state_dim))


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
        backend_mean = np.asarray(floatX(np.asarray(mean.values)))
        backend_latent_mean = np.asarray(floatX(np.asarray(prior.latent_mean.values)))
        backend_cholesky = np.asarray(floatX(np.asarray(prior.latent_cholesky.values)))
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
    active_prior = prepare_active_correlated_lognormal_prior(
        activity,
        prior,
        var_name=var_name,
    )
    return _add_prepared_correlated_lognormal_state_with_activity(
        activity,
        active_prior,
        var_name=var_name,
    )


def _add_prepared_correlated_lognormal_state_with_activity(
    activity: ResolvedStateActivity,
    active_prior: CorrelatedLognormalPrior | CorrelatedPrior | None,
    /,
    *,
    var_name: str,
) -> StateVectorResult:
    """Construct a state from the package-prepared active prior.

    ``active_prior`` comes from :func:`prepare_active_correlated_prior` or
    :func:`prepare_active_correlated_lognormal_prior`. A lognormal prior, or a
    :class:`CorrelatedPrior` whose marginals are all lognormal, builds the
    lognormal graph unchanged; other marginals use
    :func:`add_correlated_copula_state`.
    """
    if isinstance(active_prior, CorrelatedPrior):
        if active_prior.lognormal is None:
            return _add_prepared_state_with_activity(
                activity, active_prior, var_name=var_name, add_state=add_correlated_copula_state,
            )
        active_prior = active_prior.lognormal
    return _add_prepared_state_with_activity(
        activity, active_prior, var_name=var_name, add_state=add_correlated_lognormal_state,
    )


def _add_prepared_state_with_activity(
    activity: ResolvedStateActivity,
    active_prior: Any,
    /,
    *,
    var_name: str,
    add_state: Any,
) -> StateVectorResult:
    """Construct a full state from an active prior and its state builder."""
    state_dim = activity.state_dim
    add_coords(activity.zero_sensitivity.coords, model_dims=(state_dim,))

    if activity.n_active == activity.n_state:
        assert active_prior is not None
        result = add_state(
            active_prior,
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
        assert active_prior is not None
        result = add_state(
            active_prior,
            var_name=f"{var_name}_active",
        )
        latent = result.latent
        active_state = result.state

    full_state = fixed_value
    if active_state is not None:
        full_state = pt.set_subtensor(full_state[active_indices], active_state)
    state = pm.Deterministic(var_name, full_state, dims=state_dim)
    return StateVectorResult(latent=latent, state=state, activity=activity)


def _check_active_prior_labels(
    activity: ResolvedStateActivity, mean: xr.DataArray, prior_state_dim: str, *, label: str
) -> None:
    """Require the prior and the activity to share state labels and coordinates."""
    state_dim = activity.state_dim
    if prior_state_dim != state_dim:
        raise ValueError(
            f"{label} prior and state activity must use the same "
            f"state dimension; found {prior_state_dim!r} and {state_dim!r}."
        )
    activity_index = activity.zero_sensitivity.coords[state_dim].to_index()
    if not mean.coords[state_dim].to_index().equals(activity_index):
        raise ValueError(
            f"{label} prior labels must exactly match state-activity labels in the same order."
        )
    prior_coords, activity_coords = compute(
        mean.coords.to_dataset(), activity.zero_sensitivity.coords.to_dataset(),
    )
    xr.merge([prior_coords, activity_coords], join="exact", compat="equals")


def _active_state_array(
    values: xr.DataArray, state_dim: str, active_indices: np.ndarray, var_name: str
) -> xr.DataArray:
    """Select active entries onto the namespaced active-state dimension."""
    active_dim, active_coords = _active_state_coords(values.coords[state_dim], active_indices, f"{var_name}_active")
    return xr.DataArray(
        values.isel({state_dim: active_indices}).values,
        dims=(active_dim,),
        coords=active_coords,
        name=values.name,
        attrs=values.attrs,
    )


def prepare_active_correlated_lognormal_prior(
    activity: ResolvedStateActivity,
    prior: CorrelatedLognormalPrior,
    /,
    *,
    var_name: str,
) -> CorrelatedLognormalPrior | None:
    """Align and subset one correlated prior for active-state construction.

    This eager graph-construction boundary materializes the prior and activity
    coordinate datasets together before comparing auxiliary labels. Borrowed
    inputs remain unchanged and shared Dask coordinate tasks run once.
    """
    mean = prior.mean
    _check_active_prior_labels(activity, mean, prior.state_dim, label="Correlated LogNormal")
    if activity.n_active == 0:
        return None
    if activity.n_active == activity.n_state:
        return prior

    active_indices = activity.active_indices
    active_mean = _active_state_array(mean, activity.state_dim, active_indices, var_name)
    active_covariance = prior.arithmetic_covariance.isel(
        {activity.state_dim: active_indices, prior.covariance_dim: active_indices}
    ).values
    return CorrelatedLognormalPrior(active_mean, active_covariance)


def _as_lognormal_prior(prior: CorrelatedLognormalPrior | CorrelatedPrior) -> CorrelatedLognormalPrior | None:
    """Return the validated lognormal prior, given directly or carried by an all-lognormal wrapper.

    ``None`` when a :class:`CorrelatedPrior` has a non-lognormal marginal.
    """
    return prior.lognormal if isinstance(prior, CorrelatedPrior) else prior


_ASYMPTOTIC_ERFC_START = 10.0
"""Above this ``a = u / sqrt(2)``, ``log erfc(a)`` uses its asymptotic series."""


def _negative_log_normal_sf(u: TensorVariable) -> TensorVariable:
    """Return ``-log Phi(-u)`` accurately for all ``u``, with JAX-supported ops only.

    ``u <= 0``: ``-log1p(-Phi(u))``. Moderate ``u``: ``-log(erfc(a) / 2)``.
    Large ``u``: ``a**2 + log(2 a sqrt(pi)) - log(S(a))`` with the asymptotic
    series ``S`` of ``erfc(a) a sqrt(pi) exp(a**2)``, whose first omitted term
    is below 1e-11 at the switch. The result is floored at the smallest
    normal float so its logarithm and gradient stay finite far in the lower
    tail, where the Weibull state is zero to working precision.
    """
    a = u / np.sqrt(2.0)
    lower = -pt.log1p(-0.5 * pt.erfc(-pt.minimum(a, 0.0)))
    middle = -pt.log(0.5 * pt.erfc(pt.clip(a, 0.0, _ASYMPTOTIC_ERFC_START)))
    large = pt.maximum(a, _ASYMPTOTIC_ERFC_START)
    inverse = 1.0 / pt.square(large)
    # S(a) = sum_n (-1)**n (2n - 1)!! / (2 a**2)**n, n = 0..6, in Horner form.
    series = 10395 / 64
    for coefficient in (-945 / 32, 105 / 16, -15 / 8, 3 / 4, -1 / 2, 1.0):
        series = coefficient + inverse * series
    upper = pt.square(large) + pt.log(2.0 * np.sqrt(np.pi) * large) - pt.log(series)
    value = pt.switch(a <= 0, lower, pt.switch(a <= _ASYMPTOTIC_ERFC_START, middle, upper))
    return pt.maximum(value, np.finfo(np.asarray(floatX(0.0)).dtype).tiny)


def _copula_marginal_map(
    latent: TensorVariable,
    family: np.ndarray,
    mean: np.ndarray,
    shape: np.ndarray,
    scale: np.ndarray,
) -> TensorVariable:
    """Return ``F_i^-1(Phi(u_i))`` for correlated standard-normal ``u``.

    Every branch uses only built-in PyTensor operations, so the graph pickles
    to spawned sampler processes. Both sides of each switch are evaluated at
    clamped arguments so that neither produces NaN or infinite gradients.
    """
    pieces: list[tuple[np.ndarray, TensorVariable]] = []
    for name in dict.fromkeys(family):
        index = np.flatnonzero(family == name)
        u = latent if index.size == family.size else latent[index]
        k = pt.as_tensor_variable(floatX(shape[index]))
        theta = pt.as_tensor_variable(floatX(scale[index]))
        if name == "lognormal":
            value = theta * pt.exp(k * u)
        elif name == "normal":
            value = pt.as_tensor_variable(floatX(mean[index])) + theta * u
        elif name == "gamma":
            u = pt.minimum(u, GAMMA_LATENT_CAP)
            lower = 0.5 * pt.erfc(-pt.minimum(u, 0) / np.sqrt(2.0))
            upper = 0.5 * pt.erfc(pt.maximum(u, 0) / np.sqrt(2.0))
            value = theta * pt.switch(u > 0, pt.gammainccinv(k, upper), pt.gammaincinv(k, lower))
        else:
            value = theta * pt.exp(pt.log(_negative_log_normal_sf(u)) / k)
        pieces.append((index, value))
    if len(pieces) == 1:
        return pieces[0][1]
    state = pt.zeros(family.size)
    for index, value in pieces:
        state = pt.set_subtensor(state[index], value)
    return state


def add_correlated_copula_state(
    prior: CorrelatedPrior,
    /,
    *,
    var_name: str,
) -> CorrelatedStateResult:
    """Add a whitened Gaussian-copula state with per-state marginals.

    The whitened latent is ``{var_name}_latent``, as for
    :func:`add_correlated_lognormal_state`, and the state is
    ``x_i = F_i^-1(Phi((L z)_i))`` with ``L`` the prior's latent-correlation
    Cholesky factor.

    Args:
        prior: Arithmetic moments, marginals and latent correlation.
        var_name: Name of the user-facing state.

    Returns:
        The whitened latent, the state, and the supplied prior.

    Raises:
        ValueError: If the factor or marginal parameters are not finite in
            PyMC's configured floating-point dtype.

    Notes:
        This must run in an active ``pm.Model``. Validation finishes before
        the model is changed. The same moment-matched closure caveat as
        :func:`add_correlated_lognormal_state` applies.

        Every marginal compiles for PyMC's sampler and for JAX samplers.
        Under JAX, gamma states use TensorFlow Probability's inverse
        regularized incomplete gamma (the ``tfp-nightly`` dependency).
    """
    state_dim = prior.state_dim
    family = np.asarray(prior.marginal.values).astype(str)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        mean = np.asarray(floatX(np.asarray(prior.mean.values)))
        cholesky = np.asarray(floatX(np.asarray(prior.latent_correlation_cholesky.values)))
        shape = np.asarray(floatX(np.asarray(prior.marginal_shape.values)))
        scale = np.asarray(floatX(np.asarray(prior.marginal_scale.values)))
    shaped = family != "normal"
    if (
        not np.isfinite(cholesky).all()
        or (np.diag(cholesky) <= 0).any()
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
        or not np.isfinite(shape[shaped]).all()
        or (shape[shaped] <= 0).any()
    ):
        raise ValueError("Copula prior factors and marginal parameters must remain finite in the model float dtype.")

    add_coords(prior.mean.coords, model_dims=(state_dim,))
    latent = pm.Normal(f"{var_name}_latent", 0.0, 1.0, dims=state_dim)
    correlated = pt.dot(pt.as_tensor_variable(cholesky), latent)
    state = pm.Deterministic(
        var_name, _copula_marginal_map(correlated, family, mean, shape, scale), dims=state_dim,
    )
    return CorrelatedStateResult(latent=latent, state=state, prior=prior)


def add_correlated_state_with_activity(
    activity: ResolvedStateActivity,
    prior: CorrelatedLognormalPrior | CorrelatedPrior,
    /,
    *,
    var_name: str,
) -> StateVectorResult:
    """Construct a correlated state with each state's marginal and exact fixed values.

    A validated lognormal prior, given directly or as an all-lognormal
    :class:`CorrelatedPrior`, uses
    :func:`add_correlated_lognormal_state_with_activity`, so the graph is
    unchanged. Otherwise the active states use
    :func:`add_correlated_copula_state`.

    Args:
        activity: Resolved activity in canonical full-state order.
        prior: Arithmetic moments and marginals for the full state.
        var_name: Name of the full user-facing state vector.

    Returns:
        Effective whitened latent, full state vector, and supplied activity.
    """
    lognormal = _as_lognormal_prior(prior)
    if lognormal is not None:
        return add_correlated_lognormal_state_with_activity(activity, lognormal, var_name=var_name)
    active_prior = prepare_active_correlated_prior(activity, prior, var_name=var_name)
    return _add_prepared_correlated_lognormal_state_with_activity(activity, active_prior, var_name=var_name)


def prepare_active_correlated_prior(
    activity: ResolvedStateActivity,
    prior: CorrelatedLognormalPrior | CorrelatedPrior,
    /,
    *,
    var_name: str,
) -> CorrelatedLognormalPrior | CorrelatedPrior | None:
    """Subset a correlated prior to its active states before graph construction.

    The active prior keeps each active state's marginal. Its latent
    correlation is the active block of the full one, because the copula
    matches covariances pair by pair. Pass the result to
    ``_add_prepared_correlated_lognormal_state_with_activity``.

    Args:
        activity: Resolved activity in canonical full-state order.
        prior: Arithmetic moments and marginals for the full state.
        var_name: Name of the full user-facing state vector.

    Returns:
        The active prior, or ``None`` when every state is fixed.

    Raises:
        ValueError: If the prior and activity labels differ.
    """
    lognormal = _as_lognormal_prior(prior)
    if lognormal is not None:
        return prepare_active_correlated_lognormal_prior(activity, lognormal, var_name=var_name)
    assert isinstance(prior, CorrelatedPrior)
    mean = prior.mean
    _check_active_prior_labels(activity, mean, prior.state_dim, label="Correlated")
    if activity.n_active == 0:
        return None
    if activity.n_active == activity.n_state:
        return prior
    active_indices = activity.active_indices
    state_dim = activity.state_dim
    active_mean = _active_state_array(mean, state_dim, active_indices, var_name)
    active_marginal = _active_state_array(prior.marginal, state_dim, active_indices, var_name)
    active_covariance = prior.arithmetic_covariance.isel(
        {state_dim: active_indices, prior.covariance_dim: active_indices}
    ).values
    return CorrelatedPrior(active_mean, active_covariance, marginal=active_marginal)


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
            [pd.Index(site_labels[selected_sites]), pd.Index(period_labels)],
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
