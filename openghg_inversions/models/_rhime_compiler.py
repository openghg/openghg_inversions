"""Private PyMC compiler for normalized RHIME linear forward plans.

The frozen plan records contain only labelled xarray designs, prior metadata,
semantic identifiers, and requested variable names. The compiler validates a
complete plan before registering any objects on the active PyMC model, then
creates each state once and applies its ordered forward terms.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.models.components import add_model_data, add_state_vector
from openghg_inversions.models.coords import attach_coord_registry, CoordRegistry
from openghg_inversions.models.state_activity import (
    ResolvedStateActivity,
    StateActivity,
    detect_zero_sensitivity,
    resolve_state_activity,
)


@dataclass(frozen=True)
class _StatePlan:
    """Describe one semantic state and its backend variable metadata.

    Args:
        state_id: Stable semantic identifier for the state.
        variable_name: Backend variable name used for the state prior.
        prior_args: Prior metadata forwarded to the backend compiler.
        state_activity: Optional active/fixed policy for the full state vector.
    """

    state_id: str
    variable_name: str
    prior_args: Mapping[str, Any]
    state_activity: StateActivity | None = None


@dataclass(frozen=True)
class _ForwardTermPlan:
    """Describe one labelled linear application of a state to observations.

    Args:
        term_id: Stable semantic identifier for the forward term.
        state_id: Semantic identifier of the state used by this term.
        design: Selected two-dimensional labelled design matrix.
        data_name: Backend model-data name for ``design``.
        deterministic_name: Backend name for the term contribution.
        coefficient: Fixed finite scalar multiplying the linear contribution.
    """

    term_id: str
    state_id: str
    design: xr.DataArray
    data_name: str
    deterministic_name: str
    coefficient: float = 1.0


@dataclass(frozen=True)
class _FluxPlan:
    """Describe ordered flux states and terms independently of compiler strategy.

    Args:
        states: State plans in stable declaration order.
        terms: Forward terms in stable semantic order.
        observation_dim: Shared observation dimension across term designs.
        total_name: Backend deterministic name for the summed contribution.
    """

    states: tuple[_StatePlan, ...]
    terms: tuple[_ForwardTermPlan, ...]
    observation_dim: str = "nmeasure"
    total_name: str = "mu"


@dataclass(frozen=True)
class _CompiledFlux:
    """Expose compiled tensors keyed by their semantic identifiers.

    Args:
        mu: Total observation-space contribution.
        states: User-facing state tensors keyed by state ID.
        latents: Effective backend latent tensors keyed by state ID. All-fixed
            states map to ``None``.
        terms: Forward contribution tensors keyed by term ID.
    """

    mu: TensorVariable
    states: Mapping[str, TensorVariable]
    latents: Mapping[str, TensorVariable | None]
    terms: Mapping[str, TensorVariable]


def _require_unique(values: list[str], label: str) -> None:
    """Require unique values in a plan identifier or backend-name collection.

    Args:
        values: Names or identifiers to validate.
        label: Human-readable collection label for errors.

    Raises:
        ValueError: If ``values`` contains duplicates.
    """
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"Flux plan {label} must be unique; duplicates: {sorted(duplicates)!r}.")


def _uses_reparameterized_lognormal(state: _StatePlan) -> bool:
    """Return whether a state prior creates an additional backend latent."""
    pdf = str(state.prior_args.get("pdf", "")).lower()
    return pdf == "lognormal" and bool(state.prior_args.get("reparameterise", False))


def _state_backend_names(
    state: _StatePlan,
    activity: ResolvedStateActivity,
) -> tuple[str, ...]:
    """Return every backend name created by a resolved state vector."""
    if activity.n_active == activity.n_state:
        if _uses_reparameterized_lognormal(state):
            return (state.variable_name, f"{state.variable_name}_latent")
        return (state.variable_name,)

    names = [
        state.variable_name,
        f"{state.variable_name}_is_active",
        f"{state.variable_name}_fixed_value",
    ]
    if activity.n_active:
        names.append(f"{state.variable_name}_active")
        if _uses_reparameterized_lognormal(state):
            names.append(f"{state.variable_name}_active_latent")
    return tuple(names)


def _aggregate_state_design(
    terms: list[_ForwardTermPlan],
    *,
    observation_dim: str,
) -> xr.DataArray:
    """Combine effective designs for exact-zero detection for one state.

    A state is exactly insensitive only when every coefficient-scaled term is
    zero for that state. Concatenation avoids cancellation between terms while
    retaining the canonical state coordinates needed for labelled policies.

    Args:
        terms: Validated forward terms sharing one semantic state.
        observation_dim: Common observation dimension.

    Returns:
        One labelled design containing every effective term column.
    """
    effective_designs = [term.design * term.coefficient for term in terms]
    if len(effective_designs) == 1:
        return effective_designs[0]
    return xr.concat(effective_designs, dim=observation_dim)


def _coords_for_dims(data: xr.DataArray, dims: tuple[str, ...]) -> dict[str, xr.DataArray]:
    """Return non-scalar coordinates defined only over selected dimensions.

    Args:
        data: Labelled design matrix to inspect.
        dims: Dimensions whose coordinates should be selected.

    Returns:
        Coordinate mapping suitable for exact layout comparison.
    """
    dim_set = set(dims)
    return {
        str(name): coord
        for name, coord in data.coords.items()
        if coord.dims and set(coord.dims).issubset(dim_set)
    }


def _require_exact_layout(
    reference: xr.DataArray,
    candidate: xr.DataArray,
    dims: tuple[str, ...],
    *,
    label: str,
) -> None:
    """Require equal sizes and exact coordinates over selected dimensions.

    Args:
        reference: Reference design matrix.
        candidate: Design matrix to compare.
        dims: Dimensions whose sizes and coordinates must match.
        label: Human-readable layout label for errors.

    Raises:
        ValueError: If sizes or labelled coordinates differ.
    """
    if any(reference.sizes[dim] != candidate.sizes[dim] for dim in dims):
        raise ValueError(f"Flux plan terms must have exact {label} sizes.")

    reference_coords = _coords_for_dims(reference, dims)
    candidate_coords = _coords_for_dims(candidate, dims)
    if reference_coords.keys() != candidate_coords.keys():
        raise ValueError(f"Flux plan terms must have exact {label} coordinates.")
    if any(not coord.identical(candidate_coords[name]) for name, coord in reference_coords.items()):
        raise ValueError(f"Flux plan terms must have exact {label} coordinates.")


def _preflight_state_priors(
    plan: _FluxPlan,
    activities: Mapping[str, ResolvedStateActivity],
    registry: CoordRegistry,
) -> None:
    """Validate state prior metadata in isolated scratch models.

    Args:
        plan: Flux plan whose priors should be checked.
        activities: Resolved activity contract keyed by semantic state ID.
        registry: Validated global coordinate registry for the term designs.

    Raises:
        ValueError: If a prior cannot be constructed with its state dimensions.
    """
    coords = {name: np.asarray(values).tolist() for name, values in registry.pymc_coords.items()}
    with pm.Model(coords=coords, model=None) as model:
        attach_coord_registry(model, CoordRegistry())
        for state in plan.states:
            try:
                add_state_vector(
                    activities[state.state_id],
                    prior_args=dict(state.prior_args),
                    var_name=state.variable_name,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Flux plan prior for state {state.state_id!r} is invalid.") from exc


def _validate_flux_plan(plan: _FluxPlan) -> dict[str, ResolvedStateActivity]:
    """Validate a complete flux plan without mutating the active PyMC model.

    Args:
        plan: Ordered state and forward-term plan.

    Returns:
        Resolved state activities keyed by semantic state ID.

    Raises:
        TypeError: If a term design is not an xarray ``DataArray``.
        ValueError: If identifiers, names, designs, coordinates, priors, state
            references, or coefficients violate the compiler contract.
    """
    if not plan.states:
        raise ValueError("Flux plans require at least one state.")
    if not plan.terms:
        raise ValueError("Flux plans require at least one forward term.")

    _require_unique([state.state_id for state in plan.states], "state IDs")
    _require_unique([term.term_id for term in plan.terms], "term IDs")
    _require_unique([state.variable_name for state in plan.states], "variable names")
    _require_unique([term.data_name for term in plan.terms], "data names")
    _require_unique([term.deterministic_name for term in plan.terms], "deterministic names")

    state_ids = {state.state_id for state in plan.states}
    terms_by_state = {
        state.state_id: [term for term in plan.terms if term.state_id == state.state_id]
        for state in plan.states
    }
    unknown_state_ids = sorted({term.state_id for term in plan.terms} - state_ids)
    if unknown_state_ids:
        raise ValueError(f"Flux terms reference unknown state IDs: {unknown_state_ids!r}.")
    unused_state_ids = sorted(state_id for state_id, terms in terms_by_state.items() if not terms)
    if unused_state_ids:
        raise ValueError(f"Flux states have no forward terms: {unused_state_ids!r}.")

    total_collisions = [term for term in plan.terms if term.deterministic_name == plan.total_name]
    compatible_single_term = len(plan.terms) == 1 and len(total_collisions) == 1
    if total_collisions and not compatible_single_term:
        raise ValueError(
            "Flux total name collides with a term deterministic name; "
            "this is only valid for a single-term compatibility plan."
        )

    registry = CoordRegistry()
    reference_observations: xr.DataArray | None = None
    for term in plan.terms:
        if not isinstance(term.design, xr.DataArray):
            raise TypeError(f"Flux term {term.term_id!r} design must be an xarray DataArray.")
        if term.design.ndim != 2 or plan.observation_dim not in term.design.dims:
            raise ValueError(
                f"Flux term {term.term_id!r} design must be two-dimensional and include "
                f"{plan.observation_dim!r}."
            )
        missing_coords = [str(dim) for dim in term.design.dims if dim not in term.design.coords]
        if missing_coords:
            raise ValueError(
                f"Flux term {term.term_id!r} design requires dimension coordinates for {missing_coords!r}."
            )
        if isinstance(term.coefficient, bool) or not isinstance(term.coefficient, Real):
            raise ValueError(f"Flux term {term.term_id!r} coefficient must be a finite scalar.")
        if not math.isfinite(float(term.coefficient)):
            raise ValueError(f"Flux term {term.term_id!r} coefficient must be a finite scalar.")

        if reference_observations is None:
            reference_observations = term.design
        else:
            _require_exact_layout(
                reference_observations,
                term.design,
                (plan.observation_dim,),
                label="observation",
            )
        try:
            registry.add(
                term.design.coords,
                model_dims=tuple(str(dim) for dim in term.design.dims),
            )
        except ValueError as exc:
            raise ValueError("Flux plan terms have incompatible global coordinates.") from exc

    for state_id, terms in terms_by_state.items():
        reference = terms[0].design
        state_dims = tuple(str(dim) for dim in reference.dims if dim != plan.observation_dim)
        for term in terms[1:]:
            candidate_dims = tuple(str(dim) for dim in term.design.dims if dim != plan.observation_dim)
            if candidate_dims != state_dims:
                raise ValueError(f"Flux terms for state {state_id!r} must have exact state dimensions.")
            _require_exact_layout(reference, term.design, state_dims, label="state")

    activities = {
        state.state_id: resolve_state_activity(
            detect_zero_sensitivity(
                _aggregate_state_design(
                    terms_by_state[state.state_id],
                    observation_dim=plan.observation_dim,
                ),
                output_dim=plan.observation_dim,
            ),
            state.state_activity,
        )
        for state in plan.states
    }
    backend_names = [
        name for state in plan.states for name in _state_backend_names(state, activities[state.state_id])
    ]
    backend_names.extend(term.data_name for term in plan.terms)
    backend_names.extend(
        term.deterministic_name
        for term in plan.terms
        if not (compatible_single_term and term.deterministic_name == plan.total_name)
    )
    backend_names.append(plan.total_name)
    _require_unique(backend_names, "backend names")

    _preflight_state_priors(plan, activities, registry)
    return activities


def _compile_loop_sum(plan: _FluxPlan) -> _CompiledFlux:
    """Validate and compile a loop-sum plan into the active PyMC model.

    Args:
        plan: Ordered state and forward-term plan.

    Returns:
        Compiled total mean and semantic state/term tensor mappings.

    Raises:
        TypeError: If a term design is not an xarray ``DataArray``.
        ValueError: If the complete plan is invalid.
    """
    activities = _validate_flux_plan(plan)

    state_tensors: dict[str, TensorVariable] = {}
    latent_tensors: dict[str, TensorVariable | None] = {}
    term_tensors: dict[str, TensorVariable] = {}
    ordered_outputs: list[TensorVariable] = []
    state_by_id = {state.state_id: state for state in plan.states}

    for term in plan.terms:
        design = term.design.transpose(plan.observation_dim, ...)
        data = add_model_data(design, term.data_name)
        if term.state_id not in state_tensors:
            state = state_by_id[term.state_id]
            state_vector = add_state_vector(
                activities[state.state_id],
                prior_args=dict(state.prior_args),
                var_name=state.variable_name,
            )
            state_tensors[state.state_id] = state_vector.state
            latent_tensors[state.state_id] = state_vector.latent

        output = pt.dot(data, state_tensors[term.state_id])
        if term.coefficient != 1.0:
            output = term.coefficient * output
        deterministic = pm.Deterministic(
            term.deterministic_name,
            output,
            dims=plan.observation_dim,
        )
        term_tensors[term.term_id] = deterministic
        ordered_outputs.append(deterministic)

    if len(ordered_outputs) == 1 and plan.terms[0].deterministic_name == plan.total_name:
        total_mu = ordered_outputs[0]
    else:
        total_mu = pm.Deterministic(
            plan.total_name,
            cast(Any, pt.stack(ordered_outputs, axis=0)).sum(axis=0),
            dims=plan.observation_dim,
        )

    return _CompiledFlux(
        mu=total_mu,
        states=MappingProxyType(state_tensors),
        latents=MappingProxyType(latent_tensors),
        terms=MappingProxyType(term_tensors),
    )
