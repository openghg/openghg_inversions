"""Independent level and prescribed-departure/curtain boundary corrections.

Preparation retains labelled, potentially lazy operators. Model construction
and diagnostics are explicit eager boundaries. Widths are caller-owned;
changing rows, reference or frequency never recalibrates them.
"""

from __future__ import annotations

from typing import Literal
import json

import dask
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.linalg import helmert
import xarray as xr

from openghg_inversions.array_ops import to_dense
from openghg_inversions.inversion_inputs import DatetimeLike, make_freq_indicator
from openghg_inversions.inversion_data._units import mole_fraction_unit_scale
from openghg_inversions.models.additive_boundary import AdditiveBoundaryResult
from openghg_inversions.models.boundary import bind_boundary_operands, require_centered_precision
from openghg_inversions.models.components import ConcentrationTerm
from openghg_inversions.models.coords import add_model_data, component_coords

__all__ = [
    "prepare_boundary_correction",
    "materialize_boundary_correction",
    "add_boundary_correction",
    "correction_design",
    "calibrate_pattern_width",
    "functional_sd",
    "annotate_boundary_correction_trace",
]

SCHEMA = "independent-boundary-correction-v1"


def prepare_boundary_correction(
    operands: xr.Dataset,
    observations: xr.DataArray,
    *,
    method: Literal["level", "A", "curtain_contrasts", "independent_curtains"],
    level_sd: float | None = None,
    pattern_sd: float | None = None,
    reference: float | Literal["whole_window_transport_mean"] | None = None,
    frequency: str | None = None,
    anchor_time: DatetimeLike | None = None,
    level_by_period: bool = False,
) -> xr.Dataset:
    """Prepare a portable labelled Gaussian correction to prescribed physical BC.

    ``operands`` contains raw spatial ``H_bc_anomaly=A0``, dimensionless
    ``H_bc_mass=M`` and one scalar ``bc_reference=c0`` in observation units.
    Inputs are borrowed. Numerical payloads stay lazy, including whole-window
    reference selection. Period membership reads receptor observation times.

    ``method`` selects ``g delta``, ``g delta + A v``, ``g delta + M Q z``,
    or ``M u`` respectively, where ``g=M1`` and Q is equal-weight orthonormal
    Helmert. Level/curtain widths have concentration units; A widths are
    dimensionless. All coefficients are independent, unbounded zero-mean
    Gaussians. Positive widths must be explicit; no production defaults exist.
    Independent curtains omit delta. The existing weighted additive prior is
    a separate component and is unaffected.

    A supplied scalar reference is in observation units. The named policy uses
    ``c=c0+sum(A0)/sum(M)`` over all retained rows, equivalent to sum(b0)/sum(g)
    without constructing large absolute concentrations. Recentring uses
    ``A=A0+(c0-c)M`` in float64. It preserves b0 but changes the A prior.

    Frequency gates independent coefficients by receptor time, with an explicit
    inversion-start anchor. A global level is retained unless level_by_period
    selects period levels instead. The scalar c and widths remain unchanged.
    Save the returned Dataset with the package's MultiIndex serialization
    helpers; it contains the complete replay operators, widths and provenance.
    """
    if method not in {"level", "A", "curtain_contrasts", "independent_curtains"}:
        raise ValueError(f"Unknown boundary correction method {method!r}.")
    has_level = method != "independent_curtains"
    has_pattern = method != "level"
    for enabled, width, name in ((has_level, level_sd, "level_sd"), (has_pattern, pattern_sd, "pattern_sd")):
        if enabled and (width is None or not np.isfinite(width) or width <= 0):
            raise ValueError(f"{name} must be explicitly finite and positive for {method}.")
        if not enabled and width is not None:
            raise ValueError(f"{name} is unused for {method}; omit it.")
    if level_by_period and (frequency is None or not has_level):
        raise ValueError("level_by_period requires frequency and a level correction.")
    if frequency is not None and anchor_time is None:
        raise ValueError("Coefficient frequency requires an explicit inversion-start anchor_time.")
    if anchor_time is not None and (frequency is None or np.isnat(np.datetime64(anchor_time))):
        raise ValueError("anchor_time must be a valid inversion start with coefficient frequency.")
    row = str(observations.dims[0])
    if row not in {"nmeasure", "time"} and row in operands.dims:
        operands = operands.rename({row: "nmeasure"})
    bound = bind_boundary_operands(operands, observations)
    a0 = bound.H_bc_anomaly.transpose(row, "bc_region").astype(np.float64)
    m = bound.H_bc_mass.transpose(row, "bc_region").astype(np.float64)
    c0 = bound.bc_reference.astype(np.float64)
    if reference is None:
        c = c0
        policy = "stored_scalar"
    elif isinstance(reference, str):
        if reference != "whole_window_transport_mean":
            raise ValueError("reference must be a finite scalar or 'whole_window_transport_mean'.")
        c = c0 + a0.sum(skipna=False) / m.sum(skipna=False)
        policy = reference
    else:
        if np.ndim(reference) != 0 or not np.isfinite(reference):
            raise ValueError("reference must be a finite scalar or 'whole_window_transport_mean'.")
        c = xr.DataArray(float(reference))
        policy = "explicit_scalar"
    c = c.assign_attrs(c0.attrs)
    a = (a0 + (c0 - c) * m).assign_attrs(a0.attrs)
    g = m.sum("bc_region", skipna=False)
    period = (
        make_freq_indicator(observations.time, frequency, anchor_time=anchor_time)
        if frequency is not None
        else xr.zeros_like(observations, dtype=int)
    )
    periods = np.unique(period.data)
    indicator = period == xr.DataArray(
        periods, dims="bc_coefficient_period", coords={"bc_coefficient_period": periods}
    )
    level = (
        (indicator * g).rename(bc_coefficient_period="bc_level_term")
        if level_by_period
        else g.expand_dims(bc_level_term=[0], axis=1)
    )
    if not has_level:
        level = level.isel(bc_level_term=slice(0, 0))
    count = m.sizes["bc_region"]
    boundary_states = [str(label) for label in operands.get_index("bc_region")]
    if method == "A":
        operator = a
    elif method == "independent_curtains":
        operator = m
    elif method == "curtain_contrasts":
        q = xr.DataArray(
            helmert(count, full=False).T,
            dims=("bc_region", "bc_contrast"),
            coords={"bc_region": m.bc_region, "bc_contrast": np.arange(count - 1)},
        )
        operator = xr.dot(m, q, dim="bc_region")
    else:
        operator = a.isel(bc_region=slice(0, 0))
    state = next(dim for dim in operator.dims if dim != row)
    # Flatten period-major, retaining explicit support labels without a second MultiIndex.
    pattern = (indicator * operator).transpose(row, "bc_coefficient_period", state)
    labels = [f"p{p}:{j}" for p in periods for j in range(operator.sizes[state])]
    pattern = xr.DataArray(
        pattern.data.reshape((observations.size, len(labels))),
        dims=(row, "bc_pattern_term"),
        coords={**dict(observations.coords), "bc_pattern_term": labels},
    )
    if not has_pattern:
        pattern = pattern.isel(bc_pattern_term=slice(0, 0))
    units = c0.attrs["units"]
    result = xr.Dataset(
        {
            "bc_prescribed_centered": a.sum("bc_region", skipna=False).assign_attrs(units=units),
            "bc_reference_shift": (c * g).assign_attrs(units=units),
            "bc_reference": c,
            "bc_mass_response": g.assign_attrs(units="1"),
            "bc_level_operator": level.transpose(row, "bc_level_term").assign_attrs(units="1"),
            "bc_pattern_operator": pattern.assign_attrs(units=units if method == "A" else "1"),
            "bc_level_sd": xr.DataArray(level_sd or 0.0, attrs={"units": units}),
            "bc_pattern_sd": xr.DataArray(
                pattern_sd or 0.0, attrs={"units": "1" if method == "A" else units}
            ),
            "bc_period_index": period,
        },
        attrs={
            "boundary_correction_schema": SCHEMA,
            "method": method,
            "boundary_states": json.dumps(boundary_states),
            "contrast_basis": "scipy.linalg.helmert(K, full=False).T"
            if method == "curtain_contrasts"
            else "none",
            "reference_policy": policy,
            "coefficient_frequency": frequency or "whole_window",
            "coefficient_anchor": str(anchor_time) if anchor_time is not None else "whole_window",
            "level_support": "period" if level_by_period else "global" if has_level else "absent",
            "coefficient_distribution": "independent zero-mean unbounded Gaussian",
        },
    )
    if has_pattern:
        result = result.assign_coords(
            bc_pattern_period=("bc_pattern_term", np.repeat(periods, operator.sizes[state])),
            bc_pattern_state=(
                "bc_pattern_term",
                np.tile(
                    boundary_states
                    if method in {"A", "independent_curtains"}
                    else [f"Helmert{j + 1}" for j in range(count - 1)],
                    len(periods),
                ),
            ),
        )
    for kind in ("level", "pattern"):
        result.coords[f"bc_{kind}_term"].attrs.update(
            units=units if kind == "level" or method != "A" else "1",
            support="real",
            prior="independent zero-mean Gaussian",
            method=method,
            boundary_states=result.attrs["boundary_states"],
            contrast_basis=result.attrs["contrast_basis"],
            reference_policy=policy,
            coefficient_frequency=frequency or "whole_window",
            coefficient_anchor=str(anchor_time) if anchor_time is not None else "whole_window",
            concentration_units=units,
            level_support=result.attrs["level_support"],
        )
    return result


def materialize_boundary_correction(prepared: xr.Dataset, observations: xr.DataArray) -> xr.Dataset:
    """Validate a prepared/reloaded correction at its eager PyMC boundary.

    Row and coefficient indexes must be unique and exactly aligned. Numerical
    operands are jointly computed and densified without modifying the input.
    """
    require_centered_precision()
    if prepared.attrs.get("boundary_correction_schema") != SCHEMA:
        raise ValueError("Unsupported boundary correction schema.")
    row = str(observations.dims[0])
    dims = {
        "bc_prescribed_centered": (row,),
        "bc_reference_shift": (row,),
        "bc_reference": (),
        "bc_mass_response": (row,),
        "bc_level_operator": (row, "bc_level_term"),
        "bc_pattern_operator": (row, "bc_pattern_term"),
        "bc_level_sd": (),
        "bc_pattern_sd": (),
        "bc_period_index": (row,),
    }
    arrays = {name: prepared[name].transpose(*order) for name, order in dims.items()}
    for array in arrays.values():
        for dim in array.dims:
            if dim not in array.indexes or not array.indexes[dim].is_unique:
                raise ValueError(f"Boundary correction requires unique indexed {dim} labels.")
    selected = xr.Dataset(arrays, attrs=prepared.attrs)
    selected, _ = xr.align(selected, observations, join="exact", copy=False)
    units = observations.attrs.get("units")
    if "observation_units" in observations.coords:
        native_units = np.unique(observations.observation_units.data)
        if native_units.size != 1:
            raise ValueError("Prepare each boundary correction in one channel's concentration units.")
        units = str(native_units[0])
    if units is None:
        units = selected.bc_reference.attrs["units"]
    concentration_names = {
        "bc_prescribed_centered",
        "bc_reference_shift",
        "bc_reference",
        "bc_level_sd",
        "bc_pattern_operator" if selected.attrs["method"] == "A" else "bc_pattern_sd",
    }
    normalized = {}
    for name, value in selected.data_vars.items():
        if name == "bc_period_index":
            normalized[name] = value
            continue
        target = units if name in concentration_names else "1"
        scale = mole_fraction_unit_scale(value.attrs["units"], context=name) / mole_fraction_unit_scale(
            target,
            context="boundary correction observation units",
        )
        normalized[name] = (value * scale).assign_attrs({**value.attrs, "units": target})
    selected = xr.Dataset(normalized, attrs=selected.attrs)
    for kind in ("level", "pattern"):
        dim = f"bc_{kind}_term"
        selected = selected.assign_coords(
            {
                dim: selected[dim].assign_attrs(
                    concentration_units=units,
                    units=units if kind == "level" or selected.attrs["method"] != "A" else "1",
                )
            }
        )
    selected = dask.compute(selected.map(to_dense, keep_attrs=True))[0]
    if any(not np.isfinite(value.data).all() for value in selected.data_vars.values()):
        raise ValueError("Boundary correction operands must be finite.")
    for kind in ("level", "pattern"):
        width = float(selected[f"bc_{kind}_sd"])
        if width < 0 or (selected.sizes[f"bc_{kind}_term"] and width == 0):
            raise ValueError("Active boundary coefficients require positive widths.")
    return selected


def correction_design(prepared: xr.Dataset) -> np.ndarray:
    """Stack eager level and pattern operators for physical correction coefficients.

    Columns follow the concatenated level/pattern state axes. Widths are not
    included: multiply each column by its coefficient SD to obtain the factor
    for independent standard-normal coordinates.
    """
    return np.column_stack((prepared.bc_level_operator.data, prepared.bc_pattern_operator.data))


def add_boundary_correction(
    prepared: xr.Dataset,
    *,
    output_dim: str = "nmeasure",
    prefix: str = "",
    owner: str | None = None,
) -> AdditiveBoundaryResult:
    """Build physical level/pattern coefficients from an eager validated Dataset.

    Register all replay operands and retain prescribed, level, pattern and total
    absolute concentrations separately. No multiplicative ``bc`` state exists.
    Compose the returned concentration term using likelihood_representation.
    """
    data = {
        name: add_model_data(component_coords(value, owner, observation_dim=output_dim), f"{prefix}{name}")
        for name, value in prepared.data_vars.items()
    }
    coefficients, latents, signals = [], [], []
    for kind in ("level", "pattern"):
        dim = f"bc_{kind}_term"
        if prepared.sizes[dim]:
            registered_dim = str(
                component_coords(prepared[f"bc_{kind}_operator"], owner, observation_dim=output_dim).dims[1]
            )
            coefficient = pm.Normal(
                f"{prefix}bc_{kind}", mu=0.0, sigma=data[f"bc_{kind}_sd"], dims=registered_dim
            )
            coefficients.append(coefficient)
            latents.append(coefficient)
            signal = data[f"bc_{kind}_operator"] @ coefficient
        else:
            signal = pt.zeros((prepared.sizes[output_dim],))
        signals.append(pm.Deterministic(f"{prefix}mu_bc_{kind}", signal, dims=output_dim))
    centered = data["bc_prescribed_centered"] + signals[0] + signals[1]
    pm.Deterministic(
        f"{prefix}mu_bc_prescribed",
        data["bc_prescribed_centered"] + data["bc_reference_shift"],
        dims=output_dim,
    )
    output = pm.Deterministic(f"{prefix}mu_bc", centered + data["bc_reference_shift"], dims=output_dim)
    return AdditiveBoundaryResult(
        ConcentrationTerm(output, centered, prepared.bc_reference_shift),
        pt.concatenate(coefficients) if coefficients else pt.zeros((0,)),
        tuple(latents),
    )


def calibrate_pattern_width(operator: xr.DataArray, target_sd: float, *, coefficient_dim: str) -> float:
    """Eagerly translate a median pointwise SD target into one coefficient SD.

    This explicitly computes the operator; it never changes a prepared prior.
    The result is target_sd / median_i ||operator_i||. A zero median is
    unattainable and rejected. Inputs and target use compatible numeric units.
    """
    norm = to_dense(np.sqrt((operator.astype(np.float64) ** 2).sum(coefficient_dim))).compute()
    median = float(norm.median())
    if not np.isfinite(target_sd) or target_sd <= 0 or not np.isfinite(median) or median <= 0:
        raise ValueError("Calibration requires a positive finite target and median operator row norm.")
    return target_sd / median


def functional_sd(factor: xr.DataArray, functional: xr.DataArray, *, observation_dim: str) -> xr.DataArray:
    """Return exact SD ||l.T F||, retaining any labelled functional axes lazily.

    F includes the coefficient widths, with independent standard-normal inputs.
    Functional weights must cover exactly the same observation labels. A row
    mean uses weights 1/N; a temporal difference uses +1 and -1 at its two rows.
    """
    factor, functional = xr.align(factor, functional, join="exact", copy=False)
    projected = xr.dot(functional, factor, dim=observation_dim)
    coefficient_dim = next(dim for dim in factor.dims if dim != observation_dim)
    return np.sqrt((projected**2).sum(coefficient_dim))


def annotate_boundary_correction_trace(trace: xr.DataTree) -> xr.DataTree:
    """Attach physical units/support from restored coefficient coordinates.

    Call after restoring the model coordinate registry. Prepared operators and
    the scalar reference in constant_data suffice for labelled numerical replay.
    """
    metadata = {}
    for group in trace.children.values():
        for name, coordinate in group.coords.items():
            for suffix in ("bc_level_term", "bc_pattern_term"):
                if str(name).endswith(suffix) and "method" in coordinate.attrs:
                    metadata[str(name).removesuffix(suffix)] = coordinate.attrs
    for group in trace.children.values():
        for prefix, attributes in metadata.items():
            units = attributes["concentration_units"]
            pattern_units = "1" if attributes["method"] == "A" else units
            trace.attrs[f"{prefix}boundary_correction_schema"] = SCHEMA
            for kind in ("level", "pattern"):
                variable = f"{prefix}bc_{kind}"
                if variable in group:
                    group[variable].attrs.update(
                        units=units if kind == "level" else pattern_units,
                        support="real",
                        prior="independent zero-mean Gaussian",
                    )
            for suffix in (
                "mu_bc",
                "mu_bc_level",
                "mu_bc_pattern",
                "mu_bc_prescribed",
                "bc_prescribed_centered",
                "bc_reference_shift",
                "bc_reference",
                "bc_level_sd",
            ):
                if (variable := f"{prefix}{suffix}") in group:
                    group[variable].attrs["units"] = units
            for suffix in ("bc_pattern_sd", "bc_pattern_operator", "bc_level_operator", "bc_mass_response"):
                if (variable := f"{prefix}{suffix}") in group:
                    group[variable].attrs["units"] = (
                        pattern_units
                        if suffix == "bc_pattern_sd"
                        else units
                        if suffix == "bc_pattern_operator" and attributes["method"] == "A"
                        else "1"
                    )
    return trace


def _boundary_correction_roles(model: pm.Model, *, prefix: str = "") -> dict[str, str]:
    """Name the physical quantities exposed by an independent correction graph."""
    return {
        f"{prefix}{role}": f"{prefix}{variable}"
        for role, variable in {
            "boundary_concentration": "mu_bc",
            "prescribed_boundary_concentration": "mu_bc_prescribed",
            "boundary_level_correction": "mu_bc_level",
            "boundary_pattern_correction": "mu_bc_pattern",
            "boundary_level_coefficient": "bc_level",
            "boundary_pattern_coefficient": "bc_pattern",
        }.items()
        if f"{prefix}{variable}" in model
    }
