"""Decode fitted boundary facts and reconstruct graph-free scientific quantities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json

import numpy as np
import xarray as xr

from openghg_inversions.postprocessing.inversion_output import trace_group


_QUANTITY_ROLES = {
    "concentration": "boundary_concentration",
    "scale": "boundary_scale",
    "mean_shift": "boundary_mean_shift",
    "anomaly": "boundary_correction_anomaly",
    "correction": "boundary_correction",
    "reference": "boundary_reference_concentration",
    "mean_contribution": "boundary_mean_shift_concentration",
    "anomaly_contribution": "boundary_anomaly_concentration",
}


@dataclass(frozen=True)
class BoundaryOutputFacts:
    """Validated current fitted names and optional additive zero-anomaly policy."""

    roles: Mapping[str, str]
    constants: xr.Dataset
    weights: xr.DataArray | None = None
    zero_anomaly: bool = False


def decode_boundary_facts(
    trace: xr.DataTree, *, roles: Mapping[str, str] | None = None,
    channels: tuple[str, ...] = ("",),
) -> dict[str, BoundaryOutputFacts]:
    """Validate saved boundary facts once at the scientific replay boundary.

    Small fitted scale/weight facts are materialized here. Draws stay borrowed;
    no preparation, model construction or calendar calculation is performed.
    Earlier internal posterior formats require preparation and sampling again.
    """
    if roles is None:
        roles = json.loads(trace.attrs.get("rhime_variable_roles", "{}"))
    if not isinstance(roles, Mapping) or any(not isinstance(name, str) or not name for name in roles.values()):
        raise ValueError("Saved boundary roles must map scientific quantities to nonempty variable names.")
    constants = trace_group(trace, "constant_data") if "constant_data" in trace.children else xr.Dataset()
    result = {}
    for channel in channels:
        prefix = f"{channel}_" if channel else ""
        selected = {quantity: roles[prefix + role] for quantity, role in _QUANTITY_ROLES.items()
                    if prefix + role in roles}
        scale_role = prefix + "boundary_anomaly_scale"
        weights_role = prefix + "boundary_centering_weights"
        weighted_policy = scale_role in roles or weights_role in roles
        if not selected and not weighted_policy:
            continue
        version = trace.attrs.get("rhime_boundary_format_version")
        if isinstance(version, bool) or not isinstance(version, (int, np.integer)) or version != 1:
            raise ValueError("Unsupported saved boundary format; prepare again and repeat sampling.")
        weights = None
        zero_anomaly = False
        if weighted_policy:
            if scale_role not in roles or weights_role not in roles:
                raise ValueError("Weighted boundary anomalies require both scale and centering-weight roles.")
            if "correction" not in selected:
                raise ValueError("Weighted boundary anomalies require the boundary_correction role.")
            scale_name, weights_name = roles[scale_role], roles[weights_role]
            if scale_name not in constants or weights_name not in constants:
                raise ValueError("Boundary anomalies require saved scale and weights; prepare again.")
            facts = constants[[scale_name, weights_name]].compute()
            scale, weights = facts[scale_name], facts[weights_name]
            if (scale.ndim != 0 or not np.isfinite(scale.item()) or scale.item() < 0
                    or weights.ndim != 1 or weights.size == 0
                    or not np.isfinite(weights.values).all() or not (weights.values > 0).all()):
                raise ValueError("Saved zero boundary anomaly policy conflicts with its scale or weights.")
            zero_anomaly = scale.item() == 0 or weights.size == 1
            anomaly_roles = {"anomaly", "anomaly_contribution"} & selected.keys()
            if zero_anomaly and anomaly_roles:
                raise ValueError("Saved zero boundary anomaly policy conflicts with declared anomaly roles.")
            if not zero_anomaly and len(anomaly_roles) != 2:
                raise ValueError("Active boundary anomalies require saved sampled state and concentration contributions.")
        result[channel] = BoundaryOutputFacts(selected, constants, weights, zero_anomaly)
    return result


def reconstruct_boundary(
    facts: BoundaryOutputFacts, draws: xr.Dataset,
) -> dict[str, xr.DataArray]:
    """Return stable scientific quantities, including omitted zero anomalies.

    The supplied draw group determines sample labels and sizes independently
    for prior and posterior. Constants broadcast only over this group's draws.
    """
    result = {}
    for quantity, name in facts.roles.items():
        if name in draws:
            result[quantity] = draws[name]
        elif name in facts.constants:
            result[quantity] = facts.constants[name]
        else:
            raise ValueError(f"Boundary scientific outputs require role {_QUANTITY_ROLES[quantity]!r} in saved draws.")
    if facts.weights is not None:
        correction = result["correction"]
        xr.align(correction, facts.weights, join="exact", copy=False)
        if facts.zero_anomaly:
            result.setdefault("anomaly", xr.zeros_like(correction))
            if "mean_contribution" not in result:
                raise ValueError("Zero boundary anomaly reconstruction requires the saved mean contribution.")
            result.setdefault("anomaly_contribution", xr.zeros_like(result["mean_contribution"]))
        elif "anomaly" not in result or "anomaly_contribution" not in result:
            raise ValueError("Active boundary anomalies require saved sampled state and concentration contributions.")
    if "reference" in result and "concentration" in result:
        result["reference"] = result["reference"].broadcast_like(result["concentration"])
    return result
