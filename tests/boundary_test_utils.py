"""Assemble explicit scientific fixtures for the one-boundary API.

Older fixtures describe a boundary with flat ``bc_*`` options. These helpers
translate them once into a :class:`~openghg_inversions.models.boundary.Choice`
and bind it, so builder tests exercise the same choice -> bind -> add route as
the runners.
"""

import xarray as xr

from openghg_inversions.array_ops import select_gathered_data_array
from openghg_inversions.models.boundary import AdditiveChoice, MultiplicativeChoice, RawCenteredSensitivity
from openghg_inversions.rhime.specs import DEFAULT_BC_PRIOR


BOUNDARY_OPTIONS = (
    "boundary_sensitivity",
    "boundary_operands",
    "bc_prior",
    "bc_state_activity",
    "bc_freq",
    "bc_anchor_time",
    "bc_mean_shift_prior",
    "bc_centering_weights",
    "bc_anomaly_scale",
)


def raw_transport(dataset, weights=None):
    """Validate prepared centered operands, optionally adding raw centering weights."""
    data = dataset if weights is None else dataset.assign(bc_centering_weights=weights)
    return RawCenteredSensitivity.validate(data)


def centered_transport(anomaly, mass, reference, weights=None):
    """Assemble and validate raw centered transport from its three operands."""
    data = xr.Dataset(
        {"H_bc_anomaly": anomaly, "H_bc_mass": mass, "bc_reference": reference},
        attrs={"boundary_representation": "centered-v1"},
    )
    return raw_transport(data, weights=weights)


def boundary_choice(options, channel=None):
    """Return the choice described by flat fixture options, or None."""
    def get(name):
        if channel is None or name in {"bc_freq", "bc_anchor_time"}:
            return options.get(name)
        return (options.get(name) or {}).get(channel)

    periods = {"frequency": get("bc_freq"), "anchor_time": get("bc_anchor_time")}
    if get("bc_mean_shift_prior") is not None or get("bc_anomaly_scale") is not None:
        return AdditiveChoice(get("bc_mean_shift_prior"), get("bc_anomaly_scale"), **periods)
    if get("boundary_sensitivity") is None and get("boundary_operands") is None and get("bc_prior") is None:
        return None
    return MultiplicativeChoice(get("bc_prior") or dict(DEFAULT_BC_PRIOR), get("bc_state_activity"), **periods)


def boundary_transport(options, channel=None):
    """Return the absolute sensitivity or centered operands in fixture options."""
    def get(name):
        return options.get(name) if channel is None else (options.get(name) or {}).get(channel)

    operands = get("boundary_operands")
    if operands is not None:
        return raw_transport(operands, weights=get("bc_centering_weights"))
    sensitivity = get("boundary_sensitivity")
    if sensitivity is None:
        raise ValueError("A boundary choice requires boundary_sensitivity or centered operands.")
    return sensitivity


def bound_linked_model_arguments(options):
    """Bind each native channel once, preserving all other fixture arguments."""
    boundaries = {}
    for channel in ("co2", "o2"):
        choice = boundary_choice(options, channel)
        if choice is None:
            continue
        sensitivity = options[f"{channel}_sensitivity"]
        native = select_gathered_data_array(
            options["observations"],
            key=channel,
            key_dim="species",
            ragged_dim="channel_observation",
            stack_dim="observation",
        ).rename({"observation": str(sensitivity.dims[0])})
        boundaries[channel] = choice.bind(native, boundary_transport(options, channel))
    return {
        **{name: value for name, value in options.items() if name not in BOUNDARY_OPTIONS},
        "boundary": boundaries,
    }


def bound_model_arguments(options):
    """Bind one single-channel scientific fixture before model construction."""
    choice = boundary_choice(options)
    return {
        **{name: value for name, value in options.items() if name not in BOUNDARY_OPTIONS},
        "boundary": None if choice is None else choice.bind(options["observations"], boundary_transport(options)),
    }
