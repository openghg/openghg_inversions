"""Assemble explicit scientific fixtures for the one-boundary API."""

from openghg_inversions.array_ops import select_gathered_data_array
from openghg_inversions.models.boundary import prepare_boundary
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


def bound_linked_model_arguments(options):
    """Bind each native channel once, preserving all other fixture arguments."""
    boundaries = {}
    for channel in ("co2", "o2"):
        sensitivity = options[f"{channel}_sensitivity"]
        native = select_gathered_data_array(
            options["observations"],
            key=channel,
            key_dim="species",
            ragged_dim="channel_observation",
            stack_dim="observation",
        ).rename({"observation": str(sensitivity.dims[0])})
        if "observation_units" in native.coords:
            native = native.assign_attrs(units=str(native.observation_units.values[0]))
        choices = {
            "bc_frequency" if name == "bc_freq" else name: (
                options.get(name)
                if name in {"bc_freq", "bc_anchor_time"}
                else (options.get(name) or {}).get(channel)
            )
            for name in BOUNDARY_OPTIONS
        }
        if choices["bc_mean_shift_prior"] is None and choices["bc_anomaly_scale"] is None:
            choices["bc_centering_weights"] = None
        if (
            choices["boundary_sensitivity"] is not None or choices["boundary_operands"] is not None
        ) and choices["bc_mean_shift_prior"] is None and choices["bc_anomaly_scale"] is None:
            if choices["bc_prior"] is None:
                choices["bc_prior"] = dict(DEFAULT_BC_PRIOR)
        boundary = prepare_boundary(native, **choices)
        if boundary is not None:
            boundaries[channel] = boundary
    return {
        **{name: value for name, value in options.items() if name not in BOUNDARY_OPTIONS},
        "boundary": boundaries,
    }


def bound_model_arguments(options):
    """Bind one single-channel scientific fixture before model construction."""
    choices = {
        "bc_frequency" if name == "bc_freq" else name: options[name]
        for name in BOUNDARY_OPTIONS
        if name in options
    }
    if (
        choices.get("boundary_sensitivity") is not None or choices.get("boundary_operands") is not None
    ) and choices.get("bc_mean_shift_prior") is None and choices.get("bc_anomaly_scale") is None:
        if choices.get("bc_prior") is None:
            choices["bc_prior"] = dict(DEFAULT_BC_PRIOR)
    return {
        **{name: value for name, value in options.items() if name not in BOUNDARY_OPTIONS},
        "boundary": prepare_boundary(options["observations"], **choices),
    }
