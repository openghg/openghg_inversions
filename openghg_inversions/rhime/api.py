"""Select a concrete scientific recipe without changing its input contract."""

from __future__ import annotations

from typing import Any


def run(*, model: str = "standard", **kwargs: Any) -> Any:
    """Run the selected model with its ordinary recipe arguments.

    Args:
        model: ``standard`` (default), ``multisector``, ``nested``, ``co2``,
            ``co2_cached_sigma``, ``co2_o2`` or ``co2_o2_cached_sigma``.
        **kwargs: Forwarded unchanged to the selected recipe. Standard,
            multisector and nested accept their existing INI/Python inputs.
            CO2-family recipes require their respective ``prepared_inputs``;
            selection does not add acquisition or checkpoint routes.

    Returns:
        The selected recipe's existing result: a result record for standard,
        multisector or nested, or the CO2-family inference DataTree.

    Raises:
        ValueError: If the model name is unsupported.
        TypeError: If arguments do not match the selected recipe's interface.

    Established ``run_rhime*`` imports remain available through the 0.9
    compatibility cycle. See the concrete recipe for scientific options,
    ownership and output behavior.
    """
    if model == "standard":
        from .standard import run_standard

        return run_standard(**kwargs)
    if model == "multisector":
        from .multisector import run_multisector

        return run_multisector(**kwargs)
    if model == "nested":
        from .nested import run_nested

        return run_nested(**kwargs)
    if model == "co2":
        from .co2 import run_rhime_co2

        return run_rhime_co2(**kwargs)
    if model == "co2_cached_sigma":
        from .co2 import run_rhime_co2_cached_sigma

        return run_rhime_co2_cached_sigma(**kwargs)
    if model == "co2_o2":
        from .co2 import run_rhime_co2_o2_from_prepared_inputs

        return run_rhime_co2_o2_from_prepared_inputs(**kwargs)
    if model == "co2_o2_cached_sigma":
        from .co2 import run_rhime_co2_o2_cached_sigma_from_prepared_inputs

        return run_rhime_co2_o2_cached_sigma_from_prepared_inputs(**kwargs)
    raise ValueError(f"Unknown model {model!r}.")
