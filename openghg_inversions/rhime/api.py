"""Select a concrete scientific recipe without changing its input contract."""

from __future__ import annotations

from typing import Any


def run(*, model: str = "standard", **kwargs: Any) -> Any:
    """Run the selected model with its ordinary recipe arguments.

    Args:
        model: ``standard`` (default), ``multisector``, ``nested``, ``co2``,
            ``co2_o2`` or ``co2_o2_cached_sigma``.
        **kwargs: Forwarded unchanged to the selected recipe. Standard,
            multisector and nested accept their existing INI/Python inputs.
            CO2 accepts its configured prepared-input product request.
            Linked CO2/O2 recipes require ``prepared_inputs``;
            selection does not add acquisition or checkpoint routes.

    Returns:
        The selected recipe's scientific result. Standard and multisector return
        InversionOutput, CO2 returns Co2Output, nested returns NestedOutput
        with two native output views, and linked CO2/O2 recipes return an inference DataTree.

    Raises:
        ValueError: If the model name is unsupported.
        TypeError: If arguments do not match the selected recipe's interface.

    See the concrete recipe for scientific options, ownership and output
    behavior. Standard, multisector and nested callers may supply ``prepared`` to skip preparation.
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
        from .co2 import run_co2

        return run_co2(**kwargs)
    if model == "co2_o2":
        from .co2 import run_rhime_co2_o2_from_prepared_inputs

        return run_rhime_co2_o2_from_prepared_inputs(**kwargs)
    if model == "co2_o2_cached_sigma":
        from .co2 import run_rhime_co2_o2_cached_sigma_from_prepared_inputs

        return run_rhime_co2_o2_cached_sigma_from_prepared_inputs(**kwargs)
    raise ValueError(f"Unknown model {model!r}.")
