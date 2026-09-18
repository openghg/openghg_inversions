"""Derive numeric mol/mol scales at inversion-data output boundaries.

OpenGHG ``ModelScenario`` objects own dataset unit conversion. This module only
derives numeric scales for merged-data metadata and output formats.
``mole_fraction_unit_scale`` parses an OpenGHG unit expression relative to
``mol/mol`` and raises ``ValueError`` for invalid or incompatible units; it
does not mutate datasets or perform cross-site conversion.
"""

from openghg.util import (
    cf_ureg,  # pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
)


def mole_fraction_unit_scale(raw_units: str, *, context: str) -> float:
    """Return a mole-fraction unit's multiplicative scale against mol/mol.

    Args:
        raw_units: Mole-fraction units understood by OpenGHG's unit registry.
        context: Description included in validation errors.

    Returns:
        The numeric scale relative to ``mol/mol``.

    Raises:
        ValueError: If the units cannot be converted to ``mol/mol``.
    """
    try:
        quantity = cf_ureg.parse_expression(raw_units)
        if not hasattr(quantity, "to"):
            return float(quantity)
        return float(quantity.to("mol/mol").magnitude)
    except Exception as exc:
        raise ValueError(
            f"Could not convert observation units {raw_units!r} for {context} to mol/mol."
        ) from exc
