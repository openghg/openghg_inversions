"""Validate mole-fraction units and derive their scale relative to ``mol/mol``.

The parser is shared by surface preparation, boundary handling and reporting.
``mole_fraction_unit_scale`` parses an OpenGHG unit expression relative to
``mol/mol`` and raises ``ValueError`` for invalid or incompatible units; it
does not mutate datasets or perform cross-site conversion.

OpenGHG's unit registry defines ``per meg`` as 1e-6 and ``per mil`` (``‰``) as
1e-3, so a ratio anomaly such as delta(O2/N2) or Delta14CO2 would otherwise
parse as a dimensionless mole-fraction scale. ``mole_fraction_unit_scale``
rejects those labels (see :func:`is_ratio_anomaly_unit`) before parsing.
Pre-transformed per-meg delta(O2/N2) artifacts are accepted only through the
explicit :class:`~openghg_inversions.inversion_data.o2n2.DeltaO2N2Declaration`
path, which never treats them as mole fractions.
"""

from openghg.util import (
    cf_ureg,  # pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
)

# Per mille, per ten thousand and (casefolded) delta symbols.
_RATIO_ANOMALY_SYMBOLS = ("‰", "‱", "δ")
_RATIO_ANOMALY_COMPACT = frozenset({"permil", "permill", "permille", "permyriad"})


def is_ratio_anomaly_unit(unit: object) -> bool:
    """Return whether a unit label is a ratio anomaly rather than a mole fraction.

    Recognized labels, ignoring case, spaces and punctuation:

    - per meg: any label containing ``meg``, such as ``"per meg"``,
      ``"per_meg"`` and ``"permeg"``;
    - per mille: ``"per mil"``, ``"permil"``, ``"per mill"``, ``"per mille"``
      and ``"‰"``;
    - per myriad (per ten thousand): ``"per myriad"`` and ``"‱"``;
    - delta: any label containing ``delta``, ``"δ"`` or ``"Δ"``.

    Unit parsers read ``"per meg"`` as 1e-6 mol/mol and ``"per mil"`` as 1e-3
    mol/mol, so mole-fraction checks must reject these labels before
    converting units.
    """
    label = str(unit).casefold()
    compact = "".join(character for character in label if character.isalnum())
    return (
        "meg" in compact
        or "delta" in compact
        or any(symbol in label for symbol in _RATIO_ANOMALY_SYMBOLS)
        or compact in _RATIO_ANOMALY_COMPACT
    )


def mole_fraction_unit_scale(raw_units: str, *, context: str) -> float:
    """Return a mole-fraction unit's multiplicative scale against mol/mol.

    Args:
        raw_units: Mole-fraction units understood by OpenGHG's unit registry.
        context: Description included in validation errors.

    Returns:
        The numeric scale relative to ``mol/mol``.

    Raises:
        ValueError: If the units are a ratio anomaly such as per meg or per
            mil (see :func:`is_ratio_anomaly_unit`), or cannot be converted to
            ``mol/mol``.
    """
    if is_ratio_anomaly_unit(raw_units):
        raise ValueError(
            f"Cannot use ratio anomaly units {raw_units!r} for {context} as a mole fraction. "
            "Declare per-meg delta(O2/N2) O2 with DeltaO2N2Declaration instead of converting it to mol/mol."
        )
    try:
        quantity = cf_ureg.parse_expression(raw_units)
        if not hasattr(quantity, "to"):
            return float(quantity)
        return float(quantity.to("mol/mol").magnitude)
    except Exception as exc:
        raise ValueError(
            f"Could not convert observation units {raw_units!r} for {context} to mol/mol."
        ) from exc
