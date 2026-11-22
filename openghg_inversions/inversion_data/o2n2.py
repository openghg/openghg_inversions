"""Declared delta(O2/N2) observables for linked CO2/O2 inversions.

A linked preparation whose O2 rows are delta(O2/N2) rather than O2 mole
fractions declares that observable in its provenance under
:data:`O2_OBSERVABLE_PROVENANCE_KEY`. The declaration records the per-meg
definition and the reference ratio it is relative to::

    delta(O2/N2) = 1e6 * ((x_O2 / x_N2) / reference_ratio - 1)   [per meg]

The reference ratio belongs to the O2 and N2 products used to build the rows
(for example 0.20946/0.78084 for CORSO TM5 curtains), so it is always declared
with its source and never defaulted here. Rows and covariance blocks must
already be in per meg; this module does not transform observations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any, Literal

O2_OBSERVABLE_PROVENANCE_KEY = "o2_observable"
MOLE_FRACTION = "mole_fraction"
DELTA_O2_N2 = "delta_o2_n2"
O2Observable = Literal["mole_fraction", "delta_o2_n2"]
O2_OBSERVABLES: tuple[str, ...] = (MOLE_FRACTION, DELTA_O2_N2)


def is_per_meg_unit(unit: object) -> bool:
    """Return whether a unit label names per meg (``"per meg"``, ``"per_meg"``, ``"permeg"``)."""
    label = str(unit).casefold()
    return "".join(character for character in label if character.isalnum()) == "permeg"


def is_ratio_anomaly_unit(unit: object) -> bool:
    """Return whether a unit label is a ratio anomaly (per meg, per mil, delta) rather than a mole fraction.

    Unit parsers read ``"per meg"`` as 1e-6 mol/mol and ``"per mil"`` or ``"‰"`` as
    1e-3 mol/mol, so mole-fraction checks must reject these labels explicitly
    before converting units. The check covers any tracer reported as a ratio
    anomaly (for example delta(O2/N2) in per meg or Delta14CO2 in per mil).
    """
    label = str(unit).casefold()
    compact = "".join(character for character in label if character.isalnum())
    return (
        "meg" in label
        or "delta" in label
        or "\u2030" in label
        or compact in {"permil", "permill", "permille"}
    )


@dataclass(frozen=True)
class DeltaO2N2Declaration:
    """A declared delta(O2/N2) O2 channel in per meg.

    Attributes:
        reference_ratio: Positive O2/N2 mole ratio that delta is relative to.
        reference_ratio_source: Where the reference ratio comes from, such as
            the boundary product it matches.
        units: Per-meg label used by the O2 rows.
        n2_policy: Optional description of how N2 was supplied (for example
            ``"curtain"``, ``"station"`` or ``"constant"``).
    """

    reference_ratio: float
    reference_ratio_source: str
    units: str = "per meg"
    n2_policy: str | None = None

    def __post_init__(self) -> None:
        ratio = self.reference_ratio
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, int | float)
            or not math.isfinite(ratio)
            or ratio <= 0
        ):
            raise ValueError("delta(O2/N2) reference_ratio must be a finite positive number.")
        object.__setattr__(self, "reference_ratio", float(ratio))
        if not isinstance(self.reference_ratio_source, str) or not self.reference_ratio_source.strip():
            raise ValueError("delta(O2/N2) reference_ratio_source must be a non-empty string.")
        if not isinstance(self.units, str) or not is_per_meg_unit(self.units):
            raise ValueError(f"delta(O2/N2) units must be per meg, not {self.units!r}.")
        if self.n2_policy is not None and (not isinstance(self.n2_policy, str) or not self.n2_policy.strip()):
            raise ValueError("delta(O2/N2) n2_policy must be a non-empty string when given.")

    def to_provenance(self) -> dict[str, Any]:
        """Return the JSON record stored under :data:`O2_OBSERVABLE_PROVENANCE_KEY`."""
        record: dict[str, Any] = {
            "kind": DELTA_O2_N2,
            "reference_ratio": self.reference_ratio,
            "reference_ratio_source": self.reference_ratio_source,
            "units": self.units,
        }
        if self.n2_policy is not None:
            record["n2_policy"] = self.n2_policy
        return record

    @classmethod
    def from_provenance(cls, provenance: Mapping[str, Any]) -> DeltaO2N2Declaration | None:
        """Read a declaration from preparation provenance.

        Returns:
            The declaration, or ``None`` when the provenance declares no O2
            observable or declares O2 mole fractions.

        Raises:
            ValueError: If the record is not a mapping, names an unknown kind,
                has unknown fields or has invalid values.
        """
        record = provenance.get(O2_OBSERVABLE_PROVENANCE_KEY)
        if record is None:
            return None
        if not isinstance(record, Mapping):
            raise ValueError(f"Provenance {O2_OBSERVABLE_PROVENANCE_KEY!r} must be a JSON object.")
        options = dict(record)
        kind = options.pop("kind", None)
        if kind == MOLE_FRACTION and not options:
            return None
        if kind != DELTA_O2_N2:
            raise ValueError(
                f"Provenance {O2_OBSERVABLE_PROVENANCE_KEY!r} kind must be one of {O2_OBSERVABLES}."
            )
        unknown = set(options) - {"reference_ratio", "reference_ratio_source", "units", "n2_policy"}
        if unknown:
            raise ValueError(f"Unknown delta(O2/N2) declaration fields: {sorted(unknown)}.")
        try:
            return cls(**options)
        except TypeError as exc:
            raise ValueError(
                "delta(O2/N2) declarations require reference_ratio and reference_ratio_source."
            ) from exc


__all__ = [
    "O2_OBSERVABLE_PROVENANCE_KEY",
    "MOLE_FRACTION",
    "DELTA_O2_N2",
    "O2Observable",
    "O2_OBSERVABLES",
    "DeltaO2N2Declaration",
    "is_per_meg_unit",
    "is_ratio_anomaly_unit",
]
