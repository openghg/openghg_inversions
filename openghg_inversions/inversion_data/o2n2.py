"""Declarations for δ(O2/N2) observables.

A δ(O2/N2) observable is the per-meg anomaly of the O2/N2 mole ratio from a
declared reference ratio::

    delta(O2/N2) = 1e6 * ((x_O2 / x_N2) / reference_ratio - 1)   [per meg]

The reference ratio belongs to the O2 and N2 products used to build the rows
(for example 0.20946/0.78084 for CORSO TM5 curtains), so it is always declared
with its source and never defaulted here. Rows and covariance blocks must
already be in per meg; this module does not transform observations, and
per-meg values are never converted to mol/mol.

A missing reference ratio is an error: whoever built the rows used some ratio
and must declare it. The declaration may also record the observations'
calibration scale, taken from the observation metadata or the data provider.

A linked CO2/O2 preparation whose O2 rows are δ(O2/N2) rather than O2 mole
fractions stores its :class:`DeltaO2N2Declaration` in preparation provenance
under :data:`O2_OBSERVABLE_PROVENANCE_KEY`. The declaration can arrive either
in that provenance or as attributes on the O2 observations (see
:meth:`DeltaO2N2Declaration.from_attrs`); preparation stores it in provenance,
so later steps never depend on xarray attributes surviving.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from ._units import is_ratio_anomaly_unit

O2_OBSERVABLE_PROVENANCE_KEY = "o2_observable"
MOLE_FRACTION = "mole_fraction"
DELTA_O2_N2 = "delta_o2_n2"
O2_OBSERVABLES: tuple[str, ...] = (MOLE_FRACTION, DELTA_O2_N2)
#: O2 observation attribute naming the observable.
O2_OBSERVABLE_ATTR = "observable"
#: Prefix of O2 observation attributes holding declaration fields.
DELTA_O2_N2_ATTR_PREFIX = "delta_o2_n2_"
_DECLARATION_FIELDS = ("reference_ratio", "reference_ratio_source", "n2_policy", "scale")


def is_per_meg_unit(unit: object) -> bool:
    """Return whether a unit label names per meg (``"per meg"``, ``"per_meg"``, ``"permeg"``)."""
    label = str(unit).casefold()
    return "".join(character for character in label if character.isalnum()) == "permeg"


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
        scale: Optional calibration scale of the observations, from their
            metadata or the data provider.
    """

    reference_ratio: float
    reference_ratio_source: str
    units: str = "per meg"
    n2_policy: str | None = None
    scale: str | None = None

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
        for name in ("n2_policy", "scale"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"delta(O2/N2) {name} must be a non-empty string when given.")

    def to_provenance(self) -> dict[str, Any]:
        """Return the JSON record stored under :data:`O2_OBSERVABLE_PROVENANCE_KEY`."""
        record: dict[str, Any] = {
            "kind": DELTA_O2_N2,
            "reference_ratio": self.reference_ratio,
            "reference_ratio_source": self.reference_ratio_source,
            "units": self.units,
        }
        for name in ("n2_policy", "scale"):
            if getattr(self, name) is not None:
                record[name] = getattr(self, name)
        return record

    @classmethod
    def from_attrs(cls, attrs: Mapping[str, Any], *, units: str) -> DeltaO2N2Declaration | None:
        """Read a declaration from attributes on the O2 observations.

        The observable is named by ``observable`` and the declaration fields
        use the ``delta_o2_n2_`` prefix: ``delta_o2_n2_reference_ratio``,
        ``delta_o2_n2_reference_ratio_source`` and optionally
        ``delta_o2_n2_n2_policy`` and ``delta_o2_n2_scale``. Units come from
        the channel's declared units, not from an attribute.

        Returns:
            The declaration, or ``None`` when the attributes declare no O2
            observable, or declare O2 mole fractions without delta fields.

        Raises:
            ValueError: If delta fields appear without ``observable =
                "delta_o2_n2"``, the observable is unknown, a field is unknown
                or missing (including the reference ratio), or a value is
                invalid.
        """
        observable = attrs.get(O2_OBSERVABLE_ATTR)
        fields = {
            str(name).removeprefix(DELTA_O2_N2_ATTR_PREFIX): value
            for name, value in attrs.items()
            if str(name).startswith(DELTA_O2_N2_ATTR_PREFIX)
        }
        if observable is None and not fields:
            return None
        if observable == MOLE_FRACTION:
            if fields:
                raise ValueError(
                    f"O2 observations with observable 'mole_fraction' must not carry delta(O2/N2) "
                    f"attributes; found {sorted(fields)}."
                )
            return None
        if observable != DELTA_O2_N2:
            raise ValueError(
                f"O2 observations with delta(O2/N2) attributes need {O2_OBSERVABLE_ATTR}="
                f"{DELTA_O2_N2!r}; the observable must be one of {O2_OBSERVABLES}, not {observable!r}."
            )
        unknown = set(fields) - set(_DECLARATION_FIELDS)
        if unknown:
            raise ValueError(f"Unknown delta(O2/N2) attributes: {sorted(DELTA_O2_N2_ATTR_PREFIX + n for n in unknown)}.")
        if "reference_ratio" not in fields or "reference_ratio_source" not in fields:
            raise ValueError(
                "delta(O2/N2) O2 observations must declare delta_o2_n2_reference_ratio and "
                "delta_o2_n2_reference_ratio_source; the reference ratio is never defaulted."
            )
        ratio = fields["reference_ratio"]
        if hasattr(ratio, "item") and not isinstance(ratio, bool):
            fields["reference_ratio"] = ratio.item()  # NetCDF reads scalar attributes as NumPy values.
        return cls(units=units, **fields)

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
        if kind == MOLE_FRACTION:
            if options:
                raise ValueError(
                    f"Provenance {O2_OBSERVABLE_PROVENANCE_KEY!r} kind 'mole_fraction' takes no other "
                    f"fields; found {sorted(options)}."
                )
            return None
        if kind != DELTA_O2_N2:
            raise ValueError(
                f"Provenance {O2_OBSERVABLE_PROVENANCE_KEY!r} kind must be one of {O2_OBSERVABLES}."
            )
        unknown = set(options) - {*_DECLARATION_FIELDS, "units"}
        if unknown:
            raise ValueError(f"Unknown delta(O2/N2) declaration fields: {sorted(unknown)}.")
        try:
            return cls(**options)
        except TypeError as exc:
            raise ValueError(
                "delta(O2/N2) declarations require reference_ratio and reference_ratio_source; "
                "the reference ratio is never defaulted."
            ) from exc


__all__ = [
    "O2_OBSERVABLE_PROVENANCE_KEY",
    "MOLE_FRACTION",
    "DELTA_O2_N2",
    "O2_OBSERVABLES",
    "O2_OBSERVABLE_ATTR",
    "DELTA_O2_N2_ATTR_PREFIX",
    "DeltaO2N2Declaration",
    "is_per_meg_unit",
    "is_ratio_anomaly_unit",
]
