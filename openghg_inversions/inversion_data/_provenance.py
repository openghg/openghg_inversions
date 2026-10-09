"""Selected retrieval identities, independent of storage and scientific compatibility."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import xarray as xr

Identifier = str | int | tuple[str | int, ...]


@dataclass(frozen=True)
class InputProvenance:
    """Selected store, UUID and data version for one input or combined inputs.

    Missing identities remain ``"unknown"``. Tuples preserve the order of
    identifiers when retrieval combines several inputs.

    Attributes:
        store: Selected object-store identifier or identifiers.
        uuid: Selected OpenGHG object UUID or UUIDs.
        dataversion: Selected data version or versions, never a catalog guess.
    """

    store: Identifier = "unknown"
    uuid: Identifier = "unknown"
    dataversion: Identifier = "unknown"

    def __post_init__(self) -> None:
        if any(
            type(value) not in (str, int)
            and not (isinstance(value, tuple) and all(type(item) in (str, int) for item in value))
            for value in (self.store, self.uuid, self.dataversion)
        ):
            raise ValueError("Input provenance requires string or integer identifiers, or tuples of them.")


@dataclass(frozen=True)
class MergedDataProvenance:
    """OpenGHG software identity and selected inputs of a merged-data record.

    Input mappings are copied on construction; immutable input identities can
    be shared. These descriptive facts do not establish scientific compatibility.

    Attributes:
        openghg_version: OpenGHG package version used for retrieval, or unknown.
        openghg_commit: OpenGHG source revision used for retrieval, or unknown.
        observations: Observation identities keyed by retained site label.
        footprints: Footprint identities keyed by retained site label.
        flux: Flux identities keyed by source label.
        boundary: Boundary-condition identity, absent when no boundary was used.
    """

    openghg_version: str = "unknown"
    openghg_commit: str = "unknown"
    observations: Mapping[str, InputProvenance] = field(default_factory=dict)
    footprints: Mapping[str, InputProvenance] = field(default_factory=dict)
    flux: Mapping[str, InputProvenance] = field(default_factory=dict)
    boundary: InputProvenance | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.openghg_version, str) or not isinstance(self.openghg_commit, str):
            raise ValueError("OpenGHG provenance requires version and commit strings (or 'unknown').")
        for name in ("observations", "footprints", "flux"):
            values = dict(getattr(self, name))
            if any(not isinstance(label, str) or not isinstance(value, InputProvenance)
                   for label, value in values.items()):
                raise ValueError("Provenance mappings require labels and InputProvenance values.")
            object.__setattr__(self, name, values)
        if self.boundary is not None and not isinstance(self.boundary, InputProvenance):
            raise ValueError("Boundary provenance must be an InputProvenance value.")

    @classmethod
    def from_retrieval(
        cls,
        *,
        observations: Mapping[str, InputProvenance],
        footprints: Mapping[str, InputProvenance],
        flux: Mapping[str, InputProvenance],
        boundary: InputProvenance | None = None,
    ) -> MergedDataProvenance:
        """Record currently installed OpenGHG identity for a fresh retrieval."""
        import openghg

        return cls(
            openghg_version=str(getattr(openghg, "__version__", None) or "unknown"),
            openghg_commit=str(getattr(openghg, "__revisionid__", None) or "unknown"),
            observations=observations,
            footprints=footprints,
            flux=flux,
            boundary=boundary,
        )

    def retain_sites(self, sites: Iterable[str]) -> MergedDataProvenance:
        """Keep site identities in selection order, owning new input mappings."""
        sites = tuple(sites)
        return replace(
            self,
            observations={site: self.observations[site] for site in sites},
            footprints={site: self.footprints[site] for site in sites},
        )


def selected_provenance(
    value: Any, store: Any = None, *, requested_version: str | None = None
) -> InputProvenance:
    """Keep selected identifiers; infer latest only for a known retrieval request.

    OpenGHG typed wrappers can lose the generic object's selected ``_version``.
    A retrieval owner may supply ``requested_version="latest"`` when its getter
    guarantees a single UUID. Catalog ``latest_version`` alone does not identify
    arbitrary supplied data, which may come from an older version.
    """
    metadata = {} if isinstance(value, xr.Dataset) else getattr(value, "metadata", {})
    attrs = (
        value.attrs if isinstance(value, xr.Dataset) else getattr(getattr(value, "data", None), "attrs", {})
    )
    selected: dict[str, Identifier] = {}
    for name, aliases in {
        "store": ("store", "object_store"),
        "uuid": ("uuid", "UUID"),
        "dataversion": ("dataversion", "data_version"),
    }.items():
        result = next(
            (source[key] for source in (metadata, attrs) for key in aliases if source.get(key) is not None),
            None,
        )
        if name == "store" and result is None:
            result = store
        if result is None and name in {"uuid", "dataversion"}:
            result = getattr(value, "_uuid" if name == "uuid" else "_version", None)
        if name == "dataversion" and result is None and requested_version is not None:
            result = metadata.get("latest_version") if requested_version == "latest" else requested_version
        if isinstance(result, list | tuple) and all(type(item) in (str, int) for item in result):
            selected[name] = tuple(result)
        else:
            selected[name] = result if type(result) in (str, int) else "unknown"
    return InputProvenance(**selected)
