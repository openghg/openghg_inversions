"""Persisted model structure, with options owned by their scientific components."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Literal

import xarray as xr
from pydantic import Field, JsonValue

from openghg_inversions._record_schema import Record
from openghg_inversions.models.additive_sigma import AdditiveSigmaRecord
from openghg_inversions.models.fixed_error import FixedErrorRecord
from openghg_inversions.models.pollution_event import PollutionEventRecord
from openghg_inversions.models.priors import PriorArgs, PriorRecord

if TYPE_CHECKING:
    from .specs import RhimeModelSpec, SectorSpec

Name = Annotated[str, Field(min_length=1, pattern=r"\S")]


class SectorRecord(Record):
    name: Name
    flux_source: Name
    x_prior: PriorRecord | None
    variable_suffix: str

    @classmethod
    def from_runtime(cls, sector: SectorSpec, *, prior: PriorArgs | None) -> SectorRecord:
        return cls.model_validate({
            "name": sector.name, "flux_source": sector.flux_source,
            "variable_suffix": sector.variable_suffix, "x_prior": prior,
        })

    def to_runtime(self, *, trace: xr.DataTree | None = None) -> SectorSpec:
        from .specs import SectorSpec

        return SectorSpec(
            **self.model_dump(exclude={"x_prior"}),
            x_prior=None if self.x_prior is None else self.x_prior.to_runtime(trace=trace),
        )

    def for_postprocessing(self, *, trace: xr.DataTree | None = None) -> SectorSpec:
        """Retain descriptive priors while checking their registered data references."""
        from .specs import SectorSpec

        return SectorSpec(
            **self.model_dump(exclude={"x_prior"}),
            x_prior=None if self.x_prior is None else self.x_prior.for_postprocessing(trace=trace),
        )


class ModelRecord(Record):
    """Required structural choices; each component owns its metadata conversion."""

    species: Name
    domain: Name
    sectors: Annotated[list[SectorRecord], Field(min_length=1)]
    use_bc: bool
    add_offset: bool
    aggregation_error_mode: Literal["auto", "none", "dense", "low_rank", "diagonal"]
    bc_prior: PriorRecord | None
    offset_prior: PriorRecord | None
    offset_args: dict[str, JsonValue] | None
    likelihood: Annotated[
        FixedErrorRecord | AdditiveSigmaRecord | PollutionEventRecord,
        Field(discriminator="kind"),
    ] | None

    @classmethod
    def from_runtime(cls, model: RhimeModelSpec, *, priors: Mapping[str, PriorArgs]) -> ModelRecord:
        """Capture only constructed priors; never inspect unused caller arrays."""
        return cls.model_validate({
            "species": model.species, "domain": model.domain, "use_bc": model.use_bc,
            "add_offset": model.add_offset, "aggregation_error_mode": model.aggregation_error_mode,
            "offset_args": model.offset_args,
            "bc_prior": priors.get("bc"), "offset_prior": priors.get("offset_latent"),
            "sectors": [SectorRecord.from_runtime(
                sector, prior=priors.get("x" if len(model.sectors) == 1 and "x" in priors
                                         else f"x_{sector.variable_suffix}"),
            ) for sector in model.sectors],
            "likelihood": None if model.likelihood is None else model.likelihood.to_record(priors=priors),
        })

    def to_runtime(self, *, trace: xr.DataTree | None = None) -> RhimeModelSpec:
        """Restore executable prior values; symbolic dependencies need the original recipe."""
        from .specs import RhimeModelSpec

        return RhimeModelSpec(
            **self.model_dump(exclude={"sectors", "likelihood", "bc_prior", "offset_prior"}),
            sectors=tuple(sector.to_runtime(trace=trace) for sector in self.sectors),
            bc_prior=None if self.bc_prior is None else self.bc_prior.to_runtime(trace=trace),
            offset_prior=None if self.offset_prior is None else self.offset_prior.to_runtime(trace=trace),
            likelihood=None if self.likelihood is None else self.likelihood.to_runtime(trace=trace),
        )

    def for_postprocessing(self, *, trace: xr.DataTree | None = None) -> RhimeModelSpec:
        """Build a metadata view for saved-state products, retaining prior descriptions.

        The returned specification describes the sampled model. Its prior
        dictionaries may contain symbolic descriptors and are not an executable
        model recipe; use ``to_runtime`` when executable restoration is needed.
        """
        from .specs import RhimeModelSpec

        return RhimeModelSpec(
            **self.model_dump(exclude={"sectors", "likelihood", "bc_prior", "offset_prior"}),
            sectors=tuple(sector.for_postprocessing(trace=trace) for sector in self.sectors),
            bc_prior=None if self.bc_prior is None else self.bc_prior.for_postprocessing(trace=trace),
            offset_prior=None if self.offset_prior is None else self.offset_prior.for_postprocessing(trace=trace),
            likelihood=None if self.likelihood is None else self.likelihood.for_postprocessing(trace=trace),
        )
