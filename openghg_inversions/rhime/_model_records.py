"""Saved model meaning, independent of executable configuration and its defaults."""

from __future__ import annotations

from typing import Annotated, Literal, TYPE_CHECKING, Any

from pydantic import Field, model_validator

from openghg_inversions._record_schema import JsonFloat, JsonValue, Record
from openghg_inversions.models._prior_records import Name, PriorDescription
from openghg_inversions.observation_error import AggregationErrorMode

if TYPE_CHECKING:
    import xarray as xr

    from openghg_inversions.postprocessing.contracts import OutputContract
    from .params import DataConfig


class SectorDescription(Record):
    name: Name
    variable_suffix: str
    x_prior: PriorDescription | None
    # Standard preparation owns its single source; multisector records retain routing.
    flux_source: Name | None = None


class SigmaDescription(Record):
    sigma_prior: PriorDescription | None
    sigma_freq: str | None
    sigma_per_site: bool
    sigma_freq_anchor: str | None


class PollutionEventDescription(SigmaDescription):
    pollution_events_from_obs: bool
    power: PriorDescription | JsonFloat | None


class AdditiveSigmaDescription(SigmaDescription):
    use_minimum_error_floor: bool


class FixedErrorDescription(Record):
    pass


class PollutionEventRecord(Record):
    kind: Literal["pollution_event"]
    options: PollutionEventDescription


class AdditiveSigmaRecord(Record):
    kind: Literal["additive_sigma"]
    options: AdditiveSigmaDescription


class FixedErrorRecord(Record):
    kind: Literal["fixed_error"]
    options: FixedErrorDescription


class ModelDescription(Record):
    """Required saved choices and nullable descriptions of constructed priors.

    ``None`` records an unused prior; it never asks for current defaults.
    Symbolic descriptors remain metadata. This value cannot build a model.
    Preparation retains authority over species, domain and available sources.
    """

    sectors: Annotated[list[SectorDescription], Field(min_length=1)]
    use_bc: bool
    add_offset: bool
    aggregation_error_mode: AggregationErrorMode
    bc_prior: PriorDescription | None
    offset_prior: PriorDescription | None
    offset_args: dict[str, JsonValue] | None
    likelihood: Annotated[
        PollutionEventRecord | AdditiveSigmaRecord | FixedErrorRecord,
        Field(discriminator="kind"),
    ] | None

    @model_validator(mode="after")
    def unique_sector_identities(self) -> ModelDescription:
        for field in ("name", "variable_suffix"):
            values = [getattr(sector, field) for sector in self.sectors]
            if len(set(values)) != len(values):
                raise ValueError(f"Saved model sector {field} values must be unique.")
        return self

    def to_output_metadata(
        self, *, data_config: DataConfig, trace: xr.DataTree, output_contract: OutputContract,
    ) -> dict[str, Any]:
        """Bind saved descriptions to preparation and trace without executable replay."""
        from openghg_inversions.models.priors import trace_prior_records

        if len(self.sectors) != len(data_config.flux_sources):
            raise ValueError("Saved model sector count conflicts with its preparation.")
        if data_config.split_by_sectors:
            sources = [sector.flux_source for sector in self.sectors]
            if len(set(sources)) != len(sources) or set(sources) != set(data_config.flux_sources):
                raise ValueError("Saved model sources conflict with its preparation.")
        else:
            if any(sector.flux_source is not None for sector in self.sectors):
                raise ValueError("Standard model sources belong to its preparation.")
            sources = list(data_config.flux_sources)
        if self.use_bc and not data_config.use_bc:
            raise ValueError("Saved model requires unavailable prepared BC inputs.")

        descriptions = [self.bc_prior, self.offset_prior, *(sector.x_prior for sector in self.sectors)]
        if self.likelihood is not None:
            options = self.likelihood.options
            if isinstance(options, SigmaDescription):
                descriptions.append(options.sigma_prior)
            if isinstance(options, PollutionEventDescription) and isinstance(options.power, PriorDescription):
                descriptions.append(options.power)
        for description in descriptions:
            if description is not None:
                description.validate_references(trace)

        priors = trace_prior_records(trace)
        metadata = self.model_dump(mode="json")
        metadata.update(
            species=data_config.species, domain=data_config.domain, constructed_priors=priors,
            state_activity=None, bc_state_activity=None,
            likelihood=None if self.likelihood is None else self.likelihood.options.model_dump(mode="json"),
        )
        metadata["bc_prior"] = priors.get("bc")
        metadata["offset_prior"] = priors.get("offset_latent")
        if self.likelihood is not None and isinstance(self.likelihood.options, SigmaDescription):
            metadata["likelihood"]["sigma_prior"] = priors.get("sigma")
        if self.likelihood is not None and isinstance(self.likelihood.options, PollutionEventDescription):
            if "power" in priors or isinstance(self.likelihood.options.power, PriorDescription):
                metadata["likelihood"]["power"] = priors.get("power")
        for sector, source in zip(metadata["sectors"], sources):
            sector.update(flux_source=source, state_activity=None)
            role = "flux_scale" if len(self.sectors) == 1 else f"flux_scale:{sector['name']}"
            prior_name = "x" if len(self.sectors) == 1 and "x" in priors else f"x_{sector['variable_suffix']}"
            sector["x_prior"] = priors.get(prior_name)
            variable = output_contract.variable_roles.get(role)
            if variable in priors:
                sector["x_prior"] = priors[variable]
        return metadata
