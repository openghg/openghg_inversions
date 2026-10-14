"""Prototype schemas for small model records, separate from runtime arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field, JsonValue

from openghg_inversions._record_schema import Record
from openghg_inversions._serialization_codecs import _json_value

if TYPE_CHECKING:
    from .specs import RhimeModelSpec

Name = Annotated[str, Field(min_length=1, pattern=r"\S")]


class PriorDataReference(Record):
    """An array already persisted in the trace's constant_data group."""

    data: Name


type PriorRecord = dict[str, str | bool | int | float | PriorDataReference]


class SectorRecord(Record):
    name: Name
    flux_source: Name
    x_prior: PriorRecord | None
    variable_suffix: str


class FixedErrorOptions(Record):
    pass


class AdditiveSigmaOptions(Record):
    sigma_prior: PriorRecord | None
    sigma_freq: str | None
    sigma_per_site: bool
    sigma_freq_anchor: str | None
    use_minimum_error_floor: bool


class PollutionEventOptions(Record):
    sigma_prior: PriorRecord | None
    sigma_freq: str | None
    sigma_per_site: bool
    sigma_freq_anchor: str | None
    pollution_events_from_obs: bool
    power: PriorRecord | float | int | None


class FixedErrorRecord(Record):
    kind: Literal["fixed_error"]
    options: FixedErrorOptions


class AdditiveSigmaRecord(Record):
    kind: Literal["additive_sigma"]
    options: AdditiveSigmaOptions


class PollutionEventRecord(Record):
    kind: Literal["pollution_event"]
    options: PollutionEventOptions


class ModelRecord(Record):
    """Required structural choices; numerical arrays have named references."""

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
    def from_runtime(cls, model: RhimeModelSpec, *, priors: dict[str, Any]) -> ModelRecord:
        """Capture only constructed priors; never inspect unused caller arrays."""
        from dataclasses import fields

        from .specs import AdditiveSigmaSettings, FixedErrorSettings, PollutionEventSettings

        likelihood = None
        if model.likelihood is not None:
            kind = {FixedErrorSettings: "fixed_error", AdditiveSigmaSettings: "additive_sigma",
                    PollutionEventSettings: "pollution_event"}[type(model.likelihood)]
            options = {field.name: getattr(model.likelihood, field.name)
                       for field in fields(model.likelihood) if field.name != "sigma_prior"}
            if not isinstance(model.likelihood, FixedErrorSettings):
                options["sigma_prior"] = priors.get("sigma")
                options["sigma_freq_anchor"] = _json_value(options["sigma_freq_anchor"])
            if isinstance(options.get("power"), dict):
                options["power"] = priors.get("power")
            likelihood = {"kind": kind, "options": options}
        return cls.model_validate({
            **{name: getattr(model, name) for name in
               ("species", "domain", "use_bc", "add_offset", "aggregation_error_mode", "offset_args")},
            "bc_prior": priors.get("bc"), "offset_prior": priors.get("offset_latent"),
            "sectors": [{"name": sector.name, "flux_source": sector.flux_source,
                         "x_prior": priors.get("x" if len(model.sectors) == 1 and "x" in priors
                                               else f"x_{sector.variable_suffix}"),
                         "variable_suffix": sector.variable_suffix} for sector in model.sectors],
            "likelihood": likelihood,
        })

    def to_runtime(self, *, trace: Any = None) -> RhimeModelSpec:
        """Resolve references with the scientific owner, without building a graph."""
        from .specs import (
            AdditiveSigmaSettings, FixedErrorSettings, PollutionEventSettings, RhimeModelSpec, SectorSpec,
        )

        def resolve(value: Any) -> Any:
            if isinstance(value, dict):
                if set(value) == {"data"}:
                    name = value["data"]
                    if trace is None or "constant_data" not in trace or name not in trace["constant_data"]:
                        raise ValueError(f"Missing registered prior data {name!r}.")
                    return trace["constant_data"][name]
                return {key: resolve(item) for key, item in value.items()}
            return value

        values = self.model_dump()
        values["sectors"] = tuple(SectorSpec(**resolve(sector)) for sector in values["sectors"])
        for name in ("bc_prior", "offset_prior"):
            values[name] = resolve(values[name])
        if self.likelihood is not None:
            settings_type = {"fixed_error": FixedErrorSettings, "additive_sigma": AdditiveSigmaSettings,
                             "pollution_event": PollutionEventSettings}[self.likelihood.kind]
            values["likelihood"] = settings_type(**resolve(self.likelihood.options.model_dump()))
        return RhimeModelSpec(**values)
