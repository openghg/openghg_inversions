"""Typed JSON records owned by the ordinary scientific checkpoint workflows.

Numerical payloads, scientific restoration and filesystem publication retain
their existing owners. These models describe the readable completion markers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from pydantic import BeforeValidator, Field, TypeAdapter, ValidationError

from openghg_inversions._record_schema import JsonValue, Record, SchemaVersion
from openghg_inversions.postprocessing.contracts import OutputContract
from ._model_records import ModelDescription


NonemptyString = Annotated[str, Field(min_length=1)]
JsonObject = dict[str, JsonValue]


class _Envelope(Record):
    schema_version: SchemaVersion
    producer: Literal["openghg_inversions"]


class _Preparation(_Envelope):
    stage: Literal["prepare"]
    generation: NonemptyString
    prepared_inputs: NonemptyString


class _ConfiguredPreparation(_Preparation):
    data: JsonObject


class StandardPreparationRecord(_ConfiguredPreparation):
    """Standard preparation choices and their writer-assigned generation."""

    schema_name: Literal["openghg_inversions.standard.prepare"] = Field(alias="schema")


class MultisectorPreparationRecord(_ConfiguredPreparation):
    """Multisector preparation choices and their writer-assigned generation."""

    schema_name: Literal["openghg_inversions.multisector.prepare"] = Field(alias="schema")


class _Sample(_Envelope):
    stage: Literal["sample"]
    preparation: NonemptyString
    preparation_generation: NonemptyString
    posterior: NonemptyString


class _ConfiguredSample(_Sample):
    output_contract: OutputContract
    model: ModelDescription
    actual_inference: JsonObject
    provenance: JsonObject


class StandardSampleRecord(_ConfiguredSample):
    """Standard scientific roles and sampling descriptions."""

    schema_name: Literal["openghg_inversions.standard.sample"] = Field(alias="schema")


class MultisectorSampleRecord(_ConfiguredSample):
    """Multisector scientific roles and sampling descriptions."""

    schema_name: Literal["openghg_inversions.multisector.sample"] = Field(alias="schema")


class _Co2Preparation(_Preparation):
    """Original requested CO2 facts and optional affine companion."""

    schema_name: Literal["openghg_inversions.co2.prepare"] = Field(alias="schema")
    start_date: NonemptyString
    end_date: NonemptyString
    domain: NonemptyString
    affine_reconstruction: NonemptyString | None = None


class Co2PreparationWriteRecord(_Co2Preparation):
    """New CO2 preparations require valid installed-version descriptions."""

    provenance: JsonObject


class _Co2Sample(_Sample):
    schema_name: Literal["openghg_inversions.co2.sample"] = Field(alias="schema")
    inference_kind: Literal["mcmc"]


class Co2SampleWriteRecord(_Co2Sample):
    """New CO2 completion markers require valid descriptive mappings."""

    actual_inference: JsonObject
    provenance: JsonObject


_json_object = TypeAdapter(JsonObject, config=Record.model_config)


def _available_description(value: Any) -> JsonObject:
    """Keep the historical CO2 policy of unavailable descriptive metadata."""
    try:
        return _json_object.validate_python(value)
    except ValidationError:
        return {}


AvailableDescription = Annotated[JsonObject, BeforeValidator(_available_description)]


class StandardSampleReadRecord(StandardSampleRecord):
    """Require the sampler field while permitting unavailable descriptions."""

    actual_inference: AvailableDescription


class MultisectorSampleReadRecord(MultisectorSampleRecord):
    """Require the sampler field while permitting unavailable descriptions."""

    actual_inference: AvailableDescription


class Co2PreparationRecord(_Co2Preparation):
    """Read original CO2 science while permitting unavailable provenance."""

    provenance: AvailableDescription = Field(default_factory=dict)


class Co2SampleReadRecord(_Co2Sample):
    """Read scientific facts strictly, retaining forgiving CO2 descriptions."""

    actual_inference: AvailableDescription = Field(default_factory=dict)
    provenance: AvailableDescription = Field(default_factory=dict)


RecordType = TypeVar("RecordType", bound=Record)


def checkpoint_dict(record: Record) -> dict[str, Any]:
    """Revalidate Python values before projecting a record to finite JSON."""
    checked = type(record).model_validate(record.model_dump(mode="python", by_alias=True, exclude_unset=True))
    return checked.model_dump(mode="json", by_alias=True, exclude_unset=True)


def read_checkpoint(path: str | Path, record_type: type[RecordType]) -> tuple[Path, RecordType]:
    """Read an ordinary version-one checkpoint with filename and field errors."""
    path = Path(path).resolve()
    try:
        record = record_type.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (ValidationError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} is not a valid {record_type.__name__} version-1 manifest: {exc}") from exc
    return path, record
