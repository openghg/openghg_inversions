"""Msgspec alternative for the persisted metadata boundaries in issue #785.

This deliberately mirrors the Pydantic prototype's readable JSON, including
#873's references into trace ``constant_data``. No numerical codec is replaced.
Use ``convert_record``, ``decode_record`` and ``encode_record`` at boundaries:
ordinary Struct construction does not check annotated types. JSON-only ``Any``
fields remain open in generated schemas; the shared finite-JSON check owns them.

Run ``python -m openghg_inversions.experimental.metadata_records MANIFEST`` to
validate and round-trip an actual standard manifest, or add ``--schema`` to
print its JSON Schema. Install the ``schema-prototype`` extra first.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

import msgspec

from openghg_inversions._serialization_codecs import _copy_json_mapping, _json_value


NonBlank = Annotated[str, msgspec.Meta(min_length=1, pattern=r"\S")]
OutputFormat = Literal["none", "inv_out", "basic", "paris", "legacy"]


class Record(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    """Typed metadata only; validation happens at an explicit boundary."""


class PriorDataReference(Record):
    """Name of a numerical parameter already owned by trace constant_data."""

    data: NonBlank


Prior = dict[str, str | bool | int | float | PriorDataReference]


class FixedErrorOptions(Record):
    """Reported observation error has no latent mismatch options."""


class SigmaOptions(Record):
    """Options shared by the two mismatch recipes, without array payloads."""

    sigma_prior: Prior | None
    sigma_freq: str | None
    sigma_per_site: bool
    sigma_freq_anchor: str | None


class AdditiveSigmaOptions(SigmaOptions):
    use_minimum_error_floor: bool


class PollutionEventOptions(SigmaOptions):
    pollution_events_from_obs: bool
    power: Prior | float | int | None


class FixedErrorRecord(Record, tag="fixed_error", tag_field="kind"):
    options: FixedErrorOptions


class AdditiveSigmaRecord(Record, tag="additive_sigma", tag_field="kind"):
    options: AdditiveSigmaOptions


class PollutionEventRecord(Record, tag="pollution_event", tag_field="kind"):
    options: PollutionEventOptions


LikelihoodRecord = FixedErrorRecord | AdditiveSigmaRecord | PollutionEventRecord


class SectorRecord(Record):
    name: NonBlank
    flux_source: NonBlank
    x_prior: Prior | None
    variable_suffix: str


class ModelRecord(Record):
    species: NonBlank
    domain: NonBlank
    sectors: Annotated[list[SectorRecord], msgspec.Meta(min_length=1)]
    use_bc: bool
    likelihood: LikelihoodRecord | None
    add_offset: bool
    aggregation_error_mode: Literal["auto", "none", "dense", "low_rank", "diagonal"]
    bc_prior: Prior | None
    offset_prior: Prior | None
    offset_args: dict[str, Any] | None


class OutputContractRecord(Record):
    schema: Literal["openghg_inversions.output_contract"]
    schema_version: Literal[1]
    variable_roles: Annotated[dict[NonBlank, NonBlank], msgspec.Meta(min_length=1)]
    supported_output_formats: Annotated[
        list[OutputFormat], msgspec.Meta(extra_json_schema={"contains": {"const": "none"}})
    ]
    metadata: dict[str, Any]
    state_dimension_mapping: Annotated[
        dict[Literal["trace", "basis"], NonBlank],
        msgspec.Meta(extra_json_schema={
            "propertyNames": {"enum": ["trace", "basis"]},
            "oneOf": [{"maxProperties": 0}, {"required": ["trace", "basis"]}],
        }),
    ]

    def __post_init__(self) -> None:
        if "none" not in self.supported_output_formats:
            raise ValueError("supported_output_formats must include 'none'")
        if self.state_dimension_mapping and set(self.state_dimension_mapping) != {"trace", "basis"}:
            raise ValueError("state_dimension_mapping requires both 'trace' and 'basis'")


class PreparationRecord(Record):
    schema: Literal["openghg_inversions.standard.prepare"]
    schema_version: Literal[1]
    producer: Literal["openghg_inversions"]
    family: Literal["standard"]
    stage: Literal["prepare"]
    prepared_inputs: NonBlank
    data: dict[str, Any]
    basis_artifact_source: str | None


class SampleRecord(Record):
    schema: Literal["openghg_inversions.standard.sample"]
    schema_version: Literal[1]
    producer: Literal["openghg_inversions"]
    family: Literal["standard"]
    stage: Literal["sample"]
    preparation: NonBlank
    posterior: NonBlank
    output_contract: OutputContractRecord
    model: ModelRecord
    inference_kind: Literal["mcmc"]
    build_and_sample_seconds: Annotated[float, msgspec.Meta(ge=0)]
    actual_inference: dict[str, Any] | None = None
    provenance: dict[str, Any] = {}


_MANIFEST_TYPES = {
    "openghg_inversions.standard.prepare": PreparationRecord,
    "openghg_inversions.standard.sample": SampleRecord,
}
RecordT = TypeVar("RecordT", bound=Record)


def convert_record(value: Mapping[str, Any], record_type: type[RecordT]) -> RecordT:
    """Project metadata scalars, reject nonfinite values, then validate strictly.

    The existing metadata projection gives NumPy scalars and temporal values
    the same wire meaning as the public writers. Xarray and Dask inputs are
    rejected here; numerical priors must arrive as #873 data references.
    """
    finite = _copy_json_mapping(_json_value(value), "metadata record")
    # Fixed-type msgspec conversion otherwise supplies an omitted union tag.
    tag_field = record_type.__struct_config__.tag_field
    if tag_field is not None and tag_field not in finite:
        raise msgspec.ValidationError(f"Object missing required field `{tag_field}`")
    return msgspec.convert(finite, type=record_type, strict=True)


def decode_record(
    encoded: str | bytes, record_type: type[RecordT] | None = None,
) -> RecordT | PreparationRecord | SampleRecord:
    """Decode JSON with required schema fields and finite open metadata."""
    value = msgspec.json.decode(encoded)
    if record_type is None:
        try:
            record_type = _MANIFEST_TYPES[value["schema"]]
        except (KeyError, TypeError):
            raise msgspec.ValidationError("Missing or unsupported standard manifest `schema`") from None
    return convert_record(value, record_type)


def encode_record(record: Record) -> bytes:
    """Revalidate a constructed Struct and encode finite, readable JSON.

    Msgspec otherwise encodes NaN and infinity as null. Checking before encoding
    is required to avoid that information loss, including in ``Any`` fields.
    """
    validated = convert_record(msgspec.to_builtins(record), type(record))
    return msgspec.json.encode(validated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--schema", action="store_true")
    args = parser.parse_args()
    record = decode_record(args.manifest.read_bytes())
    encoded = msgspec.json.encode(msgspec.json.schema(type(record))) if args.schema else encode_record(record)
    print(msgspec.json.format(encoded, indent=2).decode())


if __name__ == "__main__":
    main()
