"""Versioned standard handoffs: small validated JSON records, separate from arrays."""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from openghg_inversions._record_schema import JsonValue, Record, SchemaVersion
from openghg_inversions.postprocessing.contracts import OutputContractRecord

from ._model_records import ModelRecord


ArtifactReference = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]


class StandardManifest(Record):
    """The producer and family are required evidence in every handoff."""

    schema_version: SchemaVersion
    producer: Literal["openghg_inversions"]
    family: Literal["standard"]


class PrepareManifest(StandardManifest):
    """Prepared-input references and external options; DataConfig owns resolution."""

    schema_name: Literal["openghg_inversions.standard.prepare"] = Field(alias="schema")
    stage: Literal["prepare"]
    prepared_inputs: ArtifactReference
    data: dict[str, JsonValue]
    basis_artifact_source: str | None


class SampleManifest(StandardManifest):
    """Samples and the saved scientific contracts needed for graph-free replay."""

    schema_name: Literal["openghg_inversions.standard.sample"] = Field(alias="schema")
    stage: Literal["sample"]
    preparation: ArtifactReference
    posterior: ArtifactReference
    output_contract: OutputContractRecord
    model: ModelRecord
    inference_kind: Literal["mcmc"]
    build_and_sample_seconds: Annotated[float, Field(ge=0)]
    actual_inference: dict[str, JsonValue] | None = None
    provenance: dict[str, JsonValue] = Field(default_factory=dict)
