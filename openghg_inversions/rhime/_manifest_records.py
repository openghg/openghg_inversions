"""Versioned standard handoffs: small validated JSON records, separate from arrays."""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints

from openghg_inversions._record_schema import JsonFloat, JsonValue, Record, SchemaVersion
from openghg_inversions.postprocessing.contracts import OutputContract

from ._model_records import ModelRecord


ArtifactReference = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]


class StandardManifest(Record):
    """The producer and family are required evidence in every handoff."""

    schema_version: SchemaVersion
    producer: Literal["openghg_inversions"]
    family: Literal["standard"]
    stage: Literal["prepare", "sample"]

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load this manifest type and validate its complete saved JSON."""
        return cls.model_validate_json(Path(path).read_bytes())

    def save(self, directory: str | Path) -> Path:
        """Validate current metadata and atomically save the completion marker.

        The runner creates the directory and saves numerical artifacts first.
        """
        record = type(self).model_validate(self.model_dump(mode="json", by_alias=True))
        destination = Path(directory).resolve()
        temporary = destination / f".{record.stage}-manifest.tmp"
        temporary.write_text(record.model_dump_json(by_alias=True, indent=2) + "\n", encoding="utf-8")
        return temporary.replace(destination / f"{record.stage}-manifest.json")


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
    output_contract: OutputContract
    model: ModelRecord
    inference_kind: Literal["mcmc"]
    build_and_sample_seconds: Annotated[JsonFloat, Field(ge=0)]
    actual_inference: dict[str, JsonValue] | None = None
    provenance: dict[str, JsonValue] = Field(default_factory=dict)
