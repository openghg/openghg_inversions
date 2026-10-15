"""Durable output metadata remains usable without importing a model backend."""

import json
import subprocess
import sys
from types import MappingProxyType

import pytest
from pydantic import ValidationError

from openghg_inversions.postprocessing.contracts import OutputContract


def test_output_contract_roundtrip_copies_roles_and_preserves_explicit_view() -> None:
    roles = {"flux_scale": "inner_scale"}
    contract = OutputContract.create(
        variable_roles=roles,
        supported_output_formats=("none", "paris"),
        metadata={"recipe": "nested", "source_order": ["fossil", "bio"]},
        state_dimension_mapping={"trace": "inner_region", "basis": "region"},
    )
    roles["flux_scale"] = "outer_scale"
    assert json.loads(contract.model_dump_json()) == contract.to_dict()
    assert OutputContract.model_validate_json(contract.model_dump_json()) == contract
    loaded = OutputContract.from_dict(json.loads(json.dumps(contract.to_dict())))
    assert loaded == contract
    assert loaded.variable_roles["flux_scale"] == "inner_scale"
    loaded.validate_requested_output("paris")
    with pytest.raises(ValueError, match="does not declare"):
        loaded.validate_requested_output("legacy")


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"variable_roles": {"flux_scale": 5}},
        {"supported_output_formats": "none"},
        {"metadata": {"invalid": float("nan")}},
        {"state_dimension_mapping": {"trace": "region"}},
        {"unknown": "silently ignored"},
    ],
)
def test_output_contract_rejects_malformed_persisted_metadata(change: dict) -> None:
    payload = OutputContract.create(variable_roles={"flux_scale": "x"}).to_dict()
    payload.update(change)
    with pytest.raises(ValueError):
        OutputContract.from_dict(payload)


def test_output_contract_create_copies_mapping_inputs() -> None:
    roles = {"flux_scale": "x"}
    metadata = {"builder": {"sources": ["bio"]}}
    dimensions = {"trace": "inner_region", "basis": "region"}
    contract = OutputContract.create(
        variable_roles=MappingProxyType(roles), metadata=MappingProxyType(metadata),
        state_dimension_mapping=MappingProxyType(dimensions),
    )
    roles["flux_scale"] = "changed"
    metadata["builder"]["sources"].append("other")
    dimensions["trace"] = "changed"
    assert contract.variable_roles == {"flux_scale": "x"}
    assert contract.metadata == {"builder": {"sources": ["bio"]}}
    assert contract.state_dimension_mapping == {"trace": "inner_region", "basis": "region"}


@pytest.mark.parametrize("operation", ["to_dict", "with_updates"])
def test_output_contract_rejects_mutated_key_collision_before_json_projection(operation: str) -> None:
    contract = OutputContract.create(variable_roles={"flux_scale": "x"})
    collision = {1: "integer evidence", "1": "string evidence"}
    contract.metadata["collision"] = collision
    with pytest.raises(ValidationError) as caught:
        getattr(contract, operation)()
    assert caught.value.errors()[0]["loc"][:2] == ("metadata", "collision")
    assert collision == {1: "integer evidence", "1": "string evidence"}


def test_reconstruction_imports_no_backend_or_recipe_in_fresh_process() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from openghg_inversions.postprocessing.output_views import make_inversion_output; "
            "assert not {'pymc', 'pytensor', 'openghg_inversions.rhime'} & sys.modules.keys()",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
