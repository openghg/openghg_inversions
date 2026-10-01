"""Durable output metadata remains usable without importing a model backend."""

import json
import subprocess
import sys

import pytest

from openghg_inversions.postprocessing.contracts import OutputContract


def test_output_contract_roundtrip_copies_roles_and_preserves_explicit_view() -> None:
    roles = {"flux_scale": "inner_scale"}
    contract = OutputContract(
        variable_roles=roles,
        supported_output_formats=("none", "paris"),
        metadata={"recipe": "nested", "source_order": ["fossil", "bio"]},
        state_dimension_mapping={"trace": "inner_region", "basis": "region"},
    )
    roles["flux_scale"] = "outer_scale"
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
    payload = OutputContract(variable_roles={"flux_scale": "x"}).to_dict()
    payload.update(change)
    with pytest.raises(ValueError):
        OutputContract.from_dict(payload)


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
