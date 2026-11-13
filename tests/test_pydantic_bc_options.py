"""Pydantic owns portable BC option fields without executing scientific work."""

from dataclasses import fields, replace
import json
from pathlib import Path

import dask.array as da
from dask import delayed
from dask.callbacks import Callback
from jsonschema import Draft202012Validator
import numpy as np
from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError
import pytest

from openghg_inversions.rhime.params import BcBasisOptions, DataConfig, RhimeConfig
from examples.pydantic_bc_options import run_example
from tests.test_staged_workflow import _params


@pytest.mark.parametrize("changes,location", [
    ({"algorithm": 123}, ("algorithm",)),
    ({"split_height": "3000"}, ("split_height",)),
    ({"split_height": True}, ("split_height",)),
    ({"paths": "bc.nc"}, ("paths",)),
    ({"unexpected": True}, ("unexpected",)),
])
def test_constructor_and_replace_report_structured_field_errors(changes, location):
    """Both supported construction paths enforce the declared fields."""
    for construct in (lambda: BcBasisOptions(**changes), lambda: replace(BcBasisOptions(), **changes)):
        with pytest.raises(ValidationError) as error:
            construct()
        assert error.value.errors()[0]["loc"] == location


def test_portable_json_and_schema_preserve_paths_without_opening_resources(tmp_path):
    """Portable values round-trip independently of whether the selected file exists."""
    options = BcBasisOptions(paths=(tmp_path / "absent.nc",), algorithm="inactive",
                             split_height=np.float64(2500.))
    assert type(options.split_height) is float
    assert options.to_params()["paths"] == (tmp_path / "absent.nc",)
    wire = json.loads(options.to_json())
    assert wire == {"paths": [str(tmp_path / "absent.nc")], "algorithm": "inactive", "split_height": 2500.}
    schema = BcBasisOptions.json_schema()
    assert schema["properties"]["split_height"]["default"] == 3000.
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(wire)
    restored = BcBasisOptions.from_json(options.to_json(), active=True)
    assert restored == options
    assert isinstance(restored.paths[0], Path)
    with pytest.raises(ValidationError, match="unexpected"):
        BcBasisOptions.from_json('{"unexpected": 1}', active=False)
    with pytest.raises(ValidationError, match="split_height"):
        BcBasisOptions.from_json('{"split_height": "2500"}', active=False)


def test_pydantic_dataclass_composes_with_current_data_owner_and_ini_registry():
    """The existing data owner and field registry use the same nested class."""
    config = RhimeConfig.from_params(_params(bc_basis={"algorithm": "twelve-region", "split_height": 2500.}),
                                    multisector=False)
    assert tuple(field.name for field in fields(BcBasisOptions)) == ("paths", "algorithm", "split_height")
    assert DataConfig.from_params(config.data.to_params()) == config.data
    assert BcBasisOptions.json_schema()["additionalProperties"] is False


def test_applicability_stays_at_the_consuming_phase():
    """Unknown algorithm names remain usable as inactive selections."""
    inactive = BcBasisOptions(algorithm="not-selected")
    assert BcBasisOptions.from_params(inactive, active=False) is inactive
    assert BcBasisOptions.from_json(inactive.to_json(), active=False) == inactive
    with pytest.raises(ValueError, match=r"bc_basis\.algorithm"):
        BcBasisOptions.from_params(inactive, active=True)
    with pytest.raises(ValueError, match=r"bc_basis\.algorithm"):
        BcBasisOptions.from_json(inactive.to_json(), active=True)


@pytest.mark.parametrize("height", [np.nan, np.inf, -np.inf])
def test_nonfinite_inactive_height_is_retained_but_never_serialized_as_null(height):
    """A runtime inactive value has no finite portable JSON representation."""
    options = BcBasisOptions(split_height=height)
    assert BcBasisOptions.from_params(options, active=False) is options
    with pytest.raises(PydanticSerializationError, match="finite"):
        options.to_json()
    with pytest.raises(PydanticSerializationError, match="finite"):
        TypeAdapter(BcBasisOptions).dump_json(options)
    with pytest.raises(ValidationError, match="finite"):
        BcBasisOptions.from_json(json.dumps({"split_height": height}), active=False)


def test_lazy_and_opaque_scalar_inputs_are_rejected_without_conversion():
    """Structural validation never invokes numerical conversion hooks."""
    @delayed
    def forbidden():
        """Fail if a borrowed scalar is executed."""
        pytest.fail("configuration executed a Dask graph")

    class Opaque:
        """Represent an object whose float conversion would execute work."""

        def __float__(self):
            """Fail if Pydantic attempts scalar coercion."""
            pytest.fail("configuration invoked __float__")

    lazy = da.from_delayed(forbidden(), shape=(), dtype=float)
    with Callback(pretask=lambda *args: pytest.fail("configuration executed Dask")):
        for value in (lazy, Opaque(), np.array(2500.)):
            with pytest.raises(ValidationError, match="eager numeric scalar"):
                BcBasisOptions(split_height=value)


def test_executable_example_exports_component_schema_and_json(tmp_path):
    """The documented example writes reusable schemas and readable options."""
    run_example(tmp_path)
    encoded = (tmp_path / "bc-basis.json").read_text()
    assert BcBasisOptions.from_json(encoded, active=True).split_height == 2500.
    Draft202012Validator.check_schema(json.loads((tmp_path / "bc-basis.schema.json").read_text()))
