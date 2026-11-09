"""Generated references preserve schema constraints without executing checks."""

import json
import sys
from types import SimpleNamespace

import dask
import numpy as np
import pytest
from xarray_validate import AttrSchema, CoordsSchema, DataArraySchema, DatasetSchema, DimsSchema

from openghg_inversions.inversion_data._merged_schema import merged_flux_schema, merged_site_schema
from scripts.schema_docs import describe_schema, load_schema, render_schema


def _forbid_execution(*args, **kwargs):
    """Report an unexpected validator or scheduler call."""
    raise AssertionError("Schema documentation executed scientific validation.")


@pytest.mark.parametrize("kwargs", [
    {},
    {"split_by_sectors": True, "use_bc": True},
    {"use_bc": True, "boundary_representation": "centered-v1"},
])
def test_merged_schema_description_preserves_structure_and_does_not_execute(kwargs):
    """Real factory variants retain dtype families, patterns and callback prose."""
    schema = merged_site_schema(**kwargs)
    original_check = schema.checks[0]
    schema.checks = [original_check, _forbid_execution]
    with dask.config.set(scheduler=_forbid_execution):
        description = describe_schema(schema)
        rendered = render_schema(description)
    assert json.loads(json.dumps(description, allow_nan=False)) == description
    assert description["data_vars"]["mf"]["dtype"] == ["integer", "floating"]
    assert description["coords"]["coords"]["time"]["dtype"] == np.dtype(np.datetime64).str
    assert description["data_vars"]["mf"]["attrs"]["attrs"]["units"]["type"] == "str"
    assert description["data_vars"]["{mf_error|mf_repeatability|mf_variability}"]["dims"] == ["time"]
    assert "at least one uncertainty field" in rendered
    assert "Pattern entries constrain matching fields without requiring a match" in rendered
    assert "Explicit error" in rendered
    assert schema.checks[0] is original_check
    assert schema.data_vars["mf"].dtype.dtype == (np.integer, np.floating)
    if kwargs.get("split_by_sectors"):
        assert "fp_x_flux_sectoral" in rendered
        assert "bc_n" in description["data_vars"]
    if kwargs.get("boundary_representation"):
        assert "bc_anomaly_n" in rendered
        assert description["data_vars"]["bc_reference"]["dims"] == []


def test_flux_schema_description_retains_selected_dimensions_and_unit_rules():
    """Time-dependent native flux and deferred conversion remain visible."""
    description = describe_schema(merged_flux_schema(timed=True, allow_scaled_units=True))
    assert description["data_vars"]["flux"]["dims"] == {"dims": ["lat", "lon", "time"], "ordered": False}
    assert "Values are neither read nor converted" in render_schema(description)


def test_renderer_handles_class_owned_schemas_and_nested_checks(monkeypatch):
    """A class method needs no ingestion-specific interface or registration."""
    class Contract:
        """Representative class-owned bound contract factory."""

        @staticmethod
        def schema(*, observation_dim="nmeasure"):
            """Build a bound-style schema with ordered dimensions."""
            return DatasetSchema(
                data_vars={
                    "anomaly": DataArraySchema(
                        dtype=(np.float32, np.float64), dims=(observation_dim, "bc_region"),
                        attrs={"units": AttrSchema(type=str)}, checks=[_forbid_execution],
                    ),
                    "{weights}": DataArraySchema(dtype=np.floating, dims=("bc_region",)),
                },
                coords=CoordsSchema({observation_dim: DataArraySchema(dtype=np.int64, dims=(observation_dim,))}),
                allow_extra_keys=False,
            )

    monkeypatch.setitem(sys.modules, "schema_docs_example", SimpleNamespace(Contract=Contract))
    schema = load_schema("schema_docs_example:Contract.schema", {"observation_dim": "rows"})
    description = describe_schema(schema)
    rendered = render_schema(description)
    assert description["data_vars"]["anomaly"]["dtype"] == [np.dtype(np.float32).str, np.dtype(np.float64).str]
    assert description["data_vars"]["{weights}"]["dtype"] == "floating"
    assert '["rows", "bc_region"]' in rendered
    assert "(ordered)" in rendered
    assert "Extra fields are rejected" in rendered
    assert "Dataset.anomaly: Relationship checks" in rendered
    assert "Report an unexpected validator or scheduler call" in rendered
    assert "Dataset.anomaly: Attributes" in rendered


def test_dataarray_reference_retains_order_shape_chunks_and_undocumented_check():
    """Additional declared constraints and missing prose are never hidden."""
    schema = DataArraySchema(
        dtype="float64", dims=DimsSchema(("lat", "lon"), ordered=False), shape=(2, 3),
        chunks=False, checks=[lambda data: None],
    )
    rendered = render_schema(describe_schema(schema))
    assert "(any order)" in rendered
    assert "shape:" in rendered and "[2, 3]" in rendered
    assert "chunks:" in rendered and "false" in rendered
    assert "No description supplied" in rendered


def test_reference_reflects_new_fields_without_a_documentation_edit():
    """A schema change reaches the generated table through the factory alone."""
    schema = merged_site_schema()
    schema.data_vars["extension"] = DataArraySchema(dtype="int16", dims=())
    rendered = render_schema(describe_schema(schema))
    assert "``extension``" in rendered
    assert "Scalar" in rendered
    assert "int16" in rendered


def test_factory_rejects_a_non_schema_result(monkeypatch):
    """A broken documentation target fails instead of producing an empty table."""
    monkeypatch.setitem(sys.modules, "schema_docs_example", SimpleNamespace(factory=lambda: {}))
    with pytest.raises(TypeError, match="must return"):
        load_schema("schema_docs_example:factory", {})
