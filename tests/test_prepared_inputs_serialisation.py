"""Tests for durable RHIME prepared-input artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import openghg_inversions.rhime.runner as rhime_runner
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.basis.operators import MultiSourceBucketBasisOperator
from openghg_inversions.inversion_data import RhimePreparedInputs, prepare_rhime_inputs
from openghg_inversions.models import RhimeModelSpec, SectorSpec
from openghg_inversions.rhime import (
    RhimeOutputSpec,
    RhimeRunSpec,
    RhimeSampler,
    run_rhime_from_prepared_inputs,
)
from openghg_inversions.rhime.outputs import RhimeOutputBundle
from openghg_inversions.serialization import (
    MULTIINDEX_DIMS_ATTR,
    inferencedata_from_datatree,
    inferencedata_to_datatree,
)


def _basis_functions() -> BasisFunctions:
    """Build a small self-contained operator and reference flux."""
    basis = xr.DataArray(
        [[1, 2]],
        dims=("lat", "lon"),
        coords={"lat": [51.0], "lon": [-2.0, -1.0]},
        name="basis",
    )
    flux = xr.DataArray(
        [[1.5, 2.5]],
        dims=("lat", "lon"),
        coords=basis.coords,
        attrs={"units": "mol m-2 s-1"},
        name="flux",
    )
    return BasisFunctions.from_flat_basis(
        basis_flat=basis,
        flux=flux,
        operator_kwargs={"state_dim": "region"},
    )


def _prepared_inputs(
    *,
    basis_artifact_path: str | None = "/path/that/does/not/exist/basis.nc",
    site_lats: tuple[float, ...] | None = (51.0, 52.0),
    site_lons: tuple[float, ...] | None = (-2.0, -1.0),
) -> RhimePreparedInputs:
    """Build canonical inputs with a site/time measurement MultiIndex."""
    nmeasure = pd.MultiIndex.from_arrays(
        [
            ["TAC", "MHD"],
            pd.to_datetime(["2019-01-01T00:00:00", "2019-01-01T01:00:00"]),
        ],
        names=["site", "time"],
    )
    inv_inputs = xr.Dataset(
        {
            "H": (("region", "nmeasure"), [[1.0, 2.0], [3.0, 4.0]]),
            "mf": ("nmeasure", [10.0, 11.0]),
            "mf_error": ("nmeasure", [0.5, 0.6]),
            "site_indicator": ("nmeasure", [0, 1]),
        },
        coords={
            "region": [0, 1],
            **xr.Coordinates.from_pandas_multiindex(nmeasure, "nmeasure"),
        },
        attrs={"prepared_by": "unit-test"},
    )
    inv_inputs["mf"].attrs["units"] = "ppm"
    return RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=_basis_functions(),
        sites=("TAC", "MHD"),
        averaging_period=("1h", None),
        basis_artifact_source="unit-test-generated",
        basis_artifact_path=basis_artifact_path,
        site_lats=site_lats,
        site_lons=site_lons,
    )


def _multisource_prepared_inputs() -> RhimePreparedInputs:
    """Build prepared inputs whose retained basis has source order B, A."""
    coords = {"lat": [51.0], "lon": [-2.0, -1.0]}
    basis_b = xr.DataArray([[1, 2]], dims=("lat", "lon"), coords=coords, name="basis")
    basis_a = xr.DataArray([[1, 1]], dims=("lat", "lon"), coords=coords, name="basis")
    basis_functions = BasisFunctions.from_multi_source_flat_basis(
        basis_flat={"B": basis_b, "A": basis_a},
        flux={
            "B": xr.full_like(basis_b, 2.0, dtype=float).rename("flux"),
            "A": xr.full_like(basis_a, 3.0, dtype=float).rename("flux"),
        },
        operator_kwargs={"state_dim": "region"},
    )
    nmeasure = pd.MultiIndex.from_arrays(
        [["TAC"], pd.to_datetime(["2019-01-01T00:00:00"])],
        names=["site", "time"],
    )
    inv_inputs = xr.Dataset(
        {
            "H": (("source", "region", "nmeasure"), np.ones((2, 2, 1))),
            "mf": ("nmeasure", [10.0]),
            "mf_error": ("nmeasure", [0.5]),
        },
        coords={
            "source": ["B", "A"],
            "region": [0, 1],
            **xr.Coordinates.from_pandas_multiindex(nmeasure, "nmeasure"),
        },
    )
    return RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=basis_functions,
        sites=("TAC",),
        averaging_period=("1h",),
        basis_artifact_source="unit-test-generated",
        site_lats=(51.0,),
        site_lons=(-2.0,),
    )


def _assert_prepared_identical(
    actual: RhimePreparedInputs,
    expected: RhimePreparedInputs,
) -> None:
    """Assert all durable prepared-input fields round-trip."""
    xr.testing.assert_identical(actual.inv_inputs, expected.inv_inputs)
    xr.testing.assert_identical(actual.basis_functions.flux, expected.basis_functions.flux)
    xr.testing.assert_identical(
        actual.basis_functions.operator.basis_matrix,
        expected.basis_functions.operator.basis_matrix,
    )
    assert type(actual.basis_functions.operator) is type(expected.basis_functions.operator)
    assert actual.basis_functions.operator.meta == expected.basis_functions.operator.meta
    assert actual.basis_functions.metadata == expected.basis_functions.metadata
    assert actual.sites == expected.sites
    assert actual.averaging_period == expected.averaging_period
    assert actual.basis_artifact_source == expected.basis_artifact_source
    assert actual.basis_artifact_path == expected.basis_artifact_path
    assert actual.site_lats == expected.site_lats
    assert actual.site_lons == expected.site_lons


@pytest.fixture(scope="module")
def prepared_from_real_route(
    tac_ch4_data_args: dict[str, Any],
    default_bc_basis_directory: Path,
) -> RhimePreparedInputs:
    """Prepare durable inputs through the real RHIME route and test object store."""
    preparation_args = dict(tac_ch4_data_args)
    flux_sources = preparation_args.pop("emissions_name")
    preparation_args.update(
        {
            "output_name": "serialisation_regression",
            "flux_sources": flux_sources,
            "basis_algorithm": "quadtree",
            "nbasis": 4,
            "use_bc": True,
            "bc_basis_directory": default_bc_basis_directory,
        }
    )
    return prepare_rhime_inputs(**preparation_args)


def test_prepared_inputs_datatree_roundtrip_is_self_contained() -> None:
    """The DataTree embeds canonical inputs, basis geometry, flux, and metadata."""
    prepared = _prepared_inputs()

    restored = RhimePreparedInputs.from_datatree(prepared.to_datatree())

    _assert_prepared_identical(restored, prepared)
    assert isinstance(restored.inv_inputs.indexes["nmeasure"], pd.MultiIndex)
    assert restored.inv_inputs.indexes["nmeasure"].names == ["site", "time"]


def test_prepared_inputs_datatree_roundtrips_nullable_metadata() -> None:
    """Nullable path and coordinate metadata remain None after decoding."""
    prepared = _prepared_inputs(basis_artifact_path=None, site_lats=None, site_lons=None)

    restored = RhimePreparedInputs.from_datatree(prepared.to_datatree())

    _assert_prepared_identical(restored, prepared)
    assert restored.averaging_period == ("1h", None)


def test_prepared_inputs_datatree_roundtrips_missing_site_coordinate() -> None:
    """A missing release coordinate uses JSON null and reloads as float NaN."""
    prepared = _prepared_inputs(site_lats=(51.0, np.nan))

    dt = prepared.to_datatree()
    restored = RhimePreparedInputs.from_datatree(dt)

    assert "NaN" not in dt.attrs["metadata"]
    assert restored.site_lats is not None
    assert restored.site_lats[0] == 51.0
    assert np.isnan(restored.site_lats[1])


def test_multisource_basis_loads_without_new_source_order_metadata() -> None:
    """BasisFunctions keeps accepting artifacts written before source-order metadata."""
    basis_dt = _multisource_prepared_inputs().basis_functions.to_datatree()
    del basis_dt["basis"].attrs["source_order"]

    restored = BasisFunctions.from_datatree(basis_dt)

    restored_operator = restored.operator
    assert isinstance(restored_operator, MultiSourceBucketBasisOperator)
    assert set(restored_operator.basis_flat) == {"A", "B"}


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_real_prepared_inputs_save_load_and_run_without_repreparation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    prepared_from_real_route: RhimePreparedInputs,
    suffix: str,
) -> None:
    """Real RHIME preparation survives both stores and runs without repeating preparation."""
    prepared = prepared_from_real_route
    serialized = prepared.to_datatree()
    multiindex_metadata = json.loads(serialized["inv_inputs"].attrs[MULTIINDEX_DIMS_ATTR])
    assert multiindex_metadata == {
        "dims": [
            {"dim": "nmeasure", "levels": ["site", "time"]},
            {"dim": "bc_region", "levels": ["bc_curtain", "bc_period"]},
        ]
    }

    artifact_path = tmp_path / f"real-prepared{suffix}"
    prepared.save(artifact_path)
    loaded = RhimePreparedInputs.load(artifact_path)

    _assert_prepared_identical(loaded, prepared)
    loaded_index = loaded.inv_inputs.indexes["nmeasure"]
    assert isinstance(loaded_index, pd.MultiIndex)
    assert loaded_index.names == ["site", "time"]
    loaded_bc_index = loaded.inv_inputs.indexes["bc_region"]
    assert isinstance(loaded_bc_index, pd.MultiIndex)
    assert loaded_bc_index.names == ["bc_curtain", "bc_period"]

    model_spec = RhimeModelSpec(
        species="ch4",
        domain="EUROPE",
        sectors=(
            SectorSpec(
                name="total",
                flux_source="total-ukghg-edgar7",
                x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                variable_suffix="total",
            ),
        ),
        use_bc=True,
    )
    run_spec = RhimeRunSpec(
        start_date="2019-01-01",
        end_date="2019-01-02",
        sites=("stale",),
        averaging_period=("stale",),
        model=model_spec,
        output=RhimeOutputSpec(output_format="none", save_inversion_output=False),
    )
    built_model = object()
    sampled = az.InferenceData()
    observed: dict[str, object] = {}

    def fail_preparation(*args: object, **kwargs: object) -> None:
        """Fail if the loaded-input runner attempts data preparation."""
        raise AssertionError("loaded prepared inputs must bypass preparation")

    def fake_builder(inv_inputs: xr.Dataset, spec: RhimeModelSpec) -> object:
        """Record the loaded builder inputs and return a sentinel model."""
        observed["builder_inputs"] = inv_inputs
        observed["builder_spec"] = spec
        return built_model

    def fake_sample(self: RhimeSampler, model: object) -> az.InferenceData:
        """Record the model passed to sampling and return a sentinel trace."""
        observed["sample_model"] = model
        return sampled

    def fake_outputs(**kwargs: Any) -> RhimeOutputBundle:
        """Record the loaded object used for output construction."""
        observed["output_prepared"] = kwargs["prepared"]
        return RhimeOutputBundle(outputs={"loaded": True})

    monkeypatch.setattr(rhime_runner, "prepare_rhime_inputs", fail_preparation)
    monkeypatch.setattr(rhime_runner, "build_rhime_model_from_spec", fake_builder)
    monkeypatch.setattr(RhimeSampler, "sample", fake_sample)
    monkeypatch.setattr(rhime_runner, "make_standard_output_bundle", fake_outputs)

    result = run_rhime_from_prepared_inputs(
        prepared_inputs=loaded,
        run_spec=run_spec,
        sampler=RhimeSampler(draws=1, tune=1, chains=1),
    )

    assert observed["builder_inputs"] is loaded.inv_inputs
    assert observed["builder_spec"] is model_spec
    assert observed["sample_model"] is built_model
    assert observed["output_prepared"] is loaded
    assert result.inv_inputs is loaded.inv_inputs
    assert result.basis_functions is loaded.basis_functions
    assert result.run_spec.sites == loaded.sites
    assert result.run_spec.averaging_period == loaded.averaging_period
    assert result.outputs == {"loaded": True}


def test_inferencedata_datatree_roundtrip_preserves_root_attrs() -> None:
    """Shared InferenceData conversion preserves artifact-level root attributes."""
    idata = az.InferenceData(
        posterior=xr.Dataset(
            {"x": (("chain", "draw"), [[1.0]])},
            coords={"chain": [0], "draw": [0]},
        ),
        attrs={"title": "root metadata", "schema_version": 3},
    )

    restored = inferencedata_from_datatree(inferencedata_to_datatree(idata))

    assert restored.attrs == idata.attrs


def test_prepared_inputs_zarr_save_overwrites_existing_artifact(tmp_path: Path) -> None:
    """Saving twice to Zarr replaces stale data from the first artifact."""
    artifact_path = tmp_path / "prepared.zarr"
    first = _prepared_inputs()
    first.inv_inputs["obsolete"] = ("nmeasure", [1, 2])
    first.save(artifact_path)

    expected = _prepared_inputs()
    expected.save(artifact_path)
    restored = RhimePreparedInputs.load(artifact_path)

    _assert_prepared_identical(restored, expected)
    assert "obsolete" not in restored.inv_inputs


@pytest.mark.parametrize(
    ("attr", "value"),
    [
        ("schema", None),
        ("schema", "not-the-prepared-input-schema"),
        ("schema_version", None),
        ("schema_version", 2),
        ("schema_version", "1"),
    ],
)
def test_prepared_inputs_rejects_missing_or_wrong_schema(
    attr: str,
    value: object,
) -> None:
    """Prepared-input loading requires its exact schema identifier and version."""
    dt = _prepared_inputs().to_datatree()
    if value is None:
        del dt.attrs[attr]
    else:
        dt.attrs[attr] = value

    with pytest.raises((TypeError, ValueError)):
        RhimePreparedInputs.from_datatree(dt)


@pytest.mark.parametrize("missing_node", ["inv_inputs", "basis_functions"])
def test_prepared_inputs_rejects_missing_required_nodes(missing_node: str) -> None:
    """Prepared-input loading rejects artifacts without either durable child."""
    original = _prepared_inputs().to_datatree()
    nodes = {name: child.copy(deep=True) for name, child in original.children.items() if name != missing_node}
    dt = xr.DataTree.from_dict(nodes)
    dt.attrs = dict(original.attrs)

    with pytest.raises((KeyError, ValueError)):
        RhimePreparedInputs.from_datatree(dt)


def test_prepared_inputs_rejects_missing_metadata() -> None:
    """Prepared-input loading requires the complete versioned metadata payload."""
    dt = _prepared_inputs().to_datatree()
    del dt.attrs["metadata"]

    with pytest.raises(ValueError):
        RhimePreparedInputs.from_datatree(dt)


@pytest.mark.parametrize(
    "raw_metadata",
    [
        None,
        json.dumps({"dims": []}),
        json.dumps({"dims": [{"dim": "nmeasure", "levels": ["time", "site"]}]}),
        "not-json",
    ],
)
def test_prepared_inputs_rejects_invalid_multiindex_metadata(raw_metadata: str | None) -> None:
    """Prepared loading requires metadata that restores nmeasure=(site, time)."""
    dt = _prepared_inputs().to_datatree()
    if raw_metadata is None:
        del dt["inv_inputs"].attrs[MULTIINDEX_DIMS_ATTR]
    else:
        dt["inv_inputs"].attrs[MULTIINDEX_DIMS_ATTR] = raw_metadata

    with pytest.raises(ValueError, match="MultiIndex"):
        RhimePreparedInputs.from_datatree(dt)


@pytest.mark.parametrize(
    ("attr", "value"),
    [
        ("schema", None),
        ("schema", "not-the-basis-schema"),
        ("schema_version", None),
        ("schema_version", 2),
        ("schema_version", "1"),
    ],
)
def test_prepared_inputs_rejects_invalid_nested_basis_schema(attr: str, value: object) -> None:
    """Prepared loading requires the exact embedded BasisFunctions v1 schema."""
    dt = _prepared_inputs().to_datatree()
    basis_dt = dt["basis_functions"]
    if value is None:
        del basis_dt.attrs[attr]
    else:
        basis_dt.attrs[attr] = value

    with pytest.raises(ValueError, match="embedded BasisFunctions"):
        RhimePreparedInputs.from_datatree(dt)


@pytest.mark.parametrize(
    "metadata",
    [
        "not-json",
        json.dumps(["not", "an", "object"]),
        json.dumps(
            {
                "sites": ["TAC", "MHD"],
                "averaging_period": ["1h", None],
                "basis_artifact_path": None,
                "site_lats": None,
                "site_lons": None,
            }
        ),
        json.dumps(
            {
                "sites": "TAC",
                "averaging_period": ["1h"],
                "basis_artifact_source": "generated",
                "basis_artifact_path": None,
                "site_lats": None,
                "site_lons": None,
            }
        ),
        json.dumps(
            {
                "sites": ["TAC", "MHD"],
                "averaging_period": ["1h"],
                "basis_artifact_source": "generated",
                "basis_artifact_path": None,
                "site_lats": None,
                "site_lons": None,
            }
        ),
        json.dumps(
            {
                "sites": ["TAC", "MHD"],
                "averaging_period": ["1h", None],
                "basis_artifact_source": "generated",
                "basis_artifact_path": 1,
                "site_lats": [51.0, 52.0],
                "site_lons": [-2.0, -1.0],
            }
        ),
        json.dumps(
            {
                "sites": ["TAC", "MHD"],
                "averaging_period": ["1h", None],
                "basis_artifact_source": "generated",
                "basis_artifact_path": None,
                "site_lats": [51.0],
                "site_lons": [-2.0, -1.0],
            }
        ),
        json.dumps(
            {
                "sites": ["TAC", "MHD"],
                "averaging_period": ["1h", None],
                "basis_artifact_source": "generated",
                "basis_artifact_path": None,
                "site_lats": [51.0, 52.0],
                "site_lons": None,
            }
        ),
    ],
)
def test_prepared_inputs_rejects_malformed_metadata(metadata: str) -> None:
    """Prepared-input metadata must be typed, complete, and site-aligned."""
    dt = _prepared_inputs().to_datatree()
    dt.attrs["metadata"] = metadata

    with pytest.raises((TypeError, ValueError)):
        RhimePreparedInputs.from_datatree(dt)


def test_loaded_prepared_inputs_run_through_existing_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A loaded artifact reaches model building, sampling, and output unchanged."""
    artifact_path = tmp_path / "prepared.nc"
    _prepared_inputs().save(artifact_path)
    loaded = RhimePreparedInputs.load(artifact_path)
    model_spec = RhimeModelSpec(
        species="ch4",
        domain="EUROPE",
        sectors=(
            SectorSpec(
                name="total",
                flux_source="total-inventory",
                x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                variable_suffix="total",
            ),
        ),
        use_bc=False,
    )
    run_spec = RhimeRunSpec(
        start_date="2019-01-01",
        end_date="2019-01-02",
        sites=("stale",),
        averaging_period=("stale",),
        model=model_spec,
        output=RhimeOutputSpec(output_format="none", save_inversion_output=False),
    )
    built_model = object()
    sampled = az.InferenceData(
        posterior=xr.Dataset(
            {"x": (("chain", "draw", "region"), np.ones((1, 1, 2)))},
            coords={"chain": [0], "draw": [0], "region": [0, 1]},
        )
    )
    observed: dict[str, Any] = {}

    def fail_preparation(*args: object, **kwargs: object) -> None:
        """Fail if execution attempts to prepare inputs again."""
        raise AssertionError("prepared-input execution must not repeat preparation")

    def fake_builder(inv_inputs: xr.Dataset, spec: RhimeModelSpec) -> object:
        """Record model-builder inputs and return the sentinel model."""
        observed["builder_inputs"] = inv_inputs
        observed["builder_spec"] = spec
        return built_model

    def fake_sample(self: RhimeSampler, model: object) -> az.InferenceData:
        """Record the sampled model and return the sentinel posterior."""
        observed["sample_model"] = model
        return sampled

    def fake_outputs(**kwargs: Any) -> RhimeOutputBundle:
        """Record output inputs and return a sentinel output bundle."""
        observed["output_prepared"] = kwargs["prepared"]
        observed["output_idata"] = kwargs["idata"]
        return RhimeOutputBundle(outputs={"loaded": True})

    monkeypatch.setattr(rhime_runner, "prepare_rhime_inputs", fail_preparation)
    monkeypatch.setattr(rhime_runner, "build_rhime_model_from_spec", fake_builder)
    monkeypatch.setattr(RhimeSampler, "sample", fake_sample)
    monkeypatch.setattr(rhime_runner, "make_standard_output_bundle", fake_outputs)

    result = run_rhime_from_prepared_inputs(
        prepared_inputs=loaded,
        run_spec=run_spec,
        sampler=RhimeSampler(draws=1, tune=1, chains=1),
    )

    assert observed["builder_inputs"] is loaded.inv_inputs
    assert observed["builder_spec"] is model_spec
    assert observed["sample_model"] is built_model
    assert observed["output_prepared"] is loaded
    assert observed["output_idata"] is sampled
    assert result.inv_inputs is loaded.inv_inputs
    assert result.run_spec.sites == loaded.sites
    assert result.run_spec.averaging_period == loaded.averaging_period
    assert result.outputs == {"loaded": True}


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_multisource_order_survives_load_run_and_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    suffix: str,
) -> None:
    """Nonlexicographic source order survives storage and the RHIME output seam."""
    original = _multisource_prepared_inputs()
    original_operator = original.basis_functions.operator
    assert isinstance(original_operator, MultiSourceBucketBasisOperator)
    state_index = original_operator.basis_matrix.indexes["region"]
    assert isinstance(state_index, pd.MultiIndex)
    state = xr.DataArray(
        np.arange(1, len(state_index) + 1, dtype=float),
        dims="region",
        coords=xr.Coordinates.from_pandas_multiindex(state_index, "region"),
    )
    expected_reconstruction = original.basis_functions.interpolate(state, flux=True)
    artifact_path = tmp_path / f"multisource{suffix}"
    original.save(artifact_path)

    loaded = RhimePreparedInputs.load(artifact_path)
    loaded_operator = loaded.basis_functions.operator
    assert isinstance(loaded_operator, MultiSourceBucketBasisOperator)
    assert list(loaded_operator.basis_flat) == ["B", "A"]
    assert tuple(loaded.basis_functions.flux.source.values) == ("B", "A")
    xr.testing.assert_identical(loaded_operator.basis_matrix, original_operator.basis_matrix)
    xr.testing.assert_identical(
        loaded.basis_functions.interpolate(state, flux=True),
        expected_reconstruction,
    )

    model_spec = RhimeModelSpec(
        species="ch4",
        domain="EUROPE",
        sectors=tuple(
            SectorSpec(
                name=source,
                flux_source=source,
                x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                variable_suffix=source.lower(),
            )
            for source in ("B", "A")
        ),
        use_bc=False,
    )
    run_spec = RhimeRunSpec(
        start_date="2019-01-01",
        end_date="2019-01-02",
        sites=loaded.sites,
        averaging_period=loaded.averaging_period,
        model=model_spec,
        output=RhimeOutputSpec(output_format="none", save_inversion_output=False),
        split_by_sectors=True,
    )
    built_model = object()
    sampled = az.InferenceData()
    observed: dict[str, object] = {}

    def fake_builder(inv_inputs: xr.Dataset, spec: RhimeModelSpec) -> object:
        """Record ordered builder inputs and return the sentinel model."""
        observed["source_order"] = tuple(inv_inputs.source.values)
        observed["model_spec"] = spec
        return built_model

    def fake_sample(self: RhimeSampler, model: object) -> az.InferenceData:
        """Return a sentinel trace for the multisource prepared run."""
        assert model is built_model
        return sampled

    def fake_outputs(**kwargs: Any) -> RhimeOutputBundle:
        """Exercise postprocessing reconstruction on the loaded retained basis."""
        output_prepared = kwargs["prepared"]
        assert isinstance(output_prepared, RhimePreparedInputs)
        output_operator = output_prepared.basis_functions.operator
        assert isinstance(output_operator, MultiSourceBucketBasisOperator)
        observed["output_order"] = tuple(output_operator.basis_flat)
        xr.testing.assert_identical(
            output_prepared.basis_functions.interpolate(state, flux=True),
            expected_reconstruction,
        )
        return RhimeOutputBundle(outputs={"postprocessed": True})

    monkeypatch.setattr(rhime_runner, "build_rhime_multisector_model_from_spec", fake_builder)
    monkeypatch.setattr(RhimeSampler, "sample", fake_sample)
    monkeypatch.setattr(rhime_runner, "make_multisector_output_bundle", fake_outputs)

    result = run_rhime_from_prepared_inputs(
        prepared_inputs=loaded,
        run_spec=run_spec,
        sampler=RhimeSampler(draws=1, tune=1, chains=1),
    )

    assert observed["source_order"] == ("B", "A")
    assert observed["model_spec"] is model_spec
    assert observed["output_order"] == ("B", "A")
    assert result.outputs == {"postprocessed": True}
