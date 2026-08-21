"""Tests for durable RHIME prepared-input artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

import openghg_inversions.rhime.multisector as rhime_multisector
import openghg_inversions.rhime.prepared as rhime_prepared
import openghg_inversions.rhime.standard as rhime_standard
from openghg_inversions.basis.basis_functions import (
    BASIS_ARTIFACT_PATH_ATTR,
    BASIS_ARTIFACT_SOURCE_ATTR,
    BasisFunctions,
)
from openghg_inversions.basis.operators import MultiSourceBucketBasisOperator
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.inversion_data import RhimePreparedInputs, prepare_rhime_inputs
from openghg_inversions.rhime import (
    RhimeModelSpec,
    RhimeOutputSpec,
    RhimeRunSpec,
    RhimeSampler,
    SectorSpec,
    run_rhime_from_prepared_inputs,
)
from openghg_inversions.serialization import (
    MULTIINDEX_DIMS_ATTR,
    encode_multiindexes_for_storage,
    inferencedata_from_datatree,
    inferencedata_to_datatree,
    load_inferencedata,
    normalise_declared_multiindex,
    restore_declared_multiindexes,
    save_inferencedata,
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


def _site_metadata(
    sites: tuple[str, ...],
    averaging_period: tuple[str | None, ...],
) -> xr.Dataset:
    """Build the canonical labeled site metadata used by prepared inputs."""
    return xr.Dataset(
        {"averaging_period": ("site", np.asarray(averaging_period, dtype=object))},
        coords={"site": list(sites)},
    )


def _prepared_inputs(
    *,
    basis_artifact_path: str | None = "/path/that/does/not/exist/basis.nc",
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
            "min_error": ("nmeasure", [0.0, 0.0]),
            "site_indicator": ("nmeasure", [0, 1]),
        },
        coords={
            "region": [0, 1],
            **xr.Coordinates.from_pandas_multiindex(nmeasure, "nmeasure"),
        },
        attrs={"prepared_by": "unit-test"},
    )
    inv_inputs["mf"].attrs["units"] = "ppm"
    basis_metadata = {BASIS_ARTIFACT_SOURCE_ATTR: "unit-test-generated"}
    if basis_artifact_path is not None:
        basis_metadata[BASIS_ARTIFACT_PATH_ATTR] = basis_artifact_path
    return RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=_basis_functions().with_metadata(basis_metadata),
        site_metadata=_site_metadata(("TAC", "MHD"), ("1h", None)),
    )


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
@pytest.mark.parametrize("representation", ["dense", "low_rank"])
def test_structured_aggregation_error_survives_prepared_input_round_trip(
    tmp_path: Path,
    suffix: str,
    representation: str,
) -> None:
    prepared = _prepared_inputs()
    inv_inputs = prepared.inv_inputs.copy()
    covariance = np.array([[0.5, 0.1], [0.1, 0.4]])
    if representation == "dense":
        inv_inputs["aggregation_error_covariance"] = (
            ("nmeasure", "nmeasure_cov"),
            covariance,
        )
    else:
        factor = np.array([[0.5], [0.2]])
        inv_inputs["low_rank_factor"] = (("nmeasure", "agg_rank"), factor)
        inv_inputs["diagonal_residual_variance"] = (
            "nmeasure",
            np.diag(covariance) - np.sum(factor**2, axis=1),
        )
    structured = RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=prepared.basis_functions,
        site_metadata=prepared.site_metadata,
    )
    path = tmp_path / f"structured-{representation}{suffix}"

    structured.save(path)
    loaded = RhimePreparedInputs.load(path)

    xr.testing.assert_identical(loaded.inv_inputs, structured.inv_inputs)


def test_derived_outputs_reject_aggregation_error_until_reconstruction_lands() -> None:
    prepared = _prepared_inputs()
    prepared.inv_inputs["aggregation_error_sd"] = ("nmeasure", [0.2, 0.3])
    model_spec = RhimeModelSpec(
        species="ch4",
        domain="EUROPE",
        sectors=(
            SectorSpec(
                name="total",
                flux_source="total",
                x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                variable_suffix="total",
            ),
        ),
        use_bc=False,
        aggregation_error_mode="diagonal",
    )
    run_spec = RhimeRunSpec(
        start_date="2019-01-01",
        end_date="2019-01-02",
        sites=prepared.sites,
        averaging_period=prepared.averaging_period,
        model=model_spec,
        output=RhimeOutputSpec(output_format="basic", save_inversion_output=False),
    )

    with pytest.raises(ValueError, match="aggregation-error covariance.*output_format='basic'"):
        run_rhime_from_prepared_inputs(prepared_inputs=prepared, run_spec=run_spec)


def _multisource_prepared_inputs() -> RhimePreparedInputs:
    """Build prepared inputs whose retained basis has source order B, A."""
    coords = {"lat": [51.0], "lon": [-2.0, -1.0]}
    basis_b = xr.DataArray([[1, 2]], dims=("lat", "lon"), coords=coords, name="basis")
    basis_a = xr.DataArray([[2, 1]], dims=("lat", "lon"), coords=coords, name="basis")
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
            "min_error": ("nmeasure", [0.0]),
            "site_indicator": ("nmeasure", [0]),
        },
        coords={
            "source": ["B", "A"],
            "region": [0, 1],
            **xr.Coordinates.from_pandas_multiindex(nmeasure, "nmeasure"),
        },
    )
    return RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=basis_functions.with_metadata({BASIS_ARTIFACT_SOURCE_ATTR: "unit-test-generated"}),
        site_metadata=_site_metadata(("TAC",), ("1h",)),
    )


def test_prepared_inputs_normalizes_h_to_basis_source_order() -> None:
    """Equivalent H source labels are selected into the operator's canonical order."""
    original = _multisource_prepared_inputs()
    reordered_h = original.inv_inputs.sel(source=["A", "B"])

    prepared = RhimePreparedInputs(
        inv_inputs=reordered_h,
        basis_functions=original.basis_functions,
        site_metadata=original.site_metadata,
    )

    assert prepared.inv_inputs.source.values.tolist() == ["B", "A"]


def test_prepared_inputs_normalizes_gathered_state_covariance_to_basis_source_order() -> None:
    """Gathered H and both prior-covariance axes follow the basis source order."""
    original = _multisource_prepared_inputs()
    state_index = pd.MultiIndex.from_tuples(
        [("A", 0), ("B", 0), ("A", 1), ("B", 1)],
        names=("source", "region_in_source"),
    )
    inv_inputs = original.inv_inputs.drop_dims(("source", "region"))
    inv_inputs["H"] = xr.DataArray(
        [[10.0], [20.0], [30.0], [40.0]],
        dims=("region", "nmeasure"),
        coords={
            **xr.Coordinates.from_pandas_multiindex(state_index, "region"),
            "nmeasure": inv_inputs["nmeasure"],
            "state_note": ("region", ["a0", "b0", "a1", "b1"]),
        },
    )
    inv_inputs["alpha_prior_mean"] = ("region", [1.0, 2.0, 3.0, 4.0])
    covariance = np.array(
        [
            [0.40, 0.01, 0.02, 0.03],
            [0.01, 0.50, 0.04, 0.05],
            [0.02, 0.04, 0.60, 0.06],
            [0.03, 0.05, 0.06, 0.70],
        ]
    )
    column_labels = [
        json.dumps(label, ensure_ascii=False, separators=(",", ":"))
        for label in state_index.tolist()
    ]
    inv_inputs["alpha_prior_covariance"] = (
        ("region", "region_cov"),
        covariance,
    )
    cross_covariance = np.arange(8, dtype=float).reshape(2, 4)
    inv_inputs["native_retained_cross_covariance"] = (
        ("native_state", "region_cov"),
        cross_covariance,
    )
    inv_inputs = inv_inputs.assign_coords(region_cov_label=("region_cov", column_labels))

    prepared = RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=original.basis_functions,
        site_metadata=original.site_metadata,
    )

    expected_index = pd.MultiIndex.from_tuples(
        [("B", 0), ("B", 1), ("A", 0), ("A", 1)],
        names=("source", "region_in_source"),
    )
    assert prepared.inv_inputs.indexes["region"].equals(expected_index)
    assert prepared.inv_inputs["H"].values[:, 0].tolist() == [20.0, 40.0, 10.0, 30.0]
    assert prepared.inv_inputs["state_note"].values.tolist() == ["b0", "b1", "a0", "a1"]
    state_order = [1, 3, 0, 2]
    np.testing.assert_array_equal(
        prepared.inv_inputs["alpha_prior_covariance"],
        covariance[np.ix_(state_order, state_order)],
    )
    assert prepared.inv_inputs["region_cov_label"].values.tolist() == [
        column_labels[index] for index in state_order
    ]
    np.testing.assert_array_equal(
        prepared.inv_inputs["native_retained_cross_covariance"],
        cross_covariance[:, state_order],
    )
    CorrelatedLognormalPrior(
        prepared.inv_inputs["alpha_prior_mean"],
        prepared.inv_inputs["alpha_prior_covariance"],
        covariance_dim="region_cov",
    )


def test_prepared_inputs_rejects_h_source_mismatch() -> None:
    """H cannot omit an operator source or introduce an unknown source."""
    original = _multisource_prepared_inputs()
    mismatched = original.inv_inputs.assign_coords(source=["B", "C"])

    with pytest.raises(ValueError, match="source labels do not match"):
        RhimePreparedInputs(
            inv_inputs=mismatched,
            basis_functions=original.basis_functions,
            site_metadata=original.site_metadata,
        )


def test_prepared_inputs_rejects_shared_basis_flux_h_source_mismatch() -> None:
    """A shared spatial basis still obtains source truth from labeled flux."""
    original = _prepared_inputs()
    flux = original.basis_functions.flux.expand_dims(source=["B", "A"])
    basis_functions = original.basis_functions.with_flux(flux)
    inv_inputs = original.inv_inputs.assign(H=original.inv_inputs["H"].expand_dims(source=["B", "C"]))

    with pytest.raises(ValueError, match="source labels do not match"):
        RhimePreparedInputs(
            inv_inputs=inv_inputs,
            basis_functions=basis_functions,
            site_metadata=original.site_metadata,
        )


def test_prepared_inputs_drops_unused_site_metadata_label() -> None:
    """Canonicalization keeps metadata only for sites present in nmeasure."""
    original = _prepared_inputs()
    site_metadata = original.site_metadata.reindex(site=["TAC", "MHD", "UNUSED"])

    prepared = RhimePreparedInputs(
        inv_inputs=original.inv_inputs,
        basis_functions=original.basis_functions,
        site_metadata=site_metadata,
    )

    assert prepared.sites == ("TAC", "MHD")


def test_prepared_inputs_normalizes_empty_averaging_period() -> None:
    """A legacy falsy period is normalized to the missing-value representation."""
    original = _prepared_inputs()
    site_metadata = original.site_metadata.copy()
    site_metadata["averaging_period"] = ("site", ["", None])

    prepared = RhimePreparedInputs(
        inv_inputs=original.inv_inputs,
        basis_functions=original.basis_functions,
        site_metadata=site_metadata,
    )

    assert prepared.averaging_period == (None, None)


def test_prepared_inputs_legacy_factory_normalizes_inputs() -> None:
    """The legacy factory adapts positional fields without parallel state."""
    original = _prepared_inputs()

    prepared = RhimePreparedInputs.from_legacy_inputs(
        original.inv_inputs,
        original.basis_functions,
        original.sites,
        original.averaging_period,
        original.basis_artifact_source,
        original.basis_artifact_path,
        (51.0, 52.0),
        (-2.0, -1.0),
    )

    xr.testing.assert_identical(prepared.site_metadata, original.site_metadata)
    np.testing.assert_array_equal(prepared.inv_inputs.release_lat, [51.0, 52.0])
    np.testing.assert_array_equal(prepared.inv_inputs.release_lon, [-2.0, -1.0])
    assert prepared.inv_inputs.release_lat.dims == ("nmeasure",)
    assert prepared.inv_inputs.release_lon.dims == ("nmeasure",)


def test_prepared_inputs_legacy_factory_preserves_observation_release_metadata() -> None:
    """Legacy site scalars do not replace higher-fidelity observation arrays."""
    original = _prepared_inputs()
    inv_inputs = original.inv_inputs.assign_coords(
        release_lat=("nmeasure", [51.1, 52.2]),
        release_lon=("nmeasure", [-2.1, -1.2]),
    )

    prepared = RhimePreparedInputs.from_legacy_inputs(
        inv_inputs,
        original.basis_functions,
        original.sites,
        original.averaging_period,
        original.basis_artifact_source,
        original.basis_artifact_path,
        (0.0, 0.0),
        (0.0, 0.0),
    )

    np.testing.assert_array_equal(prepared.inv_inputs.release_lat, [51.1, 52.2])
    np.testing.assert_array_equal(prepared.inv_inputs.release_lon, [-2.1, -1.2])


def test_prepared_inputs_preserves_site_metadata_attrs() -> None:
    """Normalization and CF preparation preserve labeled scientific metadata."""
    original = _prepared_inputs()
    site_metadata = original.site_metadata.copy(deep=True)
    site_metadata["site"].attrs["long_name"] = "observation site"
    site_metadata["averaging_period"].attrs["long_name"] = "observation averaging period"
    prepared = RhimePreparedInputs(
        inv_inputs=original.inv_inputs,
        basis_functions=original.basis_functions,
        site_metadata=site_metadata,
    )

    restored = RhimePreparedInputs.from_datatree(prepared.to_datatree())

    xr.testing.assert_identical(restored.site_metadata, site_metadata)


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
    xr.testing.assert_identical(actual.site_metadata, expected.site_metadata)
    assert actual.sites == expected.sites
    assert actual.averaging_period == expected.averaging_period
    assert actual.basis_artifact_source == expected.basis_artifact_source
    assert actual.basis_artifact_path == expected.basis_artifact_path


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
    """Nullable periods and provenance round-trip."""
    prepared = _prepared_inputs(basis_artifact_path=None)

    restored = RhimePreparedInputs.from_datatree(prepared.to_datatree())

    _assert_prepared_identical(restored, prepared)
    assert restored.averaging_period == ("1h", None)


def test_prepared_inputs_roundtrip_observation_aligned_release_metadata() -> None:
    """Release locations remain observation arrays rather than site scalars."""
    original = _prepared_inputs()
    inv_inputs = original.inv_inputs.assign_coords(
        release_lat=("nmeasure", [51.0, 52.0]),
        release_lon=("nmeasure", [-2.0, -1.0]),
    )
    prepared = RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=original.basis_functions,
        site_metadata=original.site_metadata,
    )

    restored = RhimePreparedInputs.from_datatree(prepared.to_datatree())

    assert restored.inv_inputs.release_lat.dims == ("nmeasure",)
    assert restored.inv_inputs.release_lon.dims == ("nmeasure",)
    np.testing.assert_array_equal(restored.inv_inputs.release_lat, [51.0, 52.0])
    np.testing.assert_array_equal(restored.inv_inputs.release_lon, [-2.0, -1.0])
    assert "release_lat" not in restored.site_metadata
    assert "release_lon" not in restored.site_metadata


@pytest.mark.rhime_contract
def test_site_indicator_is_derived_from_measurement_sites() -> None:
    """Supplied positional codes are replaced by the labeled measurement relation."""
    original = _prepared_inputs()
    inv_inputs = original.inv_inputs.assign(site_indicator=("nmeasure", [1, 0]))
    site_metadata = original.site_metadata.sel(site=["MHD", "TAC"])
    prepared = RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=original.basis_functions,
        site_metadata=site_metadata,
    )

    restored = RhimePreparedInputs.from_datatree(prepared.to_datatree())

    assert restored.sites == ("TAC", "MHD")
    assert restored.inv_inputs.site_indicator.values.tolist() == [0, 1]
    decoded = restored.site_metadata.site.values[restored.inv_inputs.site_indicator.values]
    np.testing.assert_array_equal(decoded, restored.inv_inputs["site"].values)


@pytest.mark.rhime_contract
@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_real_prepared_inputs_save_load_and_run_without_repreparation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    prepared_from_real_route: RhimePreparedInputs,
    suffix: str,
) -> None:
    """Freeze the labeled prepared-input round-trip and replay contract.

    Both NetCDF and Zarr must preserve dimensions, indexed and auxiliary
    coordinates, values, site alignment, and retained-basis metadata. Replaying
    either artifact must also bypass data preparation.
    """
    prepared = prepared_from_real_route
    serialized = prepared.to_datatree()
    encoded_inputs = serialized["inv_inputs"].to_dataset()
    assert encoded_inputs["nmeasure"].attrs["compress"] == "site time"
    assert encoded_inputs["bc_region"].attrs["compress"] == "bc_curtain bc_period"
    assert prepared.inv_inputs.release_lat.dims == ("nmeasure",)
    assert prepared.inv_inputs.release_lon.dims == ("nmeasure",)

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
    sampled = az.InferenceData()
    observed: dict[str, object] = {}
    original_builder = rhime_standard.build_standard_rhime_model

    def fail_preparation(*args: object, **kwargs: object) -> None:
        """Fail if the loaded-input runner attempts data preparation."""
        raise AssertionError("loaded prepared inputs must bypass preparation")

    def fake_builder(
        flux_sensitivity: xr.DataArray,
        **kwargs: Any,
    ) -> pm.Model:
        """Record the loaded builder inputs and build the real canonical graph."""
        assert kwargs["likelihood_builder"] is None
        assert kwargs["preserve_legacy_likelihood"] is False
        observed["builder_inputs"] = flux_sensitivity
        observed["observations"] = kwargs["observations"]
        return original_builder(flux_sensitivity, **kwargs)

    def fake_sample(
        self: RhimeSampler,
        model: pm.Model,
        *,
        variable_roles: dict[str, str],
    ) -> az.InferenceData:
        """Record the model passed to sampling and return a sentinel trace."""
        observed["sample_model"] = model
        assert variable_roles["concentration"] == "y"
        return sampled

    def fake_outputs(**kwargs: Any) -> None:
        """Record the loaded object used for output construction."""
        observed["output_prepared"] = kwargs["prepared"]
        kwargs["result"].outputs["loaded"] = True

    monkeypatch.setattr(rhime_standard, "retrieve_or_reload_rhime_data", fail_preparation)
    monkeypatch.setattr(rhime_standard, "build_standard_rhime_model", fake_builder)
    monkeypatch.setattr(RhimeSampler, "sample", fake_sample)
    monkeypatch.setattr(rhime_prepared, "make_standard_rhime_outputs", fake_outputs)

    result = run_rhime_from_prepared_inputs(
        prepared_inputs=loaded,
        run_spec=run_spec,
        sampler=RhimeSampler(draws=1, tune=1, chains=1),
    )

    builder_inputs = observed["builder_inputs"]
    output_prepared = observed["output_prepared"]
    assert isinstance(builder_inputs, xr.DataArray)
    assert isinstance(output_prepared, RhimePreparedInputs)
    xr.testing.assert_identical(builder_inputs, loaded.inv_inputs["H"])
    xr.testing.assert_identical(observed["observations"], loaded.inv_inputs["mf"])
    assert "y" in observed["sample_model"].named_vars
    assert result.inv_inputs is output_prepared.inv_inputs
    assert result.basis_functions is output_prepared.basis_functions
    assert result.basis_functions.operator is loaded.basis_functions.operator
    xr.testing.assert_identical(result.basis_functions.flux, loaded.basis_functions.flux)
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


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_supported_inferencedata_roundtrip_restores_multiindex(
    tmp_path: Path,
    suffix: str,
) -> None:
    """The supported trace boundary round-trips semantic observation identity."""
    nmeasure = pd.MultiIndex.from_arrays(
        [["MHD", "TAC"], pd.to_datetime(["2019-01-01", "2019-01-02"])],
        names=["site", "time"],
    )
    posterior_predictive = xr.Dataset(
        {"y": (("chain", "draw", "nmeasure"), np.ones((1, 1, 2)))},
        coords={
            "chain": [0],
            "draw": [0],
            **xr.Coordinates.from_pandas_multiindex(nmeasure, "nmeasure"),
        },
    )
    idata = az.InferenceData(
        posterior_predictive=posterior_predictive,
        attrs={"title": "semantic trace"},
    )
    path = tmp_path / f"trace{suffix}"

    save_inferencedata(idata, path)
    restored = load_inferencedata(path)

    assert restored.attrs == idata.attrs
    restored_index = restored.posterior_predictive.indexes["nmeasure"]
    assert isinstance(restored_index, pd.MultiIndex)
    assert restored_index.equals(nmeasure)


def test_storage_multiindex_metadata_declares_semantic_expectations() -> None:
    """Expanded storage form declares ownership, level order, and validation policy."""
    nmeasure = pd.MultiIndex.from_tuples(
        [("MHD", pd.Timestamp("2019-01-01"))],
        names=["site", "time"],
    )
    ds = xr.Dataset(
        {"y": ("nmeasure", [1.0])},
        coords=xr.Coordinates.from_pandas_multiindex(nmeasure, "nmeasure"),
    )

    encoded = encode_multiindexes_for_storage(ds)
    metadata = json.loads(encoded.attrs[MULTIINDEX_DIMS_ATTR])

    assert metadata == {
        "version": 1,
        "dims": [
            {
                "dim": "nmeasure",
                "levels": ["site", "time"],
                "reconstruct": True,
                "unique": True,
                "order": "preserve",
            }
        ],
    }


def test_declared_multiindex_normalizer_accepts_expanded_form() -> None:
    """Expanded semantic coordinates normalize to the same indexed representation."""
    expanded = xr.Dataset(
        {"y": ("nmeasure", [1.0, 2.0])},
        coords={
            "site": ("nmeasure", ["MHD", "TAC"]),
            "time": ("nmeasure", pd.to_datetime(["2019-01-01", "2019-01-02"])),
        },
    )

    normalized = normalise_declared_multiindex(expanded, "nmeasure", ["site", "time"])

    index = normalized.indexes["nmeasure"]
    assert isinstance(index, pd.MultiIndex)
    assert index.names == ["site", "time"]


def test_declared_multiindex_normalizer_rejects_reordered_levels() -> None:
    """An existing index must use the declared semantic level order exactly."""
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2019-01-01"), "MHD")],
        names=["time", "site"],
    )
    ds = xr.Dataset(
        {"y": ("nmeasure", [1.0])},
        coords=xr.Coordinates.from_pandas_multiindex(index, "nmeasure"),
    )

    with pytest.raises(ValueError, match="expected.*site.*time.*in that order"):
        normalise_declared_multiindex(ds, "nmeasure", ["site", "time"])


def test_declared_multiindex_restoration_rejects_duplicate_identity() -> None:
    """Strict restoration rejects duplicate semantic labels instead of guessing."""
    expanded = xr.Dataset(
        {"y": ("nmeasure", [1.0, 2.0])},
        coords={
            "site": ("nmeasure", ["MHD", "MHD"]),
            "time": ("nmeasure", pd.to_datetime(["2019-01-01", "2019-01-01"])),
        },
        attrs={
            MULTIINDEX_DIMS_ATTR: json.dumps(
                {
                    "version": 1,
                    "dims": [
                        {
                            "dim": "nmeasure",
                            "levels": ["site", "time"],
                            "reconstruct": True,
                            "unique": True,
                            "order": "preserve",
                        }
                    ],
                }
            )
        },
    )

    with pytest.raises(ValueError, match="duplicate label"):
        restore_declared_multiindexes(expanded, strict=True)


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


@pytest.mark.parametrize("missing_node", ["inv_inputs", "basis_functions", "site_metadata"])
def test_prepared_inputs_rejects_missing_required_nodes(missing_node: str) -> None:
    """Prepared-input loading requires all three durable labeled-data children."""
    original = _prepared_inputs().to_datatree()
    nodes = {name: child.copy(deep=True) for name, child in original.children.items() if name != missing_node}
    dt = xr.DataTree.from_dict(nodes)
    dt.attrs = dict(original.attrs)

    with pytest.raises((KeyError, ValueError)):
        RhimePreparedInputs.from_datatree(dt)


def test_prepared_inputs_rejects_invalid_cf_measurement_encoding() -> None:
    """Prepared loading requires CF gathering metadata for nmeasure decoding."""
    dt = _prepared_inputs().to_datatree()
    del dt["inv_inputs"]["nmeasure"].attrs["compress"]

    with pytest.raises(ValueError, match="CF gathered coordinate"):
        RhimePreparedInputs.from_datatree(dt)


def test_prepared_inputs_delegates_nested_basis_schema_to_basis_functions() -> None:
    """Outer loading delegates nested compatibility and provenance to BasisFunctions."""
    dt = _prepared_inputs().to_datatree()
    del dt["basis_functions"].attrs["schema"]
    del dt["basis_functions"].attrs["schema_version"]

    restored = RhimePreparedInputs.from_datatree(dt)

    assert restored.basis_artifact_source == "unit-test-generated"
    assert restored.basis_artifact_path == "/path/that/does/not/exist/basis.nc"


def test_prepared_inputs_rejects_site_coordinate_missing_measurement_labels() -> None:
    """Site metadata must describe every label present in nmeasure."""
    dt = _prepared_inputs().to_datatree()
    dt["site_metadata"].ds = dt["site_metadata"].to_dataset().assign_coords(site=["WRONG", "LABELS"])

    with pytest.raises(ValueError, match="missing observed site labels"):
        RhimePreparedInputs.from_datatree(dt)


def test_validated_rechecks_mutable_xarray_contents() -> None:
    """Execution and serialization can recheck nested xarray state after mutation."""
    prepared = _prepared_inputs()
    prepared.site_metadata.coords["site"] = ["WRONG", "LABELS"]

    with pytest.raises(ValueError, match="missing observed site labels"):
        prepared.validated()


def test_validated_rechecks_mutated_basis_source_labels() -> None:
    """Mutable retained-flux labels are rechecked before execution or storage."""
    prepared = _multisource_prepared_inputs()
    prepared.basis_functions.flux.coords["source"] = ["B", "C"]

    with pytest.raises(ValueError, match="flux labels must exactly match basis"):
        prepared.validated()
    with pytest.raises(ValueError, match="flux labels must exactly match basis"):
        prepared.to_datatree()


def test_prepared_inputs_rejects_non_site_metadata_variable() -> None:
    """Extension metadata remains labeled along the site dimension."""
    original = _prepared_inputs()
    site_metadata = original.site_metadata.assign(extra=("other", [1]))

    with pytest.raises(ValueError, match="variables must each have only dimension 'site'"):
        RhimePreparedInputs(
            inv_inputs=original.inv_inputs,
            basis_functions=original.basis_functions,
            site_metadata=site_metadata,
        )


def test_prepared_inputs_rejects_non_site_metadata_coordinate() -> None:
    """Auxiliary metadata coordinates cannot introduce unrelated dimensions."""
    original = _prepared_inputs()
    site_metadata = original.site_metadata.assign_coords(other=("other", [1]))

    with pytest.raises(ValueError, match="auxiliary coordinates must be scalar or site-aligned"):
        RhimePreparedInputs(
            inv_inputs=original.inv_inputs,
            basis_functions=original.basis_functions,
            site_metadata=site_metadata,
        )


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
    sampled = az.InferenceData(
        posterior=xr.Dataset(
            {"x": (("chain", "draw", "region"), np.ones((1, 1, 2)))},
            coords={"chain": [0], "draw": [0], "region": [0, 1]},
        )
    )
    observed: dict[str, Any] = {}
    original_builder = rhime_standard.build_standard_rhime_model

    def fail_preparation(*args: object, **kwargs: object) -> None:
        """Fail if execution attempts to prepare inputs again."""
        raise AssertionError("prepared-input execution must not repeat preparation")

    def fake_builder(
        flux_sensitivity: xr.DataArray,
        **kwargs: Any,
    ) -> pm.Model:
        """Record model-builder inputs and build the real canonical graph."""
        assert kwargs["likelihood_builder"] is None
        assert kwargs["preserve_legacy_likelihood"] is False
        observed["builder_inputs"] = flux_sensitivity
        observed["observations"] = kwargs["observations"]
        return original_builder(flux_sensitivity, **kwargs)

    def fake_sample(
        self: RhimeSampler,
        model: pm.Model,
        *,
        variable_roles: dict[str, str],
    ) -> az.InferenceData:
        """Record the sampled model and return the sentinel posterior."""
        observed["sample_model"] = model
        assert variable_roles["concentration"] == "y"
        return sampled

    def fake_outputs(**kwargs: Any) -> None:
        """Record output inputs and attach a sentinel product."""
        observed["output_prepared"] = kwargs["prepared"]
        observed["output_idata"] = kwargs["result"].idata
        kwargs["result"].outputs["loaded"] = True

    monkeypatch.setattr(rhime_standard, "retrieve_or_reload_rhime_data", fail_preparation)
    monkeypatch.setattr(rhime_standard, "build_standard_rhime_model", fake_builder)
    monkeypatch.setattr(RhimeSampler, "sample", fake_sample)
    monkeypatch.setattr(rhime_prepared, "make_standard_rhime_outputs", fake_outputs)

    result = run_rhime_from_prepared_inputs(
        prepared_inputs=loaded,
        run_spec=run_spec,
        sampler=RhimeSampler(draws=1, tune=1, chains=1),
    )

    builder_inputs = observed["builder_inputs"]
    output_prepared = observed["output_prepared"]
    assert isinstance(builder_inputs, xr.DataArray)
    assert isinstance(output_prepared, RhimePreparedInputs)
    xr.testing.assert_identical(builder_inputs, loaded.inv_inputs["H"])
    xr.testing.assert_identical(observed["observations"], loaded.inv_inputs["mf"])
    assert "y" in observed["sample_model"].named_vars
    assert observed["output_idata"] is sampled
    assert result.inv_inputs is output_prepared.inv_inputs
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
    sampled = az.InferenceData()
    observed: dict[str, object] = {}
    original_builder = rhime_multisector.build_multisector_rhime_model

    def fake_builder(
        flux_sensitivity: xr.DataArray,
        **kwargs: Any,
    ) -> pm.Model:
        """Record ordered builder inputs and build the real canonical graph."""
        assert kwargs["likelihood_builder"] is None
        observed["source_order"] = tuple(flux_sensitivity.source.values)
        observed["sectors"] = kwargs["sectors"]
        return original_builder(flux_sensitivity, **kwargs)

    def fake_sample(
        self: RhimeSampler,
        model: pm.Model,
        *,
        variable_roles: dict[str, str],
    ) -> az.InferenceData:
        """Return a sentinel trace for the multisource prepared run."""
        assert "y" in model.named_vars
        assert variable_roles["concentration"] == "y"
        return sampled

    def fake_outputs(**kwargs: Any) -> None:
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
        kwargs["result"].outputs["postprocessed"] = True

    monkeypatch.setattr(rhime_multisector, "build_multisector_rhime_model", fake_builder)
    monkeypatch.setattr(RhimeSampler, "sample", fake_sample)
    monkeypatch.setattr(rhime_prepared, "make_multisector_rhime_outputs", fake_outputs)

    result = run_rhime_from_prepared_inputs(
        prepared_inputs=loaded,
        run_spec=run_spec,
        sampler=RhimeSampler(draws=1, tune=1, chains=1),
    )

    assert observed["source_order"] == ("B", "A")
    assert observed["sectors"] is model_spec.sectors
    assert observed["output_order"] == ("B", "A")
    assert result.outputs == {"postprocessed": True}
