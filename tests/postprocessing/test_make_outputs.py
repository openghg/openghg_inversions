"""Tests for multisector flux reconstruction in modern postprocessing outputs."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.array_ops import concat_gather_data_arrays
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.flux_sanitization import FluxNonFiniteMetadata, NONFINITE_POLICY_ZERO_FILL
from openghg_inversions.postprocessing._basis_products import reconstruct_flux_stats
from openghg_inversions.postprocessing import make_outputs
from openghg_inversions.postprocessing.countries import Countries
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.make_outputs import (
    make_flux_outputs,
    make_multisector_country_trace_outputs,
    make_multisector_flux_trace_outputs,
)
from tests.helpers import make_trace


@pytest.mark.parametrize("source_specific", [False, True])
def test_stacked_flux_state_preserves_sector_flux_country_and_covariance(
    multisector_postprocessing_inv_out,
    fake_source_specific_multisector_basis_functions,
    source_specific,
):
    """A single labelled state reconstructs the same source products and joint totals."""
    basis = fake_source_specific_multisector_basis_functions() if source_specific else None
    legacy = multisector_postprocessing_inv_out(basis)
    groups = {}
    for group in ("prior", "posterior"):
        values = legacy.trace_group(group)
        # State storage order may differ from reporting-sector order.
        state = concat_gather_data_arrays(
            {"ocean-inventory": values.x_ocean, "ff-inventory": values.x_ff},
            key_dim="source", ragged_dim="region", stack_dim="state", join="exact",
        )
        groups[group] = xr.Dataset({"scale": state.chunk({"draw": 1})})
    trace = xr.DataTree.from_dict(groups)
    stacked = replace(legacy, trace=trace, model_metadata={
        **legacy.model_metadata,
        "variable_roles": {"flux_scale": "scale"},
    })
    borrowed = trace["posterior"]["scale"].data
    expected = make_flux_outputs(legacy, stats=["mean", "stdev"], report_flux_on_inversion_grid=False)
    actual = make_flux_outputs(stacked, stats=["mean", "stdev"], report_flux_on_inversion_grid=False)
    xr.testing.assert_allclose(actual, expected)
    assert actual.flux_total_posterior_stdev.item() == 0
    xr.testing.assert_allclose(
        make_multisector_flux_trace_outputs(stacked, report_flux_on_inversion_grid=False),
        make_multisector_flux_trace_outputs(legacy, report_flux_on_inversion_grid=False),
    )
    countries = object.__new__(Countries)
    countries.matrix = xr.DataArray(
        [[[1.0]]], dims=("country", "lat", "lon"),
        coords={"country": ["GBR"], "lat": [0.0], "lon": [0.0]},
    )
    countries.area_grid = countries.matrix.isel(country=0, drop=True)
    xr.testing.assert_allclose(
        make_multisector_country_trace_outputs(stacked, countries),
        make_multisector_country_trace_outputs(legacy, countries),
    )
    assert stacked.basis_functions is legacy.basis_functions
    assert stacked.inv_inputs is legacy.inv_inputs
    assert trace["posterior"]["scale"].data is borrowed


def test_stacked_flux_selector_restores_unequal_source_state_axes(multisector_postprocessing_inv_out):
    """Ragged source selection restores local labels without rectangular padding."""
    legacy = multisector_postprocessing_inv_out()
    flat = xr.DataArray(
        [[1, 2]], dims=("lat", "lon"), coords={"lat": [0.0], "lon": [0.0, 1.0]},
    )
    flux = xr.concat([xr.ones_like(flat), xr.ones_like(flat)],
                     dim=xr.IndexVariable("source", ["ff-inventory", "ocean-inventory"]))
    basis = BasisFunctions.from_multi_source_flat_basis(
        basis_flat={"ff-inventory": flat, "ocean-inventory": xr.ones_like(flat)},
        flux=flux, operator_kwargs={"state_dim": "region"},
    )
    values = xr.DataArray(
        [[[2.0, 3.0, 4.0]]], dims=("chain", "draw", "region"),
        coords={"chain": [0], "draw": [0], "region": basis.operator.basis_matrix.region},
    )
    stacked = replace(legacy, basis_functions=basis,
                      trace=make_trace(posterior=xr.Dataset({"x": values})),
                      model_metadata={**legacy.model_metadata, "variable_roles": {"flux_scale": "x"}})
    sectors = make_outputs._multisector_output_sectors(stacked)
    first, second = [make_outputs._sector_scale_trace(stacked, sector) for sector in sectors]
    assert first.x_posterior.dims == ("chain", "draw", "region")
    np.testing.assert_array_equal(first.region, [0, 1])
    np.testing.assert_array_equal(second.region, [0])
    np.testing.assert_array_equal(first.x_posterior, [[[2.0, 3.0]]])
    np.testing.assert_array_equal(second.x_posterior, [[[4.0]]])


def test_declared_stacked_flux_role_does_not_fall_back_to_legacy_variables(multisector_postprocessing_inv_out):
    """A broken canonical contract cannot silently select historical source states."""
    legacy = multisector_postprocessing_inv_out()
    declared = replace(legacy, model_metadata={**legacy.model_metadata, "variable_roles": {"flux_scale": "x"}})
    sector = make_outputs._multisector_output_sectors(declared)[0]
    with pytest.raises(ValueError, match="Could not find flux scale trace"):
        make_outputs._sector_scale_trace(declared, sector)


def test_legacy_sector_trace_ignores_unrelated_default_variable_name(multisector_postprocessing_inv_out):
    """Historical source states remain authoritative without a generic flux role."""
    legacy = multisector_postprocessing_inv_out()
    posterior = legacy.trace_group("posterior")
    legacy.trace["posterior"] = posterior.assign(x=posterior.x_ff + 99)
    sector = make_outputs._multisector_output_sectors(legacy)[0]
    selected = make_outputs._sector_scale_trace(legacy, sector)
    xr.testing.assert_allclose(selected.x_posterior, posterior.x_ff)


def _flux_nonfinite_metadata(data: xr.DataArray | xr.Dataset) -> FluxNonFiniteMetadata:
    """Return parsed non-finite flux metadata from an xarray object."""
    metadata = FluxNonFiniteMetadata.from_attrs(data.attrs)
    assert metadata is not None
    return metadata


def test_flux_reconstruction_does_not_guess_recipe_specific_state_dimension() -> None:
    """Generic reconstruction requires recipes to normalize their state dimensions."""
    basis = xr.DataArray(
        [[1, 2]],
        dims=("lat", "lon"),
        coords={"lat": [50.0], "lon": [-2.0, -1.0]},
    )
    flux = xr.ones_like(basis, dtype=float).rename("flux")
    basis_functions = BasisFunctions.from_flat_basis(
        basis,
        flux,
        operator_kwargs={"state_dim": "region"},
    )
    stats = xr.Dataset(
        {
            "x_posterior_mean": ("inner_region", [2.0, 3.0]),
            "x_posterior_quantile": (
                ("inner_region", "quantile"),
                [[1.5, 2.5], [2.5, 3.5]],
            ),
        },
        coords={"inner_region": [0, 1], "quantile": [0.16, 0.84]},
    )

    with pytest.raises(ValueError, match="Could not find a basis state dimension"):
        reconstruct_flux_stats(
            basis_functions,
            flux,
            stats,
            report_flux_on_inversion_grid=False,
        )


def test_multisector_flux_outputs_reconstruct_sector_and_total_flux(
    multisector_postprocessing_inv_out: Callable[..., InversionOutput],
) -> None:
    """Multisector flux postprocessing reconstructs sector fluxes before total statistics."""
    outputs = make_flux_outputs(
        multisector_postprocessing_inv_out(),
        stats=["mean", "stdev"],
        include_scale_factors=True,
        report_flux_on_inversion_grid=False,
    )

    assert "flux_ff_posterior_mean" in outputs
    assert "flux_ocean_posterior_mean" in outputs
    assert "scaling_ff_posterior_mean" in outputs
    assert "flux_total_posterior_mean" in outputs
    assert "flux_total_posterior_stdev" in outputs
    assert outputs["flux_ff_posterior_mean"].attrs["units"] == "mol/m2/s"
    assert outputs["flux_total_posterior_mean"].attrs["units"] == "mol/m2/s"
    assert float(outputs["flux_total_posterior_mean"].item()) == 2.0
    assert float(outputs["flux_total_posterior_stdev"].item()) == 0.0
    assert float(outputs["flux_ff_posterior_stdev"].item()) > 0.0
    assert _flux_nonfinite_metadata(outputs).policy == NONFINITE_POLICY_ZERO_FILL


def test_multisector_flux_outputs_support_inversion_grid_mean_stats(
    multisector_postprocessing_inv_out: Callable[..., InversionOutput],
) -> None:
    """Multisector inversion-grid stats work when statistics do not need dense quantiles."""
    outputs = make_flux_outputs(
        multisector_postprocessing_inv_out(),
        stats=["mean", "stdev"],
        include_scale_factors=False,
        report_flux_on_inversion_grid=True,
    )

    assert "flux_ff_posterior_mean" in outputs
    assert "flux_ocean_posterior_mean" in outputs
    assert "flux_total_posterior_mean" in outputs
    assert float(outputs["flux_total_posterior_mean"].item()) == 2.0
    mean_only = make_flux_outputs(
        multisector_postprocessing_inv_out(), stats=["mean"],
        include_scale_factors=False, report_flux_on_inversion_grid=True,
    )
    xr.testing.assert_allclose(mean_only, outputs[[name for name in outputs if str(name).endswith("_mean")]])


@pytest.mark.parametrize("source_specific", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("missing", ["none", "common"])
def test_multisector_mean_reduces_state_before_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
    multisector_postprocessing_inv_out: Callable[..., InversionOutput],
    fake_source_specific_multisector_basis_functions: Callable[..., BasisFunctions],
    source_specific: bool,
    dtype: str,
    missing: str,
) -> None:
    """Linear means match per-draw reconstruction with signed flux and padded groups."""
    basis = fake_source_specific_multisector_basis_functions() if source_specific else None
    inv_out = multisector_postprocessing_inv_out(basis)
    flux = inv_out.basis_functions.flux.astype(dtype) * xr.DataArray(
        [1, -1], dims="source", coords={"source": inv_out.basis_functions.flux.source}
    )
    flux = flux.expand_dims(time=np.array(["2019-01-01", "2019-01-02"], dtype="datetime64[ns]"))
    flux = flux * xr.DataArray([1, 2], dims="time", coords={"time": flux.time})
    flux = flux.astype(dtype)
    flux = flux.chunk({"lon": 1})
    inv_out.basis_functions = inv_out.basis_functions.with_flux(flux)
    ff = np.arange(1, 9, dtype=dtype).reshape(2, 4, 1)
    ocean = (ff * 2).copy()
    if missing == "common":
        ff[:, 0] = ocean[:, 0] = np.nan
    posterior = xr.Dataset(
        {"x_ff": (("chain", "draw", "region"), ff),
         "x_ocean": (("chain", "draw", "region"), ocean)},
        coords={"chain": [0, 1], "draw": np.arange(4), "region": [0]},
    )
    # Real merged traces pad the prior's missing chains and draws with NaNs.
    prior = posterior.isel(chain=[0], draw=slice(0, 3)).fillna(1)
    inv_out.trace = make_trace(
        posterior=posterior.chunk({"draw": 2}),
        prior=prior.chunk({"draw": 1}),
    )
    before = posterior.copy(deep=True)
    retained_flux = inv_out.basis_functions.flux
    reconstruct = make_outputs.reconstruct_flux_stats
    baseline = make_flux_outputs(
        inv_out, stats=["mean", "stdev"], report_flux_on_inversion_grid=False,
    )
    expected = baseline[[name for name in baseline if str(name).endswith("_mean")]]

    def reconstruct_means(*args: object, **kwargs: object) -> xr.Dataset:
        state = args[2]
        assert isinstance(state, xr.Dataset)
        assert not {"chain", "draw", "sample"}.intersection(state.dims)
        return reconstruct(*args, **kwargs)

    monkeypatch.setattr(make_outputs, "reconstruct_flux_stats", reconstruct_means)
    actual = make_flux_outputs(
        inv_out, stats=["mean"], report_flux_on_inversion_grid=False,
    )

    xr.testing.assert_allclose(actual, expected)
    for name in actual:
        assert actual[name].dtype == expected[name].dtype
        assert actual[name].dtype == np.dtype(dtype)
        assert actual[name].attrs == expected[name].attrs
    assert actual.attrs == expected.attrs
    xr.testing.assert_identical(inv_out.trace_group("posterior"), before)
    assert inv_out.basis_functions.flux is retained_flux
    assert retained_flux.chunks is not None


def test_automatic_multisector_diagnostics_reconstruct_only_means(
    monkeypatch: pytest.MonkeyPatch,
    multisector_postprocessing_inv_out: Callable[..., InversionOutput],
) -> None:
    """The automatic diagnostic writer never constructs sample-by-native-grid flux."""
    from openghg_inversions.rhime.outputs import _make_multisector_flux_diagnostics

    inv_out = multisector_postprocessing_inv_out()
    reconstruct = make_outputs.reconstruct_flux_stats

    def reconstruct_means(*args: object, **kwargs: object) -> xr.Dataset:
        assert not {"chain", "draw", "sample"}.intersection(args[2].dims)
        return reconstruct(*args, **kwargs)

    monkeypatch.setattr(make_outputs, "reconstruct_flux_stats", reconstruct_means)
    actual = _make_multisector_flux_diagnostics(inv_out)
    assert actual["flux_total_posterior_mean"].item() == 2


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf])
def test_multisector_mean_falls_back_for_partial_state_nonfinite(
    monkeypatch: pytest.MonkeyPatch,
    multisector_postprocessing_inv_out: Callable[..., InversionOutput],
    nonfinite: float,
) -> None:
    """State-dependent missingness and infinities preserve the per-draw result."""
    basis = xr.DataArray([[1, 2]], dims=("lat", "lon"), coords={"lat": [0.0], "lon": [0.0, 1.0]})
    flux = xr.DataArray(
        [[[0.0, 1.0]], [[3.0, -2.0]]], dims=("source", "lat", "lon"),
        coords={"source": ["ff-inventory", "ocean-inventory"], "lat": [0.0], "lon": [0.0, 1.0]},
    )
    inv_out = multisector_postprocessing_inv_out(
        BasisFunctions.from_flat_basis(basis, flux, operator_kwargs={"state_dim": "region"})
    )
    inv_out.trace = make_trace(posterior=xr.Dataset(
        {"x_ff": (("chain", "draw", "region"), [[[nonfinite, 2.0], [3.0, 4.0]]]),
         "x_ocean": (("chain", "draw", "region"), [[[1.0, 2.0], [3.0, 4.0]]])},
        coords={"chain": [0], "draw": [0, 1], "region": [0, 1]},
    ))
    baseline = make_flux_outputs(inv_out, stats=["mean", "stdev"], report_flux_on_inversion_grid=False)
    reconstruct = make_outputs.reconstruct_flux_stats
    calls = []

    def reconstruct_draws(*args: object, **kwargs: object) -> xr.Dataset:
        calls.append(tuple(args[2].dims))
        return reconstruct(*args, **kwargs)

    monkeypatch.setattr(make_outputs, "reconstruct_flux_stats", reconstruct_draws)
    actual = make_flux_outputs(inv_out, stats=["mean"], report_flux_on_inversion_grid=False)
    xr.testing.assert_allclose(actual, baseline[[name for name in baseline if str(name).endswith("_mean")]])
    assert calls and all("draw" in dims for dims in calls)


@pytest.mark.parametrize("stats,stats_args,optimized", [
    (None, {"stats": ["mean"]}, True),
    (["mean"], {"stats": ["stdev"]}, True),
    (None, None, False),
    (["mean", "stdev"], None, False),
    (["mean"], {"sample_dim": "lat"}, False),
    (["mean"], {"mean__sample_dim": "sample"}, False),
])
def test_multisector_mean_dispatch_respects_effective_stats(
    monkeypatch: pytest.MonkeyPatch,
    multisector_postprocessing_inv_out: Callable[..., InversionOutput],
    stats: list[str] | None,
    stats_args: dict | None,
    optimized: bool,
) -> None:
    """Only effective mean requests with the standard sample reduction use the shortcut."""
    inv_out = multisector_postprocessing_inv_out()
    calls = []
    mean_parts = make_outputs._multisector_flux_mean_parts

    def record_mean_parts(*args: object, **kwargs: object) -> object:
        calls.append(True)
        return mean_parts(*args, **kwargs)

    monkeypatch.setattr(make_outputs, "_multisector_flux_mean_parts", record_mean_parts)
    make_flux_outputs(
        inv_out, stats=stats, stats_args=stats_args,
        include_scale_factors=False, report_flux_on_inversion_grid=False,
    )
    assert bool(calls) == optimized


def test_multisector_flux_outputs_support_source_specific_basis(
    monkeypatch: pytest.MonkeyPatch,
    fake_source_specific_multisector_basis_functions: Callable[..., BasisFunctions],
    multisector_postprocessing_inv_out: Callable[..., InversionOutput],
) -> None:
    """Sector flux reconstruction handles source-specific retained basis artifacts."""
    basis_functions = fake_source_specific_multisector_basis_functions()

    def fail_flat_basis(self: BasisFunctions) -> xr.DataArray:
        raise AssertionError("source-specific multisector outputs should not materialise flat basis")

    monkeypatch.setattr(type(basis_functions), "flat_basis", fail_flat_basis)

    outputs = make_flux_outputs(
        multisector_postprocessing_inv_out(basis_functions),
        stats=["mean", "mode_kde"],
        include_scale_factors=False,
        report_flux_on_inversion_grid=False,
    )

    assert "flux_total_posterior_mean" in outputs
    assert "flux_total_posterior_mode" in outputs
    assert float(outputs["flux_total_posterior_mean"].item()) == 2.0
    assert _flux_nonfinite_metadata(outputs).policy == NONFINITE_POLICY_ZERO_FILL


def test_multisector_mode_kde_scale_factors_use_each_sector_state_dimension(
    multisector_postprocessing_inv_out: Callable[..., InversionOutput],
) -> None:
    """Ragged sector scale-factor modes chunk each sector along its own state dimension."""
    ff_basis = xr.DataArray(
        [[1, 2]],
        dims=("lat", "lon"),
        coords={"lat": [0.0], "lon": [0.0, 1.0]},
    )
    ocean_basis = xr.DataArray(
        [[1, 1]],
        dims=("lat", "lon"),
        coords={"lat": [0.0], "lon": [0.0, 1.0]},
    )
    flux = xr.DataArray(
        np.ones((2, 1, 2)),
        dims=("source", "lat", "lon"),
        coords={
            "source": ["ff-inventory", "ocean-inventory"],
            "lat": [0.0],
            "lon": [0.0, 1.0],
        },
        attrs={"units": "mol/m2/s"},
    )
    basis_functions = BasisFunctions.from_multi_source_flat_basis(
        basis_flat={
            "ff-inventory": ff_basis,
            "ocean-inventory": ocean_basis,
        },
        flux=flux,
        operator_kwargs={"state_dim": "state"},
    )
    inv_out = multisector_postprocessing_inv_out(basis_functions)
    inv_out.trace = make_trace(
        posterior=xr.Dataset(
            {
                "x_ff": (
                    ("chain", "draw", "state_ff"),
                    np.array([[[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5]]]),
                ),
                "x_ocean": (
                    ("chain", "draw", "state_ocean"),
                    np.array([[[0.5], [1.0], [1.5], [2.0]]]),
                ),
            },
            coords={"chain": [0], "draw": np.arange(4), "state_ff": [0, 1], "state_ocean": [0]},
        )
    )

    outputs = make_flux_outputs(
        inv_out,
        stats=["mode_kde"],
        include_scale_factors=True,
        report_flux_on_inversion_grid=False,
    )

    assert "scaling_ff_posterior_mode" in outputs
    assert "scaling_ocean_posterior_mode" in outputs


def test_multisector_flux_trace_materialization_is_optional(
    multisector_postprocessing_inv_out: Callable[..., InversionOutput],
) -> None:
    """The trace API preserves its eager default and exposes a lazy internal boundary."""
    inv_out = multisector_postprocessing_inv_out()

    lazy_trace = make_multisector_flux_trace_outputs(
        inv_out,
        report_flux_on_inversion_grid=False,
        materialize=False,
    )
    materialized_trace = make_multisector_flux_trace_outputs(
        inv_out,
        report_flux_on_inversion_grid=False,
    )

    assert not isinstance(lazy_trace["flux_total_posterior"].data, np.ndarray)
    assert isinstance(materialized_trace["flux_total_posterior"].data, np.ndarray)


def test_multisector_flux_total_preserves_all_missing_draws(
    multisector_postprocessing_inv_out: Callable[..., InversionOutput],
) -> None:
    """Sector totals keep padding from unequal trace-group draw counts missing."""
    inv_out = multisector_postprocessing_inv_out()
    prior = inv_out.trace_group("prior")
    extra_prior_draw = prior.isel(draw=[0]).assign_coords(draw=[prior.sizes["draw"]])
    inv_out.trace["prior"] = xr.concat([prior, extra_prior_draw], dim="draw")

    trace = make_multisector_flux_trace_outputs(
        inv_out,
        report_flux_on_inversion_grid=False,
    )

    sector_total = trace["flux_ff_posterior"] + trace["flux_ocean_posterior"]
    xr.testing.assert_allclose(trace["flux_total_posterior"], sector_total)
    assert trace["flux_total_posterior"].isel(draw=-1).isnull().all()


def test_multisector_flux_total_requires_each_sector_per_draw(
    monkeypatch: pytest.MonkeyPatch,
    multisector_postprocessing_inv_out: Callable[..., InversionOutput],
) -> None:
    """A structural zero does not hide another sector's padded missing draw."""
    inv_out = multisector_postprocessing_inv_out()

    def sector_trace(
        output: InversionOutput,
        sector: make_outputs.OutputSector,
        *,
        report_flux_on_inversion_grid: bool,
    ) -> xr.Dataset:
        del output, report_flux_on_inversion_grid
        values = (
            np.asarray([0.0, 0.0, 0.0]) if sector.variable_suffix == "ff" else np.asarray([1.0, 2.0, np.nan])
        )
        return xr.Dataset({f"flux_{sector.variable_suffix}_posterior": ("draw", values)})

    monkeypatch.setattr(make_outputs, "_sector_flux_trace_dataset", sector_trace)

    trace = make_multisector_flux_trace_outputs(inv_out)

    np.testing.assert_allclose(trace["flux_total_posterior"].isel(draw=slice(0, 2)), [1.0, 2.0])
    assert trace["flux_total_posterior"].isel(draw=-1).isnull()


def test_multisector_country_total_requires_each_sector_per_draw(
    monkeypatch: pytest.MonkeyPatch,
    multisector_postprocessing_inv_out: Callable[..., InversionOutput],
) -> None:
    """Country totals also reject draws missing any sector contribution."""
    inv_out = multisector_postprocessing_inv_out()
    countries = object.__new__(Countries)
    countries.matrix = xr.DataArray(
        [[[1.0]]],
        dims=("country", "lat", "lon"),
        coords={"country": ["GBR"], "lat": [0.0], "lon": [0.0]},
    )
    countries.area_grid = xr.DataArray(
        [[1.0]],
        dims=("lat", "lon"),
        coords={"lat": [0.0], "lon": [0.0]},
    )

    def scale_trace(
        output: InversionOutput,
        sector: make_outputs.OutputSector,
    ) -> xr.Dataset:
        del output
        values = (
            np.asarray([0.0, 0.0, 0.0]) if sector.variable_suffix == "ff" else np.asarray([1.0, 2.0, np.nan])
        )
        return xr.Dataset({"x_posterior": ("draw", values)})

    def project_country_trace(
        species: str,
        trace: xr.Dataset,
        x_to_country: xr.DataArray,
    ) -> xr.Dataset:
        del species, x_to_country
        return trace.expand_dims(country=["GBR"], flux_time=[np.datetime64("2019-01-01")])

    monkeypatch.setattr(make_outputs, "_sector_scale_trace", scale_trace)
    monkeypatch.setattr(
        make_outputs,
        "_sector_basis_functions",
        lambda *args: inv_out.basis_functions,
    )
    monkeypatch.setattr(
        make_outputs,
        "make_x_to_country_matrix",
        lambda *args, **kwargs: xr.DataArray([1.0]),
    )
    monkeypatch.setattr(Countries, "_get_country_trace", staticmethod(project_country_trace))

    trace = make_multisector_country_trace_outputs(inv_out, countries)

    np.testing.assert_allclose(
        trace["country_total_posterior"].isel(draw=slice(0, 2)).squeeze(),
        [1.0, 2.0],
    )
    assert trace["country_total_posterior"].isel(draw=-1).isnull().all()


def test_multisector_country_traces_project_basis_regions_before_scaling(
    europe_country_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_multisector_basis_functions_matching_country_grid: Callable[..., BasisFunctions],
    multisector_postprocessing_inv_out: Callable[..., InversionOutput],
) -> None:
    """Country traces map basis regions without reconstructing draw-wise spatial flux."""
    inv_out = multisector_postprocessing_inv_out(
        fake_multisector_basis_functions_matching_country_grid(europe_country_file)
    )
    countries = Countries.from_file(
        country_file=europe_country_file,
        country_code="alpha3",
        country_selections=["FRA", "GBR"],
        domain="EUROPE",
    )

    def fail_spatial_reconstruction(*args: object, **kwargs: object) -> xr.Dataset:
        """Fail if country totals reconstruct a latitude/longitude posterior trace."""
        raise AssertionError("country projection must happen before spatial reconstruction")

    monkeypatch.setattr(make_outputs, "_sector_flux_trace_dataset", fail_spatial_reconstruction)

    country_trace = make_multisector_country_trace_outputs(inv_out, countries)

    assert tuple(country_trace.country.values) == tuple(countries.matrix.country.values)
    assert set(country_trace.country.values) == {"FRA", "GBR"}
    assert "lat" not in country_trace.dims
    assert "lon" not in country_trace.dims
    assert {
        "country_total_posterior",
        "country_ff_posterior",
        "country_ocean_posterior",
    }.issubset(country_trace.data_vars)
    assert country_trace["country_total_posterior"].chunks is not None
    for when in ("prior", "posterior"):
        np.testing.assert_allclose(
            country_trace[f"country_total_{when}"].isel(draw=0),
            country_trace[f"country_ff_{when}"].isel(draw=0)
            + country_trace[f"country_ocean_{when}"].isel(draw=0),
        )
        assert country_trace[f"country_total_{when}"].attrs["units"] == "g/yr"
