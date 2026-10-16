"""Tests for multisector flux reconstruction in modern postprocessing outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, cast

import arviz as az
import numpy as np
import pytest
import xarray as xr

from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.flux_sanitization import FluxNonFiniteMetadata, NONFINITE_POLICY_ZERO_FILL
from openghg_inversions.postprocessing import make_outputs
from openghg_inversions.postprocessing.countries import Countries
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.make_outputs import (
    make_flux_outputs,
    make_multisector_country_trace_outputs,
    make_multisector_flux_trace_outputs,
)


def _flux_nonfinite_metadata(data: xr.DataArray | xr.Dataset) -> FluxNonFiniteMetadata:
    """Return parsed non-finite flux metadata from an xarray object."""
    metadata = FluxNonFiniteMetadata.from_attrs(data.attrs)
    assert metadata is not None
    return metadata


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


@pytest.mark.parametrize(
    "probability_key, shared_generator",
    [("hdi_prob", False), ("hdi__hdi_prob", False), ("hdi__hdi_prob", True)],
    ids=["generic", "override", "shared-generator"],
)
def test_multisector_flux_tiles_reuse_hdi_probability_generators(
    multisector_postprocessing_inv_out: Callable[..., InversionOutput],
    probability_key: str,
    shared_generator: bool,
) -> None:
    """HDI generators cover every tile, trace group, sector, and scale factor."""
    labels = xr.DataArray(
        np.ones((33, 1), dtype=int),
        dims=("lat", "lon"),
        coords={"lat": np.arange(33, dtype=float), "lon": [0.0]},
    )
    flux = xr.concat([labels.astype(float), labels * 3.0], dim="source").assign_coords(
        source=["ff-inventory", "ocean-inventory"]
    )
    flux.attrs["units"] = "mol/m2/s"
    basis = BasisFunctions.from_flat_basis(
        basis_flat=labels, flux=flux, operator_kwargs={"state_dim": "region"}
    )
    inv_out = multisector_postprocessing_inv_out(basis)
    probabilities = (0.68, 0.95)
    options = {"hdi_prob": 0.5, probability_key: probabilities}
    expected = make_flux_outputs(
        inv_out, stats=["hdi"], stats_args=options, report_flux_on_inversion_grid=False
    )
    generator = (probability for probability in probabilities)
    options[probability_key] = generator
    if shared_generator:
        options["hdi_prob"] = generator
    actual = make_flux_outputs(
        inv_out, stats=["hdi"], stats_args=options, report_flux_on_inversion_grid=False
    )

    xr.testing.assert_identical(actual, expected)
    assert options[probability_key] is generator
    if shared_generator:
        assert options["hdi_prob"] is generator
    assert set(actual.data_vars) == {
        f"{product}_{group}_hdi_{probability}"
        for product in ("flux_total", "flux_ff", "flux_ocean", "scaling_ff", "scaling_ocean")
        for group in ("prior", "posterior")
        for probability in (68, 95)
    }


@pytest.mark.parametrize("chunks", [None, {"lat": 1, "lon": 1, "flux_time": 1}])
def test_inversion_grid_signed_sparse_flux_preserves_lazy_region_means(chunks) -> None:
    """Signed sparse fills remain compatible across spatial and time chunks."""
    from dask.callbacks import Callback
    import sparse

    from openghg_inversions.array_ops import to_dense
    from openghg_inversions.postprocessing._basis_products import (
        _operator_region_flux_mean,
        reconstruct_flux_stats,
    )

    labels = xr.DataArray(
        [[1, 2], [1, 2]], dims=("lat", "lon"), coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]}
    )
    flux = xr.DataArray(
        np.array(
            [[[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], [[3.0, 4.0, 5.0], [-3.0, -4.0, -5.0]]],
            dtype="float32",
        ),
        dims=("lat", "lon", "flux_time"),
        coords={**labels.coords, "flux_time": [0, 1, 2]},
    ).chunk({"lat": 1, "lon": 1, "flux_time": 2})
    borrowed = flux.data
    graph = borrowed.__dask_graph__()
    basis = BasisFunctions.from_flat_basis(
        basis_flat=labels, flux=flux, operator_kwargs={"state_dim": "region"}
    )
    tasks = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):
        means = _operator_region_flux_mean(basis, flux)
    assert not tasks
    assert isinstance(means.data._meta, sparse.COO)
    assert means.dtype == flux.dtype
    dense_flux = flux.compute()
    expected_means = xr.concat(
        [dense_flux.where(labels == label).mean(["lat", "lon"]) for label in (1, 2)],
        dim=xr.IndexVariable("region", [0, 1]),
    )
    xr.testing.assert_allclose(to_dense(means).compute().transpose("region", "flux_time"), expected_means)

    state = xr.Dataset({"x": ("region", np.array([2.0, 3.0], dtype="float32"))}, coords={"region": [0, 1]})
    result = reconstruct_flux_stats(
        basis, flux, state, report_flux_on_inversion_grid=True, _chunks=chunks
    )
    expected = (expected_means * state.x).isel(region=labels - 1).drop_vars("region")
    xr.testing.assert_allclose(to_dense(result.x).compute(), expected.transpose(*result.x.dims))
    assert result.x.dtype == flux.dtype
    assert flux.data is borrowed
    assert flux.data.__dask_graph__() is graph


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
    inv_out.trace = az.from_dict(
        posterior={
            "x_ff": np.array([[[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5]]]),
            "x_ocean": np.array([[[0.5], [1.0], [1.5], [2.0]]]),
        },
        coords={"state_ff": [0, 1], "state_ocean": [0]},
        dims={"x_ff": ["state_ff"], "x_ocean": ["state_ocean"]},
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
    inference_data = cast(Any, inv_out.trace)
    prior = inference_data.prior
    extra_prior_draw = prior.isel(draw=[0]).assign_coords(draw=[prior.sizes["draw"]])
    inference_data.prior = xr.concat([prior, extra_prior_draw], dim="draw")

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
        chunks: dict[str, int] | None = None,
    ) -> xr.Dataset:
        del output, report_flux_on_inversion_grid, chunks
        values = (
            np.asarray([0.0, 0.0, 0.0])
            if sector.variable_suffix == "ff"
            else np.asarray([1.0, 2.0, np.nan])
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
            np.asarray([0.0, 0.0, 0.0])
            if sector.variable_suffix == "ff"
            else np.asarray([1.0, 2.0, np.nan])
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


@pytest.mark.parametrize("source_specific", [False, True], ids=["shared-basis", "source-specific-basis"])
@pytest.mark.parametrize("inversion_grid", [False, True], ids=["native-grid", "region-mean-grid"])
def test_multisector_flux_tiles_preserve_statistics_and_bound_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
    multisector_postprocessing_inv_out: Callable[..., InversionOutput],
    source_specific: bool,
    inversion_grid: bool,
) -> None:
    """Bounded tiles retain the original kernels, paired draws, and full region means."""
    from dask.callbacks import Callback

    from openghg_inversions.postprocessing.stats import calculate_stats

    coords = {"lat": [4.0, 3.0, 2.0, 1.0, 0.0], "lon": np.arange(7, dtype=float)}
    labels = xr.DataArray(
        np.where(np.indices((5, 7))[1] < 3, 1, 2), dims=("lat", "lon"), coords=coords
    )
    signed = (np.arange(35).reshape(5, 7) % 7 - 3).astype("float32")
    flux = xr.DataArray(
        np.stack([signed, signed[::-1] / 2 + 1])[..., None] * np.array([1, 2, 3], dtype="float32"),
        dims=("source", "lat", "lon", "flux_time"),
        coords={
            "source": ["ff-inventory", "ocean-inventory"],
            "flux_time": np.array(["2019-01-01", "2019-01-02", "2019-01-03"], dtype="datetime64[ns]"),
            **coords,
        },
        attrs={"units": "mol/m2/s"},
    )
    if inversion_grid:
        flux = flux.isel(flux_time=[0])
    if source_specific:
        basis = BasisFunctions.from_multi_source_flat_basis(
            basis_flat={"ff-inventory": labels, "ocean-inventory": xr.ones_like(labels)},
            flux=flux,
            operator_kwargs={"state_dim": "region"},
        )
    else:
        basis = BasisFunctions.from_flat_basis(
            basis_flat=labels, flux=flux, operator_kwargs={"state_dim": "region"}
        )
    inv_out = multisector_postprocessing_inv_out(basis)
    ocean_regions = 1 if source_specific else 2
    draws = np.arange(20, dtype="float32")
    ff = np.stack([draws % 5 - 2, draws % 3 - 1], axis=-1)
    ocean = np.broadcast_to((2 - draws % 5)[:, None], (20, ocean_regions)).copy()
    inv_out.trace = az.from_dict(
        posterior={"x_ff": ff[None, :16], "x_ocean": ocean[None, :16]},
        prior={"x_ff": ff[None], "x_ocean": ocean[None]},
        coords=(
            {"state_ff": [0, 1], "state_ocean": np.arange(ocean_regions)}
            if source_specific else {"region": [0, 1]}
        ),
        dims=(
            {"x_ff": ["state_ff"], "x_ocean": ["state_ocean"]}
            if source_specific else {"x_ff": ["region"], "x_ocean": ["region"]}
        ),
    )
    # A different second chain must not silently change the stable converter's behavior.
    for group in ("prior", "posterior"):
        original = getattr(inv_out.trace, group)
        two_chains = xr.concat([original, original + 100], dim="chain").assign_coords(chain=[0, 1])
        setattr(inv_out.trace, group, two_chains)
    borrowed = (inv_out.trace, inv_out.basis_functions.operator, inv_out.flux.data)
    trace_before = {group: getattr(inv_out.trace, group).copy(deep=True) for group in ("prior", "posterior")}
    flux_before = inv_out.flux.copy(deep=True)
    stats = ["mean", "stdev"] if inversion_grid else ["mean", "stdev", "quantiles", "hdi"]
    options = {"stats": stats, "hdi_prob": [0.68, 0.95]}

    sectors, traces, total = make_outputs._multisector_flux_trace_parts(
        inv_out, report_flux_on_inversion_grid=inversion_grid
    )
    assert total["flux_total_posterior"].isel(draw=-1).isnull().all().compute()
    expected = xr.merge([calculate_stats(item, **options) for item in [total, *traces]])
    expected = make_outputs._set_multisector_flux_attrs(expected, inv_out, sectors).as_numpy()
    expected = make_outputs._copy_first_flux_nonfinite_metadata(expected, [total, *traces])
    expected = make_outputs.add_basis_reconstruction_metadata(expected, inv_out.basis_functions)

    monkeypatch.setattr(make_outputs, "_FLUX_STATS_CHUNKS", {"lat": 2, "lon": 2, "flux_time": 1})
    monkeypatch.setattr(
        "openghg_inversions.postprocessing.stats._STATS_CHUNKS", {"lat": 2, "lon": 2, "flux_time": 1}
    )
    sample_payload_sizes = []

    def record_payload(key, value, dsk, state, worker_id):
        if isinstance(value, np.ndarray) and 20 in value.shape:
            sample_payload_sizes.append(value.size)

    with Callback(posttask=record_payload):
        actual = make_flux_outputs(
            inv_out,
            stats=options["stats"],
            stats_args={"hdi_prob": options["hdi_prob"]},
            include_scale_factors=False,
            report_flux_on_inversion_grid=inversion_grid,
        )

    for name in expected.data_vars:
        assert actual[name].dtype == expected[name].dtype
        if name.endswith("_stdev"):
            # Changing spatial strides may move float32 standard deviations by one ULP.
            np.testing.assert_array_max_ulp(actual[name].values, expected[name].values, maxulp=1)
            xr.testing.assert_identical(actual[name].copy(data=expected[name].data), expected[name])
        else:
            xr.testing.assert_identical(actual[name], expected[name])
    xr.testing.assert_identical(actual.coords.to_dataset(), expected.coords.to_dataset())
    assert actual.attrs == expected.attrs
    if not inversion_grid:
        assert list(actual.hdi.values) == ["lower", "higher"]
        assert actual.hdi.attrs == {"hdi_prob": 0.68}
    assert sample_payload_sizes
    # At most two sector arrays for one 2x2 tile, with every padded draw retained.
    assert max(sample_payload_sizes) <= 2 * 2 * 2 * 20
    assert inv_out.trace is borrowed[0]
    assert inv_out.basis_functions.operator is borrowed[1]
    assert inv_out.flux.data is borrowed[2]
    xr.testing.assert_identical(inv_out.flux, flux_before)
    for group, original in trace_before.items():
        xr.testing.assert_identical(getattr(inv_out.trace, group), original)
