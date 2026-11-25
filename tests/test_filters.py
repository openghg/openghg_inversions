import dask.array as da
from dask import delayed
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions import filters as filter_module
from openghg_inversions.filters import filtering, filtering_functions
from openghg_inversions.rhime.params import DataConfig
from openghg_inversions.rhime.preparation import acquire_merged_data
from openghg_inversions.inversion_data import SiteOptions
from openghg_inversions.rhime import filter_observations


@pytest.fixture
def merged_data(tac_ch4_data_args):
    return acquire_merged_data(
        DataConfig.from_params(tac_ch4_data_args, multisector=False)
    ).site_data


@pytest.mark.parametrize("longitude", [30.0, -30.0, 330.0])
@pytest.mark.parametrize("keep_missing", [False, True])
@pytest.mark.parametrize(
    "name,east_indices,west_indices",
    [
        ("daytime", list(range(9, 14)), list(range(13, 18))),
        ("daytime9to5", list(range(7, 16)), list(range(11, 20))),
        ("nighttime", [0, 1, 21, 22, 23], [1, 2, 3, 4, 5]),
        ("noon", [10], [14]),
    ],
)
def test_solar_filters_preserve_observations(name, east_indices, west_indices, longitude, keep_missing):
    """Local-hour selection keeps UTC labels and is repeatable on borrowed inputs."""
    dataset = xr.Dataset(
        {
            "mf": ("time", np.arange(24, dtype=float)),
            "release_lon": ("time", np.full(24, longitude)),
        },
        coords={"time": pd.date_range("2020-01-01", periods=24, freq="h")},
        attrs={"site": "TAC"},
    )
    dataset.time.attrs["long_name"] = "UTC observation time"
    original = dataset.copy(deep=True)
    inputs = {"TAC": dataset}
    selected = east_indices if longitude == 30.0 else west_indices
    expected = original.isel(time=selected)
    if keep_missing:
        expected = expected.reindex_like(original)

    result = filtering(inputs, name, keep_missing=keep_missing)
    repeated = filtering(inputs, name, keep_missing=keep_missing)

    xr.testing.assert_identical(dataset, original)
    xr.testing.assert_identical(result["TAC"], expected)
    xr.testing.assert_identical(repeated["TAC"], expected)


def test_pblh_filter_error(merged_data):
    with pytest.raises(NotImplementedError):
        filtering(merged_data, ["pblh"])


def test_all_filters(merged_data):
    for name in filtering_functions:
        if name != "pblh":
            filtering(merged_data, [name])


def test_filters_as_none(merged_data):
    filters = None
    filtering(merged_data, filters)


def test_filters_as_str(merged_data):
    filters = "pblh_inlet_diff"
    filtering(merged_data, filters)


def test_filters_as_list(merged_data):
    filters = ["pblh_inlet_diff", "pblh_min"]
    filtering(merged_data, filters)


def test_filters_as_dict(merged_data):
    filters = {"TAC": ["pblh_inlet_diff", "pblh_min"]}
    filtering(merged_data, filters)


def test_filters_as_dict_with_missing_site(merged_data, capsys):
    filters = {"TAC": ["pblh_inlet_diff", "pblh_min"]}
    merged_data["MHD"] = "this will be skipped!"
    filtering(merged_data, filters)

    logs = capsys.readouterr().err
    assert "Missing entry for sites ['MHD'] in filters." in logs


@pytest.mark.parametrize("name", ["unknown_filter", ""])
@pytest.mark.parametrize("request_kind", ["scalar", "list", "site_scalar", "site_list"])
@pytest.mark.parametrize("public_stage", [False, True])
def test_invalid_filter_requests_fail_before_any_data_execution(
    name, request_kind, public_stage, monkeypatch,
):
    """Validate all requested names before a preceding custom filter can compute."""
    calls = []

    @delayed
    def unused_observations():
        pytest.fail("Invalid filter request executed lazy observations")

    dataset = xr.Dataset(
        {"mf": ("time", da.from_delayed(unused_observations(), shape=(1,), dtype=float))},
        coords={"time": pd.date_range("2020-01-01", periods=1)},
    )
    monkeypatch.setattr(filter_module, "filtering_functions", filtering_functions.copy())

    @filter_module.register_filter
    def computing_filter(data, keep_missing=False):
        calls.append(1)
        return data.compute()

    request = {
        "scalar": name,
        "list": ["computing_filter", name],
        "site_scalar": {"TAC": "computing_filter", "MHD": name},
        "site_list": {"TAC": ["computing_filter"], "MHD": [name]},
    }[request_kind]
    data = {"TAC": dataset, "MHD": dataset}
    with pytest.raises(KeyError) as error:
        if public_stage:
            filter_observations(data, SiteOptions.from_inputs(sites=list(data), averaging_period=None), filters=request)
        else:
            filtering(data, request)

    assert calls == []
    assert error.value.args == (name,)


@pytest.mark.parametrize("name", ["unknown_filter", "", None])
@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("data", [{}, {".metadata": "unchanged"}])
def test_global_filters_validate_registry_without_sites(name, as_list, data):
    if name is None and not as_list:
        assert filtering(data, None) is data
        return
    request = [name] if as_list else name
    with pytest.raises(KeyError) as error:
        filtering(data, request)
    assert error.value.args == (name,)


@pytest.mark.parametrize("name", ["unknown_filter", "", None])
@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("with_active_site", [False, True])
def test_absent_site_filter_entries_remain_deferred(name, as_list, with_active_site):
    dataset = xr.Dataset({"mf": ("time", [1.0])}, coords={"time": pd.date_range("2020-01-01", periods=1)})
    data = {"TAC": dataset} if with_active_site else {".metadata": "unchanged"}
    request = {"MHD": [name] if as_list else name}
    result = filtering(data, request)
    assert result.keys() == data.keys()
    for site in data:
        assert result[site] is data[site]


@pytest.mark.parametrize("filter_request", ["noon", ["noon"], {"MHD": "noon"}])
@pytest.mark.parametrize("data", [{}, {".metadata": "unchanged"}])
def test_supported_filters_accept_inputs_without_sites(filter_request, data):
    assert filtering(data, filter_request) == data


@pytest.mark.parametrize("site_mapping", [False, True])
def test_active_none_filter_rejected_before_prior_filter(monkeypatch, site_mapping):
    monkeypatch.setattr(filter_module, "filtering_functions", filtering_functions.copy())

    @filter_module.register_filter
    def unexpected_filter(dataset, keep_missing=False):
        pytest.fail("A preceding filter executed before invalid active None was rejected")

    dataset = xr.Dataset({"mf": ("time", [1.0])}, coords={"time": pd.date_range("2020-01-01", periods=1)})
    filters = {"TAC": ["unexpected_filter"], "MHD": [None]} if site_mapping else ["unexpected_filter", None]
    with pytest.raises(KeyError) as error:
        filtering({"TAC": dataset, "MHD": dataset}, filters)
    assert error.value.args == (None,)


def test_registered_custom_filter_remains_available(monkeypatch, capsys):
    monkeypatch.setattr(filter_module, "filtering_functions", filtering_functions.copy())

    @filter_module.register_filter
    def select_last(dataset, keep_missing=False):
        """Select the final observation."""
        return dataset.isel(time=[-1])

    dataset = xr.Dataset(
        {"mf": ("time", [1.0, 2.0])},
        coords={"time": pd.date_range("2020-01-01", periods=2)},
    )
    assert filter_module.filtering_functions["select_last"] is select_last
    result = filtering({"TAC": dataset}, "select_last")
    xr.testing.assert_identical(result["TAC"], dataset.isel(time=[-1]))
    filter_module.list_filters()
    assert "select_last" in capsys.readouterr().out


def _reference_local_ratio(fp, lat, lon, release_lat, release_lon, window_size):
    """Clipped-window reference matching verification-games' audit_local_influence."""
    dlat, dlon = lat[1] - lat[0], lon[1] - lon[0]
    half = window_size // 2
    ratios = np.zeros(fp.shape[-1])
    for t in range(fp.shape[-1]):
        y = np.abs(lat - release_lat[t]).argmin()
        x = np.abs(lon - release_lon[t]).argmin()
        total = fp[:, :, t].sum()
        if abs(lat[y] - release_lat[t]) < dlat / 2 and abs(lon[x] - release_lon[t]) < dlon / 2 and total != 0:
            ratios[t] = fp[max(0, y - half) : y + half + 1, max(0, x - half) : x + half + 1, t].sum() / total
    return ratios


def _footprint_dataset(chunks=None):
    """Synthetic merged-style dataset with interior, edge, unmatched and empty footprints."""
    rng = np.random.default_rng(1050)
    lat = np.linspace(40.0, 49.0, 10)
    lon = np.linspace(-10.0, 1.0, 12)
    ntime = 8
    fp = rng.gamma(0.5, size=(lat.size, lon.size, ntime))
    release_lat = np.array([44.1, 44.1, 40.2, 48.9, 44.1, 44.1, 70.0, 44.1])
    release_lon = np.array([-5.1, -5.1, -9.8, 0.9, -5.1, -5.1, -5.1, -5.1])
    # Concentrate influence near the release point for some times so thresholds bite.
    fp[3:6, 4:7, [1, 4]] *= 40.0
    fp[4, 5, 5] *= 400.0
    fp[:, :, 7] = 0.0
    dataset = xr.Dataset(
        {
            "fp": (("lat", "lon", "time"), fp),
            "mf": ("time", np.arange(ntime, dtype=float)),
            "release_lat": ("time", release_lat),
            "release_lon": ("time", release_lon),
        },
        coords={"lat": lat, "lon": lon, "time": pd.date_range("2020-01-01", periods=ntime, freq="h")},
    )
    return dataset.chunk(chunks) if chunks is not None else dataset


@pytest.mark.parametrize("window_size", [1, 3, 5, 7])
def test_local_ratios_match_clipped_window_reference(window_size):
    dataset = _footprint_dataset()
    original = dataset.copy(deep=True)
    expected = _reference_local_ratio(
        dataset.fp.values, dataset.lat.values, dataset.lon.values,
        dataset.release_lat.values, dataset.release_lon.values, window_size,
    )
    result = filter_module.local_ratios(dataset, window_sizes=(window_size,))
    name = f"local_ratio_{window_size}x{window_size}"
    assert list(result.data_vars) == [name]
    np.testing.assert_allclose(result[name].values, expected)
    assert result[name].values[6] == 0.0  # release point outside the grid
    assert result[name].values[7] == 0.0  # empty footprint
    assert result[name].attrs["window_size"] == window_size
    xr.testing.assert_identical(dataset, original)


@pytest.mark.parametrize("window_size", [0, 4, -1, 2.0, True])
def test_local_ratio_window_must_be_positive_odd_integer(window_size):
    with pytest.raises(ValueError, match="window size"):
        filter_module.local_ratios(_footprint_dataset(), window_sizes=(window_size,))
    with pytest.raises(ValueError, match="window size"):
        filter_module.local_influence(_footprint_dataset(), window_size=window_size)


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_local_influence_threshold_must_be_a_fraction(threshold):
    with pytest.raises(ValueError, match="threshold"):
        filter_module.local_influence(_footprint_dataset(), threshold=threshold)


@pytest.mark.parametrize("window_size", [3, 5])
@pytest.mark.parametrize("threshold", [0.1, 0.15, 0.3])
@pytest.mark.parametrize("keep_missing", [False, True])
def test_local_influence_window_and_threshold_select_ratios(window_size, threshold, keep_missing):
    dataset = _footprint_dataset()
    ratio = filter_module.local_ratios(dataset, (window_size,))[f"local_ratio_{window_size}x{window_size}"].values
    keep = ratio <= threshold
    assert keep.any() and not keep.all()

    result = filtering(
        {"TAC": dataset}, "local_influence", keep_missing=keep_missing,
        filter_options={"local_influence": {"window_size": window_size, "threshold": threshold}},
    )["TAC"]

    if keep_missing:
        np.testing.assert_array_equal(result.time.values, dataset.time.values)
        np.testing.assert_array_equal(np.isfinite(result.mf.values), keep)
    else:
        xr.testing.assert_identical(result, dataset.isel(time=np.flatnonzero(keep)))


def test_local_influence_defaults_to_5x5_and_0p10():
    dataset = _footprint_dataset()
    ratio = filter_module.local_ratios(dataset)["local_ratio_5x5"].values
    result = filter_module.local_influence(dataset)
    np.testing.assert_array_equal(result.time.values, dataset.time.values[ratio <= 0.1])


@pytest.mark.parametrize("window_size", [3, 5])
@pytest.mark.parametrize("keep_missing", [False, True])
def test_cached_local_ratios_match_computed_without_fp(window_size, keep_missing):
    dataset = _footprint_dataset()
    cached = xr.merge([dataset, filter_module.local_ratios(dataset, (3, 5))]).drop_vars("fp")
    options = {"local_influence": {"window_size": window_size, "threshold": 0.15}}

    computed = filtering({"TAC": dataset}, "local_influence", keep_missing=keep_missing, filter_options=options)
    from_cache = filtering({"TAC": cached}, "local_influence", keep_missing=keep_missing, filter_options=options)

    xr.testing.assert_identical(
        from_cache["TAC"][["mf", "release_lat", "release_lon"]],
        computed["TAC"][["mf", "release_lat", "release_lon"]],
    )


def test_legacy_standard_name_local_ratio_is_used_for_5x5():
    dataset = _footprint_dataset()
    ratio = filter_module.local_ratios(dataset)["local_ratio_5x5"]
    legacy = dataset.drop_vars("fp").assign(local_ratio=ratio.assign_attrs(standard_name="local_ratio"))
    expected = filter_module.local_influence(dataset)
    result = filter_module.local_influence(legacy)
    np.testing.assert_array_equal(result.time.values, expected.time.values)

    with pytest.raises(ValueError, match="local_ratio_3x3"):
        filter_module.local_influence(legacy, window_size=3)


def test_local_influence_without_fp_or_cache_fails():
    with pytest.raises(ValueError, match="local_ratio_5x5"):
        filter_module.local_influence(_footprint_dataset().drop_vars("fp"))


def _failing_footprint(shape):
    @delayed
    def unused_footprint():
        pytest.fail("Lazy footprint was executed")

    return da.from_delayed(unused_footprint(), shape=shape, dtype=float)


def test_local_ratios_stay_lazy():
    dataset = _footprint_dataset()
    lazy = dataset.assign(fp=(dataset.fp.dims, _failing_footprint(dataset.fp.shape)))
    lazy = lazy.assign(release_lat=lazy.release_lat.chunk(), release_lon=lazy.release_lon.chunk())

    result = filter_module.local_ratios(lazy, (3, 5))

    for name in ("local_ratio_3x3", "local_ratio_5x5"):
        assert isinstance(result[name].data, da.Array)


def test_local_ratios_lazy_and_eager_agree():
    eager = filter_module.local_ratios(_footprint_dataset(), (3, 5))
    lazy = filter_module.local_ratios(_footprint_dataset(chunks={"time": 3, "lat": 4}), (3, 5))
    assert all(isinstance(lazy[name].data, da.Array) for name in lazy.data_vars)
    xr.testing.assert_allclose(lazy.compute(), eager)


def test_cached_local_influence_leaves_footprint_lazy():
    dataset = _footprint_dataset()
    ratios = filter_module.local_ratios(dataset, (5,))
    lazy = xr.merge([dataset.drop_vars("fp"), ratios]).assign(
        fp=(dataset.fp.dims, _failing_footprint(dataset.fp.shape))
    )

    result = filter_module.local_influence(lazy, threshold=0.15)

    assert isinstance(result.fp.data, da.Array)
    expected = dataset.time.values[ratios.local_ratio_5x5.values <= 0.15]
    np.testing.assert_array_equal(result.time.values, expected)


def test_filter_options_validated_before_execution():
    dataset = _footprint_dataset()
    with pytest.raises(KeyError):
        filtering({"TAC": dataset}, "local_influence", filter_options={"unknown_filter": {}})
    with pytest.raises(TypeError, match="local_influence"):
        filtering({"TAC": dataset}, "local_influence", filter_options={"local_influence": {"windw": 3}})
    with pytest.raises(TypeError, match="keep_missing"):
        filtering({"TAC": dataset}, "noon", filter_options={"noon": {"keep_missing": True}})


def test_filter_observations_forwards_filter_options():
    dataset = _footprint_dataset()
    options = {"local_influence": {"window_size": 3, "threshold": 0.3}}
    expected = filtering({"TAC": dataset}, "local_influence", filter_options=options)["TAC"]

    filtered, _ = filter_observations(
        {"TAC": dataset}, SiteOptions.from_inputs(sites=["TAC"], averaging_period=None),
        filters="local_influence", filter_options=options,
    )

    xr.testing.assert_identical(filtered["TAC"], expected)


def test_data_config_owns_and_validates_filter_options():
    params = {
        "species": "ch4", "sites": ["TAC"], "start_date": "2019-01-01", "end_date": "2019-01-02",
        "inlet": ["185m"], "domain": "EUROPE", "fp_height": ["185m"],
        "flux_sources": ["total"], "averaging_period": ["1h"],
    }
    options = {"local_influence": {"window_size": 5, "threshold": 0.15}}
    config = DataConfig.from_params({**params, "filters": ["local_influence"], "filter_options": options})
    assert config.filter_options == options
    assert config.filter_options["local_influence"] is not options["local_influence"]
    assert DataConfig.from_params(params).filter_options is None

    with pytest.raises(TypeError, match="local_influence"):
        DataConfig.from_params({**params, "filter_options": {"local_influence": {"windw": 5}}})
