"""Resolved requests keep external shorthand separate from retained run facts."""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import dask.array as da
from dask.callbacks import Callback
import pytest
import xarray as xr

import openghg_inversions.rhime as rhime
from openghg_inversions.inference.sampling import RhimeSampler
from openghg_inversions.inversion_data import RhimeMergedData
from openghg_inversions.rhime import params as rhime_params
from openghg_inversions.rhime import specs as rhime_specs


def _request(**overrides):
    return {
        "species": "ch4",
        "domain": "EUROPE",
        "sites": ["tac", "MHD"],
        "averaging_period": "1h",
        "start_date": "2019-01-01",
        "end_date": "2019-02-01",
        "output_name": "configuration",
        "output_format": "none",
        "flux_sources": ["inventory"],
        "mismatch_model": "pollution_event",
        **overrides,
    }


@pytest.mark.parametrize("multisector", [False, True])
def test_scalar_and_expanded_requests_resolve_equally(multisector):
    """All site selectors share the same scalar/sequence convention."""
    selectors = {
        "averaging_period": "1h",
        "inlet": "100m",
        "fp_height": "110m",
        "instrument": "instrument",
        "platform": "surface",
        "obs_data_level": "level",
        "met_model": "met",
        "max_level": 17,
        "time_resolved": None,
    }
    sources = ["inventory", "ocean"] if multisector else ["inventory"]
    scalar = rhime_params.resolve_rhime_config(
        _request(flux_sources=sources, **selectors), multisector=multisector
    )
    expanded = rhime_params.resolve_rhime_config(
        _request(flux_sources=sources, **{name: [value, value] for name, value in selectors.items()}),
        multisector=multisector,
    )

    assert scalar == expanded
    assert scalar.preparation.site_options.sites == ("TAC", "MHD")
    assert scalar.preparation.site_options.time_resolved == (None, None)
    assert isinstance(scalar.sampler, RhimeSampler)
    assert scalar.sampler.draws == 1000
    assert scalar.sampler.sample_posterior_predictive == ("y",)
    assert not hasattr(scalar, "run_spec")


@pytest.mark.parametrize(
    ("overrides", "option"),
    [
        ({"averaging_period": ["1h"]}, "averaging_period"),
        ({"instrument": ["one"]}, "instrument"),
        ({"max_level": [True, None]}, "max_level"),
        ({"sites": ["tac", "TAC"]}, "unique"),
        ({"unknown_option": True}, "unknown_option"),
    ],
)
def test_invalid_effective_requests_fail_without_acquisition(monkeypatch, overrides, option):
    def fail(*args, **kwargs):
        pytest.fail("option resolution must not acquire or sample")

    monkeypatch.setattr(rhime, "retrieve_or_reload_rhime_data", fail)
    monkeypatch.setattr(RhimeSampler, "sample", fail)
    with pytest.raises(ValueError, match=option):
        rhime_params.resolve_rhime_config(_request(**overrides), multisector=False)


@pytest.mark.parametrize("replace_periods", [False, True])
def test_file_overrides_precede_site_expansion(tmp_path: Path, replace_periods):
    """A replaced list is not validated against either the old or new request."""
    configured = _request(averaging_period=["invalid-length"] if replace_periods else "1h")
    path = tmp_path / "configuration.ini"
    path.write_text(
        "[RHIME.OPTIONS]\n" + "\n".join(f"{name} = {value!r}" for name, value in configured.items()),
        encoding="utf-8",
    )
    raw = rhime_params.load_rhime_config(path)
    assert raw["sites"] == ["tac", "MHD"]
    assert raw["averaging_period"] == configured["averaging_period"]
    overrides = {"sites": ["TAC", "MHD", "BSD"]}
    if replace_periods:
        overrides["averaging_period"] = "1h"

    config = rhime_params.resolve_rhime_config({**raw, **overrides}, multisector=False)
    direct = rhime_params.resolve_rhime_config({**configured, **overrides}, multisector=False)

    assert config == direct
    assert config.preparation.site_options.averaging_period == ("1h", "1h", "1h")
    assert raw["sites"] == ["tac", "MHD"]
    assert raw["averaging_period"] == configured["averaging_period"]


@pytest.mark.parametrize("invalid", [False, True])
def test_resolution_preserves_caller_containers_and_lazy_keywords(monkeypatch, invalid):
    """Configuration ownership never materializes opaque sampler payloads."""
    original = _request(
        xprior={"pdf": "normal", "mu": 1.0, "sigma": 0.5},
        inlet=[slice("10m", "100m"), None],
        sample_kwargs={"idata_kwargs": {"log_likelihood": False}, "random_seed": [2, 3]},
        posterior_predictive_kwargs={"compile_kwargs": {"mode": "FAST_COMPILE"}},
    )
    if invalid:
        original["instrument"] = ["wrong-length"]
    before = deepcopy(original)
    lazy = da.arange(3, chunks=2)
    original["sample_kwargs"]["opaque_value"] = lazy

    def fail(*args, **kwargs):
        pytest.fail("resolution must not execute sampling")

    monkeypatch.setattr(RhimeSampler, "sample", fail)
    with Callback(pretask=fail), pytest.warns(UserWarning, match="xprior"):
        if invalid:
            with pytest.raises(ValueError, match="instrument"):
                rhime_params.resolve_rhime_config(original, multisector=False)
        else:
            config = rhime_params.resolve_rhime_config(original, multisector=False)
            assert config.sampler.sample_kwargs["opaque_value"] is lazy
            assert config.preparation.site_options.inlet == (slice("10m", "100m"), None)
            assert config.model.sectors[0].x_prior == before["xprior"]

    assert original["sample_kwargs"]["opaque_value"] is lazy
    without_opaque = {**original, "sample_kwargs": dict(original["sample_kwargs"])}
    del without_opaque["sample_kwargs"]["opaque_value"]
    assert without_opaque == before


def test_retained_run_uses_prepared_metadata_without_changing_request():
    config = rhime_params.resolve_rhime_config(
        _request(averaging_period=["1h", "2h"]), multisector=False
    )
    prepared = SimpleNamespace(sites=("MHD",), averaging_period=("2h",))

    run = config.retained_run_spec(prepared)

    assert config.preparation.site_options.sites == ("TAC", "MHD")
    assert config.preparation.site_options.averaging_period == ("1h", "2h")
    assert run.sites == ("MHD",)
    assert run.averaging_period == ("2h",)
    assert (run.start_date, run.end_date) == ("2019-01-01", "2019-02-01")
    assert run.model is config.model
    assert run.output is config.output


def test_legacy_setup_remains_a_projection_of_resolved_choices():
    request = _request(
        use_bc=False,
        mismatch_model=None,
        draws="7",
        time_resolved=True,
        sample_kwargs={"random_seed": [2, 3]},
    )
    config = rhime_params.resolve_rhime_config(request, multisector=False)
    legacy = rhime_params.resolve_rhime_options(params=request, multisector=False)

    assert legacy.run_spec.model == config.model
    assert legacy.run_spec.output == config.output
    assert legacy.sampler == config.sampler
    assert tuple(site.upper() for site in legacy.run_spec.sites) == config.preparation.site_options.sites
    assert legacy.data_args["use_bc"] is config.preparation.use_bc is False
    assert legacy.data_args["averaging_period"] == "1h"
    assert legacy.data_args["time_resolved"] is True
    assert legacy.data_args["inlet"] is None
    assert legacy.data_args["instrument"] is None
    assert "use_tracer" not in legacy.data_args


def test_legacy_setup_preserves_sparse_prior_metadata():
    """Established setup metadata need not adopt canonical explicit defaults."""
    request = _request(add_offset=True)
    config = rhime_params.resolve_rhime_config(request, multisector=False)
    legacy = rhime_params.resolve_rhime_options(params=request, multisector=False)

    assert config.model.bc_prior == rhime_specs.DEFAULT_BC_PRIOR
    assert config.model.offset_prior == rhime_specs.DEFAULT_OFFSET_PRIOR
    assert config.model.likelihood.sigma_prior == rhime_specs.DEFAULT_POLLUTION_EVENT_SIGMA_PRIOR
    assert legacy.run_spec.model.bc_prior is None
    assert legacy.run_spec.model.offset_prior is None
    assert legacy.run_spec.model.likelihood.sigma_prior is None


def test_minimum_error_none_resolves_to_numeric_default():
    unspecified = rhime_params.resolve_rhime_config(_request(min_error=None), multisector=False)
    explicit = rhime_params.resolve_rhime_config(_request(min_error=0.0), multisector=False)

    assert unspecified == explicit
    assert unspecified.preparation.min_error == 0.0


def test_filtering_preserves_typed_filter_request():
    """The legacy filtering function can normalize a copy of owned choices."""
    request = _request(filters={"TAC": "six_hr_mean", "MHD": None})
    config = rhime_params.resolve_rhime_config(request, multisector=False)
    before = deepcopy(config.preparation.filters)
    data = xr.Dataset(
        {"mf": ("time", [1.0, 3.0])},
        coords={"time": [datetime(2019, 1, 1, 12), datetime(2019, 1, 1, 13)]},
    )
    merged = RhimeMergedData(
        fp_all={"TAC": data, "MHD": data}, site_options=config.preparation.site_options
    )

    filtered = rhime.filter_rhime_observations(merged, config.preparation)

    assert filtered.fp_all["TAC"].sizes["time"] == 1
    assert filtered.fp_all["TAC"].mf.item() == 2.0
    assert filtered.fp_all["MHD"].sizes["time"] == 2
    assert config.preparation.filters == before
    assert request["filters"] == before
    assert merged.fp_all["TAC"].sizes["time"] == 2


def test_configuration_api_is_public():
    for name in ("RhimeConfig", "RhimePreparationConfig", "load_rhime_config", "resolve_rhime_config"):
        assert getattr(rhime, name) is getattr(rhime_params, name)


def test_multisector_preparation_selects_sources_from_typed_request():
    """Tuple configuration labels remain a list selection at the xarray boundary."""
    config = rhime_params.resolve_rhime_config(
        _request(sites=["TAC"], flux_sources=["second", "first"], use_bc=False),
        multisector=True,
    )
    sensitivity = xr.DataArray(
        [[[1.0]], [[2.0]]],
        dims=("source", "region", "time"),
        coords={"source": ["first", "second"], "region": [0], "time": [datetime(2019, 1, 1)]},
    )
    merged = RhimeMergedData(
        {"TAC": sensitivity.sum("region").rename("fp_x_flux_sectoral").to_dataset()},
        config.preparation.site_options,
    )
    basis = SimpleNamespace(sensitivity=lambda values: sensitivity)

    site_data = rhime.build_rhime_sensitivities(
        merged, basis, config.preparation, multisector=True,
    )

    # Assignment preserves the merged dataset's labelled source axis.
    assert site_data["TAC"].H.source.values.tolist() == ["first", "second"]
    assert site_data["TAC"].H.sel(source="second").item() == 2.0
    assert config.preparation.flux_sources == ("second", "first")
