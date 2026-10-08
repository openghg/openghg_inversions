"""Unsupported generic tracer requests fail at public boundaries before I/O."""

from types import SimpleNamespace

import pytest

from openghg_inversions.inversion_data import prepare_rhime_inputs
from openghg_inversions.rhime import run_rhime, run_rhime_multisector
from openghg_inversions.rhime.params import normalise_rhime_params, params_from_config, resolve_rhime_config
from openghg_inversions.inversion_data import SiteOptions, load_rhime_data


@pytest.mark.parametrize("multisector", [False, True])
def test_option_resolution_rejects_tracer_before_required_options(multisector):
    with pytest.raises(ValueError, match="use_tracer=True.*not supported"):
        resolve_rhime_config(params={"use_tracer": True}, multisector=multisector)


@pytest.mark.parametrize("tracer_options", [{}, {"use_tracer": False}])
def test_false_or_omitted_tracer_is_accepted_during_option_normalization(tracer_options):
    assert normalise_rhime_params(tracer_options) == {}


@pytest.mark.parametrize(
    ("configured", "override", "effective"),
    [(True, None, True), (False, None, False), (True, False, False), (False, True, True)],
)
def test_ini_tracer_option_respects_overrides_before_normalization(tmp_path, configured, override, effective):
    config_file = tmp_path / "rhime.ini"
    config_file.write_text(f"[RHIME.OPTIONS]\nuse_tracer = {configured}\n", encoding="utf-8")
    assert params_from_config(config_file, normalise=False)["use_tracer"] is configured
    extra_kwargs = {} if override is None else {"use_tracer": override}
    if effective:
        with pytest.raises(ValueError, match="use_tracer=True.*not supported"):
            params_from_config(config_file, extra_kwargs=extra_kwargs)
    else:
        assert "use_tracer" not in params_from_config(config_file, extra_kwargs=extra_kwargs)


@pytest.mark.parametrize("tracer_options", [{}, {"use_tracer": False}])
@pytest.mark.parametrize("multisector", [False, True])
def test_false_or_omitted_tracer_is_not_forwarded_after_resolution(tracer_options, multisector):
    params = {
        "species": "ch4",
        "sites": ["TAC"],
        "domain": "EUROPE",
        "averaging_period": "1h",
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "output_name": "test",
        "output_format": "none",
        "flux_sources": ["anthro", "natural"] if multisector else ["total"],
        **tracer_options,
    }
    original = params.copy()
    config = resolve_rhime_config(params=params, multisector=multisector)
    assert not hasattr(config, "use_tracer")
    assert params == original


@pytest.mark.parametrize("runner", [run_rhime, run_rhime_multisector])
@pytest.mark.parametrize("supplied_merged_data", [False, True])
def test_rhime_runner_rejects_tracer_before_acquisition(runner, supplied_merged_data):
    merged = SimpleNamespace(fp_all={}) if supplied_merged_data else None
    with pytest.raises(ValueError, match="use_tracer=True.*not supported"):
        runner(use_tracer=True, merged_data=merged)


@pytest.mark.parametrize("multisector", [False, True])
def test_canonical_load_checks_supplied_layout(multisector):
    requested = SiteOptions.from_inputs(
        sites=["TAC"], averaging_period="1h", inlet=None, fp_height=None,
        instrument=None, platform=None, obs_data_level=None, met_model=None,
        max_level=None,
    )
    kwargs = dict(
        site_options=requested, species="ch4", domain="EUROPE",
        start_date="2019-01-01", end_date="2019-01-02", output_name="test",
        flux_sources=["total"], split_by_sectors=multisector,
    )
    merged = SimpleNamespace(fp_all={".split_by_sectors": multisector})
    assert load_rhime_data(**kwargs, merged_data=merged) is merged
    merged.fp_all[".split_by_sectors"] = not multisector
    with pytest.raises(ValueError, match="incompatible.*layout"):
        load_rhime_data(**kwargs, merged_data=merged)


def test_direct_preparation_rejects_tracer_before_other_options():
    with pytest.raises(ValueError, match="use_tracer=True.*not supported"):
        prepare_rhime_inputs(
            species="ch4",
            sites=[],
            domain="EUROPE",
            averaging_period=None,
            start_date="2019-01-01",
            end_date="2019-01-02",
            output_name="test",
            flux_sources=[],
            use_tracer=True,
            min_error_options={"invalid": True},
        )
