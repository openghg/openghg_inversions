"""Known retrieval facts bind nested preparations without inventing missing facts."""

from dataclasses import replace

import pytest
import xarray as xr

from openghg_inversions.inversion_data import AcquisitionFacts, RhimeMergedData, SiteOptions
from openghg_inversions.rhime import nested as nested_module
from openghg_inversions.rhime.nested import combine_nested_rhime_inputs, run_nested
from openghg_inversions.rhime.params import RhimeConfig
from tests.test_nested_core import _nested
from tests.test_staged_workflow import _params


def _config(**overrides):
    return RhimeConfig.from_params(
        _params(output_format="none", flux_sources=["inventory"], **overrides), multisector=False,
    )


def _known_facts(**overrides):
    return replace(AcquisitionFacts(
        species="ch4", domain="EUROPE", start_date="2019-01-01", end_date="2019-02-01",
    ), **overrides)


def test_combine_and_revalidation_retain_original_acquisition_facts():
    native = _nested()
    facts = _known_facts()
    prepared = combine_nested_rhime_inputs(
        native.outer, native.inner, outer_overlap_masked=True, outer_acquisition=facts,
    )
    assert prepared.outer_acquisition is facts
    validated = prepared.validated()
    assert validated.outer_acquisition is facts
    assert facts.end_date == "2019-02-01"  # The observations end much earlier.


@pytest.mark.parametrize("field,value", [
    ("species", "co2"), ("domain", "ASIA"),
    ("start_date", "2018-12-31"), ("end_date", "2019-02-02"),
])
def test_supplied_nested_known_request_conflicts_fail_before_exports(monkeypatch, field, value):
    prepared = replace(_nested(), outer_acquisition=_known_facts())

    def forbidden(*args, **kwargs):
        pytest.fail("Conflicting request reached export or materialization")

    monkeypatch.setattr(nested_module, "export_basis", forbidden)
    monkeypatch.setattr(nested_module, "materialize_nested_inputs", forbidden)
    with pytest.raises(ValueError, match=f"incompatible {field}"):
        run_nested(config=_config(**{field: value}), prepared=prepared)


class ReachedMaterialization(Exception):
    pass


def _stop_at_materialization(monkeypatch):
    def stop(*args, **kwargs):
        raise ReachedMaterialization
    monkeypatch.setattr(nested_module, "materialize_nested_inputs", stop)


def test_supplied_nested_accepts_case_and_equivalent_utc_dates(monkeypatch):
    facts = _known_facts(
        species="CH4", domain="europe", start_date="2019-01-01T01:00:00+01:00",
        end_date="2019-02-01T00:00:00Z",
    )
    prepared = replace(_nested(), outer_acquisition=facts)
    _stop_at_materialization(monkeypatch)
    with pytest.raises(ReachedMaterialization):
        run_nested(config=_config(), prepared=prepared)
    assert prepared.outer_acquisition is facts


@pytest.mark.parametrize("missing", [None, "unknown"])
def test_unknown_nested_facts_do_not_infer_request_dates_from_observation_extrema(monkeypatch, missing):
    facts = AcquisitionFacts(species=missing, domain=missing, start_date=missing, end_date=missing)
    prepared = replace(_nested(), outer_acquisition=facts)
    _stop_at_materialization(monkeypatch)
    # Reusing a handoff with unknown retrieval metadata cannot establish or
    # reject a requested window from the surviving observation timestamps.
    with pytest.raises(ReachedMaterialization):
        run_nested(config=_config(start_date="2018-01-01", end_date="2020-01-01"), prepared=prepared)
    assert prepared.outer_acquisition is facts
    assert facts.start_date == facts.end_date == missing


@pytest.mark.parametrize("field,value", [
    ("species", "co2"), ("domain", "OTHER-6km"),
    ("start_date", "2019-01-02"), ("end_date", "2019-03-01"),
])
def test_independent_inner_acquisition_conflict_fails_before_scientific_preparation(monkeypatch, field, value):
    options = SiteOptions.from_inputs(sites=["TAC"], averaging_period=["1h"])
    outer = RhimeMergedData(
        site_data={"TAC": xr.Dataset()}, flux_data={}, site_options=options,
        acquisition=_known_facts(),
    )
    inner = replace(outer, acquisition=_known_facts(domain="EUROPE-6km"))
    inner.acquisition = replace(inner.acquisition, **{field: value})

    def forbidden(*args, **kwargs):
        pytest.fail("Conflicting independent retrieval reached observation-error preparation")

    monkeypatch.setattr(nested_module, "prepare_observation_errors", forbidden)
    data = _config().data
    with pytest.raises(ValueError, match=f"incompatible {field}"):
        nested_module.prepare_nested_rhime_inputs(
            outer, inner,
            **data.select(
                "domain", "start_date", "flux_sources", "averaging_error", "filters",
                "basis_algorithm", "nbasis", "fix_basis_outer_regions", "use_bc", "bc_freq",
                "min_error", "min_error_options",
            ),
            inner_domain="EUROPE-6km", inner_basis_algorithm=None, inner_nbasis=None,
            outer_basis_inputs={}, inner_basis_inputs={},
        )


@pytest.mark.parametrize("source", ["inventory", "other-inventory"])
@pytest.mark.parametrize("labelled_domain", ["outer", "inner", "both"])
def test_supplied_nested_checks_known_scalar_source_before_export(monkeypatch, source, labelled_domain):
    native = _nested()
    outer, inner = native.outer, native.inner
    if labelled_domain in {"outer", "both"}:
        outer = replace(outer, inv_inputs=outer.inv_inputs.assign_coords(source=source))
    if labelled_domain in {"inner", "both"}:
        inner = replace(inner, inv_inputs=inner.inv_inputs.assign_coords(source=source))
    prepared = combine_nested_rhime_inputs(outer, inner, outer_overlap_masked=True)
    _stop_at_materialization(monkeypatch)
    if source == "inventory":
        with pytest.raises(ReachedMaterialization):
            run_nested(config=_config(), prepared=prepared)
    else:
        monkeypatch.setattr(nested_module, "export_basis", lambda *args: pytest.fail("exported conflicting source"))
        with pytest.raises(ValueError, match="incompatible emissions source"):
            run_nested(config=_config(), prepared=prepared)
