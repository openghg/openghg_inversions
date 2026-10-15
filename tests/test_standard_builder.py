"""The standard builder consumes eager values from an explicit shared boundary."""

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import dask.array as da
from dask import delayed, is_dask_collection
import numpy as np
import pytest
import xarray as xr

from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.models.priors import prior_records
from openghg_inversions.rhime.specs import FixedErrorSettings, RhimeModelSpec, SectorSpec
from openghg_inversions.rhime.standard import build_standard_model, materialize_standard_inputs


def _inputs() -> xr.Dataset:
    return xr.Dataset(
        {
            "H": (("region", "nmeasure"), [[1., 2.]]),
            "mf": ("nmeasure", [1., 2.]),
            "mf_error": ("nmeasure", [.2, .2]),
        },
        coords={"region": ["one"], "nmeasure": [0, 1]},
    )


def _model(prior=None) -> RhimeModelSpec:
    return RhimeModelSpec(
        species="ch4", domain="EUROPE", use_bc=False, likelihood=FixedErrorSettings(),
        sectors=(SectorSpec("total", "inventory", prior or {"pdf": "normal", "mu": 1., "sigma": .2}, "total"),),
    )


def test_materialize_once_then_build_without_replacing_requested_settings():
    calls = []

    @delayed
    def source():
        calls.append("source")
        return np.array([[1., 2.]])

    @delayed
    def unused():
        pytest.fail("disabled prior must stay lazy")

    inputs = _inputs()
    inputs["H"] = inputs["H"].copy(data=da.from_delayed(source(), shape=(1, 2), dtype=float))
    mean = inputs["H"].mean("nmeasure")
    model = replace(
        _model({"pdf": "normal", "mu": mean, "sigma": .2}),
        offset_prior={"pdf": "normal", "mu": da.from_delayed(unused(), shape=(), dtype=float), "sigma": 1.},
    )
    prepared = cast(RhimePreparedInputs, SimpleNamespace(inv_inputs=inputs))
    eager, priors = materialize_standard_inputs(prepared, model)
    assert calls == ["source"]
    assert is_dask_collection(inputs["H"])
    assert model.sectors[0].x_prior["mu"] is mean
    assert not is_dask_collection(priors["x"]["mu"])
    assert "offset_latent" not in priors

    built = build_standard_model(eager, model, priors=priors)
    assert calls == ["source"]
    assert built.output_contract is built.output_contract
    assert built.output_contract.variable_roles["flux_scale"] == "x"
    assert prior_records(built.model)["x"]["mu"] == {"data": "x_prior_mu"}
    assert model.likelihood is not None and isinstance(model.likelihood, FixedErrorSettings)


@pytest.mark.parametrize("lazy_owner", ["input", "prior"])
def test_builder_rejects_lazy_values_without_computing(lazy_owner):
    @delayed
    def unexpected():
        pytest.fail("eager builder executed caller-owned lazy values")

    inputs = _inputs()
    priors = {"x": {"pdf": "normal", "mu": 1., "sigma": .2}}
    if lazy_owner == "input":
        inputs["H"] = inputs["H"].copy(data=da.from_delayed(unexpected(), shape=(1, 2), dtype=float))
    else:
        priors["x"]["mu"] = da.from_delayed(unexpected(), shape=(), dtype=float)
    with pytest.raises(ValueError, match="materialized before building"):
        build_standard_model(inputs, _model(), priors=priors)


def test_builder_rejects_lazy_scalar_nested_in_positional_prior_without_computing():
    @delayed
    def unexpected():
        pytest.fail("eager builder executed a nested scalar prior value")

    prior = {"pdf": "normal", "mu": [da.from_delayed(unexpected(), shape=(), dtype=float)], "sigma": .2}
    model = _model(prior)
    with pytest.raises(ValueError, match="materialized before building"):
        build_standard_model(_inputs(), model, priors={"x": prior})


def test_builder_requires_materialized_parameters_for_configured_likelihood_prior():
    from openghg_inversions.rhime.specs import PollutionEventSettings

    @delayed
    def unexpected():
        pytest.fail("builder fell back to a lazy requested prior")

    sigma_prior = {"pdf": "halfnormal", "sigma": da.from_delayed(unexpected(), shape=(), dtype=float)}
    model = replace(_model(), likelihood=PollutionEventSettings(sigma_prior=sigma_prior))
    with pytest.raises(ValueError, match="Missing materialized prior.*sigma"):
        build_standard_model(_inputs(), model, priors={"x": model.sectors[0].x_prior})
