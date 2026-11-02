"""Exercise the probe's reporting boundary with supplied draws, without sampling."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from prototypes import run_biosphere_probe as runner


def _input():
    h = np.array([[-3., 2., .5], [-1., .8, .2], [-2., 1., .4]])
    truth = np.array([[1., 1., 1.], [1., 1., 1.1]])
    data = xr.Dataset(
        {
            "H": (("nmeasure", "flux_state"), h),
            "observations": (("scenario", "nmeasure"), truth @ h.T + .2),
            "observation_error": ("nmeasure", np.ones(3)),
            "fixed_prior_contribution": ("nmeasure", np.zeros(3)),
            "prior_mean": ("flux_state", np.ones(3)),
            "prior_covariance": (("flux_state", "flux_state_cov"), np.eye(3) * .04),
            "reference_flux": ("flux_state", [-100., 80., 20.]),
            "truth_scaling": (("scenario", "flux_state"), truth),
        },
        coords={"nmeasure": range(3), "flux_state": ["GPP", "TER", "FF"],
                "flux_state_cov": ["GPP", "TER", "FF"], "scenario": ["base", "ff10"]},
    )
    for name in ("H", "observations", "observation_error", "fixed_prior_contribution"):
        data[name].attrs["units"] = "ppm"
    data.reference_flux.attrs = {
        "units": "mol s-1", "scoring_domain": "UK land", "time_support": "August 2025 mean",
    }
    return data


def _arguments(monkeypatch, tmp_path, data):
    path = tmp_path / "input.nc"
    data.to_netcdf(path)
    monkeypatch.setattr(sys, "argv", ["probe", "--input", str(path), "--output", str(tmp_path / "out"),
                                    "--index", "4"])


@pytest.mark.parametrize("name", ["H", "observations", "observation_error", "fixed_prior_contribution"])
@pytest.mark.parametrize("units", ["ppb", None])
def test_rejects_non_ppm_or_missing_units_before_build(monkeypatch, tmp_path, name, units):
    data = _input()
    if units is None:
        del data[name].attrs["units"]
    else:
        data[name].attrs["units"] = units
        data[name] *= 1000
    _arguments(monkeypatch, tmp_path, data)
    monkeypatch.setattr(runner, "build_biosphere_model", lambda *a, **kw: pytest.fail("built before unit check"))
    with pytest.raises(ValueError, match=f"{name}.*ppm"):
        runner.main()
    assert not (tmp_path / "out").exists()


def test_exports_input_flux_support_and_correct_ppm_scores(monkeypatch, tmp_path):
    data = _input()
    _arguments(monkeypatch, tmp_path, data)
    model = SimpleNamespace(compile_logp=lambda: lambda point: 0., initial_point=lambda: {})
    monkeypatch.setattr(runner, "build_biosphere_model", lambda *a, **kw: model)
    draws = np.random.default_rng(17).normal(0, .01, size=(4, 32, 3))
    draws -= draws.mean(axis=(0, 1))
    estimate = np.array([1.1, .9, 1.2])
    draws += estimate
    trace = xr.DataTree.from_dict({
        "posterior": xr.Dataset(
            {"flux_scaling": (("chain", "draw", "flux_state"), draws)},
            coords={"chain": range(4), "draw": range(32), "flux_state": data.flux_state.values},
        ),
        "sample_stats": xr.Dataset({
            "diverging": (("chain", "draw"), np.zeros((4, 32), dtype=bool)),
            "tree_depth": (("chain", "draw"), np.full((4, 32), 2)),
            "n_steps": (("chain", "draw"), np.full((4, 32), 3)),
        }),
    })
    monkeypatch.setattr(runner, "RhimeSampler", lambda **kw: SimpleNamespace(sample=lambda m: trace))
    runner.main()
    out = tmp_path / "out/base/all_hours/lognormal"
    with xr.open_datatree(out / "posterior.nc") as saved:
        xr.testing.assert_equal(saved["posterior"].to_dataset(), trace["posterior"].to_dataset())
    summary = json.loads((out / "summary.json").read_text())
    expected = data.H.values @ estimate
    observed = data.observations.sel(scenario="base").values
    np.testing.assert_allclose(summary["concentration_rmse_ppm"], np.sqrt(np.mean((expected - observed)**2)))
    np.testing.assert_allclose(summary["generative_mean_rmse_ppm"], np.sqrt(np.mean((expected - data.H.values.sum(axis=1))**2)))
    assert summary["functional_units"] == "mol s-1"
    with xr.open_dataset(out / "functionals.nc") as functionals:
        for name, value in data.reference_flux.attrs.items():
            assert functionals.attrs[name] == value
        assert "July" not in functionals.attrs["interpretation"]
        assert "Full-domain" not in functionals.attrs["interpretation"]


def test_rejects_consistently_scaled_ppb_input(monkeypatch, tmp_path):
    data = _input()
    for name in ("H", "observations", "observation_error", "fixed_prior_contribution"):
        data[name] *= 1000
        data[name].attrs["units"] = "ppb"
    _arguments(monkeypatch, tmp_path, data)
    monkeypatch.setattr(runner, "build_biosphere_model", lambda *a, **kw: pytest.fail("built before unit check"))
    with pytest.raises(ValueError, match="H.*ppm"):
        runner.main()
