"""Check CLI scoring and metadata with supplied draws, without posterior sampling."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from prototypes import run_process_biosphere_probe as runner


def _input(nmeasure):
    h = np.array([[-3., 2., 1.], [-1., .5, .2], [.1, .6, .3],
                  [-2., 1., .8], [-.2, .7, .1]])[:nmeasure]
    truth = np.array([1.2, .8, 1.1])
    fixed = np.linspace(.1, .3, nmeasure)
    data = xr.Dataset(
        {
            "H": (("nmeasure", "flux_state"), h),
            "observations": (("scenario", "nmeasure"), [h @ truth + fixed + .2]),
            "truth_scaling": (("scenario", "flux_state"), [truth]),
            "fixed_prior_contribution": ("nmeasure", fixed),
            "reference_flux": ("flux_state", [-100., 60., 20.]),
            "daytime": ("nmeasure", np.isin(np.arange(nmeasure), [0, 2])),
        },
        coords={"nmeasure": range(nmeasure), "flux_state": ["g0", "a0", "h0"],
                "source": ("flux_state", ["GPP", "Ra", "Rh"]),
                "group": ("flux_state", ["north"] * 3), "scenario": ["perturbed"]},
        attrs={"observation_sd_ppm": 1., "truth_definition": "Declared controlled process scalings"},
    )
    data.reference_flux.attrs["units"] = "mol s-1"
    return data


def _arguments(monkeypatch, tmp_path, data, support="all_hours"):
    path = tmp_path / "input.nc"
    data.to_netcdf(path)
    monkeypatch.setattr(sys, "argv", ["probe", "--input", str(path), "--output", str(tmp_path / "out"),
                                    "--arm", "independent", "--scenario", "perturbed", "--support", support])


@pytest.mark.parametrize("nmeasure,support", [(3, "all_hours"), (5, "daytime")])
@pytest.mark.parametrize("transposed", [False, True])
def test_prediction_scores_accept_either_h_dimension_order(monkeypatch, tmp_path, nmeasure, support, transposed):
    data = _input(nmeasure)
    canonical_h = data.H.values.copy()
    if transposed:
        data["H"] = data.H.transpose("flux_state", "nmeasure")
    _arguments(monkeypatch, tmp_path, data, support)
    model = SimpleNamespace(compile_logp=lambda: lambda point: 0., initial_point=lambda: {})
    monkeypatch.setattr(runner, "build_process_biosphere_model", lambda *a, **kw: (model, xr.Dataset()))
    estimate = np.array([1.1, .9, 1.0])
    draws = np.random.default_rng(17).normal(0, .01, size=(4, 32, 3))
    draws -= draws.mean(axis=(0, 1))
    draws += estimate
    trace = xr.DataTree.from_dict({
        "posterior": xr.Dataset(
            {"flux_scaling": (("chain", "draw", "flux_state"), draws)},
            coords={"chain": range(4), "draw": range(32), "flux_state": data.flux_state.values},
        ),
        "sample_stats": xr.Dataset({
            "diverging": (("chain", "draw"), np.zeros((4, 32), dtype=bool)),
            "tree_depth": (("chain", "draw"), np.full((4, 32), 2)),
        }),
    })
    monkeypatch.setattr(runner, "RhimeSampler", lambda **kw: SimpleNamespace(sample=lambda m: trace))
    runner.main()
    out = tmp_path / "out" / "perturbed" / support / "independent"
    with xr.open_datatree(out / "posterior.nc") as saved:
        xr.testing.assert_equal(saved["posterior"].to_dataset(), trace["posterior"].to_dataset())
    summary = json.loads((out / "summary.json").read_text())
    truth = data.truth_scaling.sel(scenario="perturbed").values
    rows = np.arange(nmeasure) if support == "all_hours" else np.flatnonzero(data.daytime)
    for label, indices in (("fitted", rows), ("all_hours", np.arange(nmeasure))):
        h = canonical_h[indices]
        fixed = data.fixed_prior_contribution.values[indices]
        expected, fitted, prior = h @ truth + fixed, h @ estimate + fixed, h @ np.ones(3) + fixed
        observed = data.observations.values[0, indices]
        scores = summary["prediction_scores"][label]
        assert scores["n"] == len(indices)
        for name, left, right in (
            ("posterior_vs_generative", fitted, expected), ("prior_vs_generative", prior, expected),
            ("posterior_vs_observed", fitted, observed), ("prior_vs_observed", prior, observed),
        ):
            np.testing.assert_allclose(scores[name + "_rmse_ppm"], np.sqrt(np.mean((left - right)**2)))
    assert summary["concentration_rmse_ppm"] == pytest.approx(summary["prediction_scores"]["fitted"]["posterior_vs_observed_rmse_ppm"])
    assert summary["generative_mean_rmse_ppm"] == pytest.approx(summary["prediction_scores"]["fitted"]["posterior_vs_generative_rmse_ppm"])


@pytest.mark.parametrize("attribute", ["observation_sd_ppm", "truth_definition", "units"])
def test_missing_metadata_fails_before_model_build(monkeypatch, tmp_path, attribute):
    data = _input(3)
    attrs = data.reference_flux.attrs if attribute == "units" else data.attrs
    del attrs[attribute]
    _arguments(monkeypatch, tmp_path, data)
    monkeypatch.setattr(runner, "build_process_biosphere_model", lambda *a, **kw: pytest.fail("built before metadata check"))
    with pytest.raises(ValueError, match=attribute):
        runner.main()
    assert not (tmp_path / "out").exists()
