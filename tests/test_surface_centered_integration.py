"""Actual provider-to-CO2 centered preparation with independent scientific oracles."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import pytensor
import xarray as xr

from examples.surface_centered_pilot import acquire_supplied, prepare_pilot, run_example, surface_inputs
from openghg_inversions.boundary_sensitivity import align_boundary_operands
from openghg_inversions.inversion_data import MergedData, SiteOptions, acquisition
from openghg_inversions.rhime.co2.centered_boundary import restore_absolute_predictives


@pytest.mark.parametrize("mixed_transport", [False, True])
def test_offline_scientific_chain_and_absolute_saved_replay(tmp_path, request, mixed_transport):
    if mixed_transport:
        request.getfixturevalue("compatible_provider")
    retained, canonical, reduction, prepared, trace = run_example(tmp_path, mixed_transport=mixed_transport)
    assert json.loads(retained.attrs["surface_transport"]) == {
        "biosphere": "time_resolved" if mixed_transport else "integrated", "fossil": "integrated",
    }
    np.testing.assert_allclose(retained.mf_mod_sectoral.sel(source="biosphere"), [-.2, 0., .6], atol=1e-12)
    np.testing.assert_allclose(retained.mf_mod_sectoral.sel(source="fossil"), [5., 4., 5.])
    np.testing.assert_allclose(retained.bc_mod, 400.05, rtol=0, atol=1e-12)
    np.testing.assert_allclose(canonical.inv_inputs.mf, [404., 402.5, 403.5])
    # Errors are derived on the selected population, with original rows retained.
    np.testing.assert_allclose(canonical.inv_inputs.mf_error, [.5, .5, np.sqrt(.05)])
    np.testing.assert_allclose(canonical.inv_inputs.H_bc_mass, .5)
    absolute_bc = canonical.inv_inputs.H_bc_anomaly + 400 * canonical.inv_inputs.H_bc_mass
    np.testing.assert_allclose(absolute_bc.transpose("bc_region", "nmeasure"), [[199.975] * 3, [200.075] * 3])
    assert canonical.inv_inputs.bc_reference.ndim == 0

    # Independent two-cell dense Gaussian oracle. A shared bucket and symmetric
    # covariance give Pi=(1/2,1/2); the omitted contrast retains nonzero variance.
    h = np.array([[2.4, 2.4], [1.6, 2.4], [.8, 4.8]])
    b = .04 * np.array([[1., np.exp(-1.)], [np.exp(-1.), 1.]])
    pi = np.array([[.5, .5]])
    c_alpha = pi @ b @ pi.T
    effective = h @ b @ pi.T @ np.linalg.inv(c_alpha)
    residual = h @ b @ h.T - effective @ c_alpha @ effective.T
    np.testing.assert_allclose(reduction.retained_mean, [1.])
    np.testing.assert_allclose(reduction.retained_covariance, c_alpha)
    np.testing.assert_allclose(reduction.effective_observation_operator, effective)
    np.testing.assert_allclose(reduction.observation_intercept, h @ [.9, 1.1] - effective[:, 0])
    np.testing.assert_allclose(reduction.unresolved_observation_covariance, residual, atol=1e-14)
    # Equal first-row cell sensitivities leave no unresolved spatial contrast.
    assert np.diag(residual)[0] == pytest.approx(0.0, abs=1e-14)
    assert (np.diag(residual)[1:] > 0).all()
    xr.testing.assert_allclose(prepared.inv_inputs.H_bc_mass, canonical.inv_inputs.H_bc_mass)

    # Arbitrary sampled BC factors preserve the original multiplicative model.
    aligned = align_boundary_operands(canonical.inv_inputs)
    aligned_absolute = aligned.H_bc_anomaly + aligned.bc_reference * aligned.H_bc_mass
    expected_bc = (trace.prior.bc * aligned_absolute).sum("bc_region")
    xr.testing.assert_allclose(trace.prior.mu_bc, expected_bc.transpose(*trace.prior.mu_bc.dims))
    np.testing.assert_allclose(
        trace.prior.co2_flux_contribution,
        trace.prior.flux_scaling.values[..., 0, None] * effective[:, 0]
        + (h @ [.9, 1.1] - effective[:, 0]),
    )
    np.testing.assert_allclose(
        trace.prior.modelled_concentration, trace.prior.co2_flux_contribution + trace.prior.mu_bc,
    )
    xr.testing.assert_allclose(trace.observed_data.y, canonical.inv_inputs.mf)
    assert trace.prior_predictive.y.attrs["concentration_representation"] == "absolute-v1"
    absolute = trace.prior_predictive.y.copy(deep=True)
    restore_absolute_predictives(trace)
    xr.testing.assert_identical(trace.prior_predictive.y, absolute)


def test_mocked_fresh_getters_match_supplied_scientific_preparation(monkeypatch):
    obs, fp, fluxes, bc = surface_inputs()
    saved = [value.copy(deep=True) for value in (obs, fp, bc, *fluxes.values())]
    supplied = acquire_supplied(obs, fp, fluxes, bc)
    requests = []

    def get_flux(**kwargs):
        requests.append(kwargs)
        return {name: SimpleNamespace(data=value, metadata={}) for name, value in fluxes.items()}

    monkeypatch.setattr(acquisition, "get_flux_data", get_flux)
    monkeypatch.setattr(acquisition, "get_obs_data", lambda **kw: SimpleNamespace(data=obs, metadata={}))
    monkeypatch.setattr(acquisition, "get_footprint_data", lambda **kw: SimpleNamespace(data=fp, metadata={}))
    monkeypatch.setattr(acquisition, "get_bc", lambda **kw: SimpleNamespace(data=bc, metadata={}))
    monkeypatch.setattr(acquisition, "merged_scenario_data", lambda *a, **kw: pytest.fail("ModelScenario used"))
    fresh = MergedData.from_options(
        species="co2", site_options=SiteOptions.from_inputs(sites=["TOY"], averaging_period="3h"),
        domain="EUROPE", start_date="2020-01-16", end_date="2020-01-17", flux_sources=list(fluxes),
        use_bc=True, surface_transport={source: "integrated" for source in fluxes},
        boundary_representation="centered", reference=400., reference_units="ppm", mass_tolerance=1e-6,
    )
    assert requests[0]["allow_older_flux"] is False
    for name in ("fp_x_flux_sectoral", "mf_mod", "bc_anomaly_n", "bc_mass_n", "bc_reference", "bc_mod"):
        xr.testing.assert_identical(fresh.site_data["TOY"][name], supplied[name])
    with pytensor.config.change_flags(floatX="float64"):
        _, _, _, supplied_prepared = prepare_pilot(supplied, fluxes)
        _, _, _, fresh_prepared = prepare_pilot(fresh.site_data["TOY"], fresh.flux_data)
    xr.testing.assert_allclose(fresh_prepared.inv_inputs, supplied_prepared.inv_inputs)
    for borrowed, before in zip((obs, fp, bc, *fluxes.values()), saved, strict=True):
        xr.testing.assert_identical(borrowed, before)


def test_required_provider_missing_capability_fails_end_to_end_gate(tmp_path):
    """A missing capability must make the real mixed-source acceptance case red."""
    import os
    from pathlib import Path
    import subprocess
    import sys

    (tmp_path / "conftest.py").write_text(
        "import pytest\n"
        "import openghg.analyse._modelled_obs as provider\n"
        "@pytest.fixture(autouse=True)\n"
        "def broken_provider(monkeypatch):\n"
        "    monkeypatch.delattr(provider, 'fp_x_flux_time_resolved')\n"
    )
    (tmp_path / "test_gate.py").write_text(
        "from tests.test_surface_centered_integration import "
        "test_offline_scientific_chain_and_absolute_saved_replay\n"
    )
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "tests.surface_provider",
         "--confcutdir", str(tmp_path),
         str(tmp_path / "test_gate.py") + "::test_offline_scientific_chain_and_absolute_saved_replay[True]"],
        cwd=repo, env={**os.environ, "PYTHONPATH": str(repo)},
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "convolution helpers" in result.stdout
    assert "1 failed" in result.stdout
    assert "skipped" not in result.stdout
