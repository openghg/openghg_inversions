"""Raw preparations retain independent fitted operands across saved models."""

import json

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.basis import BasisFunctions
from openghg_inversions.inference.sampling import RhimeSampler
from openghg_inversions.inversion_data import PreparedInputs
from openghg_inversions.model_error import MinimumError
from openghg_inversions.boundary_sensitivity import BoundaryAlignment
from openghg_inversions.rhime import _standard_stages, multisector_stages
from openghg_inversions.rhime.params import DataConfig
from tests.test_model_owned_alignment import _raw
from tests.test_staged_workflow import _prepared


@pytest.mark.parametrize("multisector", [False, True])
def test_one_preparation_two_periods_save_exact_operands_and_replay_without_recalculation(
    tmp_path, monkeypatch, multisector,
):
    raw = _raw()
    basis = _prepared().basis_functions
    sources = ["inventory", "supplemental"] if multisector else ["inventory"]
    stages = multisector_stages if multisector else _standard_stages
    if multisector:
        source = xr.IndexVariable("source", sources)
        raw = raw.assign_coords(source=source)
        raw["H"] = xr.concat([raw.H, raw.H * 2], dim=source).variable
        basis = BasisFunctions.from_flat_basis(
            basis.flat_basis(), flux=xr.concat([basis.flux, basis.flux * 2], dim=source),
            operator_kwargs={"state_dim": "region"},
        )
    prepared = PreparedInputs(raw, basis, xr.Dataset(
        {"averaging_period": ("site", ["1h", "1h"])}, coords={"site": ["BBB", "AAA"]},
    ))
    data = DataConfig.from_params({
        "species": "ch4", "domain": "EUROPE", "sites": ["BBB", "AAA"],
        "averaging_period": "1h", "start_date": "2020-01-01", "end_date": "2020-03-01",
        "flux_sources": sources,
    }, multisector=multisector)
    manifest = stages.save_preparation(prepared, data_config=data, output_dir=tmp_path / "prepared")
    before = manifest.read_bytes()
    sampler = RhimeSampler(
        draws=2, tune=2, chains=1, sample_prior_predictive=False, sample_posterior_predictive=False,
        sample_kwargs={"cores": 1, "random_seed": 876, "progressbar": False},
    )
    saved = []
    for name, frequency in (("constant", None), ("monthly", "monthly")):
        sample = stages.sample(manifest, model={
            "bc_freq": frequency, "mismatch_model": "additive_sigma",
            "use_minimum_error_floor": True, "min_error": {"AAA": 1., "BBB": .5},
            "bc_prior": {"pdf": "normal", "mu": 1., "sigma": .1},
        }, sampler=sampler, output_dir=tmp_path / name)
        saved.append((sample, stages.load_samples(sample).model_data().copy(deep=True)))
        assert json.loads(sample.read_text())["model"]["bc_freq"] == frequency
    assert manifest.read_bytes() == before
    assert saved[0][1].hbc.shape != saved[1][1].hbc.shape

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved operands must be consumed without recalculation")

    monkeypatch.setattr(BoundaryAlignment, "prepare", forbidden)
    monkeypatch.setattr(MinimumError, "from_observations", forbidden)
    for sample, constants in saved:
        output = stages.load_samples(sample)
        xr.testing.assert_identical(output.model_data(), constants)
        np.testing.assert_allclose(output.model_data().min_error, [.5, .5, 1., 1.])
        assert "min_error" not in output.inv_inputs
        assert output.inv_inputs.H_bc.attrs["boundary_sensitivity_preparation"] == "raw-spatial"

    record = json.loads(manifest.read_text())
    record["schema_version"] = 1
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="version-2.*|matching|software version"):
        stages.load_preparation(manifest)
