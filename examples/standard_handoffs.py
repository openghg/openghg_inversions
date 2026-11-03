"""One real preparation, two priors, and two products from each saved trace.

Run ``python -m examples.standard_handoffs /tmp/standard-handoffs``. Synthetic
merged observations replace object-store acquisition; every scientific stage,
serialization boundary, sampler and product writer is real. Draw counts are
small for an executable acceptance example, not a convergence assessment.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm

from openghg_inversions.basis import BasisFunctions
from openghg_inversions.inversion_data import MergedData, SiteOptions
from openghg_inversions.rhime import stages as handoffs
from openghg_inversions.rhime.params import DataConfig, RhimeConfig
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.rhime.specs import RhimeOutputSpec
from openghg_inversions.rhime.standard import build_standard_model, materialize_standard_inputs, run_standard
from openghg_inversions.postprocessing.make_outputs import basic_output


def synthetic_inputs(destination: Path, *, multiple_sources: bool = False) -> tuple:
    """Create a small supplied-merged case and its explicit scientific resources."""
    destination.mkdir(parents=True, exist_ok=True)
    grid = {"lat": [50., 51.], "lon": [-2., -1.]}
    flux = xr.DataArray([[1., 2.], [3., 4.]], dims=("lat", "lon"), coords=grid,
                        attrs={"units": "mol m-2 s-1"})
    labels = xr.zeros_like(flux, dtype=int)
    # The saved operator's old inventory deliberately differs from current flux.
    basis_path = destination / "operator.nc"
    BasisFunctions.from_flat_basis(basis_flat=labels, flux=flux * 9).save(basis_path)
    country_path = destination / "countries.nc"
    xr.Dataset({"country": labels, "name": ("ncountries", ["United Kingdom"]),
                "country_code": ("ncountries", ["GBR"])}).to_netcdf(country_path)
    dates = pd.date_range("2019-01-02", periods=4, freq="D")
    h = np.array([1., .5, .8, 1.2])
    y = h * 1.1 + np.array([.02, -.03, .04, -.01])
    site = xr.Dataset(
        {"mf": ("time", y), "mf_repeatability": ("time", np.full(4, .08)),
         "mf_variability": ("time", np.full(4, .06)),
         "fp_x_flux": (("lat", "lon", "time"), np.broadcast_to(h / 4, (2, 2, 4)))},
        coords={**grid, "time": dates},
    ).chunk()
    site.mf.attrs["units"] = "ppm"
    sources = {"inventory": flux.to_dataset(name="flux")}
    if multiple_sources:
        supplemental = flux.copy(data=np.array([[.3, 1.2], [2.1, .4]]))
        sources["supplemental"] = supplemental.to_dataset(name="flux")
        flux = flux + supplemental
        flux.attrs["units"] = "mol m-2 s-1"
    merged = MergedData(site_data={"TAC": site}, flux_data=sources,
                             site_options=SiteOptions.from_inputs(sites=["TAC"], averaging_period="1h"))
    preparation_options = dict(species="ch4", sites=["TAC"], averaging_period="1h", domain="EUROPE",
                               start_date="2019-01-01", end_date="2019-02-01", flux_sources=["inventory"],
                               use_bc=False, basis_path=str(basis_path))
    return merged, preparation_options, country_path, flux, h, y


def run_example(destination: Path, *, multiple_sources: bool = False) -> None:
    """Run the saved/full comparison and independent normal-density check."""
    merged, preparation_options, country_path, flux, h, y = synthetic_inputs(
        destination, multiple_sources=multiple_sources,
    )
    site = merged.site_data["TAC"]
    original = site.copy(deep=True)
    preparation = DataConfig.from_params(preparation_options)
    manifest = handoffs.prepare(preparation, output_dir=destination / "prepared", merged_data=merged)
    preparation_bytes = manifest.read_bytes()
    prepared, _, _ = handoffs.load_preparation(manifest)
    np.testing.assert_allclose(prepared.inv_inputs.H, h[None, :])
    np.testing.assert_allclose(prepared.inv_inputs.mf_error, .1)
    xr.testing.assert_equal(site, original)
    assert "mf_error" not in site
    sampler = RhimeSampler(draws=30, tune=30, chains=1, sample_kwargs={"random_seed": 42})
    for index, width in enumerate((.2, .6)):
        model = {"mismatch_model": "fixed_error", "x_prior": {"pdf": "normal", "mu": 1., "sigma": width}}
        samples = handoffs.sample(manifest, model=model, sampler=sampler, output_dir=destination / f"sample-{index}")
        output = RhimeOutputSpec(output_format="basic", country_file=str(country_path),
                                 output_path=str(destination / f"basic-{index}"), save_inversion_output=False)
        replay = handoffs.postprocess(samples, output=output)
        saved = handoffs.postprocess(samples, output=RhimeOutputSpec(
            output_format="inv_out", output_path=str(destination / f"native-{index}")))
        # No intermediate saves in the equivalent full route.
        config = RhimeConfig.from_params({**preparation_options, **model, "draws": 30, "tune": 30,
                                          "chains": 1, "sample_kwargs": {"random_seed": 42},
                                          "output_format": "basic", "output_name": "rhime", "country_file": str(country_path),
                                          "output_path": str(destination / f"full-{index}"),
                                          "save_inversion_output": False, "save_trace": False}, multisector=False)
        full = run_standard(config=config, merged_data=merged)
        xr.testing.assert_allclose(replay.trace["posterior"].to_dataset(), full.trace["posterior"].to_dataset())
        xr.testing.assert_allclose(basic_output(replay, country_file=country_path),
                                   basic_output(full, country_file=country_path))
        assert manifest.read_bytes() == preparation_bytes
        assert replay.run_metadata["start_date"] == "2019-01-01"
        assert replay.run_metadata["end_date"] == "2019-02-01"
        np.testing.assert_allclose(saved.basis_functions.flux.squeeze(), flux)
        scales = saved.trace["posterior"]["x"]
        reconstructed = saved.basis_functions.state_to_flux(scales)
        expected = scales.squeeze("state", drop=True) * flux
        xr.testing.assert_allclose(reconstructed, expected.transpose(*reconstructed.dims))
        # Independent Normal prior + observation likelihood, at varied states.
        # Absolute tolerance covers the configured float32 PyTensor arithmetic.
        model_inputs, priors = materialize_standard_inputs(prepared, config.model)
        built = build_standard_model(model_inputs, config.model, priors=priors)
        logp = built.model.compile_logp()
        for state in (.7, 1., 1.4):
            reference = norm.logpdf(state, 1., width) + norm.logpdf(y, h * state, .1).sum()
            np.testing.assert_allclose(logp({"x": np.array([state])}), reference, rtol=1e-6, atol=1e-5)


if __name__ == "__main__":
    run_example(Path(sys.argv[1]))
