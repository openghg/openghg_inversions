"""Reuse one multisector preparation for Normal and Student-t inference.

Run ``uv run python -m examples.multisector_handoffs /tmp/multisector-handoffs``.
The supplied synthetic observations replace object-store acquisition. Preparation,
sampling, checkpoint codecs, flux reconstruction and output exports are real.
Small draw counts demonstrate execution parity, not sampler convergence.
For a narrated notebook, see ``docs/usage/multisector_handoffs.rst``.
"""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, t
import xarray as xr

from examples.rhime_customisation.likelihoods import likelihood_builder
from openghg_inversions.basis import BasisFunctions
from openghg_inversions.inversion_data import MergedData, SiteOptions
from openghg_inversions.postprocessing.make_outputs import make_sector_flux_outputs
from openghg_inversions.rhime import multisector_stages as handoffs
from openghg_inversions.rhime.multisector import (
    build_multisector_model,
    materialize_multisector_inputs,
    run_multisector,
)
from openghg_inversions.rhime.params import DataConfig, RhimeConfig
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.rhime.specs import RhimeOutputSpec


def synthetic_inputs(destination: Path) -> tuple:
    """Provide two distinct source fields and known source-resolved responses."""
    destination.mkdir(parents=True, exist_ok=True)
    grid = {"lat": [50., 51.], "lon": [-2., -1.]}
    inventory = xr.DataArray(
        [[1., 2.], [3., 4.]],
        dims=("lat", "lon"),
        coords=grid,
        attrs={"units": "mol m-2 s-1"},
    )
    supplemental = inventory.copy(data=np.array([[.3, 1.2], [2.1, .4]]))
    labels = xr.zeros_like(inventory, dtype=int)
    basis_path = destination / "operator.nc"
    # Retained operator metadata must never override current source inventories.
    BasisFunctions.from_flat_basis(basis_flat=labels, flux=inventory * 9).save(basis_path)
    dates = pd.date_range("2019-01-02", periods=4, freq="D")
    h = {
        "inventory": np.array([1., .5, .8, 1.2]),
        "supplemental": np.array([.2, .7, .4, .1]),
    }
    y = h["inventory"] * 1.1 + h["supplemental"] * .9 + np.array([.02, -.03, .04, -.01])
    response = np.stack([np.broadcast_to(h[source] / 4, (2, 2, 4)) for source in h])
    site = xr.Dataset(
        {
            "mf": ("time", y),
            "mf_repeatability": ("time", np.full(4, .08)),
            "mf_variability": ("time", np.full(4, .06)),
            "fp_x_flux_sectoral": (("source", "lat", "lon", "time"), response),
        },
        coords={**grid, "time": dates, "source": list(h)},
    ).chunk()
    site.mf.attrs["units"] = "ppm"
    flux = {"inventory": inventory, "supplemental": supplemental}
    merged = MergedData(
        site_data={"TAC": site},
        flux_data={source: field.to_dataset(name="flux") for source, field in flux.items()},
        site_options=SiteOptions.from_inputs(sites=["TAC"], averaging_period="1h"),
        split_by_sectors=True,
    )
    options = dict(
        species="ch4",
        sites=["TAC"],
        averaging_period="1h",
        domain="EUROPE",
        start_date="2019-01-01",
        end_date="2019-02-01",
        use_bc=False,
        flux_sources=["supplemental", "inventory"],
        basis_path=str(basis_path),
    )
    return merged, options, flux, h, y


def model_options(width: float, *, student_t: bool = False) -> dict:
    """Declare scientific sector labels independently of OpenGHG source names."""
    return {
        "sector_sources": {"biosphere": "supplemental", "fossil": "inventory"},
        "sector_priors": {
            "biosphere": {"pdf": "normal", "mu": .9, "sigma": width},
            "fossil": {"pdf": "normal", "mu": 1.1, "sigma": width},
        },
        "mismatch_model": None if student_t else "fixed_error",
    }


def run_example(destination: Path, *, draws: int = 20) -> None:
    """Compare full/prepared/saved execution against independent equations."""
    merged, options, flux, h, y = synthetic_inputs(destination)
    site = merged.site_data["TAC"]
    original = site.copy(deep=True)
    graph = site.fp_x_flux_sectoral.data

    # Prepare observation errors, current native flux and sensitivities once.
    data = DataConfig.from_params(options, multisector=True)
    manifest = handoffs.prepare(data, merged_data=merged, output_dir=destination / "prepared")
    original_manifest = manifest.read_bytes()
    original_inputs = (manifest.parent / "prepared-inputs.nc").read_bytes()
    prepared, _, _ = handoffs.load_preparation(manifest)
    assert tuple(prepared.inv_inputs.H.source.values) == tuple(merged.flux_data)
    for source in h:
        np.testing.assert_allclose(prepared.inv_inputs.H.sel(source=source).squeeze(), h[source])
        np.testing.assert_allclose(prepared.basis_functions.flux.sel(source=source), flux[source])
    np.testing.assert_allclose(prepared.inv_inputs.mf_error, .1)
    xr.testing.assert_equal(site, original)
    assert "mf_error" not in site and site.fp_x_flux_sectoral.data is graph

    sampler = RhimeSampler(
        draws=draws,
        tune=draws,
        chains=1,
        sample_kwargs={"cores": 1, "random_seed": 42},
        sample_prior_predictive=False,
        sample_posterior_predictive=False,
    )
    for index, (width, student_t) in enumerate(((.2, False), (.6, True))):
        # Prior width and likelihood change at construction, after preparation.
        model = model_options(width, student_t=student_t)
        custom = {"likelihood_builder": likelihood_builder} if student_t else {}
        sampled = handoffs.sample(
            manifest,
            model=model,
            sampler=sampler,
            output_dir=destination / f"sample-{index}",
            **custom,
        )
        output = RhimeOutputSpec(
            output_format="inv_out",
            output_name=f"replay-{index}",
            output_path=str(destination / f"native-{index}"),
        )
        replay = handoffs.postprocess(sampled, output=output)
        # The same trace can feed new reporting choices without another graph.
        quiet = handoffs.postprocess(sampled, output=RhimeOutputSpec(output_format="none"))

        # Full and prepared-input runners must reproduce the saved route.
        config = RhimeConfig.from_params(
            {
                **options,
                **model,
                "output_name": "full",
                "output_format": "inv_out",
                "save_inversion_output": False,
            },
            multisector=True,
        )
        config = replace(config, sampler=sampler)
        full = run_multisector(config=config, merged_data=merged, **custom)
        resumed = run_multisector(config=config, prepared=prepared, **custom)
        for result in (full, resumed):
            xr.testing.assert_allclose(
                replay.trace["posterior"].to_dataset(),
                result.trace["posterior"].to_dataset(),
            )
            xr.testing.assert_allclose(
                make_sector_flux_outputs(replay, stats=["mean"]),
                make_sector_flux_outputs(result, stats=["mean"]),
            )
        assert replay.trace.equals(quiet.trace)
        assert replay.run_metadata["start_date"] == "2019-01-01"
        assert replay.run_metadata["end_date"] == "2019-02-01"
        assert [sector["name"] for sector in replay.model_metadata["sectors"]] == ["biosphere", "fossil"]
        posterior = replay.trace["posterior"]
        scales = posterior[replay.variable_name("flux_scale")]
        expected_total = (
            scales.sel(source="supplemental", drop=True).mean(("chain", "draw")).squeeze() * flux["supplemental"]
            + scales.sel(source="inventory", drop=True).mean(("chain", "draw")).squeeze() * flux["inventory"]
        )
        # Native reporting preserves each inventory's within-region spatial pattern.
        diagnostics = make_sector_flux_outputs(replay, stats=["mean"], report_flux_on_inversion_grid=False)
        np.testing.assert_allclose(diagnostics["flux_total_posterior_mean"].squeeze(), expected_total)

        # Check the concrete probability model against independent equations.
        model_inputs, priors = materialize_multisector_inputs(prepared, config.model)
        built = build_multisector_model(model_inputs, config.model, priors=priors, **custom)
        logp = built.model.compile_logp()
        for biosphere, fossil in ((.7, .8), (.9, 1.1), (1.3, 1.4)):
            mean = h["supplemental"] * biosphere + h["inventory"] * fossil
            density = t.logpdf(y, df=4., loc=mean, scale=.1) if student_t else norm.logpdf(y, mean, .1)
            reference = norm.logpdf(biosphere, .9, width) + norm.logpdf(fossil, 1.1, width) + density.sum()
            np.testing.assert_allclose(logp({"x_biosphere_active": np.array([biosphere]),
                                            "x_fossil_active": np.array([fossil])}),
                                       reference, rtol=1e-6, atol=1e-5)
        assert manifest.read_bytes() == original_manifest
        assert (manifest.parent / "prepared-inputs.nc").read_bytes() == original_inputs


if __name__ == "__main__":
    run_example(Path(sys.argv[1]))
