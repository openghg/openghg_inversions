"""Offline native-template linked experiment with known positive flux truth.

Synthetic assumptions: one unit-area footprint cell, four directional endpoint
masses of 1/4, hourly integrated transport, explicit zero unresolved covariance,
0.05 ppm measurement error SD, noiseless truth observations, and distinct already
signed GPP/TER/FF O2 responses.
The scalar baseline injections are physical ppm. They are controls, not tuning.
"""

from __future__ import annotations

from dask import compute
import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.inversion_data import MergedData, SiteOptions, prepare_surface_site
from openghg_inversions.inversion_data.acquisition import AcquisitionFacts
from openghg_inversions.observation_error import AggregationError
from openghg_inversions.rhime.co2 import Co2O2PreparedInputs, prepare_linked_surface_templates


STATE_LABELS = ["gpp", "ter", "ff", "co2_ocean", "o2_ocean"]
TRUTH = np.array([1.15, .85, 1.2, 1., 1.])
SOURCE_STATE = {channel: {source: source for source in ("gpp", "ter", "ff", f"{channel}_ocean")}
                for channel in ("co2", "o2")}


def synthetic_surface_inputs(
    *, injected_baseline: bool = False,
) -> tuple[MergedData, MergedData]:
    """Prepare tiny independent native channels through prepare_surface_site.

    CO2 retains six hourly rows; O2 omits the first and fourth row. Zero and
    injected controls share the same noiseless flux truth.
    All source and endpoint numerical arrays are Dask-backed.

    Args:
        injected_baseline: Add the declared physical atmospheric shifts.

    Returns:
        Independently prepared CO2 and O2 surface data with native row supports.
    """
    times = pd.date_range("2020-01-01", periods=6, freq="h")
    footprint = xr.Dataset(
        {"fp": (("lat", "lon", "time"), np.ones((1, 1, 6)), {"units": "s m2 / mol"})},
        coords={"lat": [0.0], "lon": [0.0], "time": times, "height": [10.]},
        attrs={"model": "synthetic", "met_model": "synthetic"},
    )
    for direction in "nesw":
        axis = "lon" if direction in "ns" else "lat"
        footprint[f"particle_locations_{direction}"] = xr.DataArray(
            np.full((1, 1, 6), .25), dims=("height", axis, "time"), attrs={"units": "1"},
        )
    patterns = {
        "gpp": -np.array([.2, .6, 1., .9, .4, .1]),
        "ter": np.array([.8, .7, .5, .4, .6, .9]),
        "ff": np.array([.2, .7, .3, .8, .4, .6]),
        "co2_ocean": np.array([.1, .12, .16, .2, .18, .14]),
        "o2_ocean": np.array([.3, .2, .1, .15, .25, .35]),
    }
    channels = []
    for channel, reference in (("co2", 400.), ("o2", 209000.)):
        selected = np.arange(6) if channel == "co2" else np.array([1, 2, 4, 5])
        ratios = {"gpp": -1.1, "ter": -1.2, "ff": -1.4}
        sources = SOURCE_STATE[channel]
        fluxes = {}
        response = np.zeros(6)
        for source in sources:
            values = patterns[source] * (ratios.get(source, 1.) if channel == "o2" else 1.)
            fluxes[source] = xr.Dataset(
                {"flux": (("lat", "lon", "time"), values[None, None, :] * 1e-6,
                          {"units": "mol / m2 / s"})},
                coords={"lat": [0.], "lon": [0.], "time": times},
            ).chunk({"time": 3})
            response += values * TRUTH[STATE_LABELS.index(source)]
        delta = (0.35 if channel == "co2" else -0.45) if injected_baseline else 0.
        obs = xr.Dataset(
            {"mf": ("time", reference + response[selected] + delta,
                    {"units": "ppm"}),
             "mf_error": ("time", np.full(len(selected), .05), {"units": "ppm"})},
            coords={"time": times[selected]},
            attrs={"species": channel, "site": "SYNTH", "inlet": "10m",
                   "sampling_period": 3600, "platform": "surface"},
        )
        boundary = xr.Dataset(coords=footprint.coords, attrs={"time_period": "1 hour"})
        for direction in "nesw":
            boundary[f"vmr_{direction}"] = xr.full_like(
                footprint[f"particle_locations_{direction}"], reference,
            ).assign_attrs(units="ppm")
        site = prepare_surface_site(
            obs.chunk({"time": 2}), footprint.chunk({"time": 3}), fluxes,
            species=channel, transport=dict.fromkeys(sources, "integrated"),
            boundary_data=boundary.chunk({"time": 3}), boundary_representation="centered",
            boundary_reference=reference, boundary_reference_units="ppm",
            boundary_mass_tolerance=1e-6, output_units="ppm",
        )
        channels.append(MergedData(
            site_data={"SYNTH": site}, flux_data=fluxes,
            site_options=SiteOptions.from_inputs(
                sites=["SYNTH"], inlet="10m", fp_height="10m", averaging_period="1h", platform="surface",
            ), boundary_data=boundary, split_by_sectors=True,
            acquisition=AcquisitionFacts(species=channel, domain="SYNTHETIC",
                                         start_date="2020-01-01", end_date="2020-01-02", use_bc=True),
        ))
    return channels[0], channels[1]


def synthetic_linked_inputs(
    *, injected_baseline: bool = False, fixed_outer_truth: bool = False,
) -> Co2O2PreparedInputs:
    """Declare positive prior, source map and explicit zero unresolved covariance.

    Args:
        injected_baseline: Add the declared physical atmospheric shifts.
        fixed_outer_truth: Set fixed outer prior means to their known truth.

    Returns:
        Linked native templates in physical ppm, without frozen shared preparation.
    """
    co2, o2 = synthetic_surface_inputs(injected_baseline=injected_baseline)
    prior_mean = np.array([TRUTH[0], TRUTH[1], 1., 1., 1.]) if fixed_outer_truth else np.ones(5)
    mean = xr.DataArray(prior_mean, dims="state", coords={
        "state": STATE_LABELS, "source": ("state", ["gpp", "ter", "ff", "ocean", "ocean"]),
        "tracer_scope": ("state", ["shared", "shared", "shared", "co2", "o2"]),
    })
    prior = CorrelatedLognormalPrior(mean, xr.DataArray(
        np.eye(5) * .04, dims=("state", "state_covariance"), coords={"state": STATE_LABELS, "state_covariance": STATE_LABELS},
    ))
    index = pd.MultiIndex.from_tuples(
        [(channel, row) for channel, data in (("co2", co2), ("o2", o2))
         for row in range(sum(ds.sizes["time"] for ds in data.site_data.values()))],
        names=["species", "channel_observation"],
    )
    covariance = xr.DataArray(np.zeros((len(index), len(index))), dims=("observation", "observation_cov"))
    covariance = covariance.assign_coords(xr.Coordinates.from_pandas_multiindex(index, "observation"))
    covariance = covariance.assign_coords(xr.Coordinates.from_pandas_multiindex(
        index.set_names(["species_cov", "channel_observation_cov"]), "observation_cov",
    )).assign_coords(observation_units=("observation", ["ppm"] * len(index)),
                     observation_units_cov=("observation_cov", ["ppm"] * len(index)))
    return prepare_linked_surface_templates(
        co2, o2, retained_prior=prior, source_state=SOURCE_STATE,
        aggregation_error=AggregationError("dense", np.zeros(len(index)), covariance=covariance),
        provenance={"synthetic": True, "control": "injected" if injected_baseline else "zero",
                    "positive_flux_truth": dict(zip(STATE_LABELS, TRUTH.tolist(), strict=True))},
    )


def prepare_control(
    *, injected_baseline: bool = False, mode: str = "multiplicative",
) -> Co2O2PreparedInputs:
    """Freeze native operators and generate targets with the shared recipe.

    Args:
        injected_baseline: Inject +0.35 ppm CO2 and -0.45 ppm O2 atmospheric levels.
        mode: Supported fixed, multiplicative or multiplicative_shrunk outer mode.

    Returns:
        Frozen preparation with jointly materialized noiseless truth targets.
        The original transport arrays remain borrowed.
    """
    from openghg_inversions.biosphere_prior import calibrate_biosphere_prior
    from openghg_inversions.models.shared_baseline import SharedBaselineChoice
    from openghg_inversions.rhime.co2 import prepare_linked_shared_inputs

    linked = synthetic_linked_inputs(injected_baseline=injected_baseline, fixed_outer_truth=mode == "fixed")
    mean = linked.retained_prior.mean
    outer = mean.copy(data=[True, True, False, True, True])
    sampled = mean.copy(data=[True, True, False, False, False])
    calibration = None
    if mode == "multiplicative_shrunk":
        weights = xr.DataArray([[-1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]],
                              dims=("process", "state"), coords={"process": ["GPP", "TER"], **mean.coords},
                              attrs={"units": "kg/year"})
        reference = xr.DataArray([-1., 1.], dims="process", coords={"process": ["GPP", "TER"]},
                                 attrs={"units": "kg/year"})
        calibration = calibrate_biosphere_prior(linked.retained_prior, biosphere=sampled,
                                               weights=weights, reference=reference, residual_sd_multiplier=.5)
    prepared = prepare_linked_shared_inputs(
        linked, choices={name: SharedBaselineChoice(None, 1.0, mode) for name in ("co2", "o2")},
        outer=outer, sampled_outer=sampled,
        calibrated_prior=calibration,
    )
    from dataclasses import replace
    from openghg_inversions.rhime.co2 import reconstruct_linked_shared_baseline

    # Raw surface data above establish transport/support. Generate synthetic
    # concentration targets after freezing the full-support atmospheric means.
    truth = xr.DataArray(TRUTH, dims="state", coords={"state": STATE_LABELS})
    selected = prepared.shared_baseline["co2"].data.state
    coefficients = xr.Dataset({"flux_scaling": truth.sel(state=selected)})
    for channel, delta in (("co2", .35), ("o2", -.45)):
        coefficients[f"{channel}_bc_mean_shift"] = xr.DataArray(
            delta if injected_baseline else 0.,
        )
    native = reconstruct_linked_shared_baseline({channel: bound.data for channel, bound in prepared.shared_baseline.items()}, coefficients)
    # Explicit target boundary: resolve both related native targets together.
    native = compute(native)[0]
    targets = np.empty(prepared.observations.size, dtype=np.float64)
    for channel, values in native.items():
        positions = np.flatnonzero(prepared.observations.species.values == channel)
        row = prepared.shared_baseline[channel].data.attrs["native_observation_dim"]
        labels = prepared.observations.channel_observation.values[positions]
        targets[positions] = values.modelled_concentration.sel({row: labels}).data
    return replace(prepared, observations=prepared.observations.copy(data=targets),
                   provenance={**prepared.provenance, "synthetic_target_generator": "reviewed linked shared-baseline reconstruction"})


def run_control(output_dir, *, injected_baseline=False, mode="multiplicative", likelihood="fixed_independent", draws=250, tune=250):
    """Run saved preparation, configured sampling and graph-free saved products.

    Submit this function on Slurm. The report records recovery errors without
    imposing a scientific acceptance threshold on this tiny illustrative model.

    Args:
        output_dir: Destination for preparation, samples, products and recovery.json.
        injected_baseline: Inject the declared physical atmospheric shifts.
        mode: Supported shared outer mode.
        likelihood: fixed_independent, fixed_ou or inferred_ou.
        draws: Retained posterior draws per chain.
        tune: Warmup draws per chain.

    Returns:
        Recovery diagnostics and physical flux and atmospheric level summaries.
    """
    import json
    from pathlib import Path
    from openghg_inversions.rhime.co2 import Co2O2OutputOptions
    from openghg_inversions.rhime.co2.co2_o2_handoffs import save_preparation, sample, postprocess

    destination = Path(output_dir)
    preparation = save_preparation(prepare_control(injected_baseline=injected_baseline, mode=mode),
                                   output_dir=destination / "prepare")
    request = dict(recipe="co2_o2_shared_baseline", channels={name: {"units": "ppm"} for name in ("co2", "o2")},
                   likelihood={"kind": "fixed_independent"},
                   sampling={"draws": draws, "tune": tune, "chains": 2, "nuts_sampler": "pymc",
                             "random_seed": 246, "target_accept": .95})
    if likelihood in {"fixed_ou", "inferred_ou"}:
        request["likelihood"] = {"kind": "fixed_ou", "tau_hours": 2.}
        if likelihood == "fixed_ou":
            request["likelihood"]["fixed_site_amplitudes"] = .1
        else:
            request["likelihood"]["site_amplitude_prior"] = {"pdf": "halfnormal", "sigma": .1}
            request["likelihood"]["initial_site_amplitudes"] = .05
            request["sampling"].pop("target_accept")
    elif likelihood != "fixed_independent":
        raise ValueError("Unsupported synthetic likelihood.")
    samples = sample(preparation, model=request, output_dir=destination / "sample")
    products = postprocess(samples, output=Co2O2OutputOptions(output_path=destination / "products",
                                                           output_name="synthetic", save_concentrations=True))
    posterior = products.idata["posterior"]
    from openghg_inversions.inference.diagnostics import posterior_summary
    names = ["flux_scaling", "co2_bc_mean_shift", "o2_bc_mean_shift"]
    if "ou_site_amplitude" in posterior:
        names.append("ou_site_amplitude")
    diagnostics = posterior_summary(posterior.to_dataset()[names])
    baseline_truth = {"co2": .35, "o2": -.45} if injected_baseline else {"co2": 0., "o2": 0.}
    report = {"control": "injected" if injected_baseline else "zero", "outer_mode": mode, "likelihood": likelihood,
              "baseline_truth_ppm": baseline_truth, "positive_flux_truth": dict(zip(STATE_LABELS, TRUTH.tolist(), strict=True)),
              "baseline_posterior_mean_ppm": {name: float(posterior[f"{name}_bc_mean_shift"].mean()) for name in ("co2", "o2")},
              "physical_flux_posterior_mean": products.outputs["state_scaling"].flux_scale_posterior.mean(("chain", "draw")).values.tolist()}
    report["baseline_95pct_interval_ppm"] = {
        name: np.quantile(posterior[f"{name}_bc_mean_shift"].values, [.025, .975]).tolist()
        for name in ("co2", "o2")
    }
    report["diagnostics"] = {
        "max_rhat": max(float(value.sel(metric="r_hat").max()) for value in diagnostics.data_vars.values()),
        "min_ess_bulk": min(float(value.sel(metric="ess_bulk").min()) for value in diagnostics.data_vars.values()),
        "divergences": int(products.idata["sample_stats"].diverging.sum()),
    }
    (destination / "recovery.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["fixed", "multiplicative", "multiplicative_shrunk"], default="multiplicative")
    parser.add_argument("--likelihood", choices=["fixed_independent", "fixed_ou", "inferred_ou"], default="fixed_independent")
    parser.add_argument("--draws", type=int, default=250)
    parser.add_argument("--tune", type=int, default=250)
    args = parser.parse_args()
    for injected in (False, True):
        run_control(f"{args.output_dir}/{'injected' if injected else 'zero'}", injected_baseline=injected,
                    mode=args.mode, likelihood=args.likelihood, draws=args.draws, tune=args.tune)
