"""Run linked CO2/O2 checkpoints using synthetic same-unit scientific inputs.

With PyMC 6/Numba, set ``floatX=float64`` in ``PYTENSOR_FLAGS`` before
running ``python -m examples.co2_o2_handoffs /tmp/co2-o2-handoffs``. Retain
other flag entries and replace an existing ``floatX`` setting. The two channels have unequal observation axes and a nonzero cross covariance. Short
chains demonstrate the workflow; they do not establish convergence.
"""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.rhime.co2 import Co2O2PreparedInputs, prepare_co2_o2_inputs


FACTS = {"start_date": "2020-01-01", "end_date": "2020-02-01", "domain": "EUROPE"}


def synthetic_inputs() -> Co2O2PreparedInputs:
    """Construct borrowed labelled inputs with shared land and separate ocean states."""
    state = ["gpp:1", "ter:1", "ff:1", "co2-ocean:1", "o2-ocean:1"]
    source = ["GPP", "TER", "FF", "ocean", "ocean"]
    mean = xr.DataArray(np.ones(5), dims="state", coords={
        "state": state, "source": ("state", source),
        "tracer_scope": ("state", ["shared", "shared", "shared", "co2", "o2"]),
    }, attrs={"units": "dimensionless flux scale"})
    co2 = xr.DataArray([2., 3.], dims="co2_measure", coords={
        "co2_measure": ["c1", "c2"], "site": ("co2_measure", ["MHD", "MHD"]),
        "time": ("co2_measure", np.array(["2020-01-10", "2020-01-12"], dtype="datetime64[ns]")),
    })
    o2 = xr.DataArray([-4., -5., -6.], dims="o2_measure", coords={
        "o2_measure": ["o1", "o2", "o3"], "site": ("o2_measure", ["MHD"] * 3),
        "time": ("o2_measure", np.array(["2020-01-11", "2020-01-13", "2020-01-14"],
                                        dtype="datetime64[ns]")),
    })
    prepared = prepare_co2_o2_inputs(
        co2_observations=co2, o2_observations=o2,
        co2_prior_forward_mean=co2 - .25, o2_prior_forward_mean=o2 + .5,
        co2_sensitivity=xr.DataArray([[1., 2., 3., 4., 0.], [.5, 1., 1.5, 2., 0.]],
                                    dims=("co2_measure", "state"),
                                    coords={"co2_measure": co2.co2_measure, "state": mean.state}),
        o2_sensitivity=xr.DataArray([[-1., -2., -3., 0., 5.], [-.5, -1., -1.5, 0., 2.5],
                                    [-.2, -.4, -.6, 0., 1.]], dims=("o2_measure", "state"),
                                   coords={"o2_measure": o2.o2_measure, "state": mean.state}),
        o2_co2_flux_ratio=xr.DataArray([-1.1, -1., -1.4], dims="state", coords={
            "state": state[:3], "source": ("state", source[:3]),
        }, attrs={"direction": "O2 flux per CO2 flux", "sign_convention": "signed; positive CO2 flux has negative O2 loading",
                  "provenance": "synthetic example"}),
        o2_co2_flux_ratio_unavailable_reason=None,
        co2_aggregation_covariance=xr.DataArray([[1., .2], [.2, 1.5]],
            dims=("co2_measure", "co2_measure_cov"),
            coords={"co2_measure": co2.co2_measure, "co2_measure_cov": co2.co2_measure.values}),
        co2_o2_aggregation_covariance=xr.DataArray([[-.1, -.05, -.02], [-.08, -.04, -.01]],
            dims=("co2_measure", "o2_measure"),
            coords={"co2_measure": co2.co2_measure, "o2_measure": o2.o2_measure}),
        o2_aggregation_covariance=xr.DataArray(np.diag([2., 2.5, 3.]),
            dims=("o2_measure", "o2_measure_cov"),
            coords={"o2_measure": o2.o2_measure, "o2_measure_cov": o2.o2_measure.values}),
        retained_prior=CorrelatedLognormalPrior(mean, np.diag([.01, .02, .03, .04, .05])),
        co2_units="ppm", o2_units="ppm", provenance={**FACTS, "example": "synthetic linked inputs"},
    )
    errors = xr.ones_like(prepared.observations).rename("fixed_independent_error_sd")
    return replace(prepared, independent_error_sd=errors)


def run_example(destination: Path, *, draws: int = 20) -> None:
    """Save inputs, check priors, sample and replay basic scientific products."""
    from openghg_inversions.inference.sampling import RhimeSampler
    from openghg_inversions.rhime.co2 import co2_o2_handoffs as handoffs
    from openghg_inversions.rhime.co2.co2_o2_outputs import Co2O2OutputOptions

    model = {"channels": {"co2": {"units": "ppm"}, "o2": {"units": "ppm"}}}
    preparation = handoffs.save_preparation(synthetic_inputs(), output_dir=destination / "prepare")
    handoffs.prior_predictive(preparation, model=model, output_dir=destination / "prior",
                             draws=draws, random_seed=19)
    samples = handoffs.sample(preparation, model=model, output_dir=destination / "sample", sampler=RhimeSampler(
        draws=draws, tune=draws, chains=2, nuts_sampler="pymc",
        sample_kwargs={"random_seed": 19}, sample_prior_predictive=draws,
        sample_posterior_predictive=True,
    ))
    handoffs.postprocess(samples, output=Co2O2OutputOptions(
        output_format="basic", output_name="linked", output_path=destination / "products",
    ))


if __name__ == "__main__":
    run_example(Path(sys.argv[1]))
