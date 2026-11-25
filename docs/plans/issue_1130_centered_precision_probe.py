"""Probe centered fixed-OU inference with a float64 cache and local data precision."""

import argparse
import json
from contextlib import ExitStack
from unittest.mock import patch

from pymc.logprob.transforms import Transform

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from boundary_test_utils import bound_model_arguments
from test_co2_centered_boundary import _inputs, _kwargs
from openghg_inversions.inference.cached_sigma import make_cached_sigma_compound_step
from openghg_inversions.models import components, coords
from openghg_inversions.models.boundary import contracts
from openghg_inversions.models.boundary.centered import Correction
from openghg_inversions.models.boundary.multiplicative import CenteredMultiplicativeBoundary
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.postprocessing.boundary_outputs import restore_absolute_predictives
from openghg_inversions.rhime.co2 import build_co2_cached_sigma_model, build_co2_model
from openghg_inversions.rhime.specs import DEFAULT_BC_PRIOR


CENTERED_DATA = {
    "H_bc_anomaly",
    "H_bc_mass",
    "bc_reference",
    "mu_bc_reference",
    "boundary_reference_shift",
}


def local_double_data(name, value, *, dims):
    """Preserve selected data dtypes while retaining ordinary model registration."""
    if name not in CENTERED_DATA:
        return ordinary_data(name, value, dims=dims)
    model = pm.modelcontext(None)
    tensor = pytensor.shared(np.asarray(value, dtype=np.float64), name=model.name_for(name))
    model.register_data_var(tensor, dims=dims)
    return tensor


def local_bc_prior(name, prior_params, **kwargs):
    """Keep the physical multiplicative boundary transform in float64."""
    if name == "bc":
        kwargs["dtype"] = "float64"
    return ordinary_prior(name, prior_params, **kwargs)


class _CenteredLog(Transform):
    """Map a scale deviation on (-1, infinity) to the original log-scale coordinate."""

    name = "log1p"

    def forward(self, value, *inputs):
        """Map the physical deviation to the log scale."""
        return pt.log1p(value)

    def backward(self, value, *inputs):
        """Compute the physical deviation directly in the sampled dtype."""
        return pt.expm1(value)

    def log_jac_det(self, value, *inputs):
        """Return log(exp(value)) for the elementwise inverse transform."""
        return value


def _add_expm1_boundary(boundary, *, output_dim="nmeasure", prefix="", owner="bc"):
    """Prototype the default scale prior on deviations for the all-active fixture.

    Args:
        boundary: Bound centered multiplicative boundary with the default prior.
        output_dim: Model observation dimension.
        prefix: Prefix for registered boundary quantities.
        owner: Namespace for component-owned coordinates.

    Returns:
        The centered affine term with the original support, prior and log-scale
        coordinates. The physical scale is exposed as a deterministic.

    Raises:
        AssertionError: If the fixture uses a different prior or activity policy.
    """
    assert boundary.choice.prior == DEFAULT_BC_PRIOR and boundary.choice.activity is None
    registered = boundary._register_sensitivity(output_dim=output_dim, prefix=prefix, owner=owner)
    data = boundary.data
    sensitivity = data.H_bc_anomaly + data.bc_reference * data.H_bc_mass
    assert (sensitivity != 0).any(output_dim).all()
    deviation = pm.TruncatedNormal(
        f"{prefix}bc_deviation",
        mu=0.0,
        sigma=DEFAULT_BC_PRIOR["sigma"],
        lower=-1.0,
        dims=registered.state_dim,
        default_transform=None,
        transform=_CenteredLog(),
    )
    pm.Deterministic(f"{prefix}bc", deviation + 1.0, dims=registered.state_dim)
    signal = pt.dot(registered.anomaly, deviation) + registered.reference * pt.dot(registered.mass, deviation)
    block = components.AffineBlock(sensitivity, deviation)
    return boundary._term(registered, [Correction(deviation, signal, block, deviation)], signal)


def run_case(*, reference, kind, mode, sample):
    """Compare the centered model at identical physical states across dtypes."""
    inputs = _inputs()
    kwargs = _kwargs(inputs)
    operands = kwargs["boundary_operands"]
    shift_delta = (reference - 400.0) * operands.H_bc_mass.sum("bc_region")
    operands["bc_reference"] = operands.bc_reference.copy(data=np.asarray(reference))
    kwargs["observations"] = inputs.mf.copy(data=inputs.mf.values + shift_delta.values)
    kwargs["state_activity"] = None
    units = "ppm" if reference < 1000.0 else "per meg"
    for name in ("observations", "observation_error", "fixed_prior_contribution"):
        kwargs[name] = kwargs[name].assign_attrs(units=units)
    for name in ("bc_reference", "H_bc_anomaly"):
        operands[name] = operands[name].assign_attrs(units=units)
    if kind == "multiplicative":
        kwargs.pop("bc_mean_shift_prior")
        kwargs.pop("bc_anomaly_scale")
    options = bound_model_arguments(kwargs)
    ou = {"tau_hours": {"AAA": 3.0, "BBB": 7.0}, "fixed_site_amplitudes": 0.4}
    plain = build_co2_model(
        inputs.H,
        likelihood_builder=add_fixed_ou_gaussian_likelihood,
        likelihood_kwargs=ou,
        **options,
    )
    cached = build_co2_cached_sigma_model(inputs.H, **options, **ou)
    assert cached.shared_cache.precision.dtype == "float64"
    model = cached.model
    prior = model.logp(vars=model.free_RVs, jacobian=True)
    native = model.logp(vars=model.observed_RVs)
    quadratic = model.replace_rvs_by_values([cached.cached_state_logp])[0]
    values = model.value_vars
    absolute_mean, centered_mean = model.replace_rvs_by_values(
        [model["modelled_concentration"], cached.modelled_mean]
    )
    evaluate = model.compile_fn(
        [
            native,
            quadratic,
            absolute_mean,
            *pt.grad(prior + native, values),
            *pt.grad(prior + quadratic, values),
        ],
        inputs=values,
    )
    point = model.initial_point()
    points = [
        point,
        {
            name: value
            + np.full_like(
                value, 2.0 / reference if kind == "multiplicative" and name.startswith("bc") else 0.0625
            )
            for name, value in point.items()
        },
    ]
    evaluated = [evaluate(item) for item in points]
    prior_logp_fn = model.compile_logp(vars=model.free_RVs, jacobian=True)
    plain_likelihood = plain.compile_logp(vars=plain.observed_RVs)
    cached_likelihood = model.compile_logp(vars=model.observed_RVs)
    native_mean_fn = model.compile_fn([absolute_mean, centered_mean], inputs=values)
    fused_mean_fn = model.compile_fn(
        [absolute_mean, centered_mean, native, *pt.grad(native, values)], inputs=values
    )
    with model:
        prior_draw = pm.sample_prior_predictive(draws=2, var_names=["y"], random_seed=13)
    restore_absolute_predictives(prior_draw)
    restored = prior_draw["observed_data"]["y"].values
    if mode == "ordinary":
        np.testing.assert_allclose(
            [entry[0] for entry in evaluated],
            [entry[1] for entry in evaluated],
            atol=2e-9,
            rtol=0,
        )
    isolated_native_logp = [float(cached_likelihood(item)) for item in points]
    if mode == "local-expm1" and kind == "multiplicative":
        np.testing.assert_allclose(isolated_native_logp, [entry[0] for entry in evaluated], atol=1e-7, rtol=0)
    sampling = {}
    if sample:
        step = make_cached_sigma_compound_step(
            model=model,
            sigma=None,
            states=cached.states,
            modelled_mean=cached.modelled_mean,
            target=cached.target,
            shared_cache=cached.shared_cache,
            initial_cache=cached.initial_cache,
            cached_state_logp=cached.cached_state_logp,
        )
        with model:
            posterior = pm.sample(
                draws=6,
                tune=6,
                chains=1,
                cores=1,
                step=step,
                random_seed=13,
                progressbar=False,
                compute_convergence_checks=False,
                idata_kwargs={"log_likelihood": True},
            )
            pm.sample_posterior_predictive(
                posterior, var_names=["y"], extend_inferencedata=True, random_seed=13, progressbar=False
            )
        restore_absolute_predictives(posterior)
        sampling = {
            "finite_posterior_predictions": bool(np.isfinite(posterior["posterior_predictive"]["y"]).all()),
            "posterior_dtypes": {
                name: str(array.dtype) for name, array in posterior["posterior"].data_vars.items()
            },
            "finite_log_likelihood": bool(np.isfinite(posterior["log_likelihood"]["y"]).all()),
        }
        assert sampling["finite_log_likelihood"] and sampling["finite_posterior_predictions"]
    names = sorted((CENTERED_DATA | {"Y", "co2_sensitivity", "ou_site_amplitude"}) & set(model.named_vars))
    return {
        "mode": mode,
        "sampling_smoke": sampling,
        "floatX": pytensor.config.floatX,
        "boundary": kind,
        "reference": reference,
        "data_dtypes": {name: model[name].dtype for name in names},
        "point_dtypes": {name: str(value.dtype) for name, value in point.items()},
        "prediction_dtype": str(prior_draw["prior_predictive"]["y"].dtype),
        "restored_observation_max_error": float(np.max(np.abs(restored - kwargs["observations"].values))),
        "plain_logp": [float(plain_likelihood(item)) for item in points],
        "native_logp": [float(entry[0]) for entry in evaluated],
        "state_cache_logp": [float(entry[1]) for entry in evaluated],
        "absolute_mean": [entry[2].tolist() for entry in evaluated],
        "mean_declared_dtypes": [absolute_mean.dtype, centered_mean.dtype],
        "centered_mean": [native_mean_fn(item)[1].tolist() for item in points],
        "centered_mean_with_gradient": [fused_mean_fn(item)[1].tolist() for item in points],
        "isolated_native_logp": isolated_native_logp,
        "prior_logp": [float(prior_logp_fn(item)) for item in points],
        "native_gradient": [
            np.concatenate([np.ravel(v) for v in entry[3 : 3 + len(values)]]).tolist() for entry in evaluated
        ],
        "cached_gradient": [
            np.concatenate([np.ravel(v) for v in entry[3 + len(values) :]]).tolist() for entry in evaluated
        ],
    }


ordinary_data = coords.pm.Data
ordinary_prior = components.parse_prior


def main():
    """Run the chosen precision experiment in a fresh PyTensor process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("ordinary", "guard-only", "local", "local-prior", "local-expm1"))
    parser.add_argument("--sample", action="store_true", help="Sample the larger multiplicative case.")
    args = parser.parse_args()
    with ExitStack() as stack:
        if args.mode != "ordinary":
            stack.enter_context(patch.object(contracts, "require_centered_precision", lambda: None))
        if args.mode in ("local", "local-prior", "local-expm1"):
            stack.enter_context(patch.object(coords.pm, "Data", local_double_data))
        if args.mode == "local-prior":
            stack.enter_context(patch.object(components, "parse_prior", local_bc_prior))
        if args.mode == "local-expm1":
            stack.enter_context(patch.object(CenteredMultiplicativeBoundary, "add", _add_expm1_boundary))
        for reference in (400.000027, 209300.0021):
            for kind in ("additive", "multiplicative"):
                result = run_case(
                    reference=reference,
                    kind=kind,
                    mode=args.mode,
                    sample=args.sample and kind == "multiplicative" and reference > 1000,
                )
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
