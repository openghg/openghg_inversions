"""Research-only linked centered-BC benchmark; production precision policy is unchanged.

The configuration must name every physical prior/error policy. No policy defaults
are inferred from old traces. Outputs describe a controlled technical benchmark,
not scientific acceptance. Run each variant in a fresh process for memory data.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr

from openghg_inversions.array_ops import select_gathered_data_array
from openghg_inversions.models.coords import add_coords
from openghg_inversions.models.state_activity import prepare_linear_sensitivity, resolve_state_activity
from openghg_inversions.rhime.co2 import Co2O2PreparedInputs
from openghg_inversions.rhime.co2 import centered_boundary as cb
from openghg_inversions.rhime.co2 import co2_cached_sigma_model as cm
from openghg_inversions.rhime.co2 import co2_o2_cached_sigma_model as lm
from openghg_inversions.rhime.co2 import co2_o2_model as linked
from openghg_inversions.serialization import decode_cf_multiindexes, encode_cf_multiindexes


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def load_site(path, site):
    """Select both covariance axes before the public artifact loader validates it."""
    with xr.open_datatree(path) as opened:
        datasets = {}
        for name, node in opened.children.items():
            ds = node.to_dataset()
            indexes = json.loads(ds.attrs["multiindex_dims_json"])
            datasets[name] = decode_cf_multiindexes(ds, indexes) if indexes else ds
        joint = datasets["joint"]
        rows = np.flatnonzero(np.isin(joint.site.values, np.atleast_1d(site)))
        if not rows.size:
            raise ValueError(f"No observations for {site}")
        joint = joint.isel(observation=rows, observation_cov=rows)
        np.testing.assert_array_equal(joint.indexes["observation"], joint.indexes["observation_cov"])
        datasets["joint"] = joint
        for channel in ("co2", "o2"):
            labels = joint.channel_observation.values[joint.species.values == channel]
            ds = datasets[f"{channel}_sensitivity"]
            dim = ds[f"{channel}_effective_sensitivity"].dims[0]
            datasets[f"{channel}_sensitivity"] = ds.sel({dim: labels})
        if "independent_error" in datasets:
            datasets["independent_error"] = datasets["independent_error"].isel(observation=rows)
        encoded = {}
        for name, ds in datasets.items():
            indexes = json.loads(ds.attrs["multiindex_dims_json"])
            encoded[name] = (encode_cf_multiindexes(ds, indexes) if indexes else ds).load()
        tree = xr.DataTree.from_dict(encoded)
        tree.attrs = dict(opened.attrs)
    return Co2O2PreparedInputs.from_datatree(tree)


def prepare(config):
    prepared = load_site(config["prepared"], config["site"])
    observations = prepared.observations
    absolute = observations.copy(deep=True)
    operands = {}
    for channel in ("co2", "o2"):
        raw = xr.load_dataset(config["boundary_operands"][channel])
        sensitivity = getattr(prepared, f"{channel}_sensitivity")
        native = select_gathered_data_array(
            observations,
            key=channel,
            key_dim="species",
            ragged_dim="channel_observation",
            stack_dim="observation",
        )
        native = native.rename({"observation": sensitivity.dims[0]})
        aligned = cb.bind_boundary_operands(raw, native)
        baseline = aligned.H_bc_anomaly.sum("bc_region") + aligned.bc_reference * aligned.H_bc_mass.sum(
            "bc_region"
        )
        rows = np.flatnonzero(observations.species.values == channel)
        # The public loader verifies native labels against gathered joint labels.
        absolute.data[rows] += baseline.sel(
            {sensitivity.dims[0]: observations.channel_observation.values[rows]}
        ).values
        operands[channel] = raw
    error = xr.DataArray(
        [config["independent_error_sd_ppm"][str(s)] for s in observations.species.values],
        dims=observations.dims,
        coords=observations.coords,
        name="independent_error_sd",
        attrs={"units": "ppm"},
    )
    ones = np.ones(prepared.retained_prior.mean.size, dtype="float64")
    native_mean = prepared.fixed_prior_contribution.values + np.concatenate(
        [
            prepared.co2_sensitivity.values @ ones,
            prepared.o2_sensitivity.values @ ones,
        ]
    )
    native_closure = float(np.max(abs(observations.values - native_mean)))
    if config["observations"] == "controlled_model_consistent_truth_plus_joint_gaussian_noise":
        # Explicitly different technical target: preserve and report native closure.
        truth = native_mean + (absolute.values - observations.values)
        noise = config["observation_noise"]
        if "controlled_observations" in config:
            saved = xr.load_dataset(config["controlled_observations"])
            for name, expected in (
                ("species", observations.species),
                ("native_label", observations.channel_observation),
                ("site", observations.site),
                ("time", observations.time),
            ):
                np.testing.assert_array_equal(saved[name].values, expected.values)
            assert saved.attrs["seed"] == noise["seed"]
            np.testing.assert_allclose(saved.truth_model_mean.values, truth, rtol=0, atol=1e-9)
            np.testing.assert_array_equal(saved.independent_error_sd.values, error.values)
            absolute = absolute.copy(data=saved.observed_concentration.values)
        else:
            covariance = prepared.aggregation_error.covariance.values.copy()
            covariance += np.diag(error.values**2)
            hours = observations.time.values.astype("datetime64[ns]").astype("int64") / 3.6e12
            for channel in ("co2", "o2"):
                mask = observations.species.values == channel
                amplitude = noise["truth_ou_amplitude_ppm"][channel]
                same_site = observations.site.values[:, None] == observations.site.values[None, :]
                covariance += (
                    amplitude**2
                    * np.exp(-abs(hours[:, None] - hours[None, :]) / config["tau_hours"])
                    * mask[:, None]
                    * mask[None, :]
                    * same_site
                )
            added = np.linalg.cholesky(covariance) @ np.random.default_rng(noise["seed"]).normal(
                size=observations.size
            )
            absolute = absolute.copy(data=truth + added)
    elif config["observations"] != "saved_native_UOB_enhancement_plus_one_corrected_boundary_no_added_noise":
        raise ValueError("Unsupported explicitly named observation target")
    provenance = dict(
        prepared.provenance,
        precision_native_affine_closure_ppm=native_closure,
        precision_observation_policy=config["observations"],
        precision_observations_sha256=hashlib.sha256(absolute.values.tobytes()).hexdigest(),
    )
    return replace(
        prepared,
        observations=absolute,
        independent_error_sd=error,
        boundary_sensitivity={},
        boundary_operands=operands,
        provenance=provenance,
    )


def input_fingerprint(prepared):
    digest = hashlib.sha256()
    arrays = [
        prepared.observations,
        prepared.fixed_prior_contribution,
        prepared.co2_sensitivity,
        prepared.o2_sensitivity,
        prepared.aggregation_error.covariance,
        prepared.independent_error_sd,
        prepared.retained_prior.mean,
        prepared.retained_prior.arithmetic_covariance,
    ]
    for channel in ("co2", "o2"):
        arrays.extend(prepared.boundary_operands[channel][name] for name in cb.OPERAND_NAMES)
    for array in arrays:
        values = np.asarray(array.values)
        digest.update(str((array.dims, values.shape, values.dtype)).encode())
        digest.update(values.tobytes())
        for dim in array.dims:
            index = array.indexes.get(dim)
            digest.update(repr(None if index is None else index.tolist()).encode())
        digest.update(repr(array.attrs).encode())
    return digest.hexdigest()


@contextmanager
def research_recipe(dtype, capture):
    """Replace ordinary callable components only inside this research process."""
    original_projection = cm._centered_cached_projection
    original_boundary = linked.add_centered_boundary
    original_likelihood = lm._add_cached_likelihood
    projections = []

    def projection(operands, activity, **kwargs):
        design, fixed = original_projection(operands, activity, **kwargs)
        projections.append(design.shape)
        return design.astype(dtype).astype("float64"), fixed

    def boundary(operands, *, prior, activity=None, prefix="", output_dim="nmeasure", owner="bc"):
        anomaly = operands.H_bc_anomaly.transpose(output_dim, "bc_region")
        mass = operands.H_bc_mass.transpose(output_dim, "bc_region")
        reference = operands.bc_reference
        sensitivity = prepare_linear_sensitivity(anomaly + reference * mass, output_dim=output_dim)
        resolved = resolve_state_activity(sensitivity.removed, activity)
        vector = cb.add_state_vector(
            resolved, prior_args=prior, var_name=f"{prefix}bc", coordinate_owner=owner
        )
        design, fixed = original_projection(operands, resolved, output_dim=output_dim)
        dim = f"{prefix}research_active_bc"
        matrix = xr.DataArray(
            design.astype(dtype),
            dims=(output_dim, dim),
            coords={output_dim: anomaly[output_dim], dim: np.arange(design.shape[1])},
        )
        add_coords(matrix.coords, model_dims=matrix.dims)
        data = pytensor.shared(matrix.values, name=f"{prefix}research_active_design", borrow=True)
        pm.modelcontext(None).register_data_var(data, dims=matrix.dims)
        intercept = cb.add_model_data(
            xr.DataArray(fixed, dims=output_dim, coords={output_dim: anomaly[output_dim]}),
            f"{prefix}research_fixed_intercept",
        )
        centered = intercept
        if resolved.n_active:
            centered = centered + pt.dot(
                pt.cast(data, "float64"), vector.state[resolved.active_indices] - 1.0
            )
        shift = (reference * mass.sum("bc_region")).rename(f"{prefix}bc_reference_shift")
        absolute = pm.Deterministic(f"{prefix}mu_bc", centered + cb.add_model_data(shift), dims=output_dim)
        # Research-only data slot is a complete active B, not public anomaly data;
        # this component makes no public API or saved-operand codec claim.
        return cb.CenteredBoundaryResult(
            cb.LinearComponentResult(data, vector.latent, vector.state, absolute, resolved), centered, shift
        )

    def likelihood(terms, **kwargs):
        active = [term for term in terms if term.coefficients is not None]
        coefficients = pt.concatenate([term.coefficients for term in active])
        pm.Deterministic("research_cache_coefficients", coefficients)
        cached = original_likelihood(terms, **kwargs)
        capture["cached"] = cached
        capture["widths"] = [term.active_design.shape[1] for term in active]
        assert len(terms) == len(active) == 3, "Bounded scope is flux + CO2 BC + O2 BC, without offsets"
        assert len(projections) == 2
        for term, design in zip(active[1:], projections, strict=True):
            assert term.active_design.shape == design
        return cached

    # Both 64 and 32 use the same collapsed recipe, isolating storage precision.
    lm._centered_cached_projection = projection
    linked.add_centered_boundary = boundary
    lm._add_cached_likelihood = likelihood
    try:
        yield
    finally:
        lm._centered_cached_projection = original_projection
        linked.add_centered_boundary = original_boundary
        lm._add_cached_likelihood = original_likelihood


def build(prepared, config):
    return lm.build_co2_o2_cached_sigma_model(
        observations=prepared.observations,
        fixed_prior_contribution=prepared.fixed_prior_contribution,
        co2_sensitivity=prepared.co2_sensitivity,
        o2_sensitivity=prepared.o2_sensitivity,
        aggregation_error=prepared.aggregation_error,
        retained_prior=prepared.retained_prior,
        independent_error_sd=prepared.independent_error_sd,
        boundary_operands=prepared.boundary_operands,
        bc_prior=config["bc_prior"],
        tau_hours=config["tau_hours"],
        site_amplitude_prior=config["site_amplitude_prior"],
    )


def consistency(cached):
    model = cached.model
    public = model.logp(vars=model.observed_RVs)
    variables = [
        cached.modelled_mean,
        public,
        cached.cached_state_logp,
        model["research_cache_coefficients"],
        cached.amplitude,
    ]
    state_values = [model.rvs_to_values[rv] for rv in cached.states]
    public, cache = model.replace_rvs_by_values([public, cached.cached_state_logp])
    expressions = model.replace_rvs_by_values(variables)
    expressions += [
        pt.concatenate([g.ravel() for g in pt.grad(public, state_values)]),
        pt.concatenate([g.ravel() for g in pt.grad(cache, state_values)]),
    ]
    evaluate = model.compile_fn(expressions, inputs=model.value_vars, on_unused_input="ignore")
    point = model.initial_point()
    checks = []
    for deviation in (0.0, 1e-8, 1e-5):
        changed = {key: value.copy() for key, value in point.items()}
        # Normal physical priors used by the controlled benchmark expose these keys.
        for key in ("co2_bc", "o2_bc"):
            if key in changed:
                changed[key] = changed[key] + deviation * np.arange(1, changed[key].size + 1)
        mean, logp, cached_logp, coefficients, sigma, grad, cached_grad = evaluate(changed)
        refreshed = cached.target.refresh(sigma)
        cached.shared_cache.update(refreshed)
        mean, logp, cached_logp, coefficients, sigma, grad, cached_grad = evaluate(changed)
        numerical = cached.target.fixed_contribution + cached.target.design @ coefficients
        residual = cached.target.observations - numerical
        direct = cached.target.prepared.evaluate(residual, sigma).log_likelihood
        np.testing.assert_allclose(mean, numerical, rtol=2e-13, atol=2e-9)
        np.testing.assert_allclose(
            [logp, cached_logp, refreshed.log_likelihood(coefficients)], direct, rtol=2e-10, atol=2e-7
        )
        np.testing.assert_allclose(grad, cached_grad, rtol=2e-9, atol=2e-5)
        checks.append(
            {
                "deviation": deviation,
                "max_mean_gap_ppm": float(np.max(abs(mean - numerical))),
                "public_minus_cache_logp": float(logp - cached_logp),
            }
        )
    return checks


def baseline_integrity(original, collapsed):
    """Check collapsed64 against the unpatched production979 recipe."""
    functions = []
    for cached in (original, collapsed):
        model = cached.model
        expressions = model.replace_rvs_by_values(
            [
                cached.modelled_mean,
                model["modelled_concentration"],
                model.logp(vars=model.free_RVs, jacobian=True),
                model.logp(vars=model.observed_RVs),
            ]
        )
        functions.append(model.compile_fn(expressions, inputs=model.value_vars, on_unused_input="ignore"))
    point = original.model.initial_point()
    assert set(point) == set(collapsed.model.initial_point())
    checks = []
    for deviation in (0.0, 1e-8, 1e-5):
        changed = {key: value.copy() for key, value in point.items()}
        for key in ("co2_bc", "o2_bc"):
            if key in changed:
                changed[key] += deviation * np.arange(1, changed[key].size + 1)
        expected, actual = [function(changed) for function in functions]
        for left, right in zip(expected[:2], actual[:2], strict=True):
            np.testing.assert_allclose(left, right, rtol=2e-13, atol=2e-9)
        np.testing.assert_array_equal(expected[2], actual[2])
        np.testing.assert_allclose(expected[3], actual[3], rtol=2e-10, atol=2e-7)
        checks.append(
            {
                "deviation": deviation,
                "max_mean_gap_ppm": float(np.max(abs(expected[0] - actual[0]))),
                "loglikelihood_gap": float(expected[3] - actual[3]),
                "prior_and_jacobian_gap": float(expected[2] - actual[2]),
            }
        )
    return checks
