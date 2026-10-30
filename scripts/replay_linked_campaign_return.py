"""Replay frozen July linked CO2/O2 states through PyMC's real return path.

No inference is run. Retained states are read from the successful campaign and
repeated as synthetic warmup; they are not the original warmup trajectories.
The sampling loop is replaced by NDArray recording plus PyMC ``_sample_return``.
The production runner and RhimeSampler execute the return sequence, with the
storage strategy selected explicitly and the allocation phases instrumented.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import resource
import sys
import subprocess
import traceback
from unittest.mock import patch
import weakref

import numpy as np
import pandas as pd
import pymc as pm
from pymc.backends.arviz import DataTreeConverter
from pymc.backends.ndarray import NDArray
from pymc.sampling.mcmc import _sample_return
from pymc.stats import log_density
import pytensor
import xarray as xr

import openghg_inversions

from profile_linked_return import phase

from openghg_inversions.array_ops import select_gathered_data_array
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.inference import sampling
from openghg_inversions.inference.diagnostics import posterior_summary
from openghg_inversions.models.boundary import bind_boundary_operands
from openghg_inversions.observation_error import AggregationError
from openghg_inversions.rhime.co2 import Co2O2PreparedInputs
from openghg_inversions.rhime.co2 import co2_o2_cached_sigma_runner as runner
from openghg_inversions.rhime.co2.co2_o2_preparation import _stored_array
from openghg_inversions.rhime.sampling import RhimeSampler, sample_rhime_model
from openghg_inversions.serialization import (
    decode_cf_multiindexes,
    restore_declared_multiindexes,
    save_trace,
)


CAMPAIGN = Path(
    "/group/chem/acrg/verification_games_round_2/verification-games-work/"
    "ope225-bc-offset-20261029-r1/linked-uob-retry-r2"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_prepared(request: dict) -> Co2O2PreparedInputs:
    """Decode trusted frozen inputs, omitting the unused dense covariance.

    The public loader validates the dense 10628-square covariance before the
    campaign replaces it with the already-reviewed rank-256 sidecar. This
    benchmark verifies artifact hashes, decodes the same noncovariance arrays,
    and uses the sidecar directly, avoiding that unrelated eigendecomposition.
    """
    manifest_path = Path(request["prepared_manifest"])
    assert file_sha256(manifest_path) == request["prepared_manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "public-assembly-and-six-bindings-passed"
    assert manifest["exact_co2_marginal"]
    for key in ("shared_prepared", "low_rank"):
        assert file_sha256(Path(manifest[key]["path"])) == manifest[key]["sha256"]
    with xr.open_datatree(manifest["shared_prepared"]["path"]) as tree:
        datasets = {}
        for name, node in tree.children.items():
            dataset = node.to_dataset()
            if name == "joint":
                dataset = dataset.drop_vars("aggregation_error_covariance").drop_dims("observation_cov")
            indexes = [dim for dim in json.loads(dataset.attrs["multiindex_dims_json"]) if dim in dataset.dims]
            dataset = decode_cf_multiindexes(dataset, indexes) if indexes else dataset
            datasets[name] = dataset.load()
        root_attrs = dict(tree.attrs)
    joint = datasets["joint"]
    observations = _stored_array(joint, "observed_concentration")
    with xr.open_dataset(manifest["low_rank"]["path"]) as source:
        factor = xr.DataArray(
            source.low_rank_factor.values, dims=("observation", "agg_rank"),
            coords={"observation": observations.observation},
        )
        diagonal = observations.copy(data=source.diagonal_residual_variance.values)
    prior = datasets["retained_prior"]
    prepared = Co2O2PreparedInputs(
        observations=observations,
        fixed_prior_contribution=_stored_array(joint, "fixed_prior_contribution"),
        co2_sensitivity=_stored_array(datasets["co2_sensitivity"], "co2_effective_sensitivity"),
        o2_sensitivity=_stored_array(datasets["o2_sensitivity"], "o2_effective_sensitivity"),
        o2_co2_flux_ratio=None,
        o2_co2_flux_ratio_unavailable_reason=root_attrs["o2_co2_flux_ratio_unavailable_reason"],
        aggregation_error=AggregationError(
            "low_rank", np.sum(factor.values**2, axis=1) + diagonal.values,
            factor=factor, diagonal_variance=diagonal,
        ),
        retained_prior=CorrelatedLognormalPrior(
            _stored_array(prior, "arithmetic_mean"), _stored_array(prior, "arithmetic_covariance"),
            covariance_dim=prior.attrs["covariance_dim"],
        ),
        provenance=json.loads(root_attrs["provenance_json"]),
        independent_error_sd=_stored_array(datasets["independent_error"], "independent_error_sd"),
        boundary_operands={channel: datasets[f"{channel}_boundary_operands"] for channel in ("co2", "o2")},
    )
    assert prepared.co2_sensitivity.shape == prepared.o2_sensitivity.shape == (5314, 334)
    assert factor.shape[1] == 256
    np.testing.assert_array_equal(prepared.independent_error_sd.values, 1e-6)
    return prepared


def compare_serialized(result_file: Path, source_file: Path, chains: int, draws: int) -> dict:
    """Compare public data, attributes, coordinates and hashes in bounded reads."""
    compared = {}
    with xr.open_datatree(source_file) as original, xr.open_datatree(result_file) as replay:
        for name in ("rhime_recipe", "rhime_priors", "rhime_variable_roles", "rhime_model_metadata"):
            if name.endswith(("priors", "roles", "metadata")):
                assert json.loads(original.attrs[name]) == json.loads(replay.attrs[name]), name
            else:
                assert original.attrs[name] == replay.attrs[name], name
        assert original.children.keys() == replay.children.keys()
        for group, node in replay.children.items():
            expected = restore_declared_multiindexes(original[group].to_dataset())
            actual = restore_declared_multiindexes(node.to_dataset())
            indexers = {dim: slice(0, n) for dim, n in [("chain", chains), ("draw", draws)] if dim in expected.dims}
            expected = expected.isel(indexers)
            assert set(actual.data_vars) == set(expected.data_vars), group
            xr.testing.assert_equal(actual.coords.to_dataset(), expected.coords.to_dataset())
            compared[group] = {}
            for name, variable in actual.data_vars.items():
                reference = expected[name]
                assert variable.dims == reference.dims and variable.dtype == reference.dtype, (group, name)
                assert variable.attrs == reference.attrs, (group, name, variable.attrs, reference.attrs)
                actual_digest, source_digest = hashlib.sha256(), hashlib.sha256()
                max_abs_error = max_relative_error = 0.0
                unequal = 0
                # Chain then draw yields canonical C-order sample bytes while
                # bounding temporary concentration payloads to 16 draws.
                chain_indices = range(chains) if "chain" in variable.dims else [None]
                draw_starts = range(0, draws, 16) if "draw" in variable.dims else [None]
                for chain in chain_indices:
                    for start in draw_starts:
                        selection = {}
                        if chain is not None:
                            selection["chain"] = chain
                        if start is not None:
                            selection["draw"] = slice(start, min(start + 16, draws))
                        left = np.asarray(variable.isel(selection).values)
                        right = np.asarray(reference.isel(selection).values)
                        actual_digest.update(np.ascontiguousarray(left).view(np.uint8).data)
                        source_digest.update(np.ascontiguousarray(right).view(np.uint8).data)
                        if left.dtype.kind in "fc":
                            finite = np.isfinite(left) & np.isfinite(right)
                            error = np.abs(left[finite] - right[finite])
                            max_abs_error = max(max_abs_error, float(np.max(error, initial=0)))
                            denominator = np.maximum(np.abs(right[finite]), np.finfo(float).tiny)
                            max_relative_error = max(max_relative_error, float(np.max(error / denominator, initial=0)))
                            unequal += int(np.count_nonzero(~((left == right) | (np.isnan(left) & np.isnan(right)))))
                            tolerance = {"rtol": 1e-10, "atol": 1e-8} if group == "log_likelihood" else {"rtol": 1e-12, "atol": 1e-9}
                            np.testing.assert_allclose(left, right, **tolerance, err_msg=f"{group}/{name}")
                        else:
                            np.testing.assert_array_equal(left, right, err_msg=f"{group}/{name}")
                compared[group][name] = {
                    "shape": list(variable.shape), "dtype": str(variable.dtype),
                    "sha256": actual_digest.hexdigest(), "source_sha256": source_digest.hexdigest(),
                    "max_absolute_error": max_abs_error, "max_relative_error": max_relative_error,
                    "unequal_elements": unequal,
                }
    return compared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strategy", choices=("full", "compact"))
    parser.add_argument("--arm", choices=("anchored_site_offsets", "free_site_offsets"), default="anchored_site_offsets")
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    args = parser.parse_args()
    assert 0 < args.draws <= 1000 and 0 < args.chains <= 4 and args.tune >= 0
    assert pytensor.config.floatX == "float64"
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = args.campaign / args.arm
    request_path = source / "request.json"
    request = json.loads(request_path.read_text())
    source_trace = source / "posterior.nc"
    package_root = Path(openghg_inversions.__file__).resolve().parents[1]
    result = {
        "status": "running", "strategy": args.strategy, "arm": args.arm,
        "script_sha256": file_sha256(Path(__file__)), "python": sys.version, "pymc": pm.__version__,
        "executable": sys.executable, "imported_package": str(Path(openghg_inversions.__file__).resolve()),
        "candidate_revision": (
            subprocess.check_output(["git", "-C", str(package_root), "rev-parse", "HEAD"], text=True).strip()
            if (package_root / ".git").exists() else None
        ),
        "candidate_file_sha256": {
            name: file_sha256(package_root / name) for name in (
                "openghg_inversions/inference/sampling.py",
                "openghg_inversions/rhime/co2/co2_o2_cached_sigma_runner.py",
                "openghg_inversions/rhime/co2/co2_o2_cached_sigma_model.py",
                "openghg_inversions/rhime/co2/co2_o2_model.py",
                "openghg_inversions/models/fixed_ou.py", "pyproject.toml", "uv.lock",
            )
        },
        "dependency_versions": {name: version(name) for name in ("numpy", "scipy", "xarray", "pymc", "pytensor", "numba")},
        "native_sample_return_source": inspect.getsourcefile(_sample_return),
        "native_sample_return_sha256": hashlib.sha256(inspect.getsource(_sample_return).encode()).hexdigest(),
        "source_request": str(request_path), "source_request_sha256": file_sha256(request_path),
        "source_revision": request["ogi_revision"], "source_trace": str(source_trace),
        "chains": args.chains, "draws": args.draws, "tune": args.tune,
        "warmup_states": "Repeated retained campaign states; not original warmup trajectories",
        "phases": {},
    }

    def save_result():
        result["process_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        (out / "profile.json").write_text(json.dumps(result, indent=2) + "\n")

    @contextmanager
    def measured(name):
        previous = result.get("active_phase")
        result["active_phase"] = name
        print(json.dumps({"phase_started": name}), flush=True)
        save_result()
        with phase(result["phases"], name):
            yield
        result["active_phase"] = previous
        save_result()

    def measured_call(name, function):
        def call(*positional, **keywords):
            if name in ("explicit_log_likelihood", "deterministic_reconstruction"):
                assert all(reference() is None for reference in backend_references)
                assert all(reference() is None for reference in payload_references)
            with measured(name):
                return function(*positional, **keywords)
        return call

    backend_references = []
    payload_references = []

    def replay_sample(*, model=None, draws, tune, chains, idata_kwargs, **kwargs):
        assert model is None  # RhimeSampler supplies the active model context.
        model = pm.modelcontext(None)
        assert draws == args.draws and tune == args.tune and chains == args.chains
        compact = kwargs.get("var_names") is not None
        assert compact == (args.strategy == "compact")
        with measured("saved_state_loading_and_transforms"):
            names = [rv.name for rv in model.free_RVs]
            with xr.open_dataset(source_trace, group="posterior") as dataset:
                physical = dataset[names].isel(chain=slice(chains), draw=slice(draws)).load()
            with xr.open_dataset(source_trace, group="sample_stats") as dataset:
                original_stats = dataset.isel(chain=slice(chains), draw=slice(draws)).load()
            transformed = []
            for value in model.value_vars:
                rv = model.values_to_rvs[value]
                transform = model.rvs_to_transforms.get(rv)
                transformed.append(transform.forward(rv, *rv.owner.inputs) if transform is not None else rv)
            forward = model.compile_fn(
                inputs=model.free_RVs, outs=transformed, point_fn=False, on_unused_input="ignore",
            )
            value_names = [value.name for value in model.value_vars]
            points = []
            for chain in range(chains):
                chain_points = []
                for draw in range(draws):
                    state = {name: physical[name].values[chain, draw] for name in names}
                    values = forward(**state)
                    chain_points.append(dict(zip(value_names, values, strict=True)))
                points.append(chain_points)
            result["free_rv_names"] = names
            result["saved_free_rv_bytes"] = sum(var.nbytes for var in physical.data_vars.values())
            del physical, forward
        with measured("backend_compile"):
            variables = model.replace_rvs_by_values(model.free_RVs) if compact else None
            template = NDArray(model=model, vars=variables)
        with measured("backend_recording"):
            backends = []
            for chain in range(chains):
                backend = NDArray(
                    model=model, vars=template.vars, fn=template.fn,
                    var_shapes=template.var_shapes, var_dtypes=template.var_dtypes,
                )
                backend.setup(
                    tune + draws, chain,
                    sampler_vars=[{name: str(var.dtype) for name, var in original_stats.data_vars.items()}],
                )
                for draw in range(tune + draws):
                    index = draw % draws if draw < tune else draw - tune
                    statistics = {name: var.values[chain, index] for name, var in original_stats.data_vars.items()}
                    backend.record(points[chain][index], sampler_stats=[statistics], in_warmup=draw < tune)
                backend.close()
                backends.append(backend)
                backend_references.append(weakref.ref(backend))
                payload_references.extend(weakref.ref(payload) for payload in backend.samples.values())
                print(json.dumps({"recorded_chain": chain, "rss_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}), flush=True)
            result["backend_bytes_by_variable"] = {
                name: sum(b.samples[name].nbytes for b in backends) for name in template.varnames
            }
            result["backend_bytes"] = sum(result["backend_bytes_by_variable"].values())
            result["backend_sampler_stats_bytes"] = sum(
                array.nbytes for backend in backends for group in backend._stats for array in group.values()
            )
        # This is PyMC's actual installed return implementation, including its
        # retained slicing, conversion and automatic likelihood computation.
        with measured("native_pymc_sample_return"):
            trace = _sample_return(
                run=None, traces=backends, tune=tune, t_sampling=0,
                discard_tuned_samples=True, compute_convergence_checks=False,
                return_inferencedata=True, keep_warning_stat=False,
                idata_kwargs=dict(idata_kwargs), model=model, quiet=True,
            )
        with measured("backend_release"):
            del backends, backend, template, points
            assert all(reference() is None for reference in backend_references)
            assert all(reference() is None for reference in payload_references)
            result["backend_objects_released"] = True
            result["backend_payloads_released"] = True
        # Original diagnostics are retained scientific/sampler evidence, not
        # synthetic events invented by this replay, which performs no updates.
        trace["sample_stats"] = original_stats
        return trace

    try:
        with measured("frozen_input_loading"):
            result["source_trace_sha256"] = file_sha256(source_trace)
            prepared = frozen_prepared(request)
            result["rows_per_species"] = prepared.co2_sensitivity.shape[0]
            result["aggregation_rank"] = prepared.aggregation_error.factor.shape[1]
            native = select_gathered_data_array(
                prepared.observations, key="o2", key_dim="species",
                ragged_dim="channel_observation", stack_dim="observation",
            ).rename(observation=prepared.o2_sensitivity.dims[0])
            operands = bind_boundary_operands(prepared.boundary_operands["o2"], native)
            mass = operands.H_bc_mass.transpose(native.dims[0], "bc_region")
            weights = xr.ones_like(mass.isel({native.dims[0]: 0}, drop=True)) / 4
            weights.attrs["units"] = "1"
            np.testing.assert_allclose(mass.sum("bc_region"), 1, rtol=0, atol=1e-12)
            assert [str(site) for site in pd.unique(native.site.values)] == request["site_order"]
        sampler = RhimeSampler(
            draws=args.draws, tune=args.tune, chains=args.chains, nuts_sampler="pymc", progressbar=False,
            sample_kwargs={"cores": 4, "blas_cores": 4, "random_seed": request["seed"], "compute_convergence_checks": False},
            sample_prior_predictive=False, sample_posterior_predictive=False,
        )

        def sample_replay(built, configured_sampler, **_):
            return sample_rhime_model(
                built, configured_sampler, recompute_deterministics=args.strategy == "compact",
            )

        with (
            patch.object(pm, "sample", replay_sample),
            patch.object(runner, "sample_rhime_model", sample_replay),
            patch.object(runner, "build_co2_o2_cached_sigma_model", measured_call("model_construction", runner.build_co2_o2_cached_sigma_model)),
            patch.object(runner, "_sampler_for_cached_graph", measured_call("cached_sampler_construction", runner._sampler_for_cached_graph)),
            patch.object(DataTreeConverter, "posterior_to_xarray", measured_call("xarray_conversion", DataTreeConverter.posterior_to_xarray)),
            patch.object(log_density, "compute_log_likelihood", measured_call("automatic_log_likelihood", log_density.compute_log_likelihood)),
            patch.object(pm, "compute_log_likelihood", measured_call("explicit_log_likelihood", pm.compute_log_likelihood)),
            patch.object(sampling, "_restore_deterministics", measured_call("deterministic_reconstruction", sampling._restore_deterministics)),
            patch.object(sampling, "restore_inferencedata_coords", measured_call("coordinate_restoration", sampling.restore_inferencedata_coords)),
            patch.object(runner, "annotate_co2_o2_trace", measured_call("scientific_annotation", runner.annotate_co2_o2_trace)),
        ):
            trace = runner.run_rhime_co2_o2_cached_sigma_from_prepared_inputs(
                prepared_inputs=prepared, use_bc={"co2": True, "o2": True}, sampler=sampler,
                sigma_target_accept=0.9, state_target_accept=0.95,
                bc_prior={"co2": request["co2_bc_prior"]},
                bc_mean_shift_prior_o2={"pdf": "normal", "mu": 0.0, "sigma": request["o2_common_mean_sd_ppm"]},
                bc_anomaly_scale_o2=request["o2_anomaly_pre_centering_sd_ppm"],
                boundary_correction_sensitivity_o2=mass, bc_centering_weights_o2=weights,
                tau_hours=request["tau_hours"], site_amplitude_prior=request["site_amplitude_prior"],
                offset_prior={"o2": {"pdf": "normal", "mu": 0.0, "sigma": request["o2_offset_sd_ppm"]}},
                offset_args={"o2": {"per_site": True, "drop_first": args.arm == "anchored_site_offsets"}},
            )
        result["posterior_bytes"] = sum(var.nbytes for var in trace.posterior.data_vars.values())
        with measured("serialization"):
            save_trace(trace, out / "posterior.nc")
            result["serialized_bytes"] = (out / "posterior.nc").stat().st_size
        with measured("bounded_source_comparison"):
            result["compared_variables"] = compare_serialized(
                out / "posterior.nc", source_trace, args.chains, args.draws,
            )
        with measured("selected_diagnostics"):
            names = [
                name for name in trace.posterior.data_vars
                if any(name == prefix or name.startswith(prefix + "_") for prefix in (
                    "flux_scaling", "co2_bc", "o2_bc", "ou_site_amplitude", "o2_offset_latent",
                ))
            ]
            diagnostics = posterior_summary(trace.posterior.to_dataset()[names])
            result["diagnostic_variable_names"] = names
            save_trace(xr.DataTree.from_dict({"diagnostics": diagnostics}), out / "diagnostics.nc")
        result["status"] = "completed"
    except BaseException:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        save_result()
    print(json.dumps({"status": result["status"], "profile": str(out / "profile.json")}), flush=True)


if __name__ == "__main__":
    main()
