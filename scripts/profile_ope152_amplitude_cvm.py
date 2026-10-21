"""Compare frozen optimized NUMBA amplitudes with the private CVM constructor.

Production inputs require Slurm. Earlier experiment scripts remain immutable.
"""

# ruff: noqa: E402 -- reuse the frozen repository profiling helpers.

from collections import Counter
from contextlib import nullcontext
from datetime import UTC, datetime
import argparse
import ast
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time
from unittest.mock import patch

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pymc.blocking import DictToArrayBijection
from pymc.initial_point import make_initial_point_fn
from pytensor.compile.mode import Mode
from scipy.sparse import issparse
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import profile_ope152_amplitude as prior
from openghg_inversions.inference import cached_sigma as inference
from openghg_inversions.rhime.co2 import Co2PreparedInputs

BASELINE = "2a440c9827aa2906c79b3b1056fe0952f27e15be"
ANCHOR_SHA256 = "08cddffb167f48e478439ea3036c5eb82da1f6e1010fcfee0820a41ba44cf1c1"
SOURCE_FILES = (
    "openghg_inversions/models/fixed_ou.py", "openghg_inversions/models/cached_sigma.py",
    "openghg_inversions/inference/cached_sigma.py", "scripts/profile_ope152_amplitude.py",
    "scripts/profile_ope152_fixed_ou.py",
)


def constructor_and_sources(out):
    baseline, baseline_source = prior.reference.baseline_method(
        BASELINE, inference, "PymcCachedSigmaNutsStep", "__init__",
    )
    assert not baseline.__code__.co_freevars
    old_tree = ast.parse(baseline_source)
    cls = next(node for node in old_tree.body if isinstance(node, ast.ClassDef)
               and node.name == "PymcCachedSigmaNutsStep")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    assert not any(isinstance(node, ast.Name) and node.id in ("super", "__class__")
                   for node in ast.walk(init))
    sources = {}
    for name in SOURCE_FILES:
        current = (ROOT / name).read_text()
        frozen = prior.reference.git("show", f"{BASELINE}:{name}")
        if name != "openghg_inversions/inference/cached_sigma.py":
            assert current == frozen, f"Changed shared numerical source: {name}"
        else:
            # Only the private amplitude compile mode and its import may differ.
            tree = ast.parse(current)
            imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
                       and node.module == "pytensor.compile.mode"
                       and [alias.name for alias in node.names] == ["Mode"]]
            assert len(imports) == 1
            tree.body.remove(imports[0])
            changed = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "NUTS":
                    for keyword in list(node.keywords):
                        if keyword.arg == "compile_kwargs":
                            expected = ast.parse("{'mode': Mode(linker='cvm', optimizer='fast_run')}", mode="eval").body
                            assert ast.dump(keyword.value) == ast.dump(expected)
                            node.keywords.remove(keyword)
                            changed += 1
            assert changed == 1 and ast.dump(tree) == ast.dump(old_tree)
        sources[name] = current
    sources["baseline/inference_cached_sigma.py"] = baseline_source
    sources["harness/profile_ope152_amplitude_cvm.py"] = Path(__file__).read_text()
    for name, source in sources.items():
        destination = out / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source)
    return baseline, {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}


def constructor_context(variant, baseline):
    return (patch.object(inference.PymcCachedSigmaNutsStep, "__init__", baseline)
            if variant == "baseline" else nullcontext())


def graph_info(step, variant):
    fn = step.nuts_step._logp_dlogp_func._pytensor_function
    linker = type(fn.maker.mode.linker).__name__
    assert ("Numba" in linker) == (variant == "baseline"), (variant, linker)
    nodes = fn.maker.fgraph.toposort()
    ops = [node.op for node in nodes if isinstance(node.op, inference._PytensorSigmaLikelihoodOp)]
    assert len(ops) == 1 and ops[0] is step.likelihood_op
    return {"linker": linker, "nodes": dict(Counter(type(node.op).__name__ for node in nodes))}


def ownership(target, residual, amplitude, eta, mode_residual):
    arrays = {"residual": residual, "amplitude": amplitude, "eta": eta, "mode_residual": mode_residual}
    for owner_name, owner in (("target", target), ("prepared", target.prepared)):
        for name, value in vars(owner).items():
            if isinstance(value, np.ndarray):
                arrays[f"{owner_name}.{name}"] = value
            elif issparse(value):
                for part in ("data", "indices", "indptr"):
                    arrays[f"{owner_name}.{name}.{part}"] = getattr(value, part)
    return {name: hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
            for name, value in arrays.items()}


def micro(prepared, args, baseline):
    started = perf_counter()
    cached = prior.build(prepared)
    initial = make_initial_point_fn(model=cached.model, overrides=cached.initial_values,
                                    return_transformed=True)(args.seed)
    result = {"model_build_seconds": perf_counter() - started, "graphs": {}, "checks": [], "calls": []}
    steps, functions = {}, {}
    for variant in ("baseline", "current"):
        started = perf_counter()
        with constructor_context(variant, baseline):
            step = inference.PymcCachedSigmaNutsStep(
                cached.amplitude, target=cached.target, shared_cache=cached.shared_cache,
                initial_cache=cached.initial_cache, modelled_mean=cached.modelled_mean,
                model=cached.model, initial_point=initial, target_accept=0.9,
                rng=np.random.default_rng(args.seed),
            )
        steps[variant] = step
        functions[variant] = step.nuts_step._logp_dlogp_func._pytensor_function
        result["graphs"][variant] = {**graph_info(step, variant), "construction_seconds": perf_counter() - started}
    if args.anchors:
        with np.load(args.anchors) as anchors:
            residuals, amplitudes = anchors["residuals"], anchors["amplitudes"]
            np.testing.assert_allclose(cached.target.fixed_contribution + anchors["coefficients"] @ cached.target.design.T,
                                       anchors["modelled_means"], rtol=1e-12, atol=1e-10)
            np.testing.assert_allclose(cached.target.observations - anchors["modelled_means"], residuals,
                                       rtol=1e-12, atol=1e-10)
    else:
        assert args.expected_sha256 != prior.PREPARED_SHA256
        residuals = np.array([steps["current"]._point_values(initial)[1]] * 3)
        amplitudes = np.array([np.full(cached.target.n_group, value) for value in (0.3, 0.75, 1.2)])
    assert residuals.shape == (len(amplitudes), cached.target.n_obs)
    assert amplitudes.shape == (len(residuals), cached.target.n_group)
    prior_graph = cached.model.logp(vars=[cached.amplitude], jacobian=True)
    eta_var = steps["current"].vars[0]
    prior_fn = pytensor.function([eta_var], [prior_graph, pt.grad(prior_graph, eta_var)],
                                mode=Mode(linker="cvm", optimizer="fast_run"))
    for index, (residual, amplitude) in enumerate(zip(residuals, amplitudes, strict=True)):
        eta = np.log(amplitude).astype(eta_var.dtype)
        public = cached.target.prepared.evaluate(residual, amplitude)
        prior_value, prior_gradient = prior_fn(eta)
        oracle = (public.log_likelihood + prior_value,
                  amplitude * public.gradient_site_amplitude + prior_gradient)
        before = {}
        for variant, step in steps.items():
            step.likelihood_op.install_residual(residual)
            before[variant] = ownership(cached.target, residual, amplitude, eta,
                                        step.likelihood_op.residual.get_value(borrow=True))
        values = {}
        for repeat in range(args.repeats + 1):
            order = ("baseline", "current") if repeat % 2 == 0 else ("current", "baseline")
            for position, variant in enumerate(order):
                started, cpu = perf_counter(), process_time()
                value, gradient = functions[variant](eta)
                elapsed, cpu_elapsed = perf_counter() - started, process_time() - cpu
                np.testing.assert_allclose(value, oracle[0], rtol=1e-11, atol=1e-9)
                np.testing.assert_allclose(gradient, oracle[1], rtol=1e-10, atol=1e-8)
                values[variant] = (float(value), np.asarray(gradient).copy())
                if repeat:
                    result["calls"].append({"anchor": index, "repeat": repeat, "position": position,
                                             "variant": variant, "seconds": elapsed, "cpu_seconds": cpu_elapsed})
                else:
                    result["graphs"][variant].setdefault("warm_evaluations_seconds", []).append(elapsed)
            np.testing.assert_allclose(values["baseline"][0], values["current"][0], rtol=1e-12, atol=1e-10)
            np.testing.assert_allclose(values["baseline"][1], values["current"][1], rtol=1e-12, atol=1e-10)
        result["checks"].append({"anchor": index, "amplitude": amplitude.tolist(),
                                  "logp_absolute_error": abs(values["baseline"][0] - values["current"][0]),
                                  "gradient_max_absolute_error": float(np.max(np.abs(values["baseline"][1] - values["current"][1])))})
        for variant, step in steps.items():
            assert before[variant] == ownership(cached.target, residual, amplitude, eta,
                                                step.likelihood_op.residual.get_value(borrow=True))
    return result


def sample(prepared, args, baseline, save):
    chains = []
    for chain in range(args.chains):
        seed, timings = args.seed + chain, prior.Timings()
        record = {"chain": chain, "seed": seed, "status": "running",
                  "phase_totals": timings.total, "sweeps": timings.rows}
        chains.append(record)
        save(chains)
        with constructor_context(args.variant, baseline):
            with timings.measure("model_build_including_initial_refresh"):
                cached = prior.build(prepared)
            initial = make_initial_point_fn(model=cached.model, overrides=cached.initial_values,
                                            return_transformed=True)(seed)
            original_perform = inference._PytensorSigmaLikelihoodOp.perform

            def timed_perform(op, *inputs, **kwargs):
                with timings.measure("amplitude_likelihood"):
                    return original_perform(op, *inputs, **kwargs)

            # CVM binds perform at compilation; NUMBA resolves it at runtime.
            # Compile baseline unpatched to preserve its serialized target size.
            timer_context = (patch.object(inference._PytensorSigmaLikelihoodOp, "perform", timed_perform)
                             if args.variant == "current" else nullcontext())
            with timings.measure("sampler_construction"), timer_context:
                step = inference.make_cached_sigma_compound_step(
                    model=cached.model, sigma=cached.amplitude, states=cached.states,
                    modelled_mean=cached.modelled_mean, target=cached.target,
                    shared_cache=cached.shared_cache, initial_cache=cached.initial_cache,
                    cached_state_logp=cached.cached_state_logp, initial_point=initial,
                    sigma_target_accept=0.9, state_target_accept=0.95, rng=np.random.default_rng(seed),
                )
            assert inference._PytensorSigmaLikelihoodOp.perform is original_perform
            sigma_step, state_step = step.methods
            record["amplitude_graph"] = graph_info(sigma_step, args.variant)
            record["state_linker"] = type(state_step._logp_dlogp_func._pytensor_function.maker.mode.linker).__name__
            assert "Numba" in record["state_linker"]
            assert sigma_step.nuts_step.max_treedepth == state_step.max_treedepth == 10
            assert sigma_step.nuts_step.early_max_treedepth == state_step.early_max_treedepth == 8
            # Finish lazy NUMBA compilation with stock perform before patching.
            # Both arms evaluate the same initial densities without advancing RNGs.
            with timings.measure("initial_density_evaluations_including_lazy_compilation"):
                _, residual = sigma_step._point_values(initial)
                sigma_step.likelihood_op.install_residual(residual)
                for nuts in (sigma_step.nuts_step, state_step):
                    nuts._logp_dlogp_func.set_extra_values(initial)
                    value, gradient = nuts._logp_dlogp_func(DictToArrayBijection.map(
                        {var.name: initial[var.name] for var in nuts.vars},
                    ))
                    assert np.isfinite(value) and np.isfinite(gradient).all()
            with prior.sampling_timers(step, timings, args.tune, args.draws, lambda: save(chains)), cached.model:
                started = perf_counter()
                variables = list(dict.fromkeys(
                    [var.name for var in (*cached.states, cached.amplitude)]
                    + [name for name in ("flux_scaling", "bc") if name in cached.model.named_vars],
                ))
                trace = pm.sample(
                    draws=args.draws, tune=args.tune, chains=1, cores=1, step=step,
                    initvals=cached.initial_values, random_seed=seed, progressbar=False,
                    discard_tuned_samples=False, compute_convergence_checks=False,
                    var_names=variables, idata_kwargs={"log_likelihood": False},
                )
            sampling_seconds = perf_counter() - started
        assert len(timings.rows) == args.tune + args.draws
        for phase, group in (("warmup", "warmup_sample_stats"), ("retained", "sample_stats")):
            rows = [row for row in timings.rows if row["phase"] == phase]
            assert len(rows) == trace[group].sizes["draw"]
            np.testing.assert_array_equal([row["amplitude_likelihood_calls"] for row in rows],
                                          np.asarray(trace[group].sigma_nuts_tree_steps).ravel() + 1)
            np.testing.assert_array_equal([row.get("cache_refresh_calls", 0) for row in rows],
                                          np.asarray(trace[group].cache_refreshes).ravel())
            for row in rows:
                assert all(row[name + "_calls"] == 1 for name in (
                    "amplitude_trajectory", "state_trajectory", "sigma_wrapper", "residual_install"))
        trace.to_netcdf(args.out / f"chain-{chain}.nc")
        np.savez_compressed(args.out / f"conditioning-anchors-{chain}.npz", **timings.anchors)
        record.update(status="passed", sampling_wall_seconds=sampling_seconds,
                      sigma_value_name=sigma_step.sigma_value_name,
                      state_variables=[var.name for var in cached.states])
        save(chains)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--expected-sha256", default=prior.PREPARED_SHA256)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("micro", "sample"), default="micro")
    parser.add_argument("--variant", choices=("baseline", "current"), default="current")
    parser.add_argument("--anchors", type=Path)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--tune", type=int, default=300)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=152202610)
    args = parser.parse_args()
    if min(args.repeats, args.tune, args.draws, args.chains) <= 0:
        parser.error("Repeat, tuning, draw and chain counts must be positive.")
    if args.expected_sha256 == prior.PREPARED_SHA256 and not os.environ.get("SLURM_JOB_ID"):
        parser.error("Production-shaped work requires Slurm.")
    args.out.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    baseline, sources = constructor_and_sources(args.out)
    manifest = {
        "status": "running", "started": datetime.now(UTC).isoformat(),
        "revision": prior.reference.git("rev-parse", "HEAD").strip(), "baseline_revision": BASELINE,
        "mode": args.mode, "variant": args.variant,
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "prepared_sha256": prior.reference.sha256(args.prepared), "source_sha256": sources,
        "python": sys.version, "packages": {name: version(name) for name in ("numpy", "scipy", "pymc", "pytensor", "numba")},
        "threadpools": threadpool_info(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "environment": {key: os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTENSOR_FLAGS")},
        "scientific_target": {"tau_hours": 5.0, "amplitude_prior": {"pdf": "halfnormal", "sigma": 0.75},
                              "bc_prior": prior.reference.BC_PRIOR, "sigma_target_accept": 0.9, "state_target_accept": 0.95,
                              "max_treedepth": 10, "early_max_treedepth": 8},
        "timing_note": "Nested likelihood/trajectory/wrapper timers are not additive. Only sigma wrapper and state trajectory are disjoint. CVM binds its perform timer during construction; NUMBA compiles and warms stock perform before installing the runtime timer. Both arms warm the same densities; only CVM startup has a nested inner counter, so startup inner counts are not comparable. No unpickle/callback counter is installed. Startup, warmup, retained transitions and pm.sample wall are separate; serialization is not separately timed.",
    }
    (args.out / "working-tree.patch").write_text(prior.reference.git("diff", "HEAD", "--binary"))

    def save(chains=None):
        if chains is not None:
            manifest["chains"] = chains
        manifest["elapsed_seconds"] = perf_counter() - started
        manifest["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temporary = args.out / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.out / "manifest.json")

    try:
        save()
        assert all(pool["num_threads"] == 1 for pool in threadpool_info() if pool["user_api"] == "blas")
        assert manifest["prepared_sha256"] == args.expected_sha256
        prepared = Co2PreparedInputs.load(args.prepared)
        assert prepared.aggregation_error_mode == "low_rank"
        manifest["prepared_provenance"] = dict(prepared.provenance)
        manifest["original_dense_sha256"] = prepared.provenance.get("benchmark_source_sha256")
        if args.expected_sha256 == prior.PREPARED_SHA256:
            assert manifest["original_dense_sha256"] == prior.SOURCE_SHA256
        if args.anchors:
            assert prior.reference.sha256(args.anchors) == ANCHOR_SHA256
            manifest["anchors"] = {"path": str(args.anchors.resolve()), "sha256": ANCHOR_SHA256,
                                   "provenance": json.loads(args.anchors.with_suffix(".json").read_text())}
        if args.mode == "micro":
            manifest["micro"] = micro(prepared, args, baseline)
        else:
            sample(prepared, args, baseline, save)
        assert prior.reference.sha256(args.prepared) == manifest["prepared_sha256"]
        if args.anchors:
            assert prior.reference.sha256(args.anchors) == ANCHOR_SHA256
        for name in SOURCE_FILES:
            assert prior.reference.sha256(ROOT / name) == sources[name]
        assert prior.reference.sha256(Path(__file__)) == sources["harness/profile_ope152_amplitude_cvm.py"]
        manifest["status"] = "passed"
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
