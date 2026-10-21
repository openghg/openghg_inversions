"""Profile the fixed-OU amplitude conditional without changing its target.

Production-shaped inputs belong on Slurm. Start with ``--mode micro``; use
``--mode sample --variant baseline|current`` in separate processes afterwards.
The baseline substitutes the unchanged amplitude Op and general evaluator from
PR 926. Both variants retain that PR's cache refresh and the same PyMC NUTS.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from contextlib import contextmanager, ExitStack
from datetime import UTC, datetime
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
from pymc.blocking import DictToArrayBijection
from pymc.initial_point import make_initial_point_fn
from threadpoolctl import threadpool_info

import profile_ope152_fixed_ou as reference
from openghg_inversions.inference import cached_sigma as inference
from openghg_inversions.models import cached_sigma, fixed_ou
from openghg_inversions.rhime.co2 import Co2PreparedInputs, build_rhime_co2_cached_sigma


BASELINE = "9acd0c579b1ce61247f07892965fa1f44105e258"
PREPARED_SHA256 = "22313f482e0eb2143804f2a290cddc5f247ad293db11776977b75d631df06353"
SOURCE_SHA256 = "7ab7cf733eba7b12dd1fea7a5c8812236eda1d579486b03e5e2d27feab8866bf"
ROOT = Path(__file__).resolve().parents[1]


def old_class(module, name):
    """Compile one unchanged historical class with compatible current imports."""
    relative = Path(module.__file__).resolve().relative_to(ROOT)
    source = reference.git("show", f"{BASELINE}:{relative}")
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.ClassDef) and node.name == name)
    namespace = dict(vars(module))
    exec(compile(ast.Module(body=[node], type_ignores=[]), f"<{BASELINE}:{relative}>", "exec"), namespace)
    return namespace[name], source


def function_ast(source, class_name, method_name):
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == method_name)
    return ast.dump(method)


def build(prepared):
    return build_rhime_co2_cached_sigma(
        prepared_inputs=prepared, tau_hours=5.0,
        site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75},
        use_bc=True, bc_prior=reference.BC_PRIOR,
    )


def op_evaluate(op, sigma):
    outputs = [[None], [None]]
    op.perform(None, [sigma, op.residual.get_value(borrow=True)], outputs)
    return float(outputs[0][0]), np.asarray(outputs[1][0])


class Timings:
    """Collect nested timers; only sibling blocks may be added together."""

    def __init__(self):
        self.phase = "startup"
        self.total = defaultdict(lambda: defaultdict(float))
        self.row = None
        self.rows = []
        self.anchors = {}

    @contextmanager
    def measure(self, name):
        wall, cpu = perf_counter(), process_time()
        try:
            yield
        finally:
            elapsed = {f"{name}_seconds": perf_counter() - wall,
                       f"{name}_cpu_seconds": process_time() - cpu,
                       f"{name}_calls": 1}
            for key, value in elapsed.items():
                self.total[self.phase][key] += value
                if self.row is not None:
                    self.row[key] = self.row.get(key, 0) + value


@contextmanager
def sampling_timers(step, timings, tune, draws, checkpoint):
    """Wrap numerical boundaries and stock transitions, leaving their values intact."""
    sigma_step, state_step = step.methods
    op_type = type(sigma_step.likelihood_op)
    original_perform = op_type.perform
    original_install = op_type.install_residual
    original_refresh = cached_sigma.FixedOuCachedSigmaTarget.refresh
    original_sigma_step = inference.PymcCachedSigmaNutsStep.step
    original_nuts_step = pm.NUTS.step
    anchor_indices = {0, max(0, tune - 1), tune, tune + draws // 2, tune + draws - 1}

    def perform(self, *args, **kwargs):
        with timings.measure("amplitude_likelihood"):
            return original_perform(self, *args, **kwargs)

    def install(self, residual):
        if timings.row is not None and timings.row["sweep"] in anchor_indices:
            timings.anchors[f"residual_{timings.row['sweep']}"] = np.asarray(residual).copy()
        with timings.measure("residual_install"):
            return original_install(self, residual)

    def refresh(self, *args, **kwargs):
        with timings.measure("cache_refresh"):
            return original_refresh(self, *args, **kwargs)

    def sigma_transition(self, point):
        assert self is sigma_step
        timings.phase = "warmup" if self.tune else "retained"
        timings.row = {"sweep": len(timings.rows), "phase": timings.phase}
        if timings.row["sweep"] in anchor_indices:
            timings.anchors[f"eta_{timings.row['sweep']}"] = np.asarray(point[self.sigma_value_name]).copy()
        with timings.measure("sigma_wrapper"):
            result = original_sigma_step(self, point)
        return result

    def nuts_transition(self, point):
        name = "amplitude_trajectory" if self is sigma_step.nuts_step else "state_trajectory"
        assert self is sigma_step.nuts_step or self is state_step
        with timings.measure(name):
            result = original_nuts_step(self, point)
        if self is state_step:
            assert timings.row is not None
            timings.rows.append(timings.row)
            timings.row = None
            if len(timings.rows) == 1 or len(timings.rows) % 25 == 0:
                checkpoint()
        return result

    with ExitStack() as stack:
        for cls, name, replacement in (
            (op_type, "perform", perform), (op_type, "install_residual", install),
            (cached_sigma.FixedOuCachedSigmaTarget, "refresh", refresh),
            (inference.PymcCachedSigmaNutsStep, "step", sigma_transition),
            (pm.NUTS, "step", nuts_transition),
        ):
            stack.enter_context(patch.object(cls, name, replacement))
        yield


def micro(cached, baseline_op, baseline_evaluate, repeats, anchors):
    target = cached.target
    prepared = target.prepared
    rng = np.random.default_rng(152)
    perturbation = rng.normal(scale=0.01, size=target.n_state)
    states = [np.ones(target.n_state), np.ones(target.n_state) + perturbation,
              np.ones(target.n_state) - perturbation]
    cases = [(f"prior-state-{index}",
              target.observations - target.fixed_contribution - target.design @ state,
              [np.full(target.n_group, amplitude) for amplitude in (0.3, 0.75, 1.2)])
             for index, state in enumerate(states)]
    if anchors is not None:
        with np.load(anchors) as selected:
            residuals, amplitudes = selected["residuals"], selected["amplitudes"]
            assert residuals.shape == (len(amplitudes), target.n_obs)
            assert amplitudes.shape == (len(residuals), target.n_group)
            assert np.isfinite(residuals).all() and np.isfinite(amplitudes).all()
            np.testing.assert_allclose(
                target.fixed_contribution + selected["coefficients"] @ target.design.T,
                selected["modelled_means"], rtol=1e-12, atol=1e-10,
            )
            np.testing.assert_allclose(target.observations - selected["modelled_means"],
                                       residuals, rtol=1e-12, atol=1e-10)
            cases.extend((f"posterior-{index}", residual, [amplitude])
                         for index, (residual, amplitude) in enumerate(zip(residuals, amplitudes, strict=True)))
    current_op = inference._PytensorSigmaLikelihoodOp
    result = {"calls": [], "checks": [], "residual_installs": []}
    for state_index, residual, amplitudes in cases:
        before = residual.copy()
        ops = {"baseline": baseline_op(target), "current": current_op(target)}
        for name, op in ops.items():
            start = perf_counter()
            op.install_residual(residual)
            result["residual_installs"].append(
                {"state": state_index, "variant": name, "seconds": perf_counter() - start},
            )
        for sigma in amplitudes:
            amplitude = sigma.tolist()
            expected = baseline_evaluate(prepared, residual, sigma)
            general = prepared.evaluate(residual, sigma)
            np.testing.assert_allclose(general.log_likelihood, expected.log_likelihood, rtol=1e-12, atol=1e-9)
            np.testing.assert_allclose(general.gradient_site_amplitude,
                                       expected.gradient_site_amplitude, rtol=1e-12, atol=1e-9)
            np.testing.assert_allclose(general.gradient_residual,
                                       expected.gradient_residual, rtol=1e-12, atol=1e-9)
            actual = op_evaluate(ops["current"], sigma)
            np.testing.assert_allclose(actual[0], expected.log_likelihood, rtol=1e-12, atol=1e-9)
            np.testing.assert_allclose(actual[1], expected.gradient_site_amplitude, rtol=1e-12, atol=1e-9)
            result["checks"].append({"state": state_index, "amplitude": amplitude,
                                     "logp_absolute_error": abs(actual[0] - expected.log_likelihood),
                                     "gradient_max_absolute_error": float(np.max(
                                         np.abs(actual[1] - expected.gradient_site_amplitude)))})
            for repeat in range(repeats + 1):
                order = ("baseline", "current") if repeat % 2 == 0 else ("current", "baseline")
                for position, name in enumerate(order):
                    with ExitStack() as stack:
                        if name == "baseline":
                            stack.enter_context(patch.object(fixed_ou.FixedOuLowRank, "evaluate", baseline_evaluate))
                        cpu, wall = process_time(), perf_counter()
                        value, gradient = op_evaluate(ops[name], sigma)
                        seconds, cpu_seconds = perf_counter() - wall, process_time() - cpu
                    np.testing.assert_allclose(value, expected.log_likelihood, rtol=1e-12, atol=1e-9)
                    np.testing.assert_allclose(gradient, expected.gradient_site_amplitude, rtol=1e-12, atol=1e-9)
                    if repeat:
                        result["calls"].append({"state": state_index, "amplitude": amplitude,
                                                "repeat": repeat, "position": position, "variant": name,
                                                "seconds": seconds, "cpu_seconds": cpu_seconds})
        np.testing.assert_array_equal(residual, before)
    return result


def sample(prepared, args, baseline_op, baseline_evaluate, save):
    chains = []
    for chain in range(args.chains):
        seed = args.seed + chain
        timings = Timings()
        record = {"chain": chain, "seed": seed, "status": "running",
                  "phase_totals": timings.total, "sweeps": timings.rows}
        chains.append(record)
        save(chains)
        with ExitStack() as stack:
            if args.variant == "baseline":
                stack.enter_context(patch.object(inference, "_PytensorSigmaLikelihoodOp", baseline_op))
                stack.enter_context(patch.object(fixed_ou.FixedOuLowRank, "evaluate", baseline_evaluate))
            with timings.measure("model_build_including_initial_refresh"):
                cached = build(prepared)
            initial = make_initial_point_fn(
                model=cached.model, overrides=cached.initial_values, return_transformed=True,
            )(seed)
            with timings.measure("sampler_construction"):
                step = inference.make_cached_sigma_compound_step(
                    model=cached.model, sigma=cached.amplitude, states=cached.states,
                    modelled_mean=cached.modelled_mean, target=cached.target,
                    shared_cache=cached.shared_cache, initial_cache=cached.initial_cache,
                    cached_state_logp=cached.cached_state_logp, initial_point=initial,
                    sigma_target_accept=0.9, state_target_accept=0.95,
                    rng=np.random.default_rng(seed),
                )
            sigma_step, state_step = step.methods
            assert sigma_step.nuts_step.max_treedepth == state_step.max_treedepth == 10
            assert sigma_step.nuts_step.early_max_treedepth == state_step.early_max_treedepth == 8
            with sampling_timers(step, timings, args.tune, args.draws, lambda: save(chains)), cached.model:
                # Trigger lazy graph compilation without advancing either Markov chain.
                with timings.measure("initial_density_evaluations_including_lazy_compilation"):
                    _, residual = sigma_step._point_values(initial)
                    sigma_step.likelihood_op.install_residual(residual)
                    for nuts in (sigma_step.nuts_step, state_step):
                        nuts._logp_dlogp_func.set_extra_values(initial)
                        value, gradient = nuts._logp_dlogp_func(DictToArrayBijection.map(
                            {var.name: initial[var.name] for var in nuts.vars},
                        ))
                        assert np.isfinite(value) and np.isfinite(gradient).all()
                started = perf_counter()
                variables = list(dict.fromkeys(
                    [var.name for var in (*cached.states, cached.amplitude)]
                    + [name for name in ("flux_scaling", "bc") if name in cached.model.named_vars],
                ))
                trace = pm.sample(
                    draws=args.draws, tune=args.tune, chains=1, cores=1, step=step,
                    initvals=cached.initial_values, random_seed=seed, progressbar=False,
                    discard_tuned_samples=False, compute_convergence_checks=False,
                    var_names=variables,
                    idata_kwargs={"log_likelihood": False},
                )
            sampling_seconds = perf_counter() - started
        assert len(timings.rows) == args.tune + args.draws
        assert sum(row["phase"] == "warmup" for row in timings.rows) == args.tune
        for phase, group in (("warmup", "warmup_sample_stats"), ("retained", "sample_stats")):
            assert int(np.asarray(trace[group]["cache_refreshes"]).sum()) == int(
                timings.total[phase].get("cache_refresh_calls", 0),
            )
        trace.to_netcdf(args.out / f"chain-{chain}.nc")
        np.savez_compressed(args.out / f"conditioning-anchors-{chain}.npz", **timings.anchors)
        record.update(status="passed", sampling_wall_seconds=sampling_seconds,
                      sigma_value_name=sigma_step.sigma_value_name,
                      state_variables=[var.name for var in cached.states])
        save(chains)
    return chains


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--expected-sha256", default=PREPARED_SHA256)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("micro", "sample"), default="micro")
    parser.add_argument("--variant", choices=("baseline", "current"), default="current")
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--tune", type=int, default=300)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=152202610)
    parser.add_argument("--anchors", type=Path, help="Posterior residuals/amplitudes NPZ with a JSON provenance sidecar")
    args = parser.parse_args()
    if min(args.repeats, args.tune, args.draws, args.chains) <= 0:
        parser.error("Repeat, tuning, draw and chain counts must be positive.")
    args.out.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    baseline_op, baseline_inference_source = old_class(inference, "_PytensorSigmaLikelihoodOp")
    baseline_evaluate, baseline_fixed_source = reference.baseline_method(
        BASELINE, fixed_ou, "FixedOuLowRank", "evaluate",
    )
    cached_source = Path(cached_sigma.__file__).read_text()
    baseline_cached_source = reference.git("show", f"{BASELINE}:openghg_inversions/models/cached_sigma.py")
    assert function_ast(cached_source, "FixedOuCachedSigmaTarget", "refresh") == function_ast(
        baseline_cached_source, "FixedOuCachedSigmaTarget", "refresh",
    ), "The cache-refresh implementation must match the reference."
    assert ast.dump(next(node for node in ast.parse(cached_source).body
                         if isinstance(node, ast.FunctionDef) and node.name == "_sum_squares_gram")) == ast.dump(
        next(node for node in ast.parse(baseline_cached_source).body
             if isinstance(node, ast.FunctionDef) and node.name == "_sum_squares_gram"),
    ), "The guarded Gram implementation must match the reference."
    for method in ("_solve_transformed", "_mode_weights_and_logdet"):
        assert function_ast(Path(fixed_ou.__file__).read_text(), "FixedOuLowRank", method) == function_ast(
            baseline_fixed_source, "FixedOuLowRank", method,
        ), f"The shared covariance helper {method} must match the reference."
    manifest = {
        "status": "running", "started": datetime.now(UTC).isoformat(),
        "revision": reference.git("rev-parse", "HEAD").strip(), "baseline_revision": BASELINE,
        "mode": args.mode, "variant": args.variant,
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "prepared_sha256": reference.sha256(args.prepared), "reference_original_dense_sha256": SOURCE_SHA256,
        "prepared_provenance_note": "Existing rank-256 plus diagonal-tail public artifact; no approximation is recomputed.",
        "python": sys.version, "packages": {name: version(name) for name in ("numpy", "scipy", "pymc", "pytensor")},
        "threadpools": threadpool_info(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "environment": {key: os.environ.get(key) for key in (
            "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_CPU_BIND", "SLURM_CPU_BIND_LIST",
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTENSOR_FLAGS")},
        "scientific_target": {"tau_hours": 5.0, "amplitude_prior": {"pdf": "halfnormal", "sigma": 0.75},
                              "bc_prior": reference.BC_PRIOR, "sigma_target_accept": 0.9, "state_target_accept": 0.95,
                              "max_treedepth": 10, "early_max_treedepth": 8},
        "timing_note": "Nested: likelihood is inside amplitude trajectory; trajectory, residual install and cache refresh are inside sigma wrapper. Only sigma wrapper and state trajectory are disjoint. Startup and pm.sample wall time are reported separately; instrumentation and serialization overhead must not be called kernel time.",
        "source_sha256": {},
    }
    sources = {"baseline_inference_cached_sigma.py": baseline_inference_source,
               "baseline_fixed_ou.py": baseline_fixed_source, "baseline_cached_sigma.py": baseline_cached_source,
               "current_cached_sigma.py": cached_source,
               "current_inference_cached_sigma.py": Path(inference.__file__).read_text(),
               "current_fixed_ou.py": Path(fixed_ou.__file__).read_text(),
               "profile_ope152_fixed_ou.py": Path(reference.__file__).read_text(),
               "profile_ope152_amplitude.py": Path(__file__).read_text()}
    for name, source in sources.items():
        (args.out / name).write_text(source)
        manifest["source_sha256"][name] = hashlib.sha256(source.encode()).hexdigest()
    (args.out / "working-tree.patch").write_text(reference.git("diff", "HEAD", "--binary"))

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
        assert manifest["prepared_sha256"] == args.expected_sha256
        prepared = Co2PreparedInputs.load(args.prepared)
        assert prepared.aggregation_error_mode == "low_rank"
        manifest["prepared_provenance"] = dict(prepared.provenance)
        manifest["original_dense_sha256"] = prepared.provenance.get("benchmark_source_sha256")
        if args.expected_sha256 == PREPARED_SHA256:
            assert manifest["original_dense_sha256"] == SOURCE_SHA256
        if args.anchors is not None:
            manifest["anchors"] = {"path": str(args.anchors.resolve()),
                                   "sha256": reference.sha256(args.anchors),
                                   "provenance": json.loads(args.anchors.with_suffix(".json").read_text())}
        if args.mode == "micro":
            start = perf_counter()
            cached = build(prepared)
            manifest["startup_build_seconds"] = perf_counter() - start
            manifest["shapes"] = {"observations": cached.target.n_obs, "sites": cached.target.n_group,
                                  "rank": cached.target.prepared.rank, "state": cached.target.n_state}
            manifest["micro"] = micro(cached, baseline_op, baseline_evaluate, args.repeats, args.anchors)
        else:
            sample(prepared, args, baseline_op, baseline_evaluate, save)
        assert reference.sha256(args.prepared) == manifest["prepared_sha256"]
        for path, name in ((fixed_ou.__file__, "current_fixed_ou.py"),
                           (cached_sigma.__file__, "current_cached_sigma.py"),
                           (inference.__file__, "current_inference_cached_sigma.py")):
            assert reference.sha256(Path(path)) == manifest["source_sha256"][name]
        manifest["status"] = "passed"
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
