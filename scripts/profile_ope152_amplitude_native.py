"""Bounded three-arm comparison of the experimental native amplitude kernel.

Frozen CVM, shared Python CVM and native NUMBA use the same public input and
conditional target. Production-shaped inputs require Slurm; no sampling runs.
"""

# ruff: noqa: E402 -- import repository profiling helpers without installation.

from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime
import argparse
import ast
import hashlib
from importlib.metadata import version
from itertools import permutations
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time
from unittest.mock import patch
import warnings

import numpy as np
import pytensor
import pytensor.tensor as pt
from numba.core import serialize
from pymc.initial_point import make_initial_point_fn
from pytensor.compile.mode import Mode
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import profile_ope152_amplitude as prior
from profile_ope152_amplitude_cvm import ANCHOR_SHA256, ownership
from openghg_inversions.inference import cached_sigma as inference
from openghg_inversions.models import fixed_ou
from openghg_inversions.rhime.co2 import Co2PreparedInputs

BASELINE = "e2bb62465f43b287b08e394bdc552b8e4aa599cc"
VARIANTS = ("frozen_cvm", "shared_cvm", "native_numba")
FILES = (
    "openghg_inversions/models/fixed_ou.py", "openghg_inversions/models/cached_sigma.py",
    "openghg_inversions/inference/cached_sigma.py", "scripts/profile_ope152_amplitude.py",
    "scripts/profile_ope152_amplitude_cvm.py", "scripts/profile_ope152_fixed_ou.py",
    "scripts/ope152_native_amplitude.py", "scripts/profile_ope152_amplitude_native.py",
)


def snapshot(out):
    frozen, source = prior.reference.baseline_method(BASELINE, fixed_ou, "FixedOuLowRank", "_evaluate")
    old_nodes = {node.name: node for node in ast.parse(source).body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    current_nodes = {node.name: node for node in ast.parse(Path(fixed_ou.__file__).read_text()).body
                     if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    for name, old in old_nodes.items():
        current = current_nodes[name]
        if name == "FixedOuLowRank":
            for node in (old, current):
                node.body = [item for item in node.body if not (
                    isinstance(item, ast.FunctionDef) and item.name == "_evaluate")]
        assert ast.dump(old) == ast.dump(current), f"Unexpected change beyond shared core extraction: {name}"
    sources = {name: (ROOT / name).read_text() for name in FILES}
    for name in FILES[1:6]:
        assert sources[name] == prior.reference.git("show", f"{BASELINE}:{name}"), name
    sources["baseline/fixed_ou.py"] = source
    hashes = {}
    for name, value in sources.items():
        path = out / "sources" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        hashes[name] = hashlib.sha256(value.encode()).hexdigest()
    return frozen, hashes


def close_pair(actual, expected):
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-11, atol=1e-9)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-10, atol=1e-8)


def micro(prepared, args, frozen, native, callback_counts, counter):
    started = perf_counter()
    cached = prior.build(prepared)
    target, covariance = cached.target, cached.target.prepared
    initial = make_initial_point_fn(model=cached.model, overrides=cached.initial_values,
                                    return_transformed=True)(args.seed)
    result = {"model_build_seconds": perf_counter() - started, "graphs": {}, "checks": [],
              "calls": [], "dynamic_shared_checks": [], "invalid_checks": []}
    steps, functions = {}, {}
    perform_calls = 0
    original_perform = native.NativeAmplitudeOp.perform

    def native_perform(*inputs, **kwargs):
        nonlocal perform_calls
        perform_calls += 1
        return original_perform(*inputs, **kwargs)

    with patch.object(native.NativeAmplitudeOp, "perform", native_perform):
        for variant in VARIANTS:
            context = native.native_amplitude_context() if variant == "native_numba" else nullcontext()
            started = perf_counter()
            with warnings.catch_warnings(record=True) as caught, context:
                step = inference.PymcCachedSigmaNutsStep(
                    cached.amplitude, target=target, shared_cache=cached.shared_cache,
                    initial_cache=cached.initial_cache, modelled_mean=cached.modelled_mean,
                    model=cached.model, initial_point=initial, target_accept=0.9,
                    rng=np.random.default_rng(args.seed),
                )
            assert not any("object mode" in str(item.message).lower() for item in caught)
            steps[variant] = step
            fn = step.nuts_step._logp_dlogp_func._pytensor_function
            functions[variant] = fn
            linker = type(fn.maker.mode.linker).__name__
            assert ("Numba" in linker) == (variant == "native_numba")
            op_type = native.NativeAmplitudeOp if variant == "native_numba" else inference._PytensorSigmaLikelihoodOp
            nodes = [node for node in fn.maker.fgraph.toposort() if isinstance(node.op, op_type)]
            assert len(nodes) == 1
            if variant == "native_numba":
                assert len(nodes[0].inputs) == 6 and not vars(nodes[0].op)
                assert all(left is right for left, right in zip(nodes[0].inputs[2:], step.likelihood_op.array_inputs, strict=True))
            result["graphs"][variant] = {"linker": linker, "construction_seconds": perf_counter() - started,
                                          "nodes": dict(Counter(type(node.op).__name__ for node in fn.maker.fgraph.toposort()))}
        if args.anchors:
            with np.load(args.anchors) as anchors:
                residuals, amplitudes = anchors["residuals"], anchors["amplitudes"]
                np.testing.assert_allclose(target.fixed_contribution + anchors["coefficients"] @ target.design.T,
                                           anchors["modelled_means"], rtol=1e-12, atol=1e-10)
                np.testing.assert_allclose(target.observations - anchors["modelled_means"], residuals,
                                           rtol=1e-12, atol=1e-10)
        else:
            assert args.expected_sha256 != prior.PREPARED_SHA256
            residual = steps["shared_cvm"]._point_values(initial)[1]
            residuals = np.array([residual, residual + 0.1, residual - 0.05])
            amplitudes = np.array([np.full(target.n_group, value) for value in (0.3, 0.75, 1.2)])
        assert residuals.shape == (len(amplitudes), target.n_obs)
        assert amplitudes.shape == (len(residuals), target.n_group)
        eta_var = steps["shared_cvm"].vars[0]
        prior_graph = cached.model.logp(vars=[cached.amplitude], jacobian=True)
        prior_fn = pytensor.function([eta_var], [prior_graph, pt.grad(prior_graph, eta_var)],
                                    mode=Mode(linker="cvm", optimizer="fast_run"))
        orders = tuple(permutations(VARIANTS))
        # Revisiting the first point after all others catches stale residual capture.
        for visit, index in enumerate([*range(len(residuals)), 0]):
            revisit = visit == len(residuals)
            residual, amplitude = residuals[index], amplitudes[index]
            eta = np.log(amplitude).astype(eta_var.dtype)
            modes = covariance.mode_transform @ residual
            frozen_public = frozen(covariance, residual, amplitude)
            current_public = covariance.evaluate(residual, amplitude)
            close_pair((current_public.log_likelihood, current_public.gradient_site_amplitude),
                       (frozen_public[0], frozen_public[2]))
            np.testing.assert_allclose(current_public.gradient_residual, frozen_public[1], rtol=1e-11, atol=1e-9)
            prior_value, prior_gradient = prior_fn(eta)
            expected = {"kernel": (frozen_public[0], frozen_public[2]),
                        "graph": (frozen_public[0] + prior_value, amplitude * frozen_public[2] + prior_gradient)}
            kernel_args = (amplitude, modes, covariance.transformed_factor, covariance.mode_eigenvalues,
                           covariance.mode_site_index, covariance.base_logdet)
            before = {}
            for variant, step in steps.items():
                step.likelihood_op.install_residual(residual)
                before[variant] = ownership(target, residual, amplitude, eta,
                                             step.likelihood_op.residual.get_value(borrow=True))

            def evaluate(kind, variant):
                if kind == "graph":
                    return functions[variant](eta)
                if variant == "native_numba":
                    return native.native_amplitude_evaluate(*kernel_args)
                value = covariance._evaluate(modes, amplitude, amplitude_only=True)
                return value[0], value[2]

            for kind in ("kernel", "graph"):
                for repeat in range(1 if revisit else args.repeats + 1):
                    order = orders[(max(repeat, 1) - 1) % len(orders)]
                    for position, variant in enumerate(order):
                        counter["enabled"] = repeat == 0
                        callback_counts.clear()
                        context = (patch.object(fixed_ou.FixedOuLowRank, "_evaluate", frozen)
                                   if variant == "frozen_cvm" else nullcontext())
                        with context, warnings.catch_warnings(record=True) as caught:
                            wall, cpu = perf_counter(), process_time()
                            actual = evaluate(kind, variant)
                            seconds, cpu_seconds = perf_counter() - wall, process_time() - cpu
                            first_callbacks = dict(callback_counts)
                            if repeat == 0:
                                # Compiler-internal serialization is not a runtime callback.
                                callback_counts.clear()
                                close_pair(evaluate(kind, variant), expected[kind])
                        assert not any("object mode" in str(item.message).lower() for item in caught)
                        close_pair(actual, expected[kind])
                        if variant == "native_numba":
                            assert perform_calls == 0
                            assert all(size <= 64 for size in callback_counts), dict(callback_counts)
                        if repeat == 0:
                            result["checks"].append({"anchor": index, "revisit": revisit, "kind": kind,
                                                      "variant": variant, "warm_seconds": seconds,
                                                      "first_call_callback_bytes": first_callbacks,
                                                      "callback_bytes": dict(callback_counts),
                                                      "logp_error": abs(float(actual[0]) - float(expected[kind][0])),
                                                      "gradient_max_error": float(np.max(np.abs(actual[1] - expected[kind][1])))})
                        else:
                            result["calls"].append({"anchor": index, "kind": kind, "repeat": repeat,
                                                   "position": position, "variant": variant,
                                                   "seconds": seconds, "cpu_seconds": cpu_seconds})
            for variant, step in steps.items():
                assert before[variant] == ownership(target, residual, amplitude, eta,
                                                     step.likelihood_op.residual.get_value(borrow=True))

        counter["enabled"] = True
        native_step = steps["native_numba"]
        # The last installed conditioning point is the first anchor again.
        residual, amplitude = residuals[0], amplitudes[0]
        eta = np.log(amplitude).astype(eta_var.dtype)
        modes = covariance.mode_transform @ residual
        prior_value, prior_gradient = prior_fn(eta)
        fields = ("transformed_factor", "mode_eigenvalues", "mode_site_index", "base_logdet")
        for field, shared in zip(fields, native_step.likelihood_op.array_inputs, strict=True):
            original = shared.get_value().copy()
            changed = ((original + 1) % target.n_group if field == "mode_site_index" else
                       original * 1.001 if field == "transformed_factor" else original + 0.01)
            altered = replace(covariance, **{field: changed})
            expected = frozen(altered, modes, amplitude, amplitude_only=True)
            shared.set_value(changed)
            try:
                actual = functions["native_numba"](eta)
                close_pair(actual, (expected[0] + prior_value, amplitude * expected[2] + prior_gradient))
            finally:
                shared.set_value(original)
            np.testing.assert_array_equal(shared.get_value(), original)
            result["dynamic_shared_checks"].append(field)
        # Invalid inputs exercise the shared boundary independently of transforms.
        invalid_cases = []
        for value in (-1.0, np.nan, np.inf, 1e308):
            altered = amplitude.copy()
            altered[0] = value
            invalid_cases.append((f"amplitude-{value}", modes, altered))
        for value in (np.nan, np.inf):
            altered = modes.copy()
            altered[0] = value
            invalid_cases.append((f"residual-{value}", altered, amplitude))
        for name, residual_value, sigma in invalid_cases:
            expected = frozen(covariance, residual_value, sigma, amplitude_only=True)
            with np.errstate(over="ignore", invalid="ignore"):
                actual = native.native_amplitude_evaluate(sigma, residual_value, covariance.transformed_factor,
                                                          covariance.mode_eigenvalues, covariance.mode_site_index,
                                                          covariance.base_logdet)
            close_pair(actual, (expected[0], expected[2]))
            result["invalid_checks"].append(name)
        assert native.native_amplitude_evaluate.nopython_signatures
        assert not any(value.objectmode for value in native.native_amplitude_evaluate.overloads.values())
        assert not native.native_amplitude_evaluate.targetoptions["fastmath"]
        jit_fn = functions["native_numba"].vm.jit_fn
        assert jit_fn.nopython_signatures and not any(value.objectmode for value in jit_fn.overloads.values())
        assert not jit_fn.targetoptions.get("fastmath", False)
        result["native_kernel_signatures"] = [str(value) for value in native.native_amplitude_evaluate.nopython_signatures]
        result["native_graph_signatures"] = [str(value) for value in jit_fn.nopython_signatures]
        result["native_kernel_fastmath"] = native.native_amplitude_evaluate.targetoptions["fastmath"]
        result["native_graph_fastmath"] = jit_fn.targetoptions["fastmath"]
        result["native_perform_calls"] = perform_calls
        assert perform_calls == 0
    counter["enabled"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--expected-sha256", default=prior.PREPARED_SHA256)
    parser.add_argument("--anchors", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=152202610)
    args = parser.parse_args()
    if args.repeats <= 0 or args.repeats % 6:
        parser.error("Repeats must be a positive multiple of six for balanced three-arm ordering.")
    if args.expected_sha256 == prior.PREPARED_SHA256 and not os.environ.get("SLURM_JOB_ID"):
        parser.error("Production-shaped work requires Slurm.")
    args.out.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    frozen, sources = snapshot(args.out)
    callbacks, counter = Counter(), {"enabled": True}
    original_unpickle = serialize._numba_unpickle

    def counted_unpickle(address, data, hashed):
        if counter["enabled"]:
            callbacks[len(data)] += 1
        return original_unpickle(address, data, hashed)

    # Numba's C helper captures this Python function on its first use.
    serialize._numba_unpickle = counted_unpickle
    import ope152_native_amplitude as native

    manifest = {"status": "running", "started": datetime.now(UTC).isoformat(),
                "revision": prior.reference.git("rev-parse", "HEAD").strip(),
                "baseline_revision": prior.reference.git("rev-parse", BASELINE).strip(),
                "configuration": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
                "prepared_sha256": prior.reference.sha256(args.prepared), "source_sha256": sources,
                "python": sys.version, "packages": {name: version(name) for name in ("numpy", "scipy", "pymc", "pytensor", "numba")},
                "threadpools": threadpool_info(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "floatX": pytensor.config.floatX,
                "environment": {name: os.environ.get(name) for name in ("SLURM_JOB_ID", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTENSOR_FLAGS")},
                "scientific_target": {"tau_hours": 5.0, "amplitude_prior": {"pdf": "halfnormal", "sigma": 0.75}, "bc_prior": prior.reference.BC_PRIOR},
                "timing_note": "Outer timers only for native kernel and graph. Compilation/warm calls are separate. Counter bookkeeping is disabled during timed calls; the captured helper retains one enabled-flag branch for ordinary NumPy dtype boxing. True native code may box array outputs using 32-byte dtype callbacks; native Op.perform and captured-target payload callbacks are forbidden. Six selected posterior anchors are diagnostic, not posterior-weighted timing estimates."}
    (args.out / "working-tree.patch").write_text(prior.reference.git("diff", "HEAD", "--binary"))

    def save():
        manifest["elapsed_seconds"] = perf_counter() - started
        manifest["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    try:
        save()
        assert manifest["prepared_sha256"] == args.expected_sha256
        assert pytensor.config.floatX == "float64", "Match the previous rank-256 benchmarks' input precision."
        assert all(pool["num_threads"] == 1 for pool in threadpool_info() if pool["user_api"] == "blas")
        prepared = Co2PreparedInputs.load(args.prepared)
        assert prepared.aggregation_error_mode == "low_rank"
        manifest["original_dense_sha256"] = prepared.provenance.get("benchmark_source_sha256")
        if args.expected_sha256 == prior.PREPARED_SHA256:
            assert manifest["original_dense_sha256"] == prior.SOURCE_SHA256
            assert args.anchors is not None
        if args.anchors:
            assert prior.reference.sha256(args.anchors) == ANCHOR_SHA256
            manifest["anchors"] = {"sha256": ANCHOR_SHA256,
                                   "provenance": json.loads(args.anchors.with_suffix(".json").read_text())}
        manifest["micro"] = micro(prepared, args, frozen, native, callbacks, counter)
        for name in FILES:
            assert prior.reference.sha256(ROOT / name) == sources[name], name
        assert prior.reference.sha256(args.prepared) == manifest["prepared_sha256"]
        if args.anchors:
            assert prior.reference.sha256(args.anchors) == ANCHOR_SHA256
        manifest["status"] = "passed"
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        serialize._numba_unpickle = original_unpickle
        save()


if __name__ == "__main__":
    main()
