"""Diagnose conditional log-amplitude curvature at a common, fixed reference.

Run production inputs on Slurm with one numerical thread. This does not adapt
a metric; inverse-Hessian correlations are local approximations at a non-mode.
"""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def negative_hessian(gradient, eta, step):
    """Differentiate the transformed log-density gradient by centered differences."""
    offsets = step * np.eye(len(eta))
    return np.column_stack([-(gradient(eta + offset) - gradient(eta - offset)) / (2 * step)
                            for offset in offsets])


def summarize(curvature):
    """Report finite curvature, preserving the distinction between PD and indefinite."""
    result = {"finite": bool(np.isfinite(curvature).all()), "positive_definite": None,
              "local_correlation": None}
    if not result["finite"]:
        return result
    result["raw_antisymmetry_max_absolute"] = float(np.max(np.abs(curvature - curvature.T)))
    symmetric = curvature / 2 + curvature.T / 2
    eigenvalues = np.linalg.eigvalsh(symmetric)
    result.update(hessian=symmetric.tolist(), eigenvalues=eigenvalues.tolist(),
                  positive_definite=bool(eigenvalues[0] > 0))
    diagonal = np.diag(symmetric)
    if np.all(diagonal > 0):
        sd = np.sqrt(diagonal)
        scaled = symmetric / sd[:, None] / sd[None, :]
        scaled_eigenvalues = np.linalg.eigvalsh(scaled)
        result.update(diagonal_condition=float(diagonal.max() / diagonal.min()),
                      scaled_offdiagonal_max_absolute=float(np.max(np.abs(scaled - np.eye(len(diagonal))))),
                      scaled_eigenvalues=scaled_eigenvalues.tolist(),
                      scaled_condition=(float(scaled_eigenvalues[-1] / scaled_eigenvalues[0])
                                        if scaled_eigenvalues[0] > 0 else None))
    if result["positive_definite"]:
        covariance = np.linalg.solve(symmetric, np.eye(len(diagonal)))
        if np.isfinite(covariance).all() and np.all(np.diag(covariance) > 0):
            sd = np.sqrt(np.diag(covariance))
            correlation = covariance / sd[:, None] / sd[None, :]
            result["local_correlation"] = correlation.tolist()
            result["local_correlation_max_offdiagonal_absolute"] = float(
                np.max(np.abs(correlation - np.eye(len(diagonal)))),
            )
    return result


def self_check():
    eta, diagonal = np.array([0.2, -0.4, 0.8]), np.array([2.0, 5.0, 11.0])
    for step in (1e-3, 5e-4):
        curvature = negative_hessian(lambda value: -diagonal * value, eta, step)
        np.testing.assert_allclose(curvature, np.diag(diagonal), rtol=1e-11, atol=1e-11)
        result = summarize(curvature)
        assert result["positive_definite"]
        np.testing.assert_allclose(result["local_correlation"], np.eye(3), atol=1e-12)
        prior_curvature = negative_hessian(lambda value: 1 - np.exp(2 * value) / 0.75**2, eta, step)
        np.testing.assert_allclose(prior_curvature, np.diag(2 * np.exp(2 * eta) / 0.75**2), rtol=1e-6)
    assert summarize(np.diag([-1.0, 2.0]))["local_correlation"] is None
    assert not summarize(np.array([[np.nan]]))["finite"]
    print("Analytic diagonal, transformed-prior, indefinite and nonfinite checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--anchors", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if any(value is None for value in (args.prepared, args.anchors, args.out)):
        parser.error("--prepared, --anchors and --out are required unless --self-check is used")

    import profile_ope152_amplitude as profile
    from threadpoolctl import threadpool_info

    pools = threadpool_info()
    if any(pool["num_threads"] != 1 for pool in pools if pool["user_api"] == "blas"):
        raise RuntimeError("Start Python with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1")
    args.out.mkdir(parents=True, exist_ok=False)
    sha = profile.reference.sha256
    sources = {str(path.resolve().relative_to(profile.ROOT)): path for path in
               map(Path, (__file__, profile.__file__, profile.reference.__file__,
                          profile.fixed_ou.__file__, profile.cached_sigma.__file__, profile.inference.__file__))}
    manifest = {"status": "running", "started_at": datetime.now(UTC).isoformat(),
                "revision": profile.reference.git("rev-parse", "HEAD").strip(),
                "prepared": str(args.prepared.resolve()), "prepared_sha256": sha(args.prepared),
                "anchors": str(args.anchors.resolve()), "anchors_sha256": sha(args.anchors),
                "threadpools": pools, "source_sha256": {}, "results": [],
                "scope": "Local non-mode conditional curvature at fixed eta; not metric adaptation."}
    assert manifest["prepared_sha256"] == profile.PREPARED_SHA256
    for name, source in sources.items():
        destination = args.out / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        manifest["source_sha256"][name] = sha(source)
    metadata = args.anchors.with_suffix(".json")
    if metadata.exists():
        (args.out / "anchor-metadata.json").write_bytes(metadata.read_bytes())
        manifest["anchor_metadata_sha256"] = sha(metadata)
        assert json.loads(metadata.read_text())["npz_sha256"] == manifest["anchors_sha256"]
    started = perf_counter()
    try:
        target = profile.build(profile.Co2PreparedInputs.load(args.prepared)).target
        with np.load(args.anchors) as selected:
            residuals, amplitudes = selected["residuals"], selected["amplitudes"]
            assert residuals.shape == (len(amplitudes), target.n_obs)
            assert amplitudes.shape == (len(residuals), target.n_group)
            assert np.isfinite(residuals).all() and np.isfinite(amplitudes).all() and np.all(amplitudes > 0)
            np.testing.assert_allclose(target.fixed_contribution + selected["coefficients"] @ target.design.T,
                                       selected["modelled_means"], rtol=1e-12, atol=1e-10)
            np.testing.assert_allclose(target.observations - selected["modelled_means"], residuals,
                                       rtol=1e-12, atol=1e-10)
        eta = np.log(np.median(amplitudes, axis=0))
        manifest.update(common_sigma=np.exp(eta).tolist(), steps=[1e-3, 5e-4])
        op = profile.inference._PytensorSigmaLikelihoodOp(target)
        for index, residual in enumerate(residuals):
            op.install_residual(residual)

            def gradient(value):
                sigma = np.exp(value)
                return sigma * profile.op_evaluate(op, sigma)[1] + 1 - sigma**2 / 0.75**2

            matrices = [negative_hessian(gradient, eta, step) for step in manifest["steps"]]
            finite = all(np.isfinite(matrix).all() for matrix in matrices)
            manifest["results"].append({
                "anchor_index": index, "curvatures": [summarize(matrix) for matrix in matrices],
                "step_agreement_max_absolute": float(np.max(np.abs(matrices[0] - matrices[1]))) if finite else None,
                "step_agreement_relative_frobenius": (float(np.linalg.norm(matrices[0] - matrices[1]) /
                                                           max(np.linalg.norm(matrices[1]), np.finfo(float).tiny))
                                                       if finite else None),
            })
        manifest["gradient_evaluations"] = len(residuals) * target.n_group * 2 * len(manifest["steps"])
        assert sha(args.prepared) == manifest["prepared_sha256"]
        assert sha(args.anchors) == manifest["anchors_sha256"]
        assert all(sha(source) == manifest["source_sha256"][name] for name, source in sources.items())
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["elapsed_seconds"] = perf_counter() - started
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
