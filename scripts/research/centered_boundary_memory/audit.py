"""Supplement the original measurements with a slotted-dataclass inventory.

This builds and compiles one frozen recipe without sampling. Its array inventory
is separate evidence: it does not replace the original process-memory or timings.
"""

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path

import measure
from measure import probe, recipe, inventory, runner
from openghg_inversions.rhime.sampling import RhimeSampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check", type=Path, required=True)
    parser.add_argument("--variant", choices=measure.VARIANTS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert measure.pytensor.config.floatX == "float64", "Global arithmetic remains float64"
    assert Path(measure.numerical.__file__).resolve().is_relative_to(Path.cwd().resolve()), (
        "Set PYTHONPATH=. and run from the benchmark worktree root"
    )
    assert not args.out.exists(), "Supplementary audits never overwrite original evidence"
    config = json.loads(args.config.read_text())
    check = json.loads(args.check.read_text())
    assert check["config_sha256"] == probe.sha(args.config)
    assert check["variant"] == args.variant and check["stage"] == "check"
    prepared = probe.prepare(config)
    fingerprint = probe.input_fingerprint(prepared)
    assert fingerprint == check["input_sha256"]
    capture = {}
    with recipe(args.variant, capture):
        cached = probe.build(prepared, config)
    roots = {
        "prepared_inputs": prepared,
        "target": cached.target,
        "initial_cache": cached.initial_cache,
        "shared_cache": cached.shared_cache,
        "model_shared": {
            name: value for name, value in cached.model.named_vars.items() if hasattr(value, "get_value")
        },
    }
    inventories = {"constructed": inventory(roots)}
    sampler = runner._sampler_for_cached_graph(
        RhimeSampler(
            draws=250, tune=250, chains=1, sample_prior_predictive=False, sample_posterior_predictive=False
        ),
        cached_model=cached,
        sigma_target_accept=0.8,
        state_target_accept=0.9,
    )
    roots["sampler"] = sampler.sample_kwargs["step"]
    inventories["compiled"] = inventory(roots)
    refreshed = cached.target.refresh(0.5)
    cached.shared_cache.update(refreshed)
    inventories["refreshed"] = inventory(roots)
    prepared_arrays = [
        a for a in inventories["constructed"]["arrays"] if a["path"].startswith("prepared_inputs.")
    ]
    assert prepared_arrays, "The supplementary audit must include slotted prepared inputs"
    operand_arrays = [
        a
        for a in prepared_arrays
        if ".boundary_operands." in a["path"]
        and (".H_bc_anomaly." in a["path"] or ".H_bc_mass." in a["path"])
        and ".coords." not in a["path"]
    ]
    assert len(operand_arrays) == 4 and {a["dtype"] for a in operand_arrays} == {"float64"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    probe.write_json(
        args.out,
        {
            "scope": "supplementary retained-array audit; no sampling, process-memory or timing replacement",
            "original_measurement_revision": check["git_revision"],
            "source_package_path": str(Path(measure.numerical.__file__).resolve()),
            "versions": {
                "python": platform.python_version(),
                "numpy": measure.np.__version__,
                "pymc": measure.pm.__version__,
                "pytensor": measure.pytensor.__version__,
            },
            "audit_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "runtime_environment": {
                name: os.environ.get(name)
                for name in (
                    "PYTHONPATH",
                    "PYTENSOR_FLAGS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "DASK_NUM_WORKERS",
                    "SLURM_JOB_ID",
                    "SLURM_ARRAY_TASK_ID",
                )
            },
            "variant": args.variant,
            "config_sha256": probe.sha(args.config),
            "input_sha256": fingerprint,
            "check_sha256": probe.sha(args.check),
            "script_sha256": {p.name: probe.sha(p) for p in Path(__file__).parent.glob("*.py")},
            "array_inventory": inventories,
        },
    )
    print(f"PASS: supplementary {args.variant} inventory includes {len(prepared_arrays)} prepared arrays")


if __name__ == "__main__":
    main()
