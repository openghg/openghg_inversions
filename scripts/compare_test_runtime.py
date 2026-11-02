"""Compare numerical thread limits in serial pytest processes on one host."""

import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time


CASES = [
    "tests/test_co2_cached_sigma.py::test_named_cached_co2_model_owns_one_observed_likelihood_and_sampler_inputs",
    "tests/test_co2_cached_sigma.py::test_native_joint_outputs_use_each_draw_and_ignore_sampler_cache",
    "tests/test_co2_cached_sigma.py::test_cached_all_fixed_states_infer_amplitudes",
]


def compare(output_dir: Path) -> None:
    """Run the same cases in alternating order with fresh compiler caches."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True)
    measurements = []
    for repetition, order in enumerate(((1, 2), (2, 1))):
        for threads in order:
            name = f"repeat-{repetition}-threads-{threads}"
            with tempfile.TemporaryDirectory(prefix="ogi-runtime-") as scratch:
                scratch = Path(scratch)
                env = dict(os.environ)
                env.update({
                    "DASK_NUM_WORKERS": "2",
                    "OPENBLAS_NUM_THREADS": str(threads),
                    "OMP_NUM_THREADS": str(threads),
                    "MKL_NUM_THREADS": str(threads),
                    "MPLCONFIGDIR": str(scratch / "matplotlib"),
                    "NUMBA_CACHE_DIR": str(scratch / "numba"),
                    "OGI_TEST_RUNTIME_REPORT": str(output_dir / f"{name}-runtime.json"),
                })
                flags = [flag for flag in env.get("PYTENSOR_FLAGS", "").split(",")
                         if flag.strip() and flag.partition("=")[0].strip()
                         not in {"base_compiledir", "compiledir"}]
                env["PYTENSOR_FLAGS"] = ",".join([f"base_compiledir={scratch / 'pytensor'}", *flags])
                command = [
                    sys.executable, "-m", "pytest", "-q", "-p", "no:xdist",
                    "-p", "tests.ci_runtime", "--durations=10", f"--basetemp={scratch / 'pytest'}",
                    f"--junitxml={output_dir / f'{name}.xml'}", *CASES,
                ]
                before = resource.getrusage(resource.RUSAGE_CHILDREN)
                started = time.perf_counter()
                with (output_dir / f"{name}.log").open("w") as log:
                    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                            env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
                after = resource.getrusage(resource.RUSAGE_CHILDREN)
                measurements.append({
                    "repetition": repetition, "numerical_threads": threads,
                    "pytest_processes": 1, "dask_workers_per_process": 2,
                    "wall_seconds": time.perf_counter() - started,
                    "child_cpu_seconds": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                    "exit_code": result.returncode, "runtime_report": env["OGI_TEST_RUNTIME_REPORT"],
                })
                (output_dir / "comparison.json").write_text(json.dumps(measurements, indent=2) + "\n")
                print(json.dumps(measurements[-1]), flush=True)
                result.check_returncode()
                if not Path(env["OGI_TEST_RUNTIME_REPORT"]).is_file():
                    raise RuntimeError("pytest did not write the process runtime diagnostics")


if __name__ == "__main__":
    compare(Path(sys.argv[1]))
