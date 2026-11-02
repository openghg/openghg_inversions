"""Matched runtime comparisons preserve science and isolate execution settings."""

import json
from pathlib import Path
import subprocess

import pytest

from scripts import compare_test_runtime


@pytest.mark.parametrize("exit_code,report", [(0, True), (1, True), (0, False)])
def test_comparison_isolates_caches_preserves_flags_and_propagates_failure(tmp_path, monkeypatch, exit_code, report):
    calls = []
    output_dir = tmp_path / "reports"
    monkeypatch.setenv("PYTENSOR_FLAGS", "floatX=float64,base_compiledir=/old,compiledir=/old-C,warn_float64=raise")

    def run(command, **kwargs):
        env = kwargs["env"]
        calls.append((command, env.copy()))
        if report:
            Path(env["OGI_TEST_RUNTIME_REPORT"]).write_text('{}')
        return subprocess.CompletedProcess(command, exit_code)

    monkeypatch.setattr(compare_test_runtime.subprocess, "run", run)
    if exit_code:
        with pytest.raises(subprocess.CalledProcessError):
            compare_test_runtime.compare(output_dir)
    elif not report:
        with pytest.raises(RuntimeError, match="process runtime diagnostics"):
            compare_test_runtime.compare(output_dir)
    else:
        compare_test_runtime.compare(output_dir)
    assert [env["OPENBLAS_NUM_THREADS"] for _, env in calls] == (
        ["1"] if exit_code or not report else ["1", "2", "2", "1"]
    )
    caches = []
    for command, env in calls:
        assert "-n" not in command and "no:xdist" in command
        assert "tests.ci_runtime" in command
        assert env["DASK_NUM_WORKERS"] == "2"
        assert env["OPENBLAS_NUM_THREADS"] == env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"]
        flags = env["PYTENSOR_FLAGS"].split(",")
        assert flags[1:] == ["floatX=float64", "warn_float64=raise"]
        caches.append(flags[0])
    assert len(set(caches)) == len(caches)
    measurements = json.loads((output_dir / "comparison.json").read_text())
    assert [row["exit_code"] for row in measurements] == [exit_code] * len(calls)


def test_comparison_rejects_previous_results(tmp_path):
    (tmp_path / "repeat-0-threads-1-runtime.json").write_text('{}')
    with pytest.raises(FileExistsError):
        compare_test_runtime.compare(tmp_path)
