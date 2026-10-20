"""Check the Slurm launcher without creating environments or submitting jobs."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("exit_code", [0, 7])
@pytest.mark.parametrize(
    "flags",
    ["", "floatX=float64,base_compiledir=/old, base_compiledir = /other,warn_float64=raise"],
)
def test_slurm_tox_isolates_scratch_preserves_flags_and_cleans_up(tmp_path, flags, exit_code):
    repo_root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "capture.json"
    uvx = bin_dir / "uvx"
    uvx.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "work = pathlib.Path(os.environ['TOX_WORK_DIR'])\n"
        "assert pathlib.Path(os.environ['TMPDIR']).is_dir()\n"
        "(work / 'created-by-tox').touch()\n"
        "pathlib.Path(os.environ['TEST_CAPTURE']).write_text(json.dumps({\n"
        "    'argv': sys.argv[1:], 'work': str(work),\n"
        "    'tmp': os.environ['TMPDIR'], 'flags': os.environ['PYTENSOR_FLAGS'],\n"
        "    'mpl': os.environ['MPLCONFIGDIR'], 'workers': os.environ['PYTEST_XDIST_WORKERS'],\n"
        "}))\n"
        "sys.exit(int(os.environ['TEST_EXIT_CODE']))\n"
    )
    uvx.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "OGI_REPO_ROOT": str(repo_root),
        "SLURM_TMPDIR": str(tmp_path),
        "SLURM_JOB_ID": "12345",
        "PYTENSOR_FLAGS": flags,
        "PYTEST_XDIST_WORKERS": "1",
        "TEST_CAPTURE": str(capture),
        "TEST_EXIT_CODE": str(exit_code),
    }

    result = subprocess.run(
        ["bash", str(repo_root / "scripts/slurm_tox.sh"), "-e", "py312-openghgCur"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == exit_code, result.stderr
    recorded = json.loads(capture.read_text())
    work = Path(recorded["work"])
    assert work.parent == tmp_path
    assert Path(recorded["tmp"]) == work / "tmp"
    assert Path(recorded["mpl"]) == work / "matplotlib"
    assert recorded["workers"] == "1"
    assert recorded["argv"] == [
        "--from", "tox>=4.24", "--with", "tox-uv", "tox", "--workdir", str(work),
        "-p", "--parallel-no-spinner", "-e", "py312-openghgCur",
    ]
    expected_flags = [f"base_compiledir={work}/pytensor"]
    if flags:
        expected_flags.extend(["floatX=float64", "warn_float64=raise"])
    assert recorded["flags"].split(",") == expected_flags
    assert not work.exists()
