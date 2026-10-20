"""Check the Slurm launcher without creating environments or submitting jobs."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import check_pytensor_compiler


@pytest.mark.parametrize(
    "flags,exit_code",
    [("", 0), ("floatX=float64,base_compiledir=/old, base_compiledir = /other,warn_float64=raise", 7)],
)
@pytest.mark.parametrize(
    "overrides,tox_args",
    [
        ({}, []),
        (
            {
                "PYTEST_XDIST_WORKERS": "2",
                "OPENBLAS_NUM_THREADS": "3",
                "OMP_NUM_THREADS": "4",
                "UV_PYTHON_PREFERENCE": "only-system",
            },
            ["-p", "2"],
        ),
    ],
)
def test_slurm_tox_isolates_scratch_preserves_flags_and_cleans_up(
    tmp_path, flags, exit_code, overrides, tox_args
):
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
        "    'blas_threads': os.environ['OPENBLAS_NUM_THREADS'], 'omp_threads': os.environ['OMP_NUM_THREADS'],\n"
        "    'python_preference': os.environ['UV_PYTHON_PREFERENCE'],\n"
        "}))\n"
        "sys.exit(int(os.environ['TEST_EXIT_CODE']))\n"
    )
    uvx.chmod(0o755)
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "PYTEST_XDIST_WORKERS",
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "UV_PYTHON_PREFERENCE",
        }
    }
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "OGI_REPO_ROOT": str(repo_root),
            "SLURM_TMPDIR": str(tmp_path),
            "SLURM_JOB_ID": "12345",
            "PYTENSOR_FLAGS": flags,
            "TEST_CAPTURE": str(capture),
            "TEST_EXIT_CODE": str(exit_code),
            **overrides,
        }
    )

    result = subprocess.run(
        ["bash", str(repo_root / "scripts/slurm_tox.sh"), "-e", "py312-openghgCur", *tox_args],
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
    assert recorded["workers"] == overrides.get("PYTEST_XDIST_WORKERS", "0")
    assert recorded["blas_threads"] == overrides.get("OPENBLAS_NUM_THREADS", "1")
    assert recorded["omp_threads"] == overrides.get("OMP_NUM_THREADS", "1")
    assert recorded["python_preference"] == overrides.get("UV_PYTHON_PREFERENCE", "managed")
    assert recorded["argv"] == [
        "--from",
        "tox>=4.24",
        "--with",
        "tox-uv",
        "tox",
        "--workdir",
        str(work),
        "-e",
        "py312-openghgCur",
        *tox_args,
    ]
    expected_flags = [f"base_compiledir={work}/pytensor"]
    if flags:
        expected_flags.extend(["floatX=float64", "warn_float64=raise"])
    assert recorded["flags"].split(",") == expected_flags
    assert not work.exists()


@pytest.fixture
def compiler_preflight(monkeypatch, tmp_path):
    include = tmp_path / "include"
    include.mkdir()
    (include / "Python.h").touch()
    monkeypatch.setattr(check_pytensor_compiler.sysconfig, "get_path", lambda name: str(include))
    monkeypatch.setitem(sys.modules, "pytensor", SimpleNamespace(config=SimpleNamespace(cxx="/compiler/c++")))
    return include


def test_compiler_preflight_checks_resolved_python_with_configured_compiler(monkeypatch, compiler_preflight):
    calls = []
    monkeypatch.setattr(
        check_pytensor_compiler.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    check_pytensor_compiler.main()

    assert calls == [
        (
            (["/compiler/c++", "-x", "c++", "-fsyntax-only", f"-I{compiler_preflight}", "-"],),
            {"input": "#include <Python.h>\n", "text": True, "capture_output": True, "check": True},
        )
    ]


def test_compiler_preflight_rejects_missing_python_headers(monkeypatch, compiler_preflight):
    (compiler_preflight / "Python.h").unlink()
    monkeypatch.setattr(
        check_pytensor_compiler.subprocess, "run", lambda *args, **kwargs: pytest.fail("Compiler ran")
    )

    with pytest.raises(SystemExit, match="UV_PYTHON_PREFERENCE=only-managed") as error:
        check_pytensor_compiler.main()

    assert sys.executable in str(error.value)
    assert str(compiler_preflight / "Python.h") in str(error.value)


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("compiler missing"),
        subprocess.CalledProcessError(1, "c++", stderr="broken Python headers"),
    ],
)
def test_compiler_preflight_reports_compiler_failures(monkeypatch, compiler_preflight, failure):
    def failed_compiler(*args, **kwargs):
        raise failure

    monkeypatch.setattr(check_pytensor_compiler.subprocess, "run", failed_compiler)

    with pytest.raises(SystemExit, match="Compiler preflight failed") as error:
        check_pytensor_compiler.main()

    assert sys.executable in str(error.value)
    assert str(compiler_preflight / "Python.h") in str(error.value)
    assert "/compiler/c++" in str(error.value)
    assert (failure.stderr if isinstance(failure, subprocess.CalledProcessError) else str(failure)) in str(
        error.value
    )


def test_compiler_preflight_preserves_disabled_compiler(monkeypatch, compiler_preflight, capsys):
    sys.modules["pytensor"].config.cxx = ""
    (compiler_preflight / "Python.h").unlink()
    monkeypatch.setattr(
        check_pytensor_compiler.subprocess, "run", lambda *args, **kwargs: pytest.fail("Compiler ran")
    )

    check_pytensor_compiler.main()

    assert "cxx is disabled, skipping compilation" in capsys.readouterr().out
