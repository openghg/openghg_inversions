"""Tests for process-wide PyTensor defaults used by PyMC model paths."""

from __future__ import annotations

import os
import subprocess
import sys


def _probe_pytensor_config(code: str, *, flags: str | None = None) -> dict[str, str]:
    """Inspect PyTensor configuration in an isolated Python process."""
    env = os.environ.copy()
    retained_flags = [
        setting
        for setting in env.get("PYTENSOR_FLAGS", "").split(",")
        if setting and setting.partition("=")[0].strip() not in {"floatX", "warn_float64"}
    ]
    if flags is not None:
        retained_flags.extend(flags.split(","))
    if retained_flags:
        env["PYTENSOR_FLAGS"] = ",".join(retained_flags)
    else:
        env.pop("PYTENSOR_FLAGS", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    return dict(line.split("=", 1) for line in completed.stdout.strip().splitlines())


def test_fresh_rhime_import_defaults_pytensor_to_float32() -> None:
    """A fresh RHIME process uses the memory-conscious graph default."""
    values = _probe_pytensor_config(
        "import openghg_inversions.rhime; "
        "import pytensor; "
        "print('floatX=' + pytensor.config.floatX); "
        "print('warn=' + pytensor.config.warn_float64)"
    )

    assert values["floatX"] == "float32"
    assert values["warn"] == "warn"


def test_rhime_import_honours_explicit_pytensor_float64() -> None:
    """A process-level PyTensor precision selection overrides the default."""
    values = _probe_pytensor_config(
        "import openghg_inversions.rhime; "
        "import pytensor; "
        "print('floatX=' + pytensor.config.floatX); "
        "print('warn=' + pytensor.config.warn_float64)",
        flags="floatX=float64,warn_float64=ignore",
    )

    assert values["floatX"] == "float64"
    assert values["warn"] == "ignore"


def test_rhime_import_does_not_mutate_initialized_pytensor() -> None:
    """Notebook and host-process owners retain an initialized runtime."""
    values = _probe_pytensor_config(
        "import pytensor; "
        "pytensor.config.floatX = 'float64'; "
        "pytensor.config.warn_float64 = 'ignore'; "
        "import openghg_inversions.rhime; "
        "print('floatX=' + pytensor.config.floatX); "
        "print('warn=' + pytensor.config.warn_float64)"
    )

    assert values["floatX"] == "float64"
    assert values["warn"] == "ignore"
