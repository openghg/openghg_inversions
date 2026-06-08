"""Tests for process-wide PyTensor defaults used by PyMC model paths."""

from __future__ import annotations

import subprocess
import sys


def test_rhime_import_configures_pytensor_float32_before_pymc() -> None:
    """Importing RHIME applies the same float32 PyTensor default as legacy fixedbasis."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pytensor; "
                "print('before=' + pytensor.config.floatX); "
                "import openghg_inversions.rhime; "
                "print('after=' + pytensor.config.floatX); "
                "print('warn=' + pytensor.config.warn_float64)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    values = dict(line.split("=", 1) for line in completed.stdout.strip().splitlines())
    assert values["after"] == "float32"
    assert values["warn"] == "warn"
