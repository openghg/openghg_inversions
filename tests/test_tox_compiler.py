"""Regression tests for the tox PyTensor compiler preflight launcher."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

import pytest


LAUNCHER = Path(__file__).parents[1] / "scripts" / "run_pytest_with_pytensor_compiler.sh"


def _write_executable(path: Path, content: str) -> None:
    """Create or replace an executable test helper script.

    Args:
        path: Destination path for the helper script.
        content: Complete script contents to write.

    Raises:
        OSError: If the file cannot be written or its permissions cannot be
            changed to ``0755``.
    """
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _is_rocky_linux() -> bool:
    """Return whether the test host identifies as Rocky Linux."""
    try:
        release = platform.freedesktop_os_release()
    except OSError:
        return False
    return release.get("ID") == "rocky" or "rocky" in release.get("ID_LIKE", "").split()


@pytest.fixture
def launcher_env(tmp_path: Path) -> tuple[dict[str, str], Path, Path, Path, Path]:
    """Provide isolated fake Python, module, and child-command executables.

    The fake ``python`` reports compiler state for PyTensor probes and records
    the separate ArviZ import. The module function can optionally export a
    compiler into the launcher's shell, while the child command records its
    arguments without starting pytest.

    Args:
        tmp_path: Pytest-managed directory for the fake commands and output
            records.

    Returns:
        The isolated environment, child-command path, child-argument record,
        event-log path, and module-log path. Record paths may not exist until
        the launcher invokes their corresponding fake commands.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    event_log = tmp_path / "events.log"
    module_log = tmp_path / "module.log"
    module_init_sourced = tmp_path / "module-init-sourced"
    child_args = tmp_path / "child-args"

    _write_executable(
        bin_dir / "python",
        """#!/bin/bash
case "${2:-}" in
    *pytensor*)
        printf 'pytensor\\n' >> "${FAKE_EVENT_LOG}"
        printf '%s' "${FAKE_PYTENSOR_CXX:-}"
        ;;
    *arviz*)
        printf 'arviz\\n' >> "${FAKE_EVENT_LOG}"
        ;;
    *)
        printf 'unexpected python invocation: %s\\n' "$*" >&2
        exit 99
        ;;
esac
""",
    )
    child_command = tmp_path / "record-child"
    _write_executable(
        child_command,
        """#!/bin/bash
printf '%s\\n' "$@" > "${FAKE_CHILD_ARGS}"
exit "${FAKE_CHILD_EXIT:-0}"
""",
    )
    module_init = tmp_path / "modules.sh"
    module_init.write_text(
        """printf 'sourced\\n' > "${FAKE_MODULE_INIT_SOURCED}"
module() {
    printf '%s\\n' "$*" > "${FAKE_MODULE_LOG}"
    if [[ "${FAKE_MODULE_CONFIGURES_CXX:-0}" == "1" ]]; then
        export FAKE_PYTENSOR_CXX=g++
    fi
}
"""
    )

    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PYTENSOR_") and not key.startswith("BASH_FUNC_module")
    }
    env.update(
        {
            "PATH": str(bin_dir),
            "HOSTNAME": "unrelated-test-host",
            "FAKE_CHILD_ARGS": str(child_args),
            "FAKE_EVENT_LOG": str(event_log),
            "FAKE_MODULE_INIT_SOURCED": str(module_init_sourced),
            "FAKE_MODULE_LOG": str(module_log),
            "PYTENSOR_MODULE_INIT": str(module_init),
        }
    )
    return env, child_command, child_args, event_log, module_log


def test_launcher_forwards_arguments_and_child_status(
    launcher_env: tuple[dict[str, str], Path, Path, Path, Path],
) -> None:
    """An effective compiler lets the exact pytest command run unchanged."""
    env, child_command, child_args, event_log, _ = launcher_env
    env.update({"FAKE_PYTENSOR_CXX": "g++", "FAKE_CHILD_EXIT": "7"})
    arguments = ["-k", "alpha or beta", "tests/a path/test_example.py"]

    result = subprocess.run(
        ["/bin/bash", str(LAUNCHER), str(child_command), *arguments],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 7
    assert child_args.read_text().splitlines() == arguments
    assert event_log.read_text().splitlines() == ["pytensor", "arviz"]


def test_launcher_bootstraps_default_module_on_blue_pebble(
    launcher_env: tuple[dict[str, str], Path, Path, Path, Path],
) -> None:
    """Blue Pebble auto mode loads the default module and rechecks PyTensor."""
    env, child_command, child_args, event_log, module_log = launcher_env
    env.update({"HOSTNAME": "bp1-test", "FAKE_MODULE_CONFIGURES_CXX": "1"})

    result = subprocess.run(
        ["/bin/bash", str(LAUNCHER), str(child_command), "tests/test_array_ops.py"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert module_log.read_text().splitlines() == ["load gcc/12.3.0-sknc"]
    assert event_log.read_text().splitlines() == ["pytensor", "pytensor", "arviz"]
    assert child_args.read_text().splitlines() == ["tests/test_array_ops.py"]


def test_launcher_uses_configured_module_in_always_mode(
    launcher_env: tuple[dict[str, str], Path, Path, Path, Path],
) -> None:
    """Explicit bootstrap uses the configured module on an unrelated host."""
    env, child_command, _, _, module_log = launcher_env
    env.update(
        {
            "PYTENSOR_COMPILER_BOOTSTRAP": "always",
            "PYTENSOR_COMPILER_MODULE": "gcc/custom",
            "FAKE_MODULE_CONFIGURES_CXX": "1",
        }
    )

    result = subprocess.run(
        ["/bin/bash", str(LAUNCHER), str(child_command)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert module_log.read_text().splitlines() == ["load gcc/custom"]


def test_launcher_fails_before_pytest_when_recheck_has_no_compiler(
    launcher_env: tuple[dict[str, str], Path, Path, Path, Path],
) -> None:
    """A successful module command cannot bypass the compiler recheck."""
    env, child_command, child_args, event_log, module_log = launcher_env
    env["PYTENSOR_COMPILER_BOOTSTRAP"] = "always"

    result = subprocess.run(
        ["/bin/bash", str(LAUNCHER), str(child_command)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert module_log.read_text().splitlines() == ["load gcc/12.3.0-sknc"]
    assert event_log.read_text().splitlines() == ["pytensor", "pytensor"]
    assert not child_args.exists()
    assert "still has no configured C++ compiler" in result.stderr
    assert "PYTENSOR_FLAGS=cxx=/path/to/g++" in result.stderr


@pytest.mark.skipif(_is_rocky_linux(), reason="Rocky Linux is an automatic compiler-bootstrap target")
def test_launcher_does_not_load_modules_automatically_on_unrelated_host(
    launcher_env: tuple[dict[str, str], Path, Path, Path, Path],
) -> None:
    """Auto mode fails early without attempting modules on unrelated hosts."""
    env, child_command, child_args, event_log, module_log = launcher_env
    module_init_sourced = Path(env["FAKE_MODULE_INIT_SOURCED"])

    result = subprocess.run(
        ["/bin/bash", str(LAUNCHER), str(child_command)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert event_log.read_text().splitlines() == ["pytensor", "pytensor"]
    assert not module_init_sourced.exists()
    assert not module_log.exists()
    assert not child_args.exists()
    assert "PYTENSOR_COMPILER_BOOTSTRAP=always" in result.stderr
