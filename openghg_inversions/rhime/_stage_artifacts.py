"""Serialization and path mechanics shared by scientific workflows.

Recipes own artifact schemas, required contents, validation policy and execution
order. These functions only convert, write and locate artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any

from openghg_inversions._serialization_codecs import _json_value

__all__ = ["artifact_path", "json_value", "write_json"]


def json_value(value: Any) -> Any:
    """Convert values to JSON-compatible data at an explicit serialization boundary.

    DataArrays are computed here and encoded with dimension coordinates;
    callers choose which scientific values belong in their artifact.
    """
    return _json_value(value, materialize_dataarrays=True)


def write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Write a JSON mapping, creating parent directories and replacing the file.

    Uses :func:`json_value` to materialize any array values for serialization.
    Nonfinite JSON numbers are rejected. Returns the resolved output path.
    """
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_value(value), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def artifact_path(path: Path) -> str:
    """Return a path relative to ``RUN_ROOT`` when contained there, else absolute."""
    run_root = os.environ.get("RUN_ROOT")
    if run_root is not None:
        try:
            return str(path.resolve().relative_to(Path(run_root).resolve()))
        except ValueError:
            pass
    return str(path.resolve())


def _output_path(output_dir: Path, requested: str | Path | None, default_name: str) -> Path:
    """Resolve an output path and keep it within the declared stage directory."""
    if requested is None:
        path = (output_dir / default_name).resolve()
    else:
        requested_path = Path(requested)
        path = (
            requested_path.resolve()
            if requested_path.is_absolute()
            else (output_dir / requested_path).resolve()
        )
    try:
        path.relative_to(output_dir)
    except ValueError:
        raise ValueError(f"Output path {path} must be beneath stage output directory {output_dir}.") from None
    return path


def _stage_output_directory(output_dir: str | Path) -> Path:
    """Create a stage directory and reject pre-existing symlink redirects."""
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    symlinks = [str(path) for path in destination.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError(
            f"Stage output directory {destination} contains symlink(s), which are not safe output targets: "
            f"{symlinks!r}."
        )
    return destination


def _filename_component(name: str, value: Any) -> str:
    """Validate a value interpolated into staged output filenames."""
    text = str(value)
    path = Path(text)
    if not text or text in {".", ".."} or path.is_absolute() or path.name != text:
        raise ValueError(
            f"Staged RHIME {name} {value!r} must be a non-empty filename component without directories."
        )
    return text
