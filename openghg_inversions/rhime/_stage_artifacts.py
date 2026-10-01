"""Serialization, path and content-digest mechanics shared by scientific workflows.

Recipes own artifact schemas, required contents, validation policy and execution
order. These functions only convert, write, locate and fingerprint artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

__all__ = ["artifact_path", "file_identity", "json_value", "write_json"]


def json_value(value: Any) -> Any:
    """Convert values to JSON-compatible data at an explicit serialization boundary.

    DataArrays are computed here and encoded with dimension coordinates;
    callers choose which scientific values belong in their artifact.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, slice):
        return {
            "type": "slice",
            "start": json_value(value.start),
            "stop": json_value(value.stop),
            "step": json_value(value.step),
        }
    if isinstance(value, xr.DataArray):
        materialized = value.compute()
        return {
            "dims": [str(dim) for dim in materialized.dims],
            "coords": {
                str(dim): json_value(materialized.coords[dim].to_numpy())
                for dim in materialized.dims
                if dim in materialized.coords
            },
            "values": json_value(materialized.to_numpy()),
        }
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [json_value(item) for item in value]
    return value


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


def file_identity(path: Path) -> str:
    """Read a file in bounded chunks and return its ``sha256:`` content digest."""
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


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
