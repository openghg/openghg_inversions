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


def _validate_filename_component(name: str, value: Any) -> None:
    """Validate one value interpolated into an output filename."""
    text = str(value)
    path = Path(text)
    if not text or text in {".", ".."} or path.is_absolute() or path.name != text:
        raise ValueError(
            f"{name} {value!r} must be a non-empty filename component without directories."
        )


def read_manifest(
    path: str | Path, schema: str, required_keys: tuple[str, ...] = (),
) -> tuple[Path, dict[str, Any]]:
    """Read a version-one manifest and require the caller's envelope fields."""
    path = Path(path).resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or record.get("schema") != schema
            or type(record.get("schema_version")) is not int or record["schema_version"] != 1):
        raise ValueError(f"{path} is not a {schema} version-1 manifest.")
    missing = [name for name in required_keys if name not in record]
    if missing:
        raise ValueError(f"Manifest {path} is missing {missing!r}.")
    return path, record


def reference(manifest: Path, value: Any) -> Path:
    """Resolve a nonempty artifact reference relative to its manifest."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"Manifest {manifest} requires a nonempty artifact reference; found {value!r}.")
    return (manifest.parent / value).resolve()


def check_checkpoint_targets(
    destination: Path, *, stage: str | None, payloads: tuple[str, ...], referenced: tuple[Path, ...] = (),
) -> None:
    """Protect concrete writes from redirects and dependency aliases."""
    protected = list(referenced)
    targets = payloads if stage is None else (*payloads, f".{stage}-manifest.tmp", f"{stage}-manifest.json")
    for name in targets:
        target = _output_path(destination, None, name)
        if any(target == path or (target.exists() and path.exists() and target.samefile(path))
               for path in protected):
            raise ValueError(f"Checkpoint target {target} would overwrite another checkpoint artifact.")
        protected.append(target)


def start_checkpoint(
    directory: str | Path, stage: str, *, payloads: tuple[str, ...], referenced: tuple[Path, ...] = (),
) -> Path:
    """Check concrete targets and remove the old marker immediately before writes."""
    destination = Path(directory).resolve()
    check_checkpoint_targets(destination, stage=stage, payloads=payloads, referenced=referenced)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{stage}-manifest.json").unlink(missing_ok=True)
    return destination


def finish_checkpoint(directory: Path, stage: str, record: Mapping[str, Any]) -> Path:
    """Publish the completion marker after the caller has written every payload."""
    temporary = write_json(directory / f".{stage}-manifest.tmp", record)
    return temporary.replace(directory / f"{stage}-manifest.json")
