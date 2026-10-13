"""Convert declared RHIME filesystem types at frontend boundaries."""

from collections.abc import Mapping
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin


def _convert_path(value: Any, annotation: Any, base_dir: Path) -> Any:
    """Convert only filesystem members of a declared type, preserving sentinels."""
    if value is None:
        return None
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation is Path:
        if isinstance(value, Path) and value.is_absolute():
            return value
        if not isinstance(value, str | Path):
            raise ValueError(f"Expected a filesystem path, got {type(value).__name__}.")
        path = Path(value).expanduser()
        return path if path.is_absolute() else (base_dir / path).resolve()
    if origin in (UnionType, Union):
        if isinstance(value, bool) and bool in args:
            return value
        for member in args:
            if member is Path or get_origin(member) in (list, tuple):
                return _convert_path(value, member, base_dir)
    if origin in (list, tuple) and args and args[0] is Path:
        if isinstance(value, origin) and all(isinstance(path, Path) and path.is_absolute() for path in value):
            return value
        if isinstance(value, str | Path):
            raise ValueError("Expected a collection of filesystem paths, not a single path.")
        return origin(_convert_path(item, Path, base_dir) for item in value)
    return value


def resolve_resource_paths(
    params: Mapping[str, Any], *, annotations: Mapping[str, Any], base_dir: Path,
) -> dict[str, Any]:
    """Convert external paths using caller-owned types and frontend origins.

    Already resolved absolute paths and path tuples are forwarded unchanged.
    This helper knows no concrete configuration, output or scientific owners.
    """
    return {
        name: _convert_path(value, annotations.get(name), base_dir)
        for name, value in params.items()
    }
