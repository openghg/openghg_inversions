"""Convert declared RHIME filesystem types at frontend boundaries."""

from collections.abc import Mapping
from dataclasses import fields, replace
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from .params import RhimeConfig
from .specs import RhimeOutputSpec


def _convert_path(value: Any, annotation: Any, base_dir: Path) -> Any:
    """Convert only filesystem members of a declared type, preserving sentinels."""
    if value is None:
        return None
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation is Path:
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
        if isinstance(value, str | Path):
            raise ValueError("Expected a collection of filesystem paths, not a single path.")
        return origin(_convert_path(item, Path, base_dir) for item in value)
    return value


def resolve_resource_paths(params: Mapping[str, Any], *, base_dir: Path) -> dict[str, Any]:
    """Copy options and convert filesystem fields declared by their owners."""
    annotations = get_type_hints(RhimeConfig) | get_type_hints(RhimeOutputSpec)
    return {
        name: _convert_path(value, annotations.get(name), base_dir)
        for name, value in params.items()
    }


def resolve_config_paths(config: RhimeConfig) -> RhimeConfig:
    """Resolve paths on a directly constructed request without mutating it."""
    paths = resolve_resource_paths(
        {setting.name: getattr(config, setting.name) for setting in fields(config)},
        base_dir=Path.cwd(),
    )
    output_paths = resolve_resource_paths(
        {setting.name: getattr(config.output, setting.name) for setting in fields(config.output)},
        base_dir=Path.cwd(),
    )
    paths["output"] = replace(config.output, **output_paths)
    return replace(config, **paths)
