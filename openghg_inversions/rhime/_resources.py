"""Resolve RHIME resource paths at runner and file-frontend boundaries."""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from .params import RhimeConfig

RESOURCE_PATH_OPTIONS = frozenset(
    {
        "basis_directory",
        "basis_path",
        "basis_partition_path",
        "basis_output_path",
        "landsea_path",
        "outer_regions_path",
        "bc_basis_directory",
        "country_directory",
        "country_file",
    }
)


def resolve_resource_paths(params: Mapping[str, Any], *, base_dir: Path) -> dict[str, Any]:
    """Copy options, resolving selected paths against their frontend's base."""
    resolved = dict(params)
    for name in RESOURCE_PATH_OPTIONS:
        value = resolved.get(name)
        if not isinstance(value, str | Path) or not value:
            continue
        path = Path(value).expanduser()
        resolved[name] = str(path if path.is_absolute() else (base_dir / path).resolve())
    return resolved


def resolve_config_paths(config: RhimeConfig) -> RhimeConfig:
    """Resolve invocation paths without changing the borrowed request.

    Historical outer-map names retain their compatibility adapter's directory
    convention; the adapter resolves them before scientific preparation.
    """
    names = RESOURCE_PATH_OPTIONS - {"country_file"}
    if config.country_directory is not None:
        names = names - {"outer_regions_path"}
    paths = resolve_resource_paths(config.select(*names), base_dir=Path.cwd())
    output_paths = resolve_resource_paths(
        {"country_file": config.output.country_file},
        base_dir=Path.cwd(),
    )
    return replace(config, **paths, output=replace(config.output, **output_paths))
