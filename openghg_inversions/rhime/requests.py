"""The single RHIME request loader shared by full runs and saved phases.

It reads one explicit source, applies invocation overrides and translates
deprecated spellings at the runner frontend, so configuration owners in
:mod:`openghg_inversions.rhime.params` stay canonical-only.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any, get_type_hints

from ._resources import resolve_resource_paths
from ._runner_compat import REQUEST_ALIASES, migration_errors, translate_rhime_aliases
from .ini import read_rhime_ini
from .params import REQUEST_CLAIM_NAMES, DataConfig, rhime_option_groups
from .specs import RhimeOutputSpec


def load_rhime_params(
    *,
    config_file: str | Path | None = None,
    params_file: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    phase_groups: Collection[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Read one explicit request, apply overrides and group it by owner.

    This is the single request loader for full runs and saved phases. Raw
    overrides replace file values before alias translation, so a canonical
    spelling wins over a deprecated one. ``CONFIG_FILE`` is never consulted.

    Args:
        config_file: Explicit INI file, or ``None``.
        params_file: Explicit JSON file, or ``None``. Pass at most one source.
        overrides: Invocation values, flat or grouped. Relative paths resolve
            from the working directory.
        phase_groups: Owner groups a saved phase uses. Overrides for other groups
            are unsuitable, only these groups are resolved, and repeated data
            facts belong to ``model`` when ``data`` is inactive.

    Returns:
        Values keyed by every owner group (``data``, ``model``, ``sampler``,
        ``output``); groups outside ``phase_groups`` are empty.

    Raises:
        ValueError: If both sources are given, the source is not a mapping, or any
            name is unknown or unsuitable for the phase. All such names are
            reported together with migration guidance.
    """
    if config_file is not None and params_file is not None:
        raise ValueError("Pass at most one of `config_file` or `params_file`.")
    source = Path(config_file if config_file is not None else params_file).resolve() if (
        config_file is not None or params_file is not None) else None
    params: Any = {}
    if config_file is not None:
        params = read_rhime_ini(source)
    elif source is not None:
        params = json.loads(source.read_text(encoding="utf-8"))
    owners = rhime_option_groups()
    active = set(owners) if phase_groups is None else set(phase_groups)
    claims_owner = "data" if "data" in active else "model"
    groups: dict[str, dict[str, Any]] = {group: {} for group in owners}
    unsuitable = []
    # Classify names only: values of inactive groups are never resolved.
    for values, is_override in ((params, False), (overrides or {}, True)):
        if not isinstance(values, Mapping):
            raise ValueError(f"RHIME parameters must be a mapping; got {type(values).__name__}.")
        for key, value in values.items():
            if key in owners and not isinstance(value, Mapping):
                raise ValueError(f"`{key}` must be a mapping.")
            for name, item in (value.items() if key in owners else [(key, value)]):
                canonical = REQUEST_ALIASES.get(name, name)
                if key in owners:
                    owner = key
                elif name == "use_tracer":
                    owner = "data"
                elif canonical in REQUEST_CLAIM_NAMES:
                    owner = claims_owner
                else:
                    owner = next((group for group, names in owners.items() if canonical in names), None)
                known = owner is not None and (canonical in owners[owner] or name == "use_tracer")
                if not known or (is_override and owner not in active):
                    unsuitable.append(f"{key}.{name}" if key in owners else name)
                    continue
                groups[owner][name] = item
    if unsuitable:
        where = "" if phase_groups is None else " for this phase"
        raise ValueError(
            " ".join([f"Unsupported RHIME parameter(s){where}: {sorted(unsuitable)!r}.",
                      *migration_errors(name.rsplit(".", 1)[-1] for name in unsuitable)])
        )
    annotations = get_type_hints(DataConfig) | get_type_hints(RhimeOutputSpec)
    result: dict[str, dict[str, Any]] = {group: {} for group in owners}
    for group in active:
        result[group] = resolve_resource_paths(
            translate_rhime_aliases(groups[group]),
            annotations=annotations, base_dir=Path.cwd(),
        )
    return result
