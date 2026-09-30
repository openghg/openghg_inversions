"""Compatibility imports; implementation lives in :mod:`openghg_inversions.recipes._stage_artifacts`."""

from openghg_inversions.recipes._stage_artifacts import (
    artifact_path as artifact_path,
    file_identity as file_identity,
    json_value as json_value,
    write_json as write_json,
)

__all__ = ['artifact_path', 'file_identity', 'json_value', 'write_json']
