"""Checkpoint writers must not replace stores that own their inputs."""

from pathlib import Path

import pytest

from openghg_inversions.rhime._stage_artifacts import check_checkpoint_targets


@pytest.mark.parametrize("direction", ["inside_input", "contains_input"])
def test_checkpoint_rejects_overlapping_store_paths(tmp_path: Path, direction: str) -> None:
    store = tmp_path / "prepared.zarr"
    store.mkdir()
    payload = store / "arrays"
    payload.write_bytes(b"original prepared data")
    destination, target, referenced = (
        (store, "posterior.nc", store)
        if direction == "inside_input"
        else (tmp_path, store.name, payload)
    )

    with pytest.raises(ValueError, match="would overwrite"):
        check_checkpoint_targets(destination, stage=None, payloads=(target,), referenced=(referenced,))

    assert payload.read_bytes() == b"original prepared data"


@pytest.mark.parametrize("alias", ["same_path", "hardlink", "symlink"])
def test_checkpoint_rejects_existing_input_alias(tmp_path: Path, alias: str) -> None:
    original = tmp_path / "prepared.nc"
    original.write_bytes(b"original prepared data")
    target = tmp_path / "posterior.nc"
    if alias == "hardlink":
        target.hardlink_to(original)
    elif alias == "symlink":
        target.symlink_to(original)
    else:
        target = original

    with pytest.raises(ValueError, match="would overwrite"):
        check_checkpoint_targets(tmp_path, stage=None, payloads=(target.name,), referenced=(original,))

    assert original.read_bytes() == b"original prepared data"


def test_checkpoint_accepts_separate_outputs_beside_input_store(tmp_path: Path) -> None:
    store = tmp_path / "prepared.zarr"
    store.mkdir()
    payload = store / "arrays"
    payload.write_bytes(b"original prepared data")

    check_checkpoint_targets(tmp_path, stage="sample", payloads=("posterior.nc",), referenced=(store,))

    assert payload.read_bytes() == b"original prepared data"
    assert not (tmp_path / "posterior.nc").exists()
