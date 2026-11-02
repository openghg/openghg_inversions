"""Disk regressions for output metadata meaning and malformed envelopes."""

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.serialization import save_datatree
from tests.helpers import make_trace


@pytest.fixture
def output(fake_multisector_basis_functions):
    return InversionOutput(
        trace=make_trace(posterior=xr.Dataset({"x": (("chain", "draw"), [[1.0]])})),
        inv_inputs=xr.Dataset(),
        basis_functions=fake_multisector_basis_functions(),
    )


@pytest.mark.parametrize("suffix", ["nc", "zarr"])
def test_metadata_semantics_roundtrip(output, tmp_path, suffix):
    output.run_metadata = {"split_by_sectors": np.bool_(False), "count": np.int64(3)}
    array = np.array([False, True])
    output.model_metadata = {"nested": {"flags": array, "weights": np.array([1.5, 2.5])}}
    path = tmp_path / f"output.{suffix}"
    output.save(path)
    restored = InversionOutput.load(path)
    assert restored.run_metadata["split_by_sectors"] is False
    assert restored.is_multisector is False
    assert type(restored.run_metadata["count"]) is int
    assert restored.model_metadata == {"nested": {"flags": [False, True], "weights": [1.5, 2.5]}}
    assert output.model_metadata["nested"]["flags"] is array
    assert isinstance(output.run_metadata["split_by_sectors"], np.bool_)


@pytest.mark.parametrize("suffix", ["nc", "zarr"])
@pytest.mark.parametrize("raw", ["not JSON", "[]", '{"value": NaN}', 3])
def test_malformed_present_metadata_rejected(output, tmp_path, suffix, raw):
    tree = output.to_datatree()
    tree.attrs["run_metadata"] = raw
    path = tmp_path / f"malformed.{suffix}"
    save_datatree(tree, path)
    with pytest.raises(ValueError, match="run_metadata"):
        InversionOutput.load(path)


@pytest.mark.parametrize("suffix", ["nc", "zarr"])
@pytest.mark.parametrize("version", [None, 2, "1"])
def test_invalid_artifact_version_rejected(output, tmp_path, suffix, version):
    tree = output.to_datatree()
    if version is None:
        del tree.attrs["schema_version"]
    else:
        tree.attrs["schema_version"] = version
    path = tmp_path / f"version.{suffix}"
    save_datatree(tree, path)
    with pytest.raises(ValueError, match="schema_version"):
        InversionOutput.load(path)


def test_absent_metadata_is_optional(output):
    tree = output.to_datatree()
    del tree.attrs["run_metadata"]
    assert InversionOutput.from_datatree(tree).run_metadata == {}


def test_unsupported_metadata_rejected_without_stringification(output):
    output.provenance = {"unsupported": object()}
    with pytest.raises(TypeError, match="object"):
        output.to_datatree()
