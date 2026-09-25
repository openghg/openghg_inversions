"""Versioned affine reconstruction artifact round trips and trust boundary."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.basis.affine_flux_map import AffineFluxMap
from openghg_inversions.basis.affine_flux_map_io import (
    AffineFluxMapArtifact,
    from_datatree,
    load,
    save,
    to_datatree,
)
from openghg_inversions.basis.operators import BucketBasisOperator, MultiSourceBucketBasisOperator


def _artifact(*, representation: str = "bucket", multisource: bool = False) -> AffineFluxMapArtifact:
    coords = {"lat": [50.0, 51.0], "lon": [-2.0, -1.0]}
    mean = xr.DataArray(
        [[0.5, 1.2], [0.9, 1.6]],
        dims=("lat", "lon"),
        coords=coords,
        attrs={"units": "1", "long_name": "Native prior mean"},
    )
    if multisource:
        mean = xr.concat(
            [mean, mean + 0.3], dim=xr.IndexVariable("native_source", ["fossil", "bio"])
        ).assign_attrs(units="1")
        operator = MultiSourceBucketBasisOperator(
            {
                "fossil": xr.DataArray([[1, 1], [2, 2]], dims=("lat", "lon"), coords=coords),
                "bio": xr.DataArray([[1, 2], [1, 2]], dims=("lat", "lon"), coords=coords),
            }
        )
    else:
        operator = BucketBasisOperator(xr.DataArray([[1, 1], [2, 2]], dims=("lat", "lon"), coords=coords))
    flux = (mean * -2).assign_attrs(units="kg m-2 s-1", long_name="Signed flux")
    if representation == "bucket":
        prolongation = operator
    elif multisource:
        explicit = np.zeros((2, 2, 2, 4))
        explicit[0, :, :, :2] = np.equal([[1, 1], [2, 2]], np.array([1, 2])[:, None, None]).transpose(1, 2, 0)
        explicit[1, :, :, 2:] = np.equal([[1, 2], [1, 2]], np.array([1, 2])[:, None, None]).transpose(1, 2, 0)
        prolongation = xr.DataArray(
            explicit,
            dims=(*mean.dims, "state"),
            coords={**mean.coords, "state": operator.basis_matrix.state},
            attrs={"units": "1"},
        )
    else:
        prolongation = operator.basis_matrix.transpose("lat", "lon", "state").assign_attrs(units="1")
    return AffineFluxMapArtifact(
        AffineFluxMap(mean, flux, prolongation, state_dim="state"),
        prepared_inputs_id="sha256:example-prepared-content",
        projection_provenance={"projection_strategy": "bucket"},
        reconstruction_provenance={"producer": "test", "sequence": [1, True, None]},
        source_provenance={"native_sources": ["fossil", "bio"]} if multisource else {},
    )


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
@pytest.mark.parametrize("representation", ["bucket", "explicit"])
def test_roundtrip_preserves_separate_ingredients_and_reconstruction(
    tmp_path: Path,
    suffix: str,
    representation: str,
) -> None:
    original = _artifact(representation=representation)
    tree = to_datatree(original)
    assert set(tree.children) == {"native_mean", "flux", "prolongation"}
    assert "reference_state" not in str(tree)
    assert "country" not in str(tree)
    assert "state_to_flux" not in str(tree)
    path = tmp_path / f"affine{suffix}"
    save(original, path)
    restored = load(path)

    assert restored.affine_map.representation == representation
    assert restored.prepared_inputs_id == original.prepared_inputs_id
    assert dict(restored.projection_provenance) == dict(original.projection_provenance)
    assert dict(restored.reconstruction_provenance) == dict(original.reconstruction_provenance)
    assert dict(restored.source_provenance) == dict(original.source_provenance)
    xr.testing.assert_equal(restored.affine_map.native_mean, original.affine_map.native_mean)
    xr.testing.assert_equal(restored.affine_map.flux, original.affine_map.flux)
    state_index = (
        original.affine_map.prolongation.indexes["state"]
        if representation == "explicit"
        else original.affine_map.prolongation.basis_matrix.indexes["state"]
    )
    reference = xr.DataArray([0.4, 1.7], dims="state", coords={"state": state_index}, attrs={"units": "1"})
    state = xr.DataArray(
        [[0.9, 1.3], [0.2, 2.1]],
        dims=("draw", "state"),
        coords={"draw": [0, 1], "state": state_index},
        attrs={"units": "1"},
    )
    xr.testing.assert_allclose(
        restored.affine_map.state_to_native(state, reference_state=reference),
        original.affine_map.state_to_native(state, reference_state=reference),
    )
    xr.testing.assert_allclose(
        restored.affine_map.state_to_flux(state, reference_state=reference),
        original.affine_map.state_to_flux(state, reference_state=reference),
    )


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
@pytest.mark.parametrize("representation", ["bucket", "explicit"])
def test_gathered_state_and_native_source_order_roundtrip(
    tmp_path: Path,
    suffix: str,
    representation: str,
) -> None:
    original = _artifact(representation=representation, multisource=True)
    path = tmp_path / f"gathered{suffix}"
    save(original, path)
    restored = load(path)
    assert restored.affine_map.native_mean.native_source.values.tolist() == ["fossil", "bio"]
    if representation == "bucket":
        assert "native_source" not in restored.affine_map.prolongation.basis_matrix.dims
    state_index = (
        original.affine_map.prolongation.indexes["state"]
        if representation == "explicit"
        else original.affine_map.prolongation.basis_matrix.indexes["state"]
    )
    restored_index = (
        restored.affine_map.prolongation.indexes["state"]
        if representation == "explicit"
        else restored.affine_map.prolongation.basis_matrix.indexes["state"]
    )
    assert isinstance(restored_index, pd.MultiIndex)
    assert restored_index.equals(state_index)
    assert restored_index.names == state_index.names
    assert "source_state" not in restored.affine_map.native_mean.dims
    reference = xr.DataArray(
        [0.2, 1.4, 0.5, 1.8], dims="state", coords={"state": state_index}, attrs={"units": "1"}
    )
    state = (reference + xr.DataArray([0.1, -0.2, 0.3, 0.4], dims="state")).assign_attrs(units="1")
    np.testing.assert_allclose(
        restored.affine_map.state_to_flux(state, reference_state=reference),
        original.affine_map.state_to_flux(state, reference_state=reference),
    )


@pytest.mark.parametrize(
    ("offending", "group"),
    [
        ("fp_x_flux", "flux"),
        ("Pi", "prolongation"),
        ("native_B", "native_mean"),
        ("native_covariance", "native_mean"),
        ("state_to_flux", "flux"),
        ("country_by_state", "prolongation"),
        ("quantity_residual_covariance", "flux"),
        ("reporting_sector_mapping", "native_mean"),
    ],
)
def test_rejects_prohibited_payload_variables(offending: str, group: str) -> None:
    tree = to_datatree(_artifact(representation="explicit"))
    ds = tree[group].to_dataset()
    ds[offending] = xr.DataArray(1)
    tree[group] = xr.DataTree(ds)
    with pytest.raises(ValueError, match=offending):
        from_datatree(tree)


@pytest.mark.parametrize(
    "offending",
    [
        "fp_x_flux",
        "Pi",
        "native_B",
        "native_covariance",
        "state_to_flux",
        "country_map",
        "residual_block",
        "reporting_sector_mapping",
    ],
)
def test_rejects_prohibited_payload_groups_and_attributes(offending: str) -> None:
    tree = to_datatree(_artifact())
    tree[offending] = xr.DataTree(xr.Dataset())
    with pytest.raises(ValueError, match=offending):
        from_datatree(tree)
    tree = to_datatree(_artifact())
    tree.attrs[offending] = "unexpected"
    with pytest.raises(ValueError, match=offending):
        from_datatree(tree)


@pytest.mark.parametrize("offending", ["fp_x_flux", "Pi", "country_map", "reporting_sector_mapping"])
def test_rejects_prohibited_payload_coordinates(offending: str) -> None:
    tree = to_datatree(_artifact())
    tree["flux"] = xr.DataTree(tree["flux"].to_dataset().assign_coords({offending: 1}))
    with pytest.raises(ValueError, match=offending):
        from_datatree(tree)


def test_rejects_missing_corrupt_and_unsupported_payload_before_map_construction() -> None:
    tree = to_datatree(_artifact())
    del tree["flux"]
    with pytest.raises(ValueError, match="flux"):
        from_datatree(tree)
    tree = to_datatree(_artifact())
    tree.attrs["schema_version"] = 99
    with pytest.raises(ValueError, match="schema"):
        from_datatree(tree)
    tree = to_datatree(_artifact())
    tree.attrs["prepared_inputs_id"] = ""
    with pytest.raises(ValueError, match="prepared_inputs_id"):
        from_datatree(tree)
    tree = to_datatree(_artifact())
    tree.attrs["projection_provenance"] = "{broken"
    with pytest.raises(ValueError, match="projection_provenance"):
        from_datatree(tree)
    tree = to_datatree(_artifact(representation="explicit"))
    tree["prolongation"] = xr.DataTree(xr.Dataset())
    with pytest.raises(ValueError, match="prolongation"):
        from_datatree(tree)


def test_rejects_non_json_provenance_and_missing_identity() -> None:
    value = _artifact().affine_map
    with pytest.raises(ValueError, match="prepared_inputs_id"):
        AffineFluxMapArtifact(value, "", {}, {})
    with pytest.raises(ValueError, match="projection_provenance"):
        AffineFluxMapArtifact(value, "id", {"bad": np.nan}, {})
    with pytest.raises(ValueError, match="reporting_sector_mapping"):
        AffineFluxMapArtifact(value, "id", {}, {"nested": {"reporting_sector_mapping": {}}})


def test_serialization_prunes_ancillary_attrs_without_mutating_inputs() -> None:
    artifact = _artifact()
    operator = artifact.affine_map.prolongation
    operator.basis_flat.attrs["long_name"] = "Bucket IDs"
    tree = to_datatree(artifact)
    assert tree["prolongation"].to_dataset()["basis_flat"].attrs == {}
    assert operator.basis_flat.attrs["long_name"] == "Bucket IDs"
    assert artifact.affine_map.flux.attrs["long_name"] == "Signed flux"
    assert tree["flux"].to_dataset()["flux"].attrs == {"units": "kg m-2 s-1"}
