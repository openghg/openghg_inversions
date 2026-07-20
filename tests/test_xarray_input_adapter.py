"""Focused tests for source-neutral xarray RHIME input adaptation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr

from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.inversion_data import prepare_rhime_inputs_from_xarray
from openghg_inversions.models import components as model_components
from openghg_inversions.models.coords import CoordRegistry, attach_coord_registry
from openghg_inversions.models.rhime import RhimeModelSpec, SectorSpec, build_rhime_model
from openghg_inversions.rhime import (
    RhimeOutputSpec,
    RhimeRunSpec,
    RhimeSampler,
    run_rhime_from_prepared_inputs,
)
from openghg_inversions.rhime.outputs import RhimeOutputBundle
from openghg_inversions.sigma import SigmaAlignment


def _basis_functions(source_order: list[str] | None = None) -> BasisFunctions:
    """Create a two-cell self-contained basis with distinctive retained flux."""
    basis = xr.DataArray(
        [[1], [2]],
        dims=("lat", "lon"),
        coords={"lat": [50.0, 51.0], "lon": [-2.0]},
        name="basis",
    )
    flux = xr.DataArray(
        [[2.0], [3.0]],
        dims=("lat", "lon"),
        coords=basis.coords,
        name="flux",
        attrs={"units": "mol m-2 s-1"},
    )
    if source_order is not None:
        flux = xr.concat(
            [flux.expand_dims(source=[source]) * (index + 1) for index, source in enumerate(source_order)],
            dim="source",
        )
    return BasisFunctions.from_flat_basis(
        basis,
        flux,
        operator_kwargs={"state_dim": "region"},
    )


def _unequal_source_basis_functions() -> BasisFunctions:
    """Create source-specific bases with two ocean and one fossil region."""
    coords = {"lat": [50.0, 51.0], "lon": [-2.0, -1.0]}
    ocean_basis = xr.DataArray(
        [[1, 1], [2, 2]],
        dims=("lat", "lon"),
        coords=coords,
        name="basis",
    )
    fossil_basis = xr.DataArray(
        [[1, 1], [1, 1]],
        dims=("lat", "lon"),
        coords=coords,
        name="basis",
    )
    flux = xr.concat(
        [
            xr.full_like(ocean_basis, 2.0, dtype=float).expand_dims(source=["ocean-inventory"]),
            xr.full_like(fossil_basis, 3.0, dtype=float).expand_dims(source=["ff-inventory"]),
        ],
        dim="source",
    ).rename("flux")
    flux.attrs["units"] = "mol m-2 s-1"
    return BasisFunctions.from_multi_source_flat_basis(
        {"ocean-inventory": ocean_basis, "ff-inventory": fossil_basis},
        flux,
        operator_kwargs={"state_dim": "region"},
    )


def _site_dataset(
    site: str,
    times: list[str],
    *,
    cache_name: str = "fp_x_flux",
    source_order: list[str] | None = None,
    fixed_baseline: bool = False,
    sampled_baseline: bool = False,
) -> xr.Dataset:
    """Build a minimal site-time dataset following the adapter contract."""
    time = np.asarray(times, dtype="datetime64[ns]")
    base = np.arange(1, 2 * len(time) + 1, dtype=float).reshape(2, 1, len(time))
    if source_order is None:
        cache = xr.DataArray(
            base,
            dims=("lat", "lon", "time"),
            coords={"lat": [50.0, 51.0], "lon": [-2.0], "time": time},
            name=cache_name,
        )
    else:
        cache = xr.DataArray(
            np.stack([base * (index + 1) for index in range(len(source_order))]),
            dims=("source", "lat", "lon", "time"),
            coords={
                "source": source_order,
                "lat": [50.0, 51.0],
                "lon": [-2.0],
                "time": time,
            },
            name=cache_name,
        )

    cache.attrs["units"] = "ppm"
    obs = xr.DataArray(np.arange(len(time), dtype=float) + 400.0, dims="time", coords={"time": time})
    dataset = xr.Dataset(
        {
            cache_name: cache,
            "mf": obs,
            "mf_error": xr.ones_like(obs),
            "mf_repeatability": xr.ones_like(obs) * 0.5,
            "mf_variability": xr.ones_like(obs) * 0.75,
            "release_lat": xr.DataArray(50.0 + len(site) / 100),
            "release_lon": xr.DataArray(-2.0),
        }
    )
    for name in ("mf", "mf_error", "mf_repeatability", "mf_variability"):
        dataset[name].attrs["units"] = "ppm"
    if fixed_baseline:
        dataset["fixed_baseline"] = xr.full_like(obs, 390.0)
        dataset["fixed_baseline"].attrs["units"] = "ppm"
    if sampled_baseline:
        dataset["H_bc"] = xr.DataArray(
            np.full((2, len(time)), 0.5),
            dims=("bc_region", "time"),
            coords={"bc_region": ["north", "south"], "time": time},
            attrs={"units": "ppm"},
        )
    return dataset


def _unequal_source_site_dataset(site: str, times: list[str]) -> xr.Dataset:
    """Create a site cache compatible with the unequal source-specific basis."""
    source_order = ["ocean-inventory", "ff-inventory"]
    dataset = _site_dataset(
        site,
        times,
        cache_name="fp_x_flux_sectoral",
        source_order=source_order,
        fixed_baseline=True,
        sampled_baseline=True,
    ).drop_dims(("lat", "lon"))
    time = np.asarray(times, dtype="datetime64[ns]")
    values = np.arange(1, 2 * 2 * 2 * len(time) + 1, dtype=float).reshape(2, 2, 2, len(time))
    dataset["fp_x_flux_sectoral"] = xr.DataArray(
        values,
        dims=("source", "lat", "lon", "time"),
        coords={
            "source": source_order,
            "lat": [50.0, 51.0],
            "lon": [-2.0, -1.0],
            "time": time,
        },
        attrs={"units": "ppm"},
    )
    dataset["quality_flag"] = xr.DataArray(
        np.arange(len(time), dtype=np.int16) + 1,
        dims="time",
        coords={"time": time},
    )
    return dataset


def _canonical_site_data(
    data: Mapping[str, xr.Dataset], basis_functions: BasisFunctions
) -> dict[str, xr.Dataset]:
    """Replace each test cache with the equivalent canonical sensitivity."""
    canonical: dict[str, xr.Dataset] = {}
    for site, dataset in data.items():
        cache_name = "fp_x_flux_sectoral" if "fp_x_flux_sectoral" in dataset else "fp_x_flux"
        site_data = dataset.copy(deep=False)
        site_data["H"] = basis_functions.sensitivity(site_data[cache_name]).rename("H")
        site_data["H"].attrs["units"] = site_data[cache_name].attrs["units"]
        canonical[site] = site_data
    return canonical


def test_mapping_and_datatree_inputs_are_equivalent() -> None:
    """Ordered per-site mappings and equivalent DataTrees produce identical inputs."""
    basis_functions = _basis_functions()
    per_site = {site: _site_dataset(site, ["2021-01-01", "2021-01-02"]) for site in ("MHD", "TAC")}
    tree = xr.DataTree.from_dict(per_site)

    from_mapping = prepare_rhime_inputs_from_xarray(
        per_site,
        basis_functions=basis_functions,
        averaging_period="1h",
    )
    from_tree = prepare_rhime_inputs_from_xarray(
        tree,
        basis_functions=basis_functions,
        averaging_period="1h",
    )

    xr.testing.assert_identical(from_mapping.inv_inputs, from_tree.inv_inputs)
    assert from_mapping.sites == from_tree.sites == ("MHD", "TAC")


@pytest.mark.parametrize("layout", ["site-local", "dense", "stacked"])
def test_direct_dataset_inputs_are_rejected(layout: str) -> None:
    """Every direct Dataset layout is outside the mapping/DataTree contract."""
    dataset = _site_dataset("TAC", ["2021-01-01"])
    if layout == "dense":
        dataset = dataset.expand_dims(site=["TAC"])
    elif layout == "stacked":
        dataset = dataset.stack(nmeasure=("time",))

    with pytest.raises(TypeError, match="mapping|DataTree"):
        prepare_rhime_inputs_from_xarray(dataset, basis_functions=_basis_functions())  # type: ignore[arg-type]


def test_nested_datatree_is_rejected() -> None:
    """A DataTree must expose each site as a direct child Dataset."""
    tree = xr.DataTree.from_dict({"group/TAC": _site_dataset("TAC", ["2021-01-01"])})

    with pytest.raises(ValueError, match="direct child|nested"):
        prepare_rhime_inputs_from_xarray(tree, basis_functions=_basis_functions())


@pytest.mark.parametrize("site", [1, b"TAC", ""], ids=["numeric", "bytes", "empty"])
def test_mapping_site_identifiers_must_be_nonempty_strings(site: object) -> None:
    """Mapping keys are validated as site identifiers without string coercion."""
    data = {site: _site_dataset("TAC", ["2021-01-01"])}

    with pytest.raises((TypeError, ValueError), match="site|Site"):
        prepare_rhime_inputs_from_xarray(
            data,  # type: ignore[arg-type]
            basis_functions=_basis_functions(),
        )


def test_requested_sites_must_be_unique_strings() -> None:
    """Explicit site selection rejects duplicate identifiers before gathering."""
    data = {"TAC": _site_dataset("TAC", ["2021-01-01"])}

    with pytest.raises(ValueError, match="unique|duplicate"):
        prepare_rhime_inputs_from_xarray(
            data,
            sites=["TAC", "TAC"],
            basis_functions=_basis_functions(),
        )


@pytest.mark.parametrize("site", [1, b"TAC", ""], ids=["numeric", "bytes", "empty"])
def test_requested_site_identifiers_are_not_coerced(site: object) -> None:
    """Explicit selection rejects non-string and empty site identifiers."""
    data = {"TAC": _site_dataset("TAC", ["2021-01-01"])}

    with pytest.raises((TypeError, ValueError), match="site|Site"):
        prepare_rhime_inputs_from_xarray(
            data,
            sites=[site],  # type: ignore[list-item]
            basis_functions=_basis_functions(),
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda ds: ds.drop_vars("time"),
        lambda ds: ds.isel(time=slice(0, 0)),
        lambda ds: ds.assign_coords(time=np.asarray(["2021-01-01"] * 2, dtype="datetime64[ns]")),
        lambda ds: ds.assign_coords(time=["first", "second"]),
        lambda ds: ds.assign_coords(time=np.asarray(["2021-01-01", "NaT"], dtype="datetime64[ns]")),
    ],
    ids=["missing", "empty", "duplicate", "non-datetime", "NaT"],
)
def test_invalid_time_coordinates_are_rejected(mutate: Callable[[xr.Dataset], xr.Dataset]) -> None:
    """Site time coordinates must be explicit, nonempty, unique datetimes without NaT."""
    site_data = _site_dataset("TAC", ["2021-01-01", "2021-01-02"])
    invalid = mutate(site_data)

    with pytest.raises(ValueError, match="time"):
        prepare_rhime_inputs_from_xarray({"TAC": invalid}, basis_functions=_basis_functions())


def test_nonmonotonic_times_and_mapping_order_are_preserved() -> None:
    """Valid nonmonotonic site times retain both their order and mapping site order."""
    data = {
        "TAC": _site_dataset("TAC", ["2021-01-03", "2021-01-01"]),
        "MHD": _site_dataset("MHD", ["2021-01-04", "2021-01-02"]),
    }

    prepared = prepare_rhime_inputs_from_xarray(data, basis_functions=_basis_functions())

    assert prepared.sites == ("TAC", "MHD")
    assert list(prepared.inv_inputs.coords["site"].values) == ["TAC", "TAC", "MHD", "MHD"]
    np.testing.assert_array_equal(
        prepared.inv_inputs.coords["time"].values,
        np.asarray(
            ["2021-01-03", "2021-01-01", "2021-01-04", "2021-01-02"],
            dtype="datetime64[ns]",
        ),
    )


def test_stationary_release_coordinates_are_broadcast_to_observations() -> None:
    """Paired scalar and singleton station locations become nmeasure coordinates."""
    tac = _site_dataset("TAC", ["2021-01-01", "2021-01-02"])
    tac["release_lat"] = xr.DataArray(51.0)
    tac["release_lon"] = xr.DataArray(-2.0)
    mhd = _site_dataset("MHD", ["2021-01-03", "2021-01-04"])
    mhd["release_lat"] = xr.DataArray([52.0], dims="release_point")
    mhd["release_lon"] = xr.DataArray([-1.0], dims="release_point")

    prepared = prepare_rhime_inputs_from_xarray(
        {"TAC": tac, "MHD": mhd},
        basis_functions=_basis_functions(),
    )

    assert prepared.inv_inputs.release_lat.dims == ("nmeasure",)
    assert prepared.inv_inputs.release_lon.dims == ("nmeasure",)
    np.testing.assert_array_equal(prepared.inv_inputs.release_lat, [51.0, 51.0, 52.0, 52.0])
    np.testing.assert_array_equal(prepared.inv_inputs.release_lon, [-2.0, -2.0, -1.0, -1.0])


def test_mobile_release_coordinates_remain_time_aligned() -> None:
    """Paired mobile locations retain their exact per-observation values."""
    tac = _site_dataset("TAC", ["2021-01-01", "2021-01-02"])
    tac["release_lat"] = xr.DataArray([51.0, 51.5], dims="time", coords={"time": tac.time})
    tac["release_lon"] = xr.DataArray([-2.0, -1.5], dims="time", coords={"time": tac.time})
    mhd = _site_dataset("MHD", ["2021-01-03"])
    mhd["release_lat"] = xr.DataArray([52.0], dims="time", coords={"time": mhd.time})
    mhd["release_lon"] = xr.DataArray([-1.0], dims="time", coords={"time": mhd.time})

    prepared = prepare_rhime_inputs_from_xarray(
        {"TAC": tac, "MHD": mhd},
        basis_functions=_basis_functions(),
    )

    np.testing.assert_array_equal(prepared.inv_inputs.release_lat, [51.0, 51.5, 52.0])
    np.testing.assert_array_equal(prepared.inv_inputs.release_lon, [-2.0, -1.5, -1.0])


@pytest.mark.parametrize(
    "case",
    ["unpaired", "missing-site", "mixed-layout", "arbitrary-dimension", "nonfinite"],
)
def test_invalid_release_coordinate_pairs_are_rejected(case: str) -> None:
    """Release locations reject incomplete, mismatched, arbitrary, and nonfinite data."""
    tac = _site_dataset("TAC", ["2021-01-01", "2021-01-02"])
    mhd = _site_dataset("MHD", ["2021-01-03"])
    if case == "unpaired":
        tac = tac.drop_vars("release_lon")
    elif case == "missing-site":
        mhd = mhd.drop_vars(("release_lat", "release_lon"))
    elif case == "mixed-layout":
        tac["release_lat"] = xr.DataArray([51.0, 51.5], dims="time", coords={"time": tac.time})
        tac["release_lon"] = xr.DataArray(-2.0)
    elif case == "arbitrary-dimension":
        tac["release_lat"] = xr.DataArray([51.0, 51.5], dims="track")
        tac["release_lon"] = xr.DataArray([-2.0, -1.5], dims="track")
    else:
        tac["release_lat"] = xr.DataArray(np.inf)

    with pytest.raises(ValueError, match="release|coordinate|location"):
        prepare_rhime_inputs_from_xarray(
            {"TAC": tac, "MHD": mhd},
            basis_functions=_basis_functions(),
        )


@pytest.mark.parametrize(
    "variable",
    [
        "mf",
        "mf_error",
        "mf_repeatability",
        "mf_variability",
        "fp_x_flux",
        "fixed_baseline",
        "H_bc",
    ],
)
def test_every_concentration_field_requires_units(variable: str) -> None:
    """Every present concentration-valued field must declare a nonempty unit string."""
    site_data = _site_dataset(
        "TAC",
        ["2021-01-01"],
        fixed_baseline=True,
        sampled_baseline=True,
    )
    site_data[variable].attrs.pop("units")

    with pytest.raises(ValueError, match=rf"{variable}.*units|units.*{variable}"):
        prepare_rhime_inputs_from_xarray({"TAC": site_data}, basis_functions=_basis_functions())


@pytest.mark.parametrize("invalid_units", ["", b"ppm", 1.0])
def test_mf_units_must_be_a_nonempty_string(invalid_units: object) -> None:
    """Observation units reject empty and non-string metadata."""
    site_data = _site_dataset("TAC", ["2021-01-01"])
    site_data["mf"].attrs["units"] = invalid_units

    with pytest.raises(ValueError, match="mf.*units|units.*mf"):
        prepare_rhime_inputs_from_xarray({"TAC": site_data}, basis_functions=_basis_functions())


@pytest.mark.parametrize("variable", ["mf_error", "fp_x_flux", "fixed_baseline", "H_bc"])
def test_concentration_units_must_match_mf_exactly(variable: str) -> None:
    """The adapter rejects exact-string unit mismatches instead of converting them."""
    site_data = _site_dataset(
        "TAC",
        ["2021-01-01"],
        fixed_baseline=True,
        sampled_baseline=True,
    )
    site_data[variable].attrs["units"] = "ppb"

    with pytest.raises(ValueError, match="units|convert"):
        prepare_rhime_inputs_from_xarray({"TAC": site_data}, basis_functions=_basis_functions())


def test_mf_units_must_match_across_sites() -> None:
    """All sites must use the same exact observation-unit string."""
    data = {
        "MHD": _site_dataset("MHD", ["2021-01-01"]),
        "TAC": _site_dataset("TAC", ["2021-01-02"]),
    }
    data["TAC"]["mf"].attrs["units"] = "ppb"

    with pytest.raises(ValueError, match="units"):
        prepare_rhime_inputs_from_xarray(data, basis_functions=_basis_functions())


def test_canonical_h_requires_units() -> None:
    """A directly supplied H field must declare the observation concentration units."""
    basis_functions = _basis_functions()
    site_data = _canonical_site_data(
        {"TAC": _site_dataset("TAC", ["2021-01-01"])},
        basis_functions,
    )["TAC"].drop_vars("fp_x_flux")
    site_data["H"].attrs.pop("units")

    with pytest.raises(ValueError, match="H.*units|units.*H"):
        prepare_rhime_inputs_from_xarray({"TAC": site_data}, basis_functions=basis_functions)


@pytest.mark.parametrize(
    ("label_kind", "source_order"),
    [
        ("numeric", [1, 2]),
        ("bytes", [b"ocean", b"fossil"]),
        ("duplicate", ["ocean", "ocean"]),
        ("empty", ["ocean", ""]),
    ],
)
def test_invalid_source_labels_are_rejected(
    label_kind: str,
    source_order: list[object],
) -> None:
    """Source labels must be nonempty unique Python or NumPy strings."""
    del label_kind
    site_data = _site_dataset(
        "TAC",
        ["2021-01-01"],
        cache_name="fp_x_flux_sectoral",
        source_order=source_order,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="source"):
        prepare_rhime_inputs_from_xarray(
            {"TAC": site_data},
            basis_functions=_basis_functions(source_order),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "variable",
    ["mf", "mf_error", "mf_repeatability", "mf_variability", "fixed_baseline", "H_bc"],
)
def test_nonfinite_required_values_are_rejected(variable: str) -> None:
    """Required observation and optional baseline values must be finite."""
    site_data = _site_dataset(
        "TAC",
        ["2021-01-01"],
        fixed_baseline=True,
        sampled_baseline=True,
    )
    site_data[variable].data.reshape(-1)[0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        prepare_rhime_inputs_from_xarray({"TAC": site_data}, basis_functions=_basis_functions())


def test_nonfinite_projected_h_is_rejected() -> None:
    """Canonical projected sensitivity must contain only finite active values."""
    basis_functions = _basis_functions()
    site_data = _canonical_site_data(
        {"TAC": _site_dataset("TAC", ["2021-01-01"])},
        basis_functions,
    )["TAC"].drop_vars("fp_x_flux")
    site_data["H"].data.reshape(-1)[0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        prepare_rhime_inputs_from_xarray({"TAC": site_data}, basis_functions=basis_functions)


def test_emptied_site_is_reported_after_gathering(monkeypatch: pytest.MonkeyPatch) -> None:
    """A site removed during gathering is named instead of leaving stale metadata."""
    gathered = xr.Dataset(
        {"mf": ("nmeasure", [400.0])},
        coords={
            "nmeasure": [0],
            "site": ("nmeasure", ["MHD"]),
            "time": ("nmeasure", np.asarray(["2021-01-01"], dtype="datetime64[ns]")),
        },
    ).set_index(nmeasure=["site", "time"])
    monkeypatch.setattr(
        "openghg_inversions.inversion_data.xarray_adapter.make_inv_inputs",
        lambda *args, **kwargs: gathered,
    )
    data = {
        "MHD": _site_dataset("MHD", ["2021-01-01"]),
        "TAC": _site_dataset("TAC", ["2021-01-02"]),
    }

    with pytest.raises(ValueError, match="TAC"):
        prepare_rhime_inputs_from_xarray(data, basis_functions=_basis_functions())


def test_min_error_per_site_defaults_to_false_and_allows_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omission matches existing RHIME preparation while True remains an override."""
    captured: list[bool] = []
    canonical = xr.Dataset(
        coords={
            "nmeasure": [0],
            "site": ("nmeasure", ["TAC"]),
            "time": ("nmeasure", np.asarray(["2021-01-01"], dtype="datetime64[ns]")),
        }
    ).set_index(nmeasure=["site", "time"])

    def capture_make_inv_inputs(*args: object, **kwargs: object) -> xr.Dataset:
        """Capture the forwarded per-site minimum-error flag."""
        captured.append(bool(kwargs["min_error_per_site"]))
        return canonical

    monkeypatch.setattr(
        "openghg_inversions.inversion_data.xarray_adapter.make_inv_inputs",
        capture_make_inv_inputs,
    )
    data = {"TAC": _site_dataset("TAC", ["2021-01-01"])}

    prepare_rhime_inputs_from_xarray(data, basis_functions=_basis_functions())
    prepare_rhime_inputs_from_xarray(
        data,
        basis_functions=_basis_functions(),
        min_error_per_site=False,
    )
    prepare_rhime_inputs_from_xarray(
        data,
        basis_functions=_basis_functions(),
        min_error_per_site=True,
    )

    assert captured == [False, False, True]


def test_cached_and_canonical_h_inputs_match_without_retaining_caches() -> None:
    """Projection drops known caches while retaining time-aligned extensions."""
    basis_functions = _basis_functions()
    cached = {
        "MHD": _site_dataset("MHD", ["2021-01-01", "2021-01-02"]),
        "TAC": _site_dataset("TAC", ["2021-01-03"]),
    }
    cached["MHD"]["quality_flag"] = xr.DataArray([1, 2], dims="time", coords={"time": cached["MHD"].time})
    cached["TAC"]["quality_flag"] = xr.DataArray([3], dims="time", coords={"time": cached["TAC"].time})
    canonical = _canonical_site_data(cached, basis_functions)
    for dataset in canonical.values():
        dataset["fp_x_flux_sectoral"] = dataset["fp_x_flux"] * 99.0

    from_cache = prepare_rhime_inputs_from_xarray(cached, basis_functions=basis_functions)
    from_h = prepare_rhime_inputs_from_xarray(canonical, basis_functions=basis_functions)

    xr.testing.assert_identical(from_cache.inv_inputs, from_h.inv_inputs)
    assert from_cache.inv_inputs.sizes["nmeasure"] == 3
    assert list(from_cache.inv_inputs.coords["site"].values) == ["MHD", "MHD", "TAC"]
    assert not {"fp_x_flux", "fp_x_flux_sectoral"} & set(from_cache.inv_inputs)
    assert not {"fp_x_flux", "fp_x_flux_sectoral"} & set(from_h.inv_inputs)
    np.testing.assert_array_equal(from_cache.inv_inputs["quality_flag"], [1, 2, 3])
    assert "fp_x_flux" in cached["MHD"]


def test_time_aligned_extensions_must_be_present_at_every_site() -> None:
    """Adapter gathering rejects a partial extension instead of dropping it."""
    site_data = {
        "MHD": _site_dataset("MHD", ["2021-01-01"]),
        "TAC": _site_dataset("TAC", ["2021-01-02"]),
    }
    site_data["MHD"]["quality_flag"] = xr.DataArray(
        [1],
        dims="time",
        coords={"time": site_data["MHD"].time},
    )

    with pytest.raises(ValueError, match="same data variables.*quality_flag"):
        prepare_rhime_inputs_from_xarray(site_data, basis_functions=_basis_functions())


def test_multisector_cache_preserves_source_order_and_retained_flux() -> None:
    """Sector projection preserves source order, operator, and retained flux."""
    source_order = ["ocean-inventory", "ff-inventory"]
    basis_functions = _basis_functions(source_order)
    site_data = {
        "TAC": _site_dataset(
            "TAC",
            ["2021-01-01", "2021-01-02"],
            cache_name="fp_x_flux_sectoral",
            source_order=source_order,
        )
    }

    prepared = prepare_rhime_inputs_from_xarray(site_data, basis_functions=basis_functions)

    assert prepared.basis_functions.operator is basis_functions.operator
    xr.testing.assert_identical(prepared.basis_functions.flux, basis_functions.flux)
    assert list(prepared.inv_inputs["H"].coords["source"].values) == source_order
    assert "fp_x_flux_sectoral" not in prepared.inv_inputs


def test_multisector_cache_rejects_total_retained_flux() -> None:
    """Multisector H cannot silently reuse one total prior flux for every source."""
    source_order = ["ocean-inventory", "ff-inventory"]
    site_data = {
        "TAC": _site_dataset(
            "TAC",
            ["2021-01-01"],
            cache_name="fp_x_flux_sectoral",
            source_order=source_order,
        )
    }

    with pytest.raises(ValueError, match="total flux cannot be reused"):
        prepare_rhime_inputs_from_xarray(site_data, basis_functions=_basis_functions())


@pytest.mark.parametrize(
    "retained_sources",
    [
        ["ocean-inventory", "different-inventory"],
        ["ff-inventory", "ocean-inventory"],
    ],
)
def test_multisector_cache_rejects_retained_flux_source_mismatch(
    retained_sources: list[str],
) -> None:
    """Retained prior flux names and order must exactly match sensitivity sources."""
    source_order = ["ocean-inventory", "ff-inventory"]
    site_data = {
        "TAC": _site_dataset(
            "TAC",
            ["2021-01-01"],
            cache_name="fp_x_flux_sectoral",
            source_order=source_order,
        )
    }

    with pytest.raises(ValueError, match="retained prior flux sources/order"):
        prepare_rhime_inputs_from_xarray(
            site_data,
            basis_functions=_basis_functions(retained_sources),
        )


def test_multisector_cache_rejects_retained_source_order_mismatch() -> None:
    """Source-specific retained semantics must use the same source order as H."""
    source_order = ["ocean-inventory", "ff-inventory"]
    shared = _basis_functions()
    basis = shared.flat_basis()
    assert isinstance(basis, xr.DataArray)
    source_specific = BasisFunctions.from_multi_source_flat_basis(
        {
            "ff-inventory": basis,
            "ocean-inventory": basis,
        },
        flux=_basis_functions(source_order).flux,
        operator_kwargs={"state_dim": "region"},
    )
    site_data = {
        "TAC": _site_dataset(
            "TAC",
            ["2021-01-01"],
            cache_name="fp_x_flux_sectoral",
            source_order=source_order,
        )
    }

    with pytest.raises(ValueError, match="sources/order"):
        prepare_rhime_inputs_from_xarray(site_data, basis_functions=source_specific)


def test_canonical_gathered_h_is_accepted_without_a_projection_cache() -> None:
    """Direct H retains the exact source-specific gathered state contract."""
    basis_functions = _unequal_source_basis_functions()
    cached = _unequal_source_site_dataset("TAC", ["2021-01-01"])
    sensitivity = basis_functions.sensitivity(cached["fp_x_flux_sectoral"]).rename("H")
    sensitivity.attrs["units"] = "ppm"
    direct_site = cached.drop_vars("fp_x_flux_sectoral").drop_dims(("source", "lat", "lon"))
    direct_site["H"] = sensitivity

    prepared = prepare_rhime_inputs_from_xarray(
        {"TAC": direct_site},
        basis_functions=basis_functions,
    )

    state_index = prepared.inv_inputs["H"].indexes["region"]
    assert isinstance(state_index, pd.MultiIndex)
    assert state_index.tolist() == [
        ("ocean-inventory", 0),
        ("ocean-inventory", 1),
        ("ff-inventory", 0),
    ]
    assert prepared.inv_inputs["H"].dims == ("region", "nmeasure")


def test_canonical_h_must_match_retained_basis_state_coordinate() -> None:
    """Canonical H with an unrelated state coordinate is rejected eagerly."""
    basis_functions = _basis_functions()
    site_data = _canonical_site_data(
        {"TAC": _site_dataset("TAC", ["2021-01-01"])},
        basis_functions,
    )["TAC"].drop_vars("fp_x_flux")
    unrelated_h = site_data["H"].assign_coords(region=[10, 11])
    site_data = site_data.drop_dims("region")
    site_data["H"] = unrelated_h

    with pytest.raises(ValueError, match="does not match the retained basis operator state coordinate"):
        prepare_rhime_inputs_from_xarray({"TAC": site_data}, basis_functions=basis_functions)


def test_canonical_h_requires_exact_dimension_order() -> None:
    """Canonical H must use the eager region-time dimension contract."""
    basis_functions = _basis_functions()
    site_data = _canonical_site_data(
        {"TAC": _site_dataset("TAC", ["2021-01-01"])},
        basis_functions,
    )["TAC"].drop_vars("fp_x_flux")
    site_data["H"] = site_data["H"].transpose("time", "region")

    with pytest.raises(ValueError, match="H must have exactly dimensions"):
        prepare_rhime_inputs_from_xarray({"TAC": site_data}, basis_functions=basis_functions)


def test_h_bc_requires_exact_dimensions() -> None:
    """Sampled baseline sensitivity rejects extra non-canonical dimensions."""
    basis_functions = _basis_functions()
    site_data = _site_dataset("TAC", ["2021-01-01"], sampled_baseline=True)
    site_data["H_bc"] = site_data["H_bc"].expand_dims(extra=[0])

    with pytest.raises(ValueError, match="H_bc must have exactly dimensions"):
        prepare_rhime_inputs_from_xarray({"TAC": site_data}, basis_functions=basis_functions)


@pytest.mark.parametrize(
    ("fixed_values", "fixed_units", "mf_units", "message"),
    [
        ([np.inf], None, None, "only finite values"),
        ([390.0], "ppm", "ppb", "do not match mf units"),
    ],
)
def test_fixed_baseline_rejects_nonfinite_values_and_unit_mismatch(
    fixed_values: list[float],
    fixed_units: str | None,
    mf_units: str | None,
    message: str,
) -> None:
    """Fixed baselines are finite and use declared observation units without conversion."""
    basis_functions = _basis_functions()
    site_data = _site_dataset("TAC", ["2021-01-01"], fixed_baseline=True)
    site_data["fixed_baseline"][:] = fixed_values
    if fixed_units is not None:
        site_data["fixed_baseline"].attrs["units"] = fixed_units
    if mf_units is not None:
        for name in ("mf", "mf_error", "mf_repeatability", "mf_variability"):
            site_data[name].attrs["units"] = mf_units

    with pytest.raises(ValueError, match=message):
        prepare_rhime_inputs_from_xarray({"TAC": site_data}, basis_functions=basis_functions)


def _likelihood_data(*, fixed: bool) -> xr.Dataset:
    """Create one-observation canonical data for baseline model-mean tests."""
    dataset = xr.Dataset(
        {
            "mf": ("nmeasure", np.asarray([10.0], dtype=np.float32)),
            "mf_error": ("nmeasure", np.asarray([1.0], dtype=np.float32)),
            "min_error": ("nmeasure", np.asarray([0.0], dtype=np.float32)),
            "site_indicator": ("nmeasure", [0]),
            "sigma_freq_index": ("nmeasure", [0]),
        },
        coords={"nmeasure": [0]},
    )
    if fixed:
        dataset["fixed_baseline"] = ("nmeasure", np.asarray([3.0], dtype=np.float32))
    return dataset


@pytest.mark.parametrize(
    ("fixed", "sampled", "expected_mean"),
    [(True, False, 4.0), (False, True, 3.0), (True, True, 6.0)],
)
def test_fixed_and_sampled_baselines_are_independent_additive_terms(
    monkeypatch: pytest.MonkeyPatch,
    fixed: bool,
    sampled: bool,
    expected_mean: float,
) -> None:
    """Fixed, sampled, and combined baselines add directly to the observation mean."""
    captured: dict[str, object] = {}

    def capture_normal(name: str, *args: object, **kwargs: object) -> None:
        """Capture the composed likelihood mean without creating a random variable."""
        captured["name"] = name
        captured["mu"] = kwargs["mu"]

    monkeypatch.setattr(model_components.pm, "Normal", capture_normal)
    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        model_components.add_inferpymc_likelihood_component(
            _likelihood_data(fixed=fixed),
            mu=pt.as_tensor_variable(np.asarray([1.0], dtype=np.float32)),
            mu_bc=(pt.as_tensor_variable(np.asarray([2.0], dtype=np.float32)) if sampled else None),
            sigprior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
            sigma_alignment=SigmaAlignment.from_frequency(_likelihood_data(fixed=fixed)["site_indicator"]),
            no_model_error=True,
        )

    assert captured["name"] == "y"
    np.testing.assert_allclose(captured["mu"].eval(), [expected_mean])  # type: ignore[union-attr]


def test_adapter_fixed_and_sampled_baselines_build_together() -> None:
    """Adapter output can build a model with both baseline modes and no fake sector."""
    basis_functions = _basis_functions()
    prepared = prepare_rhime_inputs_from_xarray(
        {
            "TAC": _site_dataset(
                "TAC",
                ["2021-01-01", "2021-01-02"],
                fixed_baseline=True,
                sampled_baseline=True,
            )
        },
        basis_functions=basis_functions,
    )

    model = build_rhime_model(
        prepared.inv_inputs,
        sigma_alignment=SigmaAlignment.from_frequency(prepared.inv_inputs["site_indicator"]),
        x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
        bc_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.1},
        sigma_prior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
        use_bc=True,
    )

    assert {"x", "bc", "mu_bc", "fixed_baseline", "y"}.issubset(model.named_vars)
    assert all("baseline" not in name for name in model.named_vars if name.startswith("x_"))


def test_adapter_output_executes_through_prepared_runner_without_openghg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adapted inputs execute through the prepared runner without data preparation."""
    basis_functions = _basis_functions()
    prepared = prepare_rhime_inputs_from_xarray(
        {"TAC": _site_dataset("TAC", ["2021-01-01"], fixed_baseline=True)},
        basis_functions=basis_functions,
        averaging_period="1h",
    )
    model_spec = RhimeModelSpec(
        species="co2",
        domain="EUROPE",
        sectors=(
            SectorSpec(
                name="total",
                flux_source="total-inventory",
                x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                variable_suffix="total",
            ),
        ),
        use_bc=False,
        sigma_prior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
    )
    run_spec = RhimeRunSpec(
        start_date="2021-01-01",
        end_date="2021-01-02",
        sites=("stale",),
        averaging_period=(None,),
        model=model_spec,
        output=RhimeOutputSpec(output_format="none", save_inversion_output=False),
    )

    def fail_prepare(**kwargs: object) -> None:
        """Fail if the prepared-input runner attempts data preparation."""
        raise AssertionError("The prepared runner must not call OpenGHG preparation.")

    monkeypatch.setattr("openghg_inversions.rhime.runner.prepare_rhime_inputs", fail_prepare)
    monkeypatch.setattr(RhimeSampler, "sample", lambda self, model: az.InferenceData())

    result = run_rhime_from_prepared_inputs(prepared_inputs=prepared, run_spec=run_spec)

    xr.testing.assert_identical(result.inv_inputs, prepared.inv_inputs)
    assert result.run_spec.sites == ("TAC",)
    assert result.model is not None
    assert "fixed_baseline" in result.model.named_vars


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_unequal_source_regions_round_trip_and_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    """A genuinely gathered multisector checkpoint survives storage and execution."""
    source_order = ["ocean-inventory", "ff-inventory"]
    basis_functions = _unequal_source_basis_functions()
    site_data = {
        "TAC": _unequal_source_site_dataset("TAC", ["2021-01-03", "2021-01-01"]),
        "MHD": _unequal_source_site_dataset("MHD", ["2021-01-04"]),
    }
    prepared = prepare_rhime_inputs_from_xarray(
        site_data,
        basis_functions=basis_functions,
        averaging_period={"TAC": "2h", "MHD": "1h"},
    )

    artifact = tmp_path / f"verification-games-inputs{suffix}"
    prepared.save(artifact)
    loaded = type(prepared).load(artifact)

    assert loaded.sites == ("TAC", "MHD")
    assert loaded.averaging_period == ("2h", "1h")
    assert loaded.inv_inputs.sizes["nmeasure"] == 3
    assert list(loaded.inv_inputs.coords["site"].values) == ["TAC", "TAC", "MHD"]
    np.testing.assert_array_equal(
        loaded.inv_inputs.coords["time"].values,
        np.asarray(["2021-01-03", "2021-01-01", "2021-01-04"], dtype="datetime64[ns]"),
    )
    region_index = loaded.inv_inputs["H"].indexes["region"]
    assert isinstance(region_index, pd.MultiIndex)
    assert region_index.names == ["source", "region_in_source"]
    assert region_index.tolist() == [
        ("ocean-inventory", 0),
        ("ocean-inventory", 1),
        ("ff-inventory", 0),
    ]
    assert loaded.inv_inputs["H"].dims == ("region", "nmeasure")
    assert loaded.inv_inputs["H"].attrs["units"] == "ppm"
    assert not {"fp_x_flux", "fp_x_flux_sectoral"} & set(loaded.inv_inputs)
    np.testing.assert_array_equal(loaded.inv_inputs["quality_flag"], [1, 2, 1])
    xr.testing.assert_identical(loaded.basis_functions.flux, basis_functions.flux)
    xr.testing.assert_identical(
        loaded.basis_functions.operator.basis_matrix,
        basis_functions.operator.basis_matrix,
    )

    model_spec = RhimeModelSpec(
        species="co2",
        domain="EUROPE",
        sectors=tuple(
            SectorSpec(
                name=source,
                flux_source=source,
                x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                variable_suffix=source.split("-")[0],
            )
            for source in source_order
        ),
        use_bc=True,
        bc_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.1},
        sigma_prior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
    )
    run_spec = RhimeRunSpec(
        start_date="2021-01-01",
        end_date="2021-01-05",
        sites=("stale",),
        averaging_period=(None,),
        model=model_spec,
        output=RhimeOutputSpec(output_format="none", save_inversion_output=False),
        split_by_sectors=True,
    )

    def fail_prepare(**kwargs: object) -> None:
        """Fail if the prepared-input runner attempts data preparation."""
        raise AssertionError("The prepared runner must not call OpenGHG preparation.")

    def skip_diagnostics(**kwargs: object) -> RhimeOutputBundle:
        """Keep this regression focused on the prepared-input execution seam."""
        output_prepared = kwargs["prepared"]
        assert isinstance(output_prepared, type(loaded))
        xr.testing.assert_identical(output_prepared.inv_inputs, loaded.inv_inputs)
        xr.testing.assert_identical(output_prepared.site_metadata, loaded.site_metadata)
        return RhimeOutputBundle(outputs={"executed": True})

    monkeypatch.setattr("openghg_inversions.rhime.runner.prepare_rhime_inputs", fail_prepare)
    monkeypatch.setattr(RhimeSampler, "sample", lambda self, model: az.InferenceData())
    monkeypatch.setattr(
        "openghg_inversions.rhime.runner.make_multisector_output_bundle",
        skip_diagnostics,
    )

    result = run_rhime_from_prepared_inputs(prepared_inputs=loaded, run_spec=run_spec)

    xr.testing.assert_identical(result.inv_inputs, loaded.inv_inputs)
    assert result.run_spec.sites == ("TAC", "MHD")
    assert result.model is not None
    assert {"x_ocean", "x_ff", "bc", "fixed_baseline", "y"}.issubset(result.model.named_vars)
    assert result.outputs == {"executed": True}
