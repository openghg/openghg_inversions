import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.basis._functions import (
    basis,
    basis_functions,
    _flux_fp_from_fp_all,
    _mean_fp_times_mean_flux,
)
from openghg_inversions.basis import (
    basis_functions_wrapper,
    bucketbasisfunction,
    quadtreebasisfunction,
    fixed_outer_regions_basis,
)
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.basis.operators import (
    BucketBasisOperator,
    MultiSourceBucketBasisOperator,
)
from openghg_inversions.basis._helpers import apply_fp_basis_functions, fp_sensitivity
from openghg_inversions.inversion_data import data_processing_surface_notracer

from helpers import basis_function, footprint
from helpers import (
    convert_old_multisector_H_to_gathered,
    make_basis_flat_from_blocks,
    make_fp_x_flux,
    make_fp_x_flux_sectoral,
    expected_H_from_basis_sum,
    old_apply_fp_basis_functions_like,
)


def test_fp_x_flux(tac_ch4_data_args):
    """Test that mean(fp) * mean(flux) is invariant to duplicating sites and shifting time coords.

    This is a regression test for legacy preprocessing helpers `_flux_fp_from_fp_all` and
    `_mean_fp_times_mean_flux`: changing site membership or time coordinates (without changing
    values) should not change the mean footprint×flux product.
    """
    fp_all, *_ = data_processing_surface_notracer(**tac_ch4_data_args)
    emissions_name = [next(iter(fp_all[".flux"].keys()))]

    flux1, fp1 = _flux_fp_from_fp_all(fp_all, emissions_name)
    mean_fp_flux1 = _mean_fp_times_mean_flux(flux1, fp1)

    # add new site with same footprint -- this should not change the mean over time
    fp_all["ABC"] = fp_all["TAC"]

    flux2, fp2 = _flux_fp_from_fp_all(fp_all, emissions_name)
    mean_fp_flux2 = _mean_fp_times_mean_flux(flux2, fp2)

    xr.testing.assert_allclose(mean_fp_flux1, mean_fp_flux2)

    # shift time of second site -- this should not change the mean over time
    max_time = pd.Timedelta(fp_all["TAC"].time.max().values - fp_all["TAC"].time.min().values)
    new_time = fp_all["TAC"].time + max_time
    fp_all["ABC"] = fp_all["ABC"].assign_coords(time=new_time)

    flux3, fp3 = _flux_fp_from_fp_all(fp_all, emissions_name)
    mean_fp_flux3 = _mean_fp_times_mean_flux(flux3, fp3)

    xr.testing.assert_allclose(mean_fp_flux1, mean_fp_flux3)


def test_quadtree_basis_function(tac_ch4_data_args, raw_data_path):
    """Check if quadtree basis created with seed 42 and TAC CH4 args matches
    a basis created with the same arguments and saved to file.

    This is to check against changes in the code from when this test was made
    (13 Feb 2024)
    """
    fp_all, *_ = data_processing_surface_notracer(**tac_ch4_data_args)
    emissions_name = next(iter(fp_all[".flux"].keys()))
    basis_func = quadtreebasisfunction(
        emissions_name=[emissions_name], fp_all=fp_all, start_date="2019-01-01", seed=42, domain="EUROPE"
    )

    basis_func_reloaded = basis(
        domain="EUROPE", basis_case="quadtree_ch4-test_basis", basis_directory=raw_data_path / "basis"
    )

    # TODO: create new "fixed" basis function file, since we've switched basis functions from
    # dataset to data array
    xr.testing.assert_allclose(basis_func, basis_func_reloaded.basis)


def test_bucket_basis_function(tac_ch4_data_args, raw_data_path):
    """Check if quadtree basis created with seed 42 and TAC CH4 args matches
    a basis created with the same arguments and saved to file.

    This is to check against changes in the code from when this test was made
    (13 Feb 2024)
    """
    fp_all, *_ = data_processing_surface_notracer(**tac_ch4_data_args)
    emissions_name = next(iter(fp_all[".flux"].keys()))
    basis_func = bucketbasisfunction(
        emissions_name=[emissions_name],
        fp_all=fp_all,
        start_date="2019-01-01",
        domain="EUROPE",
        nbasis=98,
    )

    basis_func_reloaded = basis(
        domain="EUROPE", basis_case="bucket_ch4-test_basis", basis_directory=raw_data_path / "basis"
    )

    # TODO: create new "fixed" basis function file, since we've switched basis functions from
    # dataset to data array
    xr.testing.assert_allclose(basis_func, basis_func_reloaded.basis)


def test_fixed_outer_region_basis_function(tac_ch4_data_args, raw_data_path):
    """Check if fixed outer region basis created with seed 42 and TAC CH4 args matches
    a basis created with the same arguments and saved to file.

    This is to check against changes in the code from when this test was made
    (2 Sep 2024)
    """
    fp_all, *_ = data_processing_surface_notracer(**tac_ch4_data_args)
    emissions_name = next(iter(fp_all[".flux"].keys()))
    basis_func = fixed_outer_regions_basis(
        emissions_name=[emissions_name],
        fp_all=fp_all,
        start_date="2019-01-01",
        domain="EUROPE",
        basis_algorithm="weighted",
    )

    basis_func_reloaded = basis(
        domain="EUROPE",
        basis_case="fixed_outer_region_ch4-test_basis",
        basis_directory=raw_data_path / "basis",
    )

    # TODO: create new "fixed" basis function file, since we've switched basis functions from
    # dataset to data array
    xr.testing.assert_allclose(basis_func, basis_func_reloaded.basis)


def test_fp_sensitivity_one_flux():
    """Test fp_sensitivity with one flux sector."""
    nlat, nlon = 10, 12
    nbasis = 3
    basis_func = basis_function(nlat, nlon, nbasis)
    fp = footprint(nlat, nlon, "2019-01-01", "2019-01-02", 2)

    fp_and_data = {"TAC": xr.Dataset({"fp_x_flux": fp}), ".flux": {"a": 1}}

    fp_and_data = fp_sensitivity(fp_and_data, basis_func)

    h = fp_and_data["TAC"].H

    # the footprint values at time 0 are 1, and at time 1 are 2
    np.testing.assert_allclose(2 * h.isel(time=0), h.isel(time=1))


def test_fp_sensitivity_two_flux_sectors():
    """Check that we can apply a common basis function to two separate sources."""
    nlat, nlon = 10, 12
    nbasis = 3
    basis_func = basis_function(nlat, nlon, nbasis)

    fp1 = footprint(nlat, nlon, "2019-01-01", "2019-01-02", 2)
    fp2 = footprint(nlat, nlon, "2019-01-01", "2019-01-02", 2)
    fp = xr.concat([fp1.expand_dims({"source": ["a"]}), fp2.expand_dims({"source": ["b"]})], dim="source")
    fp_and_data = {"TAC": xr.Dataset({"fp_x_flux_sectoral": fp}), ".flux": {"a": 1, "b": 2}}

    fp_and_data = fp_sensitivity(fp_and_data, basis_func)

    for source in ["a", "b"]:
        h = fp_and_data["TAC"].H.sel(source=source).dropna("region")

        # the footprint values at time 0 are 1, and at time 1 are 2
        np.testing.assert_allclose(2 * h.isel(time=0), h.isel(time=1))


def test_fp_sensitivity_two_flux_sectors_two_basis_funcs():
    """Check that we can apply separate basis functions to separate sources."""
    nlat, nlon = 10, 12
    nbasis1 = 3
    nbasis2 = 4
    basis_func1 = basis_function(nlat, nlon, nbasis1)
    basis_func2 = basis_function(nlat, nlon, nbasis2)
    basis_func = {"a": basis_func1, "b": basis_func2}

    fp1 = footprint(nlat, nlon, "2019-01-01", "2019-01-02", 2)
    fp2 = footprint(nlat, nlon, "2019-01-01", "2019-01-02", 2)
    fp = xr.concat([fp1.expand_dims({"source": ["a"]}), fp2.expand_dims({"source": ["b"]})], dim="source")
    fp_and_data = {"TAC": xr.Dataset({"fp_x_flux_sectoral": fp}), ".flux": {"a": 1, "b": 2}}

    fp_and_data = fp_sensitivity(fp_and_data, basis_func)

    for source in ["a", "b"]:
        h = fp_and_data["TAC"].H.sel(source=source).dropna("region")

        # the footprint values at time 0 are 1, and at time 1 are 2
        np.testing.assert_allclose(2 * h.isel(time=0), h.isel(time=1))


def test_basisfunctions_sensitivity_synthetic_matches_explicit_sum():
    """Sensitivity on a tiny synthetic example matches explicit region-wise summation.

    This is the most important unit-level correctness test for the new BasisOperator/BasisFunctions
    pathway: it checks that the sensitivity (grid -> state) produced by the new code matches a
    hand-computed result for a 2x2 grid with two regions.

    Notes:
        - The flux field is set to ones because sensitivity should depend only on the basis partition
          (i.e. the operator / basis_matrix), not on flux weighting.
        - Expected result is produced by `expected_H_from_basis_sum`, which explicitly sums fp_x_flux
          over grid cells belonging to each region.
    """
    # 2x2 grid, 3 times, basis has 2 regions: top row region 1, bottom row region 2
    basis = make_basis_flat_from_blocks([[1, 1], [2, 2]])

    fp_x_flux = make_fp_x_flux(nlat=2, nlon=2, ntime=3)

    # flux is only used for interpolation_matrix etc; sensitivity uses basis_matrix only
    # but BasisFunctions requires flux; use ones
    flux = xr.ones_like(basis, dtype=float)

    bf = BasisFunctions.from_basis_flat(basis_flat=basis, flux=flux, operator_kwargs={"state_dim": "region"})

    H_new = bf.sensitivity(fp_x_flux)

    H_expected = expected_H_from_basis_sum(fp_x_flux, basis)

    xr.testing.assert_allclose(H_new, H_expected)


def test_basisfunctions_sensitivity_synthetic_multisector_shared_basis():
    """A single basis partition can be applied independently across multiple sources.

    This reflects the common use case where footprints/flux are sectoral (have a `source` dimension),
    but the spatial basis partition is shared across sectors.

    The new `BasisFunctions.sensitivity` should:
        - carry through the non-grid dimension `source`, and
        - produce the same per-source sensitivity you would get by running the single-source
          calculation separately for each source.

    This is a regression guard for xarray broadcasting / dot behaviour.
    """
    sources = ["A", "B"]
    basis = make_basis_flat_from_blocks([[1, 1], [2, 2]])

    fp_x_flux_sectoral = make_fp_x_flux_sectoral(sources=sources, nlat=2, nlon=2, ntime=3)

    flux = xr.ones_like(basis, dtype=float)

    bf = BasisFunctions.from_basis_flat(basis_flat=basis, flux=flux, operator_kwargs={"state_dim": "region"})

    H = bf.sensitivity(fp_x_flux_sectoral)

    # Expected: same as single-sector, but computed per source and carried through source dim.
    # bf.sensitivity preserves non-(lat,lon) dims, so expect (source, time, region) or (source, region, time)
    # We enforce canonical order via transpose below.
    H = H.transpose("region", "time", "source")

    # Construct expected by explicit summation for each source
    expected_pieces = []
    for s in sources:
        Hs = expected_H_from_basis_sum(fp_x_flux_sectoral.sel(source=s), basis)  # (region, time)
        expected_pieces.append(Hs.expand_dims(source=[s]))
    H_expected = xr.concat(expected_pieces, dim="source").transpose("region", "time", "source")

    xr.testing.assert_allclose(H, H_expected)


def test_basisfunctions_sensitivity_regression_matches_legacy_like():
    """Regression test: new sensitivity matches a simplified legacy implementation.

    We compare:
        - `BasisFunctions.sensitivity(fp_x_flux)` (new pathway using BasisOperator), vs
        - `old_apply_fp_basis_functions_like(fp_x_flux, basis)` (a direct/legacy-style computation).

    This test is intentionally small and synthetic to make failures easy to debug.
    """
    basis = make_basis_flat_from_blocks([[1, 1], [2, 2]])
    fp_x_flux = make_fp_x_flux(nlat=2, nlon=2, ntime=4)

    flux = xr.ones_like(basis, dtype=float)
    bf = BasisFunctions.from_basis_flat(basis_flat=basis, flux=flux, operator_kwargs={"state_dim": "region"})

    H_new = bf.sensitivity(fp_x_flux)  # (region, time)
    H_old = old_apply_fp_basis_functions_like(fp_x_flux, basis)  # (region, time)

    xr.testing.assert_allclose(H_new, H_old)


def test_synthetic_no_all_zero_state_rows_when_fp_positive_everywhere():
    """No all-zero state rows should appear when fp_x_flux is strictly positive.

    This guards against a class of bugs that historically occurred when:
        - using a dictionary of basis functions (sectoral / multi-source), and
        - padding region dimensions to a common maximum region index.

    In the new gathered/MultiIndex formulation, we should not introduce missing-region rows that are
    entirely zero when the underlying footprint×flux field is strictly positive everywhere.
    """
    basis = make_basis_flat_from_blocks([[1, 1], [2, 2]])
    fp_x_flux = make_fp_x_flux(nlat=2, nlon=2, ntime=3, values=np.ones((2, 2, 3)))

    flux = xr.ones_like(basis, dtype=float)
    bf = BasisFunctions.from_basis_flat(basis_flat=basis, flux=flux, operator_kwargs={"state_dim": "region"})

    H = bf.sensitivity(fp_x_flux)  # (region, time)

    # For strictly positive fp_x_flux and nonempty basis regions, each region row should be > 0 somewhere
    assert (H > 0).any(dim="time").all().item()


# @pytest.mark.slow
def test_basisfunctions_sensitivity_matches_apply_fp_basis_functions_real_data():
    """New sensitivity matches legacy `apply_fp_basis_functions` for real test-suite data.

    This is a higher-level integration test:
        - It uses `basis_functions_wrapper` to construct the legacy basis and compute H.
        - It then reconstructs a `BasisFunctions` instance using the same basis and a representative
          flux field, and verifies `BasisFunctions.sensitivity(ds.fp_x_flux)` matches the legacy H.

    This guards against coordinate alignment, dimension ordering, and subtle differences in the
    region-labelling conventions on real-world data.
    """
    data_args = {
        "species": "ch4",
        "sites": ["MHD", "TAC"],
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "bc_store": "inversions_tests",
        "obs_store": "inversions_tests",
        "footprint_store": "inversions_tests",
        "emissions_store": "inversions_tests",
        "inlet": ["10m", "185m"],
        "instrument": ["gcmd", "picarro"],
        "domain": "EUROPE",
        "fp_height": ["10m", "185m"],
        "fp_model": "NAME",
        "emissions_name": ["total-ukghg-edgar7"],
        "averaging_period": ["1h", "1h"],
    }

    fp_all, *_ = data_processing_surface_notracer(**data_args)

    basis_args = {
        "species": "ch4",
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "emissions_name": ["total-ukghg-edgar7"],
        "nbasis": 20,
        "use_bc": True,
        "basis_algorithm": "weighted",
        "bc_basis_case": "NESW",
    }

    fp_all_with_basis = basis_functions_wrapper(fp_all, **basis_args)

    site = "MHD"
    ds = fp_all_with_basis[site]

    # old sensitivity (already computed by wrapper) is ds["H"]; but we want to call the legacy fn directly too
    H_old = apply_fp_basis_functions(ds.fp_x_flux, fp_all_with_basis[".basis"])

    # new sensitivity
    # Need a flux field; the wrapper has fp_all_with_basis[".flux"][source].data.flux
    flux_source = next(iter(fp_all_with_basis[".flux"].keys()))
    flux = fp_all_with_basis[".flux"][flux_source].data.flux

    # bf = BasisFunctions(basis_flat=fp_all_with_basis[".basis"], flux=flux)
    bf = BasisFunctions.from_basis_flat(
        basis_flat=fp_all_with_basis[".basis"], flux=flux, operator_kwargs={"state_dim": "region"}
    )
    H_new = bf.sensitivity(ds.fp_x_flux)

    # Ensure same dim order for comparison
    if H_new.dims != H_old.dims:
        H_new = H_new.transpose(*H_old.dims)

    xr.testing.assert_allclose(H_new, H_old)

    # now test vs. result of basis_functions_wrapper
    H_old = ds.H

    if H_new.dims != H_old.dims:
        H_new = H_new.transpose(*H_old.dims)

    xr.testing.assert_allclose(H_new, H_old)


# @pytest.mark.slow
def test_multisector_ragged_new_matches_old_after_conversion():
    """Ragged multi-source: new gathered H matches legacy padded H after conversion.

    Historically, multi-sector sensitivities were represented as a padded array:
        H_old(region=max_regions, time, source)
    where missing regions for a given source were represented by all-zero rows.

    The new MultiSourceBucketBasisOperator produces a gathered representation with a MultiIndex:
        H_new(region=(source, region_in_source), time)

    This test:
        1) Computes the old padded H via `fp_sensitivity` with two different bases (ragged region counts).
        2) Converts H_old -> gathered MultiIndex using `convert_old_multisector_H_to_gathered`.
        3) Computes H_new with `BasisFunctions.from_multi_source_basis_flat(...).sensitivity(...)`.
        4) Asserts equality.

    This is the key equivalence test justifying the new MultiIndex-based operator.
    """
    # --- Load test-suite data (same as notebook) ---
    data_args = {
        "species": "ch4",
        "sites": ["MHD", "TAC"],
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "bc_store": "inversions_tests",
        "obs_store": "inversions_tests",
        "footprint_store": "inversions_tests",
        "emissions_store": "inversions_tests",
        "inlet": ["10m", "185m"],
        "instrument": ["gcmd", "picarro"],
        "domain": "EUROPE",
        "fp_height": ["10m", "185m"],
        "fp_model": "NAME",
        "emissions_name": ["total-ukghg-edgar7"],
        "averaging_period": ["1h", "1h"],
    }
    fp_all, *_ = data_processing_surface_notracer(**data_args)

    # --- Make a "sectoral" fp_x_flux like your notebook did ---
    fp_all_sectoral = fp_all.copy()
    fp_all_sectoral[".flux"] = fp_all[".flux"].copy()
    fp_all_sectoral[".flux"]["sector2"] = fp_all_sectoral[".flux"]["total-ukghg-edgar7"]

    for k, v in fp_all_sectoral.items():
        if str(k).startswith("."):
            continue
        ds = v["fp_x_flux"]
        to_concat = [
            ds.expand_dims({"source": ["total-ukghg-edgar7"]}),
            ds.expand_dims({"source": ["sector2"]}),
        ]
        v["fp_x_flux_sectoral"] = xr.concat(to_concat, dim="source")

    # --- Build two different basis partitions (ragged region counts) ---
    weighted_basis_args_1 = {
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "emissions_name": ["total-ukghg-edgar7"],
        "nbasis": 20,
    }
    weighted_basis_args_2 = {
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "emissions_name": ["sector2"],
        "nbasis": 30,
    }

    basis1 = basis_functions["weighted"].algorithm(fp_all_sectoral, **weighted_basis_args_1)
    basis2 = basis_functions["weighted"].algorithm(fp_all_sectoral, **weighted_basis_args_2)

    basis_dict = {"total-ukghg-edgar7": basis1, "sector2": basis2}

    # --- Old behaviour: fp_sensitivity pads to max(region) and introduces zero rows for missing regions ---
    fp_old = fp_sensitivity(fp_all_sectoral.copy(), basis_func=basis_dict)
    site = "MHD"
    H_old = fp_old[site]["H"]  # (region=max, time, source)

    # Convert old padded to gathered multiindex region and drop all-zero rows
    H_old_gathered = convert_old_multisector_H_to_gathered(H_old)

    # --- New behaviour
    flux_dict = {k: v.data.flux for k, v in fp_all_sectoral[".flux"].items()}

    multisector_bf = BasisFunctions.from_multi_source_basis_flat(
        basis_flat=basis_dict, flux=flux_dict, operator_kwargs={"state_dim": "region"}
    )

    H_new_gathered = multisector_bf.sensitivity(fp_all_sectoral[site].fp_x_flux_sectoral)
    xr.testing.assert_allclose(H_new_gathered, H_old_gathered)


def _make_simple_state_trace(
    *,
    region_dim: str = "region",
    draw_dim: str = "draw",
    chain_dim: str | None = None,
    region_values: list[float] = [10.0, 100.0],
    draw_values: list[int] = [0, 1, 2],
) -> xr.DataArray:
    """Make a tiny deterministic PyMC-like trace array for interpolate tests."""
    state = xr.DataArray(
        np.asarray(region_values, dtype=float),
        dims=(region_dim,),
        coords={region_dim: np.arange(len(region_values))},
        name="state",
    )

    draws = xr.DataArray(np.asarray(draw_values, dtype=int), dims=(draw_dim,), name=draw_dim)
    state = state.expand_dims({draw_dim: draws})

    # Make draws vary slightly so we catch broadcasting issues.
    # region r has values region_values[r] + draw
    state = state + xr.DataArray(np.asarray(draw_values, dtype=float), dims=(draw_dim,))

    if chain_dim is not None:
        chain = xr.DataArray(np.asarray([0, 1], dtype=int), dims=(chain_dim,), name=chain_dim)
        state = state.expand_dims({chain_dim: chain})
        # Make chain 1 different to catch chain propagation.
        state = state + xr.DataArray(np.asarray([0.0, 1000.0]), dims=(chain_dim,))

    return state


# --------------------------------------------------------------------------------------
# DataTree roundtrip tests for FluxWeightedBasis/BasisFunctions wrapper
# --------------------------------------------------------------------------------------
def test_basisfunctions_roundtrip_datatree_single_source():
    """FluxWeightedBasis/BasisFunctions DataTree roundtrip for a single-source basis.

    Ensures wrapper-level serialization is stable:
        - `BasisFunctions.to_datatree()` stores operator metadata under group "basis"
          (via `operator.to_datatree()`),
        - stores flux under group "flux" with variable name "flux",
        - and `BasisFunctions.from_datatree()` reconstructs an equivalent object.

    We assert both flux and operator internals (basis_flat and basis_matrix) are identical.
    """
    basis = make_basis_flat_from_blocks([[1, 1], [2, 2]])
    flux = xr.ones_like(basis, dtype=float).rename("flux")

    bf = BasisFunctions.from_basis_flat(basis_flat=basis, flux=flux, operator_kwargs={"state_dim": "region"})

    dt = bf.to_datatree()
    bf2 = BasisFunctions.from_datatree(dt)

    assert isinstance(bf2.operator, BucketBasisOperator)
    xr.testing.assert_identical(bf2.flux, bf.flux)
    xr.testing.assert_identical(bf2.operator.basis_flat, bf.operator.basis_flat)
    xr.testing.assert_identical(bf2.operator.basis_matrix, bf.operator.basis_matrix)


def test_basisfunctions_roundtrip_datatree_multisource_flux_mapping():
    """DataTree roundtrip for multi-source basis and flux supplied as a mapping.

    This specifically exercises the constructor path where flux is provided as:
        {"A": flux_A(lat, lon), "B": flux_B(lat, lon)}
    which is concatenated into a single DataArray with a `source` dimension.

    We then roundtrip through DataTree and check:
        - operator type is MultiSourceBucketBasisOperator,
        - basis_flat dict keys are preserved,
        - basis_matrix and flux are identical.
    """
    basis_a = make_basis_flat_from_blocks([[1, 1], [2, 2]])
    basis_b = make_basis_flat_from_blocks([[1, 2], [1, 2]])
    basis = {"A": basis_a, "B": basis_b}

    # Exercise the mapping->concat codepath
    flux_a = xr.ones_like(basis_a, dtype=float) * 1.0
    flux_b = xr.ones_like(basis_b, dtype=float) * 2.0
    flux = {"A": flux_a.rename("flux"), "B": flux_b.rename("flux")}

    bf = BasisFunctions.from_multi_source_basis_flat(
        basis_flat=basis, flux=flux, operator_kwargs={"state_dim": "region"}
    )
    assert "source" in bf.flux.dims
    xr.testing.assert_allclose(bf.flux.sel(source="A", drop=True), flux_a)
    xr.testing.assert_allclose(bf.flux.sel(source="B", drop=True), flux_b)

    dt = bf.to_datatree()
    bf2 = BasisFunctions.from_datatree(dt)

    assert isinstance(bf2.operator, MultiSourceBucketBasisOperator)
    assert set(bf2.operator.basis_flat.keys()) == {"A", "B"}
    xr.testing.assert_identical(bf2.flux, bf.flux)
    xr.testing.assert_identical(bf2.operator.basis_matrix, bf.operator.basis_matrix)
    for k in bf.operator.basis_flat:
        xr.testing.assert_identical(bf2.operator.basis_flat[k], bf.operator.basis_flat[k])


# --------------------------------------------------------------------------------------
# Interpolate tests (wrapper-level): intended PyMC-like state traces
# --------------------------------------------------------------------------------------
def test_basisfunctions_interpolate_no_flux_weights_trace_dims_propagate():
    """Interpolate a PyMC-like state trace to the grid (no flux weighting).

    We construct a tiny state vector with dims (region, draw) to mimic posterior samples and verify:
        - output dims are (lat, lon, draw),
        - grid cells in a region take the corresponding region value for each draw.

    This is a regression guard for dimension propagation and dot/broadcast behaviour.
    """
    # Basis: top row region 0, bottom row region 1 (labels 1/2 -> regions 0/1)
    basis = make_basis_flat_from_blocks([[1, 1], [2, 2]])
    flux = xr.ones_like(basis, dtype=float)

    bf = BasisFunctions.from_basis_flat(basis_flat=basis, flux=flux, operator_kwargs={"state_dim": "region"})

    state = _make_simple_state_trace(region_dim="region", draw_dim="draw", chain_dim=None)

    out = bf.interpolate(state, flux=False)
    assert set(out.dims) == {"lat", "lon", "draw"}

    # Expected: top row gets region 0 value, bottom row gets region 1 value
    # region 0 base=10, region 1 base=100, plus draw offset
    expected_top = xr.DataArray(np.asarray([10.0, 11.0, 12.0]), dims=("draw",), coords={"draw": state.draw})
    expected_bottom = xr.DataArray(
        np.asarray([100.0, 101.0, 102.0]), dims=("draw",), coords={"draw": state.draw}
    )

    xr.testing.assert_allclose(out.sel(lat=0, lon=0, drop=True), expected_top)
    xr.testing.assert_allclose(out.sel(lat=0, lon=1, drop=True), expected_top)
    xr.testing.assert_allclose(out.sel(lat=1, lon=0, drop=True), expected_bottom)
    xr.testing.assert_allclose(out.sel(lat=1, lon=1, drop=True), expected_bottom)


def test_basisfunctions_interpolate_with_flux_weights_trace_dims_propagate():
    """Interpolate a PyMC-like state trace with flux-weighting.

    This tests the common "state -> flux field" step:
        flux_field(lat, lon, draw) = interpolate(state(region, draw), weights=flux(lat, lon))

    Using a non-uniform flux field on a 2x2 grid makes it easy to verify that weighting is applied
    correctly, not just that shapes line up.
    """
    basis = make_basis_flat_from_blocks([[1, 1], [2, 2]])

    # Make flux non-uniform on the grid so we verify weighting actually occurs
    flux_values = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    flux = xr.DataArray(flux_values, coords=basis.coords, dims=basis.dims, name="flux")

    bf = BasisFunctions.from_basis_flat(basis_flat=basis, flux=flux, operator_kwargs={"state_dim": "region"})

    state = _make_simple_state_trace(region_dim="region", draw_dim="draw", chain_dim=None)

    out = bf.interpolate(state, flux=True)
    assert set(out.dims) == {"lat", "lon", "draw"}

    # Expected: out(lat,lon,draw) = state(region,draw) * flux(lat,lon) for the region covering that cell
    # Top row uses region 0, bottom row uses region 1.
    region0 = xr.DataArray(np.asarray([10.0, 11.0, 12.0]), dims=("draw",), coords={"draw": state.draw})
    region1 = xr.DataArray(np.asarray([100.0, 101.0, 102.0]), dims=("draw",), coords={"draw": state.draw})

    xr.testing.assert_allclose(out.sel(lat=0, lon=0, drop=True), region0 * 1.0)
    xr.testing.assert_allclose(out.sel(lat=0, lon=1, drop=True), region0 * 2.0)
    xr.testing.assert_allclose(out.sel(lat=1, lon=0, drop=True), region1 * 3.0)
    xr.testing.assert_allclose(out.sel(lat=1, lon=1, drop=True), region1 * 4.0)


def test_basisfunctions_interpolate_trace_with_chain_dim():
    """Interpolate a state trace that includes a `chain` dimension.

    Many workflows keep posterior draws as (region, chain, draw). Even if in practice users often
    select a single chain, we ensure that:
        - chain is preserved as a non-dot dimension,
        - results differ between chains when the input differs.

    This guards against accidentally squeezing/dropping the chain dimension.
    """
    basis = make_basis_flat_from_blocks([[1, 1], [2, 2]])
    flux = xr.ones_like(basis, dtype=float)

    bf = BasisFunctions.from_basis_flat(basis_flat=basis, flux=flux, operator_kwargs={"state_dim": "region"})

    state = _make_simple_state_trace(region_dim="region", draw_dim="draw", chain_dim="chain")

    out = bf.interpolate(state, flux=False)
    assert set(out.dims) == {"lat", "lon", "draw", "chain"}

    # Spot-check one cell per region, both chains.
    # chain=0: region0=10+draw, region1=100+draw
    # chain=1: +1000
    expected_region0_chain0 = xr.DataArray(
        np.asarray([10.0, 11.0, 12.0]), dims=("draw",), coords={"draw": state.draw}
    )
    expected_region0_chain1 = expected_region0_chain0 + 1000.0

    xr.testing.assert_allclose(out.sel(lat=0, lon=0, chain=0, drop=True), expected_region0_chain0)
    xr.testing.assert_allclose(out.sel(lat=0, lon=0, chain=1, drop=True), expected_region0_chain1)


# --------------------------------------------------------------------------------------
# Multi-source equivalence: gathered MultiIndex vs legacy padded H conversion
# --------------------------------------------------------------------------------------
def test_multisource_sensitivity_matches_legacy_padded_conversion_smoke():
    """Smoke test: gathered multi-source sensitivity matches legacy padded->gathered conversion.

    This test is intentionally simple (same basis for each source, small synthetic fp_x_flux) and
    checks that:

        H_new(region=(source, region_in_source), time)
        ==
        convert_old_multisector_H_to_gathered(H_old(region, time, source))

    It provides quick feedback that the gathered/MultiIndex representation remains consistent with
    older expectations, without requiring the heavier "ragged basis + real data" test.
    """
    # Use small deterministic basis and footprints; simplest: same basis per source but different scaling.
    sources = ["A", "B"]
    basis = make_basis_flat_from_blocks([[1, 1], [2, 2]])
    basis_by_source = {s: basis for s in sources}

    fp_x_flux_sectoral = make_fp_x_flux_sectoral(sources=sources, nlat=2, nlon=2, ntime=3)
    flux = xr.ones_like(basis, dtype=float)

    # New gathered H: MultiSourceBucketBasisOperator under the wrapper
    bf = BasisFunctions.from_multi_source_basis_flat(
        basis_flat=basis_by_source, flux=flux, operator_kwargs={"state_dim": "region"}
    )
    H_new = bf.sensitivity(fp_x_flux_sectoral)

    # Construct a legacy-like padded H_old(region=max_regions, time, source) by computing each source separately
    # with the single-source basis and then padding to the maximum region count (here identical counts).
    H_pieces = []
    for s in sources:
        Hs = expected_H_from_basis_sum(
            fp_x_flux_sectoral.sel(source=s), basis, region_dim="region"
        )  # (region,time)
        H_pieces.append(Hs.expand_dims(source=[s]))
    H_old = xr.concat(H_pieces, dim="source").transpose("region", "time", "source")

    # Convert legacy padded -> gathered MultiIndex and compare (up to ordering)
    H_old_gathered = convert_old_multisector_H_to_gathered(
        H_old,
        source_dim="source",
        region_dim="region",
        gathered_dim="region",
        source_region_dim="region_in_source",
    )

    # The new operator returns region as a MultiIndex; order should match source-major stacking.
    # Ensure both are in a comparable order.
    H_new = H_new.transpose("region", "time")
    H_old_gathered = H_old_gathered.transpose("region", "time")

    xr.testing.assert_allclose(H_new, H_old_gathered)
