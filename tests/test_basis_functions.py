import xarray as xr
from openghg_inversions import utils
from openghg_inversions.basis_functions import quadtreebasisfunction
from openghg_inversions.get_data import data_processing_surface_notracer


def test_quadtree_basis_function(tac_ch4_data_args, raw_data_path):
    """Check if quadtree basis created with seed 42 and TAC CH4 args matches
    a basis created with the same arguments and saved to file.

    This is to check against changes in the code from when this test was made
    (13 Feb 2024)
    """
    fp_all, *_ = data_processing_surface_notracer(**tac_ch4_data_args)
    emissions_name = next(iter(fp_all[".flux"].keys()))
    basis_func = quadtreebasisfunction(
        emissions_name=[emissions_name], fp_all=fp_all, sites=["TAC"], start_date="2019-01-01", domain="EUROPE", species="ch4", seed=42
    )

    basis_func_reloaded = utils.basis(
        domain="EUROPE", basis_case="quadtree_ch4-test_basis", basis_directory=raw_data_path / "basis"
    )

    xr.testing.assert_allclose(basis_func, basis_func_reloaded)
