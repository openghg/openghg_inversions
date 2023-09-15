from openghg_inversions.hbmcmc.hbmcmc import fixedbasisMCMC


def test_full_inversion(tmp_path):
    mcmc_args = {
        "species": "ch4",
        "sites": ["TAC"],
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "inlet": ["185m"],
        "instrument": ["picarro"],
        "domain": "EUROPE",
        "fp_height": ["185m"],
        "fp_model": "NAME",
        "emissions_name": ["total-ukghg-edgar7"],
        "met_model": "ukv",
        "meas_period": ["1H"],
        "outputname": "test_run",
        "outputpath": str(tmp_path),
        "quadtree_basis": True,
        "save_quadtree_to_outputpath": True,
        "nit": 1,
        "burn": 0,
        "tune": 0,
        "nbasis": 4,
        "nchain": 1,
    }
    fixedbasisMCMC(**mcmc_args)
