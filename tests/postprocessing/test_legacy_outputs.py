import numpy as np
import xarray as xr

from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.legacy_outputs import make_legacy_hbmcmc_output


def test_make_legacy_hbmcmc_output_handles_mixed_nmeasure_indexes(raw_data_path, europe_country_file):
    legacy = xr.open_dataset(raw_data_path / "standard_rhime_outs.nc")
    inv_out = InversionOutput.load(raw_data_path / "inversion_output.nc")
    inv_out.times = xr.DataArray(
        inv_out.times.values,
        dims=["nmeasure"],
        coords={"nmeasure": np.arange(inv_out.obs.sizes["nmeasure"])},
        attrs=inv_out.times.attrs,
        name=inv_out.times.name,
    )

    compat = make_legacy_hbmcmc_output(
        inv_out=inv_out,
        mcmc_results={
            "xouts": legacy["xtrace"],
            "sigouts": legacy["sigtrace"],
            "bcouts": legacy["bctrace"],
        },
        sigma_freq_index=legacy["sigmafreqindex"].values,
        Hx=legacy["xsensitivity"].values.T,
        Hbc=legacy["bcsensitivity"].values.T,
        country_file=europe_country_file,
        use_bc=True,
    )

    assert (compat["nmeasure"].values == legacy["nmeasure"].values).all()
    assert compat["Yobs"].dims == ("nmeasure",)
    assert compat["Ytime"].dims == ("nmeasure",)
    assert compat["Ymodmean"].dims == ("nmeasure",)
    assert "site" not in compat["Yobs"].coords
    assert "time" not in compat["Yobs"].coords
    assert "site" not in compat["Ytime"].coords
    assert "time" not in compat["Ytime"].coords
    assert compat["Ymod68"].dims[0] == "nmeasure"
