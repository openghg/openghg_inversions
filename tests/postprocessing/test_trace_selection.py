"""Trace selection filters borrowed groups before allocating outer-join padding."""

from dataclasses import replace

import arviz as az
import dask.array as da
from dask import delayed
from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from openghg_inversions.postprocessing.inversion_output import convert_idata_to_dataset


@pytest.mark.parametrize("lazy", [False, True], ids=["numpy", "dask"])
def test_trace_selection_precedes_padding_and_keeps_borrowed_sources(
    multisector_postprocessing_inv_out, monkeypatch, lazy
):
    """Unequal draws never pad or execute large unrequested concentration payloads."""
    produced = []

    def source(values, label):
        produced.append(label)
        return values

    def payload(values, label):
        if lazy:
            return da.from_delayed(delayed(source)(values, label), values.shape, values.dtype)
        return values

    groups = {}
    for group, draws in [("prior", [1, 3, 5]), ("posterior", [0, 1, 3, 5, 7])]:
        scale = np.arange(2 * len(draws) * 3, dtype=np.float32).reshape(2, len(draws), 3)
        concentration = np.full((2, len(draws), 20000), 4, dtype=np.float32)
        groups[group] = xr.Dataset(
            {
                "custom_scale": (("chain", "draw", "region"), payload(scale, group)),
                "custom_scale_latent": (("chain", "draw", "region"), scale + 100),
                "y": (("chain", "draw", "nmeasure"), payload(concentration, group + "_y")),
            },
            coords={"chain": [4, 9], "draw": draws, "region": [9, 2, 7]},
        )
        groups[group]["custom_scale"].attrs["units"] = "1"
    # A group with no requested variable must not enlarge the selected draw axis.
    groups["prior_predictive"] = xr.Dataset(
        {"y": (("chain", "draw", "nmeasure"), np.zeros((1, 1, 20000), dtype=np.float32))},
        coords={"chain": [0], "draw": [999]},
    )
    inv_out = replace(
        multisector_postprocessing_inv_out(),
        trace=az.InferenceData(**groups),
        model_metadata={"variable_roles": {"flux_scale": "custom_scale", "baseline": "custom_scale"}},
    )
    original_payloads = {group: inv_out.trace[group]["custom_scale"].data for group in ("prior", "posterior")}
    original_graphs = {
        group: data.__dask_graph__() if lazy else None for group, data in original_payloads.items()
    }
    padding = []
    original_getitem = xr.Variable._getitem_with_mask

    def record_padding(variable, *args, **kwargs):
        padding.append(variable.dims)
        return original_getitem(variable, *args, **kwargs)

    monkeypatch.setattr(xr.Variable, "_getitem_with_mask", record_padding)
    tasks = []
    with Callback(posttask=lambda key, result, *args: tasks.append((key, getattr(result, "nbytes", 0)))):
        selected = inv_out.trace_dataset(var_roles=(role for role in ["flux_scale", "baseline"]))
        assert not tasks
        assert not produced
        result = selected.compute(scheduler="single-threaded")

    if lazy:
        assert sorted(produced) == ["posterior", "prior"]
        assert max(nbytes for _, nbytes in tasks) <= 2 * 5 * 3 * np.dtype(np.float32).itemsize
    assert all("nmeasure" not in dims for dims in padding)
    assert set(result.data_vars) == {"custom_scale_prior", "custom_scale_posterior"}
    assert set(result.dims) == {"draw", "region"}
    np.testing.assert_array_equal(result.draw, [0, 1, 3, 5, 7])
    np.testing.assert_array_equal(result.region, [9, 2, 7])
    assert result.custom_scale_prior.dtype == np.float32
    assert result.custom_scale_prior.sel(draw=[0, 7]).isnull().all()
    for group, data in original_payloads.items():
        expected = groups[group].custom_scale.isel(chain=0, drop=True)
        xr.testing.assert_equal(result[f"custom_scale_{group}"].sel(draw=expected.draw), expected)
        assert inv_out.trace[group].custom_scale.data is data
        assert inv_out.trace[group].custom_scale.attrs == {"units": "1"}
        if lazy:
            assert data.__dask_graph__() is original_graphs[group]


def test_trace_selection_exact_names_group_suffixes_and_predictive_metadata(multisector_postprocessing_inv_out):
    """Literal names, aliased roles and predictive suffixes survive preselection."""
    groups = {}
    for index, group in enumerate(["prior", "posterior", "prior_predictive", "posterior_predictive"]):
        groups[group] = xr.Dataset(
            {
                "x": (("chain", "draw"), np.array([[index]], dtype=np.float32)),
                "x_prior": (("chain", "draw"), np.array([[index + 10]], dtype=np.float32)),
                "x_latent": (("chain", "draw"), np.array([[index + 20]], dtype=np.float32)),
            },
            coords={"chain": [0], "draw": [index]},
        )
    inv_out = replace(multisector_postprocessing_inv_out(), trace=az.InferenceData(**groups))
    selected = inv_out.trace_dataset(var_names="x_prior")
    assert set(selected.data_vars) == {f"x_prior_{group}" for group in groups}
    for index, group in enumerate(groups):
        assert selected[f"x_prior_{group}"].sel(draw=index).item() == index + 10
    assert selected.x_prior_prior_predictive.attrs == {
        "units": "ppm", "long_name": "prior_predictive_observed_mole_fraction"
    }
    assert groups["prior_predictive"].x_prior.attrs == {}
    assert not inv_out.trace_dataset(var_names=[]).variables
    assert not inv_out.trace_dataset(var_names="absent").variables
    with pytest.raises(ValueError, match="var_roles or var_names"):
        inv_out.trace_dataset("flux_scale", var_names="x")
    with pytest.raises(KeyError, match="Unknown InversionOutput variable role"):
        inv_out.trace_dataset("unknown")


def test_converter_default_and_unsuffixed_group_selection():
    """All-variable defaults and constant-group use preserve existing alignment."""
    prior = xr.Dataset({"x": (("chain", "draw"), [[1.0, 2.0], [90.0, 90.0]])}, coords={"draw": [0, 2]})
    posterior = xr.Dataset({"x": (("chain", "draw"), [[3.0, 4.0]])}, coords={"draw": [1, 2]})
    constant = xr.Dataset({"h": ("region", [1.0, 2.0]), "other": ("region", [5.0, 6.0])})
    trace = az.InferenceData(prior=prior, posterior=posterior, constant_data=constant)
    expected = xr.merge(
        [prior.isel(chain=0, drop=True).rename(x="x_prior"), posterior.isel(chain=0, drop=True).rename(x="x_posterior")],
        join="outer",
    )
    xr.testing.assert_identical(convert_idata_to_dataset(trace), expected)
    xr.testing.assert_identical(
        convert_idata_to_dataset(trace, group_filters=["data"], add_suffix=False, var_names=["h"]),
        constant[["h"]],
    )
