"""Tests for multisector flux reconstruction in modern postprocessing outputs."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.flux_sanitization import FluxNonFiniteMetadata, NONFINITE_POLICY_ZERO_FILL
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.make_outputs import (
    make_flux_outputs,
    make_multisector_flux_trace_outputs,
)


def _flux_nonfinite_metadata(data: xr.DataArray | xr.Dataset) -> FluxNonFiniteMetadata:
    """Return parsed non-finite flux metadata from an xarray object."""
    metadata = FluxNonFiniteMetadata.from_attrs(data.attrs)
    assert metadata is not None
    return metadata


def test_multisector_flux_outputs_reconstruct_sector_and_total_flux(
    multisector_postprocessing_inv_out: Callable[..., InversionOutput],
) -> None:
    """Multisector flux postprocessing reconstructs sector fluxes before total statistics."""
    outputs = make_flux_outputs(
        multisector_postprocessing_inv_out(),
        stats=["mean", "stdev"],
        include_scale_factors=True,
        report_flux_on_inversion_grid=False,
    )

    assert "flux_ff_posterior_mean" in outputs
    assert "flux_ocean_posterior_mean" in outputs
    assert "scaling_ff_posterior_mean" in outputs
    assert "flux_total_posterior_mean" in outputs
    assert "flux_total_posterior_stdev" in outputs
    assert outputs["flux_ff_posterior_mean"].attrs["units"] == "mol/m2/s"
    assert outputs["flux_total_posterior_mean"].attrs["units"] == "mol/m2/s"
    assert float(outputs["flux_total_posterior_mean"].item()) == 2.0
    assert float(outputs["flux_total_posterior_stdev"].item()) == 0.0
    assert float(outputs["flux_ff_posterior_stdev"].item()) > 0.0
    assert _flux_nonfinite_metadata(outputs).policy == NONFINITE_POLICY_ZERO_FILL


def test_multisector_flux_outputs_support_inversion_grid_mean_stats(
    multisector_postprocessing_inv_out: Callable[..., InversionOutput],
) -> None:
    """Multisector inversion-grid stats work when statistics do not need dense quantiles."""
    outputs = make_flux_outputs(
        multisector_postprocessing_inv_out(),
        stats=["mean", "stdev"],
        include_scale_factors=False,
        report_flux_on_inversion_grid=True,
    )

    assert "flux_ff_posterior_mean" in outputs
    assert "flux_ocean_posterior_mean" in outputs
    assert "flux_total_posterior_mean" in outputs
    assert float(outputs["flux_total_posterior_mean"].item()) == 2.0


def test_multisector_flux_outputs_support_source_specific_basis(
    monkeypatch: pytest.MonkeyPatch,
    fake_source_specific_multisector_basis_functions: Callable[..., BasisFunctions],
    multisector_postprocessing_inv_out: Callable[..., InversionOutput],
) -> None:
    """Sector flux reconstruction handles source-specific retained basis artifacts."""
    basis_functions = fake_source_specific_multisector_basis_functions()

    def fail_flat_basis(self: BasisFunctions) -> xr.DataArray:
        raise AssertionError("source-specific multisector outputs should not materialise flat basis")

    monkeypatch.setattr(type(basis_functions), "flat_basis", fail_flat_basis)

    outputs = make_flux_outputs(
        multisector_postprocessing_inv_out(basis_functions),
        stats=["mean", "mode_kde"],
        include_scale_factors=False,
        report_flux_on_inversion_grid=False,
    )

    assert "flux_total_posterior_mean" in outputs
    assert "flux_total_posterior_mode" in outputs
    assert float(outputs["flux_total_posterior_mean"].item()) == 2.0
    assert _flux_nonfinite_metadata(outputs).policy == NONFINITE_POLICY_ZERO_FILL


def test_multisector_flux_trace_materialization_is_optional(
    multisector_postprocessing_inv_out: Callable[..., InversionOutput],
) -> None:
    """The trace API preserves its eager default and exposes a lazy internal boundary."""
    inv_out = multisector_postprocessing_inv_out()

    lazy_trace = make_multisector_flux_trace_outputs(
        inv_out,
        report_flux_on_inversion_grid=False,
        materialize=False,
    )
    materialized_trace = make_multisector_flux_trace_outputs(
        inv_out,
        report_flux_on_inversion_grid=False,
    )

    assert not isinstance(lazy_trace["flux_total_posterior"].data, np.ndarray)
    assert isinstance(materialized_trace["flux_total_posterior"].data, np.ndarray)
