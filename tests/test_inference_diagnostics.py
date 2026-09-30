"""Convergence diagnostics retain chains and require no model graph."""

import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.inference.diagnostics import posterior_summary


def test_posterior_summary_preserves_chains_and_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assessment receives the exact posterior, including distinct chain labels."""
    posterior = xr.Dataset(
        {"state": (("chain", "draw"), [[1.0, 2.0], [10.0, 20.0]])},
        coords={"chain": ["first", "second"], "draw": [4, 5]},
    )
    original = posterior.copy(deep=True)

    def fake_summary(data: xr.Dataset, **kwargs: object) -> xr.Dataset:
        assert data is posterior
        assert kwargs == {"kind": "diagnostics", "fmt": "xarray", "round_to": "none"}
        return xr.Dataset(
            {"state": ("summary", [1.0101, 399.99, np.nan])},
            coords={"summary": ["r_hat", "ess_bulk", "mcse_mean"]},
        )

    monkeypatch.setattr("openghg_inversions.inference.diagnostics.az.summary", fake_summary)
    result = posterior_summary(posterior)

    xr.testing.assert_identical(posterior, original)
    assert result.metric.values.tolist() == ["r_hat", "ess_bulk", "mcse_mean"]
    assert result.state.sel(metric="r_hat").item() == 1.0101
    assert result.state.sel(metric="ess_bulk").item() == 399.99
    assert np.isnan(result.state.sel(metric="mcse_mean").item())


def test_posterior_summary_import_is_backend_neutral() -> None:
    """Loading saved-trace diagnostics does not initialize PyMC or recipes."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import openghg_inversions.inference.diagnostics; "
            "assert 'pymc' not in sys.modules; "
            "assert 'pytensor' not in sys.modules; "
            "assert not any(name.startswith(('openghg_inversions.recipes', "
            "'openghg_inversions.rhime')) for name in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_inference_sampler_export_preserves_identity() -> None:
    """The lazy public sampler export remains the shared implementation."""
    from openghg_inversions import inference
    from openghg_inversions.inference.sampling import RhimeSampler

    assert inference.__all__ == ["RhimeSampler"]
    assert inference.RhimeSampler is RhimeSampler
    with pytest.raises(AttributeError, match="unknown_sampler"):
        getattr(inference, "unknown_sampler")
