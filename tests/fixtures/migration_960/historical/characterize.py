"""Run only against the isolated v0.7.4 checkout; never acquires data or samples.

Expectations in expected.json were transcribed from source bodies, independently
of this runner. Historical imports deliberately stay outside normal pytest so
these plain examples survive removal of the HBMCMC package.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import logging
import logging.handlers  # Define FileHandler subclasses before instrumentation.
from pathlib import Path
from unittest.mock import patch


class StopBeforeAcquisition(Exception):
    """Sentinel marking the first data retrieval boundary."""


def main() -> None:
    fixture_dir = Path(__file__).resolve().parent
    assert importlib.metadata.version("openghg-inversions") == "0.7.4"
    spec = importlib.util.find_spec("openghg_inversions")
    source_root = Path(spec.origin).resolve().parent.parent
    hashes = json.loads((fixture_dir / "source_hashes.json").read_text())
    for filename, expected_hash in hashes.items():
        assert hashlib.sha256((source_root / filename).read_bytes()).hexdigest() == expected_hash, filename
    # OpenGHG 0.16 unconditionally opens ~/openghg.log on import. Suppress only
    # this logger; do not repurpose HOME or write to the user's home directory.
    with patch("logging.FileHandler", lambda *args, **kwargs: logging.NullHandler()):
        import openghg  # noqa: F401

    from openghg_inversions import cli
    import openghg_inversions.rhime as rhime
    from openghg_inversions.rhime import standard
    from openghg_inversions.hbmcmc import run_hbmcmc as wrapper
    from openghg_inversions.postprocessing.inversion_output import convert_idata_to_dataset
    import arviz as az
    import pymc as pm
    import xarray as xr

    expected = json.loads((fixture_dir / "expected.json").read_text())
    observed = {}

    def capture_rhime(**kwargs):
        with (
            patch.object(standard, "resolve_rhime_options", wraps=standard.resolve_rhime_options) as resolve,
            patch.object(standard, "retrieve_or_reload_rhime_data", side_effect=StopBeforeAcquisition),
        ):
            try:
                standard.run_rhime(**kwargs)
            except StopBeforeAcquisition:
                pass
            else:
                raise AssertionError("Expected to stop at data retrieval")
        # Resolve a second time only to obtain the resolved object, not expected
        # values. All expectations are hand-written in the adjacent JSON.
        setup = standard.resolve_rhime_options(**resolve.call_args.kwargs)
        output = setup.run_spec.output
        return {
            **{name: getattr(setup.sampler, name) for name in ("draws", "burn", "tune", "chains")},
            "output_format": output.output_format,
            "save_trace": output.save_trace,
            "save_inversion_output": output.save_inversion_output,
            "filename_convention": output.output_filename_convention,
            "sigma_prior": setup.run_spec.model.likelihood.sigma_prior,
        }

    observed["python_rhime"] = capture_rhime(config_file=fixture_dir / "rhime.ini")
    with patch.object(rhime, "run_rhime", side_effect=capture_rhime) as dispatch:
        cli.main(["run-rhime", "-c", str(fixture_dir / "rhime.ini")])
        observed["installed_cli_dispatch"] = dispatch.side_effect(**dispatch.call_args.kwargs)

    for name, filename in (("ordinary_wrapper", "wrapper.ini"), ("explicit_wrapper", "sampling.ini")):
        with (
            patch.object(wrapper, "run_rhime", side_effect=capture_rhime) as dispatch,
            patch.object(wrapper.output, "copy_config_file"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            wrapper.main(["-c", str(fixture_dir / filename)])
            observed[name] = dispatch.side_effect(**dispatch.call_args.kwargs)

    signature = inspect.signature(wrapper.fixedbasisMCMC)
    with (
        patch.object(wrapper, "fixedbasisMCMC") as dispatch,
        patch.object(wrapper.output, "copy_config_file"),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        wrapper.main(["--legacy-fixedbasis", "-c", str(fixture_dir / "wrapper.ini")])
        bound = signature.bind(**dispatch.call_args.kwargs)
        bound.apply_defaults()
        observed["true_legacy_fixedbasis"] = {
            name: bound.arguments[name] for name in expected["routes"]["true_legacy_fixedbasis"]
        }

    assert observed == expected["routes"], (observed, expected["routes"])
    errors = {}
    for name, call in (
        ("python_omits_mismatch_model", lambda: standard.run_rhime(config_file=fixture_dir / "rhime.ini", mismatch_model=None)),
        ("true_legacy_rejects_modern_draws", lambda: wrapper._prepare_legacy_fixedbasis_params({"draws": 7})),
    ):
        try:
            call()
        except ValueError as error:
            errors[name] = str(error)
        else:
            raise AssertionError(f"Expected {name} to reject the request")
    assert errors == expected["observed_errors"]

    # Supply an artificial trace at pm.sample; this executes burn slicing, not
    # MCMC. Expected five retained draws comes from 7 - 2 independently.
    sampler = rhime.RhimeSampler(
        draws=7, burn=2, tune=3, chains=2,
        sample_prior_predictive=False, sample_posterior_predictive=False,
    )
    raw_trace = az.InferenceData(posterior=xr.Dataset({"x": (("chain", "draw"), [[0] * 7] * 2)}))
    with patch.object(pm, "sample", return_value=raw_trace) as sample:
        trace = sampler.sample(pm.Model())
    assert sample.call_args.kwargs["draws"] == 7
    assert sample.call_args.kwargs["tune"] == 3
    assert trace.posterior.sizes["draw"] == 5
    assert trace.posterior.draw.values.tolist() == [0, 1, 2, 3, 4]

    # Source policy selects chain zero before derived summaries; raw trace
    # storage itself can still contain every chain.
    witness = expected["scientific_cases"]["chain_summary"]
    trace = az.InferenceData(posterior=xr.Dataset({"x": (("chain", "draw"), witness["input_chains"])}))
    summary = convert_idata_to_dataset(trace)
    assert float(summary.x_posterior.mean()) == witness["released_chain0_mean"]
    print(json.dumps({"status": "passed", "routes": observed, "errors": errors,
                      "mock_sampling_retained_draws": 5, "chain0_witness_mean": 2}, indent=2))


if __name__ == "__main__":
    main()
