"""Compare the two original linked offset fits with bounded saved-array reads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.serialization import restore_declared_multiindexes


DEFAULT_RETRY = Path(
    "/group/chem/acrg/verification_games_round_2/verification-games-work/"
    "ope225-bc-offset-20261029-r1/linked-uob-retry-r2"
)
TIMING_STATS = {"perf_counter_diff", "perf_counter_start", "process_time_diff"}
ARMS = ("anchored_site_offsets", "free_site_offsets")


def compare_array(actual: xr.DataArray, expected: xr.DataArray) -> dict:
    """Read one chain and at most sixteen draws from each file at a time."""
    assert actual.dims == expected.dims
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.attrs == expected.attrs
    actual_hash, source_hash = hashlib.sha256(), hashlib.sha256()
    max_error = 0.0
    unequal = nonfinite = source_nonfinite = 0
    chains = range(actual.sizes["chain"]) if "chain" in actual.dims else [None]
    starts = range(0, actual.sizes["draw"], 16) if "draw" in actual.dims else [None]
    for chain in chains:
        for start in starts:
            indexers = {}
            if chain is not None:
                indexers["chain"] = chain
            if start is not None:
                indexers["draw"] = slice(start, start + 16)
            left = np.asarray(actual.isel(indexers).values)
            right = np.asarray(expected.isel(indexers).values)
            actual_hash.update(np.ascontiguousarray(left).view(np.uint8).data)
            source_hash.update(np.ascontiguousarray(right).view(np.uint8).data)
            if left.dtype.kind in "fc":
                mask = np.isfinite(left) & np.isfinite(right)
                max_error = max(max_error, float(np.max(np.abs(left[mask] - right[mask]), initial=0)))
                unequal += int(np.count_nonzero(~((left == right) | (np.isnan(left) & np.isnan(right)))))
                nonfinite += int(np.count_nonzero(~np.isfinite(left)))
                source_nonfinite += int(np.count_nonzero(~np.isfinite(right)))
            else:
                unequal += int(np.count_nonzero(left != right))
    return {
        "dims": list(actual.dims), "shape": list(actual.shape), "dtype": str(actual.dtype),
        "units": actual.attrs.get("units"), "attributes_match": True,
        "sha256": actual_hash.hexdigest(), "source_sha256": source_hash.hexdigest(),
        "bytes_exact": actual_hash.hexdigest() == source_hash.hexdigest(),
        "unequal_elements": unequal, "max_absolute_difference": max_error,
        "nonfinite_elements": nonfinite, "source_nonfinite_elements": source_nonfinite,
    }


def validate_arm(actual_dir: Path, retry_dir: Path, results: dict, output: Path) -> dict:
    result = results["arms"].setdefault(actual_dir.name, {"actual": str(actual_dir), "source": str(retry_dir), "groups": {}})
    with xr.open_datatree(actual_dir / "posterior.nc") as actual, xr.open_datatree(retry_dir / "posterior.nc") as expected:
        assert set(actual.children) == set(expected.children)
        for key in ("rhime_recipe", "boundary_representation", "rhime_priors", "rhime_variable_roles", "rhime_model_metadata"):
            assert actual.attrs[key] == expected.attrs[key], key
        result["public_root_attributes_match"] = True
        for name, node in actual.children.items():
            if name in result["groups"]:
                print(json.dumps({"arm": actual_dir.name, "group_resumed": name}), flush=True)
                continue
            left = restore_declared_multiindexes(node.to_dataset())
            right = restore_declared_multiindexes(expected[name].to_dataset())
            assert dict(left.sizes) == dict(right.sizes)
            assert set(left.data_vars) == set(right.data_vars)
            left_coords, right_coords = left.coords.to_dataset(), right.coords.to_dataset()
            left_coords.attrs, right_coords.attrs = {}, {}
            xr.testing.assert_identical(left_coords, right_coords)
            attribute_differences = {
                key: {"actual": np.asarray(left.attrs.get(key)).tolist(), "source": np.asarray(right.attrs.get(key)).tolist()}
                for key in set(left.attrs) | set(right.attrs)
                if not np.array_equal(left.attrs.get(key), right.attrs.get(key))
            }
            assert set(attribute_differences) <= {"created_at", "sampling_time"}
            indexes = {}
            for dim in left.dims:
                if isinstance(left.indexes.get(dim), pd.MultiIndex):
                    pd.testing.assert_index_equal(left.indexes[dim], right.indexes[dim], exact=True)
                    indexes[dim] = {"names": left.indexes[dim].names, "size": left.sizes[dim]}
            result["groups"][name] = {
                "sizes": dict(left.sizes), "coordinates_match": True, "multiindexes": indexes,
                "group_attribute_differences": attribute_differences,
                "variables": {var: compare_array(left[var], right[var]) for var in left.data_vars},
            }
            output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"arm": actual_dir.name, "group_complete": name}), flush=True)
        posterior = actual["posterior"].to_dataset()
        assert posterior.sizes["chain"] == 4 and posterior.sizes["draw"] == 1000
        assert posterior.sizes["observation"] == 10628
        for species in ("co2", "o2"):
            assert int((posterior.species == species).sum()) == 5314
        event_names = [n for n in actual.sample_stats.data_vars if n.endswith(("diverging", "reached_max_treedepth"))]
        events = {n: int(actual.sample_stats[n].sum()) for n in event_names}
        result["sampler_events"] = events
        assert events and not any(events.values())
        assert actual.log_likelihood.y.dims == ("chain", "draw")
        assert actual.log_likelihood.y.attrs["rhime_likelihood_scope"] == "joint_observation_vector"
    with xr.open_dataset(actual_dir / "diagnostics.nc") as actual, xr.open_dataset(retry_dir / "diagnostics.nc") as expected:
        actual, expected = (restore_declared_multiindexes(ds).load() for ds in (actual, expected))
        xr.testing.assert_identical(actual, expected)
        rhats = np.concatenate([a.sel(metric="r_hat").values.ravel() for a in actual.data_vars.values()])
        ess = np.concatenate([a.sel(metric="ess_bulk").values.ravel() for a in actual.data_vars.values()])
        finite = bool(np.isfinite(rhats).all() and np.isfinite(ess).all())
        result["selected_parameter_diagnostics"] = {
            "variables": list(actual.data_vars), "all_metrics_match_retry_exactly": True,
            "finite_rhat_and_bulk_ess": finite, "max_rhat": float(rhats.max()),
            "min_bulk_ess": float(ess.min()), "original_gate_passed": bool(finite and rhats.max() <= 1.01 and ess.min() >= 100 and not any(events.values())),
        }
        assert result["selected_parameter_diagnostics"]["original_gate_passed"]
    scientific = [item for group, data in result["groups"].items() for var, item in data["variables"].items() if group != "sample_stats" or var not in TIMING_STATS]
    result["all_non_timing_values_equal_including_matching_nan"] = all(item["unequal_elements"] == 0 for item in scientific)
    result["all_non_timing_bytes_exact"] = all(item["bytes_exact"] for item in scientific)
    result["all_posterior_and_likelihood_values_finite"] = all(item["nonfinite_elements"] == 0 for group in ("posterior", "log_likelihood") for item in result["groups"][group]["variables"].values())
    assert result["all_posterior_and_likelihood_values_finite"]
    assert result["all_non_timing_values_equal_including_matching_nan"]
    assert result["all_non_timing_bytes_exact"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--retry", type=Path, default=DEFAULT_RETRY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse completed groups from this output JSON for the same immutable fit files")
    args = parser.parse_args()
    start = time.monotonic()
    result = {
        "method": "Lazy NetCDF reads, one chain and at most sixteen draws per payload; full static arrays are bounded by model design size",
        "timing_stat_exclusions": sorted(TIMING_STATS),
        "diagnostic_scope": "Original selected parameter gates only; pre-existing derived common-mode max R-hat 1.01048914016785 exceeds 1.01 and is not reclassified",
        "arms": {},
    }
    if args.resume:
        previous = json.loads(args.output.read_text())
        result["arms"] = previous["arms"]
        for arm, completed in result["arms"].items():
            assert completed["actual"] == str(args.acceptance.resolve() / arm)
            assert completed["source"] == str(args.retry.resolve() / arm)
        result["resumed_groups"] = {arm: list(data["groups"]) for arm, data in result["arms"].items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        result["arms"][arm] = validate_arm(args.acceptance.resolve() / arm, args.retry.resolve() / arm, result, args.output)
    result.update(status="passed", elapsed_seconds=time.monotonic() - start, maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"], "maxrss_kib": result["maxrss_kib"]}))


if __name__ == "__main__":
    main()
