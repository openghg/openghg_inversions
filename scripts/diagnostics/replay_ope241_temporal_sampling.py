"""Optional OPE-241 replay against private, immutable native CORSO evidence.

Run on a Slurm compute node with the Verification Games environment and this
checkout on PYTHONPATH. Outputs go to a new directory; no fitting or input
updates occur. See README.md for input contracts and the remaining limitations.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


POLICIES = {"midpoint_ffill": "hold", "nearest_midpoint": "nearest", "linear_midpoints": "linear"}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def checked(path, expected, sources):
    path = Path(path)
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"Archive identity changed: {path}: {actual} != {expected}")
    sources[str(path)] = actual
    return path


def close(actual, expected, *, atol=1e-8):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def row_index(data):
    return pd.MultiIndex.from_arrays(
        [np.asarray(data.site).astype(str), np.asarray(data.time).astype("datetime64[ns]")],
        names=["site", "time"],
    )


def align_frame(frame, rows):
    index = row_index(frame)
    if not index.is_unique or not rows.is_unique or len(index) != len(rows):
        raise ValueError("Replay rows must be a unique complete site/time match")
    positions = index.get_indexer(rows)
    if (positions < 0).any():
        raise ValueError("Replay rows differ from the selected archive")
    return frame.iloc[positions]


def sample_sites(spec_path, output, sources):
    """Use production operands with the archived independent endpoint overlay owner."""
    from openghg_inversions.inversion_data import BoundaryTemporalPolicy
    from openghg_inversions.inversion_data._boundary import baseline_reference_sensitivities
    from verification_games.footprint_endpoints import apply_endpoint_overlay
    from verification_games.metadata import append_history

    spec = json.loads(spec_path.read_text())
    spec_hash = digest(spec_path)
    sources[str(spec_path)] = spec_hash
    checked(spec["repair_spec"]["path"], spec["repair_spec"]["sha256"], sources)
    owners = {"footprint_endpoints.py": apply_endpoint_overlay, "metadata.py": append_history}
    for dependency in spec["dependencies"]:
        checked(inspect.getfile(owners[Path(dependency["path"]).name]), dependency["sha256"], sources)
    operands, records = {}, []
    with ExitStack() as boundary_stack:
        native_fields = {}
        for species in ("co2", "o2"):
            source = spec["boundary"][species]
            boundary = checked(source["path"], source["sha256"], sources)
            opened = boundary_stack.enter_context(xr.open_dataset(boundary))
            native = opened.rename(level="height") if "level" in opened.dims else opened
            if native.vmr_n.attrs["units"] not in {"mol mol-1", "mol/mol"}:
                raise ValueError("Native oracle expects unscaled mole fraction source units")
            times = native.time.values
            np.testing.assert_array_equal(np.diff(times), np.full(len(times) - 1, np.timedelta64(3, "h")))
            # Explicit sensitivity hypothesis from the archive, not producer metadata.
            bounds = np.column_stack([times - np.timedelta64(90, "m"), times + np.timedelta64(90, "m")])
            native_fields[species] = native.assign(time_bounds=(("time", "bounds"), bounds)).assign_attrs(
                time_period="3h"
            )
        for site in spec["sites"]:
            name = site["site"]
            checked(Path(site["path"]) / ".zmetadata", site["zmetadata_sha256"], sources)
            oracle_path = spec_path.parent / "sites" / f"{name}.nc"
            receipt_path = oracle_path.with_suffix(".json")
            receipt = json.loads(receipt_path.read_text())
            if receipt["spec_sha256"] != spec_hash:
                raise ValueError(f"Wrong oracle specification for {name}")
            checked(oracle_path, receipt["sha256"], sources)
            with ExitStack() as stack:
                oracle = stack.enter_context(xr.open_dataset(oracle_path)).sel(endpoint_version="repaired").load()
                fp = stack.enter_context(xr.open_zarr(site["path"], chunks=None)).sel(time=oracle.time)
                if "overlay" in site:
                    overlay_path = checked(site["overlay"]["path"], site["overlay"]["sha256"], sources)
                    overlay = stack.enter_context(xr.open_dataset(overlay_path))
                    fp = apply_endpoint_overlay(fp, overlay, parent_path=site["path"])
                # The archived particle fractions use an empty dimensionless
                # unit label; declare it explicitly for the production kernel.
                if any(fp[f"particle_locations_{side}"].attrs.get("units", "") not in {"", "1", "dimensionless"}
                       for side in "nesw"):
                    raise ValueError("Unexpected particle-fraction units in the frozen archive")
                fp = fp.assign({
                    f"particle_locations_{side}": fp[f"particle_locations_{side}"].assign_attrs(units="1")
                    for side in "nesw"
                })
                species_results = []
                for species, reference in (("co2", 400.), ("o2", 209300.)):
                    native = native_fields[species]
                    # Declare exact labels after checking the same producer rounding
                    # tolerances used by the independent native oracle.
                    for coord, tolerance in (("height", .001), ("lat", .01), ("lon", .01)):
                        close(native[coord], fp[coord], atol=tolerance)
                    bc = native.assign_coords({coord: fp[coord] for coord in ("height", "lat", "lon")})
                    site_results = []
                    for policy_name, method in {**POLICIES, "interval_containment": "interval"}.items():
                        if method == "interval":
                            policy = BoundaryTemporalPolicy(
                                method="interval", input_semantics="interval_mean", bounds="time_bounds",
                            )
                        else:
                            policy = BoundaryTemporalPolicy(
                                method=method, input_semantics="point" if method != "hold" else None,
                                max_gap="3h" if method != "hold" else None,
                            )
                        batches = []
                        for start in range(0, fp.sizes["time"], 24):
                            batch = fp.isel(time=slice(start, start + 24))
                            values = baseline_reference_sensitivities(
                                bc, batch, species=species, reference=reference, reference_units="ppm",
                                mass_tolerance=.001, temporal_policy=policy,
                            )
                            reduced = {}
                            for kind in ("mass", "anomaly"):
                                curtains = []
                                for side in "nesw":
                                    array = values[f"bc_{kind}_{side}"]
                                    curtains.append(array.sum([d for d in array.dims if d != "time"], skipna=False))
                                reduced[kind] = xr.concat(curtains, dim=pd.Index(list("nesw"), name="bc_region"))
                            # Named shared materialization boundary for both operands.
                            batches.append(xr.Dataset(reduced).compute())
                        result = xr.concat(batches, dim="time").transpose("time", "bc_region")
                        result["anomaly"] = result.anomaly * 1e6
                        archive_policy = "interval_containment" if method == "nearest" else policy_name
                        target = oracle.sel(species=species, policy=archive_policy)
                        errors = {
                            "mass_max_abs": close(result.mass, target.curtain_mass, atol=1e-12),
                            "anomaly_max_abs_ppm": close(result.anomaly, target.curtain_anomaly_contribution),
                            "absolute_max_abs_ppm": close(
                                (result.anomaly + reference * result.mass).sum("bc_region", skipna=False),
                                target.boundary_contribution,
                            ),
                        }
                        records.append({"site": name, "species": species, "policy": policy_name,
                                        "rows": fp.sizes["time"], **errors})
                        site_results.append(result.expand_dims(policy=[policy_name]))
                    species_results.append(xr.concat(site_results, dim="policy").expand_dims(species=[species]))
                combined = xr.concat(species_results, dim="species")
                combined.to_netcdf(output / "sites" / f"{name}.nc")
                operands[name] = combined.sel(species="co2")
                print(json.dumps({"site": name, "status": "native_oracle_passed"}), flush=True)
    return operands, records


def replay_design(design, operands):
    """Reassemble the archived scientific equation with frozen calibration and rows."""
    attrs = design.attrs
    n = design.sizes["nmeasure"]
    mass, a400 = np.empty((n, 4)), np.empty((n, 4))
    sites = design.site.values.astype(str)
    for site in dict.fromkeys(sites):
        rows = np.flatnonzero(sites == site)
        values = operands[site].sel(policy=attrs["policy"], time=design.time.values[rows],
                                    bc_region=design.bc_region)
        mass[rows], a400[rows] = values.mass, values.anomaly
    c, width = attrs["reference_ppm"], attrs["pattern_coefficient_sd"]
    anomaly = a400 + (400 - c) * mass
    g = mass.sum(axis=1)
    h = anomaly + c * mass
    physical = h.sum(axis=1)
    kind = attrs["model_kind"]
    operator = h if kind == "multiplicative" else width * anomaly
    if attrs["frequency"] == "8D":
        period = design.bc_period_index.values
        # The archived gate is part of the fixed experiment, not a new calendar policy.
        gate = period[:, None] == np.arange(4)[None, :]
        operator = (gate[:, :, None] * operator[:, None, :]).reshape(n, 16)
    if kind == "multiplicative":
        factor, fixed = operator, np.zeros(n)
    elif kind == "delta_only":
        factor, fixed = 5 * g[:, None], physical
    elif kind == "delta_A":
        factor, fixed = np.column_stack([5 * g, operator]), physical
    else:
        raise ValueError(f"Unsupported archived model kind: {kind}")
    errors = {
        "mass": close(mass, design.H_bc_mass, atol=1e-12),
        "anomaly_ppm": close(anomaly, design.H_bc_anomaly),
        "correction_design_ppm": close(factor, design.correction_design),
        "fixed_boundary_ppm": close(fixed, design.fixed_boundary),
        "physical_bc_prior_ppm": close(physical, design.physical_bc_prior),
        "reference_shift_ppm": close(c * g, design.reference_shift),
        "centered_physical_bc_ppm": close(anomaly.sum(axis=1), design.centered_physical_bc),
    }
    return factor, fixed, errors


def replay_selected(selection_path, operands, sources):
    from openghg_inversions.serialization import restore_serialisation_multiindexes

    selected = json.loads(selection_path.read_text())["selection"]
    sources[str(selection_path)] = digest(selection_path)
    records = []
    for arm, entry in selected.items():
        score = Path(entry["score_directory"])
        provenance_path = checked(score / "provenance.json", entry["provenance_sha256"], sources)
        provenance = json.loads(provenance_path.read_text())
        if provenance["arm"] != arm or not provenance["fit_manifest"]["convergence_gate_passed"]:
            raise ValueError(f"Selected fit not accepted: {arm}")
        root = score.parent.parent
        design_path = root / "designs" / f"{arm}.nc"
        checked(design_path, provenance["fit_manifest"]["design_sha256"], sources)
        checked(root / "prepared.nc", provenance["fit_manifest"]["prepared_sha256"], sources)
        with xr.open_dataset(design_path) as opened:
            design = restore_serialisation_multiindexes(opened.load(), strict=True)
        if design.attrs["arm"] != arm or design.sizes["nmeasure"] != provenance["rows"]:
            raise ValueError(f"Selected design identity mismatch: {arm}")
        rows = row_index(design)
        factor, fixed, errors = replay_design(design, operands)
        concentration = align_frame(pd.read_csv(score / "concentration_components.csv.gz"), rows)
        uncertainty = align_frame(pd.read_csv(score / "bc_pointwise_sd.csv.gz"), rows)
        trace_path = checked(root / "fits" / arm / "posterior.nc", provenance["trace_sha256"], sources)
        with xr.open_dataset(trace_path, group="posterior") as opened:
            coefficient = opened.bc
            dims = [d for d in coefficient.dims if d not in {"chain", "draw"}]
            if len(dims) != 1:
                raise ValueError("Expected one labelled boundary coefficient dimension")
            dim = dims[0]
            coefficients = coefficient.sel({dim: design.bc_term.values}).transpose("chain", "draw", dim).values
        for name in ("concentration_components.csv.gz", "bc_pointwise_sd.csv.gz"):
            sources[str(score / name)] = digest(score / name)
        mean = coefficients.mean(axis=(0, 1))
        covariance = np.atleast_2d(np.cov(coefficients.reshape(-1, len(mean)), rowvar=False))
        errors.update(
            posterior_bc_mean_ppm=close(fixed + factor @ mean, concentration.bc_posterior),
            prior_bc_mean_ppm=close(
                fixed + factor @ np.full(len(mean), design.attrs["prior_mu"]), concentration.bc_prior
            ),
            prior_bc_sd_ppm=close(
                design.attrs["prior_sigma"] * np.linalg.norm(factor, axis=1), uncertainty.bc_prior_sd_ppm
            ),
            posterior_bc_sd_ppm=close(
                np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", factor, covariance, factor), 0)),
                uncertainty.bc_posterior_sd_ppm,
            ),
        )
        records.append({"arm": arm, "score_directory": str(score), "rows": len(rows),
                        "reference_ppm": design.attrs["reference_ppm"],
                        "pattern_coefficient_sd": design.attrs["pattern_coefficient_sd"],
                        "max_abs_errors": errors, "convergence_gate_passed": True})
    if len(records) != 15:
        raise ValueError("Expected all 15 authoritative selected timing/family cases")
    return records


def self_check():
    # Labelled replay must follow the selected row order, including permutations.
    rows = pd.DataFrame({"site": ["B", "A"], "time": ["2021-07-01", "2021-07-02"], "x": [2, 1]})
    np.testing.assert_array_equal(align_frame(rows, row_index(rows)[::-1]).x, [1, 2])
    try:
        align_frame(rows.iloc[[0, 0]], row_index(rows))
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate rows were accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-spec", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_test:
        print("Row alignment self-check passed")
        return
    if not all((args.timing_spec, args.selection, args.output)):
        parser.error("--timing-spec, --selection and --output are required")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "sites").mkdir()
    sources = {}
    operands, native = sample_sites(args.timing_spec, args.output, sources)
    selected = replay_selected(args.selection, operands, sources)
    result = {"status": "passed", "native_oracle": native, "selected_replays": selected,
              "source_sha256": sources, "script_sha256": digest(__file__),
              "diagnostic_input_contract": {
                  "hold_time_period": "3h, checked against the complete native grid",
                  "interval_bounds": "midpoint +/- 90 minutes, sensitivity hypothesis",
                  "particle_units": "explicit dimensionless fractions (archived empty labels)",
              },
              "limitations": ["Private July 2021 CORSO CO2/O2 evidence; producer time semantics unresolved",
                              "Frozen c, calibration, observation rows and posterior draws; no re-fitting",
                              "Nearest/interval equivalence is specific to this verified regular 3h grid",
                              "Posterior BC means and uncertainties replayed; other scores retained by provenance"]}
    (args.output / "replay.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "native_cases": len(native), "selected_cases": len(selected)}))


if __name__ == "__main__":
    main()
