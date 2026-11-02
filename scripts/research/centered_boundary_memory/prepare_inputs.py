"""Freeze actual MHD or MHD/BIR/BSD/TAC inputs for the bounded memory experiment."""

import argparse
import hashlib
import json
from pathlib import Path

import h5netcdf
import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.serialization import decode_cf_multiindexes, encode_cf_multiindexes

ROOT = Path("/group/chem/acrg/verification_games_round_2")
SOURCE = (
    ROOT
    / "verification-games-work/ope188-20261020-uob-base-base-july-v1/public-assembly-pilot-r1/shared_uob_base.nc"
)
CORRECTED = ROOT / "forward_model_intermediates/uob_o2_baseline_20261022_ope231"
REFERENCE = {"co2": 420.0, "o2": 209300.0}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def decode(ds):
    return decode_cf_multiindexes(ds, json.loads(ds.attrs["multiindex_dims_json"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", choices=("MHD", "BIR", "BSD", "TAC"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert not (args.out / "config.json").exists(), "Use a fresh immutable input directory"
    spec = json.loads((CORRECTED / "spec.json").read_text())
    manifest = {
        "scope": "technical memory benchmark; scientific acceptance pending",
        "source": str(SOURCE),
        "source_sha256": sha(SOURCE),
        "spec_sha256": sha(CORRECTED / "spec.json"),
        "producer_sha256": sha(__file__),
        "sites": args.sites,
        "footprints": {},
        "boundary_files": {},
        "checks": {},
    }
    campaign_manifest = json.loads((SOURCE.parent / "manifest.json").read_text())
    assert manifest["source_sha256"] == campaign_manifest["shared_prepared"]["sha256"]
    with xr.open_dataset(SOURCE, group="joint", engine="h5netcdf") as opened:
        joint = decode(opened)
        rows = np.flatnonzero(np.isin(joint.site.values, args.sites))
        small = (
            joint.drop_vars("aggregation_error_covariance")
            .isel(observation=rows, observation_cov=rows)
            .load()
        )
        covariance_attrs = dict(joint.aggregation_error_covariance.attrs)
    # Read selected covariance axes without loading the full 10628-square matrix.
    with h5netcdf.File(SOURCE, "r") as handle:
        var = handle.groups["joint"].variables["aggregation_error_covariance"]
        covariance = np.stack([np.asarray(var[int(i), :])[rows] for i in rows])
    np.testing.assert_allclose(covariance, covariance.T, rtol=0, atol=1e-10)
    small["aggregation_error_covariance"] = (("observation", "observation_cov"), covariance, covariance_attrs)
    nodes = {"joint": small}
    index = small.indexes["observation"]
    channel_labels = {}
    for channel in REFERENCE:
        channel_labels[channel] = (
            index[index.get_level_values("species") == channel]
            .get_level_values("channel_observation")
            .to_numpy()
        )
    for name in ("co2_sensitivity", "o2_sensitivity", "retained_prior"):
        with xr.open_dataset(SOURCE, group=name, engine="h5netcdf") as opened:
            ds = decode(opened)
            if name.endswith("_sensitivity"):
                channel = name.split("_")[0]
                ds = ds.sel({f"{channel}_measure": channel_labels[channel]})
            nodes[name] = ds.load()
    encoded = {
        name: encode_cf_multiindexes(ds, json.loads(ds.attrs["multiindex_dims_json"]))
        if json.loads(ds.attrs["multiindex_dims_json"])
        else ds
        for name, ds in nodes.items()
    }
    with xr.open_dataset(SOURCE, engine="h5netcdf") as opened:
        attrs = dict(opened.attrs)
    provenance = json.loads(attrs["provenance_json"])
    provenance.update(
        memory_benchmark_sites=args.sites, memory_benchmark_source_sha256=manifest["source_sha256"]
    )
    attrs["provenance_json"] = json.dumps(provenance)
    tree = xr.DataTree.from_dict(encoded)
    tree.attrs = attrs
    tree.to_netcdf(args.out / "native.nc", engine="h5netcdf")
    operands = {channel: [] for channel in REFERENCE}
    for site in args.sites:
        record = next(item for item in spec["sites"] if item["site"] == site)
        channel_rows = np.flatnonzero((small.species.values == "co2") & (small.site.values == site))
        times = small.time.values[channel_rows]
        labels = index[channel_rows].get_level_values("channel_observation").to_numpy()
        oxygen_rows = np.flatnonzero((small.species.values == "o2") & (small.site.values == site))
        np.testing.assert_array_equal(times, small.time.values[oxygen_rows])
        np.testing.assert_array_equal(
            labels, index[oxygen_rows].get_level_values("channel_observation").to_numpy()
        )
        assert times.size == {"MHD": 129, "BIR": 154, "BSD": 180, "TAC": 155}[site]
        with xr.open_zarr(record["path"], chunks=None) as source:
            fp = source.sel(time=times)
            weights = {
                d: fp[f"particle_locations_{d}"]
                .transpose("time", "height", "lon" if d in "ns" else "lat")
                .load()
                for d in "nesw"
            }
            mass = sum(w.values.astype("float64").sum(axis=(1, 2)) for w in weights.values())
            occupied = sum(np.count_nonzero(w.values, axis=(1, 2)) for w in weights.values())
            digest = hashlib.sha256(fp.time.values.astype("datetime64[ns]").tobytes())
            for direction, w in weights.items():
                values = w.values.astype("float64")
                assert np.isfinite(values).all() and (values >= 0).all()
                np.testing.assert_array_equal(values / 2.0**-24, np.rint(values / 2.0**-24))
                digest.update(direction.encode())
                digest.update(w.values.tobytes())
                digest.update(fp["lon" if direction in "ns" else "lat"].values.tobytes())
                digest.update(fp.height.values.tobytes())
            assert (abs(mass - 1) <= occupied * 2.0**-25 + 1e-12).all()
            assert (abs(mass - 1) < 0.001).all()
            manifest["footprints"][site] = {
                "path": record["path"],
                "record": record["record"],
                "particle_payload_sha256": digest.hexdigest(),
            }
            for channel, reference in REFERENCE.items():
                path = (
                    ROOT
                    / f"games_catalog/data/boundary_conditions/EUROPE/{channel}/CORSO_v2025/{channel}_EUROPE_CORSO_v2025_202012-202112.nc"
                )
                if channel not in manifest["boundary_files"]:
                    manifest["boundary_files"][channel] = {"path": str(path), "sha256": sha(path)}
                expected_sha = (
                    spec["bc"]["sha256"]
                    if channel == "o2"
                    else "643a89c07bbaed3719c822922a338c05349582996f9e005b6fadc23a1dbe2b78"
                )
                assert manifest["boundary_files"][channel]["sha256"] == expected_sha
                anomalies, masses, direct = [], [], []
                with xr.open_dataset(path, engine="h5netcdf") as bc:
                    selected = bc.indexes["time"].get_indexer(pd.DatetimeIndex(times), method="pad")
                    assert (selected >= 0).all()
                    age = (times - bc.time.values[selected]) / np.timedelta64(1, "h")
                    assert ((age >= 0) & (age <= 6)).all()
                    for direction in "nesw":
                        spatial = "lon" if direction in "ns" else "lat"
                        curtain = bc[f"vmr_{direction}"].rename(level="height")
                        np.testing.assert_allclose(curtain[spatial], fp[spatial], rtol=0, atol=0.01)
                        np.testing.assert_allclose(curtain.height, fp.height, rtol=0, atol=0.001)
                        values = (
                            curtain.isel(time=selected)
                            .transpose("time", "height", spatial)
                            .values.astype("float64")
                        )
                        assert np.isfinite(values).all()
                        assert (
                            ((values > 0.20) & (values < 0.22))
                            if channel == "o2"
                            else ((values > 0.0002) & (values < 0.0008))
                        ).all()
                        values *= 1e6  # Explicit repair of known CORSO unit metadata.
                        w = weights[direction].values.astype("float64")
                        masses.append(w.sum(axis=(1, 2)) / mass)
                        anomalies.append((w * (values - reference)).sum(axis=(1, 2)) / mass)
                        direct.append((w * values).sum(axis=(1, 2)) / mass)
                M, A, B = np.stack(masses, axis=1), np.stack(anomalies, axis=1), np.stack(direct, axis=1)
                np.testing.assert_allclose(A + reference * M, B, rtol=0, atol=2e-10)
                np.testing.assert_allclose(M.sum(axis=1), 1, rtol=0, atol=5e-15)
                if channel == "o2":
                    manifest["footprints"][site]["oracle_sha256"] = sha(CORRECTED / f"sites/{site}.nc")
                    with xr.open_dataset(CORRECTED / f"sites/{site}.nc") as oracle:
                        np.testing.assert_array_equal(
                            mass, oracle.raw_boundary_weight_sum.sel(time=times).values
                        )
                        np.testing.assert_allclose(
                            B.sum(axis=1), oracle.background.sel(time=times), rtol=0, atol=2e-10
                        )
                operands[channel].append(
                    xr.Dataset(
                        {
                            "H_bc_anomaly": (("nmeasure", "bc_region"), A, {"units": "ppm"}),
                            "H_bc_mass": (("nmeasure", "bc_region"), M, {"units": "1"}),
                            "bc_reference": xr.DataArray(reference, attrs={"units": "ppm"}),
                        },
                        coords={
                            "nmeasure": labels,
                            "bc_region": list("nesw"),
                            "time": ("nmeasure", times),
                            "site": ("nmeasure", [site] * times.size),
                        },
                        attrs={"boundary_representation": "centered-v1", "species": channel},
                    )
                )
                manifest["checks"][f"{channel}:{site}"] = {
                    "rows": times.size,
                    "direct_B_max_gap_ppm": float(abs(A + reference * M - B).max()),
                    "max_curtain_age_hours": float(age.max()),
                }
    for channel in REFERENCE:
        result = xr.concat(operands[channel], dim="nmeasure", data_vars="minimal")
        result = result.sel(nmeasure=channel_labels[channel])
        result.to_netcdf(args.out / f"{channel}_operands.nc", engine="h5netcdf")
    config = {
        "scope": "controlled_technical_benchmark_scientific_acceptance_pending",
        "prepared": str((args.out / "native.nc").resolve()),
        "site": args.sites,
        "boundary_operands": {c: str((args.out / f"{c}_operands.nc").resolve()) for c in REFERENCE},
        "observations": "controlled_model_consistent_truth_plus_joint_gaussian_noise",
        "observation_noise": {"seed": 20261027, "truth_ou_amplitude_ppm": {"co2": 0.5, "o2": 0.5}},
        "independent_error_sd_ppm": {"co2": 0.2, "o2": 0.2},
        "tau_hours": 5.0,
        "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.75},
        "bc_prior": {c: {"pdf": "normal", "mu": 1.0, "sigma": 1.0 / r} for c, r in REFERENCE.items()},
        "sampling": {"draws": 250, "tune": 250, "seed": 20261027},
        "label": "Bounded timing and memory experiment; no scientific acceptance claim",
    }
    manifest["files"] = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in args.out.glob("*.nc")}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=int) + "\n")
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
