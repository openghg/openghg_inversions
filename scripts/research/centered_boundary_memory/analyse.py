"""Validate and summarize the sixteen frozen PR986 memory measurements (stdlib only)."""

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

PHASES = ("prepare", "construction", "compile", "refresh", "sampling", "save")
VARIANTS = ("float64", "active_b32", "blocked64", "combined")
REVISION = "3dfc2b96886878535cb784f50616831175eb6311"
AUDIT_REVISION = "310ef7c7b5eae05a18d89918c7e3aac79eb7e00a"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sacct", type=Path, required=True)
    parser.add_argument("--audit-sacct", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    accounting = list(csv.DictReader(args.sacct.open(), delimiter="|"))
    summary = {
        "scope": "controlled timing/memory experiment; no posterior or production acceptance",
        "revision": REVISION,
        "array_job": "19775806",
        "preflight_job": "19775696",
        "run_root": str(args.run.resolve()),
        "sacct_sha256": sha(args.sacct),
        "cases": [],
    }
    audit_accounting = list(csv.DictReader(args.audit_sacct.open(), delimiter="|"))
    summary["audit_revision"] = AUDIT_REVISION
    summary["audit_job"] = "19777205"
    summary["audit_sacct_sha256"] = sha(args.audit_sacct)
    summary["supplementary_inventories"] = []
    reference_hashes = None
    for workload in ("mhd", "four_site"):
        input_hashes = set()
        config_path = args.run / "inputs" / workload / "config.json"
        config = json.loads(config_path.read_text())
        manifest_path = config_path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for name, value in manifest["files"].items():
            assert sha(config_path.parent / name) == value["sha256"]
        for variant in VARIANTS:
            preflight = json.loads((args.run / "checks" / workload / variant / "result.json").read_text())
            assert (
                preflight["stage"] == "check"
                and preflight["variant"] == variant
                and len(preflight["consistency"]) == 3
            )
            audit_path = args.run / "audits" / workload / variant / "audit.json"
            audit = json.loads(audit_path.read_text())
            assert audit["original_measurement_revision"] == REVISION
            assert audit["audit_revision"] == AUDIT_REVISION
            assert audit["variant"] == variant
            assert audit["input_sha256"] == preflight["input_sha256"]
            assert audit["config_sha256"] == sha(config_path)
            assert audit["check_sha256"] == sha(args.run / "checks" / workload / variant / "result.json")
            for name, value in audit["script_sha256"].items():
                frozen = subprocess.check_output(
                    ["git", "show", f"{AUDIT_REVISION}:scripts/research/centered_boundary_memory/{name}"]
                )
                assert hashlib.sha256(frozen).hexdigest() == value
            audit_job = audit["runtime_environment"]["SLURM_JOB_ID"]
            audit_rows = [r for r in audit_accounting if r["JobIDRaw"] in (audit_job, audit_job + ".batch")]
            assert audit_rows and all(
                r["State"] == "COMPLETED" and r["ExitCode"] == "0:0" for r in audit_rows
            )
            arrays = audit["array_inventory"]["constructed"]["arrays"]
            selected_audit_arrays = [a for a in arrays if ".coords." not in a["path"]]
            operands = [a for a in selected_audit_arrays if ".boundary_operands." in a["path"]]
            assert len(operands) == 6 and {a["dtype"] for a in operands} == {"float64"}
            supplemental = {
                "workload": workload,
                "variant": variant,
                "audit_sha256": sha(audit_path),
                "config_sha256": audit["config_sha256"],
                "input_sha256": audit["input_sha256"],
                "slurm_job_id": audit_job,
                "sacct_records": audit_rows,
                "unique_backing_bytes": {
                    p: v["retained_unique_backing_bytes"] for p, v in audit["array_inventory"].items()
                },
                "constructed_arrays": [
                    {k: v for k, v in a.items() if k != "owner"} for a in selected_audit_arrays
                ],
            }
            summary["supplementary_inventories"].append(supplemental)
            for chains in (1, 4):
                case_path = args.run / "results" / workload / variant / str(chains)
                result_path = case_path / "result.json"
                result = json.loads(result_path.read_text())
                assert result["git_revision"] == preflight["git_revision"] == REVISION
                assert result["config"] == preflight["config"] == config
                assert result["config_sha256"] == preflight["config_sha256"] == sha(config_path)
                assert result["input_sha256"] == preflight["input_sha256"]
                assert result["chains"] == result["actual_posterior_chains"] == result["cores"] == chains
                assert result["variant"] == variant and result["stage"] == "measure"
                assert audit["versions"] == result["versions"]
                assert result["mp_ctx"] == "spawn"
                expected_task = (0 if workload == "mhd" else 8) + VARIANTS.index(variant) * 2 + (chains == 4)
                assert int(result["runtime"]["environment"]["SLURM_ARRAY_TASK_ID"]) == expected_task
                assert result["runtime"]["source_package_path"] == audit["source_package_path"]
                assert result["config"]["sampling"]["draws"] == result["config"]["sampling"]["tune"] == 250
                input_hashes.add(result["input_sha256"])
                if reference_hashes is None:
                    reference_hashes = result["script_sha256"]
                assert result["script_sha256"] == preflight["script_sha256"] == reference_hashes
                for name, value in reference_hashes.items():
                    frozen = subprocess.check_output(
                        ["git", "show", f"{REVISION}:scripts/research/centered_boundary_memory/{name}"]
                    )
                    assert hashlib.sha256(frozen).hexdigest() == value
                env = result["runtime"]["environment"]
                assert all(
                    env[name] == "1"
                    for name in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    )
                )
                assert env["SLURM_CPUS_PER_TASK"] == "4" and env["SLURM_MEM_PER_NODE"] == "8192"
                max_workers, max_spawn_pids, worker_ids, cgroups, cgroup_peak = 0, 0, set(), set(), 0
                telemetry = case_path / "memory_samples.jsonl"
                for line in telemetry.open():
                    sample = json.loads(line)
                    cgroup = sample["cgroup"]
                    cgroups.add(cgroup["path"])
                    cgroup_peak = max(cgroup_peak, cgroup["cumulative_peak_bytes"])
                    if sample["phase"] == "sampling":
                        workers = [
                            p
                            for p in sample["processes"]
                            if "from multiprocessing.spawn import spawn_main;" in p["cmdline"]
                        ]
                        max_spawn_pids = max(max_spawn_pids, len(workers))
                        max_workers = max(max_workers, len({p["cmdline"] for p in workers}))
                        worker_ids.update((p["pid"], p["starttime"]) for p in workers)
                assert max_workers == (4 if chains == 4 else 0)
                assert len(cgroups) == 1
                job = env["SLURM_JOB_ID"]
                assert f"/job_{job}/" in next(iter(cgroups))
                rows = [r for r in accounting if r["JobIDRaw"] in (job, job + ".batch")]
                assert rows and all(r["State"] == "COMPLETED" and r["ExitCode"] == "0:0" for r in rows)
                inventory = result["array_inventory"]
                selected = [
                    a
                    for a in inventory["constructed"]["arrays"]
                    if a["path"]
                    in (
                        "target.design",
                        "target._transformed_rhs",
                        "target.prepared.factor",
                        "target.prepared.transformed_factor",
                    )
                    or "research_active_design" in a["path"]
                ]
                designs = [a for a in selected if "research_active_design" in a["path"]]
                assert len(designs) == 2
                assert {a["dtype"] for a in designs} == (
                    {"float32"} if variant in ("active_b32", "combined") else {"float64"}
                )
                assert all(a["dtype"] == "float64" for a in selected if a not in designs)
                summary["cases"].append(
                    {
                        "workload": workload,
                        "variant": variant,
                        "chains": chains,
                        "array_task": int(env["SLURM_ARRAY_TASK_ID"]),
                        "slurm_job_id": job,
                        "result_sha256": sha(result_path),
                        "telemetry_sha256": sha(telemetry),
                        "config_sha256": result["config_sha256"],
                        "input_sha256": result["input_sha256"],
                        "input_manifest_sha256": sha(manifest_path),
                        "versions": result["versions"],
                        "rows": result["rows"],
                        "cache_states": result["cache_states"],
                        "sampled_wholeprocess_peak_pss_bytes": max(
                            v["sampled_peak_tree_pss_bytes"] for v in result["memory"].values()
                        ),
                        "phase_peak_pss_bytes": {
                            p: result["memory"][p]["sampled_peak_tree_pss_bytes"] for p in PHASES
                        },
                        "phase_seconds": result["timings"],
                        "max_simultaneous_distinct_spawn_cmdlines": max_workers,
                        "max_simultaneous_spawn_pids": max_spawn_pids,
                        "observed_worker_identities": sorted(worker_ids),
                        "cgroup_path": next(iter(cgroups)),
                        "cgroup_cumulative_peak_bytes": cgroup_peak,
                        "sacct_records": rows,
                        "inventory_unique_backing_bytes": {
                            p: v["retained_unique_backing_bytes"] for p, v in inventory.items()
                        },
                        "selected_arrays": [{k: v for k, v in a.items() if k != "owner"} for a in selected],
                    }
                )
        assert len(input_hashes) == 1
    summary["measured_script_sha256"] = reference_hashes
    summary["audited_script_sha256"] = audit["script_sha256"]
    assert len({case["cgroup_path"] for case in summary["cases"]}) == 16
    assert len({case["array_task"] for case in summary["cases"]}) == 16
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"PASS: validated and summarized {len(summary['cases'])} cases")


if __name__ == "__main__":
    main()
