"""Compare bounded posterior means with summed Monte Carlo standard errors."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def compare(reports, accepted):
    rows = []
    for report in reports:
        family = report["family"]
        if family not in accepted:
            continue
        path = accepted[family]
        with path.open() as stream:
            historical = {row["functional"]: row for row in csv.DictReader(stream)}
        for name, values in report["functionals"].items():
            if name not in historical:
                continue
            old = historical[name]
            difference = values["mean"] - float(old["posterior_mean"])
            summed_mcse = values["mcse_mean"] + float(old["mcse_mean"])
            rows.append(
                {
                    "family": family,
                    "functional": name,
                    "new_minus_accepted_mean": difference,
                    "summed_mcse": summed_mcse,
                    "difference_over_summed_mcse": difference / summed_mcse,
                    "accepted_score_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    by_family = {report["family"]: report for report in reports}
    treatment = []
    for family in ("lognormal", "gaussian"):
        old, new = by_family.get(family + "_original"), by_family.get(family + "_shrunk")
        if old is None or new is None:
            continue
        for name in new["functionals"]:
            before, after = old["functionals"][name], new["functionals"][name]
            summed = before["mcse_mean"] + after["mcse_mean"]
            treatment.append(
                {
                    "family": family,
                    "functional": name,
                    "shrunk_minus_original_mean": after["mean"] - before["mean"],
                    "summed_mcse": summed,
                    "difference_over_summed_mcse": (after["mean"] - before["mean"]) / summed,
                }
            )
    return {
        "units": "annualized Tg CO2 / year on archived inner PARIS support",
        "comparison": "posterior means; conservative sum of MCSEs; no identical-chain requirement",
        "accepted_comparisons": rows,
        "treatment_comparisons": treatment,
        "sampling_health": [
            {
                key: r[key]
                for key in (
                    "family",
                    "max_rhat",
                    "min_bulk_ess",
                    "events",
                    "negative_biosphere_coefficient_fraction",
                )
            }
            for r in reports
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--accepted-original", type=Path, required=True)
    parser.add_argument("--accepted-shrunk", type=Path, required=True)
    args = parser.parse_args()
    reports = [
        json.loads(p.read_text())
        for p in sorted(args.root.glob("*/report.json"))
        if "preflight" not in p.parent.name
    ]
    if not reports:
        parser.error("No sampling reports found.")
    result = compare(
        reports, {"lognormal_original": args.accepted_original, "lognormal_shrunk": args.accepted_shrunk}
    )
    output = args.root / "comparison.json"
    if output.exists():
        parser.error("Comparison exists; preserve it and use a new directory.")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
