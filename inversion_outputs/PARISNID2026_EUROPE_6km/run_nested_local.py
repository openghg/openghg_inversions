"""Run nested RHIME and persist its dual-grid artifacts locally."""

from __future__ import annotations

import argparse
from pathlib import Path

from openghg_inversions.rhime import run_rhime_nested
from openghg_inversions.rhime.outputs import _save_inferencedata
from openghg_inversions.rhime.params import params_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    params = params_from_config(config_path)
    output_path = params.get("output_path")
    if output_path is None:
        raise ValueError("The nested run configuration must define output_path.")

    output_dir = Path(output_path)
    stem = (
        f"{params['output_name']}_"
        f"{params['start_date']}_{params['end_date']}_nested"
    )
    artifacts = {
        "trace": output_dir / f"{stem}_trace.nc",
        "outer": output_dir / f"{stem}_outer_prepared.nc",
        "inner": output_dir / f"{stem}_inner_prepared.nc",
        "combined": output_dir / f"{stem}_combined_prepared.nc",
    }
    existing = [path for path in artifacts.values() if path.exists()]
    if existing:
        paths = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing local run artifacts:\n{paths}")

    result = run_rhime_nested(config_file=config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_inferencedata(result.idata, artifacts["trace"])
    result.prepared_inputs.outer.save(artifacts["outer"])
    result.prepared_inputs.inner.save(artifacts["inner"])
    result.prepared_inputs.combined.save(artifacts["combined"])

    for label, path in artifacts.items():
        print(f"Saved {label}: {path}", flush=True)


if __name__ == "__main__":
    main()
