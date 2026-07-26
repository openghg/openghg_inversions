"""Publish the exact tiny likelihood-aware structural MH-flow R0 certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, cast

from openghg_inversions.experimental.rjmcmc.mh_local_search_flow_oracle import (
    AUDIT_FILENAME,
    publish_flow_oracle,
)


def parser() -> argparse.ArgumentParser:
    """Build the bounded create-only audit command."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="New output directory for audit.json and complete.json.",
    )
    result.add_argument(
        "--source-revision",
        required=True,
        help="Lowercase full 40-hex revision; must equal the current clean Git HEAD.",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic oracle and print a compact strict-JSON summary."""
    arguments = parser().parse_args(argv)
    payload = publish_flow_oracle(
        arguments.output_directory,
        source_revision=arguments.source_revision,
    )
    catalogue = cast(dict[str, object], payload["catalogue"])
    summary = {
        "status": payload["status"],
        "certificate": str(arguments.output_directory / AUDIT_FILENAME),
        "source_revision": payload["source_revision"],
        "topologies": catalogue["topologies"],
        "counts": catalogue["counts"],
    }
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
