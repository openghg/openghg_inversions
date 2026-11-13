"""Export schemas and round-trip the nested BC configuration prototype.

Run ``python -m examples.pydantic_bc_options OUTPUT_DIRECTORY``. This example
writes metadata only; it performs no acquisition, model construction or sampling.
"""

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from openghg_inversions.rhime.params import BcBasisOptions


def run_example(output_directory: Path) -> None:
    """Write a schema and portable JSON using the runtime options class.

    Args:
        output_directory: Directory created for the schema and BC JSON example.

    Raises:
        OSError: If the requested directory or files cannot be written.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    options = BcBasisOptions(algorithm="twelve-region", split_height=2500.)
    encoded = options.to_json()
    assert BcBasisOptions.from_json(encoded, active=True) == options
    (output_directory / "bc-basis.json").write_text(encoded + "\n")
    (output_directory / "bc-basis.schema.json").write_text(json.dumps(BcBasisOptions.json_schema(), indent=2) + "\n")
    try:
        BcBasisOptions(split_height="2500")
    except ValidationError as error:
        print("Constructor error:", error.errors()[0]["loc"], error.errors()[0]["msg"])
    print("Round-tripped:", encoded)
    print("Wrote schemas to", output_directory)


def _main() -> None:
    """Read the example's output directory from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    run_example(parser.parse_args().output_directory)


if __name__ == "__main__":
    _main()
