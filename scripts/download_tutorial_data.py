#!/usr/bin/env python3
"""Download and verify the pinned RHIME tutorial data release."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
from typing import Sequence
from urllib.request import urlopen

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; OpenGHG depends on toml.
    import toml as tomllib  # type: ignore[no-redef]


DATA_TAG = "v1.0.0"
DEFAULT_DIRECTORY = Path("build") / f"tutorial-data-{DATA_TAG}"
BASE_URL = (
    "https://github.com/openghg/openghg_inversions_tutorial_data/raw/"
    f"{DATA_TAG}"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(relative_path: str, directory: Path) -> Path:
    destination = directory / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urlopen(f"{BASE_URL}/{relative_path}") as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)
    return destination


def download_release(directory: Path) -> None:
    """Download release files and validate data against its manifest."""
    manifest_path = _download("manifest.toml", directory)
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = directory / entry["path"]
        if (
            not path.is_file()
            or path.stat().st_size != entry["size_bytes"]
            or _sha256(path) != entry["sha256"]
        ):
            path = _download(entry["path"], directory)
        if path.stat().st_size != entry["size_bytes"] or _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Downloaded file failed verification: {entry['path']}")
    _download("scripts/populate_store.py", directory)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    directory = args.directory.resolve()
    download_release(directory)
    print(f"Downloaded and verified tutorial data {DATA_TAG} in {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
