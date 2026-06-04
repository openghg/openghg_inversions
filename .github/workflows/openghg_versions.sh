#!/usr/bin/env bash
# Script to get non-yanked OpenGHG release versions from PyPI.
#
# Minor versions are specified as N behind the most recent minor version.
#
# For instance, if the non-yanked versions are: 0.18.0, 0.17.1, 0.17.0, 0.16.0
# then:
#
# ./openghg_versions.sh -N 0
#
# returns 0.18.0,
#
# ./openghg_versions.sh -N 1
#
# returns 0.17.1.

set -euo pipefail

minor_N=0
major_version=""
test=false

while getopts M:N:t flag; do
    case "${flag}" in
        t) test=true;;
        N) minor_N=${OPTARG};;
        M) major_version=${OPTARG};;
    esac
done

python3 - "$minor_N" "$major_version" "$test" <<'PY'
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request


PYPI_URL = "https://pypi.org/pypi/openghg/json"
VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def load_pypi_json() -> dict:
    fixture_path = os.environ.get("OPENGHG_PYPI_JSON")
    if fixture_path:
        with open(fixture_path, encoding="utf-8") as handle:
            return json.load(handle)

    with urllib.request.urlopen(PYPI_URL, timeout=30) as response:
        return json.load(response)


def release_has_installable_file(files: list[dict]) -> bool:
    return bool(files) and any(not file_info.get("yanked", False) for file_info in files)


def parse_versions(data: dict) -> list[tuple[tuple[int, int, int], str]]:
    versions: list[tuple[tuple[int, int, int], str]] = []
    for version, files in data["releases"].items():
        match = VERSION_RE.fullmatch(version)
        if match is None or not release_has_installable_file(files):
            continue
        versions.append(((int(match.group(1)), int(match.group(2)), int(match.group(3))), version))
    return sorted(versions)


def selected_version(
    versions: list[tuple[tuple[int, int, int], str]],
    *,
    major: int | None,
    minor_offset: int,
) -> str:
    if not versions:
        raise SystemExit("No installable OpenGHG releases found.")

    selected_major = major if major is not None else max(version_tuple[0] for version_tuple, _ in versions)
    major_versions = [(version_tuple, version) for version_tuple, version in versions if version_tuple[0] == selected_major]
    if not major_versions:
        raise SystemExit(f"No installable OpenGHG releases found for major version {selected_major}.")

    minor_versions = sorted({version_tuple[1] for version_tuple, _ in major_versions}, reverse=True)
    try:
        selected_minor = minor_versions[minor_offset]
    except IndexError as exc:
        raise SystemExit(
            f"No OpenGHG minor release at offset {minor_offset}; available minors: {minor_versions}"
        ) from exc

    patch_versions = [
        (version_tuple, version)
        for version_tuple, version in major_versions
        if version_tuple[1] == selected_minor
    ]
    return max(patch_versions)[1]


def main() -> int:
    minor_offset = int(sys.argv[1])
    major = int(sys.argv[2]) if sys.argv[2] else None
    test_mode = sys.argv[3] == "true"

    versions = parse_versions(load_pypi_json())
    result = selected_version(versions, major=major, minor_offset=minor_offset)

    if test_mode:
        selected_major = major if major is not None else max(version_tuple[0] for version_tuple, _ in versions)
        minor_versions = sorted(
            {version_tuple[1] for version_tuple, _ in versions if version_tuple[0] == selected_major},
            reverse=True,
        )
        print("OpenGHG installable releases:")
        for _, version in versions:
            print(version)
        print(f"Selected major version = {selected_major}")
        print("Minor versions:")
        for minor in minor_versions:
            print(minor)
        print(f"Selected release = {result}")
    else:
        print(result)

    return 0


raise SystemExit(main())
PY
