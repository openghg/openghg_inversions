#!/usr/bin/env python3
"""Created on Fri Aug 21 11:21:27 2020.

@author: al18242

This file contains a method of obtaining the version
of the code used to generate output.
---------------------------------------
Updated by Eric Saboya (Dec. 2022)

"""

import subprocess
from importlib.metadata import version as il_version
from openghg_inversions.config.paths import Paths

openghginv_path = Paths.openghginv


def code_version():
    """Use git describe to return the latest tag
    (and git hash if applicable).
    -----------------------------------

    Returns:
      version : String defining the version of the code used,
                or "Unknown" if git is unavailable
    -----------------------------------
    """

    try:
        output = subprocess.run(
            ["git", "describe"],
            capture_output=True,
            cwd=openghginv_path,
            check=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "WARNING: Unable to identify version using git."
            " Check that git is available to the python process."
        )
        version = il_version("openghg_inversions")
    else:
        # remove newlines and cast as string
        version = output.stdout.strip("\n")

    return version
