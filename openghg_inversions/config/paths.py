"""Expose the project path through ``Paths.openghginv``."""

from pathlib import Path

_openghginv_path = Path(__file__).parents[2]


class Paths:
    """Object that used to be used to store paths to obs, ACRG and LPDM directories.
    However, with the move over to OpenGHG this is generally all deprecated
    Currently, the only path is to the current openghg_inversions directory.

    All paths are pathlib.Path objects (Python >3.4)

    Paths.openghginv: path to openghg_inversions repo

    [Formerly]
    paths.acrg: path to ACRG repo
    paths.obs: path to obs folder
    path.lpdm: path to LPDM data directory
    """

    openghginv = _openghginv_path
