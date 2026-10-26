"""The finite command map retains canonical routes without redundant launchers."""

import pytest

from openghg_inversions.cli import build_parser


@pytest.mark.parametrize("command", ["run-rhime", "run-rhime-multisector", "run-rhime-nested", "run-co2"])
def test_retired_commands_receive_ordinary_parser_rejection(command):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([command, "-c", "run.ini"])
    assert exc.value.code == 2

