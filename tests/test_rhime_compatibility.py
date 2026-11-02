"""Modern INI decoding remains independent of the retired dictionary adapter."""

import warnings

from openghg_inversions.rhime.ini import read_rhime_ini


def test_reader_preserves_file_names_values_and_first_occurrence(tmp_path):
    path = tmp_path / "rhime.ini"
    path.write_text(
        '[FIRST]\noutputname = "first"\ndraws = "7"\n'
        'use_tracer = False\nspecial_recipe = {"x": [1, 2]}\n'
        '[SECOND]\noutputname = "second"\n'
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decoded = read_rhime_ini(path)
    assert not caught
    assert decoded == {
        "outputname": "first",
        "draws": "7",
        "use_tracer": False,
        "special_recipe": {"x": [1, 2]},
    }
