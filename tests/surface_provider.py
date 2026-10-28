"""Required provider operations shared by surface and end-to-end tests."""

import pytest

from openghg_inversions.inversion_data import surface


@pytest.fixture
def compatible_provider():
    surface._surface_operations()
