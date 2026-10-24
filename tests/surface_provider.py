"""Explicit capability gate shared by surface kernel and end-to-end tests."""

import pytest

from openghg_inversions.inversion_data import surface


def pytest_addoption(parser):
    parser.addoption(
        "--surface-provider", choices=("required", "unsupported"), default="required",
        help="Require resolved transport capability, or explicitly omit it for an unsupported provider.",
    )


@pytest.fixture
def compatible_provider(request):
    if request.config.getoption("--surface-provider") == "unsupported":
        pytest.skip("Explicit unsupported-provider comparison excludes resolved transport")
    surface._require_residual_month_support()
