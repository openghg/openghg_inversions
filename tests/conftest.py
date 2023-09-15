from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import pytest


# Added for import of openghg from testing directory
#sys.path.insert(0, os.path.abspath("."))


@pytest.fixture(scope="session", autouse=True)
def default_session_fixture() -> Iterator[None]:
    mock_config = {
        "object_store": {
            "inversions_tests": {"path": str(Path().resolve() / 'tests/data/test_store'), "permissions": "rw"},
        },
        "user_id": "test-id-123",
        "config_version": "2",
    }

    with patch("openghg.objectstore._local_store.read_local_config", return_value=mock_config):
        yield
