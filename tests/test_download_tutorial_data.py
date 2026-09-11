"""Tests for the pinned tutorial-data downloader."""

import hashlib
from io import BytesIO
from pathlib import Path

from scripts import download_tutorial_data


def test_download_release_uses_manifest_and_reuses_verified_data(
    monkeypatch, tmp_path: Path
) -> None:
    content = b"tutorial data"
    manifest = f'''[[files]]
path = "data/example.nc"
sha256 = "{hashlib.sha256(content).hexdigest()}"
size_bytes = {len(content)}
'''.encode()
    remote = {
        "manifest.toml": manifest,
        "data/example.nc": content,
        "scripts/populate_store.py": b"print('populate')\n",
    }
    requested = []

    def fake_urlopen(url):
        relative_path = url.removeprefix(download_tutorial_data.BASE_URL + "/")
        requested.append(relative_path)
        return BytesIO(remote[relative_path])

    monkeypatch.setattr(download_tutorial_data, "urlopen", fake_urlopen)

    download_tutorial_data.download_release(tmp_path)
    download_tutorial_data.download_release(tmp_path)

    assert (tmp_path / "data/example.nc").read_bytes() == content
    assert requested.count("data/example.nc") == 1
    assert requested.count("manifest.toml") == 2
    assert requested.count("scripts/populate_store.py") == 2
