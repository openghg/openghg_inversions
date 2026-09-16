"""Tests for the local Sphinx documentation preview command."""

from argparse import ArgumentTypeError
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from scripts import preview_docs


def test_parse_args_uses_non_mkdocs_default_and_accepts_overrides() -> None:
    """Preview applies defaults and accepts port, browser, and freshness overrides."""
    defaults = preview_docs._parse_args([])
    overridden = preview_docs._parse_args(["--port", "9123", "--no-open", "--fresh"])

    assert defaults.port == 8765
    assert defaults.no_open is False
    assert defaults.fresh is False
    assert defaults.command == "preview"
    assert overridden.port == 9123
    assert overridden.no_open is True
    assert overridden.fresh is True
    assert overridden.command == "preview"


def test_parse_args_accepts_the_clean_command() -> None:
    """The clean command needs no preview-server options."""
    assert preview_docs._parse_args(["clean"]).command == "clean"


def test_port_rejects_values_outside_the_tcp_range() -> None:
    """Port validation rejects values outside the valid TCP port range."""
    with pytest.raises(ArgumentTypeError, match="between 1 and 65535"):
        preview_docs._port("0")

    with pytest.raises(ArgumentTypeError, match="between 1 and 65535"):
        preview_docs._port("65536")


def test_build_docs_runs_the_documentation_tox_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building delegates to the docs tox environment from the repository root."""
    calls = []

    def run_fake(command, **kwargs):
        calls.append((command, kwargs))
        return CompletedProcess(command, returncode=7)

    monkeypatch.setattr(preview_docs.subprocess, "run", run_fake)

    source_directory = Path("/tmp/docs-source")
    build_root = Path("/tmp/docs-build")

    assert preview_docs._build_docs(source_directory, build_root) == 7
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == ["tox", "-e", "docs"]
    assert kwargs["cwd"] == preview_docs._REPOSITORY_ROOT
    assert kwargs["check"] is False
    assert kwargs["env"]["DOCS_SOURCE_DIR"] == str(source_directory)
    assert kwargs["env"]["DOCS_BUILD_DIR"] == str(build_root)


def test_clean_docs_removes_only_the_generated_build_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleaning removes the build root while preserving documentation sources."""
    build_root = tmp_path / "docs" / "_build"
    build_directory = build_root / "html"
    build_directory.mkdir(parents=True)
    source_file = build_root.parent / "conf.py"
    source_file.touch()
    generated_file = build_directory / "index.html"
    generated_file.touch()
    monkeypatch.setattr(preview_docs, "_BUILD_ROOT", build_root)

    preview_docs._clean_docs()

    assert not build_root.exists()
    assert source_file.exists()


def test_main_cleans_without_building_or_serving(monkeypatch: pytest.MonkeyPatch) -> None:
    """The clean command must not invoke the docs build or HTTP server."""
    calls = []
    monkeypatch.setattr(preview_docs, "_clean_docs", lambda: calls.append("clean"))
    monkeypatch.setattr(
        preview_docs, "_build_docs", lambda source, build: pytest.fail("build should not run")
    )

    assert preview_docs.main(["clean"]) == 0
    assert calls == ["clean"]


def test_open_safari_reports_a_failed_launch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A Safari launch failure leaves the user with the preview URL."""
    url = "http://127.0.0.1:8765/"

    def run_fake(command, **kwargs):
        assert command == ["open", "-a", "Safari", url]
        assert kwargs == {"check": False}
        return CompletedProcess(command, returncode=1)

    monkeypatch.setattr(preview_docs.subprocess, "run", run_fake)

    preview_docs._open_safari(url)

    assert f"Could not open Safari automatically; visit {url}" in capsys.readouterr().err


def test_open_safari_reports_when_the_open_command_cannot_start(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing or unusable open command does not stop the preview server."""
    url = "http://127.0.0.1:8765/"

    def run_fake(command, **kwargs):
        raise OSError("open is unavailable")

    monkeypatch.setattr(preview_docs.subprocess, "run", run_fake)

    preview_docs._open_safari(url)

    assert f"Could not open Safari automatically; visit {url}" in capsys.readouterr().err


def test_serve_docs_binds_loopback_serves_build_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The server exposes only the build directory on loopback and closes on Ctrl-C."""
    opened_urls = []

    class FakeServer:
        def __init__(self, address, handler):
            self.address = address
            self.handler = handler
            self.closed = False

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            self.closed = True

    servers = []

    def server_factory(address, handler):
        server = FakeServer(address, handler)
        servers.append(server)
        return server

    monkeypatch.setattr(preview_docs, "ThreadingHTTPServer", server_factory)
    monkeypatch.setattr(preview_docs, "_open_safari", opened_urls.append)

    assert preview_docs._serve_docs(9123, open_safari=True) == 0

    server = servers[0]
    assert server.address == ("127.0.0.1", 9123)
    assert Path(server.handler.keywords["directory"]) == preview_docs._BUILD_DIRECTORY
    assert opened_urls == ["http://127.0.0.1:9123/"]
    assert server.closed is True


def test_serve_docs_reports_bind_errors_without_opening_browser(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bind error is reported and never attempts to launch Safari."""

    def fail_to_bind(address, handler):
        raise OSError("address already in use")

    def fail_to_open(url):
        raise AssertionError(f"browser should not open for {url}")

    monkeypatch.setattr(preview_docs, "ThreadingHTTPServer", fail_to_bind)
    monkeypatch.setattr(preview_docs, "_open_safari", fail_to_open)

    assert preview_docs._serve_docs(9123, open_safari=True) == 1
    assert "Could not serve documentation on 127.0.0.1:9123" in capsys.readouterr().err


def test_main_stops_when_the_documentation_build_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed docs build returns its status without starting the server."""
    monkeypatch.setattr(preview_docs, "_build_docs", lambda source, build: 4)

    def fail_to_serve(port, *, open_safari, build_directory):
        raise AssertionError("server should not start after a failed build")

    monkeypatch.setattr(preview_docs, "_serve_docs", fail_to_serve)

    assert preview_docs.main([]) == 4
    assert "Documentation build failed" in capsys.readouterr().err


def test_main_builds_and_serves_from_persistent_directories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preview reuses the project docs and build directories for incremental builds."""

    def build_fake(source_directory: Path, build_root: Path) -> int:
        assert source_directory == preview_docs._DOCS_DIRECTORY
        assert build_root == preview_docs._BUILD_ROOT
        return 0

    def serve_fake(port, *, open_safari, build_directory):
        assert port == 9123
        assert open_safari is False
        assert build_directory == preview_docs._BUILD_DIRECTORY
        return 0

    monkeypatch.setattr(preview_docs, "_build_docs", build_fake)
    monkeypatch.setattr(preview_docs, "_serve_docs", serve_fake)

    assert preview_docs.main(["--port", "9123", "--no-open"]) == 0


def test_main_fresh_cleans_before_building(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh preview discards cached output before starting the docs build."""
    calls = []
    monkeypatch.setattr(preview_docs, "_clean_docs", lambda: calls.append("clean"))

    def build_fake(source_directory: Path, build_root: Path) -> int:
        calls.append(("build", source_directory, build_root))
        return 0

    monkeypatch.setattr(preview_docs, "_build_docs", build_fake)
    monkeypatch.setattr(
        preview_docs,
        "_serve_docs",
        lambda port, *, open_safari, build_directory: calls.append("serve") or 0,
    )

    assert preview_docs.main(["--fresh", "--no-open"]) == 0
    assert calls == [
        "clean",
        ("build", preview_docs._DOCS_DIRECTORY, preview_docs._BUILD_ROOT),
        "serve",
    ]


@pytest.mark.parametrize(
    ("platform", "arguments", "expected_open"),
    [
        ("darwin", [], True),
        ("darwin", ["--no-open"], False),
        ("linux", [], False),
    ],
)
def test_main_selects_browser_behavior_for_platform_and_option(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    arguments: list[str],
    expected_open: bool,
) -> None:
    """Safari opens by default only on macOS.

    The explicit no-open option must suppress it on macOS as well.
    """
    serve_calls = []
    monkeypatch.setattr(preview_docs, "_build_docs", lambda source, build: 0)
    monkeypatch.setattr(preview_docs.sys, "platform", platform)

    def serve_fake(port, *, open_safari, build_directory):
        serve_calls.append((port, open_safari))
        return 0

    monkeypatch.setattr(preview_docs, "_serve_docs", serve_fake)

    assert preview_docs.main(["--port", "9123", *arguments]) == 0
    assert serve_calls == [(9123, expected_open)]
