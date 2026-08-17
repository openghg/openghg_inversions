"""Tests for the local Sphinx documentation preview command."""

from argparse import ArgumentTypeError
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from scripts import preview_docs


def test_parse_args_uses_non_mkdocs_default_and_accepts_overrides() -> None:
    """The CLI defaults away from port 8000 and accepts explicit options."""
    defaults = preview_docs._parse_args([])
    overridden = preview_docs._parse_args(["--port", "9123", "--no-open"])

    assert defaults.port == 8765
    assert defaults.no_open is False
    assert overridden.port == 9123
    assert overridden.no_open is True


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

    assert preview_docs._build_docs() == 7
    assert calls == [
        (
            ["tox", "-e", "docs"],
            {"cwd": preview_docs._REPOSITORY_ROOT, "check": False},
        )
    ]


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
    monkeypatch.setattr(preview_docs, "_build_docs", lambda: 4)

    def fail_to_serve(port, *, open_safari):
        raise AssertionError("server should not start after a failed build")

    monkeypatch.setattr(preview_docs, "_serve_docs", fail_to_serve)

    assert preview_docs.main([]) == 4
    assert "Documentation build failed" in capsys.readouterr().err


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
    monkeypatch.setattr(preview_docs, "_build_docs", lambda: 0)
    monkeypatch.setattr(preview_docs.sys, "platform", platform)

    def serve_fake(port, *, open_safari):
        serve_calls.append((port, open_safari))
        return 0

    monkeypatch.setattr(preview_docs, "_serve_docs", serve_fake)

    assert preview_docs.main(["--port", "9123", *arguments]) == 0
    assert serve_calls == [(9123, expected_open)]
