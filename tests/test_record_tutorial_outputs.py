"""Test the explicit RHIME tutorial-output recorder."""

import nbformat
from pathlib import Path
import pytest
import subprocess

from scripts import record_tutorial_outputs


def test_run_reports_captured_subprocess_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], returncode=2, stdout="preparation started\n", stderr="store conflict\n"
        ),
    )

    with pytest.raises(RuntimeError, match=r"(?s)preparation started.*store conflict"):
        record_tutorial_outputs._run(["example", "command"])


def test_prepare_data_uses_an_isolated_openghg_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    recorder_home = tmp_path / "recorder-home"
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def fake_run(command, *, cwd=record_tutorial_outputs._ROOT, env=None):
        calls.append((command, env))
        return "same-commit" if command[:2] == ["git", "rev-parse"] else "same-commit"

    monkeypatch.setattr(record_tutorial_outputs, "_RECORDER_HOME", recorder_home)
    monkeypatch.setattr(record_tutorial_outputs, "_run", fake_run)

    record_tutorial_outputs._prepare_data(data_directory)

    populate_env = calls[-1][1]
    assert populate_env is not None
    assert populate_env["HOME"] == str(recorder_home)


def test_build_docs_invokes_sphinx_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))

    record_tutorial_outputs._build_docs()

    assert calls[0][0] == [
        "sphinx-build",
        "-M",
        "html",
        "docs",
        "docs/_build",
        "--keep-going",
    ]
    assert "tox" not in calls[0][0]


def test_replace_outputs_preserves_inputs_and_updates_adjacent_results() -> None:
    document = """.. jupyter-input::

   value = 1
   value

.. jupyter-output::

   old

Prose.
"""

    replaced = record_tutorial_outputs._replace_outputs(document, ["{'value': 1}"])

    assert "   value = 1\n   value" in replaced
    assert "   {'value': 1}" in replaced
    assert "old" not in replaced
    assert replaced.endswith("Prose.\n")


def test_replace_outputs_rejects_unpaired_cells() -> None:
    document = """.. jupyter-input::

   value = 1
"""

    with pytest.raises(ValueError, match="counts must match"):
        record_tutorial_outputs._replace_outputs(document, ["1"])


def test_recorded_outputs_accepts_text_results_and_ignores_stdout() -> None:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "value",
                outputs=[
                    nbformat.v4.new_output("stream", name="stdout", text="sampling log\n"),
                    nbformat.v4.new_output(
                        "execute_result",
                        data={"text/plain": "{'done': True}"},
                        execution_count=1,
                    ),
                ],
            )
        ]
    )

    assert record_tutorial_outputs._recorded_outputs(notebook) == ["{'done': True}"]


def test_recorded_outputs_ignores_stderr_when_a_cell_has_a_result() -> None:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "value",
                outputs=[
                    nbformat.v4.new_output("stream", name="stderr", text="warning\n"),
                    nbformat.v4.new_output(
                        "execute_result",
                        data={"text/plain": "{'done': True}"},
                        execution_count=1,
                    ),
                ],
            )
        ]
    )

    assert record_tutorial_outputs._recorded_outputs(notebook) == ["{'done': True}"]


@pytest.mark.parametrize(
    ("name", "inputs"),
    [("rhime_standard_tutorial", 4), ("rhime_multisector_tutorial", 3)],
)
def test_tutorials_are_manual_downloadable_notebooks(name: str, inputs: int) -> None:
    document = (record_tutorial_outputs._ROOT / "docs" / "usage" / f"{name}.rst").read_text(encoding="utf-8")

    assert f":jupyter-download-notebook:`download it as a Jupyter notebook <{name}>`" in document
    assert f".. jupyter-kernel:: python3\n   :id: {name}" in document
    assert len(record_tutorial_outputs._directives(document.splitlines(), "jupyter-input")) == inputs
    assert len(record_tutorial_outputs._directives(document.splitlines(), "jupyter-output")) == inputs
    assert ".. jupyter-execute::" not in document
    assert ".. code-block:: python" not in document
