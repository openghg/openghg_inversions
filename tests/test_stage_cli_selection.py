"""Staged CLI selection happens once, before configuration resolution."""

from types import SimpleNamespace

import pytest

from openghg_inversions.cli import main
from openghg_inversions.rhime import stages


@pytest.mark.parametrize("model", ["standard", "multisector", "co2"])
@pytest.mark.parametrize("command", ["prepare", "prior-predictive", "sample", "postprocess"])
def test_cli_selects_recipe_once(monkeypatch, tmp_path, model, command):
    calls = []
    config = object()

    def load_params(**kwargs):
        calls.append(("load", kwargs))
        return {"choice": "resolved"}

    def resolve_config(params):
        calls.append(("resolve", params))
        return config

    def operation(**kwargs):
        calls.append(("operation", kwargs))
        return {"manifest_path": "prepare.json", "status": "pass", "artifacts": {"posterior": "posterior.nc"}}

    def select(model):
        calls.append(("select", model))
        return SimpleNamespace(
            load_params=load_params, resolve_config=resolve_config, **{command.replace("-", "_"): operation}
        )

    monkeypatch.setattr(stages, "select_stages", select)
    arguments = [command, "--model", model, "--params-file", "params.json", "--output-dir", str(tmp_path)]
    if command != "prepare":
        arguments += ["--prepared-inputs", "prepared.nc", "--preparation-manifest", "prepare.json"]
    if command == "postprocess":
        arguments += ["--posterior", "posterior.nc", "--sample-manifest", "sample.json"]
    main(arguments)
    assert [name for name, _ in calls] == ["select", "load", "resolve", "operation"]
    assert calls[0][1] == model
    assert calls[-1][1]["setup"] is config
    assert calls[-1][1]["output_dir"] == tmp_path.resolve()
    assert "model" not in calls[-1][1]


def test_selector_rejects_unknown_recipe():
    with pytest.raises(ValueError, match="Unsupported staged model"):
        stages.select_stages("unknown")
