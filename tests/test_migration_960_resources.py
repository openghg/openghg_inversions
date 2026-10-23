"""Resource evidence for #960; expected settings live outside compatibility code.

The JSON cases can be reused by a future converter after HBMCMC is removed.
Only the driver imports the current compatibility adapter. NetCDF contents are
synthetic schema markers: these checks select paths, never load scientific data.
"""

import json
import shutil
import warnings
from pathlib import Path

import pytest
import xarray as xr

FIXTURES = Path(__file__).parent / "fixtures" / "migration_960" / "resources"
CASES = json.loads((FIXTURES / "cases.json").read_text())


def _plain(value, root):
    if isinstance(value, Path):
        return value.relative_to(root).as_posix()
    if isinstance(value, dict):
        return {key: _plain(item, root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item, root) for item in value]
    return value


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_characterize_resource_selection(case, tmp_path, monkeypatch):
    from openghg_inversions.hbmcmc.compatibility import adapt_resource_params

    monkeypatch.chdir(tmp_path)
    for name, kind in case.get("files", {}).items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        attrs = {"schema": "openghg_inversions.flux_weighted_basis"} if kind == "retained" else {}
        xr.Dataset({"basis": ("lat", [1])}, attrs=attrs).to_netcdf(path)

    if case.get("no_discovery"):
        def forbidden_glob(*args, **kwargs):
            pytest.fail("superseded, disabled or invalid selections must not discover files")

        monkeypatch.setattr(Path, "glob", forbidden_glob)

    kwargs = {"overrides": case.get("overrides")}
    if "base_dir" in case:
        kwargs["base_dir"] = tmp_path / case["base_dir"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        if "error" in case:
            error_type = {"ValueError": ValueError, "FileNotFoundError": FileNotFoundError}[case["error"]["type"]]
            with pytest.raises(error_type) as caught:
                adapt_resource_params(case["params"], **kwargs)
            assert str(caught.value) == case["error"]["message"]
        else:
            actual = adapt_resource_params(case["params"], **kwargs)
            assert _plain(actual, tmp_path) == case["expected"]


def test_characterize_entrypoint_paths_are_invocation_relative(tmp_path, monkeypatch):
    from openghg_inversions.hbmcmc.compatibility import params_from_config
    from openghg_inversions.rhime.requests import load_params

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    source = config_dir / "invocation.ini"
    shutil.copyfile(FIXTURES / "invocation.ini", source)
    monkeypatch.chdir(tmp_path)
    expected = json.loads((FIXTURES / "invocation.expected.json").read_text())
    with pytest.warns(DeprecationWarning):
        assert _plain(params_from_config(source), tmp_path) == expected
    grouped = load_params(config_file=source)
    assert _plain({name: value for group in grouped.values() for name, value in group.items()}, tmp_path) == expected
