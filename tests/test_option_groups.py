"""Nested option groups and their deprecated flat spellings."""

from dataclasses import dataclass
from contextlib import nullcontext
from importlib import import_module
import json
import warnings

import pytest
import xarray as xr

from openghg_inversions.rhime.ini import read_rhime_ini
from openghg_inversions.rhime.params import (
    DATA_OPTION_GROUPS, BcBasisOptions, DataConfig, OptionGroup, RhimeConfig, group_params, nest_flat_options,
)
from openghg_inversions.rhime.requests import load_params
from tests.test_staged_workflow import _params, _prepared


@dataclass(frozen=True)
class _Split:
    """Fields whose names contain underscores."""

    split_height: float = 0.0
    height: float = 0.0


def test_flat_names_are_split_by_dataclass_fields_not_underscores():
    groups = (OptionGroup("bc_basis", _Split),)
    nested = nest_flat_options({"bc_basis_split_height": 1.0, "bc_basis_height": 2.0, "bc_store": "x"}, groups)
    assert nested == {"bc_basis": {"split_height": 1.0, "height": 2.0}, "bc_store": "x"}


def test_stacked_prefix_nests_under_the_prefixed_key():
    groups = (OptionGroup("bc_basis", BcBasisOptions),)
    nested = nest_flat_options({"inner_bc_basis_algorithm": "NESW", "bc_basis_algorithm": "x"}, groups,
                               prefix="inner_")
    assert nested == {"inner_bc_basis": {"algorithm": "NESW"}, "bc_basis_algorithm": "x"}


def test_group_in_both_forms_is_rejected():
    with pytest.raises(ValueError, match="both as a mapping and as flat"):
        nest_flat_options({"bc_basis": {}, "bc_basis_algorithm": "NESW"}, DATA_OPTION_GROUPS)
    with pytest.raises(ValueError, match="both as a mapping and as flat"):
        group_params(_params(bc_basis={"algorithm": "NESW"}, bc_basis_split_height=2.0))


@pytest.mark.parametrize("entries", [
    [("bc_basis_algorithm", "NESW"), ("data", {"bc_basis": {"algorithm": "twelve-region"}})],
    [("bc_basis", {"algorithm": "twelve-region"}), ("data", {"bc_basis_algorithm": "NESW"})],
])
@pytest.mark.parametrize("reverse", [False, True])
def test_request_rejects_mixed_basis_spellings_across_locations(tmp_path, entries, reverse):
    """Reject mixed spelling within either source regardless of placement or key order."""
    options = dict(reversed(entries) if reverse else entries)
    source = tmp_path / "params.json"
    source.write_text(json.dumps(options))
    with pytest.raises(ValueError, match="bc_basis.*both as a mapping and as flat"):
        load_params(params_file=source)
    with pytest.raises(ValueError, match="bc_basis.*both as a mapping and as flat"):
        load_params(overrides=options)
    ini = tmp_path / "params.ini"
    ini.write_text("[RUN]\n" + "\n".join(f"{name} = {value!r}" for name, value in options.items()))
    with pytest.raises(ValueError, match="bc_basis.*both as a mapping and as flat"):
        read_rhime_ini(ini)
    with pytest.raises(ValueError, match="bc_basis.*both as a mapping and as flat"):
        load_params(config_file=ini)


def test_inactive_basis_fields_are_checked_without_resolving_values(tmp_path):
    """Check names in inactive preparation options without opening paths or parsing values."""
    source = tmp_path / "params.json"
    phases = {"model", "sampler"}
    source.write_text(json.dumps({"data": {"bc_basis": {"paths": 42, "split_heigth": 2.5}}}))
    with pytest.raises(ValueError, match=r"data\.bc_basis\.split_heigth"):
        load_params(params_file=source, phase_groups=phases)
    source.write_text(json.dumps({"data": {"bc_basis": {"paths": 42}}}))
    assert load_params(params_file=source, phase_groups=phases)["data"] == {}


def test_nested_request_needs_no_warning_and_round_trips():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        config = RhimeConfig.from_params(
            _params(bc_basis={"algorithm": "twelve-region", "split_height": 2.5}), multisector=False)
        params = config.data.to_params()
        assert params["bc_basis"] == {"paths": None, "algorithm": "twelve-region", "split_height": 2.5}
        assert DataConfig.from_params(params) == config.data


def test_flat_python_request_is_deprecated_but_equivalent():
    nested = RhimeConfig.from_params(_params(bc_basis={"algorithm": "twelve-region"}), multisector=False)
    with pytest.warns(DeprecationWarning, match="bc_basis_algorithm.*nested `bc_basis`"):
        flat = RhimeConfig.from_params(_params(bc_basis_algorithm="twelve-region"), multisector=False)
    assert flat.data == nested.data


@pytest.mark.parametrize("multisector", [False, True], ids=["standard", "multisector"])
def test_preparation_manifest_round_trip_reads_nested_v3_and_flat_v2(tmp_path, multisector):
    """Save and reopen a current preparation and the former v2 layout."""
    from examples.multisector_handoffs import synthetic_inputs
    from openghg_inversions.rhime import _standard_stages, multisector_stages

    basis = {"paths": None, "algorithm": "twelve-region", "split_height": 2000.0}
    output = tmp_path / "prepared"
    if multisector:
        stages = multisector_stages
        merged, params, *_ = synthetic_inputs(tmp_path / "inputs")
        data = DataConfig.from_params({**params, "bc_basis": basis}, multisector=True)
        manifest = stages.prepare(data, output_dir=output, merged_data=merged)
    else:
        stages = _standard_stages
        data = RhimeConfig.from_params(_params(bc_basis=basis), multisector=False).data
        manifest = stages.save_preparation(_prepared(), data_config=data, output_dir=output)

    record = json.loads(manifest.read_text())
    assert record["schema_version"] == 3
    assert record["data"]["bc_basis"] == basis
    prepared, recorded, generation = stages.load_preparation(manifest)
    assert recorded == data
    assert generation == record["generation"]

    record["schema_version"] = 2
    record["data"].update({f"bc_basis_{key}": value for key, value in record["data"].pop("bc_basis").items()})
    legacy = manifest.with_name("v2-prepare.json")
    legacy.write_text(json.dumps(record))
    with pytest.warns(DeprecationWarning, match="bc_basis_algorithm"):
        old_prepared, old_data, old_generation = stages.load_preparation(legacy)
    xr.testing.assert_identical(old_prepared.inv_inputs, prepared.inv_inputs)
    assert old_data == recorded
    assert old_generation == generation


def test_unknown_group_field_is_rejected():
    with pytest.raises(ValueError, match="Unsupported `bc_basis` option"):
        RhimeConfig.from_params(_params(bc_basis={"algorithm": "NESW", "height": 1.0}), multisector=False)


def test_ini_keeps_flat_spelling_without_warning(tmp_path):
    path = tmp_path / "request.ini"
    values = _params(bc_basis_algorithm="twelve-region", bc_basis_split_height=2.5)
    path.write_text("[RUN]\n" + "\n".join(f"{key} = {value!r}" for key, value in values.items()))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        decoded = read_rhime_ini(path)
        assert decoded["bc_basis"] == {"algorithm": "twelve-region", "split_height": 2.5}
        assert "bc_basis_algorithm" not in decoded
        config = RhimeConfig.from_params(load_params(config_file=path), multisector=False)
    assert config.data.bc_basis == BcBasisOptions(algorithm="twelve-region", split_height=2.5)


def test_nested_override_updates_file_group_by_field(tmp_path):
    path = tmp_path / "request.ini"
    values = _params(bc_basis_algorithm="twelve-region", bc_basis_split_height=2.5)
    path.write_text("[RUN]\n" + "\n".join(f"{key} = {value!r}" for key, value in values.items()))
    loaded = load_params(config_file=path, overrides={"bc_basis": {"split_height": 4.0}})
    assert loaded["data"]["bc_basis"] == {"algorithm": "twelve-region", "split_height": 4.0}


@pytest.mark.parametrize("runner", [
    "openghg_inversions.rhime.nested",
    "examples.rhime_customisation.runner",
    "examples.rhime_customisation.custom_basis_runner",
])
@pytest.mark.parametrize("flat", [False, True], ids=["nested-override", "flat-override"])
def test_runners_preserve_ini_basis_algorithm_when_overriding_height(tmp_path, monkeypatch, runner, flat):
    """A split-height override preserves the matching algorithm through each entry point."""
    path = tmp_path / "request.ini"
    values = _params(
        basis_algorithm="quadtree", bc_basis_algorithm="twelve-region",
        bc_basis_split_height=3000.0, output_format="none", use_bc=True,
    )
    path.write_text("[RUN]\n" + "\n".join(f"{key} = {value!r}" for key, value in values.items()))
    options = {"bc_basis_split_height": 2000.0} if flat else {"bc_basis": {"split_height": 2000.0}}
    module = import_module(runner)
    if runner == "openghg_inversions.rhime.nested":
        run = module.run_nested
        options["inner_domain"] = "6km"
    else:
        run = module.run_custom_rhime

    class BasisSelected(Exception):
        """Stop once the runner reaches boundary-basis resolution."""

    def check_basis(selection, *, use_bc):
        """Check the selection before the runner accesses any store."""
        assert use_bc
        assert selection == BcBasisOptions(algorithm="twelve-region", split_height=2000.0)
        raise BasisSelected

    monkeypatch.setattr(module, "resolve_bc_basis", check_basis)
    warning = pytest.warns(DeprecationWarning, match="bc_basis_split_height") if flat else nullcontext()
    with warning, pytest.raises(BasisSelected):
        run(config_file=path, **options)


def test_flat_json_request_is_deprecated(tmp_path):
    path = tmp_path / "params.json"
    path.write_text(json.dumps(_params(bc_basis_algorithm="twelve-region")))
    with pytest.warns(DeprecationWarning, match="bc_basis_algorithm"):
        loaded = load_params(params_file=path)
    assert loaded["data"]["bc_basis"] == {"algorithm": "twelve-region"}
