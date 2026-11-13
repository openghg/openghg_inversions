"""Nested option groups and their deprecated flat spellings."""

from dataclasses import dataclass
import json
import warnings

import pytest

from openghg_inversions.rhime.ini import read_rhime_ini
from openghg_inversions.rhime.params import (
    DATA_OPTION_GROUPS, BcBasisOptions, DataConfig, OptionGroup, RhimeConfig, group_params, nest_flat_options,
)
from openghg_inversions.rhime.requests import load_params
from tests.test_staged_workflow import _params


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


def test_old_flat_manifest_data_still_loads(tmp_path):
    config = RhimeConfig.from_params(_params(bc_basis={"paths": [tmp_path / "bc.nc"]}), multisector=False).data
    record = config.to_params()
    del record["bc_basis"]
    record.update(bc_basis_paths=[str(tmp_path / "bc.nc")], bc_basis_algorithm="NESW", bc_basis_split_height=3000.0)
    with pytest.warns(DeprecationWarning):
        loaded = DataConfig.from_params(record)
    assert loaded.bc_basis == config.bc_basis


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


def test_flat_json_request_is_deprecated(tmp_path):
    path = tmp_path / "params.json"
    path.write_text(json.dumps(_params(bc_basis_algorithm="twelve-region")))
    with pytest.warns(DeprecationWarning, match="bc_basis_algorithm"):
        loaded = load_params(params_file=path)
    assert loaded["data"]["bc_basis"] == {"algorithm": "twelve-region"}
