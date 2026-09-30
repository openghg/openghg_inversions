"""Compatibility imports; implementation lives in :mod:`openghg_inversions.recipes.co2.configuration`."""

from openghg_inversions.recipes.co2.configuration import (
    Co2O2RunSetup as Co2O2RunSetup,
    Co2RunSetup as Co2RunSetup,
    Runner as Runner,
    _bool as _bool,
    _cached_likelihood as _cached_likelihood,
    _frozen as _frozen,
    _model as _model,
    _number as _number,
    _number_or_mapping as _number_or_mapping,
    _ordinary_likelihood as _ordinary_likelihood,
    _prepared_inputs as _prepared_inputs,
    _prior as _prior,
    _reject_unknown as _reject_unknown,
    _resolve_co2 as _resolve_co2,
    _resolve_linked as _resolve_linked,
    _sampling as _sampling,
    _string as _string,
    _table as _table,
    _take as _take,
    co2_config_templates as co2_config_templates,
    load_co2_family_config as load_co2_family_config,
    resolve_co2_family_config as resolve_co2_family_config,
)

__all__ = ['Co2O2RunSetup', 'Co2RunSetup', 'co2_config_templates', 'load_co2_family_config', 'resolve_co2_family_config']
