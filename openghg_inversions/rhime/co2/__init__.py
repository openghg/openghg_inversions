"""Public preparation, model-building, and runner APIs for the CO2 family.

The package exposes separate CO2-only, cached fixed-OU, and linked CO2/O2
recipes while reusing canonical RHIME inputs through explicit composition.
"""

from .co2_model import build_co2_model
from .co2_preparation import Co2PreparedInputs, prepare_co2_inputs
from .co2_runner import (
    co2_model_input_names,
    prepare_co2_scalar_sigma_eigenbasis,
    run_rhime_co2,
)
from .co2_cached_sigma_model import Co2CachedSigmaModel, build_co2_cached_sigma_model
from .co2_cached_sigma_runner import (
    co2_cached_sigma_input_names,
    run_rhime_co2_cached_sigma,
)
from .co2_o2_model import (
    build_co2_o2_model,
    evaluate_co2_o2_prior_forward_mean,
)
from .co2_o2_preparation import Co2O2PreparedInputs, prepare_co2_o2_inputs
from .co2_o2_runner import run_rhime_co2_o2_from_prepared_inputs
from .configuration import (
    Co2O2RunSetup,
    Co2RunSetup,
    co2_config_templates,
    load_co2_family_config,
    resolve_co2_family_config,
)

__all__ = [
    "Co2O2RunSetup",
    "Co2O2PreparedInputs",
    "Co2PreparedInputs",
    "Co2CachedSigmaModel",
    "Co2RunSetup",
    "build_co2_cached_sigma_model",
    "build_co2_model",
    "build_co2_o2_model",
    "co2_model_input_names",
    "co2_config_templates",
    "prepare_co2_scalar_sigma_eigenbasis",
    "co2_cached_sigma_input_names",
    "evaluate_co2_o2_prior_forward_mean",
    "prepare_co2_o2_inputs",
    "prepare_co2_inputs",
    "load_co2_family_config",
    "resolve_co2_family_config",
    "run_rhime_co2",
    "run_rhime_co2_cached_sigma",
    "run_rhime_co2_o2_from_prepared_inputs",
]
