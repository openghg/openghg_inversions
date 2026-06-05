"""Compatibility script for running old fixedbasis-style configs through RHIME.

This entry point preserves the historical ``run_hbmcmc.py`` command-line
surface for old INI files, but translates supported fixedbasis-style parameter
names and calls the modern ``run_rhime`` pathway.

Run as:
    $ python run_hbmcmc.py [start end -c config.ini]
e.g.
    $ python run_hbmcmc.py
    $ python run_hbmcmc.py 2012-01-01 2013-01-01 -c hbmcmc_ch4_run.ini

start - Start of date range to use for MCMC inversion (YYYY-MM-DD)
end - End of date range to use for MCMC inversion (YYYY-MM-DD) (must be after start)
-c / --config - configuration file. See config/ folder for templates and examples of this input file.

If start and end are specified these will supersede the values within the configuration file, if present.
If -c option is not specified, this script will look for configuration file within the
acrg_hbmcmc/ directory called `hbmcmc_input.ini`.

To generate a config file from the template run this script as:
    $ python run_hbmcmc.py -r  [-c config.ini]

The MCMC run *will not be executed*. This will be named for your -c input or, if not specified, this will
create a configuration file called `hbmcmc_input.ini` within your acrg_hbmcmc/ directory and exit.
This file will need to be edited to add parameters for your MCMC run.
"""

import json
import sys
import argparse
import inspect
from pathlib import Path
from shutil import copyfile
from typing import Any
import warnings

import openghg_inversions.hbmcmc.hbmcmc_output as output

from openghg_inversions._timing import log_timing, timed, timer_seconds, timer_start
from openghg_inversions.config import config
from openghg_inversions.config.paths import Paths
from openghg_inversions.inversion_data import prepare_rhime_inputs
from openghg_inversions.rhime import run_rhime
from openghg_inversions.rhime.params import make_rhime_runner_setup, normalise_rhime_params


_RUN_HBMCMC_RHIME_ALIASES = {
    "nit": "draws",
    "nchain": "chains",
    "verbose": "progressbar",
    "sampler_kwargs": "sample_kwargs",
}
_LOGNORMAL_PRIOR_NAMES = ("xprior", "x_prior", "bcprior", "bc_prior")
_MIN_ERROR_METHODS = {"residual", "percentile"}


def fixed_basis_expected_param() -> list[str]:
    """Define required parameters for openghg_inversions.hcmcmc.fixedbasisMCMC().

    Expected parameters currently include:
      species, sites, averaging_period, domain, start_date, end_date,
      outputpath, outputname

    Returns:
      expected_param: required parameter names
    """
    expected_param = [
        "species",
        "sites",
        "averaging_period",
        "domain",
        "start_date",
        "end_date",
        "outputpath",
        "outputname",
    ]

    return expected_param


def extract_mcmc_type(config_file: str | Path, default: str = "fixed_basis") -> str:
    """Find value which describes the MCMC function to use.

    Checks the input configuration file the "mcmc_type" keyword within
    the "MCMC.TYPE" section. If not present, the default is used.

    Args:
      config_file:
        Configuration file name. Should be an .ini file.
      default:
        Default keyword for MCMC function to use.

    Returns:
      Keyword for MCMC function to use
    """
    mcmc_type_section = "MCMC.TYPE"
    mcmc_type_keyword = "mcmc_type"
    param_mcmc_type = config.extract_params(config_file, section=mcmc_type_section)

    if param_mcmc_type is not None and mcmc_type_keyword in param_mcmc_type:
        mcmc_type = param_mcmc_type[mcmc_type_keyword]
    else:
        mcmc_type = default

    return mcmc_type


def _legacy_option_enabled(value: Any) -> bool:
    """Return whether a legacy option value should be treated as enabled."""
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "false", "none", "0"}
    return True


def _translate_legacy_aliases(params: dict[str, Any]) -> None:
    """Translate legacy run_hbmcmc parameter names to RHIME names in-place."""
    for old, new in _RUN_HBMCMC_RHIME_ALIASES.items():
        if old not in params:
            continue
        if new in params:
            print(f"Ignoring deprecated run_hbmcmc parameter {old!r} because {new!r} was also supplied.")
        else:
            params[new] = params[old]
        del params[old]


def _normalise_legacy_output_format(params: dict[str, Any]) -> None:
    """Map old HBMCMC output names to the modern compatibility output."""
    paris_postprocessing = params.pop("paris_postprocessing", False)
    if _legacy_option_enabled(paris_postprocessing):
        params["output_format"] = "paris"
        return

    raw_output_format = params.get("output_format")
    if raw_output_format is None:
        params["output_format"] = "legacy"
        return

    params["output_format"] = str(raw_output_format).lower()


def _translate_calculate_min_error(params: dict[str, Any]) -> None:
    """Translate legacy ``calculate_min_error`` to the modern ``min_error`` option."""
    if "calculate_min_error" not in params:
        return

    value = params.pop("calculate_min_error")
    if not _legacy_option_enabled(value):
        return

    method = str(value).strip().lower()
    if method not in _MIN_ERROR_METHODS:
        raise ValueError(
            "`calculate_min_error` is deprecated and can only be translated when set to "
            f"one of {sorted(_MIN_ERROR_METHODS)!r}; use `min_error` instead."
        )

    warnings.warn(
        "`calculate_min_error` is deprecated. The run_hbmcmc compatibility shim is translating "
        "it to `min_error`.",
        FutureWarning,
        stacklevel=3,
    )
    params["min_error"] = method


def _translate_reparameterise_log_normal(params: dict[str, Any]) -> None:
    """Translate legacy lognormal reparameterisation flag into prior dictionaries."""
    value = params.pop("reparameterise_log_normal", False)
    if not _legacy_option_enabled(value):
        return

    warnings.warn(
        "`reparameterise_log_normal` is deprecated. The run_hbmcmc compatibility shim is setting "
        "`reparameterise=True` in lognormal emissions and BC prior dictionaries.",
        DeprecationWarning,
        stacklevel=3,
    )
    for name in _LOGNORMAL_PRIOR_NAMES:
        prior = params.get(name)
        if not isinstance(prior, dict):
            continue
        if str(prior.get("pdf", "")).lower() != "lognormal":
            continue
        prior = prior.copy()
        prior["reparameterise"] = True
        params[name] = prior


def _translate_legacy_options(params: dict[str, Any]) -> None:
    """Translate legacy fixedbasis options that have modern RHIME equivalents."""
    _translate_calculate_min_error(params)
    _translate_reparameterise_log_normal(params)


def fixedbasis_params_to_rhime(params: dict[str, Any]) -> dict[str, Any]:
    """Translate fixedbasis-style script/config parameters into RHIME arguments.

    The compatibility shim deliberately stays at the entrypoint boundary:
    legacy config spellings are normalised here, then the modern ``run_rhime``
    API performs its existing validation and spec construction.
    """
    translated = dict(params)
    mcmc_type = translated.pop("mcmc_type", "fixed_basis")
    if mcmc_type != "fixed_basis":
        raise ValueError(f"Unsupported run_hbmcmc mcmc_type {mcmc_type!r}; expected 'fixed_basis'.")

    _translate_legacy_aliases(translated)
    _normalise_legacy_output_format(translated)
    _translate_legacy_options(translated)
    translated["output_filename_convention"] = "legacy"
    if "save_inversion_output" not in translated and translated["output_format"] != "inv_out":
        translated["save_inversion_output"] = False
    return normalise_rhime_params(translated)


def validate_rhime_params(params: dict[str, Any]) -> None:
    """Validate translated single-sector RHIME params before script side effects."""
    make_rhime_runner_setup(
        params=params,
        multisector=False,
        data_param_names=set(inspect.signature(prepare_rhime_inputs).parameters),
    )


def hbmcmc_extract_param(
    config_file: str | Path,
    mcmc_type: str | None = "fixed_basis",
    print_param: bool | None = True,
    **command_line,
):
    """Extract fixedbasis-style parameters from an input configuration file.

    Checks the mcmc_type to extract the required parameters.

    Args:
      config_file:
        Configuration file name. Should be an .ini file.
      mcmc_type:
        Keyword for MCMC function to use.
        Default = "fixed_basis" (only option at present)
      print_param:
        When set to True, print out extracted parameter names.
        Default = True
      command_line:
        Any additional command line arguments to be added to the param
        dictionary or to supersede values contained within the config file.

    Returns:
      dict:
        Dictionary of parameter names and values from the fixedbasis-style
        configuration file plus command-line overrides.

    Raises:
        ValueError if expected parameter is missing or has `None` value.
    """
    expected_param = fixed_basis_expected_param() if mcmc_type == "fixed_basis" else []

    # If an expected parameter has been passed from the command line,
    # this does not need to be within the config file
    for key, value in command_line.items():
        if key in expected_param and value is not None:
            expected_param.remove(key)

    mcmc_type_section = "MCMC.TYPE"
    param = config.extract_params(
        config_file, expected_param=expected_param, ignore_sections=[mcmc_type_section]
    )

    # Command line values added to param (or supersede inputs from the config
    # file)
    for key, value in command_line.items():
        if value is not None:
            param[key] = value

    # If configuration file does not include values for the
    # required parameters - produce an error
    for ep in expected_param:
        if ep not in param or not param[ep]:
            raise ValueError(f"Required parameter '{ep}' has not been defined")

    if print_param:
        print("\nInput parameters: ")
        for key, value in param.items():
            print(f"{key} = {value}")

    return param


def build_parser(default_config_file: Path) -> argparse.ArgumentParser:
    """Build the legacy run_hbmcmc argument parser."""
    parser = argparse.ArgumentParser(description="Running Hierarchical Bayesian MCMC script")
    parser.add_argument("start", help="Start date string of the format YYYY-MM-DD", nargs="?")
    parser.add_argument("end", help="End date string of the format YYYY-MM-DD", nargs="?")
    parser.add_argument(
        "-c", "--config", help="Name (including path) of configuration file", default=default_config_file
    )
    parser.add_argument(
        "-r",
        "--generate",
        action="store_true",
        help="Generate template config file and exit (does not run MCMC simulation)",
    )
    parser.add_argument(
        "--kwargs",
        type=json.loads,
        help='Pass keyword arguments to mcmc function. Format: \'{"key1": "val1", "key2": "val2"}\'.',
    )
    parser.add_argument(
        "--output-path",
        help="Path to write ini file and results to.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run fixedbasis-style configs through the modern RHIME path."""
    openghginv_path = Paths.openghginv
    config_file = openghginv_path / "hbmcmc" / "hbmcmc_input.ini"

    args = build_parser(config_file).parse_args(argv)

    config_file = Path(args.config)
    command_line_args = {}
    if args.start:
        command_line_args["start_date"] = args.start
    if args.end:
        command_line_args["end_date"] = args.end
    if args.output_path:
        command_line_args["outputpath"] = args.output_path

    if args.kwargs:
        command_line_args.update(args.kwargs)

    if args.generate is True:
        template_file = openghginv_path / "hbmcmc" / "config" / "hbmcmc_input_template.ini"
        if config_file.exists():
            write = input(f"Config file {config_file} already exists.\nOverwrite? (y/n): ")
            if write.lower() == "y" or write.lower() == "yes":
                copyfile(template_file, config_file)
            else:
                sys.exit("Previous configuration file has not been overwritten.")
        else:
            copyfile(template_file, config_file)
        sys.exit(f"New configuration file has been generated: {config_file}")

    if not config_file.exists():
        raise ValueError(
            f"Configuration file cannot be found.\nPlease check path and filename are correct: {config_file}"
        )

    timing_start = timer_start()
    mcmc_type = extract_mcmc_type(config_file)
    print(f"Using MCMC type: {mcmc_type} - routing fixedbasis-style config to run_rhime(...)")
    param = hbmcmc_extract_param(config_file, mcmc_type, **command_line_args)
    log_timing("run_hbmcmc.config_extract", timer_seconds(timing_start))

    with timed("run_hbmcmc.fixedbasis_to_rhime_translation"):
        rhime_params = fixedbasis_params_to_rhime(param)

    with timed("run_hbmcmc.validation"):
        validate_rhime_params(rhime_params)

    with timed("run_hbmcmc.config_copy"):
        output.copy_config_file(str(config_file), param=param, **command_line_args)

    run_rhime(**rhime_params)


if __name__ == "__main__":
    main()
