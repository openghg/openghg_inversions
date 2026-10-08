"""Compatibility script for running old fixedbasis-style configs through RHIME.

This entry point preserves the historical ``run_hbmcmc.py`` command-line
surface for old INI files, but translates supported fixedbasis-style parameter
names and calls the modern ``run_rhime`` pathway.

Run as:
    $ python -m openghg_inversions.hbmcmc.run_hbmcmc [start end] -c config.ini
e.g.
    $ python -m openghg_inversions.hbmcmc.run_hbmcmc 2012-01-01 2013-01-01 -c hbmcmc_ch4_run.ini

start - Start of date range to use for MCMC inversion (YYYY-MM-DD)
end - End of date range to use for MCMC inversion (YYYY-MM-DD) (must be after start)
-c / --config - existing fixedbasis-style configuration file.
--all-chains - use every sampled chain in derived outputs. By default this
compatibility entry point warns and continues to use chain 0.

If start and end are specified these will supersede the values within the configuration file, if present.
"""

import argparse
import json
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from openghg_inversions._timing import log_timing, timed, timer_seconds, timer_start
from openghg_inversions.config import config
from openghg_inversions.models.additive_sigma import DEFAULT_ADDITIVE_SIGMA_PRIOR
from openghg_inversions.rhime import PollutionEventSettings, RhimeConfig, run_rhime
from openghg_inversions.hbmcmc.compatibility import fixedbasis_params_to_rhime as fixedbasis_params_to_rhime


_ADDITIVE_SIGMA_LIKELIHOOD = "additive_sigma"
_ADDITIVE_SIGMA_PRIOR = "additive_sigma_prior"
_ADDITIVE_SIGMA_OPTION_NAMES = (
    "sigma_prior",
    "sigma_freq",
    "sigma_per_site",
    "sigma_freq_anchor",
    "no_model_error",
)
_REQUIRED_FIXEDBASIS_PARAMS = (
    "species",
    "sites",
    "averaging_period",
    "domain",
    "start_date",
    "end_date",
    "outputpath",
    "outputname",
)


def _select_additive_sigma_model_options(
    raw_params: dict[str, Any],
    rhime_params: dict[str, Any],
) -> dict[str, Any] | None:
    """Resolve an INI additive selection into explicit model options.

    This compatibility entry point owns the policy that additive sigma never
    consumes aggregation error. The reusable component remains capable of
    serving CO2-family recipes which do own fixed aggregation covariance.
    """
    likelihood = raw_params.get("likelihood")
    additive_sigma_prior = raw_params.get(_ADDITIVE_SIGMA_PRIOR)
    if likelihood is None:
        if additive_sigma_prior is not None:
            raise ValueError("`additive_sigma_prior` requires likelihood='additive_sigma'.")
        return None
    if not isinstance(likelihood, str) or likelihood.strip().lower() != _ADDITIVE_SIGMA_LIKELIHOOD:
        raise ValueError(
            "run_hbmcmc supports only likelihood='additive_sigma'; omit `likelihood` "
            "to retain the historical pollution-event likelihood."
        )
    if rhime_params.get("aggregation_error_mode", "none") != "none":
        raise ValueError(
            "run_hbmcmc additive_sigma does not support `aggregation_error_mode`; "
            "remove it or set it to 'none'."
        )
    options: dict[str, Any] = {
        "mismatch_model": "additive_sigma",
        "use_minimum_error_floor": True,
        "aggregation_error_mode": "none",
    }
    rhime_params.pop("pollution_events_from_obs", None)
    rhime_params.pop("power", None)
    options.update(
        {name: rhime_params[name] for name in _ADDITIVE_SIGMA_OPTION_NAMES if name in rhime_params}
    )
    if options.get("sigma_freq") not in (None, "monthly"):
        options.setdefault("sigma_freq_anchor", rhime_params["start_date"])
    resolved_prior = (
        rhime_params.get("sigma_prior", DEFAULT_ADDITIVE_SIGMA_PRIOR)
        if additive_sigma_prior is None
        else additive_sigma_prior
    )
    if not isinstance(resolved_prior, Mapping):
        name = "sigma_prior" if additive_sigma_prior is None else _ADDITIVE_SIGMA_PRIOR
        raise ValueError(f"`{name}` must be a mapping/dict.")
    resolved_prior = dict(resolved_prior)
    options["sigma_prior"] = resolved_prior
    return options


def _validate_country_file(params: dict[str, Any]) -> None:
    """Reject a configured country file that does not exist."""
    country_file = params.get("country_file")
    if country_file is not None and str(country_file).strip():
        country_file_path = Path(country_file)
        if not country_file_path.exists():
            raise FileNotFoundError(f"Configured country_file does not exist: {country_file_path}")


def hbmcmc_extract_param(
    config_file: str | Path,
    print_param: bool | None = True,
    **command_line,
):
    """Extract fixedbasis-style parameters from an input configuration file.

    Args:
      config_file:
        Configuration file name. Should be an .ini file.
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
        ValueError: If the configuration selects a removed MCMC route or a
            required parameter is missing or falsey.
    """
    param = config.extract_params(config_file)

    # Command line values added to param (or supersede inputs from the config
    # file)
    for key, value in command_line.items():
        if value is not None:
            param[key] = value

    mcmc_type = param.pop("mcmc_type", "fixed_basis")
    if mcmc_type != "fixed_basis":
        raise ValueError(f"Unsupported run_hbmcmc mcmc_type {mcmc_type!r}; expected 'fixed_basis'.")

    # If configuration file does not include values for the
    # required parameters - produce an error
    for ep in _REQUIRED_FIXEDBASIS_PARAMS:
        if ep not in param or not param[ep]:
            raise ValueError(f"Required parameter '{ep}' has not been defined")

    if print_param:
        print("\nInput parameters: ")
        for key, value in param.items():
            print(f"{key} = {value}")

    return param


def build_parser() -> argparse.ArgumentParser:
    """Build the legacy run_hbmcmc argument parser."""
    parser = argparse.ArgumentParser(description="Run a fixedbasis-style config through RHIME")
    parser.add_argument("start", help="Start date string of the format YYYY-MM-DD", nargs="?")
    parser.add_argument("end", help="End date string of the format YYYY-MM-DD", nargs="?")
    parser.add_argument(
        "-c", "--config", help="Name (including path) of an existing configuration file", required=True
    )
    parser.add_argument(
        "--kwargs",
        type=json.loads,
        help='Pass RHIME keyword arguments. Format: \'{"key1": "val1", "key2": "val2"}\'.',
    )
    parser.add_argument(
        "--output-path",
        help="Path to write ini file and results to.",
    )
    parser.add_argument(
        "--all-chains",
        action="store_true",
        help="Use every sampled chain in derived outputs (recommended).",
    )
    return parser


def _copy_config_file(config_file: str | Path, param: dict[str, Any], **command_line: Any) -> None:
    """Copy the effective legacy configuration alongside the inversion output."""
    output_path = Path(param["outputpath"])
    output_filename = output_path / (
        f"{str(param['species']).upper()}_{param['domain']}_{param['outputname']}_{param['start_date']}.ini"
    )
    config_lines = Path(config_file).read_text(encoding="utf-8")

    if command_line:
        print("Adding inputs from command line to file")
        keyword_added = False
        for key, value in command_line.items():
            match = re.search(rf"\s*{re.escape(key)}\s*=\s*\S+", config_lines)
            if match is None:
                if not keyword_added:
                    config_lines += "\n\n[ADDED_FROM_COMMAND_LINE]\n"
                    config_lines += (
                        "; This section contains additional commands specified on the command line "
                        "with no equivalent entry in this file\n"
                    )
                config_lines += f"\n{key} = {value!r}\n"
                keyword_added = True
                continue

            original_line = match.group()
            current_key, current_value = original_line.split("=", maxsplit=1)
            config_lines = config_lines.replace(
                original_line,
                current_key + "=" + current_value.replace(current_value.strip(), repr(value)),
            )

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Copying input configuration file to: {output_filename}")
    output_filename.write_text(config_lines, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """Run a fixedbasis-style config through RHIME."""
    args = build_parser().parse_args(argv)

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

    if not config_file.exists():
        raise ValueError(
            f"Configuration file cannot be found.\nPlease check path and filename are correct: {config_file}"
        )

    timing_start = timer_start()
    param = hbmcmc_extract_param(config_file, **command_line_args)
    log_timing("run_hbmcmc.config_extract", timer_seconds(timing_start))

    print("Routing fixedbasis-style config to run_rhime(...)")

    with timed("run_hbmcmc.fixedbasis_to_rhime_translation"):
        rhime_params = fixedbasis_params_to_rhime(param)
        additive_sigma_options = _select_additive_sigma_model_options(param, rhime_params)
        no_model_error = bool(rhime_params.pop("no_model_error", False))
        legacy_unused_sigma_settings: PollutionEventSettings | None = None
        legacy_minimum_error_floor = False
        if additive_sigma_options is not None:
            no_model_error = bool(additive_sigma_options.get("no_model_error", no_model_error))
            if no_model_error:
                for name in _ADDITIVE_SIGMA_OPTION_NAMES:
                    rhime_params.pop(name, None)
                rhime_params.update(
                    mismatch_model="fixed_error",
                    aggregation_error_mode="none",
                )
                legacy_minimum_error_floor = True
            else:
                rhime_params.update(additive_sigma_options)
                rhime_params.pop("no_model_error", None)
        elif no_model_error:
            legacy_unused_sigma_settings = PollutionEventSettings(
                sigma_prior=rhime_params.pop("sigma_prior", None),
                sigma_freq=rhime_params.pop("sigma_freq", None),
                sigma_per_site=rhime_params.pop("sigma_per_site", True),
                sigma_freq_anchor=rhime_params.pop("sigma_freq_anchor", None),
            )
            rhime_params.pop("pollution_events_from_obs", None)
            rhime_params.pop("power", None)
            rhime_params["mismatch_model"] = "fixed_error"
        else:
            rhime_params["mismatch_model"] = "pollution_event"

    with timed("run_hbmcmc.validation"):
        resolved_config = RhimeConfig.from_params(rhime_params, multisector=False)

    _validate_country_file(rhime_params)

    if not args.all_chains:
        warnings.warn(
            "run_hbmcmc.py is preserving historical chain-0-only derived outputs. "
            "Pass --all-chains to use every sampled chain (recommended). Pooling chains "
            "does not establish convergence.",
            UserWarning,
            stacklevel=2,
        )

    # TODO(#423): Validate BC and saved fp-basis files, including glob matches and readability.
    with timed("run_hbmcmc.config_copy"):
        _copy_config_file(config_file, param=param, **command_line_args)

    run_rhime(
        preserve_legacy_likelihood=additive_sigma_options is None,
        _compatibility_unused_sigma_settings=legacy_unused_sigma_settings,
        _compatibility_minimum_error_floor=legacy_minimum_error_floor,
        compatibility_output_chain=None if args.all_chains else 0,
        config=resolved_config,
    )


if __name__ == "__main__":
    main()
