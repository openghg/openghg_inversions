"""Wrapper script to read in parameters from a configuration file and run underlying MCMC script.

Run as:
    $ python run_hbmcmc.py [start end -c config.ini]
e.g.
    $ python run_hbmcmc.py
    $ python run_hbmcmc.py 2012-01-01 2013-01-01 -c hbmcmc_ch4_run.ini

start - Start of date range to use for MCMC inversion (YYYY-MM-DD)
end - End of date range to use for MCMC inversion (YYYY-MM-DD) (must be after start)
-c / --config - configuration file. See config/ folder for templates and examples of this input file.

If start and end are specified these will superceed the values within the configuration file, if present.
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
from pathlib import Path
from shutil import copyfile
from collections.abc import Callable

import openghg_inversions.hbmcmc.hbmcmc as mcmc
import openghg_inversions.hbmcmc.hbmcmc_output as output

from openghg_inversions.config import config
from openghg_inversions.config.paths import Paths


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


def extract_mcmc_type(config_file: str, default: str = "fixed_basis") -> str:
    """Find value which describes the MCMC function to use.

    Checks the input configuation file the "mcmc_type" keyword within
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


def define_mcmc_function(mcmc_type: str) -> Callable:
    """Links mcmc_type name to function.

    Args:
      mcmc_type (str):
        Keyword for MCMC function to use.
        Current option "fixed_basis" (openghg_inversions.hbmcmc.fixedbasisMCMC())

    Returns:
      Function
    """
    function_dict = {"fixed_basis": mcmc.fixedbasisMCMC}

    return function_dict[mcmc_type]


def hbmcmc_extract_param(
    config_file: str, mcmc_type: str | None = "fixed_basis", print_param: bool | None = True, **command_line
):
    """Extract parameters from input configuration file and associated MCMC function.

    Checks the mcmc_type to extract the required parameters.

    Args:
      config_file:
        Configuration file name. Should be an .ini file.
      mcmc_type:
        Keyword for MCMC function to use.
        Default = "fixed_basis" (only option at present)
      print_param:
        Went set to True, print out extracted parameter names.
        Default = True
      command_line:
        Any additional command line arguments to be added to the param
        dictionary or to superceed values contained within the config file.

    Returns:
      function,collections.OrderedDict:
        MCMC function to use, dictionary of parameter names and values passed
        to MCMC function

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

    # Command line values added to param (or superceed inputs from the config
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


if __name__ == "__main__":
    openghginv_path = Paths.openghginv
    config_file = openghginv_path / "hbmcmc" / "hbmcmc_input.ini"

    parser = argparse.ArgumentParser(description="Running Hierarchical Bayesian MCMC script")
    parser.add_argument("start", help="Start date string of the format YYYY-MM-DD", nargs="?")
    parser.add_argument("end", help="End date sting of the format YYYY-MM-DD", nargs="?")
    parser.add_argument(
        "-c", "--config", help="Name (including path) of configuration file", default=config_file
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

    args = parser.parse_args()

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
            "Configuration file cannot be found.\n"
            f"Please check path and filename are correct: {config_file}"
        )

    mcmc_type = extract_mcmc_type(config_file)
    mcmc_function = define_mcmc_function(mcmc_type)
    print(f"Using MCMC type: {mcmc_type} - function {mcmc_function.__name__}(...)")

    param = hbmcmc_extract_param(config_file, mcmc_type, **command_line_args)

    output.copy_config_file(config_file, param=param, **command_line_args)

    mcmc_function(**param)
