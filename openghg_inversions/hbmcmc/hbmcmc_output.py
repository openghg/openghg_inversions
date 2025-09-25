"""Functions for configuring HBMCMC inversion output files."""

import os
import re
import xarray as xr
from pathlib import Path
from openghg_inversions.config import config


def ncdf_encoding(ds_in: xr.Dataset) -> dict:
    """Define encoding for netCDF4 files.
    Args:
      ds_in: xarray dataset to define encoding for

    Returns:
      Dictionary with encoding parameters for netCDF4 files.
    """
    # variables with variable length data types shouldn't be compressed
    # e.g. object ("O") or unicode ("U") type
    do_not_compress = []
    dtype_pat = re.compile(r"[<>=]?[UO]")  # regex for Unicode and Object dtypes
    for dv in ds_in.data_vars:
        if dtype_pat.match(ds_in[dv].data.dtype.str):
            do_not_compress.append(dv)
    encoding = {var: {"zlib": True, "complevel": 5, "shuffle": True} for var in ds_in.data_vars if var not in do_not_compress}
    
    return encoding


def check_and_create_folder(outputpath: str | Path) -> None:
    """Check folder exists and create if not.

    Args:
      outputpath: path of folder to check exists
    """
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)


def define_output_filename(
    outputpath: str | Path, species: str, domain: str, outputname: str, start_date: str, ext: str = ".nc"
) -> str:
    """Create output file name to write results to.

    Output filename based on the format:
    'outputpath'/'species'_'domain'_'outputname'_'start_date''ext'
    e.g. /home/user/output/CH4_EUROPE_test_2014-01-01.nc

    Args:
        outputpath: Directory where to save outputfile.
        species: Atmospheric trace gas species of interest (e.g. 'co2').
        domain: Name of modelling domain used (e.g. 'EUROPE').
        outputname: Additional str to include in filename (e.g. 'Test').
        start_date: Start date of inversion in format YYYY-MM-DD.
        ext: File extension. Defaults to .nc.

    Returns:
        str: Fullpath with filename of output file.
    """
    outputname = os.path.join(outputpath, f"{species.upper()}_{domain}_{outputname}_{start_date}{ext}")

    return outputname


def copy_config_file(config_file: str, param: dict | None = None, **command_line) -> None:
    """Creating a copy of the inputs used to run MCMC code based
    on the input config file and any additional parameters
    specified on the command line.

    Writes output file to same location as MCMC output
    (output filename based on define_output_filename()
    function with '.ini' extension)

    Values to create the output filename and location are either
    extracted from the config_file directly or from the input
    param dictionary if specified.

    Any additional command line arguments can be specified as keyword arguments.
    e.g. start_date="2018-01-01", end_date="2019-01-01"

    Args:
      config_file:
        Input configuration file name. Should be an .ini file.
      param:
        Optional param dictionary used directly in input to MCMC code.
        Just included as a convenience so config_file doesn't have to be
        read twice but should be the same inputs contained within
        the configuration file unless they have been superceeded by command
        line arguments.
      **command_line :
        Any additional keyword arguments from the command line input.
    """
    param_for_output_name = ["outputpath", "species", "domain", "outputname", "start_date"]

    if param is not None:
        parameters = {k: param[k] for k in param_for_output_name}
    else:
        parameters = config.extract_params(config_file, names=param_for_output_name)

        for key, value in command_line.items():
            if key in param_for_output_name and value is not None:
                parameters[key] = value

    output_filename = define_output_filename(ext=".ini", **parameters)
    # copyfile(config_file,output_filename)

    check_and_create_folder(parameters["outputpath"])

    raw_config_file = open(config_file, encoding="utf-8")
    config_lines = raw_config_file.read()

    if len(command_line) > 0:
        print("Adding inputs from command line to file")

        keyword_added = False
        for key, value in command_line.items():
            search_str = re.compile(rf"\s*{key}\s*=\s*\S+")
            found = re.search(search_str, config_lines)

            if found is None:
                if not keyword_added:
                    config_lines += "\n\n[ADDED_FROM_COMMAND_LINE]\n"
                    config_lines += "; This section contains additional commands specified on the command line with no equivalent entry in this file\n"
                if isinstance(value, str):
                    config_lines += f"\n{key} = '{value}'\n"
                else:
                    config_lines += f"\n{key} = {value}\n"

                keyword_added = True
            else:
                org_line = found.group()
                current_key, current_value = org_line.split("=")
                to_replace = current_value.strip()  # .replace('"','').replace("'",'')

                new_value = repr(value)
                # if isinstance(value,str) and to_replace == 'None':
                #    new_value = f"'{value}'"
                # else:
                #    new_value = str(value)

                new_line = current_key + "=" + current_value.replace(to_replace, new_value)
                config_lines = config_lines.replace(org_line, new_line)

    print(f"Copying input configuration file to: {output_filename}")
    output_file = open(output_filename, "w", encoding="utf-8")
    output_file.write(config_lines)
