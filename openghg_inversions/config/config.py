#!/usr/bin/env python2
"""Configuration file utilities for INI format files.

This module allows configuration files in the INI format to be read and used.

Example of a section in the configuration file::

    [MEASUREMENTS]
    # Measurement details

    sites = ["GSN"]           ; Sites to read the data from as a list
    species = "chcl3"
    start_date = "2015-01-01" ; Default start date used if none specified on the command line
    end_date = "2015-02-01"   ; Default start date used if none specified on the command line
    domain = "EASTASIA"
    network = "AGAGE"

Configuration file format:

- Sections are included in square brackets
- Parameter name and value pairs are separated by an equals sign
- Values can be specified with the same syntax as when creating a python object 
  e.g. '' for string, [] for lists (and also for np.array - will be converted 
  if their type has been set as an array)
- ; and # symbols can be used to create new line and inline comments

Section headings can be of the form [NAME] or [GROUP.NAME]. This allows parameters 
to be separated into several section headings in the configuration file for clarity 
but grouped into one overall classification when inputted based on the GROUP name.

param_type dictionary:

To specify inputs and the types they should be cast to (e.g. str, array, boolean etc) 
a nested dictionary should be created. This can be passed into several functions as 
the param_type argument.

This should be of the form of one of the following:

- {'SECTION_GROUP1':{'param1':str,'param2':float},'SECTION_GROUP2':{'param3':list,'param4':np.array}}
- {'SECTION1':{'param1':str},'SECTION2':'{param2':float},'SECTION3':{'param3':list},'SECTION4':{'param4':np.array}}
- OrderedDict(['SECTION_GROUP1':OrderedDict([('param1':str),('param2':float)]),'SECTION_GROUP2':OrderedDict([('param3':list),('param4':np.array)]))
- OrderedDict(['SECTION1':{'param1':str},'SECTION2':{'param2':float},'SECTION3':{'param3':list},'SECTION4':{'param4':np.array}])

This can either be created directly or a template configuration file can be created and 
a param_type dictionary created from this using the generate_param_dict() function.
These template files should be kept within the acrg_config/templates/ directory and, 
after creation, should not be altered unless you wish to change the format for all 
config files of this type.

Note: if param_type is not defined, the code will attempt to cast the inputs to the 
most sensible type. This should be fine in most cases but could cause issues.
This also means the the set of input parameters will not be checked for any missing values.

How to run:

The main functions to use for reading in parameters from a config file are:

- all_param(config_file,...)      ; Extract all parameters from a configuration file.
- extract_params(config_file,...) ; Extract specific parameters from a file either based on parameter names, sections or groups.

A param_type dictionary can be defined both to fix expected inputs and to explictly 
specify the parameter types.

@author: rt17603
"""

import configparser
from collections import OrderedDict
import numpy as np
import os


def open_config(config_file):
    """Open configuration files in the ini format.

    Args:
        config_file: Filename for input configuration file.

    Returns:
        configparser.ConfigParser: Parsed configuration object.
    """
    config = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    config.optionxform = str  # Keeps case when inputting option names

    with open(config_file, encoding="utf-8") as fp:
        config.read_file(fp)

    return config


def generate_param_dict(config_file):
    """The generate_param_dict function creates a param_type nested dictionary from an input configuration
    file.
    This could be used on some fixed template config file to generate the parameter type dictionary which
    can then be applied to other configuration files of the same type.

    Args:
        config_file (str) :
            Filename for template configuration file (str).

    Returns:
        nested OrderedDict :
            Parameter type dictionary from configuration file input with sections as keys for parameters
            and value types.
    """
    config = open_config(config_file)

    sections = config.sections()  # Extract all section names from the config file

    param_type = OrderedDict([])

    for section in sections:
        section_param = list(config[section].keys())
        if section_param:
            types = [type(convert(value)) for value in list(config[section].values())]
            section_types = OrderedDict([(key, value) for key, value in zip(section_param, types)])
            param_type[section] = section_types

    return param_type


def generate_from_template(template_file, output_file):
    """Generate an example configuration file based on a template file.
    
    Template files are normally used to inform the expected format of any input configuration file.

    Args:
        template_file: Input template file in expected .ini format.
        output_file: Name of output file including path information.

    Note:
        Writes output file. If output file is already present, the user will be asked whether 
        the file should be overwritten. If the response is 'N' or 'no' or an unrecognised 
        input an exception will be raised.
    """
    if os.path.exists(output_file):
        answer = input(
            f"This action with overwrite existing {output_file} file. Do you wish to proceed (Y/N): "
        )
        if answer.lower() == "y" or answer.lower() == "yes":
            out = open(output_file, "w", encoding="utf-8")
        elif answer.lower() == "n" or answer.lower() == "no":
            raise Exception("Configuration file has not been generated.")
        else:
            raise Exception(
                f"Did not understand input: '{answer}'. Configuration file has not been regenerated."
            )
    else:
        out = open(output_file, "w", encoding="utf-8")

    copy = False
    with open(template_file, encoding="utf-8") as fname:
        for line in fname:
            if copy:
                out.write(line)
            elif not line.strip():
                # print('Empty line',i)
                continue
            elif line.strip().startswith("##"):
                # print("##",i)
                continue
            else:
                copy = True
                # print("Writing out from line: {}".format(i))
                out.write(line)

    print(f"Configuration file: {output_file} has been generated.")

    out.close()


def str_check(string, error=True):
    """The str_check function is used as part of checking the input from a configuration file.
    This function ensures the input remains as a string and removes any " or ' characters.

    Args:
        string (str) :
            Value input from config file
        error (bool, optional) :
            Print error message if unable to evaluate (bool).

    Returns:
        string (formatted)
    """
    if string is None:
        return None
    # Remove any ' or " symbols surrounding the input string from the config_file
    string = string.strip()  # Strip any whitespace just in case
    if (string[0] == "'" and string[-1] == "'") or (string[0] == '"' and string[-1] == '"'):
        string = string[1:-1]

    try:
        out = str(string)
        out.encode("ascii", "ignore")
    except (TypeError, SyntaxError):
        if error:
            print(f"WARNING: Could not convert input parameter '{string}' to str.")
        return None

    return out


def eval_check(string, error=True):
    """Evaluate the input string from a configuration file to a python object.
    
    Examples:
        - '1' would evalute to an int object 1
        - '1.' or '1.0' would evaluate to float object 1.0
        - "[1,2,3]" would evalute to a list object [1,2,3]
        - "1,2,3" would evaluate to a tuple object (1,2,3)
    
    See eval() documentation for full list.

    Args:
        string: Value input from config file.
        error: Print error message if unable to evaluate.

    Returns:
        tuple: Python object and boolean indicating success.
            If unable to convert string to python object, returns (None, False).
    """
    try:
        out = eval(string)
    except (NameError, SyntaxError):
        try:
            out = eval(
                "'" + string + "'"
            )  # An input string without quotes cannot be evaluated so try adding quotes
        except (NameError, SyntaxError):
            if error:
                print(f"WARNING: Could not evaluate input '{string}' to any type.")
            return None, False

    return out, True


def list_check(string, force_convert=True, error=True):
    """Convert input string to a list.

    Args:
        string: Value input from config file.
        force_convert: Specifies whether conversion to a list should be forced 
            (i.e. out = [out] if unable to evaluate any other way). Default = True.
        error: Print error message if unable to evaluate.

    Returns:
        list or None: List if conversion successful, None if unable to convert to list.
    """
    out, _ = eval_check(string)  # Try evaluating input

    if not isinstance(out, list | str):  # If not already a list
        try:
            out = list(out)
        except TypeError:
            if force_convert:
                out = [out]
            else:
                if error:
                    print(f"WARNING: Could not convert input parameter '{out}' to list.")
                return None
    elif isinstance(out, str):
        out = [out]

    return out


# def float_check(string,error=True):
#
#    out = eval_check(string)
#
#    try:
#        out = float(out)
#    except ValueError:
#        if error:
#            print 'Could not convert input parameter to float: {0}'.format(out)
#        return None
#
#    return out

# def int_check(string,error=True):
#
#    out = eval_check(string)
#
#    try:
#        out = int(out)
#    except ValueError:
#        if error:
#            print 'Could not convert input parameter to int: {0}'.format(out)
#        return None
#
#    return out


def convert(string, value_type=None):
    """Convert the input string to the specified value_type.
    
    If no value_type is stated, the function attempts to discern the appropriate type.

    Args:
        string: Value input from config file.
        value_type: Object type. Values accepted: "str", "list". Optional.

    Returns:
        Python object of specified type, or None if unable to convert to value_type 
        or find a suitable type when value_type is not specified.
    """
    if value_type is str:
        out = str_check(string)
    elif value_type is list:
        out = list_check(string)
    elif value_type is np.ndarray:
        out = list_check(string)
        out = np.array(out)
    #    elif value_type is float:
    #        out = float_check(string)
    #    elif value_type is int:
    #        out = int_check(string)
    else:
        out, check = eval_check(string, error=False)
        if out is None and check is False:
            out = str_check(out, error=False)

    return out


# def check_params(param,param_type,keys=[],optional_param=[],raise_exception=True):
#    '''
#    The check_params function
#    '''
#    names = param.keys()
#
#    if keys:
#        check_keys = keys
#    else:
#        check_keys = param_type.keys()
#
#    check_names = []
#    for key in check_keys:
#        try:
#            check_names.extend(param_type[key].keys())
#        except KeyError:
#            print "Key '{0}' cannot be found in param_type passed to check_params function".format(key)
#            return None
#
#    for name in check_names:
#        if name not in names:
#            if name not in optional_param:
#                if raise_exception:
#                    raise Exception("Parameter '{0}' must be specified in configuration file".format(name))
#                else:
#                    return False
#
#    return True


def all_parameters_in_param_type(param_type):
    """The all_parameters_in_param_type function extracts all parameters (regardless of section/section_group) for a given
    param_type nested dictionary.

    Args:
        param_type (dict) :
            Nested dictionary of expected parameter names and types.
            Key for each parameter dictionary can be the section heading or the overall group (e.g. for [MCMC.MEASUREMENTS], section group should be 'MCMC').
            See module header for expected formats of param_type dictionary.

    Returns:
        list :
            list of all parameters in param_type dictionary
    """
    types = param_type  # Get dictionary containing parameter names and types
    keys = list(types.keys())  # Extract all section/section_group headings
    parameters = []
    for key in keys:
        parameters.extend(
            list(types[key].keys())
        )  # For each section/section_group extract all the associated parameters

    return parameters


def find_param_key(param_type, section=None, section_group=None):
    """The find_param_key function checks whether the keys within the param_type dictionary are for sections
    (e.g. 'MCMC.MEASUREMENTS') or groups (e.g. 'MCMC') and returns the relevant key(s).
    One of section or section_group should be specified.
    Returned key_type is one of 'section' or 'section_group'.

    Args:
        param_type (dict) :
            Nested dictionary of expected parameter names and types.
            Key for each parameter dictionary can be the section heading or the overall group (e.g. for [MCMC.MEASUREMENTS], section group should be 'MCMC').
            See module header for expected formats of param_type dictionary.
        section (str, optional) :
            Name of section in config file for the parameter
        section_group (str, optional) :
            Name of group in config file for the parameter

    Returns:
        list,str:
            keys, key_type
    """
    types = param_type  # Get dictionary containing parameter names and types
    all_keys = list(types.keys())  # Extract all section/classification keys

    # Find parameter class if not specified (should be defined as first part of section split by '.' e.g. MCMC.MEASUREMENTS, section_group='MCMC')
    if not section_group:
        if not section:
            raise ValueError("One of `section` or `section_group` must be provided.")
        section_group = section.split(".")[0]

    if section in all_keys:
        keys = [section]
        key_type = "section"
    elif section_group in all_keys:
        keys = [section_group]
        key_type = "section_group"
    elif section_group in [k.split(".")[0] for k in all_keys]:
        keys = [k for k in all_keys if k.split(".")[0].lower() == section_group.lower()]
        # keys_starter = [k.split('.')[0] for k in all_keys]
        # keys = all_keys[keys_starter.index(section_group)]
        key_type = "section"
    else:
        keys = [None]
        key_type = None
        # raise Exception('Section/Classification {0}/{1} does not match to any key in input param_type'.format(section_group,section))
        # print('Param class cannot be found i for section of parameters not defined. Using {0} as default'.format(section_groups[0]))
        # section_group = section_groups[0]

    return keys, key_type


def get_value(name, config, section, param_type=None):
    """The get_value function extracts the value of a parameter from the configuration file.
    This value is then converted to the type specified within param_type (default from mcmc_param_type()
    function).

    Args:
        name (str) :
            Name of the parameter to extract
        config (ConfigParser) :
            ConfigParser object created using the configparser class. Data from config file should have
            been read in.
        section (str) :
            Name of section in config file for the parameter
        param_type (dict, optional) :
            nested dictionary of parameter classes and expected parameter names and types.
            Key for each parameter dictionary can be the section heading or the overall classification
            (e.g. for [MCMC.MEASUREMENTS], classification should be 'MCMC').
            See module header for expected formats of param_type dictionary.

    Returns:
        value

        If param_type is specified
            If neither section nor section_group can be identified within param_type dictionary:
                Exception raised and program exited
            If parameter name cannot be identified within param_type dictionary:
                Exception raised and program exited
    """
    if param_type:
        keys, key_type = find_param_key(param_type, section)
        key = keys[0]  # Should only ever be one key for a section
        types = param_type
        try:
            value_type = types[key][name]  # Find specified type of object for input parameter
        except KeyError:
            # raise Exception('Input parameter {0} in section {1} not expected (not found in param_type dictionary [{2}][{0}])'.format(name,section,key))
            print(f"Type for input name '{name}' is not specified.")
            value_type = None
    else:
        # print "Type for input name '{0}' is not specified.".format(name)
        # value_type = str
        value_type = None

    # For int, float and bool object functions within config module exist to cast directly to these types
    if value_type == int:
        value = config.getint(section, option=name)
    elif value_type == float:
        value = config.getfloat(section, option=name)
    elif value_type == bool:
        value = config.getboolean(section, option=name)
    else:  # For lists, np.ndarrays and str objects we extract as a string and manually format
        value = config.get(section, option=name)
        value = convert(value, value_type=value_type)

    return value


def extract_params(
    config_file,
    expected_param=[],
    section=None,
    section_group=None,
    names=[],
    ignore_sections=[],
    ignore_section_groups=[],
    # optional_param=[],optional_section=[],optional_section_group=[],
    exclude_not_found=True,
    allow_new=False,
    param_type=None,
):
    """Extract parameter names and values from a configuration file.
    
    The parameters which are extracted is dependent on whether the section, section_group 
    and/or names variables are specified. A param_type dictionary can be defined to ensure 
    variables are cast to the correct types.

    Args:
        config_file: Filename for input configuration file.
        expected_param: Parameters within the configuration file which must be returned.
            An error will be raised if these parameters are not present.
        section: Extract parameters from section name(s).
        section_group: Extract parameters from all sections with this group (these groups).
            If section and section_group are both specified - section takes precedence.
        names: Parameter names to extract (within section or section_group, if specified).
        ignore_sections: Sections to ignore when reading in the configuration file 
            (even if parameters are specified).
        ignore_section_groups: Sections groups to ignore when reading in the configuration 
            file (even if parameters are specified).
        exclude_not_found: Whether to remove parameters which are not found in the input 
            file or include them as None. Default = True.
        allow_new: If a param_type is specified, whether to allow unrecognised parameters 
            to be added without printing a warning. Default = False (i.e. Warning will be printed).
        param_type: Nested dictionary of sections or groups and expected parameter names and types.
            See module header for expected formats of param_type dictionary.

    Returns:
        OrderedDict: Parameter names and values.
    """
    # Open config file with configparser
    config = open_config(config_file)

    all_sections = config.sections()  # Extract all section names from the config file

    if section:
        if isinstance(section, str):
            section = [section]
        select_sections = []
        for s in section:
            if s in all_sections:
                select_sections.append(s)
            else:
                # raise KeyError('Specified section {0} could not be found in configuration file: {1}'.format(s,config_file))
                print(f"Specified section {s} could not be found in configuration file: {config_file}")
                return None
    elif section_group:
        if isinstance(section_group, str):
            section_group = [section_group]
        select_sections = []
        for sg in section_group:
            s_sections = [
                s for s in all_sections if s.split(".")[0].lower() == sg.lower()
            ]  # Find all sections covered by section_group (section_group.name)
            select_sections.extend(s_sections)
        if not select_sections:
            # raise KeyError('No sections could be found for specified section_group {0} in configuration file: {1}'.format(section_group,config_file))
            print(
                f"No sections could be found for specified section_group {section_group} in configuration file: {config_file}"
            )
            return None
    elif ignore_sections:
        select_sections = all_sections
        for es in ignore_sections:
            if es in select_sections:
                select_sections.remove(es)
            else:
                # raise KeyError('Specified section {0} could not be found in configuration file: {1}'.format(es,config_file))
                print(f"Specified section {es} could not be found in configuration file: {config_file}")
    elif ignore_section_groups:
        select_sections = all_sections
        for esg in ignore_section_groups:
            ignore_s = [s for s in all_sections if s.split(".")[0].lower() == esg.lower()]
            if not ignore_s:
                # raise KeyError('No sections could be found for specified section_group {0} in configuration file: {1}'.format(esg,config_file))
                print(
                    f"No sections could be found for specified section_group {esg} in configuration file: {config_file}"
                )
            for es in ignore_s:
                if es in select_sections:
                    select_sections.remove(es)
    else:
        select_sections = all_sections  # Find all sections

    # Extracting parameter names from the input config file. Filter by names if already present
    extracted_names = []
    match_section = []
    # pdb.set_trace() # REMOVE
    if names:
        for sect in select_sections:
            k = list(config[sect].keys())
            for name in names:
                if name in k:
                    extracted_names.append(name)
                    match_section.append(sect)
    else:
        for sect in select_sections:
            k = list(config[sect].keys())
            s = [sect] * len(k)
            extracted_names.extend(
                k
            )  # List of the parameter names within the input file, within specified sections
            match_section.extend(s)  # Associated list with the section heading for each parameter

    # Creating list of names we want to put into the parameter dictionary based on inputs (e.g. section, section_group)
    # pdb.set_trace() # REMOVE
    if not names:
        if param_type:
            if section_group:
                keys = []
                for i, sg in enumerate(section_group):
                    if i == 0:
                        k, key_type = find_param_key(section_group=sg, param_type=param_type)
                    else:
                        k = find_param_key(section_group=sg, param_type=param_type)[0]
                    keys.extend(k)
                # keys,key_type = find_param_key(section_group=section_group,param_type=param_type)
                # keys = [key_value]
            elif section:
                keys = []
                for i, s in enumerate(section):
                    if i == 0:
                        k, key_type = find_param_key(section=s, param_type=param_type)
                    else:
                        k = find_param_key(section=s, param_type=param_type)[0]
                    keys.extend(k)
                # keys,key_type = find_param_key(section=section,param_type=param_type)
                # keys = [key_value]
            elif ignore_sections:
                keys = list(param_type.keys())
                for es in ignore_sections:
                    key_type = find_param_key(section=es, param_type=param_type)[1]
                    if es in keys:
                        keys.remove(es)
            elif ignore_section_groups:
                keys = list(param_type.keys())
                for esg in ignore_section_groups:
                    ignore_s = [k for k in keys if k.split(".")[0].lower() != esg.lower()]
                    key_type = find_param_key(section_group=esg, param_type=param_type)[1]
                    for es in ignore_s:
                        if es in keys:
                            keys.remove(es)
            else:
                keys = list(param_type.keys())

            # print('Keys to extract input names from param_type: {0}'.format(keys))
            names = []
            if (section and key_type == "section_group") or (ignore_sections and key_type == "section_group"):
                print(
                    "WARNING: Cannot create list of necessary input parameters based on param_type input. Please check all inputs are included (or excluded) manually."
                )
                names = extracted_names  # Set to just match names extracted from file
            #            elif (section_group and key_type == 'section_group') or (section and key_type == 'section') or (section_group and key_type == 'section'):
            else:
                for key in keys:
                    names.extend(list(param_type[key].keys()))
        #            else:
        #                names = all_parameters_in_param_type(param_type) # Extract all parameter names from param_type dictionary
        else:
            names = extracted_names.copy()  # Set to just match names extracted from file

    for ep in expected_param:
        if ep not in names:
            names.append(ep)

    # pdb.set_trace() # REMOVE

    #    if optional_section:
    #        keys,key_type = find_param_key(section=optional_section,param_type=param_type)
    #        if key_type == 'section_group':
    #            print('WARNING: Cannot create list of optional parameters for the section based on param_type input. Please add parameters to optional_parameters list.')
    #        else:
    #            for key in keys:
    #                optional_param.extend(list(param_type[key].keys()))
    #
    #    if optional_section_group:
    #        keys,key_type = find_param_key(section_group=optional_section_group,param_type=param_type)
    #        for key in keys:
    #            optional_param.extend(list(param_type[key].keys()))

    param = OrderedDict({})

    for name in names:
        if name in extracted_names:
            try:
                index = extracted_names.index(name)
            except ValueError:
                print(
                    f"WARNING: Parameter '{name}' not found in configuration file (check specified section {section} or section_group {section_group} is correct)."
                )
            else:
                param[name] = get_value(name, config, match_section[index], param_type)
        elif name in expected_param:
            if section:
                raise KeyError(
                    f"Expected parameter '{name}' not found in input configuration file in section '{section}'"
                )
            elif section_group:
                raise KeyError(
                    f"Expected parameter '{name}' not found in input configuration file within section_group '{section_group}'"
                )
            else:
                raise KeyError(f"Expected parameter '{name}' not found in input configuration file.")
        elif not param_type:
            print(
                f"WARNING: Parameter '{name}' not found in configuration file (check specified section {section} or section_group {section_group} is correct)."
            )
        elif exclude_not_found:
            pass
        else:
            param[name] = None

    for index, extracted_name in enumerate(extracted_names):
        if extracted_name not in param:
            if not allow_new:
                print(
                    f"WARNING: Unknown parameter '{extracted_name}' found in configuration file. Please add to template file and define input type."
                )
            param[extracted_name] = get_value(extracted_name, config, match_section[index])

    # if exclude_not_found:
    #    param = OrderedDict([(key,value) for key,value in param.iteritems() if value != None])

    # print("names",names)

    return param


def all_param(
    config_file,
    expected_param=[],
    param_type=None,
    exclude_not_found=False,
    allow_new=False,
    # optional_param=[]
):
    """Extract all parameters from a config file.
    
    If param_type specified will cast to the specified types, otherwise will attempt 
    to discern the parameter types from the form of the values.

    Args:
        config_file: Filename for input configuration file.
        expected_param: Parameters within the configuration file which must be returned.
            An error will be raised if these parameters are not present.
        param_type: Nested dictionary of sections or groups and expected parameter names and types.
            See module header for expected formats of param_type dictionary.
        exclude_not_found: Whether to remove parameters which are not found in the input 
            file or include them as None.
        allow_new: If a param_type is specified, whether to allow unrecognised parameters 
            to be added without printing a warning. Default = False (i.e. Warning will be printed).

    Returns:
        OrderedDict: Parameter names and values.
    """
    param = OrderedDict({})
    param = extract_params(
        config_file,
        # optional_param=optional_param,
        expected_param=expected_param,
        param_type=param_type,
        exclude_not_found=exclude_not_found,
        allow_new=allow_new,
    )

    return param
