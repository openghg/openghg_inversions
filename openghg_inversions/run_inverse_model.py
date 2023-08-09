'''
Wrapper script to read in parameters from a configuration file and run underlying MCMC script.

Run as:
    $ python run_inverse_model.py [start end -m model -c config.ini]
e.g.
    $ python run_inverse_model.py
    $ python run_inverse_model.py 2012-01-01 2013-01-01 -m hbmcmc -c hbmcmc_ch4_run.ini
    $ python run_inverse_model.py 2021-01-01 2021-02-02 -m multi_gas -c ch4_c2h6_test.ini

start - Start of date range to use for MCMC inversion (YYYY-MM-DD)
end - End of date range to use for MCMC inversion (YYYY-MM-DD) (must be after start)
-c / --config - configuration file. See config/ folder for templates and examples of this input file.
-m / --model - model name. Currently either 'hbmcmc' or 'multi_gas'
If start and end are specified these will superceed the values within the configuration file, if present.
If -c option is not specified, this script will look for a configuration file within the 
hbmcmc/config or multi_gas_model/config directories. 

To generate a config file from the template run this script as:
    $ python run_inverse modelj.py -r  [-c config.ini]

The MCMC run *will not be executed*. This will be named for your -c input or, if not specified, this will 
create a configuration file called `hbmcmc_input.ini` within your acrg_hbmcmc/ directory and exit. 
This file will need to be edited to add parameters for your MCMC run.

-----------------------------------------------------------------------------------
Updated in August 2023 to include options for either HBMCMC or the multi gas model
-----------------------------------------------------------------------------------
'''

import os
import sys
import argparse
from shutil import copyfile

from openghg_inversions.hbmcmc import hbmcmc
from openghg_inversions.hbmcmc import hbmcmc_output

from openghg_inversions.multi_gas_model import multi_gas_model

import openghg_inversions.config.config as config
from openghg_inversions.config.paths import Paths

def fixed_basis_expected_param():
    '''
    Define required parameters for openghg_inversions.hcmcmc.fixedbasisMCMC()
    
    Expected parameters currently include:
      species, sites, meas_period, start_date, end_date, domain, 
      outputpath, outputname
    
    Returns:
      Required parameter names (list)
    '''
    expected_param = ["species", "sites", "meas_period", "domain","start_date",
                      "end_date", "outputpath", "outputname"]

    return expected_param

def extract_mcmc_type(config_file,default="fixed_basis"):
    '''
    Find value which describes the MCMC function to use.
    Checks the input configuation file the "mcmc_type" keyword within 
    the "MCMC.TYPE" section. If not present, the default is used.
    -----------------------------------
    Args:
      config_file (str):
        Configuration file name. Should be an .ini file.
      default (str):
        ***    
    Returns:
      Keyword for MCMC function to use (str)
    -----------------------------------
    '''
    mcmc_type_section = "MCMC.TYPE"
    mcmc_type_keyword = "mcmc_type"
    param_mcmc_type = config.extract_params(config_file, 
                                            section=mcmc_type_section)
    
    if param_mcmc_type is not None and mcmc_type_keyword in param_mcmc_type:
        mcmc_type = param_mcmc_type[mcmc_type_keyword]
    else:
        mcmc_type = default
       
    return mcmc_type
    
def define_mcmc_function(mcmc_type):
    '''
    Links mcmc_type name to function.
    -----------------------------------
    Current options:
      mcmc_type (str):
        "fixed_basis" : openghg_inversions.hbmcmc.fixedbasisMCMC()
    
    Returns:
      Function
    -----------------------------------
    '''
    function_dict = {"fixed_basis":hbmcmc.fixedbasisMCMC}
    
    return function_dict[mcmc_type]

def hbmcmc_extract_param(config_file, mcmc_type="fixed_basis", print_param=True,**command_line):
    '''
    Extract parameters from input configuration file and associated 
    MCMC function. Checks the mcmc_type to extract the required 
    parameters.
    -----------------------------------
    Args:
      config_file (str):
        Configuration file name. Should be an .ini file.
      mcmc_type (str, optional):
        Keyword for MCMC function to use.
        Default = "fixed_basis" (only option at present)
      print_param (bool, optional):
        Print out extracted parameter names.
        Default = True
      command_line:
        Any additional command line arguments to be added to the param 
        dictionary or to superceed values contained within the config file.
    
    Returns:
      function,collections.OrderedDict:
        MCMC function to use, dictionary of parameter names and values passed 
        to MCMC function
    -----------------------------------
    ''' 
    if mcmc_type == "fixed_basis":
        expected_param = fixed_basis_expected_param()

    # If an expected parameter has been passed from the command line, 
    # this does not need to be within the config file
    for key,value in command_line.items():
        if key in expected_param and value is not None:
            expected_param.remove(key)

    mcmc_type_section = "MCMC.TYPE"    
    param = config.extract_params(config_file,
                                  expected_param=expected_param,
                                  ignore_sections=[mcmc_type_section])

    # Command line values added to param (or superceed inputs from the config 
    # file)
    for key,value in command_line.items():
        if key != 'model':
            if value is not None:
                param[key] = value
    
    # If configuration file does not include values for the
    # required parameters - produce an error
    for ep in expected_param:
        if not param[ep]:
            raise Exception(f"Required parameter '{ep}' has not been defined")

    if print_param:
        print("\nInput parameters: ")
        for key,value in param.items():
            print(f"{key} = {value}")

    return param    

def multi_gas_extract_param(config_file,default_config_file,**command_line):
    '''
    Extract parameters from the input configuration file for the 'multi gas model'.
    Checks the type of these inputs against those in the template config file.
    
    Args:
        config_file (str):  
            Config file name. Either uses the default or extracts this from the command line.
        default_config_file (str):  
            Config file name with default values and types.
        command_line:
            Any additional command line arguments to be added to the param 
            dictionary or to superceed values contained within the config file (have not tested this).
    Returns:
        function,collections.OrderedDict:
            MCMC function to use, dictionary of parameter names and values passed 
            to MCMC function
    '''
    
    param_type = config.generate_param_dict(default_config_file)
    
    all_params = []
    for k in param_type.keys():
        all_params += (list(param_type[k].keys()))
    
    for key,value in command_line_args.items():
        if key in param_type and value is not None:
            param_type.remove(key)
    
    param = config.extract_params(config_file)
    
    for key,value in command_line_args.items():
        if key != 'model':
            if value is not None:
                param[key] = value
            
    return param

def extract_multi_gas_model_type(config_file):
    '''
    Find value which describes the model setup to use.
    Checks the input configuation file the "mcmc_type" keyword.
    -----------------------------------
    Args:
      config_file (str):
        Configuration file name. Should be an .ini file.
      default (str):
        ***    
    Returns:
      Keyword for MCMC function to use (str)
    -----------------------------------
    '''
    mcmc_type_keyword = "mcmc_type"
    param_mcmc_type = config.extract_params(config_file)
    
    if param_mcmc_type is not None and mcmc_type_keyword in param_mcmc_type:
        mcmc_type = param_mcmc_type[mcmc_type_keyword]
    else:
        sys.exit('There is no mcmc_type in the .ini file.')
       
    return mcmc_type

if __name__=="__main__":
    openghginv_path = Paths.openghginv
    
    parser = argparse.ArgumentParser(description="Running Hierarchical Bayesian MCMC script")
    parser.add_argument("start", help="Start date string of the format YYYY-MM-DD",nargs="?")                  
    parser.add_argument("end", help="End date sting of the format YYYY-MM-DD",nargs="?")
    parser.add_argument("-c","--config",help="Name (including path) of configuration file")
    parser.add_argument("-r","--generate",action='store_true',help="Generate template config file and exit (does not run MCMC simulation)")
    parser.add_argument("-m","--model",help="Model type, either 'hbmcmc' or 'multi_gas'")

    args = parser.parse_args()
    
    command_line_args = {}
    if args.start:
        command_line_args["start_date"] = args.start
    if args.end:
        command_line_args["end_date"] = args.end
    if args.model:
        command_line_args["model"] = args.model
        if args.model == 'hbmcmc':
            default_config_file = os.path.join(openghginv_path,"hbmcmc/config/openghg_hbmcmc_input_template.ini")
            config_file = default_config_file
        elif args.model == 'multi_gas':
            default_config_file = os.path.join(openghginv_path,"multi_gas_model/config/model_inputs_template.ini")
            config_file = default_config_file
        else:
            sys.exit(f"Model name incorrectly specified, you must choose either -m 'hbmcmc' or -m 'multi_gas'")
            
    else:
        sys.exit(f"No model specified, you must choose either -m 'hbmcmc' or -m 'multi_gas'")
        
    config_file = args.config or config_file
    
    if args.generate == True:
        if args.model == 'hbmcmc':
            template_file = os.path.join(openghginv_path,"hbmcmc/config/openghg_hbmcmc_input_template.ini")
        elif args.model == 'multi_gas':
            template_file = os.path.join(openghginv_path,"multi_gas_model/config/model_inputs_template.ini")
        if os.path.exists(config_file):
            write = input(f"Config file {config_file} already exists.\nOverwrite? (y/n): ")
            if write.lower() == "y" or write.lower() == "yes":        
                copyfile(template_file,config_file)
            else:
                sys.exit(f"Previous configuration file has not been overwritten.")
        else:
            print(f'Creating new config file based on {template_file}')
            copyfile(template_file,config_file)
                
    if not os.path.exists(config_file):
        if config_file == default_config_file:
            sys.exit("No configuration file detected.\n"
                     "To generate a template configuration file run again"
                     " with -r flag:\n  $ python run_inverse_model.py -r")
        else:
            sys.exit("Configuration file cannot be found.\n"
                     f"Please check path and filename are correct: {config_file}")
    
    if args.model == 'hbmcmc':
        mcmc_type = extract_mcmc_type(config_file)
        mcmc_function = define_mcmc_function(mcmc_type)
        print(f"Using MCMC type: {mcmc_type} - function {mcmc_function.__name__}(...)")
        
        param = hbmcmc_extract_param(config_file,mcmc_type,**command_line_args)

        hbmcmc_output.copy_config_file(config_file,param=param,**command_line_args)

        mcmc_function(**param)
        
    elif args.model == 'multi_gas':
        model_type = extract_multi_gas_model_type(config_file)
        print(f"\nUsing multi gas model in {model_type} mode...")
        
        param = multi_gas_extract_param(config_file,default_config_file,**command_line_args)
        
        multi_gas_model.multi_gas_model(**param)
        