#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 09:36:00 2023

@author: cv18710, based on code in the ACRG repo

Multiple gas MCMC inversion with customisable inputs, mcmc type and output types.

Run from the command line using: python run_multi_gas_model.py 'start_date' 'end_date' -c /path/to/.ini file
"""

import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from config import config
import argparse
from openghg_inversions.multi_gas_model import multi_gas_model

from acrg.config.paths import Paths as paths
data_path = paths.data

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Running MCMC inversion for multiple gases')
    parser.add_argument("start_date", help="Start date with format YYYY-MM-DD",nargs="?")                  
    parser.add_argument("end_date", help="End date with format YYYY-MM-DD",nargs="?")
    parser.add_argument("-c","--config",help='Name (including path) of configuration file')
    
    args = parser.parse_args()
    
    config_file = args.config or args.config_file
    command_line_args = {}
    if args.start_date:
        command_line_args["start_date"] = args.start_date
    if args.end_date:
        command_line_args["end_date"] = args.end_date
    
    param_type = config.generate_param_dict('/user/home/cv18710/code/multiple_gas_inverse_model/model_inputs_template.ini')
    
    all_params = []
    for k in param_type.keys():
        all_params += (list(param_type[k].keys()))
    
    for key,value in command_line_args.items():
        if key in param_type and value is not None:
            param_type.remove(key)
    
    param = config.extract_params(config_file)
    
    for key,value in command_line_args.items():
        if value is not None:
            param[key] = value
    
    #param = config.all_param(config_file,param_type=param_type)

    multi_gas_model.multi_gas_model(**param)