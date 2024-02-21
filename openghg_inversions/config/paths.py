# ****************************************************************************
# Created: 7 Nov. 2022
# Author: Eric Saboya, School of Geographical Sciences, University of Bristol
# Contact: ericsaboya@bristol.ac.uk
# ****************************************************************************
# About
# Orginally created by Matt Rigby (ACRG) and updated by Eric Saboya (Nov. 2022)
#
# Creates a class that stores file paths required for finding data and model
# output.
# To create a user-specific file, copy this to config/paths.yaml, and edit
#
# Feel free to add other paths, as required.
#
# Update by Ben Adam (ACRG), 20th February 2024:
#
# The move away from the ACRG repository means the only relevant path here is 
# the one pointing to the current directory, Paths.openghginv. I have kept 
# everything else commented out, for reference
# ****************************************************************************

from pathlib import Path

_openghginv_path = Path(__file__).parents[2]
# _openghginv_config_path = Path(__file__).parents[0]

# _user_defined_data_paths = sorted(_openghginv_config_path.glob("paths.y*ml"))

# if len(_user_defined_data_paths) == 0:
#     _data_paths_file = _openghginv_config_path / "templates/paths_default.yaml"
# else:
#     _data_paths_file = _user_defined_data_paths[0]

# with open(_data_paths_file, 'r') as f:
#     _data_paths = yaml.load(f, Loader = yaml.SafeLoader)


class Paths:
    """
   Object that used to be used to store paths to obs, ACRG and LPDM directories.
   However, with the move over to OpenGHG this is generally all deprecated
   Currently, the only path is to the current openghg_inversions directory

    All paths are pathlib.Path objects (Python >3.4)

    Paths.openghginv: path to openghg_inversions repo

    [Formerly]
    paths.acrg: path to ACRG repo
    paths.obs: path to obs folder
    path.lpdm: path to LPDM data directory
    """

    openghginv = _openghginv_path


#    obs = Path(_data_paths["obs_folder"])
#    lpdm = Path(_data_paths["lpdm_folder"])
#    if "data_folder" in _data_paths:
#        data = Path(_data_paths["data_folder"])
