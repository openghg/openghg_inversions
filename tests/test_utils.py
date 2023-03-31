# ****************************************************************************
# Created: 1 Feb. 2023
# Author: Gareth Jones
# Contributor: Eric Saboya
# ****************************************************************************
# About
# Some very simple and possibly fragile tests for utils.py 
# ****************************************************************************
import os
import numpy as np 

from openghg_inversions import utils
from openghg_inversions.utils import load_json

def test_open_ds_site():
    """
    Reads CO2 concentrations from Mace Head and checks median
    value is between 350-550 ppm 
    """
    obspth = os.path.join('/group/chemistry/acrg/obs/MHD')
    mhd_data = utils.open_ds(os.path.join(obspth,'ICOS-picarro41_MHD_20100107_co2-20230113.nc'))
    co2_median = np.nanmedian(mhd_data.co2.data)
    assert co2_median<550 and co2_median>350

def test_open_ds_fp():
    """
    Reads a HighTRes NAME footprint for Mace Head
    to test "chunking" option works
    """
    fppth = os.path.join('/group/chemistry/acrg/LPDM/fp_NAME/EUROPE')
    mhd_data = utils.open_ds(os.path.join(fppth, 'MHD-10magl_UKV_co2_EUROPE_202112.nc'), chunks=50)
    assert mhd_data.fp_HiTRes.data.chunksize[0] == 50


# These are just some very simple and possibly fragile tests
# to make sure these files load correctly
def test_load_site_info():
    site_data = load_json(filename="site_info.json")
    for k in ["ADR", "ALT", "AMB", "AMS", "AMT", "AND", "ARH", "ASC"]:
        assert k in site_data

def test_load_species_info():
    species_info = load_json(filename="species_info.json")

    assert species_info["Desflurane"] == {
        "alt": [],
        "group": "Anaesthetics",
        "long_name": "Desflurane",
        "mol_mass": "168.03",
        "print_string": "Desflurane",
        "units": "ppt",
    }


def test_synonyms():
    """
    Tests synonyms for some species
    """
    species_info = utils.load_json(filename="species_info.json")
    expected = ["CO2","CH4","O2","CFC-12","CFC-11","SF6"]
    for k in ["co2","CO2","ch4","CH4","o2","cfc12","cfc-11","sf6"]:
        assert utils.synonyms(k,species_info) in expected






