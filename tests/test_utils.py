from openghg_inversions.utils import load_json

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
