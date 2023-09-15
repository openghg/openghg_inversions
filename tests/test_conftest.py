from openghg.objectstore._local_store import read_local_config
from openghg.retrieve import search


def test_default_session_fixture():
    """The default session fixture should mock `read_local_config`
    so that the object store path is set to:

    <location of inversions>/openghg_inversions/tests/data/test_store
    """
    conf = read_local_config()

    assert conf
    assert 'inversions_tests' in conf['object_store'].keys()
    assert 'openghg_inversions/tests/data/test_store' in conf['object_store']['inversions_tests']['path']


def test_obs_in_test_store():
    results = search(site='tac', species='ch4', data_type='surface')
    assert results


def test_footprints_in_test_store():
    results = search(site='tac', data_type='footprints')
    assert results


def test_bc_in_test_store():
    results = search(species='ch4', data_type='boundary_conditions')
    assert results


def test_flux_in_test_store():
    results = search(species='ch4', data_type='emissions')
    assert results
