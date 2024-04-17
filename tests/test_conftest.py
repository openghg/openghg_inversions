from openghg.retrieve import search


def test_default_session_fixture(using_zarr_store):
    """The default session fixture should mock `read_local_config`
    so that the object store path is set to:

    <temp dir>/openghg_inversions_testing_store
    """
    from openghg.objectstore._local_store import read_local_config

    conf = read_local_config()

    assert conf
    assert "inversions_tests" in conf["object_store"]

    if using_zarr_store:
        assert "openghg_inversions_zarr_testing_store" in conf["object_store"]["inversions_tests"]["path"]
    else:
        assert "openghg_inversions_testing_store" in conf["object_store"]["inversions_tests"]["path"]


def test_obs_in_test_store():
    results = search(site="tac", species="ch4", data_type="surface", store="inversions_tests")
    assert results


def test_footprints_in_test_store():
    results = search(site="tac", data_type="footprints")
    assert results


def test_bc_in_test_store():
    results = search(species="ch4", data_type="boundary_conditions")
    assert results


def test_flux_in_test_store(using_zarr_store):
    data_type = "flux" if using_zarr_store else "emissions"
    results = search(species="ch4", data_type=data_type)
    assert results
