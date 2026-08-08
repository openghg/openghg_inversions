Installation and Setup
======================

As OpenGHG Inversions is dependent on OpenGHG, please ensure that when
running locally you are using Python 3.12 or later on Linux or MacOS.
Please see the `OpenGHG project <https://github.com/openghg/openghg/>`__
for further installation instructions of OpenGHG and setting up an
object store.

Recommended development install using Pixi
------------------------------------------

OpenGHG Inversions and OpenGHG read and write NetCDF/HDF5 data through
``xarray``, ``h5netcdf``, ``h5py``, and ``netcdf4``. Plain ``pip`` or
``uv`` installs can select PyPI wheels with incompatible bundled HDF5
libraries. The Pixi workspace in this repository installs the compiled
HDF5/NetCDF stack from conda-forge and installs ``openghg_inversions`` in
editable mode.

Install `Pixi <https://pixi.prefix.dev/latest/installation/>`__, then run:

.. code:: bash

   git clone https://github.com/openghg/openghg_inversions.git
   cd openghg_inversions
   pixi install -e dev
   pixi run -e dev python -c "import openghg_inversions, h5py, h5netcdf, netCDF4"

The ``dev`` environment includes the test, lint, type-checking, and tox
tools used by the repository:

.. code:: bash

   pixi run -e dev test
   pixi run -e dev lint
   pixi run -e dev typecheck
   pixi run -e dev tox

To run the optional real country-file HDF5 smoke check on a machine that
can access the ACRG country files, set the country directory and run the
Pixi task:

.. code:: bash

   OPENGHG_COUNTRY_FILE_SMOKE_DIR=/group/chem/acrg/LPDM/countries pixi run -e dev country-file-smoke

The smoke check opens ``country_EUROPE_EEZ_PARIS_gapfilled.nc`` and
``country_EUROPE.nc`` with xarray's default backend, ``h5netcdf``, and
``netcdf4``, then exercises
``openghg_inversions._country_file.load_country_dataset``. It prints the
``xarray``, ``h5netcdf``, ``h5py``, and ``netCDF4`` versions and the
per-engine result. Without ``OPENGHG_COUNTRY_FILE_SMOKE_DIR``, the
real-file tests are skipped so uv/pip CI does not need access to cluster
data.

If you need to test against a local OpenGHG checkout, install only the
local package code into the Pixi environment. The ``--no-deps`` flag is
important because it prevents ``pip`` from replacing Pixi's conda-forge
HDF5/NetCDF packages with wheels.

.. code:: bash

   pixi run -e dev python -m pip install --no-deps -e ~/Documents/openghg

Avoid running commands such as ``pip install -U h5py h5netcdf netcdf4``
inside the Pixi environment. That can reintroduce the binary mismatch
that Pixi is intended to avoid.

Setup a virtual environment
---------------------------

Check that you have Python 3.12 or greater:

.. code:: bash

   python --version

(Note for Bristol ACRG group: If you are on Blue Pebble, the default
anaconda module ``lang/python/anaconda`` is Python 3.9. Use
``module avail`` to list other options;
``lang/python/miniconda/3.12.2.inc-perl-5.30.0`` will work.)

Make a virtual environment

.. code:: bash

   python -m venv openghg_inv

Next activate the environment

.. code:: bash

   source openghg_inv/bin/activate

Installation using ``pip``
--------------------------

First you’ll need to clone the repository

.. code:: bash

   git clone https://github.com/openghg/openghg_inversions.git

Next make sure ``pip`` and related install tools are up to date and then
install OpenGHG Inversions using the editable install flag (``-e``)

.. code:: bash

   pip install --upgrade pip setuptools wheel
   pip install -e openghg_inversions

Optionally, install the developer requirements (there is more
information about this in the “Contributing” section below):

.. code:: bash

   pip install -r requirements-dev.txt

Verify that PyMC is using fast linear algebra libraries
-------------------------------------------------------

At this point, run

.. code:: bash

   python -c "import pymc"

This should run without printing any messages. If you receive a message
about ``pymc`` or ``pytensor`` using the ``numpy`` C-API, then your
inversions might run slowly because the fast linear algebra libraries
used by ``numpy`` haven’t been found.

Solutions to this are:

1. Use the Pixi development environment above, which installs ``numpy``
   and the NetCDF/HDF5 stack from conda-forge.
2. Try ``python -m pip install numpy`` after upgrading
   ``pip, setuptools, wheel``.
3. Create a ``conda`` env, install ``numpy`` using ``conda``, then use
   ``pip`` to upgrade ``pip, setuptools, wheel`` and install
   ``openghg_inversions``.
