.. _installation-and-setup:

Install and set up OpenGHG Inversions
=====================================

Choose the setup that matches your task:

* To run inversions with a released version, install the package in a virtual
  environment.
* To contribute to OpenGHG Inversions, use the repository's Pixi environment.

OpenGHG Inversions supports Python 3.10 or later on Linux and macOS. Inversions
that acquire data through OpenGHG also need access to a configured OpenGHG
object store; see the `OpenGHG project documentation
<https://docs.openghg.org/>`_ for that separate setup.

Install the released package
----------------------------

Create and activate a virtual environment, then install the package from PyPI:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install openghg-inversions

Confirm that the installed package imports and report its version:

.. code-block:: bash

   python -c "import importlib.metadata; print(importlib.metadata.version('openghg-inversions'))"

Start with :doc:`conceptual_inversion`, then use the
:doc:`model recipe chooser <model_recipes>` to select a runnable workflow.

Contributor setup with Pixi
---------------------------

The repository's recommended contributor environment uses `Pixi
<https://pixi.prefix.dev/latest/installation/>`_. Pixi keeps the compiled
NetCDF/HDF5 stack from conda-forge consistent across OpenGHG, xarray,
``h5netcdf``, ``h5py``, and ``netcdf4``.

Clone the repository, install the development environment, and verify its core
imports:

.. code-block:: bash

   git clone https://github.com/openghg/openghg_inversions.git
   cd openghg_inversions
   pixi install -e dev
   pixi run -e dev python -c "import openghg_inversions, h5py, h5netcdf, netCDF4"

The main contributor checks are:

.. code-block:: bash

   pixi run -e dev test
   pixi run -e dev lint
   pixi run -e dev typecheck

Build and preview the documentation with:

.. code-block:: bash

   pixi run -e dev docs-preview

The preview is served at ``http://127.0.0.1:8765/`` and opens in Safari on
macOS. Use ``--no-open`` to keep it in the terminal or ``--fresh`` to discard
cached Sphinx doctrees:

.. code-block:: bash

   pixi run -e dev docs-preview --no-open
   pixi run -e dev docs-preview --fresh

The `repository README
<https://github.com/openghg/openghg_inversions#installation>`_ documents
additional contributor tasks, including the optional country-file smoke test
and the larger ``uv_dev`` environment.

Use a local OpenGHG checkout
----------------------------

To test repository code against a local OpenGHG checkout, replace only the
OpenGHG package code inside the Pixi environment:

.. code-block:: bash

   pixi run -e dev python -m pip install --no-deps -e /path/to/openghg

The ``--no-deps`` flag prevents ``pip`` from replacing Pixi's conda-forge
NetCDF/HDF5 libraries with unrelated wheels.

Troubleshoot compiled dependencies
----------------------------------

If importing ``h5py``, ``h5netcdf``, or ``netCDF4`` reports an HDF5 library
error, recreate the Pixi development environment before changing individual
packages. Avoid upgrading those compiled packages separately with ``pip``
inside the Pixi environment, because doing so can mix incompatible binary
libraries.
