Installation and Setup
======================

As OpenGHG Inversions is dependent on OpenGHG, please ensure that when
running locally you are using Python 3.10 or later on Linux or MacOS.
Please see the `OpenGHG project <https://github.com/openghg/openghg/>`__
for further installation instructions of OpenGHG and setting up an
object store.

Setup a virtual environment
---------------------------

Check that you have Python 3.10 or greater:

.. code:: bash

   python --version

(Note for Bristol ACRG group: If you are on Blue Pebble, the default
anaconda module ``lang/python/anaconda`` is Python 3.9. Use
``module avail`` to list other options;
``lang/python/miniconda/3.10.10.cuda-12`` or
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

Solutions to this are: 1. try ``python -m pip install numpy`` after
upgrading ``pip, setuptools, wheel`` 2. create a ``conda`` env, install
``numpy`` using ``conda``, then use ``pip`` to upgrade
``pip, setuptools, wheel`` and install ``openghg_inversions``
