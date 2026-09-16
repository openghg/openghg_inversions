Developing OpenGHG Inversions
=============================

These pages describe the programming conventions used for scientific model
development. They are normative for new RHIME work.

.. toctree::
   :maxdepth: 2

   rhime_model_development
   validation_and_xarray
   releasing

Scientific notation in prose
----------------------------

Use Sphinx's inline math role for a scientific quantity as it appears in an
equation, for example :math:`H`, :math:`x`, or :math:`H_{bc}b`. Use monospace
literals for exact Python, configuration, command-line, dimension, coordinate,
or data-variable names, for example ``run_rhime``, ``flux_sources``, and ``H``.
When a mathematical quantity and an implementation identifier share a name,
state the connection explicitly rather than using code formatting as a
substitute for mathematical notation.
