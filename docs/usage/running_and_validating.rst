Running and validating inversions
=================================

Use the staged workflow to separate preparation, prior prediction, sampling,
diagnosis, and postprocessing into inspectable artifacts. It currently supports
standard, multisector and CO₂-only model recipes. The CO₂-only route accepts
a saved coherent prepared-input artifact and supports ordinary and cached
fixed-OU execution; see :doc:`co2_model_family` for its commands and output
boundary. Linked CO₂/O₂ staging remains follow-up work.

Scientific validation is broader than a successful software run. Interpret
convergence diagnostics and predictive checks using the
:ref:`standard tutorial diagnostic workflow <standard-rhime-diagnostics>` and
the machine-readable checks in :doc:`staged_workflow`. Interpret them in the
context of the assumptions and limitations described in :doc:`how an
atmospheric inversion works <conceptual_inversion>`. Sensitivity tests and
independent comparisons remain essential scientific work, but the user guide
does not yet provide a dedicated end-to-end procedure for them.

.. toctree::
   :maxdepth: 1

   staged_workflow
