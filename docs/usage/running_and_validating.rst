Running and validating inversions
=================================

Use the staged workflow to separate preparation, prior prediction, sampling,
diagnosis, and postprocessing into inspectable artifacts. It currently supports
the standard and multisector model recipes; the CO₂ model family does not have
a staged CLI route.

Scientific validation is broader than a successful software run. Interpret
convergence diagnostics and predictive checks using the
:ref:`standard tutorial diagnostic workflow <standard-rhime-diagnostics>` and
the machine-readable checks in :doc:`staged_workflow`. Before posterior
sampling, follow the :doc:`prior_predictive_checking` tutorial to inspect what
the configured model implies on the concentration scale. Interpret all of
these checks in the context of the assumptions and limitations described in
:doc:`how an atmospheric inversion works <conceptual_inversion>`. Sensitivity
tests and independent comparisons remain essential scientific work, but the
user guide does not yet provide a dedicated end-to-end procedure for them.

.. toctree::
   :maxdepth: 1

   staged_workflow
   prior_predictive_checking
