Running and validating inversions
=================================

Use the staged workflow to separate preparation, prior prediction, sampling,
diagnosis, and postprocessing into inspectable artifacts. It currently supports
the standard and multisector model recipes; the CO₂ model family does not have
a staged CLI route.

Scientific validation is broader than a successful software run. Interpret
convergence diagnostics, predictive checks, sensitivity tests, and independent
comparisons in the context of the assumptions described in :doc:`how an
atmospheric inversion works <conceptual_inversion>`.

.. toctree::
   :maxdepth: 1

   staged_workflow
