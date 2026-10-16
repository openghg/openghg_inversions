Choose a RHIME model recipe
===========================

Choose a recipe by the state and observation relationships required by the
scientific question. Do not infer the recipe from ``species`` alone: the
standard recipe can be scientifically appropriate for more than one gas, while
the linked CO₂/O₂ recipe has a specific shared-state and tracer structure.

.. list-table::
   :header-rows: 1
   :widths: 18 31 27 24

   * - Recipe
     - Choose it when
     - Current starting point
     - Continue with
   * - Standard
     - One flux component is scaled by one labelled state.
     - Complete acquisition-to-output runner and staged workflow.
     - :doc:`standard tutorial <rhime_standard_tutorial>`
   * - Multisector
     - Several named flux sectors need separately inferred states and a shared
       observation likelihood.
     - Complete acquisition-to-output runner and staged workflow.
     - :doc:`multisector tutorial <rhime_multisector_tutorial>`
   * - CO₂-only
     - A correlated positive state and coherent-reduction covariance are
       supplied for one CO₂ observation channel.
     - Advanced prepared-input replay; acquisition and complete configuration
       are not part of this entry point.
     - :doc:`CO₂ model family <co2_model_family>`
   * - Linked CO₂/O₂
     - CO₂ and O₂ channels constrain shared and tracer-specific retained
       states through one joint covariance.
     - Advanced prepared-input replay; there is no complete
       ``run_rhime_co2_o2`` workflow or standalone O₂ recipe.
     - :doc:`CO₂ model family <co2_model_family>`

The :doc:`standard model family <standard_model_family>` is the main route for
new users. “Advanced” on the CO₂ family describes the preparation knowledge and
workflow complexity it requires; it does not by itself state API stability,
configuration availability, operational adoption, or scientific validation.
