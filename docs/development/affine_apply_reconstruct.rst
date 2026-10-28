Prototype: apply and reconstruct
================================

This design experiment makes two operations visible on existing scientific
components. ``apply(state)`` supplies a concentration contribution used in the
likelihood. ``reconstruct(state)`` supplies the corresponding physical output.
Recipes still compose flux, boundary and offset contributions explicitly.
The prototype adds no model-selection or staged-workflow interface.

The two maps have different state and output spaces. Their shared operation
names explain the recipe; they do not require a common protocol or framework.
See :doc:`rhime_model_development` for the component and ownership principles,
and :doc:`/usage/affine_flux_map` for the retained-state reconstruction contract.

Run the :download:`offline example <../../examples/affine_apply_reconstruct.py>`
from the repository to compare both pairs of outputs on small synthetic arrays::

   python -m examples.affine_apply_reconstruct

The example checks coherent application against a separately reconstructed
native scaling field. It does not sample a posterior or read scientific data.

Centered boundary component
----------------------------

``CenteredBoundaryMap`` in ``rhime.co2.centered_boundary`` owns the factored
boundary expression after model-data registration. Let :math:`A` denote the
anomaly sensitivity, :math:`M` the dimensionless mass sensitivity, :math:`c`
the concentration reference, and :math:`b` the original boundary coefficients.
With :math:`q = c M \mathbf{1}`, its operations are

.. math::

   \operatorname{apply}(b) &= A b + c M (b - \mathbf{1}), \\
   \operatorname{reconstruct}(b) &= \operatorname{apply}(b) + q.

The likelihood uses observations :math:`y-q`. Reconstruction supplies absolute
boundary concentration. Subtracting a deterministic reference changes neither
the measurement-error nor model-error covariance. The prior and fixed/active
policy still describe the physical coefficients :math:`b`; activity is resolved
from the full absolute design :math:`A+cM`.

Inside ``add_centered_boundary``, the registered tensors make both operations
explicit:

.. code-block:: python

   from openghg_inversions.rhime.co2.centered_boundary import CenteredBoundaryMap

   boundary_map = CenteredBoundaryMap(
       anomaly=a,
       mass=m,
       reference=c,
       reference_shift=shift_data,
   )
   likelihood_contribution = boundary_map.apply(vector.state)
   absolute_boundary = boundary_map.reconstruct(vector.state)

Here ``a``, ``m``, ``c`` and ``shift_data`` are tensors registered from aligned,
materialized operands in the active model. The map keeps anomaly, mass and
reference separate so the small likelihood contribution is composed directly.
Computing an absolute concentration first and subtracting its reference later
would reintroduce the cancellation this representation avoids. Existing trace
annotation restores observed and predictive concentrations using the retained
reference shift.

Coherent flux component
------------------------

``CoherentForwardMap`` in ``rhime.co2.co2_affine_output`` pairs prepared
observation sensitivity and intercept with an existing
``BoundCo2AffineFluxMap``. The ``coherent_forward_map`` factory constructs the
pair from matching preparation and reconstruction inputs:

.. code-block:: python

   from openghg_inversions.rhime.co2.co2_affine_output import coherent_forward_map

   flux_map = coherent_forward_map(prepared, reconstruction)
   concentration = flux_map.apply(state)
   native_flux = flux_map.reconstruct(state)

``prepared`` is a ``Co2PreparedInputs`` value, ``reconstruction`` is its matched
``BoundCo2AffineFluxMap``, and ``state`` is a labelled retained state, optionally
with sample dimensions. These operations borrow their arrays and preserve
labelled, lazy execution.

For retained state :math:`\alpha`, effective observation sensitivity
:math:`H_\alpha` and prepared intercept :math:`d`, application evaluates

.. math::

   \operatorname{apply}(\alpha) = d + H_\alpha \alpha.

This is the coherent flux contribution in observation concentration units;
boundary and offset terms remain separate. The intercept is
:math:`d = Hm - H_\alpha\alpha_{\mathrm{ref}}`. Reconstruction evaluates the
retained-state-conditional signed native flux mean

.. math::

   \operatorname{reconstruct}(\alpha)
   = F\left[m + U_* (\alpha - \alpha_{\mathrm{ref}})\right],

where :math:`m` is the native scaling mean, :math:`F` the signed reference flux,
and :math:`U_*` the conditional prolongation. The bound reconstruction owns the
authoritative retained reference :math:`\alpha_{\mathrm{ref}}`. This state
reference and the boundary concentration reference :math:`c` have different
roles and units.

Reconstruction supplies a conditional mean for each retained-state draw.
Unresolved native covariance is not sampled or added to these flux draws.
The coherent likelihood retains its prepared unresolved observation covariance;
the paired operations do not change its uncertainty model.

Scope of the experiment
------------------------

The boundary map uses PyTensor ``TensorVariable`` values at the graph boundary.
Its labelled alignment and eager materialization remain separate, explicit
operations. The coherent map uses labelled xarray values and the existing flux
reconstruction. This prototype does not establish a backend-neutral map API.

The centered ``floatX=float64`` guard remains in place. The cached likelihood's
numerical projection and quadratic equations are unchanged; naming graph
operations does not replace their numerical lowering. Prepared and sampled
artifact formats retain their existing owners and contents.
