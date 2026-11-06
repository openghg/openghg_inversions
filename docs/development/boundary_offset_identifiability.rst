Boundary corrections and observation offsets
============================================

This note explains structural overlap between additive BC corrections and
observation offsets. It concerns their likelihood directions; native-basis
covariance design is separate work.

Known overlap for normalized NESW transport
-------------------------------------------

Let ``M`` be the native dimensionless boundary-mass response, ``w`` its normalized
boundary-state centering weights, and ``Q`` a basis for the state anomaly
constraint ``w.T @ u = 0``. The sampled boundary correction contributes

.. math::

   g\delta+MQ\alpha,\qquad g=M\mathbf1.

Here ``delta`` is one boundary-product correction per channel over the full
inversion. For complete, jointly normalized transport with the default NESW
curtain basis and binary temporal assignment, ``g = ones``. This is a known
property of that preparation, including its observation averaging.

A global observation offset has this same direction. Unrestricted site or
site-period indicators also contain it: summing their columns gives ``ones``.
The likelihood therefore identifies a combined constant, even though boundary
product bias and persistent observation bias have distinct scientific roles.
Proper priors can produce a proper posterior without making that attribution
identifiable from the observations.

Current offsets default to ``per_site=True, drop_first=False``. Explicit
``drop_first=True`` fixes the first encountered site's offsets to zero,
including every period when a frequency is selected. Global offsets reject
frequency and drop-first settings. Centered site contrasts are not currently
an offset option.

What anchored or centered contrasts achieve
-------------------------------------------

For two sites, an anchored design keeps only the second site's indicator.
A centered design can use ``z = (+1 at site A, -1 at site B)``. More generally,
site contrasts satisfy a declared positive site-weight constraint. These site
weights are distinct from the boundary-state weights ``w``.

Either construction excludes the unrestricted constant site-offset direction.
With ``g = ones``, the boundary mean and site contrasts alone are linearly
independent. Boundary anomalies may nevertheless reproduce site contrasts.
Thus these constructions remove one structural redundancy; they do not
establish identifiability of the joint boundary, offset and flux model.

Anchoring or centering can define relative site effects, with their common
component assigned to :math:`\delta`. This identifying convention does not
establish that the anchor site's physical bias, or the network's mean bias,
is zero. If :math:`\delta` is interpreted solely as boundary-product bias,
the convention becomes an explicit attribution assumption. Neither convention
supplies external calibration evidence.

Changing the offset constraint changes the prior
------------------------------------------------

An anchored two-site representation gives site constants
``(delta, delta+beta_B)``. Re-expressing those same predictions with equal-site
centered contrasts gives ``delta_new = delta + beta_B/2`` and
``eta = (-beta_B/2, beta_B/2)``. If the original ``delta`` and ``beta_B`` are
independent with unit variance, ``var(delta_new)=1.25`` and
``cov(delta_new, eta)=(-0.25, 0.25)``. Assigning fresh independent priors after
recoding changes the joint prior. The redefined common intercept preserves
predictions, but is not automatically the same physical boundary correction.
A proposed centered-offset model must state
that scientific choice explicitly or transform the original joint prior.

Runtime warning
---------------

The model warns once per channel when an additive boundary mean with
``g = ones`` is combined with an unrestricted global, site or site-period
offset constant. The warning explains that the likelihood identifies only the
combined constant; attribution requires priors or external information.
The combination remains permitted, and the warning makes no claim that
contrasts remove every boundary confounding. It inspects ``g`` only at the
existing joint model-materialization boundary, after selected inputs are eager.

Historical investigations, illustrative matrix examples and proposed empirical
checks are preserved in :doc:`the investigation note
<../plans/boundary_offset_investigations>`.
