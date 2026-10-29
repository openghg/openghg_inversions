Boundary corrections and observation offsets
============================================

This note records a focused identifiability investigation for additive BC
corrections. It concerns the likelihood directions of boundary and offset
terms; native-basis covariance design is separate work.

Known overlap for normalized NESW transport
-------------------------------------------

Let ``G`` be the dimensionless additive BC response, ``w`` its normalized
boundary-state centering weights, and ``Q`` a basis for the state anomaly
constraint ``w.T @ u = 0``. The sampled boundary correction contributes

.. math::

   g\delta+GQ\alpha,\qquad g=G\mathbf1.

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

Earlier investigations
----------------------

The Linear records contain related controlled experiments. These are
historical results, not reruns with the currently normalized operators:

* `OPE-190 <https://linear.app/openghg-inversions/issue/OPE-190>`_ deferred a
  common observation offset after finding strong BC/outer-state confounding.
* `OPE-191 <https://linear.app/openghg-inversions/issue/OPE-191>`_ and
  `OPE-192 <https://linear.app/openghg-inversions/issue/OPE-192>`_ considered
  one scalar applied to every non-BIR site, with zero at BIR. This differs
  from independent offsets for every non-reference site.
* `OPE-195 <https://linear.app/openghg-inversions/issue/OPE-195>`_ screened the
  additive mean/anomaly BC formulation without selecting a reference site.
  Full design rank did not establish precise recovery of its common mean.
* `OPE-206 <https://linear.app/openghg-inversions/issue/OPE-206>`_ compared
  scalar offsets and centered BC corrections. Centered BCs absorbed injected
  observation offsets as well as boundary perturbations; no default changed.
* `OPE-225 <https://linear.app/openghg-inversions/issue/OPE-225>`_ separately
  proposes independent site offsets with a reference and explicitly retains
  BC/flux separation as an acceptance question.

These records motivate testing attribution of known errors, rather than
accepting a formulation from concentration fit or sampler convergence alone.

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

Two normalized, full-rank NESW examples
---------------------------------------

Use six observations ordered ``A,A,A,B,B,B`` and four BC states ordered
``N,E,S,W``. Set ``w = (1,1,1,1)/4`` and let ``Q`` be any full-rank basis for
``sum(u)=0``. The current Helmert construction supplies such a basis.

The first response has independently estimable baseline coefficients after
removing the unrestricted site-offset constant:

.. code-block:: python

   G_identifiable = [
       [1,    0,    0,    0],
       [0,    1,    0,    0],
       [0,    0,    1,    0],
       [0,    0,    0,    1],
       [0.25, 0.25, 0.25, 0.25],
       [0.5,  0,    0,    0.5],
   ]

The second response confounds the site contrast with a BC anomaly:

.. code-block:: python

   G_confounded = [
       [1,   0,   0,   0],
       [0,   1,   0,   0],
       [0.5, 0.5, 0,   0],
       [0,   0,   1,   0],
       [0,   0,   0,   1],
       [0,   0,   0.5, 0.5],
   ]

Every row sums to one and both matrices have rank four. For the second,
``v = (1,1,-1,-1)`` has zero boundary-state mean and ``G @ v = z`` exactly.
Changing this anomaly and the centered site offset in opposite directions
leaves every prediction unchanged. The anchored indicator is
``site_B = (ones-z)/2``, so anchoring remains confounded too.

.. list-table:: Joint ranks for ``[G @ ones, G @ Q, offset_design]``
   :header-rows: 1

   * - Offset design
     - Columns
     - First response: rank
     - Second response: rank
   * - Both site indicators
     - 6
     - 5
     - 4
   * - Site B indicator only
     - 5
     - 5
     - 4
   * - Centered contrast ``z``
     - 5
     - 5
     - 4

These ranks assume other model contributions are known. Additional flux terms
can introduce further overlap. Missing site-period combinations can also
produce zero offset columns; a cross-component diagnostic must distinguish
that internal rank loss from overlap between components.

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

Runtime warning and a narrow follow-up
--------------------------------------

Warn once per channel when an additive boundary mean with ``g = ones`` is
combined with an unrestricted global, site or site-period offset constant.
Explain that the likelihood identifies only the combined constant; attribution
requires priors or external information. Do not reject the scientifically
possible combination or claim contrasts remove every boundary confounding.
Use the normalized-NESW contract where known; otherwise inspect ``g`` only at
the existing joint model-materialization boundary. Do not compute or densify
borrowed lazy inputs during configuration or preparation to issue a warning.

A bounded empirical investigation should:

1. Reproduce the two matrices' ranks and predictions under exact null moves.
2. Compare anchored and centered designs with transformed priors first, then
   assess any deliberately changed scientific prior separately.
3. Inspect actual site/period transport from one representative prepared run,
   including missing combinations and within-site changes in inflow direction.
4. Run a small synthetic recovery experiment with known boundary and site
   biases, recording attribution sensitivity to the chosen priors and flux
   terms. Positive-definite observation covariance does not remove exact rank
   loss, though it affects precision and near-confounding.

General rank diagnostics remain investigation work, rather than a new runtime
feature. No conclusion here establishes scientific acceptance for a network.
