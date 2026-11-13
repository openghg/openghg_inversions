:orphan:

Boundary and offset identifiability investigations
=================================================

This planning note preserves historical investigations, illustrative matrix
examples and proposed follow-up checks. It does not record acceptance of a
scientific formulation for a network. In the examples, ``G`` denotes the
native dimensionless boundary-mass response (``M`` in the maintained
:doc:`explanation <../development/boundary_offset_identifiability>`).

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
