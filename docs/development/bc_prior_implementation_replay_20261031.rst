Boundary correction implementation replay: October 2026
=======================================================

Purpose and dependency choice
-----------------------------

This is an implementation comparison and validation record for the independent
level/A-pattern/curtain-contrast API described in
:doc:`../usage/boundary_pattern_priors`. It does not select production BC
uncertainties or establish physical flux recovery.

The feature worktree is ``.codex-worktrees/bc-pattern-priors-20261031`` on
``codex/bc-pattern-priors``. Remote branches and GitHub PRs were checked before
choosing the base on 31 October 2026:

* devel: ``80782dd4``; numerical centering is already merged.
* `PR 745 <https://github.com/openghg/openghg_inversions/pull/745>`_:
  ``6726c6dc``; independent concentration-unit additive priors, still open.
* `PR 989 <https://github.com/openghg/openghg_inversions/pull/989>`_:
  ``df09dd52``; concentration-term accounting, based on 745, still open.
* `PR 991 <https://github.com/openghg/openghg_inversions/pull/991>`_:
  ``22b2057c``; shared species-neutral components, based on 989, still open.

The new implementation starts at current PR 991. Its merge ``7d2d83c4``
incorporates the later level-only and additive fixes from 745/989. The frozen
experiment revision ``c913da1a`` is an ancestor, retained as a historical
reference rather than used as the development base. This feature therefore
depends on the full open stack; its implementation is not a recommendation to
merge the parent priors or adopt the illustrative widths.

Input, model and comparison definitions
---------------------------------------

The July 2021 WUR BASE preparation has 5,314 retained site-hours at 25 sites,
240 adjustable inner flux states and 24 adjustable outer states. Its SHA256 is
``056c8ad508ebbb02fdaf234026dea2a789cd7df63a061945192638f4408e9ad6``.
The authoritative source and experiment manifests are under
``/group/chem/acrg/verification_games_round_2/verification-games-work``:

* ``ope225-wur-base-bc-offset-20261029-r2/prepared.nc``;
* ``ope225-wur-bc-patterns-20261030-r1/preparation.json``;
* ``ope225-wur-bc-frequency-20261030-r1/preparation.json``.

The comparison quantities in the deterministic tests are the saved correction
operators, prescribed physical BC, centered concentrations, residuals and
normalized Gaussian likelihoods. There is no truth-based score in these
checks. In later sampling comparisons, the reference is the matched frozen
posterior for the same arm. Flux functionals are conditional retained-state
inner-PARIS totals, using the original scorer's coherent flux-functional
weights; concentration functionals are retained-row means in ppm. Neither is
a new assessment against WUR flux truth.

Prescribed physical BC is :math:`b_0=(A+cM)\mathbf1`. The fitted physical BC is
:math:`b_0+d`. The complete baseline in these experiments is physical BC plus
its correction, adjustable outer flux and ``fixed_full_minus_fixed_inner``.
The full model still contains ``H_all @ x + fixed_full + b0 + d``. Outer flux
and the coherent fixed contribution remain separately accounted for.

The prior/reference flux is UOB BASE; the observations are the historical
full WUR BASE concentration product. The monthly arms retain the original
complete correlated arithmetic-lognormal
flux prior. The 8D arms retain their frozen shrunk-lognormal biosphere
covariance, checked by its manifest hash and state labels. This report does
not compare a frequency fit against an original-prior monthly fit; scientific
frequency comparisons require the matched shrunk-lognormal monthly control.

The likelihood retains reported observation errors, full-model rank-512
aggregation covariance plus diagonal residual variance, and independent
within-site OU blocks. The Gaussian aggregation term is the inherited moment
approximation to unresolved flux. OU timescale is fixed at 5 hours; independent
site amplitudes have HalfNormal(0.75 ppm) priors. There are no site offsets.
The scalar BC reference is 409.8810652476134 ppm in all five arms. Level width
is 5 ppm; A coefficient width is 0.8656536988131538; Helmert width is
1.5207787909022383 ppm. The latter two give pattern-only median pointwise SD
1 ppm. These are experimental widths, not calibrated production defaults.

Deterministic results
---------------------

The adapter verifies prepared/design hashes, retains the frozen files, and
writes new labelled correction artifacts. The full 5,314-row correction
matrices match their frozen counterparts exactly in every arm. This establishes
the same zero prior means and observation-space Gaussian covariance after
mapping standardized experiment coordinates to physical coefficients. Exact
functional SDs use :math:`\|l^TF\|`, without an observation-square covariance.

The table gives concentration-space prior diagnostics in ppm. A mean's SD is
uncertainty in the retained-row whole-window mean, not a mean pointwise SD.
Only the pattern component enters the last column; differences use the 4,833
same-site pairs exactly 24 hours apart.

.. list-table:: Frozen-design replay and separate prior components
   :header-rows: 1

   * - Arm
     - Coefficients
     - Level mean SD
     - Pattern mean SD
     - Median pattern 24h difference SD
   * - Level only
     - 1
     - 5.000000
     - 0
     - 0
   * - Level + A
     - 5
     - 5.000000
     - 0.332509
     - 0.619647
   * - Level + Helmert
     - 4
     - 5.000000
     - 0.903685
     - 0.126502
   * - Global level + 8D A
     - 17
     - 5.000000
     - 0.531839
     - 0.684168
   * - Period levels + 8D A
     - 20
     - 2.504798
     - 0.531839
     - 0.684168

The scalar reference and coefficient widths remain fixed across frequencies.
Equal pointwise variance therefore coexists with different temporal covariance
and whole-window uncertainty. Period levels also permit jumps: with unit mass
response, a difference crossing two independent periods has level SD
:math:`5\sqrt2` ppm, although the median across all 24-hour pairs is zero.

Model preflights select three rows per site and coefficient period, evaluate
nonzero physical coefficients, and compare ordinary observed log density with
the cached quadratic. Initial five-arm checks have maximum absolute mean
difference :math:`3.6\times10^{-15}` ppm and maximum absolute log-density
difference :math:`1.1\times10^{-12}`. The final adapter additionally evaluates
the saved experimental design directly at the corresponding standardized
point and compares its residual and ordinary Gaussian density.

Focused equation and replay tests cover general :math:`g\ne1`, all K A
coefficients, iid-curtain/Helmert covariance equivalence at the required widths,
unbounded signed A corrections, global/period level support, strict labels,
scalar-reference invariance under frequency changes, reference-dependent A
covariance, and float64 O2 corrections around 209,300 ppm. Both ordinary and
cached single/linked models agree in densities and gradients at nonzero
coefficients. Trace roundtrips preserve physical units/support and reconstruct
prescribed, level, pattern and total BC contributions. Predictive reference
restoration is idempotent. Dask checks retain shared graphs and compute related
flux/correction payloads once at the model boundary.

Local validation records 40 focused tests and 205 broader existing
boundary, linked, replay and cached-likelihood tests passing on locked Python
3.13. Changed-path Ruff and whitespace checks pass. The offline example runs.

Cluster array 19861094 tested implementation revision ``fc75da5d``:

* Locked Python 3.12: 5,097 passed, 8 skipped, 4 deselected (43m40s job).
* Locked Python 3.13: 5,097 passed, 8 skipped, 4 deselected (36m09s job).
* Python 3.12 with OpenGHG devel: no failures before the one-hour time limit;
  interrupted at 36% of the suite. This is not a passing full-suite result.
* ``docs-full``, repository Ruff and ``borrowed-types`` completed successfully.
  Rendered usage, API landing and module pages were inspected. New heading
  and inline-literal warnings found in the build were corrected; regenerated
  API pages are included. A repeat build checks these corrections.
* Full typing failed on both candidate and parent PR 991: 142 versus 140
  diagnostics. The two additions were the new module's untyped PyMC import
  (the same missing third-party typing as existing modules) and an xarray
  square-root return annotation. The latter was corrected using an
  xarray-preserving expression; the repeat check compares against the same
  parent, without suppressing the inherited failures.

All three full-suite allocations reached approximately 8 GiB resident memory.
The development-dependency retry uses 12 GiB and a longer wall-time allowance;
it does not change dependencies or scientific inputs. The focused sampling
job 19861096 remains pinned to ``fc75da5d``. Subsequent documentation and
calibration-expression fixes do not change its manifest-specified widths or
model operators.

Reproduction and cluster validation
-----------------------------------

Use a fresh output root and the implementation worktree's locked environment:

.. code-block:: bash

   PYTENSOR_FLAGS=floatX=float64 python scripts/replay_bc_priors.py \
     --root inversion_outputs/bc-prior-replay-20261031-r3 --prepare
   PYTENSOR_FLAGS=floatX=float64 python scripts/replay_bc_priors.py \
     --root inversion_outputs/bc-prior-replay-20261031-r3 --check-only

The adapter records the actual revision, tracked diff hash, script/lock hashes,
Python executable and numerical dependency versions. The initial deterministic
receipts are in ``inversion_outputs/bc-prior-replay-20261031-r2``. Final pinned
receipts and sampling output use the fresh ``r3`` root. The frozen scripts and
their provenance assertions are unchanged. The standalone locked environment
uses PyMC 6.3.2, PyTensor 3.3.2, NumPy 2.2.6, SciPy 1.15.3 and xarray 2025.9.1.

The planned focused sampling replay is monthly ``delta_A`` against its own
frozen original-prior fit: four chains, 1,000 tuning and 1,000 retained draws,
seed 20260924, state target acceptance 0.95 and OU-amplitude acceptance 0.9.
Run it through ``scripts/slurm_bc_prior_replay.sh`` with slurm-wakeup. Sampling
is rejected on a login node. The comparison reports sampling health, separate
level/pattern/outer and complete-baseline concentration means, and inner flux
functionals with both runs' Monte Carlo standard errors. A shared seed does
not require paired or bit-for-bit draws. No sampling result is claimed yet.

Scientific limitations
----------------------

The frozen experiments retain historical endpoint defects. This matched replay
tests implementation equivalence, not clean physical validation of BC priors.
Validated replacements are in
``forward_model_intermediates/name_endpoint_repair_20261031_ope239_r2``;
``usage.txt``, ``catalog-registration.json`` and ``sites/WES.nc`` were inspected.
Material sites on this roster are WES and KRE, with smaller source-version
differences at LIN and TOH. Corrected scientific tests require a new versioned
preparation for both implementations, holding WUR observations, surface
footprints, fluxes, row order and non-BC terms fixed. Old object-store UUIDs
do not automatically apply the endpoint overlay.

CORSO midpoint labels and the archived forward-fill policy are preserved;
instantaneous versus interval-average interpretation remains unconfirmed.
The BC provenance audit attributes the apparent historical exported mismatch
to different input generations, rather than an OGI export/model defect. No
biosphere-prior, endpoint, outer-flux or temporal-interpolation change is part
of this implementation.
