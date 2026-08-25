Releases and branch maintenance
===============================

OpenGHG Inversions supports published PyPI packages and tagged releases. The
``devel`` branch is an unsupported integration branch: users should not depend
on it for scientific work.

Branch roles
------------

``main``
   The current stable release line. Current-line hotfixes start here. We do not
   maintain branches for older minor releases.

``devel``
   Integration for the next monthly release. Only reviewed, release-ready
   changes merge here.

``feature/*``
   Work that is not ready to release. Open a draft pull request for visibility.
   Feature branches track ``devel`` only; do not merge both ``main`` and
   ``devel`` into them.

``release/vX.Y.Z``
   A short-lived branch created by the release-preparation workflow. It is
   deleted after publication.

``hotfix/X.Y.Z-description``
   A short-lived fix branch from ``main``. A tag is immutable and cannot be a
   pull-request target. While 0.6 is current, for example, a 0.6 hotfix targets
   ``main`` rather than ``v0.6.0``.

Monthly release
---------------

The monthly reminder opens a tracking issue on the first Monday of each month.
From the Actions page, run **Prepare release** with the next minor version and
``source=devel``. The workflow:

* creates ``release/vX.Y.0``;
* updates ``pyproject.toml``;
* assembles Towncrier fragments into ``CHANGELOG.md``;
* commits and pushes the result;
* opens a pull request to ``main``; and
* dispatches ordinary CI for the generated commit.

``devel`` remains open while the release pull request is stabilized. Apply a
release-blocking fix to the release branch and ``devel``; do not leave a fix on
only one side.

After the release pull request merges, run the real-data ``test_inversions``
suite on the exact resulting ``main`` commit. The local tests marked ``slow``
are development contracts, not a substitute for this release assessment.

HPC evidence and scientific approval
------------------------------------

The external ``hpc-ci`` tool pins the tested Git SHA and collects compact
evidence. For a release pull request, its manual trigger can fetch the PR ref;
for the final merged commit, pass an explicit ``main`` ref and verify that the
reported SHA matches ``git rev-parse origin/main``. Keep the ``hpc-ci``
collection, configuration/data identity, Slurm job references, summaries and
interpretable scientific outputs together.

Upload or link the compact evidence somewhere reachable by HTTPS, normally the
release tracking issue or release pull request. A maintainer reviews the real
results and runs **Record HPC release check** with:

* the full tested commit SHA;
* the reviewed outcome;
* the HTTPS evidence link; and
* a short summary.

The workflow refuses commits outside ``main`` history and records the fixed
``HPC / test_inversions`` status against that exact SHA. Configure the
``hpc-release-review`` GitHub environment with required reviewers so both
manual attestations and future ``hpc-ci`` ``repository_dispatch`` callbacks
pause for scientific approval.

Publication
-----------

Run **Publish to PyPI** with the version on the approved ``main`` commit. The
workflow refuses to publish unless:

* the requested version, ``pyproject.toml`` and Towncrier heading agree;
* no unassembled news fragments remain;
* ``CI Gate`` succeeded on the exact commit;
* ``HPC / test_inversions`` succeeded on the same commit; and
* the protected ``pypi`` environment is approved.

It builds and clean-installs the wheel, checks the installed metadata version,
publishes through PyPI trusted publishing, creates the GitHub release, rebuilds
the released documentation, and opens a ``main`` to ``devel`` forward-port
pull request. Auto-merge is enabled when repository settings and required
checks allow it. The release branch is deleted only after publication.

If automated publication fails after PyPI accepts the artifact, do not publish
the same version again. Create the missing GitHub release for that tag and run
**Forward released main to devel** manually. If it fails before PyPI accepts
the artifact, correct the workflow or release commit, rerun the exact-SHA
tests when the commit changes, and dispatch publication again.

Current-line hotfix
-------------------

For an urgent bug affecting the published version:

#. Create ``hotfix/X.Y.Z-description`` from ``main``.
#. Add the fix, a focused regression test, and a Towncrier bugfix fragment.
#. Merge the reviewed pull request to ``main``.
#. Run **Prepare release** with the patch version and ``source=main``.
#. Merge the generated release pull request.
#. Run and scientifically review ``test_inversions`` on the final ``main`` SHA.
#. Record the HPC check and run **Publish to PyPI**.
#. Confirm that the automatic ``main`` to ``devel`` pull request merges.

Once ``main`` advances to a new minor version, the older minor is unsupported.
Do not create a maintenance branch without first changing this policy.

Feature drift and inactive pull requests
----------------------------------------

Every Monday, **Pull request maintenance** checks open pull requests targeting
``devel``. A branch behind ``devel`` receives ``needs-devel-sync`` and one bot
comment that is updated in place. Apply ``auto-sync-devel`` to a same-repository
pull request to opt into GitHub's clean update-branch operation. Conflicts get
``devel-sync-conflict`` and remain for a person to resolve. Forks receive a
reminder but are never mutated.

Inactive pull requests receive ``stale`` after 30 days and close after another
14 days. Activity removes the stale state. ``keep-open`` exempts a pull request.
Closing never deletes its branch, and work can be reopened later.

Manual fallbacks
----------------

Every automation leaves ordinary Git and GitHub operations available:

* prepare a release branch locally, update the version, and run
  ``towncrier build --version X.Y.Z --yes``;
* open release and forward-port pull requests manually;
* record the exact-SHA HPC status with the attestation workflow after uploading
  evidence; and
* create a GitHub release manually only after the same CI and HPC gates pass.

Never bypass or recreate a successful PyPI publication. When a commit changes,
its previous HPC result does not apply.
