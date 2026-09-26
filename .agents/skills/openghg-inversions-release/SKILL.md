---
name: openghg-inversions-release
description: Prepare, publish, and verify OpenGHG Inversions releases; create current-line hotfix pull requests; or promote an existing devel pull request into a safe sibling hotfix. Use for requests such as “make the next minor release”, “make a bugfix release”, “prepare but do not publish”, or “turn PR 123 into a hotfix”.
---

# OpenGHG Inversions releases and hotfixes

Read the [release runbook](../../../docs/development/releasing.rst) completely before acting. Treat it and the repository workflows as authoritative when they differ from this skill.

## Choose the operation

- **Next minor release:** release from `devel` with `source=devel` and a version whose patch component is zero.
- **Bugfix release:** release the current stable line from `main` with `source=main` and the next unused patch version. Do not create a patch release merely because `main` advanced.
- **Prepare only:** stop after the generated release pull request and report what remains.
- **Publish:** continue through the protected PyPI environment only when the user asked to make or publish the release. Environment approval and any admin merge override remain explicit permissions.
- **New hotfix PR:** branch from current `main`, add the focused fix, regression test, and Towncrier bugfix fragment, then open a PR to `main`.
- **Promote a `devel` PR:** create a sibling hotfix PR as described below. Do not casually retarget the existing PR.

Use an isolated worktree when the current checkout has unrelated changes or is on another task branch. Preserve user files and unrelated work.

## Release workflow

1. Inspect the current branches, latest release, open release PRs, working tree, Towncrier fragments, `.zenodo.json`, and relevant workflow state. Resolve the intended version from repository state; ask only if more than one version is genuinely plausible.
2. Confirm the source commit has a successful exact-commit `CI Gate`.
3. Run **Prepare release** with the selected version and source. Inspect the generated diff and contributor-candidate checklist, then attach the PR to the task. Update `.zenodo.json` on the release branch when needed; every contributor must have a `type`. Any release-branch update must receive fresh exact-commit CI.
4. Merge only when the required checks pass on the final release commit. Do not use an admin override unless the user explicitly authorized it for this work.
5. Run **Publish to PyPI** only from the approved `main` commit. Approve the protected environment only when authorized. Never move an existing tag or overwrite a PyPI artifact.
6. Verify the GitHub release and tag, PyPI wheel and source distribution, release metadata, Zenodo webhook acceptance, and the version DOI. Allow Zenodo time to process; do not redeliver an accepted webhook while processing is plausible.
7. Confirm the automatic `main`-to-`devel` forward-port, its CI, documentation deployment, and release-branch cleanup. Report any expected superseded workflow separately from failures.

## Promote a devel PR to a hotfix

Treat “convert” as **promote**, not as permission to change the existing PR's base.

1. Inspect the PR's base, head repository, commits, full diff, dependencies on unreleased `devel`, checks, and Towncrier fragment.
2. Confirm the fix affects the published stable version and is separable from unrelated feature work. If not, explain why it is unsuitable for a hotfix instead of manufacturing a large patch.
3. Create `hotfix/X.Y.Z-description` from current `main`. Transplant only the required commits when they apply cleanly; otherwise reapply the minimal diff while preserving attribution in the commit or PR description.
4. Ensure the sibling hotfix contains a focused regression test and a bugfix fragment, then validate it and open a PR to `main`. Link the original `devel` PR in both directions.
5. Leave the original PR open until the published hotfix is forward-ported. Afterwards, confirm `devel` contains the fix, then close the original as redundant or retain only its remaining unreleased changes.

Never retarget a `devel`-based PR to `main` unless comparison proves it contains no `devel`-only history or changes. Retargeting an ordinary feature branch can silently include every unreleased commit.

## Handoff

Lead with the outcome. Include the version, release/PR links, exact CI status, PyPI and Zenodo results when applicable, forward-port state, and any remaining action. Do not present a release as complete while a required publication or verification step remains.
