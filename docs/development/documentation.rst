Maintain the documentation
==========================

The API reference combines a hand-maintained landing page with generated
module pages. Keeping these two parts separate makes the public API easier to
navigate without requiring maintainers to duplicate detailed signatures and
docstrings.

Update the hand-maintained API reference
----------------------------------------

``docs/reference/index.rst`` is the API landing page. Its introductory text,
section headings, and autosummary tables are maintained by hand. Update it
when a public object is added, removed, renamed, or moved, or when an object
belongs under a different reader-facing heading. Adding a module does not add
its public objects to the landing page automatically.

The autosummary entries must use fully qualified import paths. Sphinx reads
the current signatures and summaries from the implementation, so do not copy
those details into the landing page. Update the relevant docstring instead.

Generated signatures keep annotations beside documented parameter
descriptions, rather than crowding them into the signature itself. They also
omit module prefixes from displayed object names; the fully qualified import
path remains available in the page URL, module heading, and cross-reference
target. These are site-wide presentation choices controlled by
``autodoc_typehints``, ``autodoc_typehints_description_target``, and
``add_module_names`` in ``docs/conf.py``. Change those settings only when the
whole API reference should adopt a different style.

Regenerate module pages
-----------------------

The detailed module pages in ``docs/reference/openghg_inversions.*.rst`` are
generated from the package layout. Regenerate them after adding, removing, or
moving a Python module:

.. code-block:: bash

   pixi run -e dev tox -e docs-api

This command uses the templates in ``docs/_templates`` and does not generate
or overwrite ``docs/reference/index.rst``. Review and commit the resulting
changes. Do not edit generated module pages by hand; change a template, the
Sphinx configuration, or the source docstring and regenerate instead.

Build and check the documentation
---------------------------------

For an incremental local build and browser preview, run:

.. code-block:: bash

   pixi run -e dev docs-preview --no-open

The preview command runs ``tox -e docs``. It does not regenerate API module
pages, so it is safe to use while editing the hand-maintained landing page.
Stop and rerun it after changing a source file. Add ``--fresh`` if cached
Sphinx output obscures a change.

Before opening a pull request, regenerate API pages when required and run the
full build:

.. code-block:: bash

   pixi run -e dev tox -e docs-full
   git diff --check

``docs-full`` regenerates the module pages before building them. Use
``docs-api`` on its own when you only need to refresh and review those pages.

``docs-strict`` is available as a diagnostic build that treats Sphinx warnings
and missing cross-references as errors. The existing documentation has a
warning backlog, so this environment does not yet provide a clean pass/fail
check for a pull request. Review its output for new warnings in changed pages.

Inspect the rendered API landing and affected module pages as well as the
command output. A successful build cannot by itself confirm that the grouping
and descriptions are useful to readers.

Review a pull-request preview
-----------------------------

The **Validate Docs** check uploads a ``docs-versioning-preview`` artifact containing a
local copy of the documentation site. Download it from the completed Actions
run, extract it, and run the following command in the extracted directory:

.. code-block:: bash

   python3 -m http.server 8765 --bind 127.0.0.1

Open ``http://127.0.0.1:8765/``. Use this web server rather than opening the
HTML files directly: the version selector loads its index over HTTP. The
preview's links use this local address; it does not publish to GitHub Pages.

Ordinary documentation changes preview the pull request as ``devel``. Changes
to the versioning configuration or publishing machinery also build
``v0.7.4`` and ``v0.6.0`` as representative historical releases. For those
changes, check that the root opens 0.7.4, development pages display an
unreleased notice, and the selector and archive reach both release builds.
Try a nested usage or API page as well as each homepage. The fixed historical
versions exercise both build layouts; they do not select the supported release
for the public site.

Publish versioned documentation
-------------------------------

GitHub Pages keeps release documentation at ``/vX.Y.Z/`` and development
documentation at ``/devel/``, under
``https://openghg.github.io/openghg_inversions/``. The site root and
``/stable/`` lead to the highest published stable release that has a successful
documentation build. A development build or an older release backfill does
not move this entry point backwards. Release snapshots remain available in
the version selector and the ``/versions/`` archive; availability of archived
documentation does not imply continuing support for that release. See
:doc:`releasing` for the supported release policy.

The **Build & Deploy Versioned Docs** workflow builds one version at a time, installing
the package from the selected ref in its own build environment. It uses the
Sphinx configuration and API generation from that ref, with the maintained
version-selector configuration appended for older releases. It replaces only
that version's generated output and preserves other published versions.
Publication also updates the shared ``versions.json`` selector list and
redirects. Existing root HTML page links redirect to the corresponding stable
page when available, or to the stable homepage when the page was removed.

Changes to ``devel`` trigger development builds. Release publication dispatches
the maintained workflow on ``main`` with the release tag as its ``docs_ref``
input. To backfill or rebuild a published release, run this workflow from
``main`` in GitHub Actions, or use GitHub CLI:

.. code-block:: bash

   gh workflow run docs.yaml --ref main -f docs_ref=v0.7.4

Substitute the desired published release tag. Drafts and prereleases are not
published by this workflow. To rebuild development documentation manually,
use ``docs_ref=devel``. A failed build leaves the published site unchanged;
inspect its Actions logs and uploaded build artifact before retrying. Historical
dependency requirements may need a separately reviewed build adjustment.

For the initial rollout, follow this order:

#. Upgrade ``main`` with the maintained workflow, publishing helper, and
   selector configuration. Keep the ``devel`` pull request unmerged: merging
   it automatically publishes development documentation, which becomes the
   default if no release snapshot exists yet.
#. Drain queued or running legacy documentation deployments, including their
   GitHub Pages publication runs, by waiting for completion or cancelling
   them. Avoid changes that trigger the old ``devel`` workflow until the
   branch upgrade is complete; do not rerun legacy deployments.
#. Dispatch the maintained workflow on ``main`` with the latest supported
   0.7 release tag as ``docs_ref``. Wait for GitHub Pages publication and
   verify that the public root and ``/stable/`` lead to that release.
#. Only then merge the versioning changes into ``devel``. Verify that its
   automatic deployment preserves the stable release and default, then
   backfill ``v0.6.0``.

Never dispatch ``docs.yaml`` with ``--ref`` set to a historical release tag:
its old workflow can replace the entire site. Do not rewrite release tags to
install the new workflow.

After each publication, check the version label, selector, archive, and
``/stable/`` destination on the hosted site. Check a nested API or usage page
and an existing root link as well as the homepage. After publishing ``devel``
or an older release, confirm that the supported release and its entry point
are still present. The ``gh-pages`` commit history records previous complete
site trees for recovery if a legacy workflow accidentally replaces the site.
