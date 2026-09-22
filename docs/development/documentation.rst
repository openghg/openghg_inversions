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
