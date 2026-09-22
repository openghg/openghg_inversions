HBMCMC compatibility namespace
==============================

``openghg_inversions.hbmcmc`` is no longer an inversion API. In 0.8, the
direct ``fixedbasisMCMC`` and ``inferpymc`` implementation and its preparation,
postprocessing, and plotting helpers were removed. The 0.7.x release line is
the final line containing the direct implementation.

Only ``openghg_inversions.hbmcmc.run_hbmcmc`` remains as a transitional
command-line wrapper for supported fixedbasis-style INI files. It translates
the old vocabulary and always calls :func:`openghg_inversions.rhime.run_rhime`.
It does not accept ``--legacy-fixedbasis`` or generate legacy templates.

Use :doc:`../usage/legacy_and_migration` to migrate an INI file, Python call,
batch script, or historical output workflow. New code should use the public
:mod:`openghg_inversions.rhime` API.
