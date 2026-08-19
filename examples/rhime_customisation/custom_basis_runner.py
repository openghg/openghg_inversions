"""Run standard RHIME with project-owned emissions basis construction.

This copy-and-modify example keeps RHIME's supported acquisition, filtering,
sensitivity, labelled-input, model, sampling, and output stages.  Only the
emissions-basis stage is replaced by :func:`build_project_basis`.  The example
either loads a self-contained retained ``BasisFunctions`` artifact or creates
a guarded region-constrained basis whose regions stay within land/ocean
classes, remain connected, and satisfy a project eccentricity threshold.
The verification-games workflow weights that strategy with summed absolute
cached ``fp_x_flux``. At this earlier runner stage projected sensitivities do
not exist, so the example deliberately substitutes the public
``basis_weights_from_fp_all`` field while preserving the guarded strategy and
class composition.

Use :func:`run_custom_rhime` from Python or :func:`main` from the command line.
Acquired xarray objects are treated as borrowed.  Generated-basis fitting is
an explicit eager algorithm boundary, artifact loading materializes the saved
artifact, and model arrays materialize later at the named PyMC boundary.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import xarray as xr

from openghg_inversions.basis import (
    BasisFunctions,
    basis_functions_from_fp_all_flat_basis,
    basis_weights_from_fp_all,
    load_country_region_classes,
)
from openghg_inversions.basis.algorithms import (
    ConnectedComponentPartitionStep,
    ConnectedComponentSplitStrategy,
    GreedySplitStrategy,
    InertialSplitStep,
    LatLonGridGeometry,
    MaxChildPCAEccentricity,
    region_constrained_basis,
)
from openghg_inversions.inversion_data import RhimeMergedData
from openghg_inversions.rhime import (
    RhimeResult,
    assemble_rhime_inputs,
    build_rhime_sensitivities,
    build_standard_rhime_model_result,
    filter_rhime_observations,
    make_standard_rhime_result,
    materialize_pymc_inputs,
    params_from_config,
    resolve_rhime_options,
    retrieve_or_reload_rhime_data,
    sample_rhime_model,
    standard_model_input_names,
    with_prepared_rhime_sites,
)


DEFAULT_MAX_CHILD_PCA_ECCENTRICITY = 10.0


def _land_ocean_classes(
    *,
    domain: str,
    country_directory: str | Path | None = None,
) -> xr.DataArray:
    """Reduce the public country grid to the project's land/ocean classes.

    Args:
        domain: RHIME domain used to select the country grid.
        country_directory: Optional directory containing the domain country
            file.

    Returns:
        An eager object-valued ``(lat, lon)`` class grid preserving the loaded
        coordinates. Positive country codes are ``land``; null, zero, and
        negative codes are ``ocean``.

    Raises:
        FileNotFoundError: If the requested country grid does not exist.
        KeyError: If the country-grid artifact does not contain ``country``.
    """
    country_classes = load_country_region_classes(
        domain,
        country_directory,
    )
    return xr.where(country_classes > 0, "land", "ocean").astype(object).rename("basis_class")


def _guarded_basis(
    merged: RhimeMergedData,
    data_args: Mapping[str, Any],
    *,
    max_child_pca_eccentricity: float,
) -> BasisFunctions:
    """Build the project's connected, eccentricity-guarded retained basis.

    The public weight adapter intentionally materializes a derived 2-D weight
    field from borrowed footprint and flux inputs here. The project then owns
    normalization, class composition, guarded splitting, and conversion of the
    flat labels to retained ``BasisFunctions`` without mutating ``merged``.

    Args:
        merged: Filtered, borrowed RHIME observations and flux data.
        data_args: Resolved standard RHIME data and basis options. This stage
            consumes required ``domain`` and optional ``flux_sources``,
            ``nbasis``, and ``country_directory``.
        max_child_pca_eccentricity: Project limit on each proposed child's
            physical-coordinate PCA eccentricity.

    Returns:
        Retained basis functions fitted within connected land/ocean regions.

    Raises:
        FileNotFoundError: If the requested country grid does not exist.
        KeyError: If ``domain`` or country-grid content is missing.
        ValueError: If zero-filled weights do not have a positive finite
            maximum, or geometry, allocation, or guarded splitting is invalid.
    """
    # 1. Turn the filtered footprints and flux into one spatial importance map.
    weights = basis_weights_from_fp_all(
        merged.fp_all,
        data_args.get("flux_sources"),
        abs_flux=True,
    )
    finite_weights = weights.fillna(0.0).astype(np.float64)
    maximum = float(finite_weights.max())
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("Project basis weights must have a positive finite maximum.")
    normalized_weights = finite_weights / maximum

    # 2. Define scientific boundaries that no basis region may cross.
    region_classes = _land_ocean_classes(
        domain=data_args["domain"],
        country_directory=data_args.get("country_directory"),
    )
    geometry = LatLonGridGeometry.from_dataarray(normalized_weights)
    # 3. Compose the split algorithm and its shape/connectivity safeguards.
    strategy = ConnectedComponentSplitStrategy(
        split_strategy=GreedySplitStrategy(
            split_step=ConnectedComponentPartitionStep(
                split_step=InertialSplitStep(
                    balanced=True,
                    geometry=geometry,
                ),
                connectivity=1,
            ),
            split_acceptance=MaxChildPCAEccentricity(
                max_child_pca_eccentricity=max_child_pca_eccentricity,
                geometry=geometry,
            ),
        ),
        connectivity=1,
    )
    # 4. Allocate and split regions within the land/ocean classes.
    basis_flat = (
        region_constrained_basis(
            normalized_weights,
            region_classes,
            int(data_args.get("nbasis", 100)),
            allocation="weight",
            min_regions_per_class=1,
            split_strategy=strategy,
        )
        .astype(np.int16)
        .rename("basis")
    )
    provenance: dict[str, str | int | float] = {
        "openghg_inversions:basis_artifact_source": "project-guarded",
        "openghg_inversions:project_basis_strategy": "connected_component_balanced_inertial",
        "openghg_inversions:project_basis_connectivity": 1,
        "openghg_inversions:project_basis_max_child_pca_eccentricity": float(max_child_pca_eccentricity),
        "openghg_inversions:project_basis_class_policy": "land_ocean",
        "openghg_inversions:project_basis_weights": "basis_weights_from_fp_all_abs_flux_normalized",
    }
    basis_flat.attrs.update(provenance)
    # 5. Attach the current flux so standard RHIME sensitivity code can use it.
    return basis_functions_from_fp_all_flat_basis(
        fp_all=merged.fp_all,
        basis_flat=basis_flat,
        metadata=provenance,
    )


def build_project_basis(
    merged: RhimeMergedData,
    data_args: Mapping[str, Any],
    *,
    project_basis_path: str | Path | None = None,
    max_child_pca_eccentricity: float = DEFAULT_MAX_CHILD_PCA_ECCENTRICITY,
) -> BasisFunctions:
    """Return the project-selected retained emissions basis.

    Args:
        merged: Filtered, borrowed RHIME observations and flux data.
        data_args: Resolved standard RHIME data and basis options.
        project_basis_path: Optional ``.nc`` or ``.zarr`` artifact previously
            written by :meth:`BasisFunctions.save`.  The artifact is
            self-contained, so its serialized operator, metadata, and flux are
            retained rather than replaced with flux from ``merged``.
        max_child_pca_eccentricity: Project limit passed to the guarded split
            policy when generating a basis. Defaults to ``10`` and is ignored
            when ``project_basis_path`` is supplied.

    Returns:
        Loaded or newly fitted retained basis functions.

    Raises:
        OSError: If the requested artifact cannot be opened.
        KeyError: If required artifact content or generated-basis options are
            missing.
        ValueError: If the artifact or generated-basis configuration is
            invalid.

    Notes:
        Loading eagerly materializes the artifact so no open file handle is
        retained.  Without a path, project basis fitting is the named eager
        basis-generation boundary.
    """
    if project_basis_path is not None:
        return BasisFunctions.load(project_basis_path)
    return _guarded_basis(
        merged,
        data_args,
        max_child_pca_eccentricity=max_child_pca_eccentricity,
    )


def run_custom_rhime(
    *,
    config_file: str | Path | None = None,
    project_basis_path: str | Path | None = None,
    max_child_pca_eccentricity: float | None = None,
    **kwargs: Any,
) -> RhimeResult:
    """Run standard single-sector RHIME with a project-owned basis stage.

    Args:
        config_file: Optional RHIME INI configuration file.
        project_basis_path: Optional self-contained ``BasisFunctions`` artifact
            to use instead of fitting the project guarded basis.
        max_child_pca_eccentricity: Optional project split-policy threshold.
            When omitted, the merged config/keyword value is used, falling back
            to ``10``. This option is removed before standard RHIME resolution.
        **kwargs: Standard RHIME option names that override values from
            ``config_file``.

    Returns:
        The sampled result and any outputs requested by the RHIME options.

    Raises:
        OSError: If configured artifact, input, or output I/O fails.
        ValueError: If RHIME options, the custom basis, or prepared inputs are
            invalid.

    Notes:
        This workflow may retrieve or reload data, may eagerly fit or load a
        basis, eagerly materializes PyMC model inputs, runs sampling, and writes
        outputs requested by the resolved RHIME options.
    """
    params = (
        params_from_config(config_file, extra_kwargs=kwargs, normalise=False)
        if config_file is not None
        else dict(kwargs)
    )
    configured_project_basis_path = params.pop("project_basis_path", None)
    if project_basis_path is None:
        project_basis_path = configured_project_basis_path
    configured_eccentricity = params.pop(
        "max_child_pca_eccentricity",
        DEFAULT_MAX_CHILD_PCA_ECCENTRICITY,
    )
    if max_child_pca_eccentricity is None:
        max_child_pca_eccentricity = float(configured_eccentricity)
    setup = resolve_rhime_options(params=params, multisector=False)

    merged = retrieve_or_reload_rhime_data(setup.data_args, multisector=False)
    filtered = filter_rhime_observations(merged, setup.data_args)

    # CUSTOMISATION POINT: the standard runner calls build_rhime_basis here.
    # This project function either loads a saved basis or composes the guarded
    # basis-building tools above. Every stage after this call is standard RHIME.
    basis_functions = build_project_basis(
        filtered,
        dict(setup.data_args),
        project_basis_path=project_basis_path,
        max_child_pca_eccentricity=max_child_pca_eccentricity,
    )
    site_data = build_rhime_sensitivities(
        filtered,
        basis_functions,
        setup.data_args,
        multisector=False,
    )
    prepared = assemble_rhime_inputs(filtered, basis_functions, site_data, setup.data_args)
    run_spec = with_prepared_rhime_sites(setup.run_spec, prepared)

    # Cross the explicit eager PyMC boundary without changing canonical inputs.
    model_inputs = materialize_pymc_inputs(
        prepared,
        variable_names=standard_model_input_names(prepared, run_spec.model),
    )
    build_and_sample_start = perf_counter()
    model_build_result = build_standard_rhime_model_result(
        prepared=prepared,
        model_inputs=model_inputs,
        run_spec=run_spec,
    )
    idata = sample_rhime_model(model_build_result, setup.sampler)

    return make_standard_rhime_result(
        prepared=prepared,
        run_spec=run_spec,
        sampler=setup.sampler,
        model_build_result=model_build_result,
        idata=idata,
        build_and_sample_seconds=perf_counter() - build_and_sample_start,
    )


def _json_object(value: str) -> dict[str, Any]:
    """Parse one command-line JSON object containing additional RHIME options.

    Args:
        value: JSON text to parse.

    Returns:
        The decoded JSON object.

    Raises:
        argparse.ArgumentTypeError: If the text is invalid JSON or decodes to
            a non-object value.
    """
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--kwargs must decode to a JSON object")
    return parsed


def main(argv: Sequence[str] | None = None) -> RhimeResult:
    """Parse command-line options and run the custom-basis RHIME workflow.

    Args:
        argv: Command-line tokens excluding the program name. ``None`` reads
            tokens from :data:`sys.argv`.

    Returns:
        The RHIME result returned by :func:`run_custom_rhime`.

    Raises:
        SystemExit: If argument parsing fails or help is requested.

    Notes:
        Data access, eager basis and model materialization, sampling, and
        requested output writes are delegated to :func:`run_custom_rhime`; its
        exceptions propagate unchanged.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", nargs="?", type=Path, help="RHIME INI configuration file")
    parser.add_argument(
        "--project-basis-path",
        type=Path,
        help="self-contained BasisFunctions .nc or .zarr artifact",
    )
    parser.add_argument(
        "--max-child-pca-eccentricity",
        type=float,
        help="project guarded-basis threshold (default: 10)",
    )
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--output-name")
    parser.add_argument("--draws", type=int)
    parser.add_argument("--tune", type=int)
    parser.add_argument("--chains", type=int)
    parser.add_argument(
        "--kwargs",
        type=_json_object,
        default={},
        metavar="JSON",
        help="additional standard RHIME options as a JSON object",
    )
    args = parser.parse_args(argv)

    overrides = dict(args.kwargs)
    for name in ("start_date", "end_date", "output_path", "output_name", "draws", "tune", "chains"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    return run_custom_rhime(
        config_file=args.config_file,
        project_basis_path=args.project_basis_path,
        max_child_pca_eccentricity=args.max_child_pca_eccentricity,
        **overrides,
    )


if __name__ == "__main__":
    main()
