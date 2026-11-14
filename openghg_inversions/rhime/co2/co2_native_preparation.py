"""Build a CO2 preparation from configured acquisition and native choices.

The steps run in scientific order: acquire (or accept) merged surface data,
select the boundary representation the configured choice needs, select
observations, derive observation errors, build the basis and canonical inputs,
reduce the native Gaussian prior coherently, and form the CO2 artifact with its
affine companion. Saving belongs to :mod:`openghg_inversions.rhime.co2.handoffs`.

The native prior is either one dimensionless scaling field shared by every
flux source, or separate per-source fields (GPP, TER, FF, ocean, ...) with
their own amplitudes and kernels, an optional GPP/TER correlation, and one
joint reduction over the stacked ``(source, lat, lon)`` state, so the
retained state is (source, region).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.array_ops import force_align
from openghg_inversions.basis import prepare_basis
from openghg_inversions.basis.affine_flux_map_io import AffineFluxMapArtifact
from openghg_inversions.basis.basis_functions import FluxWeightedBasis, basis_functions_from_flat_basis
from openghg_inversions.basis.covariance_products import (
    PreserveBucketProlongation,
    RetainedProjectionStrategy,
    SuppliedRestriction,
    absolute_weighted_regional_mean_restriction,
    covariance_natural_prolongation,
)
from openghg_inversions.basis.operators import MultiSourceBucketBasisOperator
from openghg_inversions.coherent_reduction import reduce_native_gaussian
from openghg_inversions.correlated_source_covariance import CorrelatedSourceCovariance
from openghg_inversions.inversion_data import MergedData, PreparedInputs, select_boundary_representation
from openghg_inversions.inversion_data.observation_errors import prepare_observation_errors
from openghg_inversions.models.boundary import Preparation
from openghg_inversions.models.boundary.contracts import validate_centering_weights
from openghg_inversions.native_b_calibration import calibrate_native_b_amplitudes
from openghg_inversions.native_covariance import (
    ClassMixedExponentialCovariance,
    GeodesicExponentialKernel,
    InvertibleNativeCovarianceAction,
    SeparableExponentialCovariance,
)
from openghg_inversions.native_flux import resolve_runtime_flux
from openghg_inversions.utils import areagrid
from openghg_inversions.rhime.preparation import (
    assemble_inputs,
    build_sensitivities,
    filter_observations,
    resolve_basis_inputs,
    resolve_bc_basis,
)

from .co2_affine_output import import_explicit_affine_flux_map, produce_bucket_affine_flux_map
from .co2_preparation import Co2PreparedInputs, prepare_co2_inputs

if TYPE_CHECKING:
    from .configuration import (
        Co2MultiSourceNativePrior, Co2NativePreparationConfig, Co2NativePrior, Co2NativeSourcePrior,
    )

NATIVE_SOURCE_DIM = "native_source"
"""Native source dimension of per-source fields, distinct from the retained ``source`` level."""


def prepare_native_co2(
    native: Co2NativePreparationConfig,
    *,
    domain: str,
    start_date: str,
    end_date: str,
    boundary: Preparation,
    merged_data: MergedData | None = None,
) -> tuple[Co2PreparedInputs, AffineFluxMapArtifact]:
    """Acquire and prepare one coherent CO2 artifact and its affine companion.

    Selected basis and weight files are read before acquisition. Without
    ``merged_data``, CO2 surface data are acquired for the requested window;
    supplied merged data are checked against that window instead. One merged
    dataset serves either boundary representation.

    With a single-field prior, all flux sources share one dimensionless
    native scaling field, so their summed response ``fp_x_flux`` is the
    native sensitivity. With per-source priors, each source's response in
    ``fp_x_flux_sectoral`` is its native sensitivity, every source keeps its
    own regions of the basis partition, and
    :func:`prepare_multisource_native_co2_inputs` reduces the stacked
    sources jointly. The first configured source weights basis design.

    Args:
        native: Configured acquisition and native preparation choices.
        domain: Transport domain.
        start_date: Requested window start.
        end_date: Requested window end.
        boundary: Boundary preparation the configured model needs.
        merged_data: Optional borrowed merged data, skipping acquisition.

    Returns:
        The CO2 prepared inputs and their unsaved affine companion.

    Raises:
        ValueError: If supplied data conflict with the request, cannot provide
            the boundary representation, or preparation contracts fail.
    """
    # Resolve file-backed resources before any acquisition.
    basis_inputs = resolve_basis_inputs(native.basis)
    bc_basis = resolve_bc_basis(native.basis)
    weights = None
    if native.bc_centering_weights_path is not None:
        weights = _load_centering_weights(native.bc_centering_weights_path)
    from .configuration import Co2MultiSourceNativePrior

    multisource = native.native_prior if isinstance(native.native_prior, Co2MultiSourceNativePrior) else None
    resources = MultiSourceNativeResources.load(multisource) if multisource is not None else None

    acquisition = dict(native.acquisition)
    flux_sources = tuple(acquisition["flux_sources"])
    use_bc = bool(acquisition.get("use_bc", True))
    if merged_data is None:
        merged_data = MergedData.from_options(
            species="co2", domain=domain, start_date=start_date, end_date=end_date, **acquisition,
        )
    else:
        merged_data.validate_for_preparation(
            species="co2", domain=domain, start_date=start_date, end_date=end_date, split_by_sectors=False,
        )

    merged_data.validate_scientific_inputs(
        split_by_sectors=False, use_bc=use_bc, flux_sources=flux_sources,
        require_footprints=basis_inputs.get("basis_path") is None and basis_inputs.get("basis_partition") is None,
    )
    site_data = select_boundary_representation(merged_data.site_data, boundary.representation)
    filtered, site_options = filter_observations(site_data, merged_data.site_options, filters=native.filters)
    filtered = prepare_observation_errors(filtered, averaging_error=native.averaging_error)

    runtime_flux = resolve_runtime_flux(merged_data.flux_data, explicit_surface=True)
    if multisource is None:
        basis_functions = prepare_basis(
            site_data=filtered, runtime_flux=runtime_flux,
            weighting_flux=merged_data.flux_data[flux_sources[0]]["flux"],
            domain=domain, start_date=start_date, basis_algorithm=native.basis.basis_algorithm,
            nbasis=native.basis.nbasis, fix_outer_regions=native.basis.fix_basis_outer_regions,
            **basis_inputs,
        )
    else:
        basis_functions = _multisource_basis(
            native, basis_inputs, filtered, merged_data, runtime_flux, flux_sources=flux_sources,
            domain=domain, start_date=start_date,
        )
    sensitivities = build_sensitivities(
        filtered if multisource is None else _sectoral_response_only(filtered), basis_functions,
        flux_sources=flux_sources, use_bc=use_bc, bc_basis=bc_basis, multisector=multisource is not None,
    )
    canonical = assemble_inputs(
        sensitivities, basis_functions, site_options=site_options,
        domain=domain, start_date=start_date, use_bc=use_bc,
    )
    if multisource is not None:
        assert resources is not None
        operator = cast(MultiSourceBucketBasisOperator, basis_functions.operator)
        return prepare_multisource_native_co2_inputs(
            canonical, multisource,
            native_sensitivity=_native_source_sensitivity(filtered, canonical, operator.source_labels),
            resources=resources, aggregation_error_rank=native.aggregation_error_rank,
            bc_centering_weights=weights,
            provenance={"native_preparation": {
                "boundary_representation": boundary.representation,
                "flux_sources": list(flux_sources),
                "basis_algorithm": native.basis.basis_algorithm,
            }},
        )

    native_sensitivity = _native_sensitivity(filtered, canonical)
    prior = cast("Co2NativePrior", native.native_prior)
    native_mean = xr.DataArray(
        np.full((native_sensitivity.sizes["lat"], native_sensitivity.sizes["lon"]), prior.mean),
        dims=("lat", "lon"), coords={"lat": native_sensitivity.lat, "lon": native_sensitivity.lon},
        attrs={"units": "1"},
    )
    covariance = SeparableExponentialCovariance(
        native_sensitivity.lat, native_sensitivity.lon, sigma=prior.sigma,
        correlation_length=prior.correlation_length,
        latitude_correlation_length=prior.latitude_correlation_length,
        longitude_correlation_length=prior.longitude_correlation_length,
    )
    reduction = reduce_native_gaussian(
        covariance=covariance, strategy=PreserveBucketProlongation(basis_functions.operator.basis_matrix),
        state_dim=basis_functions.operator.meta.state_dim, native_mean=native_mean,
        native_sensitivity=native_sensitivity, observation_dim="observation",
    )
    prepared = prepare_co2_inputs(
        canonical, reduction, aggregation_error_rank=native.aggregation_error_rank,
        bc_centering_weights=weights,
        provenance={"native_preparation": {
            "boundary_representation": boundary.representation,
            "native_prior": asdict(prior),
            "flux_sources": list(flux_sources),
            "basis_algorithm": native.basis.basis_algorithm,
        }},
    )
    affine = produce_bucket_affine_flux_map(prepared, native_mean, prepared_inputs_id="unsaved")
    return prepared, affine


def _native_sensitivity(site_data: dict[str, xr.Dataset], canonical: PreparedInputs) -> xr.DataArray:
    """Gather each retained row's native ``fp_x_flux`` in canonical observation order.

    Canonical rows are grouped by site in retained-site order, so gathering
    each site's retained times in turn reproduces the ``nmeasure`` labels.
    """
    index = cast(pd.MultiIndex, canonical.inv_inputs.indexes["nmeasure"])
    sites, times = index.get_level_values("site"), index.get_level_values("time")
    parts = [
        site_data[site]["fp_x_flux"].sel(time=times[sites == site]).reset_coords(drop=True)
        for site in pd.unique(sites)
    ]
    gathered = xr.concat(parts, dim="time", join="exact").transpose("time", "lat", "lon")
    if not pd.MultiIndex.from_arrays([np.repeat(pd.unique(sites), [len(p.time) for p in parts]),
                                      gathered.indexes["time"]]).equals(index):
        raise ValueError("Canonical CO2 rows are not grouped by retained site.")
    return gathered.drop_vars("time").rename(time="observation").assign_coords(
        xr.Coordinates.from_pandas_multiindex(index, "observation")
    )


@dataclass(frozen=True)
class MultiSourceNativeResources:
    """File-backed inputs of a per-source native prior, read before acquisition.

    Attributes:
        class_labels: Integer class labels on ``(lat, lon)`` or
            ``(source, lat, lon)``, or ``None`` for one class per source.
        retained_regions: Retained region labels per source, as written in
            the basis partition, or ``None`` to retain every region.
        calibration_targets: Functional weights and SD targets, or ``None``.
    """

    class_labels: xr.DataArray | None = None
    retained_regions: Mapping[str, tuple[int, ...]] | None = None
    calibration_targets: xr.Dataset | None = None

    @classmethod
    def load(cls, prior: Co2MultiSourceNativePrior) -> MultiSourceNativeResources:
        """Read and check the configured class labels, retained regions and targets."""
        class_labels = None
        if prior.class_labels_path is not None:
            with xr.open_dataset(prior.class_labels_path) as dataset:
                if "class_labels" not in dataset:
                    raise ValueError(f"{prior.class_labels_path} must contain a class_labels variable.")
                class_labels = dataset["class_labels"].load()
            if not np.issubdtype(class_labels.dtype, np.integer):
                raise ValueError("Native class_labels must be integers.")
            if set(class_labels.dims) not in ({"lat", "lon"}, {"source", "lat", "lon"}):
                raise ValueError("Native class_labels must use dimensions (lat, lon) or (source, lat, lon).")
        retained = None
        if prior.retained_regions_path is not None:
            record = json.loads(Path(prior.retained_regions_path).read_text())
            if not isinstance(record, Mapping) or not all(
                isinstance(labels, list) and labels and all(type(label) is int for label in labels)
                for labels in record.values()
            ):
                raise ValueError("retained_regions_path must map each source to a non-empty list of integer labels.")
            retained = {str(source): tuple(sorted(set(labels))) for source, labels in record.items()}
        targets = None
        if prior.calibration is not None:
            targets = xr.load_dataset(prior.calibration.targets_path)
            given = [name for name in ("target_sd", "target_relative_sd") if name in targets]
            if "functional_weights" not in targets or len(given) != 1:
                raise ValueError(
                    f"{prior.calibration.targets_path} must contain functional_weights and exactly one of "
                    "target_sd or target_relative_sd."
                )
        return cls(class_labels=class_labels, retained_regions=retained, calibration_targets=targets)


def prepare_multisource_native_co2_inputs(
    canonical: PreparedInputs,
    prior: Co2MultiSourceNativePrior,
    *,
    native_sensitivity: xr.DataArray,
    resources: MultiSourceNativeResources | None = None,
    aggregation_error_rank: int | None = 512,
    bc_centering_weights: xr.DataArray | None = None,
    provenance: Mapping[str, Any] | None = None,
    observation_batch_size: int = 64,
) -> tuple[Co2PreparedInputs, AffineFluxMapArtifact]:
    """Reduce per-source native priors jointly into the CO2 artifact.

    Each source ``s`` has a dimensionless native scaling field with constant
    mean ``m_s`` and covariance ``sigma_s**2 K_s``; GPP and TER may be
    correlated under one shared kernel. One coherent reduction over the
    stacked ``(native_source, lat, lon)`` state gives the retained (source,
    region) prior ``C_alpha`` with its GPP/TER cross blocks, ``H_alpha`` and
    the unresolved covariance ``A`` that includes every cross-source term.

    Args:
        canonical: Canonical inputs whose basis is a
            :class:`~openghg_inversions.basis.operators.MultiSourceBucketBasisOperator`
            with one partition per configured source; its flux supplies the
            restriction weights, calibration flux and affine flux.
        prior: Per-source native priors.
        native_sensitivity: Native ``H`` on ``(observation, native_source,
            lat, lon)`` in canonical observation order and units, with the
            basis source order.
        resources: Class labels, retained regions and calibration targets
            read from the configured files.
        aggregation_error_rank: LRPD rank, or ``None`` for exact storage.
        bc_centering_weights: Optional additive-boundary centering weights.
        provenance: Extra JSON-safe provenance recorded with the artifact.
        observation_batch_size: Covariance right-hand sides per block.

    Returns:
        The CO2 prepared inputs and their unsaved affine companion: the
        bucket map for ``restriction = "bucket"``, otherwise the exact
        covariance-natural lift ``U_* = B Pi.T C_alpha^-1``.

    Raises:
        ValueError: If sources, grids, class labels, regions or targets do
            not match, or a reduction contract fails.
    """
    resources = resources or MultiSourceNativeResources()
    basis = canonical.basis_functions
    operator = basis.operator
    if not isinstance(operator, MultiSourceBucketBasisOperator):
        raise ValueError("Per-source native priors require a source-specific (multisource) bucket basis.")
    sources = operator.source_labels
    if set(sources) != set(prior.sources):
        raise ValueError(f"Native prior sources {sorted(prior.sources)!r} differ from basis sources {sorted(sources)!r}.")
    sensitivity = native_sensitivity.transpose("observation", NATIVE_SOURCE_DIM, "lat", "lon")
    if list(map(str, sensitivity.indexes[NATIVE_SOURCE_DIM])) != list(sources):
        raise ValueError(f"Native sensitivity sources must follow the basis order {list(sources)!r}.")
    latitude, longitude = sensitivity["lat"], sensitivity["lon"]
    grid = xr.Dataset(coords={"lat": latitude, "lon": longitude})

    flux = _source_flux(basis, operator, grid)
    native_mean = xr.DataArray(
        np.stack([np.full((latitude.size, longitude.size), prior.sources[source].mean) for source in sources]),
        dims=(NATIVE_SOURCE_DIM, "lat", "lon"),
        coords={NATIVE_SOURCE_DIM: list(sources), "lat": latitude, "lon": longitude},
        attrs={"units": "1"},
    )
    classes = {source: _source_class_labels(resources.class_labels, source, grid) for source in sources}
    sigmas, calibration = _source_sigmas(prior, sources, flux, classes, resources.calibration_targets, grid)
    covariance = _source_covariance(prior, sources, sigmas, classes, latitude, longitude)

    state_dim = operator.meta.state_dim
    strategy: RetainedProjectionStrategy
    if prior.restriction == "bucket":
        strategy = PreserveBucketProlongation(
            operator._native_prolongation(native_mean, native_dims=(NATIVE_SOURCE_DIM, "lat", "lon"))
        )
        restriction = None
    else:
        restriction = _flux_weighted_restriction(operator, flux, resources.retained_regions, grid)
        strategy = SuppliedRestriction(
            restriction, require_positivity_preserving=True, name="absolute_flux_weighted_regional_mean",
        )
    reduction = reduce_native_gaussian(
        covariance=covariance, strategy=strategy, state_dim=state_dim, native_mean=native_mean,
        native_sensitivity=sensitivity, observation_dim="observation",
        observation_batch_size=observation_batch_size,
    )
    if restriction is not None:
        canonical = _retained_canonical(canonical, reduction.retained_mean.indexes[state_dim], state_dim)

    record = _multisource_record(prior, sources, sigmas, calibration, classes, covariance, reduction, state_dim)
    metadata = {**dict(provenance or {})}
    native_record = {**dict(metadata.get("native_preparation", {})), **record}
    metadata["native_preparation"] = native_record
    prepared = prepare_co2_inputs(
        canonical, reduction, aggregation_error_rank=aggregation_error_rank,
        bc_centering_weights=bc_centering_weights, provenance=metadata,
    )
    source_provenance = {"native_prior": record}
    if restriction is None:
        affine = produce_bucket_affine_flux_map(
            prepared, native_mean, prepared_inputs_id="unsaved", source_provenance=source_provenance,
        )
    else:
        lift = covariance_natural_prolongation(
            covariance=covariance, restriction=restriction, state_dim=state_dim,
            rhs_block_size=observation_batch_size,
        )
        affine = import_explicit_affine_flux_map(
            prepared, native_mean, basis.flux, lift, prepared_inputs_id="unsaved",
            source_provenance=source_provenance,
        )
    return prepared, affine


def _multisource_basis(
    native: Co2NativePreparationConfig,
    basis_inputs: Mapping[str, Any],
    site_data: dict[str, xr.Dataset],
    merged_data: MergedData,
    runtime_flux: xr.DataArray,
    *,
    flux_sources: Sequence[str],
    domain: str,
    start_date: str,
) -> FluxWeightedBasis:
    """Give every flux source its own copy, or its own layer, of the basis partition.

    A partition with a ``source`` dimension supplies one layer per source; a
    shared partition, configured or generated from the first source's
    weights, is repeated for each source so that each keeps its own regions.
    """
    sectoral_flux = resolve_runtime_flux(merged_data.flux_data, split_by_sectors=True, explicit_surface=True)
    sources = [str(source) for source in sectoral_flux.indexes["source"]]
    partition = basis_inputs.get("basis_partition")
    if isinstance(partition, xr.DataArray):
        if "source" in partition.dims:
            missing = sorted(set(sources) - {str(source) for source in partition.indexes["source"]})
            if missing:
                raise ValueError(f"The basis partition has no layer for flux source(s) {missing!r}.")
            layers = {source: partition.sel(source=source, drop=True) for source in sources}
        else:
            layers = {source: partition for source in sources}
        return prepare_basis(
            site_data=site_data, runtime_flux=sectoral_flux,
            weighting_flux=merged_data.flux_data[flux_sources[0]]["flux"],
            domain=domain, start_date=start_date, basis_algorithm=native.basis.basis_algorithm,
            nbasis=native.basis.nbasis, fix_outer_regions=native.basis.fix_basis_outer_regions,
            **{**basis_inputs, "basis_partition": layers},
        )
    generated = prepare_basis(
        site_data=site_data, runtime_flux=runtime_flux,
        weighting_flux=merged_data.flux_data[flux_sources[0]]["flux"],
        domain=domain, start_date=start_date, basis_algorithm=native.basis.basis_algorithm,
        nbasis=native.basis.nbasis, fix_outer_regions=native.basis.fix_basis_outer_regions,
        **basis_inputs,
    )
    shared = getattr(generated.operator, "basis_flat", None)
    if not isinstance(shared, xr.DataArray):
        raise ValueError("Per-source native priors need a bucket partition to repeat for each source.")
    return basis_functions_from_flat_basis(
        flux=sectoral_flux, basis_flat={source: shared for source in sources},
        metadata=getattr(generated, "metadata", None),
    )


def _sectoral_response_only(site_data: dict[str, xr.Dataset]) -> dict[str, xr.Dataset]:
    """Drop source-resolved diagnostics so only the projected response owns ``source``.

    The gathered state MultiIndex has a ``source`` level, which cannot coexist
    with a ``source`` dimension that a surface diagnostic such as
    ``mf_mod_sectoral`` would otherwise keep in each site dataset.
    """
    return {
        site: dataset.drop_vars([
            name for name, value in dataset.data_vars.items()
            if "source" in value.dims and name != "fp_x_flux_sectoral"
        ])
        for site, dataset in site_data.items()
    }


def _native_source_sensitivity(
    site_data: dict[str, xr.Dataset], canonical: PreparedInputs, sources: Sequence[str],
) -> xr.DataArray:
    """Gather each retained row's per-source ``fp_x_flux_sectoral`` in canonical order."""
    index = cast(pd.MultiIndex, canonical.inv_inputs.indexes["nmeasure"])
    sites, times = index.get_level_values("site"), index.get_level_values("time")
    parts = []
    for site in pd.unique(sites):
        if "fp_x_flux_sectoral" not in site_data[site]:
            raise ValueError(f"Site {site!r} lacks fp_x_flux_sectoral, which per-source native priors need.")
        parts.append(site_data[site]["fp_x_flux_sectoral"].sel(time=times[sites == site]).reset_coords(drop=True))
    gathered = xr.concat(parts, dim="time", join="exact").sel(source=list(sources))
    gathered = gathered.transpose("time", "source", "lat", "lon")
    if not pd.MultiIndex.from_arrays([np.repeat(pd.unique(sites), [len(p.time) for p in parts]),
                                      gathered.indexes["time"]]).equals(index):
        raise ValueError("Canonical CO2 rows are not grouped by retained site.")
    return gathered.drop_vars("time").rename(time="observation", source=NATIVE_SOURCE_DIM).assign_coords(
        xr.Coordinates.from_pandas_multiindex(index, "observation")
    )


def _source_flux(
    basis: FluxWeightedBasis, operator: MultiSourceBucketBasisOperator, grid: xr.Dataset,
) -> xr.DataArray:
    """Return the window-mean signed prior flux on ``(native_source, lat, lon)``."""
    flux = basis.flux
    if operator.source_dim not in flux.dims:
        raise ValueError("Per-source native priors require source-resolved basis flux.")
    if "time" in flux.dims:
        flux = flux.mean("time", keep_attrs=True)
    flux = flux.rename({operator.source_dim: NATIVE_SOURCE_DIM}).sel({NATIVE_SOURCE_DIM: list(operator.source_labels)})
    return force_align(flux, grid, dims=("lat", "lon")).transpose(NATIVE_SOURCE_DIM, "lat", "lon").fillna(0.0)


def _source_class_labels(labels: xr.DataArray | None, source: str, grid: xr.Dataset) -> xr.DataArray | None:
    """Select one source's class labels on the native grid."""
    if labels is None:
        return None
    if "source" in labels.dims:
        if source not in {str(item) for item in labels.indexes["source"]}:
            raise ValueError(f"Native class_labels have no layer for source {source!r}.")
        labels = labels.sel(source=source, drop=True)
    return force_align(labels, grid, dims=("lat", "lon")).transpose("lat", "lon")


def _source_sigmas(
    prior: Co2MultiSourceNativePrior,
    sources: Sequence[str],
    flux: xr.DataArray,
    classes: Mapping[str, xr.DataArray | None],
    targets: xr.Dataset | None,
    grid: xr.Dataset,
) -> tuple[dict[str, float], dict[str, Any] | None]:
    """Use configured amplitudes and calibrate the others against SD targets."""
    sigmas: dict[str, float] = {}
    record: dict[str, Any] | None = None
    for source in sources:
        configured = prior.sources[source]
        if configured.sigma is not None:
            sigmas[source] = configured.sigma
            continue
        assert prior.calibration is not None and targets is not None
        weights = force_align(targets["functional_weights"], grid, dims=("lat", "lon"))
        name = "target_sd" if "target_sd" in targets else "target_relative_sd"
        target = targets[name]
        if "source" in target.dims:
            if source not in {str(item) for item in target.indexes["source"]}:
                raise ValueError(f"Calibration targets have no {name} for source {source!r}.")
            target = target.sel(source=source, drop=True)
        result = calibrate_native_b_amplitudes(
            flux.sel({NATIVE_SOURCE_DIM: [source]}).rename({NATIVE_SOURCE_DIM: "source"}),
            weights,
            correlation_length=configured.correlation_length,
            latitude_correlation_length=configured.latitude_correlation_length,
            longitude_correlation_length=configured.longitude_correlation_length,
            class_labels=classes[source],
            statistic=cast(Any, prior.calibration.statistic),
            target_sd=target if name == "target_sd" else None,
            target_relative_sd=target if name == "target_relative_sd" else None,
        )
        sigmas[source] = float(result.amplitudes.sel(source=source))
        diagnostics = result.diagnostics.sel(source=source)
        if record is None:
            record = {"statistic": prior.calibration.statistic, "targets": name,
                      "targets_path": str(prior.calibration.targets_path), "sources": {}}
        record["sources"][source] = {
            "sigma": sigmas[source],
            "fit_status": str(diagnostics["fit_status"].values),
            "fit_rms_log_residual": float(diagnostics["fit_rms_log_residual"]),
            "functionals": [str(item) for item in diagnostics.indexes["functional"]],
            "achieved_to_target_ratio": [
                float(item) for item in np.ravel(diagnostics["achieved_to_target_ratio"].values)
            ],
        }
    return sigmas, record


def _source_covariance(
    prior: Co2MultiSourceNativePrior,
    sources: Sequence[str],
    sigmas: Mapping[str, float],
    classes: Mapping[str, xr.DataArray | None],
    latitude: xr.DataArray,
    longitude: xr.DataArray,
) -> CorrelatedSourceCovariance:
    """Build each source's covariance and join GPP and TER under one kernel.

    A correlated GPP/TER pair needs one shared kernel, including one class
    partition: differing class labels are refused here rather than letting one
    source silently use the other's classes.
    """
    if prior.gpp_ter_correlation != 0.0:
        gpp, ter = cast(str, prior.gpp_source), cast(str, prior.ter_source)
        if not _same_class_partition(classes[gpp], classes[ter]):
            raise ValueError(
                f"gpp_ter_correlation requires one shared kernel, but sources {gpp!r} and {ter!r} have "
                "different class labels in class_labels_path; give them the same classes or set rho = 0."
            )
    blocks: dict[str, InvertibleNativeCovarianceAction] = {}
    for source in sources:
        configured = prior.sources[source]
        length = configured.correlation_length
        latitude_length = configured.latitude_correlation_length
        longitude_length = configured.longitude_correlation_length
        geodesic = configured.geodesic_correlation_length_km
        if geodesic is None:
            blocks[source] = SeparableExponentialCovariance(
                latitude, longitude, sigma=sigmas[source], class_labels=classes[source], correlation_length=length,
                latitude_correlation_length=latitude_length, longitude_correlation_length=longitude_length,
            )
            continue
        partner = _shared_kernel_partner(prior, source)
        if (
            partner in blocks
            and isinstance(blocks[partner], ClassMixedExponentialCovariance)
            and _same_labels(classes[source], classes[cast(str, partner)])
            and prior.sources[cast(str, partner)].kernel_key == configured.kernel_key
        ):
            # Share the dense factors of a correlated partner's identical kernel.
            blocks[source] = cast(ClassMixedExponentialCovariance, blocks[partner]).with_sigma(sigmas[source])
            continue
        labels = classes[source]
        if labels is None:
            labels = xr.zeros_like(latitude * longitude, dtype=np.int64).transpose("lat", "lon")
        blocks[source] = ClassMixedExponentialCovariance(
            latitude, longitude, class_labels=labels, sigma=sigmas[source],
            class_kernels=_geodesic_kernels(geodesic, labels, source), correlation_length=length,
            latitude_correlation_length=latitude_length, longitude_correlation_length=longitude_length,
        )
    correlations = None
    if prior.gpp_ter_correlation != 0.0:
        correlations = {(cast(str, prior.gpp_source), cast(str, prior.ter_source)): prior.gpp_ter_correlation}
    return CorrelatedSourceCovariance(blocks, correlations, source_dim=NATIVE_SOURCE_DIM)


def _same_labels(first: xr.DataArray | None, second: xr.DataArray | None) -> bool:
    """Return whether two sources' class labels are the same array (or both absent)."""
    if first is None or second is None:
        return first is None and second is None
    return bool(np.array_equal(first.transpose("lat", "lon").values, second.transpose("lat", "lon").values))


def _same_class_partition(first: xr.DataArray | None, second: xr.DataArray | None) -> bool:
    """Return whether two sources' class labels split the grid into the same classes."""
    def partition(labels: xr.DataArray | None) -> np.ndarray | None:
        if labels is None:
            return None
        _, inverse = np.unique(labels.transpose("lat", "lon").values.ravel(), return_inverse=True)
        # Number classes by first appearance so equal partitions compare equal whatever their label values.
        order = {value: index for index, value in enumerate(dict.fromkeys(inverse.tolist()))}
        return np.array([order[value] for value in inverse.tolist()])

    a, b = partition(first), partition(second)
    if a is None or b is None:
        single = [p for p in (a, b) if p is not None]
        return all(int(p.max()) == 0 for p in single)
    return bool(np.array_equal(a, b))


def _shared_kernel_partner(prior: Co2MultiSourceNativePrior, source: str) -> str | None:
    """Return the correlated GPP/TER partner of ``source``, if any."""
    if prior.gpp_ter_correlation == 0.0:
        return None
    pair = (prior.gpp_source, prior.ter_source)
    if source not in pair:
        return None
    return pair[1] if source == pair[0] else pair[0]


def _geodesic_kernels(
    lengths: float | Mapping[str, float], labels: xr.DataArray, source: str,
) -> dict[object, GeodesicExponentialKernel]:
    """Resolve configured class keys against the labels' own values."""
    present = {str(label): label.item() for label in np.unique(labels.values)}
    if not isinstance(lengths, Mapping):
        return {label: GeodesicExponentialKernel(float(lengths)) for label in present.values()}
    missing = sorted(set(lengths) - set(present))
    if missing:
        raise ValueError(f"Geodesic classes {missing!r} of source {source!r} are absent from its class labels.")
    return {present[key]: GeodesicExponentialKernel(float(length)) for key, length in lengths.items()}


def _flux_weighted_restriction(
    operator: MultiSourceBucketBasisOperator,
    flux: xr.DataArray,
    retained: Mapping[str, tuple[int, ...]] | None,
    grid: xr.Dataset,
) -> xr.DataArray:
    """Build ``Pi`` from ``|flux| * cell area`` regional means on the basis's own state labels.

    Regions outside ``retained`` are inactive: they get no retained state and
    are marginalized into the unresolved covariance.
    """
    state_dim = operator.meta.state_dim
    states = operator.basis_matrix.indexes[state_dim]
    regions: dict[str, xr.DataArray] = {}
    labels: list[tuple[str, object]] = []
    for source in operator.source_labels:
        partition = force_align(operator.basis_flat[source], grid, dims=("lat", "lon")).transpose("lat", "lon")
        values = np.asarray(partition.values).astype(np.int64)
        present = [int(label) for label in np.unique(values)]
        # The bucket operator numbers each source's regions in sorted label order.
        source_states = [label for state_source, label in states if state_source == source]
        if len(source_states) != len(present) or min(present) < 0:
            raise ValueError(f"The {source!r} basis partition does not match its bucket states.")
        keep = present if retained is None or source not in retained else list(retained[source])
        unknown = sorted(set(keep) - set(present))
        if unknown:
            raise ValueError(f"Retained regions {unknown!r} are absent from the {source!r} basis partition.")
        regions[source] = partition.copy(data=np.where(np.isin(values, keep), values, -1))
        state_of = dict(zip(present, source_states, strict=True))
        labels.extend((source, state_of[label]) for label in keep)
    if retained is not None:
        extra = sorted(set(retained) - set(operator.source_labels))
        if extra:
            raise ValueError(f"retained_regions_path names unknown source(s) {extra!r}.")
    area = xr.DataArray(
        areagrid(grid["lat"].values, grid["lon"].values), dims=("lat", "lon"),
        coords={"lat": grid["lat"], "lon": grid["lon"]},
    )
    weights = flux * area
    empty: dict[str, list[int]] = {}
    for source, source_regions in regions.items():
        labels_flat = np.asarray(source_regions.values).ravel()
        magnitude = np.abs(np.asarray(weights.sel({NATIVE_SOURCE_DIM: source}).transpose("lat", "lon").values)).ravel()
        active = labels_flat >= 0
        present, inverse = np.unique(labels_flat[active], return_inverse=True)
        totals = np.bincount(inverse, weights=magnitude[active], minlength=present.size)
        zero = [int(label) for label, total in zip(present, totals, strict=True) if not total > 0.0]
        if zero:
            empty[source] = zero
    if empty:
        raise ValueError(
            "The absolute-flux-weighted restriction needs prior flux in every retained region, but these "
            f"(source: partition labels) have zero window-mean |flux|: {empty!r}. A shared or generated "
            "partition repeated per source often has such regions (for example biosphere over sea, ocean "
            "over land); leave them out with retained_regions_path to marginalize them, or use a "
            "source-specific partition or restriction = 'bucket'."
        )
    restriction = absolute_weighted_regional_mean_restriction(
        regions, weights, state_dim=state_dim, native_source_dim=NATIVE_SOURCE_DIM,
        source_dim=operator.source_dim, region_dim=operator.region_in_source_dim,
    )
    # Active regions are numbered in ascending label order within each source,
    # so the retained bucket-state labels replace those positions one for one.
    index = pd.MultiIndex.from_tuples(labels, names=(operator.source_dim, operator.region_in_source_dim))
    restriction = restriction.drop_vars([state_dim, operator.source_dim, operator.region_in_source_dim])
    return restriction.assign_coords(xr.Coordinates.from_pandas_multiindex(index, state_dim)).assign_attrs(
        definition="|flux_si| * area_i / sum_(j in r) |flux_sj| * area_j within one source and region",
        policy="absolute-flux-weighted regional mean",
    )


def _retained_canonical(canonical: PreparedInputs, retained: pd.Index, state_dim: str) -> PreparedInputs:
    """Keep only the retained states of canonical inputs, in the reduction's order."""
    inputs = canonical.inv_inputs
    available = inputs.indexes[state_dim]
    missing = [label for label in retained if label not in available]
    if missing:
        raise ValueError(f"Retained states {missing[:3]!r} are absent from the canonical inputs.")
    selection = {state_dim: available.get_indexer(retained)}
    covariance_dim = f"{state_dim}_cov"
    if covariance_dim in inputs.dims:
        selection[covariance_dim] = inputs.indexes[covariance_dim].get_indexer(retained)
    return PreparedInputs(
        inv_inputs=inputs.isel(selection),
        basis_functions=canonical.basis_functions,
        site_metadata=canonical.site_metadata,
    )


def _multisource_record(
    prior: Co2MultiSourceNativePrior,
    sources: Sequence[str],
    sigmas: Mapping[str, float],
    calibration: Mapping[str, Any] | None,
    classes: Mapping[str, xr.DataArray | None],
    covariance: CorrelatedSourceCovariance,
    reduction: Any,
    state_dim: str,
) -> dict[str, Any]:
    """Return the JSON-safe record of the native prior that defined the reduction."""
    retained = reduction.retained_mean.indexes[state_dim].get_level_values(0)
    sources_record: dict[str, Any] = {}
    for source in sources:
        configured: Co2NativeSourcePrior = prior.sources[source]
        block = covariance.source_covariances[source]
        geodesic = configured.geodesic_correlation_length_km
        labels = classes[source]
        sources_record[source] = {
            "mean": configured.mean,
            "sigma": sigmas[source],
            "sigma_origin": "configured" if configured.sigma is not None else "calibrated",
            "kernel": "separable_exponential" if geodesic is None else "class_mixed_geodesic_exponential",
            "latitude_correlation_length_deg": float(getattr(block, "latitude_correlation_length")),
            "longitude_correlation_length_deg": float(getattr(block, "longitude_correlation_length")),
            "geodesic_correlation_length_km": (
                dict(geodesic) if isinstance(geodesic, Mapping) else geodesic
            ),
            "class_labels": labels is not None,
            "classes": int(np.unique(labels.values).size) if labels is not None else 1,
            "retained_states": int((retained == source).sum()),
        }
    return {
        "native_prior": prior.to_params(),
        "native_source_priors": {
            "type": type(covariance).__name__,
            "source_dim": NATIVE_SOURCE_DIM,
            "source_order": list(sources),
            "gpp_ter_correlation": prior.gpp_ter_correlation,
            "gpp_source": prior.gpp_source,
            "ter_source": prior.ter_source,
            "sign_convention": "Corr(x_GPP, x_TER) of native scaling anomalies",
            "sources": sources_record,
        },
        "restriction": prior.restriction,
        "calibration": dict(calibration) if calibration is not None else None,
    }


def _load_centering_weights(path: Path) -> xr.DataArray:
    """Load and check the one ``bc_centering_weights`` array before acquisition starts."""
    with xr.open_dataset(path) as dataset:
        if "bc_centering_weights" not in dataset:
            raise ValueError(f"{path} must contain a bc_centering_weights variable.")
        weights = dataset["bc_centering_weights"].load()
    validate_centering_weights(weights.transpose("bc_curtain", "bc_period"))
    return weights


__all__ = [
    "NATIVE_SOURCE_DIM",
    "MultiSourceNativeResources",
    "prepare_multisource_native_co2_inputs",
    "prepare_native_co2",
]
