"""TOML loading and semantic configuration for the concrete CO2 recipes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from importlib.resources import files
from importlib.resources.abc import Traversable
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import tomllib


from openghg_inversions.basis.boundary import BOUNDARY_BASIS_ALGORITHMS
from openghg_inversions.boundary_sensitivity import BoundaryRepresentation
from openghg_inversions.inversion_data import SiteOptions, check_surface_request
from openghg_inversions.models.boundary import (
    AdditiveChoice, AnomalyScalingChoice, Choice, MultiplicativeChoice, boundary_preparation,
)
from openghg_inversions.models.boundary.preparation import check_representation
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.models.priors import positive_prior_args
from openghg_inversions.models.scalar_sigma import add_scalar_sigma_eigen_likelihood
from openghg_inversions.models.site_sigma import add_site_sigma_gaussian_likelihood
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.rhime.specs import DEFAULT_BC_PRIOR, normalise_optional_mapping

from .co2_cached_sigma_runner import validate_cached_co2_sampler
from .inference import uses_cached_ou


def _frozen(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {key: _frozen(value) if isinstance(value, Mapping) else value for key, value in values.items()}
    )


def _table(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a table.")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} keys must be strings.")
    return dict(value)


def _take(options: dict[str, object], key: str, path: str) -> object:
    try:
        return options.pop(key)
    except KeyError as exc:
        raise ValueError(f"Missing required option {path}.{key}.") from exc


def _reject_unknown(options: Mapping[str, object], path: str) -> None:
    if options:
        names = ", ".join(f"{path}.{name}" for name in sorted(options))
        raise ValueError(f"Unknown configuration option(s): {names}.")



# Names are checked for every phase; their values remain owned by the consuming parser.
_CO2_REQUEST_OPTION_NAMES = frozenset({
    "format_version", "recipe", "prepared_inputs", "model", "likelihood", "sampling", "outputs",
})
# A saved preparation is imported with ``path``; a new one is built from the
# remaining tables, each parsed by its own owner below.
_NATIVE_PREPARATION_OPTION_NAMES = frozenset({
    "acquisition", "observations", "basis", "native_prior", "aggregation_error_rank", "bc_centering_weights_path",
})
_PREPARATION_OPTION_NAMES = frozenset({
    "path", "reconstruction_path", "start_date", "end_date", "domain", "boundary_representation",
}) | _NATIVE_PREPARATION_OPTION_NAMES
_SITE_OPTION_NAMES = frozenset({
    "sites", "averaging_period", "inlet", "fp_height", "instrument", "platform", "obs_data_level",
    "met_model", "max_level", "time_resolved",
})
_ACQUISITION_STRING_NAMES = frozenset({
    "bc_store", "obs_store", "footprint_store", "flux_store", "flux_domain", "fp_model", "fp_species",
    "calibration_scale", "bc_input", "flux_non_finite_check", "boundary_reference_units",
})
_ACQUISITION_OPTION_NAMES = _SITE_OPTION_NAMES | _ACQUISITION_STRING_NAMES | {
    "flux_sources", "surface_transport", "use_bc", "boundary_reference", "boundary_mass_tolerance",
}
_BASIS_OPTION_NAMES = frozenset({
    "algorithm", "nbasis", "path", "partition_path", "landsea_path", "outer_regions_path",
    "fix_outer_regions", "bc_algorithm", "bc_split_height", "bc_paths",
})
_NATIVE_PRIOR_OPTION_NAMES = frozenset({
    "mean", "sigma", "correlation_length", "latitude_correlation_length", "longitude_correlation_length",
})
# A ``sources`` table selects separate native fields per flux source.
_MULTISOURCE_NATIVE_PRIOR_OPTION_NAMES = frozenset({
    "sources", "gpp_ter_correlation", "gpp_source", "ter_source", "restriction", "class_labels_path",
    "retained_regions_path", "calibration",
})
_NATIVE_SOURCE_PRIOR_OPTION_NAMES = _NATIVE_PRIOR_OPTION_NAMES | {"geodesic_correlation_length_km"}
_NATIVE_CALIBRATION_OPTION_NAMES = frozenset({"targets_path", "statistic"})
NATIVE_RESTRICTIONS = ("bucket", "absolute_flux_weighted_mean")
_SAMPLING_OPTION_NAMES = frozenset({
    "draws", "burn", "tune", "chains", "nuts_sampler", "progressbar", "sample_prior_predictive",
    "sample_posterior_predictive", "target_accept", "random_seed",
})
_LIKELIHOOD_OPTION_NAMES = frozenset({
    "kind", "no_model_error", "sigma_prior", "fixed_model_mismatch", "fixed_site_amplitudes",
    "site_amplitude_prior", "tau_hours", "initial_site_amplitudes", "eigenbasis_path",
})
_PRIOR_OPTION_NAMES = frozenset({
    "pdf", "mu", "sigma", "lower", "upper", "nu", "alpha", "beta", "lam", "mean", "stdev", "reparameterise",
})
_MODEL_SECTION_OPTION_NAMES = {
    "boundary": frozenset({
        "enabled", "type", "prior", "frequency", "anchor_time", "mean_shift_prior", "anomaly_scale", "anomalies",
        "mean_by_period", "scale_sd", "reference",
    }),
    "boundary.mean_shift_prior": _PRIOR_OPTION_NAMES,
    "boundary.prior": _PRIOR_OPTION_NAMES,
    "offset": frozenset({"prior", "frequency", "per_site", "drop_first"}),
    "offset.prior": _PRIOR_OPTION_NAMES,
    "biosphere": frozenset({"path"}),
}


def _validate_section_option_names(
    requests: Iterable[Mapping[str, object]], sections: Mapping[str, frozenset[str]],
) -> None:
    """Check source/override names once without resolving any section values."""
    unknown: dict[str, object] = {}
    for config in requests:
        for path, allowed in sections.items():
            section: object = config
            for part in path.split(".") if path else ():
                section = section.get(part) if isinstance(section, Mapping) else None
            if isinstance(section, Mapping):
                prefix = f"{path}." if path else ""
                unknown.update({f"{prefix}{name}": value for name, value in section.items() if name not in allowed})
    _reject_unknown(unknown, "config")


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{path} must be a non-empty string.")
    return value.strip()


def _bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean.")
    return value


def _number(value: object, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{path} must be numeric.")
    result = float(value)
    if not isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{path} must be {qualifier}.")
    return result


def _number_or_mapping(
    value: object,
    path: str,
    *,
    positive: bool,
    nonnegative: bool = False,
) -> float | Mapping[str, float]:
    def checked(item: object, item_path: str) -> float:
        result = _number(item, item_path, positive=positive)
        if nonnegative and result < 0.0:
            raise ValueError(f"{item_path} must be finite and non-negative.")
        return result

    if isinstance(value, Mapping):
        table = _table(value, path)
        if not table:
            raise ValueError(f"{path} must not be empty.")
        return MappingProxyType(
            {_string(key, f"{path} key"): checked(item, f"{path}.{key}") for key, item in table.items()}
        )
    return checked(value, path)


def _prior(value: object, path: str, *, positive: bool = False) -> Mapping[str, Any]:
    prior = _table(value, path)
    pdf = _string(_take(prior, "pdf", path), f"{path}.pdf").casefold().replace("-", "")
    required = {
        "normal": ("mu", "sigma"),
        "truncatednormal": ("mu", "sigma", "lower"),
        "halfnormal": ("sigma",),
        "halfstudentt": ("nu", "sigma"),
        "gamma": ("alpha", "beta"),
        "exponential": ("lam",),
        "uniform": ("lower", "upper"),
    }
    result: dict[str, Any] = {"pdf": pdf}
    if pdf == "lognormal":
        parameter_keys = {"mu", "sigma"}
        moment_keys = {"mean", "stdev"}
        if parameter_keys & prior.keys() and moment_keys & prior.keys():
            raise ValueError(f"{path} must use either mu/sigma or mean/stdev for a LogNormal prior.")
        names = ("mean", "stdev") if moment_keys & prior.keys() else ("mu", "sigma")
        for name in names:
            result[name] = _number(
                _take(prior, name, path),
                f"{path}.{name}",
                positive=name in {"mean", "stdev", "sigma"},
            )
        if "reparameterise" in prior:
            result["reparameterise"] = _bool(prior.pop("reparameterise"), f"{path}.reparameterise")
    elif pdf not in required:
        raise ValueError(f"{path}.pdf is not a supported configuration prior.")
    else:
        for name in required[pdf]:
            result[name] = _number(
                _take(prior, name, path),
                f"{path}.{name}",
                positive=name in {"sigma", "nu", "alpha", "beta", "lam"},
            )
    if pdf == "truncatednormal" and "upper" in prior:
        result["upper"] = _number(prior.pop("upper"), f"{path}.upper")
    _reject_unknown(prior, path)
    if pdf in {"uniform", "truncatednormal"} and "upper" in result:
        if result["lower"] >= result["upper"]:
            raise ValueError(f"{path}.lower must be less than {path}.upper.")
    if positive:
        result = positive_prior_args(result)
    return _frozen(result)


def validate_co2_config(config: Mapping[str, object]) -> dict[str, Any]:
    """Validate the CO2 file envelope without resolving its sections.

    Missing headers select CO2 for phase-only requests. Component
    parsers own scientific values and resource paths. The request loader checks
    section names; this envelope check leaves inactive sections unresolved.
    """
    options = _table(config, "config")
    version = options.get("format_version", 1)
    if type(version) is not int or version != 1:
        raise ValueError("config.format_version must be the integer 1.")
    if options.get("recipe", "co2") != "co2":
        raise ValueError("config.recipe must be 'co2' for CO2 execution.")
    _reject_unknown({name: value for name, value in options.items() if name not in _CO2_REQUEST_OPTION_NAMES}, "config")
    return options


def _validate_co2_option_names(*requests: Mapping[str, object]) -> None:
    """Check request names independently of header values and winning overrides."""
    from .co2_outputs import Co2OutputOptions
    from openghg_inversions.postprocessing.paris_options import ParisOptions

    _validate_section_option_names(requests, {
        "": _CO2_REQUEST_OPTION_NAMES,
        "prepared_inputs": _PREPARATION_OPTION_NAMES,
        "prepared_inputs.acquisition": _ACQUISITION_OPTION_NAMES,
        "prepared_inputs.observations": frozenset({"filters", "averaging_error"}),
        "prepared_inputs.basis": _BASIS_OPTION_NAMES,
        "prepared_inputs.native_prior": _NATIVE_PRIOR_OPTION_NAMES | _MULTISOURCE_NATIVE_PRIOR_OPTION_NAMES,
        "prepared_inputs.native_prior.calibration": _NATIVE_CALIBRATION_OPTION_NAMES,
        "model": frozenset({"boundary", "offset", "biosphere"}),
        **{f"model.{path}": names for path, names in _MODEL_SECTION_OPTION_NAMES.items()},
        "likelihood": _LIKELIHOOD_OPTION_NAMES,
        "likelihood.sigma_prior": _PRIOR_OPTION_NAMES,
        "likelihood.site_amplitude_prior": _PRIOR_OPTION_NAMES,
        "sampling": _SAMPLING_OPTION_NAMES | {"method", "sigma_target_accept", "state_target_accept"},
        "outputs": frozenset(Co2OutputOptions.__dataclass_fields__) | {"source_to_sector"},
        "outputs.paris_postprocessing_kwargs": frozenset(ParisOptions.__dataclass_fields__),
    })


def _path(value: object, path: str) -> Path:
    """Resolve one configured filesystem path from the invocation directory."""
    if not isinstance(value, str | Path):
        raise TypeError(f"{path} must be a filesystem path.")
    if isinstance(value, str):
        value = _string(value, path)
    return Path(value).expanduser().resolve()


def _positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer.")
    return value


@dataclass(frozen=True)
class Co2BasisSelection:
    """Flux and boundary basis choices for a new CO2 preparation.

    Attribute names match the basis resolvers in
    :mod:`openghg_inversions.rhime.preparation`, which read them directly.
    Configured paths resolve from the invocation directory.
    """

    domain: str
    use_bc: bool = True
    basis_algorithm: str | None = "weighted"
    nbasis: int = 100
    basis_path: Path | None = None
    basis_partition_path: Path | None = None
    basis_partition_paths: tuple[Path, ...] | None = None
    landsea_path: Path | None = None
    outer_regions_path: Path | None = None
    fix_basis_outer_regions: bool = False
    bc_basis_paths: tuple[Path, ...] | None = None
    bc_basis_algorithm: str = "NESW"
    bc_basis_split_height: float = 3000.0

    @classmethod
    def from_params(cls, value: object, *, domain: str, use_bc: bool) -> Co2BasisSelection:
        """Parse ``[prepared_inputs.basis]`` for the acquired domain."""
        options = _table(value, "prepared_inputs.basis")
        result: dict[str, Any] = {"domain": domain, "use_bc": use_bc}
        for name, field in (("algorithm", "basis_algorithm"), ("bc_algorithm", "bc_basis_algorithm")):
            if name in options:
                result[field] = _string(options.pop(name), f"prepared_inputs.basis.{name}")
        if "nbasis" in options:
            result["nbasis"] = _positive_int(options.pop("nbasis"), "prepared_inputs.basis.nbasis")
        for name, field in (("path", "basis_path"), ("partition_path", "basis_partition_path"),
                            ("landsea_path", "landsea_path"), ("outer_regions_path", "outer_regions_path")):
            if name in options:
                result[field] = _path(options.pop(name), f"prepared_inputs.basis.{name}")
        if "fix_outer_regions" in options:
            result["fix_basis_outer_regions"] = _bool(
                options.pop("fix_outer_regions"), "prepared_inputs.basis.fix_outer_regions",
            )
        if "bc_split_height" in options:
            result["bc_basis_split_height"] = _number(
                options.pop("bc_split_height"), "prepared_inputs.basis.bc_split_height",
            )
        if "bc_paths" in options:
            paths = options.pop("bc_paths")
            if isinstance(paths, str | bytes) or not isinstance(paths, Sequence) or not paths:
                raise TypeError("prepared_inputs.basis.bc_paths must be a non-empty list of paths.")
            result["bc_basis_paths"] = tuple(
                _path(item, "prepared_inputs.basis.bc_paths item") for item in paths
            )
        _reject_unknown(options, "prepared_inputs.basis")
        selected = cls(**result)
        # region_constrained also needs a configured class map, which waits on #454.
        algorithms = ("quadtree", "weighted")
        if selected.basis_path is None and selected.basis_partition_path is None and (
            selected.basis_algorithm not in algorithms
        ):
            raise ValueError(f"prepared_inputs.basis.algorithm must be one of {algorithms!r}.")
        if selected.bc_basis_algorithm not in BOUNDARY_BASIS_ALGORITHMS:
            raise ValueError(
                f"prepared_inputs.basis.bc_algorithm must be one of {BOUNDARY_BASIS_ALGORITHMS!r}."
            )
        return selected


@dataclass(frozen=True)
class Co2NativePrior:
    """Native Gaussian prior on dimensionless flux scaling, before reduction.

    The prior has a constant mean and a separable exponential covariance over
    the native latitude/longitude grid, with lengths in degrees.
    """

    sigma: float
    mean: float = 1.0
    correlation_length: float = 1.5
    latitude_correlation_length: float | None = None
    longitude_correlation_length: float | None = None

    @classmethod
    def from_params(cls, value: object) -> Co2NativePrior:
        """Parse ``[prepared_inputs.native_prior]``."""
        options = _table(value, "prepared_inputs.native_prior")
        path = "prepared_inputs.native_prior"
        result: dict[str, Any] = {"sigma": _number(_take(options, "sigma", path), f"{path}.sigma", positive=True)}
        if "mean" in options:
            result["mean"] = _number(options.pop("mean"), f"{path}.mean")
        for name in ("correlation_length", "latitude_correlation_length", "longitude_correlation_length"):
            if name in options:
                result[name] = _number(options.pop(name), f"{path}.{name}", positive=True)
        _reject_unknown(options, path)
        return cls(**result)


@dataclass(frozen=True)
class Co2NativeSourcePrior:
    """Native Gaussian prior of one flux source's dimensionless scaling field.

    The field has the constant ``mean`` and covariance ``sigma**2 K``. ``K``
    is the separable exponential kernel with lengths in degrees, blocked by
    class when class labels are configured. Classes named in
    ``geodesic_correlation_length_km`` use a dense great-circle exponential
    kernel with that length in km instead; a single number makes every class
    geodesic. Without ``sigma`` the amplitude is calibrated against the
    configured functional SD targets.

    Attributes:
        sigma: Native grid-cell standard deviation, or ``None`` to calibrate.
        mean: Constant native scaling mean.
        correlation_length: Separable fallback length in degrees.
        latitude_correlation_length: Optional separable latitude length.
        longitude_correlation_length: Optional separable longitude length.
        geodesic_correlation_length_km: Geodesic length for every class, or a
            mapping from class label (as written in TOML) to length.
    """

    sigma: float | None = None
    mean: float = 1.0
    correlation_length: float = 1.5
    latitude_correlation_length: float | None = None
    longitude_correlation_length: float | None = None
    geodesic_correlation_length_km: float | Mapping[str, float] | None = None

    @classmethod
    def from_params(cls, value: object, path: str) -> Co2NativeSourcePrior:
        """Parse one ``[prepared_inputs.native_prior.sources.<source>]`` table."""
        options = _table(value, path)
        result: dict[str, Any] = {}
        if "sigma" in options:
            result["sigma"] = _number(options.pop("sigma"), f"{path}.sigma", positive=True)
        if "mean" in options:
            result["mean"] = _number(options.pop("mean"), f"{path}.mean")
        for name in ("correlation_length", "latitude_correlation_length", "longitude_correlation_length"):
            if name in options:
                result[name] = _number(options.pop(name), f"{path}.{name}", positive=True)
        if "geodesic_correlation_length_km" in options:
            result["geodesic_correlation_length_km"] = _number_or_mapping(
                options.pop("geodesic_correlation_length_km"), f"{path}.geodesic_correlation_length_km",
                positive=True,
            )
        _reject_unknown(options, path)
        return cls(**result)

    def to_params(self) -> dict[str, Any]:
        """Return the table :meth:`from_params` reads back, omitting unset options."""
        result: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if value is not None:
                result[item.name] = dict(value) if isinstance(value, Mapping) else value
        return result

    @property
    def kernel_key(self) -> tuple[object, ...]:
        """Resolved unit-kernel choice; sources that share a kernel have equal keys."""
        geodesic = self.geodesic_correlation_length_km
        return (
            self.latitude_correlation_length or self.correlation_length,
            self.longitude_correlation_length or self.correlation_length,
            tuple(sorted(geodesic.items())) if isinstance(geodesic, Mapping) else geodesic,
        )


@dataclass(frozen=True)
class Co2NativeBCalibration:
    """Calibrate unset source amplitudes against functional SD targets.

    The targets file holds ``functional_weights`` on ``(functional, lat, lon)``
    (for example a country mask times cell area) and either ``target_sd`` in
    the units of ``flux * functional_weights`` or dimensionless
    ``target_relative_sd``, on ``(source, functional)`` or ``(functional,)``.
    Each calibrated source gets one amplitude
    (:func:`~openghg_inversions.native_b_calibration.calibrate_native_b_amplitudes`)
    using its own separable kernel, class labels and the window-mean prior flux.

    Attributes:
        targets_path: NetCDF file with the functionals and their targets.
        statistic: ``"geometric-mean"`` (the A1-style anchor) or ``"median"``.
    """

    targets_path: Path
    statistic: str = "geometric-mean"

    @classmethod
    def from_params(cls, value: object) -> Co2NativeBCalibration:
        """Parse ``[prepared_inputs.native_prior.calibration]``."""
        path = "prepared_inputs.native_prior.calibration"
        options = _table(value, path)
        result: dict[str, Any] = {
            "targets_path": _path(_take(options, "targets_path", path), f"{path}.targets_path"),
        }
        if "statistic" in options:
            statistic = _string(options.pop("statistic"), f"{path}.statistic")
            if statistic not in ("geometric-mean", "median"):
                raise ValueError(f"{path}.statistic must be 'geometric-mean' or 'median'.")
            result["statistic"] = statistic
        _reject_unknown(options, path)
        return cls(**result)

    def to_params(self) -> dict[str, Any]:
        """Return the table :meth:`from_params` reads back."""
        return {"targets_path": str(self.targets_path), "statistic": self.statistic}


@dataclass(frozen=True)
class Co2MultiSourceNativePrior:
    """Separate native Gaussian fields per flux source, reduced jointly.

    Each source has its own scaling field and covariance
    (:class:`Co2NativeSourcePrior`). Sources are independent except GPP and
    TER, whose scaling anomalies have correlation ``gpp_ter_correlation``
    under one shared kernel
    (:class:`~openghg_inversions.correlated_source_covariance.CorrelatedSourceCovariance`).
    The retained state is (source, region): every source keeps its own
    regions of the basis partition.

    ``restriction`` selects the retained coordinates: ``"bucket"`` keeps the
    basis buckets as the conditional-mean lift (the single-field choice), and
    ``"absolute_flux_weighted_mean"`` uses regional means weighted by
    ``|window-mean flux| * cell area``, with the covariance-natural lift as
    affine companion. Only the latter may restrict the retained regions with
    ``retained_regions_path``; other regions are marginalized into the
    aggregation error.

    Attributes:
        sources: Per-source priors keyed by acquisition flux source.
        gpp_ter_correlation: Correlation ``rho`` of the GPP and TER native
            scaling anomalies, with ``|rho| < 1``.
        gpp_source: GPP flux source label, required when ``rho != 0``.
        ter_source: TER flux source label, required when ``rho != 0``.
        restriction: Retained-coordinate policy, one of
            :data:`NATIVE_RESTRICTIONS`.
        class_labels_path: Optional NetCDF with integer ``class_labels`` on
            ``(lat, lon)`` or ``(source, lat, lon)``; classes are uncorrelated.
        retained_regions_path: Optional JSON mapping each source to the basis
            region labels it retains.
        calibration: Optional amplitude calibration for sources without
            ``sigma``.
    """

    sources: Mapping[str, Co2NativeSourcePrior]
    gpp_ter_correlation: float = 0.0
    gpp_source: str | None = None
    ter_source: str | None = None
    restriction: str = "bucket"
    class_labels_path: Path | None = None
    retained_regions_path: Path | None = None
    calibration: Co2NativeBCalibration | None = None

    @classmethod
    def from_params(cls, value: object) -> Co2MultiSourceNativePrior:
        """Parse ``[prepared_inputs.native_prior]`` with a ``sources`` table."""
        path = "prepared_inputs.native_prior"
        options = _table(value, path)
        shared = sorted(_NATIVE_PRIOR_OPTION_NAMES & set(options))
        if shared:
            raise ValueError(
                f"{path}.{shared[0]} belongs to the single-field prior; with {path}.sources give it per "
                f"source under {path}.sources.<source>."
            )
        sources = _table(_take(options, "sources", path), f"{path}.sources")
        if not sources:
            raise ValueError(f"{path}.sources must name at least one flux source.")
        result: dict[str, Any] = {
            "sources": MappingProxyType({
                _string(name, f"{path}.sources key"): Co2NativeSourcePrior.from_params(
                    item, f"{path}.sources.{name}",
                )
                for name, item in sources.items()
            }),
        }
        if "gpp_ter_correlation" in options:
            rho = _number(options.pop("gpp_ter_correlation"), f"{path}.gpp_ter_correlation")
            if abs(rho) >= 1.0:
                raise ValueError(f"{path}.gpp_ter_correlation must satisfy |rho| < 1.")
            result["gpp_ter_correlation"] = rho
        for name in ("gpp_source", "ter_source"):
            if name in options:
                result[name] = _string(options.pop(name), f"{path}.{name}")
        if "restriction" in options:
            restriction = _string(options.pop("restriction"), f"{path}.restriction")
            if restriction not in NATIVE_RESTRICTIONS:
                raise ValueError(f"{path}.restriction must be one of {NATIVE_RESTRICTIONS!r}.")
            result["restriction"] = restriction
        for name in ("class_labels_path", "retained_regions_path"):
            if name in options:
                result[name] = _path(options.pop(name), f"{path}.{name}")
        if "calibration" in options:
            result["calibration"] = Co2NativeBCalibration.from_params(options.pop("calibration"))
        _reject_unknown(options, path)
        prior = cls(**result)
        prior._check(path)
        return prior

    def _check(self, path: str) -> None:
        """Check cross-option rules that need the whole table."""
        for name in ("gpp_source", "ter_source"):
            label = getattr(self, name)
            if label is not None and label not in self.sources:
                raise ValueError(f"{path}.{name} {label!r} is not one of {path}.sources {tuple(self.sources)!r}.")
        if self.gpp_ter_correlation != 0.0:
            if self.gpp_source is None or self.ter_source is None:
                raise ValueError(f"{path}.gpp_ter_correlation requires {path}.gpp_source and {path}.ter_source.")
            if self.gpp_source == self.ter_source:
                raise ValueError(f"{path}.gpp_source and {path}.ter_source must differ.")
            if self.sources[self.gpp_source].kernel_key != self.sources[self.ter_source].kernel_key:
                raise ValueError(
                    f"{path}.gpp_ter_correlation requires one shared kernel: give "
                    f"{path}.sources.{self.gpp_source} and {path}.sources.{self.ter_source} the same "
                    "correlation lengths and geodesic classes."
                )
        uncalibrated = [name for name, source in self.sources.items() if source.sigma is None]
        if self.calibration is None and uncalibrated:
            raise ValueError(
                f"{path}.sources.{uncalibrated[0]}.sigma is required without {path}.calibration."
            )
        if self.calibration is not None and not uncalibrated:
            raise ValueError(f"{path}.calibration has no source to calibrate; every source sets sigma.")
        for name in uncalibrated:
            if self.sources[name].geodesic_correlation_length_km is not None:
                raise ValueError(
                    f"{path}.sources.{name}: calibration supports separable kernels only; set sigma for a "
                    "source with geodesic classes."
                )
        if self.class_labels_path is None:
            for name, source in self.sources.items():
                if isinstance(source.geodesic_correlation_length_km, Mapping):
                    raise ValueError(
                        f"{path}.sources.{name}.geodesic_correlation_length_km names classes, which requires "
                        f"{path}.class_labels_path."
                    )
        if self.retained_regions_path is not None and self.restriction != "absolute_flux_weighted_mean":
            raise ValueError(
                f"{path}.retained_regions_path requires restriction = 'absolute_flux_weighted_mean'."
            )

    def to_params(self) -> dict[str, Any]:
        """Return the table :meth:`from_params` reads back, omitting defaults left unset."""
        result: dict[str, Any] = {"sources": {name: source.to_params() for name, source in self.sources.items()}}
        if self.gpp_ter_correlation != 0.0:
            result["gpp_ter_correlation"] = self.gpp_ter_correlation
        for name in ("gpp_source", "ter_source"):
            if getattr(self, name) is not None:
                result[name] = getattr(self, name)
        result["restriction"] = self.restriction
        for name in ("class_labels_path", "retained_regions_path"):
            if getattr(self, name) is not None:
                result[name] = str(getattr(self, name))
        if self.calibration is not None:
            result["calibration"] = self.calibration.to_params()
        return result


def _native_prior(value: object) -> Co2NativePrior | Co2MultiSourceNativePrior:
    """Select the single shared field or separate per-source fields."""
    options = _table(value, "prepared_inputs.native_prior")
    if "sources" in options:
        return Co2MultiSourceNativePrior.from_params(options)
    multisource = sorted(_MULTISOURCE_NATIVE_PRIOR_OPTION_NAMES & set(options))
    if multisource:
        raise ValueError(
            f"prepared_inputs.native_prior.{multisource[0]} requires per-source priors in "
            "prepared_inputs.native_prior.sources."
        )
    return Co2NativePrior.from_params(options)


@dataclass(frozen=True)
class Co2NativePreparationConfig:
    """Acquisition and native preparation choices for a new CO2 preparation.

    Attributes:
        acquisition: Keywords for
            :meth:`~openghg_inversions.inversion_data.MergedData.from_options`,
            other than species, domain and window. Surface transport is
            required, so boundaries keep centered operands for either
            representation.
        filters: Observation filters, as for
            :func:`~openghg_inversions.rhime.preparation.filter_observations`.
        averaging_error: Include variability when deriving observation errors.
        basis: Flux and boundary basis choices.
        native_prior: Native prior used for the coherent reduction and the
            affine companion: one field shared by every flux source, or
            separate per-source fields reduced jointly.
        aggregation_error_rank: LRPD rank for the unresolved covariance, or
            ``None`` to store it exactly.
        bc_centering_weights_path: Optional netCDF file holding one
            ``bc_centering_weights`` array, for weighted additive boundaries.
    """

    acquisition: Mapping[str, Any]
    basis: Co2BasisSelection
    native_prior: Co2NativePrior | Co2MultiSourceNativePrior
    filters: Any = None
    averaging_error: bool = True
    aggregation_error_rank: int | None = 512
    bc_centering_weights_path: Path | None = None

    @classmethod
    def from_params(cls, prepared: dict[str, object], *, domain: str) -> Co2NativePreparationConfig:
        """Pop and parse the build-route tables of ``[prepared_inputs]``."""
        acquisition = _acquisition(_take(prepared, "acquisition", "prepared_inputs"))
        result: dict[str, Any] = {
            "acquisition": _frozen(acquisition),
            "basis": Co2BasisSelection.from_params(
                prepared.pop("basis", {}), domain=domain, use_bc=acquisition.get("use_bc", True),
            ),
            "native_prior": _native_prior(_take(prepared, "native_prior", "prepared_inputs")),
        }
        prior = result["native_prior"]
        if isinstance(prior, Co2MultiSourceNativePrior):
            if set(prior.sources) != set(acquisition["flux_sources"]):
                raise ValueError(
                    "prepared_inputs.native_prior.sources must name exactly the acquisition flux_sources "
                    f"{sorted(acquisition['flux_sources'])!r}; got {sorted(prior.sources)!r}."
                )
            if result["basis"].basis_path is not None:
                raise ValueError(
                    "prepared_inputs.basis.path cannot serve per-source native priors; use partition_path "
                    "(optionally with a source dimension) or a basis algorithm."
                )
            basis = result["basis"]
            if basis.basis_partition_path is None and basis.fix_basis_outer_regions:
                raise ValueError(
                    "prepared_inputs.basis.fix_outer_regions cannot serve per-source native priors: the "
                    "fixed-outer basis groups belong to one shared operator. Save that partition and pass it "
                    "as partition_path instead."
                )
        observations = _table(prepared.pop("observations", {}), "prepared_inputs.observations")
        if "filters" in observations:
            filters = observations.pop("filters")
            if isinstance(filters, Mapping):
                filters = {str(site): value for site, value in filters.items()}
            elif not isinstance(filters, str | list):
                raise TypeError("prepared_inputs.observations.filters must be a name, list or table.")
            result["filters"] = filters
        if "averaging_error" in observations:
            result["averaging_error"] = _bool(
                observations.pop("averaging_error"), "prepared_inputs.observations.averaging_error",
            )
        _reject_unknown(observations, "prepared_inputs.observations")
        if "aggregation_error_rank" in prepared:
            rank = prepared.pop("aggregation_error_rank")
            result["aggregation_error_rank"] = None if rank == "exact" else _positive_int(
                rank, "prepared_inputs.aggregation_error_rank (or 'exact')",
            )
        if "bc_centering_weights_path" in prepared:
            result["bc_centering_weights_path"] = _path(
                prepared.pop("bc_centering_weights_path"), "prepared_inputs.bc_centering_weights_path",
            )
        return cls(**result)


def _acquisition(value: object) -> dict[str, Any]:
    """Parse ``[prepared_inputs.acquisition]`` into acquisition keywords."""
    path = "prepared_inputs.acquisition"
    options = _table(value, path)
    sites: dict[str, Any] = {name: options.pop(name) for name in _SITE_OPTION_NAMES if name in options}
    for name in SiteOptions.REQUIRED_INPUT_NAMES:
        if name not in sites:
            raise ValueError(f"Missing required option {path}.{name}.")
    if isinstance(sites["sites"], str):
        sites["sites"] = [sites["sites"]]
    result: dict[str, Any] = {"site_options": SiteOptions.from_inputs(**sites)}
    sources = _take(options, "flux_sources", path)
    if isinstance(sources, str):
        sources = [sources]
    if not isinstance(sources, list) or not sources:
        raise TypeError(f"{path}.flux_sources must be a non-empty list of source names.")
    result["flux_sources"] = tuple(_string(item, f"{path}.flux_sources item") for item in sources)
    transport = _table(_take(options, "surface_transport", path), f"{path}.surface_transport")
    result["surface_transport"] = {
        source: _string(mode, f"{path}.surface_transport.{source}") for source, mode in transport.items()
    }
    for name in sorted(_ACQUISITION_STRING_NAMES & set(options)):
        result[name] = _string(options.pop(name), f"{path}.{name}")
    if "use_bc" in options:
        result["use_bc"] = _bool(options.pop("use_bc"), f"{path}.use_bc")
    if "boundary_reference" in options:
        result["boundary_reference"] = _number(options.pop("boundary_reference"), f"{path}.boundary_reference")
    if "boundary_mass_tolerance" in options:
        result["boundary_mass_tolerance"] = _number(
            options.pop("boundary_mass_tolerance"), f"{path}.boundary_mass_tolerance",
        )
    _reject_unknown(options, path)
    check_surface_request(
        species="co2", site_options=result["site_options"], flux_sources=result["flux_sources"],
        transport=result["surface_transport"], use_bc=result.get("use_bc", True),
        boundary_reference=result.get("boundary_reference"),
        boundary_reference_units=result.get("boundary_reference_units"),
        boundary_mass_tolerance=result.get("boundary_mass_tolerance"),
    )
    return result


@dataclass(frozen=True)
class Co2PreparationConfig:
    """Choices for importing a saved CO2 preparation or building a new one.

    ``[prepared_inputs]`` either names a saved artifact with ``path`` or, with
    an ``acquisition`` table, describes a new preparation (``native``). Paths
    resolve from the invocation directory. The affine companion is an
    association of the preparation, independent of current output choices.

    The prepared boundary representation is resolved by the configured
    ``[model.boundary]`` choice: ``boundary_representation`` may select one it
    can bind, and otherwise its default is used (see
    :func:`~openghg_inversions.models.boundary.boundary_preparation`). Only the
    boundary table is read from ``[model]``; other model options belong to
    :class:`Co2ModelConfig`.
    """

    prepared_path: Path | None = None
    affine_path: Path | None = None
    start_date: str | None = None
    end_date: str | None = None
    domain: str | None = None
    boundary_representation: BoundaryRepresentation | None = None
    boundary: Choice | None = None
    native: Co2NativePreparationConfig | None = None

    @classmethod
    def from_params(cls, config: Mapping[str, object], *, require_source: bool = False) -> Co2PreparationConfig:
        """Parse prepared-input options and the boundary choice they serve.

        Args:
            config: Complete CO2 request.
            require_source: Require ``path`` or an ``acquisition`` table.

        Raises:
            ValueError: If options are missing, unknown, conflicting, or the
                boundary choice cannot use the requested representation.
        """
        options = _table(config, "config")
        prepared = _table(options.get("prepared_inputs", {}), "prepared_inputs")
        result: dict[str, Any] = {}
        native = bool(_NATIVE_PREPARATION_OPTION_NAMES & set(prepared))
        if native and "path" in prepared:
            raise ValueError("prepared_inputs.path imports a saved preparation; remove the build options.")
        if require_source and not native and "path" not in prepared:
            raise ValueError(
                "Missing required option prepared_inputs.path, or prepared_inputs.acquisition to build one."
            )
        for name, argument in (("path", "prepared_path"), ("reconstruction_path", "affine_path")):
            if name in prepared:
                result[argument] = _path(prepared.pop(name), f"prepared_inputs.{name}")
        for name in ("start_date", "end_date", "domain"):
            if name in prepared:
                result[name] = _string(prepared.pop(name), f"prepared_inputs.{name}")
        if "boundary_representation" in prepared:
            result["boundary_representation"] = check_representation(prepared.pop("boundary_representation"))
        model = options.get("model", {})
        if isinstance(model, Mapping) and "boundary" in model:
            result["boundary"] = _boundary_choice(_table(model["boundary"], "model.boundary"), "model.boundary")
        needs = boundary_preparation(result.get("boundary"), result.get("boundary_representation"))
        if native:
            if "affine_path" in result:
                raise ValueError("A new preparation produces its own affine companion; remove reconstruction_path.")
            for name in ("start_date", "end_date", "domain"):
                if name not in result:
                    raise ValueError(f"Missing required option prepared_inputs.{name} for a new preparation.")
            result["native"] = Co2NativePreparationConfig.from_params(prepared, domain=result["domain"])
            weights = result["native"].bc_centering_weights_path
            if needs.centering_weights and weights is None:
                raise ValueError("The weighted additive boundary requires prepared_inputs.bc_centering_weights_path.")
            if weights is not None and needs.representation != "centered":
                raise ValueError("prepared_inputs.bc_centering_weights_path requires centered boundaries.")
            use_bc = result["native"].acquisition.get("use_bc", True)
            if not use_bc and (result.get("boundary") is not None or "boundary_representation" in result):
                raise ValueError(
                    "prepared_inputs.acquisition.use_bc = false cannot serve a configured boundary or "
                    "boundary_representation; enable use_bc or remove them."
                )
        _reject_unknown(prepared, "prepared_inputs")
        return cls(**result)

    def preparation_arguments(self) -> dict[str, Any]:
        """Return explicit keywords for the saved preparation boundary."""
        return {item.name: getattr(self, item.name) for item in fields(self)}


def _random_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("sampling.random_seed must be a non-negative integer.")
    return value


def resolve_co2_prior_seed(config: Mapping[str, object]) -> int | None:
    """Resolve only the random seed consumed by prior predictive sampling.

    MCMC options are unused by prior prediction and remain unvalidated here.
    """
    options = _table(_table(config, "config").get("sampling", {}), "sampling")
    return _random_seed(options["random_seed"]) if "random_seed" in options else None


def _sampling(value: object) -> RhimeSampler:
    options = _table(value, "sampling")
    kwargs: dict[str, Any] = {}
    for name in ("draws", "burn", "tune", "chains"):
        if name not in options:
            continue
        raw = options.pop(name)
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError(f"sampling.{name} must be an integer.")
        minimum = 0 if name in ("burn", "tune") else 1
        if raw < minimum:
            raise ValueError(f"sampling.{name} must be >= {minimum}.")
        kwargs[name] = raw
    if "nuts_sampler" in options:
        kwargs["nuts_sampler"] = _string(options.pop("nuts_sampler"), "sampling.nuts_sampler")
    if "progressbar" in options:
        kwargs["progressbar"] = _bool(options.pop("progressbar"), "sampling.progressbar")
    if "sample_prior_predictive" in options:
        prior_predictive = options.pop("sample_prior_predictive")
        if not isinstance(prior_predictive, bool | int) or (
            isinstance(prior_predictive, int)
            and not isinstance(prior_predictive, bool)
            and prior_predictive < 0
        ):
            raise TypeError("sampling.sample_prior_predictive must be a boolean or non-negative integer.")
        kwargs["sample_prior_predictive"] = prior_predictive
    if "sample_posterior_predictive" in options:
        posterior = options.pop("sample_posterior_predictive")
        if isinstance(posterior, bool):
            kwargs["sample_posterior_predictive"] = posterior
        elif isinstance(posterior, Sequence) and not isinstance(posterior, str | bytes):
            kwargs["sample_posterior_predictive"] = tuple(
                _string(item, "sampling.sample_posterior_predictive item") for item in posterior
            )
        else:
            raise TypeError(
                "sampling.sample_posterior_predictive must be a boolean or list of variable names."
            )

    sample_kwargs: dict[str, Any] = {}
    if "target_accept" in options:
        target_accept = _number(options.pop("target_accept"), "sampling.target_accept")
        if not 0.0 < target_accept < 1.0:
            raise ValueError("sampling.target_accept must be between zero and one.")
        sample_kwargs["target_accept"] = target_accept
    posterior_kwargs: dict[str, Any] = {}
    if "random_seed" in options:
        random_seed = _random_seed(options.pop("random_seed"))
        sample_kwargs["random_seed"] = random_seed
        posterior_kwargs["random_seed"] = random_seed
    _reject_unknown(options, "sampling")
    if sample_kwargs:
        kwargs["sample_kwargs"] = sample_kwargs
    if posterior_kwargs:
        kwargs["posterior_predictive_kwargs"] = posterior_kwargs
    sampler = RhimeSampler(**kwargs)
    if sampler.burn >= sampler.draws:
        raise ValueError("sampling.burn must be less than sampling.draws.")
    return sampler


def _multiplicative_boundary(options: dict[str, object], periods: dict[str, Any], path: str) -> MultiplicativeChoice:
    """Resolve the options of a multiplicative boundary, the default type."""
    prior = _prior(options.pop("prior"), f"{path}.prior") if "prior" in options else dict(DEFAULT_BC_PRIOR)
    _reject_unknown(options, path)
    return MultiplicativeChoice(prior, **periods)


def _mean_shift_options(options: dict[str, object], path: str) -> dict[str, Any]:
    """Resolve the optional mean-shift prior and its per-period switch shared by centred corrections."""
    mean_shift_prior = (
        _prior(options.pop("mean_shift_prior"), f"{path}.mean_shift_prior")
        if "mean_shift_prior" in options else None
    )
    mean_by_period = (
        _bool(options.pop("mean_by_period"), f"{path}.mean_by_period") if "mean_by_period" in options else False
    )
    return {"mean_prior": mean_shift_prior, "mean_by_period": mean_by_period}


def _additive_boundary(options: dict[str, object], periods: dict[str, Any], path: str) -> AdditiveChoice:
    """Resolve the options of centred additive boundary corrections."""
    mean = _mean_shift_options(options, path)
    anomaly_scale = _number(_take(options, "anomaly_scale", path), f"{path}.anomaly_scale")
    anomalies = _string(options.pop("anomalies"), f"{path}.anomalies") if "anomalies" in options else "weighted"
    _reject_unknown(options, path)
    return AdditiveChoice(
        mean["mean_prior"], anomaly_scale, anomalies=anomalies,  # type: ignore[arg-type]
        mean_by_period=mean["mean_by_period"], **periods,
    )


def _anomaly_scaling_boundary(
    options: dict[str, object], periods: dict[str, Any], path: str
) -> AnomalyScalingChoice:
    """Resolve the options of a boundary that scales the prescribed anomaly about a reference."""
    mean = _mean_shift_options(options, path)
    scale_sd = _number(_take(options, "scale_sd", path), f"{path}.scale_sd", positive=True)
    reference: object = options.pop("reference", "stored")
    if not isinstance(reference, str):
        reference = _number(reference, f"{path}.reference")
    _reject_unknown(options, path)
    return AnomalyScalingChoice(
        mean["mean_prior"], scale_sd, reference=reference,  # type: ignore[arg-type]
        mean_by_period=mean["mean_by_period"], **periods,
    )


# Each boundary ``type`` names one choice family and owns the parser for its options.
_BOUNDARY_TYPES: dict[str, Callable[[dict[str, object], dict[str, Any], str], Choice]] = {
    MultiplicativeChoice.kind: _multiplicative_boundary,
    AdditiveChoice.kind: _additive_boundary,
    AnomalyScalingChoice.kind: _anomaly_scaling_boundary,
}


def _boundary_choice(boundary: dict[str, object], path: str) -> Choice | None:
    """Resolve one boundary table at ``path``; ``None`` when explicitly disabled.

    ``type`` selects the family and defaults to ``"multiplicative"``, scaling
    absolute or centered sensitivity. Options belonging to another family are
    rejected rather than used to infer it.
    """
    enabled = _bool(boundary.pop("enabled", True), f"{path}.enabled")
    if not enabled:
        if boundary:
            raise ValueError(f"{path} options require {path}.enabled=true.")
        return None
    kind = _string(boundary.pop("type", MultiplicativeChoice.kind), f"{path}.type")
    if kind not in _BOUNDARY_TYPES:
        raise ValueError(f"{path}.type must be one of {sorted(_BOUNDARY_TYPES)!r}, not {kind!r}.")
    periods: dict[str, Any] = {
        name: _string(boundary.pop(name), f"{path}.{name}")
        for name in ("frequency", "anchor_time") if name in boundary
    }
    return _BOUNDARY_TYPES[kind](boundary, periods, path)


def _model(value: object, path: str = "model") -> dict[str, Any]:
    """Resolve boundary and offset choices from one model table at ``path``.

    The linked recipe passes ``channels.<name>`` so messages name the table the
    user actually wrote.
    """
    options = _table(value, path)
    result: dict[str, Any] = {}
    if "biosphere" in options:
        biosphere_path = f"{path}.biosphere"
        biosphere = _table(options.pop("biosphere"), biosphere_path)
        result["biosphere_prior_path"] = _path(_take(biosphere, "path", biosphere_path), f"{biosphere_path}.path")
        _reject_unknown(biosphere, biosphere_path)
    if "boundary" in options:
        result["boundary"] = _boundary_choice(_table(options.pop("boundary"), f"{path}.boundary"), f"{path}.boundary")
    if "offset" in options:
        result.update(_offset_model(options.pop("offset"), f"{path}.offset"))
    _reject_unknown(options, path)
    return result


def _offset_model(value: object, path: str) -> dict[str, Any]:
    """Resolve the offset prior and runner options from the offset table at ``path``."""
    result: dict[str, Any] = {}
    offset = _table(value, path)
    result["offset_prior"] = _prior(_take(offset, "prior", path), f"{path}.prior")
    offset_args: dict[str, Any] = {}
    if "frequency" in offset:
        frequency = offset.pop("frequency")
        if frequency is not None:
            frequency = _string(frequency, f"{path}.frequency")
        offset_args["offset_freq"] = frequency
    for config_name, runner_name in (("per_site", "per_site"), ("drop_first", "drop_first")):
        if config_name in offset:
            offset_args[runner_name] = _bool(offset.pop(config_name), f"{path}.{config_name}")
    _reject_unknown(offset, path)
    if offset_args.get("per_site") is False:
        if offset_args.get("offset_freq") is not None:
            raise ValueError(f"A global {path} does not accept frequency.")
        if offset_args.get("drop_first") is True:
            raise ValueError(f"A global {path} does not support drop_first=true.")
    result["offset_args"] = _frozen(offset_args)
    return result


def _co2_likelihood(value: object, *, matched_ou: bool = False) -> dict[str, Any]:
    options = _table(value, "likelihood")
    kind = _string(_take(options, "kind", "likelihood"), "likelihood.kind")
    result: dict[str, Any] = {}
    if kind == "additive_sigma":
        no_model_error = _bool(options.pop("no_model_error", False), "likelihood.no_model_error")
        result["no_model_error"] = no_model_error
        if "sigma_prior" in options:
            if no_model_error:
                raise ValueError("likelihood.sigma_prior cannot be combined with no_model_error=true.")
            result["sigma_prior"] = _prior(
                options.pop("sigma_prior"), "likelihood.sigma_prior", positive=True
            )
        if "fixed_model_mismatch" in options:
            fixed = _number(
                options.pop("fixed_model_mismatch"),
                "likelihood.fixed_model_mismatch",
            )
            if fixed < 0.0:
                raise ValueError("likelihood.fixed_model_mismatch must be non-negative.")
            result["fixed_model_mismatch"] = fixed
    elif kind == "site_sigma":
        kwargs: dict[str, Any] = {}
        if "fixed_site_amplitudes" in options:
            fixed_site_amplitudes = options.pop("fixed_site_amplitudes")
            if not isinstance(fixed_site_amplitudes, Mapping):
                raise TypeError("likelihood.fixed_site_amplitudes must be a site mapping.")
            kwargs["fixed_site_amplitudes"] = _number_or_mapping(
                fixed_site_amplitudes,
                "likelihood.fixed_site_amplitudes",
                positive=False,
                nonnegative=True,
            )
        if "site_amplitude_prior" in options:
            kwargs["site_amplitude_prior"] = _prior(
                options.pop("site_amplitude_prior"),
                "likelihood.site_amplitude_prior",
                positive=True,
            )
        if len(kwargs) != 1:
            raise ValueError(
                "site_sigma requires exactly one of fixed_site_amplitudes or site_amplitude_prior."
            )
        result.update(
            likelihood_builder=add_site_sigma_gaussian_likelihood,
            likelihood_kwargs=_frozen(kwargs),
        )
    elif kind == "fixed_ou":
        kwargs = {
            "tau_hours": _number_or_mapping(
                _take(options, "tau_hours", "likelihood"),
                "likelihood.tau_hours",
                positive=True,
            )
        }
        if "fixed_site_amplitudes" in options:
            kwargs["fixed_site_amplitudes"] = _number_or_mapping(
                options.pop("fixed_site_amplitudes"),
                "likelihood.fixed_site_amplitudes",
                positive=False,
                nonnegative=True,
            )
        if "site_amplitude_prior" in options:
            kwargs["site_amplitude_prior"] = _prior(
                options.pop("site_amplitude_prior"),
                "likelihood.site_amplitude_prior",
                positive=True,
            )
        if len(kwargs) != 2:
            raise ValueError(
                "fixed_ou requires exactly one of fixed_site_amplitudes or site_amplitude_prior."
            )
        if matched_ou and "site_amplitude_prior" in kwargs and "initial_site_amplitudes" in options:
            kwargs["initial_site_amplitudes"] = _number_or_mapping(
                options.pop("initial_site_amplitudes"), "likelihood.initial_site_amplitudes", positive=True,
            )
        result.update(
            likelihood_builder=add_fixed_ou_gaussian_likelihood,
            likelihood_kwargs=_frozen(kwargs),
        )
    elif kind == "scalar_sigma":
        result.update(
            likelihood_builder=add_scalar_sigma_eigen_likelihood,
            likelihood_kwargs=_frozen(
                {
                    "eigenbasis_path": Path(
                        _string(
                            _take(options, "eigenbasis_path", "likelihood"),
                            "likelihood.eigenbasis_path",
                        )
                    ),
                    "sigma_prior": _prior(
                        _take(options, "sigma_prior", "likelihood"),
                        "likelihood.sigma_prior",
                        positive=True,
                    ),
                }
            ),
        )
    else:
        raise ValueError(
            "likelihood.kind must be one of 'additive_sigma', 'site_sigma', 'fixed_ou', or 'scalar_sigma'."
        )
    _reject_unknown(options, "likelihood")
    return result


def load_co2_family_config(path: str | Path | Traversable) -> Mapping[str, object]:
    """Load one CO2-family TOML file without applying recipe semantics."""
    resource = Path(path) if isinstance(path, str | Path) else path
    with resource.open("rb") as config_file:
        return tomllib.load(config_file)


def co2_config_templates() -> Mapping[str, Traversable]:
    """Return installed CO2-family TOML templates keyed by file name."""
    directory = files("openghg_inversions.rhime").joinpath("config")
    return MappingProxyType(
        {name: directory.joinpath(name) for name in ("co2.toml", "co2_fixed_ou.toml", "co2_o2.toml")}
    )


@dataclass(frozen=True)
class Co2ModelConfig:
    """Scientific choices for ordinary CO2, independent of arrays and inference.

    Construction through :meth:`from_params` validates only model and likelihood
    sections. Ordinary containers are owned; opaque numerical values are borrowed.
    Defaults omitted here remain owned by the concrete scientific recipe.
    The boundary choice is a prior specification; inference binds it to the preparation.
    """

    boundary: Choice | None = None
    offset_prior: Mapping[str, Any] | None = None
    offset_args: Mapping[str, Any] | None = None
    no_model_error: bool | None = None
    sigma_prior: Mapping[str, Any] | None = None
    fixed_model_mismatch: float | None = None
    likelihood_builder: Callable[..., Any] | None = None
    likelihood_kwargs: Mapping[str, Any] | None = None
    biosphere_prior_path: Path | None = None

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, Mapping):
                object.__setattr__(self, item.name, normalise_optional_mapping(value))

    @classmethod
    def from_params(cls, config: Mapping[str, object]) -> Co2ModelConfig:
        """Parse active science without opening artifacts or creating a sampler."""
        options = _table(config, "config")
        model = _model(options.get("model", {}))
        likelihood = _co2_likelihood(_take(options, "likelihood", "config"), matched_ou=True)
        kwargs = likelihood.get("likelihood_kwargs", {})
        if "eigenbasis_path" in kwargs:
            likelihood["likelihood_kwargs"] = _frozen({
                **kwargs, "eigenbasis_path": kwargs["eigenbasis_path"].expanduser().resolve(),
            })
        return cls(**model, **likelihood)

    def sampler_arguments(self) -> dict[str, Any]:
        """Return graph-independent policy choices without opening artifacts."""
        return {"likelihood_builder": self.likelihood_builder, "likelihood_kwargs": self.likelihood_kwargs or {}}

    def model_arguments(self) -> dict[str, Any]:
        """Bind these choices as keywords, loading a selected biosphere prior.

        Call after output preflight: this opens the calibrated prior artifact.
        """
        from .prior_configuration import load_biosphere_prior

        result = {item.name: getattr(self, item.name) for item in fields(self)
                  if item.name != "biosphere_prior_path" and getattr(self, item.name) is not None}
        if self.biosphere_prior_path is not None:
            result["biosphere_prior"] = load_biosphere_prior(self.biosphere_prior_path)
        return result

    def artifact_paths(self) -> tuple[Path, ...]:
        """Identify the calibrated prior and likelihood artifacts consumed during construction."""
        eigenbasis = (self.likelihood_kwargs or {}).get("eigenbasis_path")
        return tuple(Path(path).resolve() for path in (self.biosphere_prior_path, eigenbasis) if path is not None)


# NUTS-only sampling options, refused with the exact Gaussian estimator.
_NUTS_ONLY_SAMPLING_OPTIONS = ("burn", "tune", "chains", "target_accept", "sigma_target_accept", "state_target_accept")
CO2_INFERENCE_METHODS = ("nuts", "gaussian")


@dataclass(frozen=True)
class _Co2Sampling:
    """Resolved sampler, the matched OU transition tuning controls and the inference method.

    ``method="gaussian"`` uses only the sampler's ``draws``, random seed and
    predictive choices.
    """

    sampler: RhimeSampler
    sigma_target_accept: float = 0.8
    state_target_accept: float = 0.9
    method: Literal["nuts", "gaussian"] = "nuts"


def _resolve_co2_sampling(config: Mapping[str, object], *, model: Mapping[str, Any]) -> _Co2Sampling:
    """Resolve sampling options after the scientific model has selected its graph.

    The configured backend is selected from resolved scientific choices.
    Additive sigma uses NumPyro; supported custom likelihoods use PyMC. Inferred
    OU amplitudes use the matched PyMC transition and its two explicit controls.

    Returns:
        object: Resolved sampler and matched OU tuning controls, with attributes
            ``sampler``, ``sigma_target_accept``, and ``state_target_accept``.
    """
    options = _table(config, "config")
    sampling = _table(options.get("sampling", {}), "sampling")
    if "nuts_sampler" in sampling:
        raise ValueError("Configured CO2 selects its sampler from the likelihood; remove sampling.nuts_sampler.")
    method = _string(sampling.pop("method", "nuts"), "sampling.method")
    if method not in CO2_INFERENCE_METHODS:
        raise ValueError(f"sampling.method must be one of {list(CO2_INFERENCE_METHODS)!r}, not {method!r}.")
    if method == "gaussian":
        from .co2_gaussian import check_co2_gaussian_likelihood, require_gaussian_precision

        require_gaussian_precision()
        check_co2_gaussian_likelihood(model)
        unused = sorted(set(_NUTS_ONLY_SAMPLING_OPTIONS) & sampling.keys())
        if unused:
            raise ValueError(
                f"sampling.method = 'gaussian' draws exact independent samples; remove NUTS-only "
                f"option(s) {', '.join(f'sampling.{name}' for name in unused)}."
            )
        return _Co2Sampling(sampler=_sampling(sampling), method="gaussian")
    cached = uses_cached_ou(model)
    targets = {}
    for name, default in (("sigma_target_accept", 0.8), ("state_target_accept", 0.9)):
        if name in sampling and not cached:
            raise ValueError(f"sampling.{name} requires inferred fixed-OU amplitudes.")
        target = _number(sampling.pop(name, default), f"sampling.{name}")
        if not 0.0 < target < 1.0:
            raise ValueError(f"sampling.{name} must be between zero and one.")
        targets[name] = target
    sampling["nuts_sampler"] = "pymc" if model.get("likelihood_builder") is not None else "numpyro"
    sampler = _sampling(sampling)
    if cached:
        validate_cached_co2_sampler(sampler)
    return _Co2Sampling(
        sampler=sampler,
        sigma_target_accept=targets["sigma_target_accept"], state_target_accept=targets["state_target_accept"],
    )


__all__ = [
    "Co2PreparationConfig", "Co2NativePreparationConfig", "Co2BasisSelection", "Co2NativePrior",
    "Co2ModelConfig", "co2_config_templates",
    "load_co2_family_config", "resolve_co2_prior_seed", "validate_co2_config",
]
