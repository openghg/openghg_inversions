from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from pytensor.compile.mode import Mode

import openghg_inversions.hbmcmc.inversion_pymc as legacy_mcmc
import openghg_inversions.inversion_data.preparation as prep_module
import openghg_inversions.models as models
import openghg_inversions.models._rhime_compiler as rhime_compiler_module
import openghg_inversions.models.rhime as rhime_models_module
import openghg_inversions.postprocessing.inversion_output as inversion_output_module
import openghg_inversions.rhime as rhime_public
import openghg_inversions.rhime.outputs as rhime_outputs
import openghg_inversions.rhime.params as rhime_params
import openghg_inversions.rhime.runner as rhime_module
import openghg_inversions.rhime.sampling as rhime_sampling
import openghg_inversions.rhime.specs as rhime_specs
from openghg_inversions.basis.basis_functions import (
    BASIS_ARTIFACT_PATH_ATTR,
    BASIS_ARTIFACT_SOURCE_ATTR,
    BasisFunctions,
)
from openghg_inversions.basis.operators import BasisMeta, BasisOperator, BucketBasisOperator
from openghg_inversions.cli import main
from openghg_inversions.flux_sanitization import (
    NONFINITE_POLICY_ZERO_FILL,
    FluxNonFiniteMetadata,
    NonFiniteFluxWarning,
)
from openghg_inversions.inversion_data import RhimePreparedInputs, prepare_rhime_inputs
from openghg_inversions.inversion_inputs import make_inv_inputs
from openghg_inversions.models import (
    StateActivity,
    build_rhime_model,
    build_rhime_model_from_spec,
    build_rhime_multisector_model,
    build_rhime_multisector_model_from_spec,
    safe_pymc_name,
)
from openghg_inversions.models._rhime_compiler import (
    _compile_loop_sum,
    _FluxPlan,
    _ForwardTermPlan,
    _StatePlan,
)
from openghg_inversions.models.coords import CoordRegistry, attach_coord_registry
from openghg_inversions.postprocessing._basis_products import (
    BASIS_ARTIFACT_PATH_OUTPUT_ATTR,
    BASIS_ARTIFACT_SOURCE_LOADED_DATATREE,
    BASIS_ARTIFACT_SOURCE_OUTPUT_ATTR,
    BASIS_RECONSTRUCTION_OPERATOR_BACKED,
    BASIS_RECONSTRUCTION_PATH_ATTR,
)
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.make_outputs import observation_inputs_for_outputs
from openghg_inversions.postprocessing.make_paris_outputs import PARIS_LATEST_COUNTRIES
from openghg_inversions.rhime import (
    RhimeModelSpec,
    RhimeOutputSpec,
    RhimeResult,
    RhimeRunSpec,
    RhimeSampler,
    SectorSpec,
    params_from_config,
    resolve_flux_sources,
    run_rhime,
    run_rhime_from_prepared_inputs,
    run_rhime_multisector,
)
from openghg_inversions.sigma import SigmaAlignment


@pytest.fixture(scope="module")
def rhime_inv_inputs(mhd_and_tac_fp_data) -> xr.Dataset:
    return make_inv_inputs(
        mhd_and_tac_fp_data,
        sites=["MHD", "TAC"],
        bc_freq="3h",
        min_error=0.0,
        start_date="2019-01-01",
    )


def _flux_nonfinite_metadata(data: xr.DataArray | xr.Dataset) -> FluxNonFiniteMetadata:
    """Return parsed non-finite flux metadata from an xarray object."""
    metadata = FluxNonFiniteMetadata.from_attrs(data.attrs)
    assert metadata is not None
    return metadata


@pytest.fixture
def multisector_inv_inputs(rhime_inv_inputs: xr.Dataset) -> xr.Dataset:
    ds = rhime_inv_inputs.copy()
    ds["H"] = xr.concat(
        [
            rhime_inv_inputs["H"].expand_dims(source=["total-ukghg-edgar7"]),
            (2.0 * rhime_inv_inputs["H"]).expand_dims(source=["sector-2"]),
        ],
        dim="source",
    )
    return ds


@pytest.fixture
def builder_args(rhime_inv_inputs: xr.Dataset) -> dict:
    """Build low-level model arguments including prepared sigma alignment."""
    return {
        "sigma_alignment": SigmaAlignment.from_frequency(
            rhime_inv_inputs["site_indicator"],
            frequency="3h",
            anchor_time="2019-01-01",
        ),
        "x_prior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
        "bc_prior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
        "sigma_prior": {"pdf": "uniform", "lower": 0.1, "upper": 10.0},
        "offset_prior": {"pdf": "normal", "mu": 0, "sigma": 1},
        "add_offset": False,
        "use_bc": True,
        "pollution_events_from_obs": True,
        "no_model_error": False,
        "power": 1.99,
    }


def _fake_basis_functions(*, artifact_source: str = "generated") -> BasisFunctions:
    """Build a minimal one-cell basis artifact for RHIME tests."""
    basis = xr.DataArray(
        [[1]],
        dims=("lat", "lon"),
        coords={"lat": [0.0], "lon": [0.0]},
        name="basis",
    )
    flux = xr.ones_like(basis, dtype=float).rename("flux")
    return BasisFunctions.from_flat_basis(
        basis_flat=basis,
        flux=flux,
        operator_kwargs={"state_dim": "region"},
        metadata={BASIS_ARTIFACT_SOURCE_ATTR: artifact_source},
    )


def _fake_basis_functions_matching_country_grid(country_file: Path) -> BasisFunctions:
    """Build a one-region basis artifact on the grid of a test country file."""
    with xr.open_dataset(country_file) as country_grid:
        lat = country_grid.lat.values.copy()
        lon = country_grid.lon.values.copy()
    basis = xr.DataArray(
        np.ones((lat.size, lon.size), dtype=int),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="basis",
    )
    flux = xr.ones_like(basis, dtype=float).rename("flux")
    flux.attrs["units"] = "mol/m2/s"
    return BasisFunctions.from_flat_basis(
        basis_flat=basis,
        flux=flux,
        operator_kwargs={"state_dim": "region"},
        metadata={BASIS_ARTIFACT_SOURCE_ATTR: "country-grid-test"},
    )


def _two_region_basis_functions_matching_country_grid(country_file: Path) -> BasisFunctions:
    """Build a non-uniform two-region basis artifact on the grid of a test country file."""
    with xr.open_dataset(country_file) as country_grid:
        lat = country_grid.lat.values.copy()
        lon = country_grid.lon.values.copy()
    nlat = lat.size
    nlon = lon.size
    basis_values = np.ones((nlat, nlon), dtype=int)
    basis_values[nlat // 2 :, :] = 2
    basis = xr.DataArray(
        basis_values,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="basis",
    )
    flux_values = np.linspace(1.0, 3.0, nlat * nlon, dtype=float).reshape(nlat, nlon)
    flux = xr.DataArray(
        flux_values,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="flux",
    )
    flux.attrs["units"] = "mol/m2/s"
    return BasisFunctions.from_flat_basis(
        basis_flat=basis,
        flux=flux,
        operator_kwargs={"state_dim": "region"},
        metadata={BASIS_ARTIFACT_SOURCE_ATTR: "two-region-test"},
    )


def _unsanitized_nonfinite_basis_functions_matching_country_grid(country_file: Path) -> BasisFunctions:
    """Build an old-style basis artifact whose retained flux has non-finite values."""
    basis_functions = _two_region_basis_functions_matching_country_grid(country_file)
    flux = basis_functions.flux.copy()
    flux.values[0, 0] = np.nan
    flux.values[-1, -1] = np.inf
    flux.attrs = {"units": "mol/m2/s"}
    return type(basis_functions)(
        operator=basis_functions.operator,
        flux=flux,
        metadata=dict(basis_functions.metadata),
    )


class _RecordingBasisOperator(BasisOperator):
    """Basis operator test double that records product reconstruction calls."""

    kind = "recording-test"

    def __init__(self, basis_flat: xr.DataArray) -> None:
        operator_cls = cast(Any, BucketBasisOperator)
        self._operator = operator_cls(
            basis_flat=basis_flat,
            meta=BasisMeta(state_dim="region"),
            region_labels="range0",
        )
        self.interpolate_calls: list[tuple[xr.DataArray, xr.DataArray | None]] = []
        self.basis_matrix_accesses = 0

    @property
    def meta(self) -> BasisMeta:
        """Basis metadata from the wrapped operator."""
        return self._operator.meta

    @property
    def basis_matrix(self) -> xr.DataArray:
        """Basis matrix from the wrapped operator, recording direct use."""
        self.basis_matrix_accesses += 1
        return self._operator.basis_matrix

    def interpolate(self, state: xr.DataArray, weights: xr.DataArray | None = None) -> xr.DataArray:
        """Record interpolation before delegating to the wrapped operator."""
        self.interpolate_calls.append((state, weights))
        return self._operator.interpolate(state, weights=weights)

    def to_datatree(self) -> xr.DataTree:
        """Serialization is not needed for this test double."""
        raise NotImplementedError

    @classmethod
    def from_datatree(cls, dt: xr.DataTree) -> _RecordingBasisOperator:
        """Deserialization is not needed for this test double."""
        raise NotImplementedError


def _recording_basis_functions_matching_country_grid(
    country_file: Path,
) -> tuple[BasisFunctions, _RecordingBasisOperator]:
    """Build a retained basis artifact with a recording operator."""
    with xr.open_dataset(country_file) as country_grid:
        lat = country_grid.lat.values.copy()
        lon = country_grid.lon.values.copy()
    basis = xr.DataArray(
        np.ones((lat.size, lon.size), dtype=int),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="basis",
    )
    flux = xr.ones_like(basis, dtype=float).rename("flux")
    flux.attrs["units"] = "mol/m2/s"
    operator = _RecordingBasisOperator(basis)
    basis_functions = BasisFunctions(
        operator=operator,
        flux=flux,
        metadata={BASIS_ARTIFACT_SOURCE_ATTR: "recording-test"},
    )
    return basis_functions, operator


def _site_dataset(values: list[float] | None = None) -> xr.Dataset:
    """Build a minimal site dataset with footprint-times-flux values."""
    values = values if values is not None else [2.0, 3.0]
    time = np.array(
        [f"2019-01-01T{hour:02d}:00:00" for hour in range(len(values))],
        dtype="datetime64[ns]",
    )
    fp_x_flux = xr.DataArray(
        np.array(values, dtype=float).reshape(len(values), 1, 1),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [0.0], "lon": [0.0]},
        name="fp_x_flux",
        attrs={"units": "1e-9"},
    )
    return xr.Dataset(
        {
            "fp_x_flux": fp_x_flux,
            "mf": (
                "time",
                np.linspace(10.0, 10.0 + len(values) - 1, len(values)),
                {"units": "1e-9"},
            ),
            "mf_error": ("time", np.ones(len(values)), {"units": "1e-9"}),
            "mf_repeatability": ("time", np.full(len(values), 0.5), {"units": "1e-9"}),
            "mf_variability": ("time", np.full(len(values), 0.25), {"units": "1e-9"}),
        },
        coords={"time": time},
    )


def _minimal_inv_inputs() -> xr.Dataset:
    """Build a minimal single-measurement RHIME inversion-input dataset."""
    return xr.Dataset(
        {"H": (("region", "nmeasure"), [[1.0]])},
        coords={"region": [0], "nmeasure": [0]},
    )


def _minimal_prepared_inv_inputs(sites: tuple[str, ...] = ("TAC",)) -> xr.Dataset:
    """Build minimal inversion inputs satisfying the durable prepared contract."""
    nmeasure = pd.MultiIndex.from_arrays(
        [list(sites), pd.date_range("2019-01-01", periods=len(sites), freq="h")],
        names=["site", "time"],
    )
    return xr.Dataset(
        {
            "H": (("region", "nmeasure"), [np.ones(len(sites))]),
            "site_indicator": ("nmeasure", np.arange(len(sites), dtype=int)),
        },
        coords={
            "region": [0],
            **xr.Coordinates.from_pandas_multiindex(nmeasure, "nmeasure"),
        },
    )


def _prepared_site_metadata(
    sites: tuple[str, ...] = ("TAC",),
    averaging_period: tuple[str | None, ...] = ("1h",),
) -> xr.Dataset:
    """Build labeled site metadata for the canonical prepared-input API."""
    return xr.Dataset(
        {"averaging_period": ("site", np.asarray(averaging_period, dtype=object))},
        coords={"site": list(sites)},
    )


def _site_options(
    sites: list[str],
    *,
    averaging_period: list[str | None] | str | None = None,
    inlet: list[str | None] | str | None = None,
    fp_height: list[str | None] | str | None = None,
    instrument: list[str | None] | str | None = None,
    platform: list[str | None] | str | None = None,
    obs_data_level: list[str | None] | str | None = None,
    met_model: list[str | None] | str | None = None,
    max_level: list[int | None] | int | None = None,
) -> prep_module._SiteOptions:
    """Build normalized site-aligned options for private preparation tests."""
    return prep_module._SiteOptions.from_inputs(
        sites=sites,
        averaging_period=averaging_period,
        inlet=inlet,
        fp_height=fp_height,
        instrument=instrument,
        platform=platform,
        obs_data_level=obs_data_level,
        met_model=met_model,
        max_level=max_level,
    )


def _minimal_output_inv_inputs() -> xr.Dataset:
    """Build minimal inversion inputs for output adapter tests."""
    inv_inputs = xr.Dataset(
        {
            "H": (("region", "nmeasure"), [[1.0]]),
            "mf": ("nmeasure", [10.0]),
            "mf_error": ("nmeasure", [1.0]),
            "mf_repeatability": ("nmeasure", [0.5]),
            "mf_variability": ("nmeasure", [0.25]),
            "site_indicator": ("nmeasure", [0]),
        },
        coords={
            "region": [0],
            "nmeasure": [0],
            "site": ("nmeasure", ["TAC"]),
            "time": ("nmeasure", np.array(["2019-01-01T00:00:00"], dtype="datetime64[ns]")),
        },
    )
    inv_inputs["mf"].attrs["units"] = "ppm"
    return inv_inputs.set_index(nmeasure=["site", "time"])


def _minimal_output_specs(
    output_format: rhime_specs.OutputFormat = "inv_out",
) -> tuple[RhimeModelSpec, RhimeOutputSpec, RhimeRunSpec]:
    """Build minimal RHIME specs for output helper tests."""
    model_spec = RhimeModelSpec(
        species="ch4",
        domain="EUROPE",
        sectors=(
            SectorSpec(
                name="FF",
                flux_source="ff-inventory",
                x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                variable_suffix="ff",
            ),
        ),
    )
    output_spec = RhimeOutputSpec(
        output_format=output_format,
        output_name="test",
        save_inversion_output=False,
    )
    run_spec = RhimeRunSpec(
        "2019-01-01",
        "2019-01-02",
        ("TAC",),
        ("1h",),
        model_spec,
        output_spec,
    )
    return model_spec, output_spec, run_spec


def _minimal_output_idata() -> az.InferenceData:
    """Build a minimal posterior trace with one region."""
    return az.from_dict(
        posterior={"x": np.ones((1, 1, 1))},
        coords={"region": [0]},
        dims={"x": ["region"]},
    )


def _posterior_only_idata(model: pm.Model, variable_names: tuple[str, ...]) -> az.InferenceData:
    """Build deterministic posterior variables using a PyMC model's coordinates.

    Args:
        model: Built model defining the requested variables and their dimensions.
        variable_names: Posterior variable names to include.

    Returns:
        One-chain, one-draw inference data containing only the requested
        posterior variables.
    """
    coords = {"chain": np.arange(1), "draw": np.arange(1)}
    posterior_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for variable_name in variable_names:
        model_dims = model.named_vars_to_dims[variable_name]
        for dim in model_dims:
            coord = model.coords[dim]
            assert coord is not None
            coords[dim] = np.asarray(coord)
        dims = ("chain", "draw", *model_dims)
        shape = tuple(len(coords[dim]) for dim in dims)
        posterior_vars[variable_name] = (dims, np.ones(shape))

    return az.InferenceData(posterior=xr.Dataset(posterior_vars, coords=coords))


def _postprocessing_output_idata(nregion: int = 1) -> az.InferenceData:
    """Build a small complete trace for modern postprocessing smoke tests."""
    region = np.arange(nregion)
    nmeasure = pd.MultiIndex.from_arrays(
        [["TAC"], np.array(["2019-01-01T00:00:00"], dtype="datetime64[ns]")],
        names=["site", "time"],
    )
    x_posterior = np.stack(
        [np.linspace(1.0 + draw, 2.0 + draw, nregion) for draw in range(3)],
        axis=0,
    )
    x_prior = np.stack(
        [np.linspace(0.8 + draw, 1.8 + draw, nregion) for draw in range(3)],
        axis=0,
    )
    idata = az.from_dict(
        posterior={
            "x": x_posterior[np.newaxis, :, :],
            "epsilon": np.full((1, 3, 1), 2.0),
            "mu_bc": np.full((1, 3, 1), 0.1),
        },
        prior={
            "x": x_prior[np.newaxis, :, :],
            "mu_bc": np.full((1, 3, 1), 0.05),
        },
        posterior_predictive={"y": np.full((1, 3, 1), 10.0)},
        prior_predictive={"y": np.full((1, 3, 1), 9.0)},
        constant_data={
            "hx": np.ones(1),
            "hbc": np.full(1, 0.1),
            "min_error": np.zeros(1),
        },
        coords={"region": region, "nmeasure": np.arange(len(nmeasure))},
        dims={
            "x": ["region"],
            "epsilon": ["nmeasure"],
            "mu_bc": ["nmeasure"],
            "y": ["nmeasure"],
            "hx": ["nmeasure"],
            "hbc": ["nmeasure"],
            "min_error": ["nmeasure"],
        },
    )
    nmeasure_coords = xr.Coordinates.from_pandas_multiindex(nmeasure, "nmeasure")
    for group in idata.groups():
        ds = idata[group]
        if "nmeasure" in ds.dims:
            setattr(idata, group, ds.assign_coords(nmeasure_coords))
    return idata


def _rename_idata_data_vars(idata: az.InferenceData, rename: dict[str, str]) -> az.InferenceData:
    """Rename data variables across all InferenceData groups."""
    groups: dict[str, xr.Dataset] = {}
    for group in idata.groups():
        ds = idata[group]
        group_rename = {old: new for old, new in rename.items() if old in ds.data_vars}
        groups[group] = ds.rename_vars(group_rename) if group_rename else ds.copy()
    return cast(Any, az.InferenceData)(**groups)


def _modern_postprocessing_inv_out(
    country_file: Path, basis_functions: BasisFunctions | None = None
) -> InversionOutput:
    """Build a modern output with enough groups for real postprocessing helpers."""
    basis_functions = basis_functions or _fake_basis_functions_matching_country_grid(country_file)
    nregion = basis_functions.operator.basis_matrix.sizes[basis_functions.operator.meta.state_dim]
    inv_inputs = _minimal_output_inv_inputs()
    inv_inputs = inv_inputs.drop_dims("region").assign_coords(region=np.arange(nregion))
    inv_inputs["H"] = xr.DataArray(
        np.ones((nregion, inv_inputs.sizes["nmeasure"])),
        dims=("region", "nmeasure"),
        coords={"region": inv_inputs.region, "nmeasure": inv_inputs.nmeasure},
    )
    for var_name in ("mf", "mf_error", "mf_repeatability", "mf_variability"):
        inv_inputs[var_name].attrs["units"] = "1e-09 mol/mol"

    return InversionOutput(
        trace=_postprocessing_output_idata(nregion=nregion),
        inv_inputs=inv_inputs,
        basis_functions=basis_functions,
        run_metadata={
            "start_date": "2019-01-01",
            "end_date": "2019-01-02",
            "sites": ["TAC"],
            "split_by_sectors": False,
        },
        model_metadata={"species": "ch4", "domain": "EUROPE"},
    )


def _with_column_prior_factors(inv_out: InversionOutput) -> InversionOutput:
    """Return a copy with OCO-style column prior correction factors."""
    inv_inputs = inv_out.inv_inputs.copy()
    coords = {"nmeasure": inv_inputs["nmeasure"]}
    inv_inputs["mf_prior_factor"] = xr.DataArray([0.2], dims=("nmeasure",), coords=coords)
    inv_inputs["mf_prior_upper_level_factor"] = xr.DataArray([0.3], dims=("nmeasure",), coords=coords)
    return replace(inv_out, inv_inputs=inv_inputs)


class _SpyBasisFunctions:
    """BasisFunctions test double that records direct sensitivity calls."""

    basis_artifact_source = "datatree"

    def __init__(self, sensitivity: xr.DataArray) -> None:
        self.sensitivity_calls: list[xr.DataArray] = []
        self._sensitivity = sensitivity

    @property
    def flux(self) -> xr.DataArray:
        raise AssertionError("RHIME preparation should not materialise flux from BasisFunctions.")

    @property
    def operator(self) -> object:
        return SimpleNamespace(source_labels=None)

    def flat_basis(self) -> xr.DataArray:
        return xr.DataArray([[1]], dims=("lat", "lon"))

    def sensitivity(self, fp_x_flux: xr.DataArray, fillna: bool = True) -> xr.DataArray:
        self.sensitivity_calls.append(fp_x_flux)
        return self._sensitivity


class _DynamicSpyBasisFunctions(_SpyBasisFunctions):
    """BasisFunctions test double that derives sensitivity shape from input time."""

    def __init__(self) -> None:
        super().__init__(xr.DataArray())

    def sensitivity(self, fp_x_flux: xr.DataArray, fillna: bool = True) -> xr.DataArray:
        self.sensitivity_calls.append(fp_x_flux)
        return xr.DataArray(
            np.ones((1, fp_x_flux.sizes["time"])),
            dims=("region", "time"),
            coords={"region": [0], "time": fp_x_flux.time},
            name="H",
        )


class _DynamicSectorSpyBasisFunctions(_SpyBasisFunctions):
    """BasisFunctions test double that derives source-resolved sensitivity."""

    def __init__(self) -> None:
        super().__init__(xr.DataArray())
        self._source_labels: tuple[str, ...] = ()

    @property
    def operator(self) -> object:
        """Expose source labels without constructing a basis matrix."""
        return SimpleNamespace(source_labels=self._source_labels)

    def sensitivity(self, fp_x_flux: xr.DataArray, fillna: bool = True) -> xr.DataArray:
        self.sensitivity_calls.append(fp_x_flux)
        self._source_labels = tuple(str(source) for source in fp_x_flux.source.values)
        return xr.DataArray(
            np.ones((1, fp_x_flux.sizes["time"], fp_x_flux.sizes["source"])),
            dims=("region", "time", "source"),
            coords={"region": [0], "time": fp_x_flux.time, "source": fp_x_flux.source},
            name="H",
        )


def test_compile_loop_sum_reuses_state_and_sums_named_terms() -> None:
    """One shared state feeds named coefficient-scaled terms and their total."""
    coords = {"region": ["r0", "r1"], "nmeasure": ["obs0", "obs1", "obs2"]}
    design_a = xr.DataArray(
        [[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]],
        dims=("region", "nmeasure"),
        coords=coords,
    )
    design_b = xr.DataArray(
        [[4.0, 3.0, 2.0], [1.0, 2.0, 3.0]],
        dims=("region", "nmeasure"),
        coords=coords,
    )
    plan = _FluxPlan(
        states=(
            _StatePlan(
                state_id="shared",
                variable_name="x_shared",
                prior_args={"pdf": "uniform", "lower": 1.0, "upper": 2.0},
            ),
        ),
        terms=(
            _ForwardTermPlan(
                term_id="term_a",
                state_id="shared",
                design=design_a,
                data_name="h_a",
                deterministic_name="mu_a",
            ),
            _ForwardTermPlan(
                term_id="term_b",
                state_id="shared",
                design=design_b,
                data_name="h_b",
                deterministic_name="mu_b",
                coefficient=-0.25,
            ),
        ),
    )

    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        compiled = _compile_loop_sum(plan)
        trace = pm.sample_prior_predictive(
            draws=3,
            var_names=["x_shared", "mu_a", "mu_b", "mu"],
            random_seed=402,
            compile_kwargs={"mode": Mode(linker="py", optimizer="fast_run")},
        )

    assert [rv.name for rv in model.free_RVs].count("x_shared") == 1
    assert compiled.states["shared"] is model["x_shared"]
    assert compiled.latents["shared"] is model["x_shared"]
    assert compiled.terms["term_a"] is model["mu_a"]
    assert compiled.terms["term_b"] is model["mu_b"]
    prior = cast(Any, trace).prior
    assert set(prior.data_vars) == {"x_shared", "mu_a", "mu_b", "mu"}
    trace_coords = {"region": prior["region"], "nmeasure": prior["nmeasure"]}
    expected_a = xr.dot(
        prior["x_shared"],
        design_a.assign_coords(trace_coords),
        dim="region",
    )
    expected_b = -0.25 * xr.dot(
        prior["x_shared"],
        design_b.assign_coords(trace_coords),
        dim="region",
    )
    xr.testing.assert_allclose(
        prior["mu_a"],
        expected_a.transpose(*prior["mu_a"].dims).rename("mu_a"),
    )
    xr.testing.assert_allclose(
        prior["mu_b"],
        expected_b.transpose(*prior["mu_b"].dims).rename("mu_b"),
    )
    xr.testing.assert_allclose(
        prior["mu"],
        (prior["mu_a"] + prior["mu_b"]).rename("mu"),
    )


def test_compile_loop_sum_separates_user_state_from_reparameterized_latent() -> None:
    """Compiled state metadata distinguishes the physical state from its latent."""
    design = xr.DataArray(
        [[1.0]],
        dims=("region", "nmeasure"),
        coords={"region": [0], "nmeasure": [0]},
    )
    plan = _FluxPlan(
        states=(
            _StatePlan(
                state_id="flux",
                variable_name="x_flux",
                prior_args={
                    "pdf": "lognormal",
                    "mean": 1.0,
                    "stdev": 0.2,
                    "reparameterise": True,
                },
            ),
        ),
        terms=(
            _ForwardTermPlan(
                term_id="flux",
                state_id="flux",
                design=design,
                data_name="h_flux",
                deterministic_name="mu",
            ),
        ),
    )

    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        compiled = _compile_loop_sum(plan)

    assert compiled.states["flux"] is model["x_flux"]
    assert compiled.latents["flux"] is model["x_flux_latent"]


@pytest.mark.parametrize(
    ("state_id", "coefficient", "prior_args", "error"),
    [
        ("missing", 1.0, {"pdf": "normal", "mu": 1.0, "sigma": 0.2}, "unknown state IDs"),
        ("shared", np.inf, {"pdf": "normal", "mu": 1.0, "sigma": 0.2}, "finite scalar"),
        ("shared", 1.0, {"pdf": "not-a-distribution"}, "prior.*invalid"),
    ],
)
def test_compile_loop_sum_rejects_invalid_plan_before_graph_mutation(
    state_id: str,
    coefficient: float,
    prior_args: dict[str, Any],
    error: str,
) -> None:
    """Invalid references, coefficients, and priors do not mutate the active graph."""
    design = xr.DataArray(
        [[1.0]],
        dims=("region", "nmeasure"),
        coords={"region": [0], "nmeasure": [0]},
    )
    plan = _FluxPlan(
        states=(
            _StatePlan(
                state_id="shared",
                variable_name="x_shared",
                prior_args=prior_args,
            ),
        ),
        terms=(
            _ForwardTermPlan(
                term_id="term",
                state_id=state_id,
                design=design,
                data_name="h_term",
                deterministic_name="mu_term",
                coefficient=coefficient,
            ),
        ),
    )

    with pm.Model() as model, pytest.raises(ValueError, match=error):
        _compile_loop_sum(plan)

    assert model.named_vars == {}


def test_compile_loop_sum_rejects_global_coordinate_conflicts_before_mutation() -> None:
    """Repeated backend dimensions must carry one globally consistent coordinate."""
    obs = ["obs0", "obs1"]
    plan = _FluxPlan(
        states=(
            _StatePlan("a", "x_a", {"pdf": "normal", "mu": 1.0, "sigma": 0.2}),
            _StatePlan("b", "x_b", {"pdf": "normal", "mu": 1.0, "sigma": 0.2}),
        ),
        terms=(
            _ForwardTermPlan(
                "a",
                "a",
                xr.DataArray(
                    np.ones((2, 2)),
                    dims=("region", "nmeasure"),
                    coords={"region": ["a0", "a1"], "nmeasure": obs},
                ),
                "h_a",
                "mu_a",
            ),
            _ForwardTermPlan(
                "b",
                "b",
                xr.DataArray(
                    np.ones((2, 2)),
                    dims=("region", "nmeasure"),
                    coords={"region": ["b0", "b1"], "nmeasure": obs},
                ),
                "h_b",
                "mu_b",
            ),
        ),
    )

    with (
        pm.Model() as model,
        pytest.raises(ValueError, match="incompatible global coordinates"),
    ):
        _compile_loop_sum(plan)

    assert model.named_vars == {}


def test_compile_loop_sum_preserves_term_order_independent_of_state_order() -> None:
    """The strategy follows term order while reusing independently declared states."""
    design = xr.DataArray(
        [[1.0]],
        dims=("region", "nmeasure"),
        coords={"region": [0], "nmeasure": [0]},
    )
    plan = _FluxPlan(
        states=(
            _StatePlan("b", "x_b", {"pdf": "normal", "mu": 1.0, "sigma": 0.2}),
            _StatePlan("a", "x_a", {"pdf": "normal", "mu": 1.0, "sigma": 0.2}),
        ),
        terms=(
            _ForwardTermPlan("a1", "a", design, "h_a1", "mu_a1"),
            _ForwardTermPlan("b1", "b", design, "h_b1", "mu_b1"),
            _ForwardTermPlan("a2", "a", design, "h_a2", "mu_a2"),
        ),
    )

    with pm.Model():
        compiled = _compile_loop_sum(plan)

    assert list(compiled.terms) == ["a1", "b1", "a2"]


def test_compile_loop_sum_resolves_zero_activity_across_shared_state_terms() -> None:
    """A shared state is pruned only when every effective term column is zero."""
    coords = {"region": ["first", "second", "zero"], "nmeasure": [0, 1]}
    plan = _FluxPlan(
        states=(_StatePlan("shared", "x_shared", {"pdf": "normal", "mu": 1.0, "sigma": 0.2}),),
        terms=(
            _ForwardTermPlan(
                "first",
                "shared",
                xr.DataArray(
                    [[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]], dims=("region", "nmeasure"), coords=coords
                ),
                "h_first",
                "mu_first",
            ),
            _ForwardTermPlan(
                "second",
                "shared",
                xr.DataArray(
                    [[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]], dims=("region", "nmeasure"), coords=coords
                ),
                "h_second",
                "mu_second",
            ),
        ),
    )

    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        compiled = _compile_loop_sum(plan)

    np.testing.assert_array_equal(model["x_shared_is_active"].eval(), [True, True, False])
    assert model["x_shared_active"].eval().shape == (2,)
    assert compiled.latents["shared"] is model["x_shared_active"]


def test_build_rhime_model_contains_expected_variables(
    rhime_inv_inputs: xr.Dataset, builder_args: dict
) -> None:
    """Build the low-level model with explicitly prepared sigma alignment."""
    model = build_rhime_model(rhime_inv_inputs, **builder_args)

    assert isinstance(model, pm.Model)
    expected = {"x", "mu", "bc", "mu_bc", "sigma", "epsilon", "y"}
    assert expected.issubset(model.named_vars)


def test_modern_inv_inputs_omit_component_owned_sigma_index(rhime_inv_inputs: xr.Dataset) -> None:
    """Modern inversion inputs omit component-owned sigma period indexes."""
    assert "sigma_freq_index" not in rhime_inv_inputs
    assert "sigma_period_index" not in rhime_inv_inputs


def test_build_rhime_multisector_model_contains_expected_variables(
    multisector_inv_inputs: xr.Dataset, builder_args: dict
) -> None:
    """Build the multisector model with explicitly prepared sigma alignment."""
    sectors = ["total-ukghg-edgar7", "sector-2"]
    model = build_rhime_multisector_model(multisector_inv_inputs, sectors=sectors, **builder_args)

    expected = {
        "x_total_ukghg_edgar7",
        "mu_total_ukghg_edgar7",
        "x_sector_2",
        "mu_sector_2",
        "mu",
        "bc",
        "mu_bc",
        "sigma",
        "epsilon",
        "y",
    }
    assert expected.issubset(model.named_vars)
    region_coord = model.coords["region"]
    assert region_coord is not None
    assert len(region_coord) == multisector_inv_inputs.sizes["region"]


def test_build_rhime_multisector_model_uses_sector_names_for_variables(
    multisector_inv_inputs: xr.Dataset, builder_args: dict
) -> None:
    """Distinct sector priors retain named states and additive mu deterministics."""
    sector_priors = {
        "FF": {"pdf": "uniform", "lower": 1.0, "upper": 2.0},
        "ocean": {"pdf": "uniform", "lower": 10.0, "upper": 11.0},
    }
    model = build_rhime_multisector_model(
        multisector_inv_inputs,
        sectors=["FF", "ocean"],
        sector_sources={"FF": "total-ukghg-edgar7", "ocean": "sector-2"},
        sector_variable_suffixes={"FF": "ff", "ocean": "ocean"},
        sector_priors=sector_priors,
        **builder_args,
    )

    expected_trace_names = {
        "x_ff",
        "mu_ff",
        "x_ocean",
        "mu_ocean",
        "mu",
    }
    expected_model_names = expected_trace_names | {
        "bc",
        "mu_bc",
        "sigma",
        "epsilon",
        "y",
    }
    assert expected_model_names.issubset(model.named_vars)
    free_rv_names = [rv.name for rv in model.free_RVs]
    assert free_rv_names.count("x_ff") == 1
    assert free_rv_names.count("x_ocean") == 1

    with model:
        trace = pm.sample_prior_predictive(
            draws=4,
            var_names=sorted(expected_trace_names),
            random_seed=412,
        )

    prior = cast(Any, trace).prior
    assert set(prior.data_vars) == expected_trace_names
    assert np.all((prior["x_ff"] >= 1.0) & (prior["x_ff"] <= 2.0))
    assert np.all((prior["x_ocean"] >= 10.0) & (prior["x_ocean"] <= 11.0))
    np.testing.assert_allclose(prior["mu"], prior["mu_ff"] + prior["mu_ocean"])


def test_build_rhime_multisector_model_selects_sources_by_label(
    multisector_inv_inputs: xr.Dataset,
    builder_args: dict,
) -> None:
    """Sector routing is independent of the prepared source-coordinate order."""
    reversed_inputs = multisector_inv_inputs.sel(source=["sector-2", "total-ukghg-edgar7"])

    model = build_rhime_multisector_model(
        reversed_inputs,
        sectors=["FF", "ocean"],
        sector_sources={"FF": "total-ukghg-edgar7", "ocean": "sector-2"},
        sector_priors={
            "FF": {"pdf": "normal", "mu": 1.0, "sigma": 0.2},
            "ocean": {"pdf": "normal", "mu": 1.0, "sigma": 0.3},
        },
        **builder_args,
    )

    expected_ff = reversed_inputs["H"].sel(source="total-ukghg-edgar7").transpose("nmeasure", "region")
    expected_ocean = reversed_inputs["H"].sel(source="sector-2").transpose("nmeasure", "region")
    np.testing.assert_allclose(model["hx_ff"].get_value(), expected_ff.values)
    np.testing.assert_allclose(model["hx_ocean"].get_value(), expected_ocean.values)


def test_concrete_and_compiled_multisector_models_accept_gathered_ragged_states(
    multisector_inv_inputs: xr.Dataset,
    builder_args: dict,
) -> None:
    """Both builders preserve and restore labelled source-specific states."""
    ff_labels = ["north", "south"]
    ocean_labels = ["atlantic", "pacific", "indian"]
    state_index = pd.MultiIndex.from_tuples(
        [
            *(("ff-inventory", label) for label in ff_labels),
            *(("ocean-inventory", label) for label in ocean_labels),
        ],
        names=["source", "region_in_source"],
    )
    nmeasure = multisector_inv_inputs.sizes["nmeasure"]
    values = np.arange(len(state_index) * nmeasure, dtype=float).reshape(len(state_index), nmeasure)
    inv_inputs = multisector_inv_inputs.drop_vars("H")
    inv_inputs = inv_inputs.drop_dims([dim for dim in ("region", "source") if dim in inv_inputs.dims])
    inv_inputs["H"] = xr.DataArray(
        values,
        dims=("state", "nmeasure"),
        coords={
            **xr.Coordinates.from_pandas_multiindex(state_index, "state"),
            "nmeasure": multisector_inv_inputs.coords["nmeasure"],
            "basis_group": ("state", ["fixed", "active", "fixed", "active", "active"]),
        },
    )
    ff_active = xr.DataArray(
        [False, True],
        dims="state",
        coords={"state": ["south", "north"]},
    )
    ff_fixed = xr.DataArray(
        [2.0, 3.0],
        dims="state",
        coords={"state": ["south", "north"]},
    )

    kwargs = {
        "sectors": ["FF", "ocean"],
        "sector_sources": {"FF": "ff-inventory", "ocean": "ocean-inventory"},
        "sector_priors": {
            "FF": {
                "pdf": "uniform",
                "lower": xr.DataArray(
                    [20.0, 10.0],
                    dims="state",
                    coords={"state": ["south", "north"]},
                ),
                "upper": xr.DataArray(
                    [21.0, 11.0],
                    dims="state",
                    coords={"state": ["south", "north"]},
                ),
            },
            "ocean": {"pdf": "normal", "mu": 1.0, "sigma": 0.3},
        },
        "sector_state_activities": {
            "FF": StateActivity(active=ff_active, fixed_value=ff_fixed),
            "ocean": StateActivity(fixed_groups=("fixed",)),
        },
        **builder_args,
    }
    model = build_rhime_multisector_model(inv_inputs, **kwargs)
    compiled = rhime_models_module._build_compiled_rhime_multisector_model(
        inv_inputs,
        **kwargs,
    )

    assert set(model.named_vars) == set(compiled.named_vars)
    assert model.named_vars_to_dims == compiled.named_vars_to_dims
    np.testing.assert_allclose(model["hx_ff"].get_value(), values[:2].T)
    np.testing.assert_allclose(model["hx_ocean"].get_value(), values[2:].T)
    np.testing.assert_allclose(compiled["hx_ff"].get_value(), values[:2].T)
    np.testing.assert_allclose(compiled["hx_ocean"].get_value(), values[2:].T)
    np.testing.assert_allclose(model["hbc"].get_value(), compiled["hbc"].get_value())
    assert model.named_vars_to_dims["x_ff"] == ("state_ff",)
    assert model.named_vars_to_dims["x_ocean"] == ("state_ocean",)
    assert compiled.named_vars_to_dims["x_ff"] == ("state_ff",)
    assert compiled.named_vars_to_dims["x_ocean"] == ("state_ocean",)
    assert model.coords["state_ff"] == (0, 1)
    assert model.coords["state_ocean"] == (0, 1, 2)
    assert compiled.coords["state_ff"] == (0, 1)
    assert compiled.coords["state_ocean"] == (0, 1, 2)

    concrete_registry = models.get_coord_registry(model)
    compiled_registry = models.get_coord_registry(compiled)
    assert concrete_registry is not None
    assert compiled_registry is not None
    for registry in (concrete_registry, compiled_registry):
        assert list(registry.original_coords["state_ff"]) == ff_labels
        assert list(registry.original_coords["state_ocean"]) == ocean_labels
        assert registry.original_coords["nmeasure"].equals(inv_inputs.indexes["nmeasure"])
        np.testing.assert_array_equal(registry.auxiliary_coords["basis_group_ff"], ["fixed", "active"])
        np.testing.assert_array_equal(
            registry.auxiliary_coords["basis_group_ocean"],
            ["fixed", "active", "active"],
        )
        assert list(registry.original_coords["state_ff_x_ff_active"]) == ["north"]
        assert list(registry.original_coords["state_ocean_x_ocean_active"]) == [
            "pacific",
            "indian",
        ]

    var_names = [
        "x_ff_active",
        "x_ff",
        "mu_ff",
        "x_ocean_active",
        "x_ocean",
        "mu_ocean",
        "mu",
    ]
    with model:
        concrete_prior = pm.sample_prior_predictive(
            draws=2,
            var_names=var_names,
            random_seed=535,
        )
    with compiled:
        compiled_prior = pm.sample_prior_predictive(
            draws=2,
            var_names=var_names,
            random_seed=535,
        )
    concrete_prior = models.restore_inferencedata_coords(
        cast(az.InferenceData, concrete_prior),
        concrete_registry,
    )
    compiled_prior = models.restore_inferencedata_coords(
        cast(az.InferenceData, compiled_prior),
        compiled_registry,
    )
    concrete_dataset = concrete_prior.prior
    compiled_dataset = compiled_prior.prior
    xr.testing.assert_allclose(concrete_dataset, compiled_dataset)
    assert list(concrete_dataset["state_ff"].values) == ff_labels
    assert list(concrete_dataset["state_ocean"].values) == ocean_labels
    assert list(concrete_dataset["state_ff_x_ff_active"].values) == ["north"]
    assert list(concrete_dataset["state_ocean_x_ocean_active"].values) == ["pacific", "indian"]
    assert np.all((concrete_dataset["x_ff_active"] >= 10.0) & (concrete_dataset["x_ff_active"] <= 11.0))
    np.testing.assert_allclose(concrete_dataset["x_ff"].sel(state_ff="south"), 2.0)
    assert concrete_dataset.indexes["nmeasure"].equals(inv_inputs.indexes["nmeasure"])
    xr.testing.assert_allclose(
        concrete_dataset["mu"],
        concrete_dataset["mu_ff"] + concrete_dataset["mu_ocean"],
    )


def test_build_rhime_multisector_model_rejects_ungathered_source_state(
    multisector_inv_inputs: xr.Dataset,
    builder_args: dict,
) -> None:
    """Source labels on a state axis require the canonical gathered MultiIndex."""
    nmeasure = multisector_inv_inputs.sizes["nmeasure"]
    inv_inputs = multisector_inv_inputs.drop_vars("H")
    inv_inputs = inv_inputs.drop_dims([dim for dim in ("region", "source") if dim in inv_inputs.dims])
    inv_inputs["H"] = xr.DataArray(
        np.ones((2, nmeasure)),
        dims=("state", "nmeasure"),
        coords={
            "state": [0, 1],
            "source": ("state", ["ff-inventory", "ocean-inventory"]),
            "nmeasure": multisector_inv_inputs.coords["nmeasure"],
        },
    )

    with pytest.raises(ValueError, match="MultiIndex containing a 'source' level"):
        build_rhime_multisector_model(
            inv_inputs,
            sectors=["FF", "ocean"],
            sector_sources={"FF": "ff-inventory", "ocean": "ocean-inventory"},
            **builder_args,
        )


def test_build_rhime_multisector_model_rejects_duplicate_gathered_states(
    multisector_inv_inputs: xr.Dataset,
    builder_args: dict,
) -> None:
    """Gathered scientific state coordinates must identify unique coefficients."""
    state_index = pd.MultiIndex.from_tuples(
        [
            ("ff-inventory", 0),
            ("ff-inventory", 0),
            ("ocean-inventory", 0),
        ],
        names=["source", "region_in_source"],
    )
    nmeasure = multisector_inv_inputs.sizes["nmeasure"]
    inv_inputs = multisector_inv_inputs.drop_vars("H")
    inv_inputs = inv_inputs.drop_dims([dim for dim in ("region", "source") if dim in inv_inputs.dims])
    inv_inputs["H"] = xr.DataArray(
        np.ones((len(state_index), nmeasure)),
        dims=("state", "nmeasure"),
        coords={
            **xr.Coordinates.from_pandas_multiindex(state_index, "state"),
            "nmeasure": multisector_inv_inputs.coords["nmeasure"],
        },
    )

    with pytest.raises(ValueError, match="unique state labels.*duplicate state.*ff-inventory"):
        build_rhime_multisector_model(
            inv_inputs,
            sectors=["FF", "ocean"],
            sector_sources={"FF": "ff-inventory", "ocean": "ocean-inventory"},
            **builder_args,
        )


def test_build_rhime_multisector_model_rejects_duplicate_prepared_sources(
    multisector_inv_inputs: xr.Dataset,
    builder_args: dict,
) -> None:
    """Duplicate source coordinates fail before label selection becomes ambiguous."""
    duplicate_sources = multisector_inv_inputs.sel(source=["total-ukghg-edgar7", "total-ukghg-edgar7"])

    with pytest.raises(ValueError, match="duplicate source 'total-ukghg-edgar7'"):
        build_rhime_multisector_model(duplicate_sources, **builder_args)


def test_build_rhime_multisector_model_rejects_padded_source_regions(
    multisector_inv_inputs: xr.Dataset,
    builder_args: dict,
) -> None:
    """Direct builders reject latent state elements introduced only by padding."""
    region_count = multisector_inv_inputs.sizes["region"]
    padded_inputs = multisector_inv_inputs.assign_coords(
        source_region_count=(
            "source",
            [region_count - 1, region_count],
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            f"Sector 'FF' -> source 'total-ukghg-edgar7' declares {region_count - 1} "
            f"region elements.*prepared H has {region_count}"
        ),
    ):
        build_rhime_multisector_model(
            padded_inputs,
            sectors=["FF", "ocean"],
            sector_sources={"FF": "total-ukghg-edgar7", "ocean": "sector-2"},
            **builder_args,
        )


def test_build_rhime_multisector_model_allows_prior_only_regions(
    multisector_inv_inputs: xr.Dataset,
    builder_args: dict,
) -> None:
    """Zero sensitivity is not mistaken for padding when layout metadata matches."""
    region_count = multisector_inv_inputs.sizes["region"]
    inv_inputs = multisector_inv_inputs.assign_coords(
        source_region_count=("source", [region_count, region_count])
    ).copy(deep=True)
    inv_inputs["H"].loc[{"source": "total-ukghg-edgar7", "region": 0}] = 0

    model = build_rhime_multisector_model(
        inv_inputs,
        sectors=["FF", "ocean"],
        sector_sources={"FF": "total-ukghg-edgar7", "ocean": "sector-2"},
        **builder_args,
    )

    assert "x_ff" in model.named_vars


def test_build_rhime_multisector_model_rejects_duplicate_sector_source_mappings(
    multisector_inv_inputs: xr.Dataset,
    builder_args: dict,
) -> None:
    """Independent sector states cannot select the same source sensitivity."""
    with pytest.raises(ValueError, match="source 'total-ukghg-edgar7'.*\\['FF', 'other'\\]"):
        build_rhime_multisector_model(
            multisector_inv_inputs,
            sectors=["FF", "other"],
            sector_sources={"FF": "total-ukghg-edgar7", "other": "total-ukghg-edgar7"},
            **builder_args,
        )


def test_build_rhime_multisector_model_names_sector_and_missing_source(
    multisector_inv_inputs: xr.Dataset,
    builder_args: dict,
) -> None:
    """Missing prepared data errors retain semantic sector and source identities."""
    with pytest.raises(ValueError, match="sector 'FF' -> source 'missing-inventory'"):
        build_rhime_multisector_model(
            multisector_inv_inputs,
            sectors=["FF", "ocean"],
            sector_sources={"FF": "missing-inventory", "ocean": "sector-2"},
            **builder_args,
        )


@pytest.mark.parametrize(
    ("sector_priors", "error_fragment"),
    [
        (
            {"FF": {"pdf": "normal", "mu": 1.0, "sigma": 0.2}},
            "missing sector prior\\(s\\): \\['ocean'\\]",
        ),
        (
            {
                "FF": {"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                "ocean": {"pdf": "normal", "mu": 1.0, "sigma": 0.3},
                "typo": {"pdf": "normal", "mu": 1.0, "sigma": 0.4},
            },
            "unused sector prior key\\(s\\): \\['typo'\\]",
        ),
    ],
)
def test_build_rhime_multisector_model_requires_exact_sector_prior_keys(
    multisector_inv_inputs: xr.Dataset,
    builder_args: dict,
    sector_priors: dict[str, dict[str, Any]],
    error_fragment: str,
) -> None:
    """Explicit per-sector priors must be complete and contain no unused keys."""
    with pytest.raises(ValueError, match=error_fragment):
        build_rhime_multisector_model(
            multisector_inv_inputs,
            sectors=["FF", "ocean"],
            sector_sources={"FF": "total-ukghg-edgar7", "ocean": "sector-2"},
            sector_priors=sector_priors,
            **builder_args,
        )


def test_concrete_multisector_model_rejects_reparameterized_name_collisions(
    multisector_inv_inputs: xr.Dataset,
    builder_args: dict,
) -> None:
    """Concrete composition preserves PyMC's generated-name collision check."""
    reparameterized_prior = {
        "pdf": "lognormal",
        "mean": 1.0,
        "stdev": 0.2,
        "reparameterise": True,
    }

    with pytest.raises(ValueError, match="x_ff_latent.*already exists"):
        build_rhime_multisector_model(
            multisector_inv_inputs,
            sectors=["FF", "other"],
            sector_sources={"FF": "total-ukghg-edgar7", "other": "sector-2"},
            sector_variable_suffixes={"FF": "ff", "other": "ff_latent"},
            sector_priors={"FF": reparameterized_prior, "other": reparameterized_prior},
            **builder_args,
        )


def test_compiled_multisector_model_preflights_reparameterized_name_collisions(
    multisector_inv_inputs: xr.Dataset,
    builder_args: dict,
) -> None:
    """The opt-in compiler retains whole-plan generated-name validation."""
    reparameterized_prior = {
        "pdf": "lognormal",
        "mean": 1.0,
        "stdev": 0.2,
        "reparameterise": True,
    }

    with pytest.raises(ValueError, match="backend names"):
        rhime_models_module._build_compiled_rhime_multisector_model(
            multisector_inv_inputs,
            sectors=["FF", "other"],
            sector_sources={"FF": "total-ukghg-edgar7", "other": "sector-2"},
            sector_variable_suffixes={"FF": "ff", "other": "ff_latent"},
            sector_priors={"FF": reparameterized_prior, "other": reparameterized_prior},
            **builder_args,
        )


def test_public_rhime_builders_default_to_concrete_construction(
    monkeypatch: pytest.MonkeyPatch,
    rhime_inv_inputs: xr.Dataset,
    multisector_inv_inputs: xr.Dataset,
    builder_args: dict,
) -> None:
    """Public concrete builders do not invoke the private compiler."""

    def fail_compile(plan: _FluxPlan) -> Any:
        raise AssertionError("Concrete builders must not invoke the flux compiler.")

    monkeypatch.setattr(rhime_models_module, "_compile_loop_sum", fail_compile)

    build_rhime_model(rhime_inv_inputs, **builder_args)
    build_rhime_multisector_model(
        multisector_inv_inputs,
        sectors=["FF", "ocean"],
        sector_sources={"FF": "total-ukghg-edgar7", "ocean": "sector-2"},
        sector_variable_suffixes={"FF": "ff", "ocean": "ocean"},
        **builder_args,
    )

    model_spec = RhimeModelSpec(species="ch4", domain="EUROPE", sectors=())
    assert model_spec.builder_strategy == "concrete"
    assert (
        inspect.signature(RhimeModelSpec).parameters["builder_strategy"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


def test_opt_in_compiled_builders_share_loop_sum_compiler(
    monkeypatch: pytest.MonkeyPatch,
    rhime_inv_inputs: xr.Dataset,
    multisector_inv_inputs: xr.Dataset,
    builder_args: dict,
) -> None:
    """Both opt-in builders normalize fluxes through the private compiler."""
    captured_plans: list[_FluxPlan] = []

    def compile_spy(plan: _FluxPlan) -> Any:
        captured_plans.append(plan)
        return rhime_compiler_module._compile_loop_sum(plan)

    monkeypatch.setattr(rhime_models_module, "_compile_loop_sum", compile_spy)

    rhime_models_module._build_compiled_rhime_model(rhime_inv_inputs, **builder_args)
    rhime_models_module._build_compiled_rhime_multisector_model(
        multisector_inv_inputs,
        sectors=["FF", "ocean"],
        sector_sources={"FF": "total-ukghg-edgar7", "ocean": "sector-2"},
        sector_variable_suffixes={"FF": "ff", "ocean": "ocean"},
        **builder_args,
    )

    assert len(captured_plans) == 2
    standard_plan, multisector_plan = captured_plans
    assert [state.variable_name for state in standard_plan.states] == ["x"]
    assert [term.deterministic_name for term in standard_plan.terms] == ["mu"]
    assert [state.variable_name for state in multisector_plan.states] == ["x_ff", "x_ocean"]
    assert [term.deterministic_name for term in multisector_plan.terms] == ["mu_ff", "mu_ocean"]


def test_concrete_and_compiled_standard_models_are_equivalent(
    rhime_inv_inputs: xr.Dataset,
    builder_args: dict,
) -> None:
    """Float32 sensitivities and the default reparameterized prior retain parity."""
    inv_inputs = rhime_inv_inputs.copy()
    inv_inputs["H"] = inv_inputs["H"].astype(np.float32)
    kwargs = {**builder_args, "x_prior": None}
    concrete = build_rhime_model(inv_inputs, **kwargs)
    compiled = rhime_models_module._build_compiled_rhime_model(inv_inputs, **kwargs)

    assert set(concrete.named_vars) == set(compiled.named_vars)
    assert concrete.named_vars_to_dims == compiled.named_vars_to_dims
    assert {"x", "x_latent", "mu"}.issubset(concrete.named_vars)
    assert concrete.named_vars_to_dims["x"] == concrete.named_vars_to_dims["x_latent"]
    np.testing.assert_allclose(concrete["hx"].get_value(), compiled["hx"].get_value())
    np.testing.assert_allclose(concrete["hbc"].get_value(), compiled["hbc"].get_value())
    assert concrete["hx"].get_value().dtype == np.dtype("float32")
    assert compiled["hx"].get_value().dtype == np.dtype("float32")

    var_names = ["x_latent", "x", "mu", "mu_bc", "sigma", "epsilon"]
    with concrete:
        concrete_prior = pm.sample_prior_predictive(
            draws=2,
            var_names=var_names,
            random_seed=402,
        )
    with compiled:
        compiled_prior = pm.sample_prior_predictive(
            draws=2,
            var_names=var_names,
            random_seed=402,
        )
    concrete_dataset = cast(Any, concrete_prior).prior
    compiled_dataset = cast(Any, compiled_prior).prior
    xr.testing.assert_allclose(concrete_dataset, compiled_dataset)

    registered_h = xr.DataArray(
        concrete["hx"].get_value(),
        dims=("nmeasure", "region"),
        coords={
            "nmeasure": concrete_dataset["nmeasure"],
            "region": concrete_dataset["region"],
        },
    )
    expected_mu = xr.dot(concrete_dataset["x"], registered_h, dim="region")
    xr.testing.assert_allclose(
        concrete_dataset["mu"],
        expected_mu.transpose(*concrete_dataset["mu"].dims).rename("mu"),
    )


def test_concrete_and_compiled_standard_models_prune_exact_zero_states(
    rhime_inv_inputs: xr.Dataset,
    builder_args: dict,
) -> None:
    """Concrete and compiled graphs retain full state around an active prior."""
    inv_inputs = rhime_inv_inputs.copy(deep=True)
    first_region = inv_inputs["H"].coords["region"][0]
    inv_inputs["H"].loc[{"region": first_region}] = 0.0
    kwargs = {
        **builder_args,
        "use_bc": False,
        "no_model_error": True,
        "state_activity": StateActivity(),
    }

    concrete = build_rhime_model(inv_inputs, **kwargs)
    compiled = rhime_models_module._build_compiled_rhime_model(inv_inputs, **kwargs)

    assert set(concrete.named_vars) == set(compiled.named_vars)
    assert concrete.named_vars_to_dims == compiled.named_vars_to_dims
    np.testing.assert_array_equal(
        concrete["x_is_active"].eval(),
        compiled["x_is_active"].eval(),
    )
    assert not bool(concrete["x_is_active"].eval()[0])
    assert concrete["x_active"] in concrete.free_RVs
    assert compiled["x_active"] in compiled.free_RVs
    with concrete:
        concrete_prior = pm.sample_prior_predictive(
            draws=2,
            var_names=["x_active", "x", "mu"],
            random_seed=417,
        )
    with compiled:
        compiled_prior = pm.sample_prior_predictive(
            draws=2,
            var_names=["x_active", "x", "mu"],
            random_seed=417,
        )
    xr.testing.assert_allclose(
        cast(Any, concrete_prior).prior,
        cast(Any, compiled_prior).prior,
    )


def test_concrete_and_compiled_models_support_all_fixed_flux_and_bc(
    rhime_inv_inputs: xr.Dataset,
    builder_args: dict,
) -> None:
    """All-fixed flux and BC states retain full deterministic contributions."""
    kwargs = {
        **builder_args,
        "no_model_error": True,
        "state_activity": StateActivity(active=False, fixed_value=2.0),
        "bc_state_activity": StateActivity(active=False, fixed_value=1.5),
    }

    concrete = build_rhime_model(rhime_inv_inputs, **kwargs)
    compiled = rhime_models_module._build_compiled_rhime_model(rhime_inv_inputs, **kwargs)

    assert set(concrete.named_vars) == set(compiled.named_vars)
    assert concrete.named_vars_to_dims == compiled.named_vars_to_dims
    for model in (concrete, compiled):
        assert model["x"] not in model.free_RVs
        assert model["bc"] not in model.free_RVs
        assert "x_active" not in model.named_vars
        assert "bc_active" not in model.named_vars
        np.testing.assert_allclose(model["x"].eval(), 2.0)
        np.testing.assert_allclose(model["bc"].eval(), 1.5)
        np.testing.assert_allclose(model["mu"].eval(), model["hx"].get_value() @ model["x"].eval())
        np.testing.assert_allclose(
            model["mu_bc"].eval(),
            model["hbc"].get_value() @ model["bc"].eval(),
        )


def test_bc_activity_is_opt_in_and_restores_exact_zero_columns(
    rhime_inv_inputs: xr.Dataset,
    builder_args: dict,
) -> None:
    """Explicit BC activity prunes zero columns while omission keeps the old graph."""
    inv_inputs = rhime_inv_inputs.copy(deep=True)
    first_region = inv_inputs["H_bc"].coords["bc_region"][0]
    inv_inputs["H_bc"].loc[{"bc_region": first_region}] = 0.0
    kwargs = {**builder_args, "no_model_error": True}

    default_model = build_rhime_model(inv_inputs, **kwargs)
    assert default_model["bc"] in default_model.free_RVs
    assert "bc_active" not in default_model.named_vars
    assert "bc_is_active" not in default_model.named_vars

    for model in (
        build_rhime_model(inv_inputs, bc_state_activity=StateActivity(), **kwargs),
        rhime_models_module._build_compiled_rhime_model(
            inv_inputs,
            bc_state_activity=StateActivity(),
            **kwargs,
        ),
    ):
        assert model["bc_active"] in model.free_RVs
        assert model["bc"] not in model.free_RVs
        assert not bool(model["bc_is_active"].eval()[0])
        assert model["bc"].eval()[0] == 1.0
        registry = models.get_coord_registry(model)
        assert registry is not None
        assert list(registry.original_coords["bc_region_bc_active"]) == list(
            inv_inputs["H_bc"].indexes["bc_region"][1:]
        )
        np.testing.assert_allclose(
            model["mu_bc"].eval(),
            model["hbc"].get_value() @ model["bc"].eval(),
        )


def test_compiled_multisector_model_rejects_unknown_state_activity_sector(
    multisector_inv_inputs: xr.Dataset,
    builder_args: dict,
) -> None:
    """Compiler normalization rejects activity overrides for absent sectors."""
    with pytest.raises(ValueError, match="unknown sector.*missing"):
        rhime_models_module._build_compiled_rhime_multisector_model(
            multisector_inv_inputs,
            sectors=["FF", "ocean"],
            sector_sources={"FF": "total-ukghg-edgar7", "ocean": "sector-2"},
            sector_state_activities={"missing": StateActivity(active=False)},
            **builder_args,
        )


def test_concrete_and_compiled_multisector_models_are_equivalent(
    multisector_inv_inputs: xr.Dataset,
    builder_args: dict,
) -> None:
    """Concrete and compiled multisector graphs preserve additive semantics."""
    kwargs = {
        "sectors": ["FF", "ocean"],
        "sector_sources": {"FF": "total-ukghg-edgar7", "ocean": "sector-2"},
        "sector_variable_suffixes": {"FF": "ff", "ocean": "ocean"},
        **builder_args,
    }
    concrete = build_rhime_multisector_model(multisector_inv_inputs, **kwargs)
    compiled = rhime_models_module._build_compiled_rhime_multisector_model(
        multisector_inv_inputs,
        **kwargs,
    )

    assert set(concrete.named_vars) == set(compiled.named_vars)
    assert concrete.named_vars_to_dims == compiled.named_vars_to_dims
    for name in ("hx_ff", "hx_ocean", "hbc"):
        np.testing.assert_allclose(concrete[name].get_value(), compiled[name].get_value())

    var_names = ["x_ff", "mu_ff", "x_ocean", "mu_ocean", "mu"]
    with concrete:
        concrete_prior = pm.sample_prior_predictive(
            draws=2,
            var_names=var_names,
            random_seed=403,
        )
    with compiled:
        compiled_prior = pm.sample_prior_predictive(
            draws=2,
            var_names=var_names,
            random_seed=403,
        )
    concrete_dataset = cast(Any, concrete_prior).prior
    compiled_dataset = cast(Any, compiled_prior).prior
    xr.testing.assert_allclose(concrete_dataset, compiled_dataset)
    xr.testing.assert_allclose(
        concrete_dataset["mu"],
        concrete_dataset["mu_ff"] + concrete_dataset["mu_ocean"],
    )


def test_build_rhime_multisector_model_requires_multiple_sectors(
    multisector_inv_inputs: xr.Dataset, builder_args: dict
) -> None:
    with pytest.raises(ValueError, match="at least two sectors"):
        build_rhime_multisector_model(
            multisector_inv_inputs,
            sectors=["total-ukghg-edgar7"],
            **builder_args,
        )


def test_models_exports_rhime_builders() -> None:
    assert models.build_rhime_model is build_rhime_model
    assert models.build_rhime_model_from_spec is build_rhime_model_from_spec
    assert models.build_rhime_multisector_model is build_rhime_multisector_model
    assert models.build_rhime_multisector_model_from_spec is build_rhime_multisector_model_from_spec
    assert models.safe_pymc_name is safe_pymc_name
    assert isinstance(models.DEFAULT_X_PRIOR, dict)
    assert isinstance(models.DEFAULT_BC_PRIOR, dict)
    assert isinstance(models.DEFAULT_SIGMA_PRIOR, dict)
    assert isinstance(models.DEFAULT_OFFSET_PRIOR, dict)


def test_resolve_flux_sources_prefers_new_name() -> None:
    assert resolve_flux_sources(flux_sources=["new"], emissions_name=["legacy"]) == ["new"]
    assert resolve_flux_sources(emissions_name=["legacy"]) == ["legacy"]


def test_direct_sector_spec_records_source_backing() -> None:
    """Direct Python specs distinguish model sectors from OpenGHG sources."""
    sector = SectorSpec(
        name="FF",
        flux_source="ff-inventory",
        x_prior={"pdf": "lognormal", "mean": 1.0, "stdev": 1.0},
        variable_suffix="ff",
    )

    assert sector.name == "FF"
    assert sector.flux_source == "ff-inventory"


def test_public_rhime_dataclasses_keep_existing_positional_order() -> None:
    """New default fields do not intercept existing positional construction."""
    sector = SectorSpec(
        name="FF",
        flux_source="ff-inventory",
        x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
        variable_suffix="ff",
    )
    model_spec = RhimeModelSpec(
        "ch4",
        "EUROPE",
        (sector,),
        True,
        True,
        False,
        False,
        False,
        1.99,
        None,
        None,
        None,
        None,
    )
    output_spec = RhimeOutputSpec(output_format="none")
    run_spec = RhimeRunSpec(
        "2019-01-01",
        "2019-01-02",
        ("TAC",),
        ("1h",),
        model_spec,
        output_spec,
        True,
    )
    output_metadata = {"path": "trace.nc"}
    result = RhimeResult(
        run_spec,
        model_spec,
        output_spec,
        _minimal_inv_inputs(),
        cast(Any, object()),
        output_metadata,
    )

    assert run_spec.split_by_sectors is True
    assert model_spec.state_activity is None
    assert model_spec.sector_state_activities is None
    assert not hasattr(run_spec, "sampler")
    assert not hasattr(run_spec, "sampling")
    assert result.output_metadata == output_metadata
    assert result.sampler == RhimeSampler()


@pytest.mark.parametrize("sector_count", [1, 2])
def test_run_rhime_from_prepared_inputs_routes_without_preparation(
    monkeypatch: pytest.MonkeyPatch,
    sector_count: int,
) -> None:
    """Prepared runs bypass preparation and select the builder from sector count."""
    model_spec, output_spec, run_spec = _minimal_output_specs(output_format="none")
    sectors = model_spec.sectors
    if sector_count == 2:
        sectors += (
            SectorSpec(
                name="Ocean",
                flux_source="ocean-inventory",
                x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                variable_suffix="ocean",
            ),
        )
    model_spec = RhimeModelSpec(
        species=model_spec.species,
        domain=model_spec.domain,
        sectors=sectors,
    )
    run_spec = RhimeRunSpec(
        run_spec.start_date,
        run_spec.end_date,
        ("STALE",),
        ("24h",),
        model_spec,
        output_spec,
        split_by_sectors=sector_count > 1,
    )
    inv_inputs = _minimal_output_inv_inputs()
    if sector_count == 1:
        inv_inputs["H"] = inv_inputs["H"].assign_coords(source=sectors[0].flux_source)
    else:
        inv_inputs["H"] = xr.concat(
            [inv_inputs["H"].expand_dims(source=[sector.flux_source]) for sector in model_spec.sectors],
            dim="source",
        )
    prepared = RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=_fake_basis_functions(),
        site_metadata=_prepared_site_metadata(),
    )
    sampler = RhimeSampler(draws=7, sample_prior_predictive=False, sample_posterior_predictive=False)
    built_model = pm.Model()
    builder_calls: list[str] = []
    output_calls: list[str] = []

    def fail_prepare(**kwargs: Any) -> None:
        raise AssertionError("prepared runs must not call prepare_rhime_inputs")

    def fail_setup(**kwargs: Any) -> None:
        raise AssertionError("prepared runs must not normalize runner parameters")

    def fail_config(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("prepared runs must not normalize config parameters")

    def build_standard(inv_inputs: xr.Dataset, spec: RhimeModelSpec) -> pm.Model:
        builder_calls.append("standard")
        return built_model

    def build_multisector(inv_inputs: xr.Dataset, spec: RhimeModelSpec) -> pm.Model:
        builder_calls.append("multisector")
        return built_model

    def fake_sample(self: RhimeSampler, model: pm.Model) -> az.InferenceData:
        assert self is sampler
        assert model is built_model
        return _minimal_output_idata()

    def make_standard_outputs(**kwargs: Any) -> rhime_outputs.RhimeOutputBundle:
        output_calls.append("standard")
        return rhime_outputs.RhimeOutputBundle()

    def make_multisector_outputs(**kwargs: Any) -> rhime_outputs.RhimeOutputBundle:
        output_calls.append("multisector")
        return rhime_outputs.RhimeOutputBundle()

    monkeypatch.setattr(rhime_module, "prepare_rhime_inputs", fail_prepare)
    monkeypatch.setattr(rhime_module, "_make_rhime_runner_setup", fail_setup)
    monkeypatch.setattr(rhime_module, "params_from_config", fail_config)
    monkeypatch.setattr(rhime_module, "build_rhime_model_from_spec", build_standard)
    monkeypatch.setattr(rhime_module, "build_rhime_multisector_model_from_spec", build_multisector)
    monkeypatch.setattr(RhimeSampler, "sample", fake_sample)
    monkeypatch.setattr(rhime_module, "make_standard_output_bundle", make_standard_outputs)
    monkeypatch.setattr(rhime_module, "make_multisector_output_bundle", make_multisector_outputs)

    result = run_rhime_from_prepared_inputs(
        prepared_inputs=prepared,
        run_spec=run_spec,
        sampler=sampler,
    )

    expected_route = "standard" if sector_count == 1 else "multisector"
    assert builder_calls == [expected_route]
    assert output_calls == [expected_route]
    assert result.sampler is sampler
    assert result.run_spec.sites == prepared.sites
    assert result.run_spec.averaging_period == prepared.averaging_period
    assert result.run_spec.split_by_sectors is (sector_count > 1)


@pytest.mark.parametrize(
    ("sector_count", "split_by_sectors"),
    [(1, True), (2, False)],
)
def test_run_rhime_from_prepared_inputs_rejects_layout_mode_mismatch_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    sector_count: int,
    split_by_sectors: bool,
) -> None:
    """Prepared runs reject disagreement between sector count and data layout."""
    model_spec, output_spec, run_spec = _minimal_output_specs(output_format="none")
    sectors = model_spec.sectors
    if sector_count == 2:
        sectors += (
            SectorSpec(
                name="Ocean",
                flux_source="ocean-inventory",
                x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                variable_suffix="ocean",
            ),
        )
    run_spec = RhimeRunSpec(
        run_spec.start_date,
        run_spec.end_date,
        run_spec.sites,
        run_spec.averaging_period,
        RhimeModelSpec(species=model_spec.species, domain=model_spec.domain, sectors=sectors),
        output_spec,
        split_by_sectors=split_by_sectors,
    )
    inv_inputs = _minimal_output_inv_inputs()
    if split_by_sectors:
        inv_inputs["H"] = inv_inputs["H"].expand_dims(source=[sectors[0].flux_source])
    prepared = RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=_fake_basis_functions(),
        site_metadata=_prepared_site_metadata(),
    )

    def fail_execution(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("layout validation must precede model building and sampling")

    monkeypatch.setattr(rhime_module, "build_rhime_model_from_spec", fail_execution)
    monkeypatch.setattr(rhime_module, "build_rhime_multisector_model_from_spec", fail_execution)
    monkeypatch.setattr(RhimeSampler, "sample", fail_execution)

    with pytest.raises(ValueError, match="split_by_sectors.*must agree"):
        run_rhime_from_prepared_inputs(prepared_inputs=prepared, run_spec=run_spec)


@pytest.mark.parametrize("sector_count", [1, 2])
def test_run_rhime_from_prepared_inputs_rejects_flag_h_layout_mismatch_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    sector_count: int,
) -> None:
    """Prepared runs reject a layout flag that disagrees with H dimensions."""
    model_spec, output_spec, run_spec = _minimal_output_specs(output_format="none")
    sectors = model_spec.sectors
    if sector_count == 2:
        sectors += (
            SectorSpec(
                name="Ocean",
                flux_source="ocean-inventory",
                x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                variable_suffix="ocean",
            ),
        )
    model_spec = RhimeModelSpec(
        species=model_spec.species,
        domain=model_spec.domain,
        sectors=sectors,
    )
    multisector = sector_count > 1
    run_spec = RhimeRunSpec(
        run_spec.start_date,
        run_spec.end_date,
        run_spec.sites,
        run_spec.averaging_period,
        model_spec,
        output_spec,
        split_by_sectors=multisector,
    )
    inv_inputs = _minimal_output_inv_inputs()
    if not multisector:
        inv_inputs["H"] = inv_inputs["H"].expand_dims(source=[sectors[0].flux_source])
    prepared = RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=_fake_basis_functions(),
        site_metadata=_prepared_site_metadata(),
    )

    def fail_execution(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("H layout validation must precede model building and sampling")

    monkeypatch.setattr(rhime_module, "build_rhime_model_from_spec", fail_execution)
    monkeypatch.setattr(rhime_module, "build_rhime_multisector_model_from_spec", fail_execution)
    monkeypatch.setattr(RhimeSampler, "sample", fail_execution)

    with pytest.raises(ValueError, match="split_by_sectors.*prepared `H` layout"):
        run_rhime_from_prepared_inputs(prepared_inputs=prepared, run_spec=run_spec)


def test_run_rhime_from_prepared_inputs_defaults_sampler_and_skips_none_output_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Prepared standard runs default the sampler and write nothing for none output."""
    model_spec, _, run_spec = _minimal_output_specs(output_format="none")
    output_spec = RhimeOutputSpec(
        output_format="none",
        output_path=str(tmp_path),
        output_name="prepared",
        save_trace=True,
        save_inversion_output=True,
    )
    run_spec = RhimeRunSpec(
        run_spec.start_date,
        run_spec.end_date,
        run_spec.sites,
        run_spec.averaging_period,
        model_spec,
        output_spec,
    )
    prepared = RhimePreparedInputs(
        inv_inputs=_minimal_output_inv_inputs(),
        basis_functions=_fake_basis_functions(),
        site_metadata=_prepared_site_metadata(),
    )
    built_model = pm.Model()
    sampled_with: list[RhimeSampler] = []

    def fake_sample(self: RhimeSampler, model: pm.Model) -> az.InferenceData:
        sampled_with.append(self)
        assert model is built_model
        return _minimal_output_idata()

    monkeypatch.setattr(rhime_module, "build_rhime_model_from_spec", lambda *args: built_model)
    monkeypatch.setattr(RhimeSampler, "sample", fake_sample)

    result = run_rhime_from_prepared_inputs(prepared_inputs=prepared, run_spec=run_spec)

    assert len(sampled_with) == 1
    assert sampled_with[0] is result.sampler
    assert result.sampler == RhimeSampler()
    assert result.outputs == {}
    assert result.inv_out is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("default_without_path", "output_path.*required"),
        ("multisector_legacy", "legacy.*supports only single-sector"),
    ],
)
def test_run_rhime_from_prepared_inputs_validates_output_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
    expected_error: str,
) -> None:
    """Prepared runs apply existing output validation before build or sample."""
    model_spec, _, run_spec = _minimal_output_specs(output_format="none")
    sectors = model_spec.sectors
    if case == "multisector_legacy":
        sectors += (
            SectorSpec(
                name="Ocean",
                flux_source="ocean-inventory",
                x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                variable_suffix="ocean",
            ),
        )
        output_spec = RhimeOutputSpec(output_format="legacy", output_path=str(tmp_path))
    else:
        output_spec = RhimeOutputSpec()
    run_spec = RhimeRunSpec(
        run_spec.start_date,
        run_spec.end_date,
        run_spec.sites,
        run_spec.averaging_period,
        RhimeModelSpec(species=model_spec.species, domain=model_spec.domain, sectors=sectors),
        output_spec,
        split_by_sectors=len(sectors) > 1,
    )
    inv_inputs = _minimal_output_inv_inputs()
    if len(sectors) > 1:
        inv_inputs["H"] = xr.concat(
            [inv_inputs["H"].expand_dims(source=[sector.flux_source]) for sector in sectors],
            dim="source",
        )
    prepared = RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=_fake_basis_functions(),
        site_metadata=_prepared_site_metadata(),
    )

    def fail_execution(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("output validation must precede model building and sampling")

    monkeypatch.setattr(rhime_module, "build_rhime_model_from_spec", fail_execution)
    monkeypatch.setattr(rhime_module, "build_rhime_multisector_model_from_spec", fail_execution)
    monkeypatch.setattr(RhimeSampler, "sample", fail_execution)

    with pytest.raises(ValueError, match=expected_error):
        run_rhime_from_prepared_inputs(prepared_inputs=prepared, run_spec=run_spec)


def test_run_rhime_from_prepared_inputs_rejects_empty_model() -> None:
    """Prepared runs reject model specifications without a sector."""
    model_spec, output_spec, run_spec = _minimal_output_specs(output_format="none")
    run_spec = RhimeRunSpec(
        run_spec.start_date,
        run_spec.end_date,
        run_spec.sites,
        run_spec.averaging_period,
        RhimeModelSpec(species=model_spec.species, domain=model_spec.domain, sectors=()),
        output_spec,
    )
    prepared = RhimePreparedInputs(
        inv_inputs=_minimal_output_inv_inputs(),
        basis_functions=_fake_basis_functions(),
        site_metadata=_prepared_site_metadata(),
    )

    with pytest.raises(ValueError, match="must contain at least one sector; found 0"):
        run_rhime_from_prepared_inputs(prepared_inputs=prepared, run_spec=run_spec)


def test_run_rhime_from_prepared_inputs_is_publicly_reexported() -> None:
    """The prepared-input runner is available from the public RHIME package."""
    assert rhime_public.run_rhime_from_prepared_inputs is run_rhime_from_prepared_inputs


def test_unreleased_sampling_compatibility_shims_are_absent() -> None:
    """Unreleased same-branch sampling compatibility names are not public API."""
    model_spec = RhimeModelSpec(
        species="ch4",
        domain="EUROPE",
        sectors=(
            SectorSpec(
                name="FF",
                flux_source="ff-inventory",
                x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                variable_suffix="ff",
            ),
        ),
    )
    output_spec = RhimeOutputSpec(output_format="none")
    run_spec = RhimeRunSpec(
        "2019-01-01",
        "2019-01-02",
        ("TAC",),
        ("1h",),
        model_spec,
        output_spec,
    )

    assert not hasattr(rhime_public, "RhimeSamplingSpec")
    assert not hasattr(rhime_sampling, "RhimeSamplingSpec")
    assert not hasattr(RhimeResult, "sampling_spec")
    with pytest.raises(TypeError, match="nit"):
        RhimeSampler(nit=7)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="sampling_spec"):
        RhimeResult(
            run_spec,
            model_spec,
            output_spec,
            _minimal_inv_inputs(),
            cast(Any, object()),
            sampling_spec=RhimeSampler(),  # type: ignore[call-arg]
        )


def test_rhime_runner_setup_builds_specs_before_preparation(tmp_path: Path) -> None:
    """Route sigma frequency into the model spec, not data preparation."""
    params = {
        "species": "ch4",
        "sites": "TAC",
        "averaging_period": "1h",
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "flux_sources": ["ff-source", "gpp-source", "ter-source", "ocean-source"],
        "sector_sources": {
            "FF": "ff-source",
            "GPP": "gpp-source",
            "TER": "ter-source",
            "ocean": "ocean-source",
        },
        "output_path": str(tmp_path),
        "output_name": "test",
        "output_format": "none",
        "x_prior": {"pdf": "normal", "mu": 1.0, "sigma": 0.5},
        "sector_priors": {
            "FF": {"pdf": "normal", "mu": 1.0, "sigma": 0.5},
            "GPP": {"pdf": "normal", "mu": 0.7, "sigma": 0.2},
            "TER": {"pdf": "normal", "mu": 1.3, "sigma": 0.3},
            "ocean": {"pdf": "normal", "mu": 1.0, "sigma": 0.5},
        },
        "sigma_freq": "8D",
        "draws": "7",
        "burn": "1",
        "tune": "2",
        "chains": "3",
        "sample_kwargs": {"random_seed": 42},
        "posterior_predictive_kwargs": {"random_seed": 43},
        "builder_strategy": "compiled",
    }

    setup = rhime_params.make_rhime_runner_setup(
        params=params,
        multisector=True,
        data_param_names=set(inspect.signature(prepare_rhime_inputs).parameters),
    )

    assert setup.data_args["flux_sources"] == ["ff-source", "gpp-source", "ter-source", "ocean-source"]
    assert setup.data_args["split_by_sectors"] is True
    assert "sector_sources" not in setup.data_args
    assert "sigma_freq" not in setup.data_args
    assert setup.run_spec.sites == ("TAC",)
    assert setup.run_spec.averaging_period == ("1h",)
    assert setup.run_spec.output.output_format == "none"
    assert setup.sampler == RhimeSampler(
        draws=7,
        burn=1,
        tune=2,
        chains=3,
        nuts_sampler="pymc",
        progressbar=False,
        sample_kwargs={"random_seed": 42},
        posterior_predictive_kwargs={"random_seed": 43},
    )
    assert [sector.name for sector in setup.run_spec.model.sectors] == ["FF", "GPP", "TER", "ocean"]
    assert [sector.flux_source for sector in setup.run_spec.model.sectors] == [
        "ff-source",
        "gpp-source",
        "ter-source",
        "ocean-source",
    ]
    assert setup.run_spec.model.sectors[0].x_prior == {"pdf": "normal", "mu": 1.0, "sigma": 0.5}
    assert setup.run_spec.model.sectors[1].x_prior == {"pdf": "normal", "mu": 0.7, "sigma": 0.2}
    assert setup.run_spec.model.sigma_freq == "8D"
    assert setup.run_spec.model.sigma_freq_anchor == "2019-01-01"
    assert setup.run_spec.model.builder_strategy == "compiled"


def test_rhime_runner_setup_rejects_unknown_builder_strategy() -> None:
    """Invalid builder selection fails during setup, before data preparation."""
    params = {
        "species": "ch4",
        "sites": ["TAC"],
        "averaging_period": ["1h"],
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "flux_sources": ["ff-source"],
        "output_name": "test",
        "output_format": "none",
        "builder_strategy": "fallback",
    }

    with pytest.raises(ValueError, match="builder_strategy.*concrete.*compiled"):
        rhime_params.make_rhime_runner_setup(
            params=params,
            multisector=False,
            data_param_names=set(inspect.signature(prepare_rhime_inputs).parameters),
        )


def test_legacy_sampling_names_are_not_rhime_config_aliases() -> None:
    """RHIME accepts modern sampler names only; legacy HBMCMC names are unsupported."""
    params = rhime_params.normalise_rhime_params(
        {
            "nit": "7",
            "nchain": "3",
            "verbose": True,
            "sampler_kwargs": {"random_seed": 42},
        }
    )

    with pytest.raises(ValueError, match="nit"):
        rhime_params.validate_supported_params(
            params,
            data_params=set(inspect.signature(prepare_rhime_inputs).parameters),
        )


@pytest.mark.parametrize(
    ("legacy_output_format", "expected_output_format"),
    [
        ("hbmcmc", "legacy"),
        ("hbmcmc_postprocessing", "legacy"),
        ("legacy", "legacy"),
    ],
)
def test_rhime_normalises_legacy_output_format_aliases(
    legacy_output_format: str, expected_output_format: str
) -> None:
    """Old HBMCMC output names now select the modern legacy formatter."""
    params = rhime_params.normalise_rhime_params({"output_format": legacy_output_format})

    assert params["output_format"] == expected_output_format


def test_build_rhime_model_from_spec_forwards_single_sector_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward sector activity and priors and construct sigma alignment."""
    sentinel = cast(pm.Model, object())
    seen: dict[str, Any] = {}
    flux_activity = StateActivity(fixed_groups=("outer",))
    bc_activity = StateActivity(active=False)

    def fake_build_rhime_model(inv_inputs: xr.Dataset, **kwargs: Any) -> pm.Model:
        seen["inv_inputs"] = inv_inputs
        seen["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(rhime_models_module, "build_rhime_model", fake_build_rhime_model)
    inv_inputs = _minimal_output_inv_inputs()
    model_spec = RhimeModelSpec(
        species="ch4",
        domain="EUROPE",
        sectors=(
            SectorSpec(
                name="FF",
                flux_source="ff-inventory",
                x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                variable_suffix="ff",
                state_activity=flux_activity,
            ),
        ),
        bc_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.1},
        bc_state_activity=bc_activity,
        sigma_per_site=False,
        sigma_freq="8D",
        sigma_freq_anchor="2019-01-01",
    )

    model = build_rhime_model_from_spec(inv_inputs, model_spec)

    assert model is sentinel
    assert seen["inv_inputs"] is inv_inputs
    assert seen["kwargs"]["x_prior"] == {"pdf": "normal", "mu": 1.0, "sigma": 0.2}
    assert seen["kwargs"]["bc_prior"] == {"pdf": "normal", "mu": 1.0, "sigma": 0.1}
    assert seen["kwargs"]["state_activity"] is flux_activity
    assert seen["kwargs"]["bc_state_activity"] is bc_activity
    alignment = seen["kwargs"]["sigma_alignment"]
    assert isinstance(alignment, SigmaAlignment)
    assert alignment.nsite == 1
    assert alignment.nperiod == 1
    np.testing.assert_array_equal(alignment.site_index, np.array([0]))
    np.testing.assert_array_equal(alignment.period_index, np.array([0]))


def test_build_rhime_model_from_spec_requires_one_sector() -> None:
    """The standard spec wrapper rejects multi-sector specs."""
    model_spec = RhimeModelSpec(
        species="ch4",
        domain="EUROPE",
        sectors=(
            SectorSpec("FF", "ff-inventory", {"pdf": "normal", "mu": 1.0, "sigma": 0.2}, "ff"),
            SectorSpec("ocean", "ocean-inventory", {"pdf": "normal", "mu": 1.0, "sigma": 0.3}, "ocean"),
        ),
    )

    with pytest.raises(ValueError, match="exactly one sector"):
        build_rhime_model_from_spec(_minimal_inv_inputs(), model_spec)


def test_build_rhime_model_from_spec_dispatches_compiled_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A serializable model-spec option selects the private compiler path."""
    sentinel = cast(pm.Model, object())
    seen: dict[str, Any] = {}

    def fake_compiled_builder(inv_inputs: xr.Dataset, **kwargs: Any) -> pm.Model:
        seen["inv_inputs"] = inv_inputs
        seen["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(
        rhime_models_module,
        "_build_compiled_rhime_model",
        fake_compiled_builder,
    )
    inv_inputs = _minimal_output_inv_inputs()
    model_spec = RhimeModelSpec(
        species="ch4",
        domain="EUROPE",
        sectors=(
            SectorSpec(
                "FF",
                "ff-inventory",
                {"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                "ff",
            ),
        ),
        builder_strategy="compiled",
    )

    model = build_rhime_model_from_spec(inv_inputs, model_spec)

    assert model is sentinel
    assert seen["inv_inputs"] is inv_inputs
    assert seen["kwargs"]["x_prior"] == {"pdf": "normal", "mu": 1.0, "sigma": 0.2}


def test_build_rhime_multisector_model_from_spec_preserves_sector_source_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec wrapper preserves source mappings and activity precedence."""
    sentinel = cast(pm.Model, object())
    seen: dict[str, Any] = {}
    shared_activity = StateActivity(fixed_value=3.0)
    mapped_activity = StateActivity(active=False, fixed_value=2.0)
    ff_activity = StateActivity(fixed_groups=("outer",))
    bc_activity = StateActivity(active=False)

    def fake_build_rhime_multisector_model(inv_inputs: xr.Dataset, **kwargs: Any) -> pm.Model:
        seen["inv_inputs"] = inv_inputs
        seen["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(
        rhime_models_module,
        "build_rhime_multisector_model",
        fake_build_rhime_multisector_model,
    )
    inv_inputs = _minimal_output_inv_inputs()
    model_spec = RhimeModelSpec(
        species="ch4",
        domain="EUROPE",
        sectors=(
            SectorSpec(
                name="FF",
                flux_source="ff-inventory",
                x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                variable_suffix="ff",
                state_activity=ff_activity,
            ),
            SectorSpec(
                name="ocean",
                flux_source="ocean-inventory",
                x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.3},
                variable_suffix="ocean",
            ),
        ),
        bc_state_activity=bc_activity,
        state_activity=shared_activity,
        sector_state_activities={"FF": mapped_activity, "ocean": mapped_activity},
    )

    model = build_rhime_multisector_model_from_spec(inv_inputs, model_spec)

    assert model is sentinel
    assert seen["inv_inputs"] is inv_inputs
    assert seen["kwargs"]["sectors"] == ["FF", "ocean"]
    assert seen["kwargs"]["sector_sources"] == {"FF": "ff-inventory", "ocean": "ocean-inventory"}
    assert seen["kwargs"]["sector_variable_suffixes"] == {"FF": "ff", "ocean": "ocean"}
    assert seen["kwargs"]["sector_priors"] == {
        "FF": {"pdf": "normal", "mu": 1.0, "sigma": 0.2},
        "ocean": {"pdf": "normal", "mu": 1.0, "sigma": 0.3},
    }
    assert seen["kwargs"]["state_activity"] is shared_activity
    assert seen["kwargs"]["sector_state_activities"] == {
        "FF": ff_activity,
        "ocean": mapped_activity,
    }
    assert seen["kwargs"]["bc_state_activity"] is bc_activity
    assert isinstance(seen["kwargs"]["sigma_alignment"], SigmaAlignment)


def test_build_rhime_multisector_model_from_spec_dispatches_compiled_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The multisector spec wrapper selects compilation only when requested."""
    sentinel = cast(pm.Model, object())
    seen: dict[str, Any] = {}

    def fake_compiled_builder(inv_inputs: xr.Dataset, **kwargs: Any) -> pm.Model:
        seen["inv_inputs"] = inv_inputs
        seen["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(
        rhime_models_module,
        "_build_compiled_rhime_multisector_model",
        fake_compiled_builder,
    )
    inv_inputs = _minimal_output_inv_inputs()
    model_spec = RhimeModelSpec(
        species="ch4",
        domain="EUROPE",
        sectors=(
            SectorSpec(
                "FF",
                "ff-inventory",
                {"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                "ff",
            ),
            SectorSpec(
                "ocean",
                "ocean-inventory",
                {"pdf": "normal", "mu": 1.0, "sigma": 0.3},
                "ocean",
            ),
        ),
        builder_strategy="compiled",
    )

    model = build_rhime_multisector_model_from_spec(inv_inputs, model_spec)

    assert model is sentinel
    assert seen["inv_inputs"] is inv_inputs
    assert seen["kwargs"]["sectors"] == ["FF", "ocean"]


def test_rhime_model_spec_rejects_unknown_builder_strategy() -> None:
    """Invalid graph-construction modes fail at model-spec construction."""
    with pytest.raises(ValueError, match="builder_strategy.*concrete.*compiled"):
        RhimeModelSpec(
            species="ch4",
            domain="EUROPE",
            sectors=(),
            builder_strategy=cast(Any, "fallback"),
        )


def test_rhime_sampler_runs_pymc_sampling_and_predictive_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RhimeSampler owns PyMC sampling, burn slicing, and predictive groups."""

    class FakePosterior:
        sizes = {"draw": 6}

    class FakeInferenceData:
        posterior = FakePosterior()
        sample_stats = xr.Dataset(
            {
                "n_steps": (("chain", "draw"), np.array([[1, 2, 3], [4, 5, 6]])),
                "tree_depth": (("chain", "draw"), np.array([[2, 2, 3], [3, 4, 4]])),
                "step_size": (("chain", "draw"), np.array([[0.1, 0.1, 0.2], [0.2, 0.2, 0.2]])),
                "acceptance_rate": (("chain", "draw"), np.array([[0.8, 0.9, 1.0], [0.7, 0.8, 0.9]])),
                "diverging": (("chain", "draw"), np.array([[False, True, False], [False, False, True]])),
            }
        )

        def __init__(self) -> None:
            self.isel_kwargs: dict[str, Any] | None = None
            self.extensions: list[Any] = []
            self.attrs: dict[str, Any] = {}

        def isel(self, **kwargs: Any) -> "FakeInferenceData":
            self.isel_kwargs = kwargs
            return self

        def groups(self) -> list[str]:
            """Return the fake InferenceData group names."""
            return ["sample_stats"]

        def extend(self, other: Any) -> None:
            self.extensions.append(other)

    fake_idata = FakeInferenceData()
    seen: dict[str, Any] = {}
    timings: list[tuple[str, dict[str, Any]]] = []

    def fake_sample(**kwargs: Any) -> Any:
        seen["sample_kwargs"] = kwargs
        return fake_idata

    def fake_prior_predictive(draws: int, model: pm.Model) -> str:
        seen["prior_predictive"] = {"draws": draws, "model": model}
        return "prior"

    def fake_posterior_predictive(trace: Any, **kwargs: Any) -> str:
        seen["posterior_predictive"] = {"trace": trace, **kwargs}
        return "posterior"

    def fake_log_timing(label: str, seconds: float, **fields: Any) -> None:
        timings.append((label, fields))

    monkeypatch.setattr("openghg_inversions.rhime.sampling.pm.sample", fake_sample)
    monkeypatch.setattr(rhime_sampling, "log_timing", fake_log_timing)
    monkeypatch.setattr(
        "openghg_inversions.rhime.sampling.pm.sample_prior_predictive",
        fake_prior_predictive,
    )
    monkeypatch.setattr(
        "openghg_inversions.rhime.sampling.pm.sample_posterior_predictive",
        fake_posterior_predictive,
    )
    model = pm.Model()
    sampler = RhimeSampler(
        draws=7,
        burn=1,
        tune=2,
        chains=3,
        nuts_sampler="numpyro",
        progressbar=True,
        sample_kwargs={"target_accept": 0.9},
        posterior_predictive_kwargs={"random_seed": 42},
    )

    idata = sampler.sample(model)

    assert idata is fake_idata
    assert seen["sample_kwargs"]["draws"] == 7
    assert seen["sample_kwargs"]["tune"] == 2
    assert seen["sample_kwargs"]["chains"] == 3
    assert seen["sample_kwargs"]["nuts_sampler"] == "numpyro"
    assert seen["sample_kwargs"]["progressbar"] is True
    assert seen["sample_kwargs"]["cores"] == 3
    assert seen["sample_kwargs"]["target_accept"] == 0.9
    assert seen["sample_kwargs"]["return_inferencedata"] is True
    assert seen["sample_kwargs"]["idata_kwargs"] == {"log_likelihood": True}
    assert fake_idata.isel_kwargs == {"draw": slice(1, None)}
    assert seen["prior_predictive"] == {"draws": 6, "model": model}
    assert seen["posterior_predictive"] == {
        "trace": fake_idata,
        "model": model,
        "var_names": ["y"],
        "random_seed": 42,
    }
    assert fake_idata.extensions == ["prior", "posterior"]
    sample_stats_fields = dict(timings)["rhime.sampler.sample_stats"]
    assert sample_stats_fields["n_steps_mean"] == 3.5
    assert sample_stats_fields["n_steps_max"] == 6.0
    assert sample_stats_fields["tree_depth_mean"] == 3.0
    assert sample_stats_fields["tree_depth_max"] == 4.0
    assert sample_stats_fields["step_size_mean"] == pytest.approx(1.0 / 6.0)
    assert sample_stats_fields["acceptance_rate_mean"] == pytest.approx(0.85)
    assert sample_stats_fields["divergences"] == 2


def test_rhime_sampler_resets_retained_draws_before_extending_predictive_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal-length predictive groups do not outer-align against burned draw labels."""
    raw_trace = az.InferenceData(
        posterior=xr.Dataset(
            {"x": (("chain", "draw"), np.arange(2000, dtype=float)[None, :])},
            coords={"chain": [0], "draw": np.arange(2000)},
        ),
        sample_stats=xr.Dataset(
            {"diverging": (("chain", "draw"), np.zeros((1, 2000), dtype=bool))},
            coords={"chain": [0], "draw": np.arange(2000)},
        ),
        log_likelihood=xr.Dataset(
            {"y": (("chain", "draw"), np.zeros((1, 2000)))},
            coords={"chain": [0], "draw": np.arange(2000)},
        ),
    )
    predictive_draws: dict[str, np.ndarray] = {}

    def fake_sample(**kwargs: Any) -> az.InferenceData:
        """Return the unsliced sampling trace."""
        return raw_trace

    def fake_prior_predictive(draws: int, model: pm.Model) -> az.InferenceData:
        """Build prior groups with zero-based draw labels."""
        predictive_draws["prior"] = np.arange(draws)
        return az.InferenceData(
            prior=xr.Dataset(
                {"x": (("chain", "draw"), np.ones((1, draws)))},
                coords={"chain": [0], "draw": predictive_draws["prior"]},
            ),
            prior_predictive=xr.Dataset(
                {"y": (("chain", "draw"), np.ones((1, draws)))},
                coords={"chain": [0], "draw": predictive_draws["prior"]},
            ),
        )

    def fake_posterior_predictive(trace: az.InferenceData, **kwargs: Any) -> az.InferenceData:
        """Record and mirror the retained posterior draw labels."""
        inference_data = cast(Any, trace)
        predictive_draws["posterior_seen"] = inference_data.posterior.draw.values.copy()
        draws = inference_data.posterior.sizes["draw"]
        return az.InferenceData(
            posterior_predictive=xr.Dataset(
                {"y": (("chain", "draw"), np.ones((1, draws)))},
                coords={"chain": [0], "draw": np.arange(draws)},
            )
        )

    monkeypatch.setattr("openghg_inversions.rhime.sampling.pm.sample", fake_sample)
    monkeypatch.setattr(
        "openghg_inversions.rhime.sampling.pm.sample_prior_predictive",
        fake_prior_predictive,
    )
    monkeypatch.setattr(
        "openghg_inversions.rhime.sampling.pm.sample_posterior_predictive",
        fake_posterior_predictive,
    )
    sampler = RhimeSampler(draws=2000, burn=1000, tune=0, chains=1)

    result = sampler.sample(pm.Model())
    inference_data = cast(Any, result)
    trace_dataset = inversion_output_module.convert_idata_to_dataset(result)

    expected_draws = np.arange(1000)
    np.testing.assert_array_equal(inference_data.posterior.draw.values, expected_draws)
    np.testing.assert_array_equal(inference_data.sample_stats.draw.values, expected_draws)
    np.testing.assert_array_equal(inference_data.log_likelihood.draw.values, expected_draws)
    np.testing.assert_array_equal(inference_data.prior.draw.values, expected_draws)
    np.testing.assert_array_equal(inference_data.prior_predictive.draw.values, expected_draws)
    np.testing.assert_array_equal(inference_data.posterior_predictive.draw.values, expected_draws)
    np.testing.assert_array_equal(predictive_draws["prior"], expected_draws)
    np.testing.assert_array_equal(predictive_draws["posterior_seen"], expected_draws)
    assert trace_dataset.sizes["draw"] == 1000
    np.testing.assert_array_equal(trace_dataset.draw.values, expected_draws)
    assert not trace_dataset.to_array().isnull().any()
    assert inference_data.attrs["burn"] == 1000
    for group_name in ("posterior", "sample_stats", "log_likelihood"):
        assert inference_data[group_name].attrs["burn"] == 1000


def test_rhime_sampler_restores_registered_coords_after_predictive_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RhimeSampler restores model-managed coordinates after predictive groups are attached."""

    class FakePosterior:
        sizes = {"draw": 2}

    class FakeInferenceData:
        posterior = FakePosterior()

        def __init__(self) -> None:
            self.extensions: list[str] = []
            self.attrs: dict[str, Any] = {}

        def isel(self, **kwargs: Any) -> "FakeInferenceData":
            return self

        def groups(self) -> list[str]:
            """Return the fake InferenceData group names."""
            return []

        def extend(self, other: str) -> None:
            self.extensions.append(other)

    fake_idata = FakeInferenceData()
    calls: list[tuple[FakeInferenceData, models.CoordRegistry, list[str]]] = []

    def fake_sample(**kwargs: Any) -> FakeInferenceData:
        return fake_idata

    def fake_prior_predictive(draws: int, model: pm.Model) -> str:
        return "prior"

    def fake_posterior_predictive(trace: Any, **kwargs: Any) -> str:
        return "posterior"

    def fake_restore(trace: FakeInferenceData, registry: models.CoordRegistry) -> FakeInferenceData:
        calls.append((trace, registry, list(trace.extensions)))
        return trace

    monkeypatch.setattr("openghg_inversions.rhime.sampling.pm.sample", fake_sample)
    monkeypatch.setattr(
        "openghg_inversions.rhime.sampling.pm.sample_prior_predictive",
        fake_prior_predictive,
    )
    monkeypatch.setattr(
        "openghg_inversions.rhime.sampling.pm.sample_posterior_predictive",
        fake_posterior_predictive,
    )
    monkeypatch.setattr("openghg_inversions.rhime.sampling.restore_inferencedata_coords", fake_restore)

    model = pm.Model()
    registry = models.CoordRegistry(
        original_coords={"nmeasure": pd.MultiIndex.from_arrays([["TAC"], pd.to_datetime(["2019-01-01"])])}
    )
    models.attach_coord_registry(model, registry)

    result = RhimeSampler(draws=2, chains=1, tune=0).sample(model)

    assert result is fake_idata
    assert calls == [(fake_idata, registry, ["prior", "posterior"])]


def test_params_from_config_maps_legacy_emissions_name(tmp_path: Path) -> None:
    config_file = tmp_path / "rhime.ini"
    config_file.write_text(
        """
[INPUT.MEASUREMENTS]
species = "ch4"
sites = ["TAC"]
averaging_period = ["1h"]
start_date = "2019-01-01"
end_date = "2019-01-02"

[INPUT.PRIORS]
domain = "EUROPE"
emissions_name = ["legacy-source"]

[RHIME.OUTPUT]
output_path = "out"
output_name = "test"
""",
        encoding="utf-8",
    )

    params = params_from_config(config_file)
    assert params["flux_sources"] == ["legacy-source"]


def test_params_from_config_parses_builder_strategy(tmp_path: Path) -> None:
    """A real RHIME INI file preserves the requested model-builder strategy."""
    config_file = tmp_path / "rhime.ini"
    config_file.write_text(
        """
[INPUT.MEASUREMENTS]
species = "ch4"
sites = ["TAC"]
averaging_period = ["1h"]
start_date = "2019-01-01"
end_date = "2019-01-02"

[INPUT.PRIORS]
domain = "EUROPE"
flux_sources = ["ff-source"]

[RHIME.OPTIONS]
builder_strategy = "compiled"

[RHIME.OUTPUT]
output_path = "out"
output_name = "test"
""",
        encoding="utf-8",
    )

    params = params_from_config(config_file)

    assert params["builder_strategy"] == "compiled"


@pytest.mark.parametrize("prior_name", ["x_prior", "bc_prior", "sigma_prior", "offset_prior"])
def test_params_from_config_rejects_malformed_prior_options(tmp_path: Path, prior_name: str) -> None:
    """Malformed structured prior config values fail during RHIME normalization."""
    config_file = tmp_path / "rhime.ini"
    config_file.write_text(
        f"""
[RHIME.PDF]
{prior_name} = {{"pdf": "normal"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=prior_name):
        params_from_config(config_file)


def test_params_from_config_rejects_malformed_sector_priors(tmp_path: Path) -> None:
    """Malformed sector prior config values name the bad option."""
    config_file = tmp_path / "rhime.ini"
    config_file.write_text(
        """
[RHIME.PDF]
sector_priors = {"FF": {"pdf": "normal"}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sector_priors"):
        params_from_config(config_file)


def test_run_rhime_rejects_string_prior_before_data_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid prior types fail before RHIME data preparation is called."""

    def fail_prepare(**kwargs):
        raise AssertionError("prepare_rhime_inputs should not be called")

    monkeypatch.setattr(rhime_module, "prepare_rhime_inputs", fail_prepare)

    with pytest.raises(ValueError, match="x_prior"):
        run_rhime(
            species="ch4",
            sites=["TAC"],
            averaging_period=["1h"],
            domain="EUROPE",
            start_date="2019-01-01",
            end_date="2019-01-02",
            output_name="test",
            output_format="none",
            flux_sources=["total-ukghg-edgar7"],
            x_prior="bad",
        )


@pytest.mark.parametrize(
    "min_error_options",
    [
        pytest.param("bad", id="not-a-mapping"),
        pytest.param({"unsupported": True}, id="unsupported-key"),
        pytest.param({"by_site": 1}, id="non-boolean-by-site"),
    ],
)
def test_run_rhime_rejects_malformed_min_error_options_before_data_preparation(
    monkeypatch: pytest.MonkeyPatch,
    min_error_options: object,
) -> None:
    """Invalid min-error options fail before RHIME data preparation."""

    def fail_prepare(**kwargs):
        raise AssertionError("prepare_rhime_inputs should not be called")

    monkeypatch.setattr(rhime_module, "prepare_rhime_inputs", fail_prepare)

    with pytest.raises(ValueError, match="min_error_options"):
        run_rhime(
            species="ch4",
            sites=["TAC"],
            averaging_period=["1h"],
            domain="EUROPE",
            start_date="2019-01-01",
            end_date="2019-01-02",
            output_name="test",
            output_format="none",
            flux_sources=["total-ukghg-edgar7"],
            min_error_options=cast(Any, min_error_options),
        )


def test_run_rhime_rejects_malformed_power_before_data_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid likelihood power values fail before RHIME data preparation."""

    def fail_prepare(**kwargs):
        raise AssertionError("prepare_rhime_inputs should not be called")

    monkeypatch.setattr(rhime_module, "prepare_rhime_inputs", fail_prepare)

    with pytest.raises(ValueError, match="power"):
        run_rhime(
            species="ch4",
            sites=["TAC"],
            averaging_period=["1h"],
            domain="EUROPE",
            start_date="2019-01-01",
            end_date="2019-01-02",
            output_name="test",
            output_format="none",
            flux_sources=["total-ukghg-edgar7"],
            power="bad",
        )


def test_run_rhime_multisector_rejects_non_mapping_sector_prior_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid sector prior values fail before RHIME data preparation is called."""

    def fail_prepare(**kwargs):
        raise AssertionError("prepare_rhime_inputs should not be called")

    monkeypatch.setattr(rhime_module, "prepare_rhime_inputs", fail_prepare)

    with pytest.raises(ValueError, match="sector_priors"):
        run_rhime_multisector(
            species="ch4",
            sites=["TAC"],
            averaging_period=["1h"],
            domain="EUROPE",
            start_date="2019-01-01",
            end_date="2019-01-02",
            output_name="test",
            output_format="none",
            flux_sources=["FF", "GPP"],
            sector_priors={"FF": "bad"},
        )


def test_run_rhime_multisector_rejects_source_keyed_xprior_before_data_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy source-keyed ``xprior`` fails before RHIME data preparation."""

    def fail_prepare(**kwargs):
        raise AssertionError("prepare_rhime_inputs should not be called")

    setattr(fail_prepare, "__signature__", inspect.signature(prepare_rhime_inputs))
    monkeypatch.setattr(rhime_module, "prepare_rhime_inputs", fail_prepare)

    with pytest.raises(ValueError, match="source-keyed priors"):
        run_rhime_multisector(
            species="ch4",
            sites=["TAC"],
            averaging_period=["1h"],
            domain="EUROPE",
            start_date="2019-01-01",
            end_date="2019-01-02",
            output_name="test",
            output_format="none",
            emissions_name=["sector-a", "sector-b"],
            xprior={
                "sector-a": {"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                "sector-b": {"pdf": "normal", "mu": 1.0, "sigma": 0.3},
            },
        )


def test_run_rhime_multisector_rejects_non_mapping_sector_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid sector-source mappings fail before RHIME data preparation."""

    def fail_prepare(**kwargs):
        raise AssertionError("prepare_rhime_inputs should not be called")

    monkeypatch.setattr(rhime_module, "prepare_rhime_inputs", fail_prepare)

    with pytest.raises(ValueError, match="sector_sources"):
        run_rhime_multisector(
            species="ch4",
            sites=["TAC"],
            averaging_period=["1h"],
            domain="EUROPE",
            start_date="2019-01-01",
            end_date="2019-01-02",
            output_name="test",
            output_format="none",
            flux_sources=["ff-source", "gpp-source"],
            sector_sources="bad",
        )


def test_run_rhime_multisector_rejects_duplicate_sanitized_sector_names() -> None:
    """Duplicate PyMC suffixes fail during setup, before RHIME data preparation."""
    with pytest.raises(ValueError, match="duplicate sanitized name"):
        rhime_module._make_rhime_runner_setup(
            params={
                "species": "ch4",
                "sites": ["TAC"],
                "averaging_period": ["1h"],
                "domain": "EUROPE",
                "start_date": "2019-01-01",
                "end_date": "2019-01-02",
                "output_name": "test",
                "output_format": "none",
                "flux_sources": ["ff-source", "gpp-source"],
                "sector_sources": {"Sector 2": "ff-source", "sector-2": "gpp-source"},
            },
            multisector=True,
        )


def test_resolve_flux_sources_rejects_duplicates() -> None:
    """Repeated retrieval identities are rejected before OpenGHG access."""
    with pytest.raises(ValueError, match="duplicate source.*ff-inventory"):
        resolve_flux_sources(flux_sources=["ff-inventory", "ff-inventory"])


def test_run_rhime_multisector_rejects_duplicate_sector_source_mappings() -> None:
    """Current independent sector states require distinct source sensitivities."""
    with pytest.raises(ValueError, match="source 'ff-inventory'.*\\['FF', 'other'\\]"):
        rhime_module._make_rhime_runner_setup(
            params={
                "species": "ch4",
                "sites": ["TAC"],
                "averaging_period": ["1h"],
                "domain": "EUROPE",
                "start_date": "2019-01-01",
                "end_date": "2019-01-02",
                "output_name": "test",
                "output_format": "none",
                "flux_sources": ["ff-inventory"],
                "sector_sources": {
                    "FF": "ff-inventory",
                    "other": "ff-inventory",
                },
            },
            multisector=True,
        )


@pytest.mark.parametrize(
    ("sector_priors", "error_fragment"),
    [
        (
            {"FF": {"pdf": "normal", "mu": 1.0, "sigma": 0.2}},
            "missing sector prior\\(s\\): \\['ocean'\\]",
        ),
        (
            {
                "FF": {"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                "ocean": {"pdf": "normal", "mu": 1.0, "sigma": 0.3},
                "oecam": {"pdf": "normal", "mu": 1.0, "sigma": 0.4},
            },
            "unused sector prior key\\(s\\): \\['oecam'\\]",
        ),
    ],
)
def test_run_rhime_multisector_rejects_inexact_sector_prior_keys(
    sector_priors: dict[str, dict[str, Any]],
    error_fragment: str,
) -> None:
    """Missing and unused sector prior keys fail before data preparation."""
    with pytest.raises(ValueError, match=error_fragment):
        rhime_module._make_rhime_runner_setup(
            params={
                "species": "ch4",
                "sites": ["TAC"],
                "averaging_period": ["1h"],
                "domain": "EUROPE",
                "start_date": "2019-01-01",
                "end_date": "2019-01-02",
                "output_name": "test",
                "output_format": "none",
                "flux_sources": ["ff-inventory", "ocean-inventory"],
                "sector_sources": {"FF": "ff-inventory", "ocean": "ocean-inventory"},
                "sector_priors": sector_priors,
            },
            multisector=True,
        )


def test_new_rhime_docs_use_flux_sources_for_examples() -> None:
    """New RHIME docs keep legacy names out of config/API examples."""
    rhime_doc = Path("docs/usage/rhime.rst").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    template = Path("openghg_inversions/config/templates/rhime_template.ini").read_text(encoding="utf-8")

    assert "flux_sources" in rhime_doc
    assert "sector_sources" in rhime_doc
    assert "flux_sources" in readme
    assert "flux_sources" in template
    assert "sector_sources" in template
    assert 'builder_strategy = "concrete"' in template
    assert 'builder_strategy="compiled"' in rhime_doc
    assert "emissions_name =" not in rhime_doc
    assert "emissions_name =" not in readme
    assert "emissions_name =" not in template
    assert "draws =" in template
    assert "chains =" in template
    assert "nit =" not in template
    assert "nchain =" not in template
    assert "sample_kwargs =" in template
    assert "sampler_kwargs =" not in template
    assert "Legacy compatibility spelling" in rhime_doc


def test_cleanup_plan_records_issue_400_decisions() -> None:
    """The cleanup plan records current PR decisions without example completions."""
    plan_doc = Path("docs/plans/clean_up_inversions_refactor.md").read_text(encoding="utf-8")

    assert "PR #434 / Issue #400" in plan_doc
    assert "#383 / PR #412 merged" not in plan_doc
    assert "one public concrete `RhimeModelSpec`" in plan_doc
    assert "Do not split the public API" in plan_doc
    assert "SemanticModel" in plan_doc
    assert "RhimeSampler" in plan_doc
    assert "openghg_inversions.rhime.runner" in plan_doc
    assert "openghg_inversions.rhime.model_specs" not in plan_doc
    assert "openghg_inversions.models.rhime" in plan_doc
    assert "prior-predictive-only" in plan_doc
    assert "Deferred Issue #431 data-preparation spec" in plan_doc
    assert "should not introduce `RhimeDataSpec`" in plan_doc
    assert "Deferred Issue #383 / Issue #429 output boundary" in plan_doc


def _rhime_preparation_args(data_args: dict, flux_sources: list[str], bc_basis_directory: Path) -> dict:
    args = data_args.copy()
    args.pop("emissions_name", None)
    args.update(
        {
            "output_name": "prep_test",
            "flux_sources": flux_sources,
            "basis_algorithm": "quadtree",
            "nbasis": 4,
            "use_bc": True,
            "bc_basis_directory": bc_basis_directory,
        }
    )
    return args


def test_rhime_prepared_inputs_contract_exposes_only_modern_fields() -> None:
    """The durable prepared-input API omits legacy merged-data side channels."""
    prepared = RhimePreparedInputs(
        inv_inputs=_minimal_prepared_inv_inputs(),
        basis_functions=_fake_basis_functions(),
        site_metadata=_prepared_site_metadata(averaging_period=("1H",)),
    )

    for legacy_attr in (
        "fp_all",
        "fp_data",
        "basis",
        "flux",
        "basis_objects",
        "return_basis_objects",
    ):
        assert not hasattr(prepared, legacy_attr)


def test_prepared_multisector_runner_accepts_gathered_source_specific_basis_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Legacy source-valued ``sector`` provenance survives load and execution.

    The auxiliary ``sector`` coordinate records the OpenGHG flux source for
    each gathered state; it is not the semantic ``SectorSpec.name``. Reloading
    preserves this coordinate, while model lowering namespaces auxiliaries
    spanning state and observation and leaves observation-only coordinates
    shared.
    """
    basis_ff = xr.DataArray(
        [[0, 0], [1, 1]],
        dims=("lat", "lon"),
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    basis_ocean = xr.DataArray(
        [[0, 1], [2, 2]],
        dims=("lat", "lon"),
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    flux = xr.DataArray(
        np.ones((2, 2, 2)),
        dims=("source", "lat", "lon"),
        coords={
            "source": ["ff-inventory", "ocean-inventory"],
            "lat": [0.0, 1.0],
            "lon": [0.0, 1.0],
        },
    )
    basis_functions = BasisFunctions.from_multi_source_flat_basis(
        basis_flat={"ff-inventory": basis_ff, "ocean-inventory": basis_ocean},
        flux=flux,
        operator_kwargs={"state_dim": "region"},
    )
    model_spec = RhimeModelSpec(
        species="ch4",
        domain="EUROPE",
        use_bc=False,
        sectors=(
            SectorSpec("FF", "ff-inventory", {"pdf": "normal", "mu": 1.0, "sigma": 0.2}, "ff"),
            SectorSpec(
                "ocean",
                "ocean-inventory",
                {"pdf": "normal", "mu": 1.0, "sigma": 0.3},
                "ocean",
            ),
        ),
    )
    fp_x_flux = xr.DataArray(
        np.ones((2, 2, 2, 1)),
        dims=("source", "lat", "lon", "time"),
        coords={
            "source": ["ff-inventory", "ocean-inventory"],
            "lat": [0.0, 1.0],
            "lon": [0.0, 1.0],
            "time": [0],
        },
    )
    inv_inputs = _minimal_output_inv_inputs().drop_dims("region")
    inv_inputs["min_error"] = xr.zeros_like(inv_inputs["mf"])
    sensitivity = (
        basis_functions.sensitivity(fp_x_flux)
        .rename(time="nmeasure")
        .assign_coords(nmeasure=inv_inputs.coords["nmeasure"])
    )
    state_dim = next(dim for dim in sensitivity.dims if dim != "nmeasure")
    state_measurement_code = np.arange(sensitivity.size).reshape(sensitivity.shape)
    inv_inputs["H"] = sensitivity.assign_coords(
        sector=(state_dim, sensitivity.coords["source"].values),
        state_measurement_code=(sensitivity.dims, state_measurement_code),
        observation_label=("nmeasure", ["shared-observation"]),
    )
    expected_state_index = inv_inputs.indexes[state_dim].copy()
    expected_sector = inv_inputs["sector"].copy(deep=True)
    expected_state_measurement_code = inv_inputs["state_measurement_code"].copy(deep=True)

    prepared = RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=basis_functions,
        site_metadata=_prepared_site_metadata(averaging_period=("1H",)),
    )
    artifact_path = tmp_path / "gathered-multisector-prepared.nc"
    prepared.save(artifact_path)
    loaded = RhimePreparedInputs.load(artifact_path)
    loaded_before_run = loaded.inv_inputs.copy(deep=True)

    loaded_state_index = loaded.inv_inputs.indexes[state_dim]
    assert isinstance(loaded_state_index, pd.MultiIndex)
    assert loaded_state_index.names == ["source", "region_in_source"]
    assert loaded_state_index.equals(expected_state_index)
    xr.testing.assert_identical(loaded.inv_inputs["sector"], expected_sector)
    xr.testing.assert_identical(
        loaded.inv_inputs["state_measurement_code"],
        expected_state_measurement_code,
    )

    run_spec = RhimeRunSpec(
        start_date="2019-01-01",
        end_date="2019-01-02",
        sites=("TAC",),
        averaging_period=("1H",),
        model=model_spec,
        output=RhimeOutputSpec(output_format="none"),
        split_by_sectors=True,
    )
    monkeypatch.setattr(
        RhimeSampler,
        "sample",
        lambda self, model: _minimal_output_idata(),
    )
    monkeypatch.setattr(
        rhime_module,
        "make_multisector_output_bundle",
        lambda **kwargs: rhime_outputs.RhimeOutputBundle(),
    )

    result = run_rhime_from_prepared_inputs(prepared_inputs=loaded, run_spec=run_spec)

    xr.testing.assert_identical(loaded.inv_inputs, loaded_before_run)
    assert result.model.named_vars_to_dims["x_ff"] == ("region_ff",)
    assert result.model.named_vars_to_dims["x_ocean"] == ("region_ocean",)
    registry = models.get_coord_registry(result.model)
    assert registry is not None
    assert registry.auxiliary_coords["sector_ff"].values.tolist() == ["ff-inventory"] * 2
    assert registry.auxiliary_coords["sector_ocean"].values.tolist() == ["ocean-inventory"] * 3
    ff_states = expected_sector.values == "ff-inventory"
    ocean_states = expected_sector.values == "ocean-inventory"
    ff_codes = registry.auxiliary_coords["state_measurement_code_ff"]
    ocean_codes = registry.auxiliary_coords["state_measurement_code_ocean"]
    assert ff_codes.dims == ("nmeasure", "region_ff")
    assert ocean_codes.dims == ("nmeasure", "region_ocean")
    expected_ff_codes = (
        expected_state_measurement_code.isel({state_dim: np.flatnonzero(ff_states)})
        .rename({state_dim: "region_ff"})
        .transpose(*ff_codes.dims)
    )
    expected_ocean_codes = (
        expected_state_measurement_code.isel({state_dim: np.flatnonzero(ocean_states)})
        .rename({state_dim: "region_ocean"})
        .transpose(*ocean_codes.dims)
    )
    np.testing.assert_array_equal(
        ff_codes,
        expected_ff_codes,
    )
    np.testing.assert_array_equal(
        ocean_codes,
        expected_ocean_codes,
    )
    observation_label = registry.auxiliary_coords["observation_label"]
    assert observation_label.dims == ("nmeasure",)
    assert observation_label.values.tolist() == ["shared-observation"]


def test_multisector_runner_rejects_shared_basis_h_layout_mismatch() -> None:
    """Retained shared-basis coordinates must match the prepared design state."""
    model_spec = RhimeModelSpec(
        species="ch4",
        domain="EUROPE",
        sectors=(
            SectorSpec("FF", "ff-inventory", {"pdf": "normal", "mu": 1.0, "sigma": 0.2}, "ff"),
            SectorSpec(
                "ocean",
                "ocean-inventory",
                {"pdf": "normal", "mu": 1.0, "sigma": 0.3},
                "ocean",
            ),
        ),
    )
    inv_inputs = xr.Dataset(
        {
            "H": (
                ("region", "nmeasure", "source"),
                np.ones((2, 1, 2)),
            )
        },
        coords={
            "region": [0, 1],
            "nmeasure": [0],
            "source": ["ff-inventory", "ocean-inventory"],
        },
    )

    with pytest.raises(
        ValueError,
        match="Retained source-specific basis.*basis has 1 regions.*prepared H has 2",
    ):
        rhime_module._validate_multisector_basis_layout(
            _fake_basis_functions(),
            model_spec,
            inv_inputs,
        )


def test_prepare_rhime_inputs_single_sector_reloads_merged_data(
    tac_ch4_data_args, merged_data_dir, merged_data_file_name, default_bc_basis_directory
) -> None:
    args = _rhime_preparation_args(
        tac_ch4_data_args,
        tac_ch4_data_args["emissions_name"],
        default_bc_basis_directory,
    )
    args.update(
        {
            "reload_merged_data": True,
            "merged_data_dir": str(merged_data_dir),
            "merged_data_name": merged_data_file_name,
        }
    )

    prepared = prepare_rhime_inputs(**args)

    assert isinstance(prepared, RhimePreparedInputs)
    assert prepared.basis_artifact_source == "generated"
    assert isinstance(prepared.basis_functions, BasisFunctions)
    assert prepared.sites == ("TAC",)
    assert "source" not in prepared.inv_inputs["H"].dims
    for legacy_attr in ("fp_all", "fp_data", "basis", "flux", "basis_objects"):
        assert not hasattr(prepared, legacy_attr)


def test_prepare_rhime_inputs_multisector_keeps_source_dimension(
    tac_ch4_data_args, default_bc_basis_directory
) -> None:
    flux_sources = ["total-ukghg-edgar7", "total-ukghg-edgar7-shuffled"]
    args = _rhime_preparation_args(tac_ch4_data_args, flux_sources, default_bc_basis_directory)
    args["split_by_sectors"] = True

    prepared = prepare_rhime_inputs(**args)

    assert prepared.basis_artifact_source == "generated"
    assert isinstance(prepared.basis_functions, BasisFunctions)
    assert "source" in prepared.inv_inputs["H"].dims
    assert list(prepared.inv_inputs["H"].coords["source"].values) == flux_sources
    assert "source_region_count" not in prepared.inv_inputs["H"].coords


def test_multisector_sensitivity_sources_fail_before_site_gathering() -> None:
    """Gathered sensitivity is validated before missing sources can be synthesized."""
    time = pd.date_range("2019-01-01", periods=2, freq="h")
    state_index = pd.MultiIndex.from_tuples(
        [("ff-inventory", 0)],
        names=["source", "region_in_source"],
    )
    sensitivity = xr.DataArray(
        np.ones((1, 2)),
        dims=("state", "time"),
        coords={
            **xr.Coordinates.from_pandas_multiindex(state_index, "state"),
            "time": time,
        },
    )
    fp_x_flux = xr.DataArray(
        np.ones((2, 2)),
        dims=("time", "source"),
        coords={
            "time": time,
            "source": ["ff-inventory", "ocean-inventory"],
        },
        name="fp_x_flux_sectoral",
    )

    class MissingSourceBasis:
        def sensitivity(self, _: xr.DataArray) -> xr.DataArray:
            return sensitivity

    merged = prep_module._MergedInversionData(
        fp_all={"TAC": xr.Dataset({"fp_x_flux_sectoral": fp_x_flux})},
        site_options=_site_options(["TAC"], averaging_period=["1H"]),
    )

    with pytest.raises(
        ValueError,
        match="Site 'TAC'.*missing source\\(s\\): \\['ocean-inventory'\\]",
    ):
        prep_module._rhime_site_data_from_basis_functions(
            merged=merged,
            basis_functions=cast(BasisFunctions, MissingSourceBasis()),
            domain="EUROPE",
            split_by_sectors=True,
            flux_sources=["ff-inventory", "ocean-inventory"],
            use_bc=False,
            bc_basis_case="NESW",
            bc_basis_directory=None,
        )


def test_multisector_site_preparation_keeps_gathered_source_state() -> None:
    """Modern preparation must not rectangularize ragged source-specific state."""
    time = pd.date_range("2019-01-01", periods=2, freq="h")
    state_index = pd.MultiIndex.from_tuples(
        [("ff-inventory", 0), ("ff-inventory", 1), ("ocean-inventory", 0)],
        names=["source", "region_in_source"],
    )
    sensitivity = xr.DataArray(
        np.ones((3, 2)),
        dims=("state", "time"),
        coords={
            **xr.Coordinates.from_pandas_multiindex(state_index, "state"),
            "time": time,
        },
    )
    fp_x_flux = xr.DataArray(
        np.ones((2, 2)),
        dims=("source", "time"),
        coords={
            "source": ["ff-inventory", "ocean-inventory"],
            "time": time,
        },
        name="fp_x_flux_sectoral",
    )

    class GatheredBasis:
        def sensitivity(self, _: xr.DataArray) -> xr.DataArray:
            return sensitivity

    merged = prep_module._MergedInversionData(
        fp_all={"TAC": xr.Dataset({"fp_x_flux_sectoral": fp_x_flux})},
        site_options=_site_options(["TAC"], averaging_period=["1H"]),
    )

    prepared = prep_module._rhime_site_data_from_basis_functions(
        merged=merged,
        basis_functions=cast(BasisFunctions, GatheredBasis()),
        domain="EUROPE",
        split_by_sectors=True,
        flux_sources=["ff-inventory", "ocean-inventory"],
        use_bc=False,
        bc_basis_case="NESW",
        bc_basis_directory=None,
    )

    xr.testing.assert_identical(prepared["TAC"]["H"], sensitivity.rename("H"))
    assert "source" not in prepared["TAC"]["H"].dims
    assert list(prepared["TAC"]["H"].indexes["state"].names) == [
        "source",
        "region_in_source",
    ]


def test_fixedbasis_preparation_adds_anchored_legacy_sigma_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy preparation retains its anchored component compatibility index."""
    times = pd.to_datetime(["2019-01-08", "2019-01-09", "2019-01-15"])
    inv_inputs = xr.Dataset(
        {
            "H": (("region", "nmeasure"), np.ones((1, 3))),
            "site_indicator": ("nmeasure", np.zeros(3, dtype=int)),
        },
        coords={"region": [0], "nmeasure": np.arange(3), "time": ("nmeasure", times)},
    )
    fp_data = {"TAC": _site_dataset([2.0, 3.0, 4.0])}
    merged = prep_module._MergedInversionData(
        fp_all=fp_data,
        site_options=_site_options(["TAC"], averaging_period=["1H"]),
    )
    basis_functions = _fake_basis_functions()

    monkeypatch.setattr(prep_module, "_prepare_merged_data", lambda **kwargs: merged)
    monkeypatch.setattr(
        prep_module,
        "basis_functions_wrapper",
        lambda **kwargs: (fp_data, {"emissions": basis_functions}),
    )
    monkeypatch.setattr(
        prep_module,
        "_apply_filters_and_drop_empty_sites",
        lambda **kwargs: (fp_data, _site_options(["TAC"], averaging_period=["1H"])),
    )
    monkeypatch.setattr(prep_module, "_set_domain_attrs", lambda *args, **kwargs: None)
    monkeypatch.setattr(prep_module, "_make_inv_inputs", lambda **kwargs: inv_inputs.copy())

    prepared = prep_module.prepare_fixedbasis_inversion_data(
        species="ch4",
        sites=["TAC"],
        domain="EUROPE",
        averaging_period=["1H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        output_name="fixedbasis_sigma",
        flux_sources=["total-ukghg-edgar7"],
        sigma_freq="8D",
        use_bc=False,
    )

    assert prepared.inv_inputs is not None
    np.testing.assert_array_equal(prepared.inv_inputs["sigma_freq_index"], [0, 1, 1])


def test_fixedbasis_preparation_uses_platform_for_sites_retained_after_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Satellite BC scaling receives platform metadata after a surface site is dropped."""
    fp_data = {"OCO2-EASTASIA": _site_dataset([2.0])}
    merged = prep_module._MergedInversionData(
        fp_all={"TAC": _site_dataset([]), **fp_data},
        site_options=_site_options(
            ["TAC", "OCO2-EASTASIA"],
            averaging_period=["1H", "1H"],
            platform=["surface", "satellite"],
        ),
    )
    retained_options = merged.site_options.select_indices([1])
    captured: dict[str, object] = {}

    monkeypatch.setattr(prep_module, "_prepare_merged_data", lambda **kwargs: merged)
    monkeypatch.setattr(
        prep_module,
        "basis_functions_wrapper",
        lambda **kwargs: (fp_data, {"emissions": _fake_basis_functions()}),
    )
    monkeypatch.setattr(
        prep_module,
        "_apply_filters_and_drop_empty_sites",
        lambda **kwargs: (fp_data, retained_options),
    )
    monkeypatch.setattr(prep_module, "_set_domain_attrs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        prep_module,
        "_make_inv_inputs",
        lambda **kwargs: _minimal_prepared_inv_inputs(sites=("OCO2-EASTASIA",)),
    )

    def capture_scaling(inv_inputs: xr.Dataset, **kwargs: object) -> xr.Dataset:
        captured.update(kwargs)
        return inv_inputs

    monkeypatch.setattr(prep_module, "_scale_satellite_bc_sensitivity_to_column_signal", capture_scaling)

    prep_module.prepare_fixedbasis_inversion_data(
        species="co2",
        sites=["TAC", "OCO2-EASTASIA"],
        domain="EASTASIA",
        averaging_period=["1H", "1H"],
        platform=["surface", "satellite"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        output_name="filtered_satellite",
        flux_sources=["test-source"],
        use_bc=False,
    )

    assert captured == {"sites": ["OCO2-EASTASIA"], "platform": ("satellite",)}


def test_rhime_preparation_uses_platform_for_sites_retained_after_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RHIME satellite BC scaling receives the filtered mixed-platform metadata."""
    merged = prep_module._MergedInversionData(
        fp_all={"TAC": _site_dataset([]), "OCO2-EASTASIA": _site_dataset([2.0])},
        site_options=_site_options(
            ["TAC", "OCO2-EASTASIA"],
            averaging_period=["1H", "1H"],
            platform=["surface", "satellite"],
        ),
    )
    filtered_merged = prep_module._MergedInversionData(
        fp_all={"OCO2-EASTASIA": _site_dataset([2.0])},
        site_options=merged.site_options.select_indices([1]),
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(prep_module, "_prepare_merged_data", lambda **kwargs: merged)
    monkeypatch.setattr(prep_module, "_filter_merged_inversion_data", lambda **kwargs: filtered_merged)
    monkeypatch.setattr(prep_module, "make_basis_functions", lambda **kwargs: _fake_basis_functions())
    monkeypatch.setattr(
        prep_module,
        "_rhime_site_data_from_basis_functions",
        lambda **kwargs: {"OCO2-EASTASIA": _site_dataset([2.0])},
    )
    monkeypatch.setattr(
        prep_module,
        "_make_inv_inputs",
        lambda **kwargs: _minimal_prepared_inv_inputs(sites=("OCO2-EASTASIA",)),
    )

    def capture_scaling(inv_inputs: xr.Dataset, **kwargs: object) -> xr.Dataset:
        captured.update(kwargs)
        return inv_inputs

    monkeypatch.setattr(prep_module, "_scale_satellite_bc_sensitivity_to_column_signal", capture_scaling)

    prepare_rhime_inputs(
        species="co2",
        sites=["TAC", "OCO2-EASTASIA"],
        domain="EASTASIA",
        averaging_period=["1H", "1H"],
        platform=["surface", "satellite"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        output_name="filtered_satellite",
        flux_sources=["test-source"],
        use_bc=False,
    )

    assert captured == {"sites": ("OCO2-EASTASIA",), "platform": ("satellite",)}


def test_prepare_rhime_inputs_uses_basis_sensitivity_without_legacy_side_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RHIME preparation derives sensitivity without legacy side channels."""
    site_data = _site_dataset([2.0])
    sensitivity = xr.DataArray(
        [[8.0]],
        dims=("state", "time"),
        coords={"state": [0], "time": site_data.time},
        name="H",
    )
    expected_sensitivity = sensitivity
    basis_functions = _SpyBasisFunctions(sensitivity)
    captured_fp_data_keys: set[str] = set()

    def fake_data_processing_surface_notracer(
        **kwargs: object,
    ) -> tuple[dict, list[str], list[str], list[str], list[str], list[str]]:
        return (
            {"TAC": site_data, ".species": "CH4"},
            ["TAC"],
            ["185m"],
            ["185m"],
            ["instrument-1"],
            ["1H"],
        )

    def fake_make_basis_functions(**kwargs: object) -> _SpyBasisFunctions:
        assert "return_basis_objects" not in kwargs
        fp_all = kwargs["fp_all"]
        assert isinstance(fp_all, dict)
        xr.testing.assert_allclose(fp_all["TAC"], site_data)
        return basis_functions

    def fake_make_inv_inputs(fp_data: dict, sites: list[str], **kwargs: object) -> xr.Dataset:
        """Capture direct-sensitivity inputs without constructing a model."""
        nonlocal captured_fp_data_keys
        captured_fp_data_keys = set(fp_data)
        assert sites == ["TAC"]
        assert set(fp_data) == {"TAC"}
        xr.testing.assert_identical(fp_data["TAC"]["H"], expected_sensitivity)
        return _minimal_prepared_inv_inputs()

    def forbidden_basis_functions_wrapper(*args: object, **kwargs: object) -> None:
        raise AssertionError("RHIME preparation should use make_basis_functions and direct sensitivity.")

    monkeypatch.setattr(
        prep_module,
        "data_processing_surface_notracer",
        fake_data_processing_surface_notracer,
    )
    monkeypatch.setattr(prep_module, "make_basis_functions", fake_make_basis_functions)
    monkeypatch.setattr(prep_module, "make_inv_inputs", fake_make_inv_inputs)
    monkeypatch.setattr(
        prep_module,
        "basis_functions_wrapper",
        forbidden_basis_functions_wrapper,
        raising=False,
    )

    prepared = prepare_rhime_inputs(
        species="ch4",
        sites=["TAC"],
        domain="EUROPE",
        averaging_period=["1H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        output_name="direct_sensitivity",
        flux_sources=["total-ukghg-edgar7"],
        use_bc=False,
    )

    assert prepared.basis_functions is basis_functions
    assert prepared.basis_artifact_source == "datatree"
    assert captured_fp_data_keys == {"TAC"}
    assert len(basis_functions.sensitivity_calls) == 1
    xr.testing.assert_allclose(basis_functions.sensitivity_calls[0], site_data["fp_x_flux"])


def test_prepare_rhime_inputs_matches_direct_sensitivity_inv_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_data = _site_dataset([2.0, 3.0])
    basis_functions = _fake_basis_functions()

    def fake_data_processing_surface_notracer(
        **kwargs: object,
    ) -> tuple[dict, list[str], list[str], list[str], list[str], list[str]]:
        return (
            {"TAC": site_data, ".species": "CH4"},
            ["TAC"],
            ["185m"],
            ["185m"],
            ["instrument-1"],
            ["1H"],
        )

    def fake_make_basis_functions(**kwargs: object) -> BasisFunctions:
        return basis_functions

    monkeypatch.setattr(
        prep_module,
        "data_processing_surface_notracer",
        fake_data_processing_surface_notracer,
    )
    monkeypatch.setattr(prep_module, "make_basis_functions", fake_make_basis_functions)

    prepared = prepare_rhime_inputs(
        species="ch4",
        sites=["TAC"],
        domain="EUROPE",
        averaging_period=["1H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        output_name="equivalence",
        flux_sources=["total-ukghg-edgar7"],
        use_bc=False,
    )

    expected_site_data = prep_module._select_fp_all_sites({"TAC": site_data}, ["TAC"])["TAC"]
    expected_site_data["H"] = basis_functions.sensitivity(expected_site_data["fp_x_flux"])
    expected_inv_inputs = make_inv_inputs(
        {"TAC": expected_site_data},
        sites=["TAC"],
        bc_freq=None,
        min_error=0.0,
        start_date="2019-01-01",
    )

    xr.testing.assert_identical(prepared.inv_inputs["H"], expected_inv_inputs["H"])
    xr.testing.assert_identical(prepared.inv_inputs["mf"], expected_inv_inputs["mf"])


def test_prepare_rhime_inputs_prunes_reloaded_merged_data_to_requested_sites(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reload pruning removes unrequested sites and site-keyed metadata."""
    captured_fp_all_keys: set[str] = set()

    def fake_load_merged_data(*args: object, **kwargs: object) -> dict:
        return {
            "TAC": _site_dataset([2.0]),
            "MHD": _site_dataset([3.0]),
            ".flux": object(),
            ".species": "CH4",
            ".scales": {"TAC": "tac-scale", "MHD": "mhd-scale"},
            ".units": 1e-9,
        }

    def fake_make_basis_functions(**kwargs: object) -> BasisFunctions:
        nonlocal captured_fp_all_keys
        fp_all = kwargs["fp_all"]
        assert isinstance(fp_all, dict)
        captured_fp_all_keys = set(fp_all)
        return _fake_basis_functions()

    def fake_make_inv_inputs(fp_data: dict, sites: list[str], **kwargs: object) -> xr.Dataset:
        """Return minimal inputs after checking retained reload sites."""
        assert sites == ["TAC"]
        return _minimal_prepared_inv_inputs()

    monkeypatch.setattr(prep_module, "load_merged_data", fake_load_merged_data)
    monkeypatch.setattr(prep_module, "make_basis_functions", fake_make_basis_functions)
    monkeypatch.setattr(prep_module, "make_inv_inputs", fake_make_inv_inputs)

    prepare_rhime_inputs(
        species="ch4",
        sites=["TAC"],
        domain="EUROPE",
        averaging_period=["1H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        output_name="reload_prune",
        flux_sources=["total-ukghg-edgar7"],
        reload_merged_data=True,
        merged_data_dir=str(tmp_path),
        use_bc=False,
    )

    assert "TAC" in captured_fp_all_keys
    assert "MHD" not in captured_fp_all_keys
    assert {key for key in captured_fp_all_keys if key.startswith(".")} == {
        ".flux",
        ".species",
        ".scales",
        ".split_by_sectors",
        ".units",
    }


def test_prepare_merged_data_reload_keeps_all_options_aligned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reloading a subset retains the complete option record for each kept site."""
    monkeypatch.setattr(
        prep_module,
        "load_merged_data",
        lambda *args, **kwargs: {
            "MHD": _site_dataset([3.0]),
            ".species": "CH4",
            ".units": 1e-9,
        },
    )

    merged = prep_module._prepare_merged_data(
        species="ch4",
        sites=["TAC", "MHD", "RGL"],
        domain="EUROPE",
        averaging_period=["1H", "2H", "3H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        output_name="reload_alignment",
        flux_sources=["total-ukghg-edgar7"],
        inlet=["100m", "200m", "300m"],
        fp_height=["110m", "210m", "310m"],
        instrument=["inst-tac", "inst-mhd", "inst-rgl"],
        platform=["surface", "flask", "site-column"],
        obs_data_level=["level-tac", "level-mhd", "level-rgl"],
        met_model=["met-tac", "met-mhd", "met-rgl"],
        max_level=[10, 20, 30],
        reload_merged_data=True,
        merged_data_dir=str(tmp_path),
        use_bc=False,
    )

    assert merged.site_options == _site_options(
        ["MHD"],
        averaging_period=["2H"],
        inlet=["200m"],
        fp_height=["210m"],
        instrument=["inst-mhd"],
        platform=["flask"],
        obs_data_level=["level-mhd"],
        met_model=["met-mhd"],
        max_level=[20],
    )
    assert set(merged.fp_all) == {"MHD", ".species", ".split_by_sectors", ".units"}


def test_site_options_direct_construction_enforces_immutable_alignment() -> None:
    """The tuple-backed record rejects direct construction with drifted fields."""
    with pytest.raises(ValueError, match="same length"):
        prep_module._SiteOptions(
            sites=("TAC", "MHD"),
            averaging_period=("1H",),
            inlet=(None, None),
            fp_height=(None, None),
            instrument=(None, None),
            platform=(None, None),
            obs_data_level=(None, None),
            met_model=(None, None),
            max_level=(None, None),
        )

    options = _site_options(["TAC"], averaging_period=["1H"])
    assert isinstance(options.sites, tuple)
    with pytest.raises(AttributeError):
        options.sites.append("MHD")  # type: ignore[attr-defined]


def test_prepare_merged_data_retrieval_keeps_requested_metadata_authoritative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A middle retrieval failure retains every option from the requested record."""

    def fake_data_processing(
        **kwargs: object,
    ) -> tuple[dict, list[str], list[str], list[str], list[str], list[str]]:
        """Return the first and third requested sites using the legacy tuple."""
        return (
            {
                "TAC": _site_dataset([2.0]),
                "RGL": _site_dataset([4.0]),
                ".species": "CH4",
            },
            ["TAC", "RGL"],
            ["100m", "300m"],
            ["110m", "310m"],
            ["inst-tac", "inst-rgl"],
            ["1H", "3H"],
        )

    monkeypatch.setattr(
        prep_module,
        "data_processing_surface_notracer",
        fake_data_processing,
    )

    merged = prep_module._prepare_merged_data(
        species="ch4",
        sites=["TAC", "MHD", "RGL"],
        domain="EUROPE",
        averaging_period=["1H", "2H", "3H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        output_name="retrieval_alignment",
        flux_sources=["inventory"],
        inlet=["100m", "200m", "300m"],
        fp_height=["110m", "210m", "310m"],
        instrument=["inst-tac", "inst-mhd", "inst-rgl"],
        platform=["surface", "flask", "site-column"],
        obs_data_level=["level-tac", "level-mhd", "level-rgl"],
        met_model=["met-tac", "met-mhd", "met-rgl"],
        max_level=[10, 20, 30],
        use_bc=False,
    )

    assert merged.site_options == _site_options(
        ["TAC", "RGL"],
        averaging_period=["1H", "3H"],
        inlet=["100m", "300m"],
        fp_height=["110m", "310m"],
        instrument=["inst-tac", "inst-rgl"],
        platform=["surface", "site-column"],
        obs_data_level=["level-tac", "level-rgl"],
        met_model=["met-tac", "met-rgl"],
        max_level=[10, 30],
    )


def test_prepare_merged_data_ignores_redundant_retrieval_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Length-correct legacy metadata cannot replace requested site pairings."""
    monkeypatch.setattr(
        prep_module,
        "data_processing_surface_notracer",
        lambda **kwargs: (
            {"TAC": _site_dataset([2.0]), ".species": "CH4"},
            ["TAC"],
            ["wrong-inlet"],
            ["110m"],
            ["inst-tac"],
            ["1H"],
        ),
    )

    merged = prep_module._prepare_merged_data(
        species="ch4",
        sites=["TAC"],
        domain="EUROPE",
        averaging_period=["1H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        output_name="retrieval_disagreement",
        flux_sources=["inventory"],
        inlet=["100m"],
        fp_height=["110m"],
        instrument=["inst-tac"],
        use_bc=False,
    )

    assert merged.site_options.inlet == ("100m",)


def test_site_options_accept_numpy_integer_max_levels() -> None:
    """NumPy integral levels normalize to immutable Python integers."""
    scalar = _site_options(["TAC"], max_level=np.int64(17))
    aligned = _site_options(["TAC", "MHD"], max_level=[None, np.int32(21)])

    assert scalar.max_level == (17,)
    assert aligned.max_level == (None, 21)
    assert type(aligned.max_level[1]) is int


def test_prepare_rhime_inputs_reload_all_sites_missing_fails_before_basis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reloading with no requested sites fails before basis construction."""
    monkeypatch.setattr(
        prep_module,
        "load_merged_data",
        lambda *args, **kwargs: {
            "RGL": _site_dataset([4.0]),
            ".species": "CH4",
            ".units": 1e-9,
        },
    )

    def fail_make_basis_functions(**kwargs: object) -> BasisFunctions:
        """Fail if basis construction starts without a requested site."""
        raise AssertionError("Basis generation should not run without a requested site.")

    monkeypatch.setattr(prep_module, "make_basis_functions", fail_make_basis_functions)

    with pytest.raises(ValueError, match="does not include any requested sites"):
        prepare_rhime_inputs(
            species="ch4",
            sites=["TAC", "MHD"],
            domain="EUROPE",
            averaging_period=["1H", "2H"],
            start_date="2019-01-01",
            end_date="2019-02-01",
            output_name="reload_all_missing",
            flux_sources=["total-ukghg-edgar7"],
            reload_merged_data=True,
            merged_data_dir=str(tmp_path),
            use_bc=False,
        )


def test_apply_filters_drops_complete_site_option_record() -> None:
    """Dropping an empty site removes every option aligned to that site."""
    site_options = _site_options(
        ["TAC", "MHD", "RGL"],
        averaging_period=["1H", "2H", "3H"],
        inlet=["100m", "200m", "300m"],
        fp_height=["110m", "210m", "310m"],
        instrument=["inst-tac", "inst-mhd", "inst-rgl"],
        platform=["surface", "flask", "site-column"],
        obs_data_level=["level-tac", "level-mhd", "level-rgl"],
        met_model=["met-tac", "met-mhd", "met-rgl"],
        max_level=[10, 20, 30],
    )
    fp_data = {
        "TAC": _site_dataset([2.0]),
        "MHD": _site_dataset([]),
        "RGL": _site_dataset([4.0]),
    }

    filtered, retained = prep_module._apply_filters_and_drop_empty_sites(
        fp_data=fp_data,
        site_options=site_options,
        filters=None,
    )

    assert set(filtered) == {"TAC", "RGL"}
    assert retained == site_options.select_indices([0, 2])


def test_filtering_prunes_scales_with_empty_sites() -> None:
    """Reload filtering prunes calibration provenance for an empty site."""
    merged = prep_module._MergedInversionData(
        fp_all={
            "TAC": _site_dataset([]),
            "MHD": _site_dataset([3.0]),
            ".scales": {"TAC": "tac-scale", "MHD": "mhd-scale"},
            ".units": 1e-9,
        },
        site_options=_site_options(["TAC", "MHD"], averaging_period=["1H", "1H"]),
    )

    filtered = prep_module._filter_merged_inversion_data(merged=merged, filters=None)

    assert filtered.sites == ("MHD",)
    assert filtered.fp_all[".scales"] == {"MHD": "mhd-scale"}
    assert filtered.fp_all[".units"] == pytest.approx(1e-9)


@pytest.mark.parametrize(
    ("averaging_period", "expected"),
    [
        ("1H", ["1H", "1H"]),
        (None, [None, None]),
    ],
)
def test_prepare_rhime_inputs_normalises_averaging_period_to_site_count(
    monkeypatch: pytest.MonkeyPatch,
    averaging_period: str | None,
    expected: list[str | None],
) -> None:
    """Scalar and None averaging periods broadcast across requested sites."""
    captured_averaging_period: list[str | None] | None = None
    site_data = {"TAC": _site_dataset([2.0]), "MHD": _site_dataset([3.0])}

    def fake_data_processing_surface_notracer(
        **kwargs: object,
    ) -> tuple[dict, list[str], list[str], list[str], list[str], list[str | None]]:
        nonlocal captured_averaging_period
        averaging_period_arg = kwargs["averaging_period"]
        assert isinstance(averaging_period_arg, list)
        captured_averaging_period = averaging_period_arg
        return (
            {**site_data, ".species": "CH4"},
            ["TAC", "MHD"],
            ["185m", "10m"],
            ["185m", "10m"],
            ["instrument-1", "instrument-2"],
            captured_averaging_period,
        )

    def fake_make_basis_functions(**kwargs: object) -> _DynamicSpyBasisFunctions:
        return _DynamicSpyBasisFunctions()

    def fake_make_inv_inputs(fp_data: dict, sites: list[str], **kwargs: object) -> xr.Dataset:
        """Return minimal gathered inputs for averaging-period normalization."""
        assert sites == ["TAC", "MHD"]
        return _minimal_prepared_inv_inputs(("TAC", "MHD"))

    monkeypatch.setattr(
        prep_module,
        "data_processing_surface_notracer",
        fake_data_processing_surface_notracer,
    )
    monkeypatch.setattr(prep_module, "make_basis_functions", fake_make_basis_functions)
    monkeypatch.setattr(prep_module, "make_inv_inputs", fake_make_inv_inputs)

    prepared = prepare_rhime_inputs(
        species="ch4",
        sites=["TAC", "MHD"],
        domain="EUROPE",
        averaging_period=averaging_period,
        start_date="2019-01-01",
        end_date="2019-02-01",
        output_name="normalise_avg",
        flux_sources=["total-ukghg-edgar7"],
        use_bc=False,
    )

    assert captured_averaging_period == expected
    assert prepared.averaging_period == tuple(expected)


def test_prepare_rhime_inputs_rejects_misaligned_averaging_period_list() -> None:
    with pytest.raises(ValueError, match="List averaging_period does not have specified length"):
        prepare_rhime_inputs(
            species="ch4",
            sites=["TAC", "MHD"],
            domain="EUROPE",
            averaging_period=["1H"],
            start_date="2019-01-01",
            end_date="2019-02-01",
            output_name="bad_avg",
            flux_sources=["total-ukghg-edgar7"],
            use_bc=False,
        )


@pytest.mark.parametrize("averaging_period", [1, ["1H", 2]])
def test_prepare_rhime_inputs_rejects_non_string_averaging_period_values(
    averaging_period: object,
) -> None:
    with pytest.raises(ValueError, match="averaging_period"):
        prepare_rhime_inputs(
            species="ch4",
            sites=["TAC", "MHD"],
            domain="EUROPE",
            averaging_period=cast(Any, averaging_period),
            start_date="2019-01-01",
            end_date="2019-02-01",
            output_name="bad_avg_type",
            flux_sources=["total-ukghg-edgar7"],
            use_bc=False,
        )


def test_run_rhime_leaves_scalar_averaging_period_for_shared_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_averaging_period: object = None
    original_signature = inspect.signature(prepare_rhime_inputs)

    def fake_prepare_rhime_inputs(**kwargs: object) -> None:
        nonlocal captured_averaging_period
        captured_averaging_period = kwargs["averaging_period"]
        raise RuntimeError("stop after data argument capture")

    setattr(fake_prepare_rhime_inputs, "__signature__", original_signature)
    monkeypatch.setattr(rhime_module, "prepare_rhime_inputs", fake_prepare_rhime_inputs)

    with pytest.raises(RuntimeError, match="stop after data argument capture"):
        run_rhime(
            species="ch4",
            sites=["TAC", "MHD"],
            domain="EUROPE",
            averaging_period="1H",
            start_date="2019-01-01",
            end_date="2019-02-01",
            output_name="avg_scalar",
            flux_sources=["total-ukghg-edgar7"],
            output_format="none",
        )

    assert captured_averaging_period == "1H"


def test_prepare_rhime_inputs_treats_min_error_none_as_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A None minimum-error value is normalized to the numeric default."""
    captured_min_error: object = None
    site_data = _site_dataset([2.0])

    def fake_data_processing_surface_notracer(
        **kwargs: object,
    ) -> tuple[dict, list[str], list[str], list[str], list[str], list[str]]:
        return (
            {"TAC": site_data, ".species": "CH4"},
            ["TAC"],
            ["185m"],
            ["185m"],
            ["instrument-1"],
            ["1H"],
        )

    def fake_make_basis_functions(**kwargs: object) -> _DynamicSpyBasisFunctions:
        return _DynamicSpyBasisFunctions()

    def fake_make_inv_inputs(fp_data: dict, sites: list[str], **kwargs: object) -> xr.Dataset:
        """Capture the normalized minimum-error value."""
        nonlocal captured_min_error
        captured_min_error = kwargs["min_error"]
        return _minimal_prepared_inv_inputs()

    monkeypatch.setattr(
        prep_module,
        "data_processing_surface_notracer",
        fake_data_processing_surface_notracer,
    )
    monkeypatch.setattr(prep_module, "make_basis_functions", fake_make_basis_functions)
    monkeypatch.setattr(prep_module, "make_inv_inputs", fake_make_inv_inputs)

    prepare_rhime_inputs(
        species="ch4",
        sites=["TAC"],
        domain="EUROPE",
        averaging_period=["1H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        output_name="min_error_none",
        flux_sources=["total-ukghg-edgar7"],
        use_bc=False,
        min_error=None,
    )

    assert captured_min_error == 0.0


def test_make_inv_inputs_boundary_propagates_valid_by_site_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preparation passes a valid per-site minimum-error choice to assembly."""
    captured_by_site: bool | None = None

    def fake_make_inv_inputs(*args: object, **kwargs: object) -> xr.Dataset:
        """Capture the canonical per-site minimum-error flag."""
        nonlocal captured_by_site
        captured_by_site = cast(bool, kwargs["min_error_per_site"])
        return _minimal_inv_inputs()

    monkeypatch.setattr(prep_module, "make_inv_inputs", fake_make_inv_inputs)

    prep_module._make_inv_inputs(
        fp_data={"TAC": _site_dataset([2.0])},
        sites=["TAC"],
        start_date="2019-01-01",
        bc_freq=None,
        min_error="residual",
        calculate_min_error=None,
        min_error_per_site=True,
    )

    assert captured_by_site is True


def test_prepare_rhime_inputs_rejects_min_error_options_before_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct preparation validates minimum-error options before data access."""

    def fail_data_processing(**kwargs: object) -> None:
        """Fail if invalid options reach the retrieval boundary."""
        raise AssertionError("Data retrieval should not run for invalid min-error options.")

    monkeypatch.setattr(
        prep_module,
        "data_processing_surface_notracer",
        fail_data_processing,
    )

    with pytest.raises(ValueError, match="unsupported option"):
        prepare_rhime_inputs(
            species="ch4",
            sites=["TAC"],
            domain="EUROPE",
            averaging_period=["1H"],
            start_date="2019-01-01",
            end_date="2019-02-01",
            output_name="invalid_min_error_options",
            flux_sources=["total-ukghg-edgar7"],
            min_error_options={"robust": False},
            use_bc=False,
        )


def test_prepare_rhime_inputs_filters_sites_before_basis_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One filter pass should feed basis, flux, and BC sensitivity construction."""
    site_data = {"TAC": _site_dataset([2.0, 3.0, 4.0])}
    basis_functions = _DynamicSpyBasisFunctions()
    filtering_calls = 0
    captured_basis_times: tuple[np.datetime64, ...] | None = None
    captured_bc_times: tuple[np.datetime64, ...] | None = None
    retained_times = tuple(site_data["TAC"].time.isel(time=[1]).values)

    def fake_data_processing_surface_notracer(
        **kwargs: object,
    ) -> tuple[dict, list[str], list[str], list[str], list[str], list[str]]:
        return (
            {**site_data, ".species": "CH4"},
            ["TAC"],
            ["185m"],
            ["185m"],
            ["instrument-1"],
            ["1H"],
        )

    def fake_filtering(fp_data: dict, filters: object) -> dict:
        nonlocal filtering_calls
        filtering_calls += 1
        assert filters == ["keep-middle"]
        assert "H" not in fp_data["TAC"]
        return {"TAC": fp_data["TAC"].isel(time=[1])}

    def fake_make_basis_functions(**kwargs: object) -> _DynamicSpyBasisFunctions:
        nonlocal captured_basis_times
        fp_all = kwargs["fp_all"]
        assert isinstance(fp_all, dict)
        captured_basis_times = tuple(fp_all["TAC"].time.values)
        return basis_functions

    def fake_bc_sensitivity(fp_data: dict, **kwargs: object) -> dict:
        """Record filtered times and add matching boundary-condition sensitivity."""
        nonlocal captured_bc_times
        captured_bc_times = tuple(fp_data["TAC"].time.values)
        fp_data["TAC"]["H_bc"] = xr.DataArray(
            np.ones((1, len(retained_times))),
            dims=("bc_region", "time"),
            coords={"bc_region": [0], "time": fp_data["TAC"].time},
        )
        return fp_data

    def fake_make_inv_inputs(fp_data: dict, sites: list[str], **kwargs: object) -> xr.Dataset:
        """Validate filtering occurred before inversion-input assembly."""
        assert sites == ["TAC"]
        assert tuple(fp_data["TAC"].time.values) == retained_times
        assert tuple(fp_data["TAC"]["H"].time.values) == retained_times
        assert tuple(fp_data["TAC"]["H_bc"].time.values) == retained_times
        return _minimal_prepared_inv_inputs()

    monkeypatch.setattr(
        prep_module,
        "data_processing_surface_notracer",
        fake_data_processing_surface_notracer,
    )
    monkeypatch.setattr(prep_module, "filtering", fake_filtering)
    monkeypatch.setattr(prep_module, "make_basis_functions", fake_make_basis_functions)
    monkeypatch.setattr(prep_module, "bc_sensitivity", fake_bc_sensitivity)
    monkeypatch.setattr(prep_module, "make_inv_inputs", fake_make_inv_inputs)

    prepared = prepare_rhime_inputs(
        species="ch4",
        sites=["TAC"],
        domain="EUROPE",
        averaging_period=["1H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        output_name="filter_before_basis",
        flux_sources=["total-ukghg-edgar7"],
        use_bc=True,
        filters=["keep-middle"],
    )

    assert prepared.sites == ("TAC",)
    assert prepared.averaging_period == ("1H",)
    assert captured_basis_times == retained_times
    assert tuple(basis_functions.sensitivity_calls[0].time.values) == retained_times
    assert captured_bc_times == retained_times
    assert filtering_calls == 1


def test_satellite_bc_sensitivity_is_scaled_to_corrected_column_signal() -> None:
    """Satellite H_bc is reduced into the same OCO corrected-column space as mf."""
    inv_inputs = xr.Dataset(
        {
            "H_bc": (
                ("bc_region", "nmeasure"),
                np.array([[100.0, 200.0], [300.0, 400.0]], dtype=float),
            ),
            "mf": ("nmeasure", np.array([50.0, 100.0], dtype=float)),
            "mf_prior_factor": ("nmeasure", np.array([0.0, 0.0], dtype=float)),
            "mf_prior_upper_level_factor": ("nmeasure", np.array([350.0, 300.0], dtype=float)),
            "site": ("nmeasure", np.array(["OCO2-EASTASIA", "OCO2-EASTASIA"])),
        }
    )

    result = prep_module._scale_satellite_bc_sensitivity_to_column_signal(
        inv_inputs,
        sites=["OCO2-EASTASIA"],
        platform=["satellite"],
    )

    np.testing.assert_allclose(
        result["H_bc"].values,
        np.array([[12.5, 50.0], [37.5, 100.0]]),
    )
    assert "satellite_column_bc_scale" in result["H_bc"].attrs


def test_surface_bc_sensitivity_is_not_scaled_by_column_factors() -> None:
    """Non-satellite H_bc is unchanged even if similarly named diagnostics exist."""
    inv_inputs = xr.Dataset(
        {
            "H_bc": (
                ("bc_region", "nmeasure"),
                np.array([[100.0, 200.0], [300.0, 400.0]], dtype=float),
            ),
            "mf": ("nmeasure", np.array([50.0, 100.0], dtype=float)),
            "mf_prior_factor": ("nmeasure", np.array([0.0, 0.0], dtype=float)),
            "mf_prior_upper_level_factor": ("nmeasure", np.array([350.0, 300.0], dtype=float)),
            "site": ("nmeasure", np.array(["TAC", "TAC"])),
        }
    )

    result = prep_module._scale_satellite_bc_sensitivity_to_column_signal(
        inv_inputs,
        sites=["TAC"],
        platform=[None],
    )

    xr.testing.assert_identical(result["H_bc"], inv_inputs["H_bc"])


def test_prepare_rhime_inputs_applies_daily_median_before_sensitivity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Daily-median filtering should aggregate times before basis projection."""
    site_dataset = _site_dataset([2.0, 3.0, 4.0]).drop_vars(["fp_x_flux", "lat", "lon"])
    site_dataset["fp_x_flux"] = xr.DataArray(
        [[[0.0, 100.0]], [[0.0, 0.0]], [[100.0, 0.0]]],
        dims=("time", "lat", "lon"),
        coords={"time": site_dataset.time, "lat": [0.0], "lon": [0.0, 1.0]},
    )
    basis_input: xr.DataArray | None = None
    final_sensitivity: xr.DataArray | None = None

    def fake_data_processing_surface_notracer(
        **kwargs: object,
    ) -> tuple[dict, list[str], list[str], list[str], list[str], list[str]]:
        return (
            {"TAC": site_dataset, ".species": "CH4"},
            ["TAC"],
            ["185m"],
            ["185m"],
            ["instrument-1"],
            ["1H"],
        )

    def fake_make_basis_functions(**kwargs: object) -> BasisFunctions:
        nonlocal basis_input
        fp_all = kwargs["fp_all"]
        assert isinstance(fp_all, dict)
        basis_input = fp_all["TAC"]["fp_x_flux"].copy()
        basis = xr.DataArray(
            [[1, 1]],
            dims=("lat", "lon"),
            coords={"lat": [0.0], "lon": [0.0, 1.0]},
        )
        return BasisFunctions.from_flat_basis(basis_flat=basis, flux=xr.ones_like(basis))

    def fake_make_inv_inputs(fp_data: dict, sites: list[str], **kwargs: object) -> xr.Dataset:
        """Capture sensitivity after daily-median filtering."""
        nonlocal final_sensitivity
        final_sensitivity = fp_data["TAC"]["H"].copy()
        return _minimal_prepared_inv_inputs()

    monkeypatch.setattr(
        prep_module,
        "data_processing_surface_notracer",
        fake_data_processing_surface_notracer,
    )
    monkeypatch.setattr(prep_module, "make_basis_functions", fake_make_basis_functions)
    monkeypatch.setattr(prep_module, "make_inv_inputs", fake_make_inv_inputs)

    prepare_rhime_inputs(
        species="ch4",
        sites=["TAC"],
        domain="EUROPE",
        averaging_period=["1H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        output_name="daily_median_filter_order",
        flux_sources=["total-ukghg-edgar7"],
        use_bc=False,
        filters=["daily_median"],
    )

    assert basis_input is not None
    assert basis_input.sizes["time"] == 1
    xr.testing.assert_allclose(basis_input, xr.zeros_like(basis_input))
    assert final_sensitivity is not None
    xr.testing.assert_identical(final_sensitivity.time, basis_input.time)
    xr.testing.assert_allclose(final_sensitivity, xr.zeros_like(final_sensitivity))


def test_prepare_rhime_inputs_filters_multisector_sites_before_basis_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multisector filtering should preserve source dimensions on retained times."""
    site_dataset = _site_dataset([2.0, 3.0, 4.0])
    flux_sources = ["total-ukghg-edgar7", "sector-2"]
    site_dataset["fp_x_flux_sectoral"] = xr.concat(
        [
            site_dataset["fp_x_flux"],
            2.0 * site_dataset["fp_x_flux"],
        ],
        dim=xr.DataArray(flux_sources, dims="source", name="source"),
    )
    site_data = {"TAC": site_dataset}
    basis_functions = _DynamicSectorSpyBasisFunctions()
    captured_split_by_sectors: object = None
    retained_times = tuple(site_dataset.time.isel(time=[2]).values)

    def fake_data_processing_surface_notracer(
        **kwargs: object,
    ) -> tuple[dict, list[str], list[str], list[str], list[str], list[str]]:
        return (
            {
                **site_data,
                ".flux": {source: object() for source in flux_sources},
                ".species": "CH4",
                ".split_by_sectors": True,
            },
            ["TAC"],
            ["185m"],
            ["185m"],
            ["instrument-1"],
            ["1H"],
        )

    def fake_filtering(fp_data: dict, filters: object) -> dict:
        assert filters == {"TAC": ["keep-last"]}
        assert "H" not in fp_data["TAC"]
        return {"TAC": fp_data["TAC"].isel(time=[2])}

    def fake_make_basis_functions(**kwargs: object) -> _DynamicSectorSpyBasisFunctions:
        nonlocal captured_split_by_sectors
        fp_all = kwargs["fp_all"]
        assert isinstance(fp_all, dict)
        captured_split_by_sectors = fp_all[".split_by_sectors"]
        assert tuple(fp_all["TAC"].time.values) == retained_times
        return basis_functions

    def fake_make_inv_inputs(fp_data: dict, sites: list[str], **kwargs: object) -> xr.Dataset:
        """Validate multisector filtering before input assembly."""
        assert sites == ["TAC"]
        assert tuple(fp_data["TAC"].time.values) == retained_times
        assert tuple(fp_data["TAC"]["H"].time.values) == retained_times
        assert fp_data["TAC"]["H"].dims == ("region", "time", "source")
        assert tuple(fp_data["TAC"]["H"].source.values) == tuple(flux_sources)
        inv_inputs = _minimal_prepared_inv_inputs()
        inv_inputs["H"] = inv_inputs["H"].expand_dims(source=flux_sources)
        return inv_inputs

    monkeypatch.setattr(
        prep_module,
        "data_processing_surface_notracer",
        fake_data_processing_surface_notracer,
    )
    monkeypatch.setattr(prep_module, "filtering", fake_filtering)
    monkeypatch.setattr(prep_module, "make_basis_functions", fake_make_basis_functions)
    monkeypatch.setattr(prep_module, "make_inv_inputs", fake_make_inv_inputs)

    prepared = prepare_rhime_inputs(
        species="ch4",
        sites=["TAC"],
        domain="EUROPE",
        averaging_period=["1H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        output_name="filter_before_sector_basis",
        flux_sources=flux_sources,
        split_by_sectors=True,
        use_bc=False,
        filters={"TAC": ["keep-last"]},
    )

    assert prepared.sites == ("TAC",)
    assert captured_split_by_sectors is True
    assert tuple(basis_functions.sensitivity_calls[0].time.values) == retained_times
    assert tuple(basis_functions.sensitivity_calls[0].source.values) == tuple(flux_sources)
    assert basis_functions.sensitivity_calls[0].name == "fp_x_flux_sectoral"


def test_prepare_rhime_inputs_filters_loaded_basis_before_sensitivity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loaded-basis runs should filter observations once before sensitivity construction."""
    site_data = {"TAC": _site_dataset([2.0, 3.0, 4.0])}
    basis_functions = _DynamicSpyBasisFunctions()
    captured_basis_times: tuple[np.datetime64, ...] | None = None
    filtering_calls = 0
    retained_times = tuple(site_data["TAC"].time.isel(time=[1]).values)

    def fake_data_processing_surface_notracer(
        **kwargs: object,
    ) -> tuple[dict, list[str], list[str], list[str], list[str], list[str]]:
        return (
            {**site_data, ".species": "CH4"},
            ["TAC"],
            ["185m"],
            ["185m"],
            ["instrument-1"],
            ["1H"],
        )

    def fake_make_basis_functions(**kwargs: object) -> _DynamicSpyBasisFunctions:
        nonlocal captured_basis_times
        assert kwargs["fp_basis_case"] == "saved_case"
        fp_all = kwargs["fp_all"]
        assert isinstance(fp_all, dict)
        captured_basis_times = tuple(fp_all["TAC"].time.values)
        return basis_functions

    def fake_filtering(fp_data: dict, filters: object) -> dict:
        nonlocal filtering_calls
        filtering_calls += 1
        assert filters == ["keep-middle"]
        assert "H" not in fp_data["TAC"]
        return {"TAC": fp_data["TAC"].isel(time=[1])}

    def fake_make_inv_inputs(fp_data: dict, sites: list[str], **kwargs: object) -> xr.Dataset:
        """Validate loaded-basis filtering before input assembly."""
        assert sites == ["TAC"]
        assert tuple(fp_data["TAC"].time.values) == retained_times
        assert tuple(fp_data["TAC"]["H"].time.values) == retained_times
        return _minimal_prepared_inv_inputs()

    monkeypatch.setattr(
        prep_module,
        "data_processing_surface_notracer",
        fake_data_processing_surface_notracer,
    )
    monkeypatch.setattr(prep_module, "make_basis_functions", fake_make_basis_functions)
    monkeypatch.setattr(prep_module, "filtering", fake_filtering)
    monkeypatch.setattr(prep_module, "make_inv_inputs", fake_make_inv_inputs)

    prepared = prepare_rhime_inputs(
        species="ch4",
        sites=["TAC"],
        domain="EUROPE",
        averaging_period=["1H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        output_name="filter_loaded_basis",
        flux_sources=["total-ukghg-edgar7"],
        fp_basis_case="saved_case",
        use_bc=False,
        filters=["keep-middle"],
    )

    assert prepared.sites == ("TAC",)
    assert captured_basis_times == retained_times
    assert tuple(basis_functions.sensitivity_calls[0].time.values) == retained_times
    assert filtering_calls == 1


def test_prepare_rhime_inputs_aligns_averaging_period_after_empty_site_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filtering should keep site metadata aligned after dropping an empty site."""
    site_data = {"TAC": _site_dataset([2.0]), "MHD": _site_dataset([])}
    captured_basis_sites: list[str] | None = None

    def fake_data_processing_surface_notracer(
        **kwargs: object,
    ) -> tuple[dict, list[str], list[str], list[str], list[str], list[str]]:
        return (
            {**site_data, ".species": "CH4"},
            ["TAC", "MHD"],
            ["185m", "10m"],
            ["185m", "10m"],
            ["instrument-1", "instrument-2"],
            ["1H", "2H"],
        )

    def fake_make_basis_functions(**kwargs: object) -> BasisFunctions:
        nonlocal captured_basis_sites
        fp_all = kwargs["fp_all"]
        assert isinstance(fp_all, dict)
        captured_basis_sites = [key for key in fp_all if not key.startswith(".")]
        return _fake_basis_functions()

    def fake_make_inv_inputs(fp_data: dict, sites: list[str], **kwargs: object) -> xr.Dataset:
        """Return minimal inputs after an empty site is removed."""
        assert sites == ["TAC"]
        return _minimal_prepared_inv_inputs()

    monkeypatch.setattr(
        prep_module,
        "data_processing_surface_notracer",
        fake_data_processing_surface_notracer,
    )
    monkeypatch.setattr(prep_module, "make_basis_functions", fake_make_basis_functions)
    monkeypatch.setattr(prep_module, "make_inv_inputs", fake_make_inv_inputs)

    prepared = prepare_rhime_inputs(
        species="ch4",
        sites=["TAC", "MHD"],
        domain="EUROPE",
        averaging_period=["1H", "2H"],
        start_date="2019-01-01",
        end_date="2019-02-01",
        output_name="filter_drop",
        flux_sources=["total-ukghg-edgar7"],
        use_bc=False,
    )

    assert prepared.sites == ("TAC",)
    assert prepared.averaging_period == ("1H",)
    assert captured_basis_sites == ["TAC"]


def test_prepare_rhime_inputs_rejects_all_sites_dropped_before_basis_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filtering should fail before basis construction when every site is empty."""
    site_data = {"TAC": _site_dataset([]), "MHD": _site_dataset([])}

    def fake_data_processing_surface_notracer(
        **kwargs: object,
    ) -> tuple[dict, list[str], list[str], list[str], list[str], list[str]]:
        return (
            {**site_data, ".species": "CH4"},
            ["TAC", "MHD"],
            ["185m", "10m"],
            ["185m", "10m"],
            ["instrument-1", "instrument-2"],
            ["1H", "2H"],
        )

    def fake_make_basis_functions(**kwargs: object) -> BasisFunctions:
        raise AssertionError("Basis generation should not run when all sites are dropped.")

    monkeypatch.setattr(
        prep_module,
        "data_processing_surface_notracer",
        fake_data_processing_surface_notracer,
    )
    monkeypatch.setattr(prep_module, "make_basis_functions", fake_make_basis_functions)

    with pytest.raises(ValueError, match="No sites remain"):
        prepare_rhime_inputs(
            species="ch4",
            sites=["TAC", "MHD"],
            domain="EUROPE",
            averaging_period=["1H", "2H"],
            start_date="2019-01-01",
            end_date="2019-02-01",
            output_name="filter_drop",
            flux_sources=["total-ukghg-edgar7"],
            use_bc=False,
        )


def test_params_from_config_rejects_unsupported_deprecated_option(tmp_path: Path) -> None:
    config_file = tmp_path / "rhime.ini"
    config_file.write_text(
        """
[INPUT.MEASUREMENTS]
species = "ch4"
sites = ["TAC"]
averaging_period = ["1h"]
start_date = "2019-01-01"
end_date = "2019-01-02"

[INPUT.PRIORS]
domain = "EUROPE"
flux_sources = ["total-ukghg-edgar7"]

[RHIME.OUTPUT]
output_path = "out"
output_name = "test"

[RHIME.DATA]
calculate_min_error = true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="calculate_min_error"):
        params_from_config(config_file)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("reparameterise_log_normal", "true"),
        ("mcmc_type", '"hmc"'),
    ],
)
def test_params_from_config_rejects_unsupported_legacy_runner_options(
    tmp_path: Path, name: str, value: str
) -> None:
    config_file = tmp_path / "rhime.ini"
    config_file.write_text(
        f"""
[INPUT.MEASUREMENTS]
species = "ch4"
sites = ["TAC"]
averaging_period = ["1h"]
start_date = "2019-01-01"
end_date = "2019-01-02"

[INPUT.PRIORS]
domain = "EUROPE"
flux_sources = ["total-ukghg-edgar7"]

[RHIME.OUTPUT]
output_path = "out"
output_name = "test"

[RHIME.MCMC]
{name} = {value}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=name):
        params_from_config(config_file)


def test_run_rhime_rejects_unknown_parameter_before_data_preparation(tmp_path: Path) -> None:
    args = {
        "species": "ch4",
        "sites": ["TAC"],
        "averaging_period": ["1h"],
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "flux_sources": ["total-ukghg-edgar7"],
        "output_path": str(tmp_path),
        "output_name": "test",
        "definitely_not_a_rhime_parameter": True,
    }

    with pytest.raises(ValueError, match="Unsupported RHIME parameter"):
        run_rhime(**args)


def test_run_rhime_rejects_unsupported_output_format(tmp_path: Path) -> None:
    args = {
        "species": "ch4",
        "sites": ["TAC"],
        "averaging_period": ["1h"],
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "flux_sources": ["total-ukghg-edgar7"],
        "output_path": str(tmp_path),
        "output_name": "test",
        "output_format": "definitely-not-supported",
    }

    with pytest.raises(ValueError, match="Unsupported RHIME output_format"):
        run_rhime(**args)


def test_run_rhime_can_validate_output_format_without_output_path() -> None:
    args = {
        "species": "ch4",
        "sites": ["TAC"],
        "averaging_period": ["1h"],
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "flux_sources": ["total-ukghg-edgar7"],
        "output_name": "test",
        "output_format": "definitely-not-supported",
    }

    with pytest.raises(ValueError, match="Unsupported RHIME output_format"):
        run_rhime(**args)


def test_required_parameter_validation_allows_missing_output_path_for_in_memory_runs() -> None:
    args = {
        "species": "ch4",
        "sites": ["TAC"],
        "averaging_period": ["1h"],
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "output_name": "test",
    }

    rhime_params.validate_required_params(args)


def test_rhime_runner_setup_forwards_satellite_platform_to_preparation() -> None:
    """Satellite runs use the public ``platform`` preparation parameter."""
    setup = rhime_params.make_rhime_runner_setup(
        params={
            "species": "co2",
            "sites": ["OCO2-EASTASIA"],
            "averaging_period": ["1h"],
            "platform": ["satellite"],
            "domain": "EASTASIA",
            "start_date": "2019-01-01",
            "end_date": "2019-01-02",
            "flux_sources": ["test-source"],
            "output_name": "satellite-test",
            "output_format": "none",
        },
        multisector=False,
        data_param_names=set(inspect.signature(prepare_rhime_inputs).parameters),
    )

    assert setup.data_args["platform"] == ["satellite"]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("species", ""),
        ("sites", []),
        ("domain", "  "),
    ],
)
def test_required_parameter_validation_rejects_empty_values(name: str, value) -> None:
    args = {
        "species": "ch4",
        "sites": ["TAC"],
        "averaging_period": ["1h"],
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "output_name": "test",
    }
    args[name] = value

    with pytest.raises(ValueError, match=name):
        rhime_params.validate_required_params(args)


def test_output_path_validation_allows_output_none_without_path() -> None:
    rhime_specs.validate_output_path_settings(
        output_format="none",
        output_path=None,
        save_trace=False,
        save_inversion_output=True,
        multisector=False,
    )


def test_output_path_validation_rejects_default_standard_save_without_path() -> None:
    with pytest.raises(ValueError, match="output_path"):
        rhime_specs.validate_output_path_settings(
            output_format="inv_out",
            output_path=None,
            save_trace=False,
            save_inversion_output=True,
            multisector=False,
        )


def test_runner_setup_defaults_inv_out_save_only_for_inv_out_format(tmp_path: Path) -> None:
    """Derived RHIME products should not save large inv_out sidecars unless requested."""
    base_params = {
        "species": "ch4",
        "sites": ["TAC"],
        "averaging_period": ["1h"],
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "flux_sources": ["total-ukghg-edgar7"],
        "output_name": "test",
    }
    data_params = set(inspect.signature(prepare_rhime_inputs).parameters)

    inv_out_setup = rhime_params.make_rhime_runner_setup(
        params={**base_params, "output_format": "inv_out", "output_path": str(tmp_path)},
        multisector=False,
        data_param_names=data_params,
    )
    paris_setup = rhime_params.make_rhime_runner_setup(
        params={**base_params, "output_format": "paris"},
        multisector=False,
        data_param_names=data_params,
    )

    assert inv_out_setup.run_spec.output.save_inversion_output is True
    assert paris_setup.run_spec.output.save_inversion_output is False


def test_output_path_validation_rejects_multisector_legacy_output() -> None:
    with pytest.raises(ValueError, match="single-sector"):
        rhime_specs.validate_output_path_settings(
            output_format="legacy",
            output_path="outputs",
            save_trace=False,
            save_inversion_output=False,
            multisector=True,
        )


def test_make_output_spec_validates_trace_save_path() -> None:
    with pytest.raises(ValueError, match="save_trace"):
        rhime_specs.make_output_spec(
            output_format="inv_out",
            output_path=None,
            output_name="test",
            save_trace=True,
            save_inversion_output=False,
            country_file=None,
            paris_postprocessing_kwargs=None,
            output_filename_convention="rhime",
            multisector=False,
        )


def test_make_output_spec_normalizes_filename_convention_case() -> None:
    spec = rhime_specs.make_output_spec(
        output_format="Legacy",
        output_path="outputs",
        output_name="test",
        save_trace=False,
        save_inversion_output=False,
        country_file=None,
        paris_postprocessing_kwargs=None,
        output_filename_convention="Legacy",
        multisector=False,
    )

    assert spec.output_format == "legacy"
    assert spec.output_filename_convention == "legacy"


def test_derived_output_filename_can_use_legacy_convention(tmp_path: Path) -> None:
    spec = RhimeOutputSpec(
        output_format="legacy",
        output_path=str(tmp_path),
        output_name="legacy_test",
        save_inversion_output=False,
        output_filename_convention="legacy",
    )

    filename = rhime_outputs._define_derived_output_filename(
        spec,
        species="ch4",
        domain="EUROPE",
        output_name=spec.output_name,
        start_date="2019-01-01",
    )

    assert filename == tmp_path / "CH4_EUROPE_legacy_test_2019-01-01.nc"


def test_make_standard_output_bundle_returns_outputs_without_mutating_result() -> None:
    model_spec, output_spec, run_spec = _minimal_output_specs()
    inv_inputs = _minimal_output_inv_inputs().assign_coords(
        release_lat=("nmeasure", [51.0]),
        release_lon=("nmeasure", [-2.0]),
    )
    prepared = RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=_fake_basis_functions(),
        site_metadata=_prepared_site_metadata(),
    )

    bundle = rhime_outputs.make_standard_output_bundle(
        output_spec=output_spec,
        run_spec=run_spec,
        model_spec=model_spec,
        idata=_minimal_output_idata(),
        prepared=prepared,
        country_file=None,
    )

    assert isinstance(bundle.inv_out, InversionOutput)
    xr.testing.assert_identical(bundle.inv_out.inv_inputs.release_lat, inv_inputs.release_lat)
    xr.testing.assert_identical(bundle.inv_out.inv_inputs.release_lon, inv_inputs.release_lon)
    assert "site_lats" not in bundle.inv_out.run_metadata
    assert "site_lons" not in bundle.inv_out.run_metadata
    assert bundle.outputs == {"inversion_output": bundle.inv_out}
    assert bundle.output_metadata == {"inversion_output_contract": "modern"}


def test_output_bundle_serializes_state_activity_spec() -> None:
    """Concrete per-sector policy dictionaries remain valid output metadata."""
    _, output_spec, run_spec = _minimal_output_specs()
    model_spec = RhimeModelSpec(
        species="ch4",
        domain="EUROPE",
        sectors=run_spec.model.sectors,
        sector_state_activities={"FF": StateActivity(active=False, fixed_value=2.0)},
    )
    prepared = RhimePreparedInputs(
        inv_inputs=_minimal_output_inv_inputs(),
        basis_functions=_fake_basis_functions(),
        site_metadata=_prepared_site_metadata(),
    )

    bundle = rhime_outputs.make_standard_output_bundle(
        output_spec=output_spec,
        run_spec=run_spec,
        model_spec=model_spec,
        idata=_minimal_output_idata(),
        prepared=prepared,
        country_file=None,
    )

    assert bundle.inv_out is not None
    policy = bundle.inv_out.model_metadata["sector_state_activities"]["FF"]
    assert policy["active"] is False
    assert policy["fixed_value"] == 2.0


def test_make_multisector_output_bundle_returns_modern_inv_out() -> None:
    """Multi-sector RHIME outputs retain the same modern inversion-output contract."""
    sectors = (
        SectorSpec(
            name="FF",
            flux_source="ff-inventory",
            x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
            variable_suffix="ff",
        ),
        SectorSpec(
            name="Ocean",
            flux_source="ocean-inventory",
            x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
            variable_suffix="ocean",
        ),
    )
    model_spec = RhimeModelSpec(species="ch4", domain="EUROPE", sectors=sectors)
    output_spec = RhimeOutputSpec(output_format="inv_out", output_name="test", save_inversion_output=False)
    run_spec = RhimeRunSpec(
        "2019-01-01",
        "2019-01-02",
        ("TAC",),
        ("1h",),
        model_spec,
        output_spec,
        split_by_sectors=True,
    )
    prepared = RhimePreparedInputs(
        inv_inputs=_minimal_output_inv_inputs(),
        basis_functions=_fake_basis_functions(),
        site_metadata=_prepared_site_metadata(),
    )
    idata = az.from_dict(
        posterior={
            "x_ff": np.ones((1, 1, 1)),
            "x_ocean": np.ones((1, 1, 1)),
        },
        coords={"region": [0]},
        dims={"x_ff": ["region"], "x_ocean": ["region"]},
    )

    bundle = rhime_outputs.make_multisector_output_bundle(
        output_spec=output_spec,
        run_spec=run_spec,
        model_spec=model_spec,
        idata=idata,
        prepared=prepared,
        country_file=None,
    )

    assert isinstance(bundle.inv_out, InversionOutput)
    assert bundle.inv_out.run_metadata["split_by_sectors"] is True
    assert bundle.output_metadata["inversion_output_contract"] == "modern"
    assert "sector_flux_diagnostics" in bundle.outputs
    assert "flux_ff_posterior_mean" in bundle.outputs["sector_flux_diagnostics"]
    assert "flux_ocean_posterior_mean" in bundle.outputs["sector_flux_diagnostics"]
    assert "flux_total_posterior_mean" in bundle.outputs["sector_flux_diagnostics"]
    assert bundle.outputs["inversion_output"] is bundle.inv_out


def test_make_multisector_output_bundle_builds_latest_paris_flux(
    europe_country_file: Path,
    fake_multisector_basis_functions_matching_country_grid: Callable[..., BasisFunctions],
    multisector_postprocessing_inv_out: Callable[..., InversionOutput],
) -> None:
    """Multi-sector PARIS output builds explicit latest total and sector flux output."""
    sectors = (
        SectorSpec(
            name="FF",
            flux_source="ff-inventory",
            x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
            variable_suffix="ff",
        ),
        SectorSpec(
            name="Ocean",
            flux_source="ocean-inventory",
            x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
            variable_suffix="ocean",
        ),
    )
    model_spec = RhimeModelSpec(species="ch4", domain="EUROPE", sectors=sectors)
    output_spec = RhimeOutputSpec(
        output_format="paris",
        output_name="test",
        save_inversion_output=False,
        paris_postprocessing_kwargs={
            "template_version": "latest",
            "inversion_grid": False,
            "flux_frequency": "yearly",
            "country_selections": list(PARIS_LATEST_COUNTRIES),
        },
    )
    run_spec = RhimeRunSpec(
        "2019-01-01",
        "2019-01-02",
        ("TAC",),
        ("1h",),
        model_spec,
        output_spec,
        split_by_sectors=True,
    )
    basis_functions = fake_multisector_basis_functions_matching_country_grid(europe_country_file)
    basis_functions.flux.attrs["time_period"] = "not-a-parseable-period"
    inv_inputs = _minimal_output_inv_inputs()
    assert basis_functions.source_labels is not None
    inv_inputs["H"] = inv_inputs["H"].expand_dims(source=list(basis_functions.source_labels))
    prepared = RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=basis_functions,
        site_metadata=_prepared_site_metadata(),
    )
    idata = cast(Any, multisector_postprocessing_inv_out().trace)

    bundle = rhime_outputs.make_multisector_output_bundle(
        output_spec=output_spec,
        run_spec=run_spec,
        model_spec=model_spec,
        idata=idata,
        prepared=prepared,
        country_file=str(europe_country_file),
    )

    assert "paris_flux" in bundle.outputs
    assert "paris_concentration" not in bundle.outputs
    assert "not implemented yet" in bundle.output_metadata["paris_note"]
    paris_flux = bundle.outputs["paris_flux"]
    assert "flux_total_posterior" in paris_flux
    assert "flux_ff_posterior" in paris_flux
    assert "flux_ocean_posterior" in paris_flux
    assert tuple(paris_flux.sector.values) == ("ff", "ocean")


def test_modern_inversion_output_save_load_roundtrip(tmp_path: Path) -> None:
    """Modern output preserves January annual flux metadata for a June run."""
    model_spec, output_spec, _ = _minimal_output_specs()
    sector = model_spec.sectors[0]
    flux_activity = StateActivity(
        active=xr.DataArray([False], dims="region", coords={"region": ["r0"]}),
        fixed_value=np.array([1.25]),
    )
    model_spec = RhimeModelSpec(
        species=model_spec.species,
        domain=model_spec.domain,
        sectors=(
            SectorSpec(
                name=sector.name,
                flux_source=sector.flux_source,
                x_prior=sector.x_prior,
                variable_suffix=sector.variable_suffix,
                state_activity=flux_activity,
            ),
        ),
        bc_state_activity=StateActivity(active=False, fixed_value=np.array(0.75)),
        builder_strategy="compiled",
    )
    run_spec = RhimeRunSpec(
        "2019-06-01",
        "2019-07-01",
        ("TAC",),
        ("1h",),
        model_spec,
        output_spec,
    )
    basis_artifact_path = str(tmp_path / "unit-basis.nc")
    basis_functions = _fake_basis_functions(artifact_source="unit-test")
    annual_flux = basis_functions.flux.expand_dims(flux_time=pd.to_datetime(["2019-01-01"]))
    annual_flux.attrs["time_period"] = "1 year"
    prepared = RhimePreparedInputs(
        inv_inputs=_minimal_output_inv_inputs(),
        basis_functions=basis_functions.with_flux(annual_flux).with_metadata(
            {BASIS_ARTIFACT_PATH_ATTR: basis_artifact_path}
        ),
        site_metadata=_prepared_site_metadata(),
    )
    idata = _minimal_output_idata()
    idata.attrs["burn"] = 1000
    cast(Any, idata).posterior.attrs["burn"] = 1000
    bundle = rhime_outputs.make_standard_output_bundle(
        output_spec=output_spec,
        run_spec=run_spec,
        model_spec=model_spec,
        idata=idata,
        prepared=prepared,
        country_file=None,
    )
    assert bundle.inv_out is not None

    output_file = tmp_path / "modern_inv_out.nc"
    bundle.inv_out.save(output_file)
    reloaded = InversionOutput.load(output_file)

    assert reloaded.species == "ch4"
    assert reloaded.domain == "EUROPE"
    assert reloaded.start_date == "2019-06-01"
    assert reloaded.end_date == "2019-07-01"
    assert reloaded.run_metadata["basis_artifact_source"] == "unit-test"
    assert reloaded.run_metadata["basis_artifact_path"] == basis_artifact_path
    assert reloaded.basis_functions.basis_artifact_path == basis_artifact_path
    assert reloaded.provenance["basis_representation"] == "operator-backed"
    assert reloaded.output_metadata["output_format"] == "inv_out"
    assert reloaded.model_metadata["builder_strategy"] == "compiled"
    saved_activity = reloaded.model_metadata["sectors"][0]["state_activity"]
    assert saved_activity["active"] == {
        "dims": ["region"],
        "coords": {"region": ["r0"]},
        "values": [False],
    }
    assert saved_activity["fixed_value"] == [1.25]
    assert reloaded.model_metadata["bc_state_activity"]["active"] is False
    assert reloaded.model_metadata["bc_state_activity"]["fixed_value"] == 0.75
    assert reloaded.trace.attrs["burn"] == 1000
    assert cast(Any, reloaded.trace).posterior.attrs["burn"] == 1000
    xr.testing.assert_identical(reloaded.inv_inputs, prepared.inv_inputs)
    xr.testing.assert_identical(reloaded.basis_functions.flux, prepared.basis_functions.flux)
    np.testing.assert_array_equal(
        reloaded.flux["flux_time"].values,
        np.array(["2019-01-01"], dtype="datetime64[ns]"),
    )
    assert reloaded.flux.attrs["time_period"] == "1 year"
    xr.testing.assert_equal(
        reloaded.basis_functions.operator.basis_matrix,
        prepared.basis_functions.operator.basis_matrix,
    )


def test_modern_inversion_output_restores_bytes_multiindex_metadata() -> None:
    """Modern output loading accepts bytes-encoded MultiIndex metadata."""
    model_spec, output_spec, run_spec = _minimal_output_specs()
    inv_inputs = _minimal_output_inv_inputs()
    prepared = RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=_fake_basis_functions(artifact_source="unit-test"),
        site_metadata=_prepared_site_metadata(),
    )
    bundle = rhime_outputs.make_standard_output_bundle(
        output_spec=output_spec,
        run_spec=run_spec,
        model_spec=model_spec,
        idata=_minimal_output_idata(),
        prepared=prepared,
        country_file=None,
    )
    assert bundle.inv_out is not None

    dt = bundle.inv_out.to_datatree()
    inv_inputs_dt = dt["inv_inputs"]
    inv_inputs_dt.attrs[inversion_output_module.MULTIINDEX_DIMS_ATTR] = inv_inputs_dt.attrs[
        inversion_output_module.MULTIINDEX_DIMS_ATTR
    ].encode()

    reloaded = InversionOutput.from_datatree(dt)

    assert isinstance(reloaded.inv_inputs.indexes["nmeasure"], pd.MultiIndex)
    assert reloaded.inv_inputs.indexes["nmeasure"].names == ["site", "time"]
    xr.testing.assert_identical(reloaded.inv_inputs, prepared.inv_inputs)


def test_modern_inversion_output_roundtrips_trace_multiindex() -> None:
    """Modern output serialization preserves restored trace measurement coordinates."""
    model_spec, output_spec, run_spec = _minimal_output_specs()
    nmeasure_index = pd.MultiIndex.from_arrays(
        [["TAC"], pd.to_datetime(["2019-01-01"])],
        names=["site", "time"],
    )
    posterior = xr.Dataset(
        {"x": (("chain", "draw", "region"), np.ones((1, 1, 1)))},
        coords={"chain": [0], "draw": [0], "region": [0]},
    )
    posterior_predictive = xr.Dataset(
        {"y": (("chain", "draw", "nmeasure"), np.ones((1, 1, 1)))},
        coords={
            "chain": [0],
            "draw": [0],
            **xr.Coordinates.from_pandas_multiindex(nmeasure_index, "nmeasure"),
        },
    )
    prepared = RhimePreparedInputs(
        inv_inputs=_minimal_output_inv_inputs(),
        basis_functions=_fake_basis_functions(artifact_source="unit-test"),
        site_metadata=_prepared_site_metadata(),
    )
    bundle = rhime_outputs.make_standard_output_bundle(
        output_spec=output_spec,
        run_spec=run_spec,
        model_spec=model_spec,
        idata=az.InferenceData(posterior=posterior, posterior_predictive=posterior_predictive),
        prepared=prepared,
        country_file=None,
    )
    assert bundle.inv_out is not None

    reloaded = InversionOutput.from_datatree(bundle.inv_out.to_datatree())

    reloaded_posterior_predictive = cast(Any, reloaded.trace).posterior_predictive
    index = reloaded_posterior_predictive.indexes["nmeasure"]
    assert isinstance(index, pd.MultiIndex)
    assert index.names == ["site", "time"]
    xr.testing.assert_identical(reloaded_posterior_predictive["y"], posterior_predictive["y"])


@pytest.mark.parametrize(
    "raw_multiindex_dims",
    [
        b"not-json",
        '{"dims": "not-a-list"}',
        '{"dims": [{"dim": "nmeasure", "levels": "site"}]}',
        '{"dims": [{"dim": "nmeasure", "levels": ["missing"]}]}',
    ],
)
def test_modern_inversion_output_ignores_malformed_multiindex_metadata(raw_multiindex_dims: object) -> None:
    """Malformed MultiIndex metadata should not break modern output loading."""
    model_spec, output_spec, run_spec = _minimal_output_specs()
    inv_inputs = _minimal_output_inv_inputs()
    prepared = RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=_fake_basis_functions(artifact_source="unit-test"),
        site_metadata=_prepared_site_metadata(),
    )
    bundle = rhime_outputs.make_standard_output_bundle(
        output_spec=output_spec,
        run_spec=run_spec,
        model_spec=model_spec,
        idata=_minimal_output_idata(),
        prepared=prepared,
        country_file=None,
    )
    assert bundle.inv_out is not None

    dt = bundle.inv_out.to_datatree()
    dt["inv_inputs"].attrs[inversion_output_module.MULTIINDEX_DIMS_ATTR] = raw_multiindex_dims

    reloaded = InversionOutput.from_datatree(dt)

    assert inversion_output_module.MULTIINDEX_DIMS_ATTR not in reloaded.inv_inputs.attrs
    assert "site" in reloaded.inv_inputs
    assert "time" in reloaded.inv_inputs
    assert not isinstance(reloaded.inv_inputs.indexes.get("nmeasure"), pd.MultiIndex)


def test_modern_inversion_output_supports_flux_outputs() -> None:
    """Modern InversionOutput feeds flux postprocessing directly."""
    from openghg_inversions.postprocessing.make_outputs import make_flux_outputs

    model_spec, output_spec, run_spec = _minimal_output_specs()
    prepared = RhimePreparedInputs(
        inv_inputs=_minimal_output_inv_inputs(),
        basis_functions=_fake_basis_functions(),
        site_metadata=_prepared_site_metadata(),
    )
    bundle = rhime_outputs.make_standard_output_bundle(
        output_spec=output_spec,
        run_spec=run_spec,
        model_spec=model_spec,
        idata=az.from_dict(
            posterior={"x": np.ones((1, 2, 1))},
            prior={"x": np.ones((1, 2, 1))},
            coords={"region": [0]},
            dims={"x": ["region"]},
        ),
        prepared=prepared,
        country_file=None,
    )
    assert bundle.inv_out is not None

    modern_flux = make_flux_outputs(
        bundle.inv_out,
        include_scale_factors=False,
        report_flux_on_inversion_grid=False,
    )

    assert "flux_posterior_mean" in modern_flux


def test_modern_flux_outputs_use_retained_basis_operator(
    monkeypatch: pytest.MonkeyPatch,
    europe_country_file: Path,
) -> None:
    """Modern flux products use retained basis operators instead of the flat view basis."""
    from openghg_inversions.postprocessing.make_outputs import make_flux_outputs

    basis_functions, operator = _recording_basis_functions_matching_country_grid(europe_country_file)
    inv_out = _modern_postprocessing_inv_out(europe_country_file, basis_functions=basis_functions)

    def fail_flat_basis(self: BasisFunctions) -> xr.DataArray:
        raise AssertionError("modern flux outputs should not materialise the flat basis view")

    monkeypatch.setattr(type(basis_functions), "flat_basis", fail_flat_basis)

    flux_outputs = make_flux_outputs(
        inv_out,
        include_scale_factors=True,
        report_flux_on_inversion_grid=False,
    )

    assert "flux_posterior_mean" in flux_outputs
    assert "scaling_posterior_mean" in flux_outputs
    assert operator.interpolate_calls


def test_modern_outputs_record_basis_reconstruction_metadata(europe_country_file: Path) -> None:
    """Modern derived outputs record stable basis reconstruction metadata."""
    from openghg_inversions.postprocessing.make_outputs import make_country_outputs, make_flux_outputs

    basis_artifact_path = "/tmp/example-basis.nc"
    basis_functions = _fake_basis_functions_matching_country_grid(europe_country_file).with_metadata(
        {
            BASIS_ARTIFACT_SOURCE_ATTR: "datatree",
            "basis_artifact_path": basis_artifact_path,
        }
    )
    inv_out = _modern_postprocessing_inv_out(europe_country_file, basis_functions=basis_functions)

    flux_outputs = make_flux_outputs(inv_out, include_scale_factors=False)
    country_outputs = make_country_outputs(inv_out, country_file=europe_country_file)

    for outputs in (flux_outputs, country_outputs):
        assert outputs.attrs[BASIS_RECONSTRUCTION_PATH_ATTR] == BASIS_RECONSTRUCTION_OPERATOR_BACKED
        assert outputs.attrs[BASIS_ARTIFACT_SOURCE_OUTPUT_ATTR] == BASIS_ARTIFACT_SOURCE_LOADED_DATATREE
        assert outputs.attrs[BASIS_ARTIFACT_PATH_OUTPUT_ATTR] == basis_artifact_path


def test_modern_operator_flux_and_country_outputs_run_on_nonuniform_basis(
    europe_country_file: Path,
) -> None:
    """Operator-backed modern products run on a non-uniform fixture."""
    from openghg_inversions.postprocessing.make_outputs import make_country_outputs, make_flux_outputs

    inv_out = _modern_postprocessing_inv_out(
        europe_country_file,
        basis_functions=_two_region_basis_functions_matching_country_grid(europe_country_file),
    )

    modern_flux = make_flux_outputs(inv_out, include_scale_factors=True)
    modern_country = make_country_outputs(inv_out, country_file=europe_country_file, country_regions="paris")

    assert "flux_posterior_mean" in modern_flux
    assert "country_posterior_mean" in modern_country


@pytest.mark.parametrize("report_flux_on_inversion_grid", [False, True])
def test_modern_flux_outputs_backfill_unsanitized_nonfinite_flux(
    europe_country_file: Path,
    report_flux_on_inversion_grid: bool,
) -> None:
    """Flux postprocessing zero-fills old retained flux artifacts with non-finite values."""
    from openghg_inversions.postprocessing.make_outputs import make_flux_outputs

    basis_functions = _unsanitized_nonfinite_basis_functions_matching_country_grid(europe_country_file)
    inv_out = _modern_postprocessing_inv_out(europe_country_file, basis_functions=basis_functions)

    with pytest.warns(NonFiniteFluxWarning, match="applying a lazy zero-fill guard"):
        outputs = make_flux_outputs(
            inv_out,
            include_scale_factors=False,
            report_flux_on_inversion_grid=report_flux_on_inversion_grid,
        )

    assert "flux_posterior_mean" in outputs
    assert np.isfinite(outputs["flux_posterior_mean"].values).all()
    assert _flux_nonfinite_metadata(outputs).policy == NONFINITE_POLICY_ZERO_FILL


def test_modern_paris_flux_outputs_use_retained_basis_operator(
    monkeypatch: pytest.MonkeyPatch,
    europe_country_file: Path,
) -> None:
    """PARIS flux outputs use retained basis operators for modern flux and country products."""
    from openghg_inversions.postprocessing.make_paris_outputs import paris_flux_output

    basis_functions, operator = _recording_basis_functions_matching_country_grid(europe_country_file)
    inv_out = _modern_postprocessing_inv_out(europe_country_file, basis_functions=basis_functions)

    def fail_flat_basis(self: BasisFunctions) -> xr.DataArray:
        raise AssertionError("modern PARIS flux outputs should not materialise the flat basis view")

    monkeypatch.setattr(type(basis_functions), "flat_basis", fail_flat_basis)

    flux_outputs = paris_flux_output(
        inv_out,
        country_file=europe_country_file,
    )

    assert "flux_total_posterior" in flux_outputs
    assert "country_flux_total_posterior" in flux_outputs
    assert "flux_total_posterior_inversion_grid" in flux_outputs
    assert flux_outputs.attrs[BASIS_RECONSTRUCTION_PATH_ATTR] == BASIS_RECONSTRUCTION_OPERATOR_BACKED
    for name in (
        "flux_total_posterior",
        "country_flux_total_posterior",
        "flux_total_posterior_inversion_grid",
    ):
        assert flux_outputs[name].dtype == np.dtype("float32")
    assert operator.interpolate_calls
    assert operator.basis_matrix_accesses


def test_modern_paris_flux_outputs_backfill_unsanitized_nonfinite_flux(europe_country_file: Path) -> None:
    """PARIS flux output completes when old retained flux artifacts contain non-finite values."""
    from openghg_inversions.postprocessing.make_paris_outputs import paris_flux_output

    basis_functions = _unsanitized_nonfinite_basis_functions_matching_country_grid(europe_country_file)
    inv_out = _modern_postprocessing_inv_out(europe_country_file, basis_functions=basis_functions)

    with pytest.warns(NonFiniteFluxWarning, match="applying a lazy zero-fill guard"):
        flux_outputs = paris_flux_output(inv_out, country_file=europe_country_file)

    assert "flux_total_posterior" in flux_outputs
    assert np.isfinite(flux_outputs["flux_total_posterior"].values).all()
    assert _flux_nonfinite_metadata(flux_outputs).policy == NONFINITE_POLICY_ZERO_FILL


def test_observation_inputs_for_outputs_stay_dataset_based() -> None:
    """Modern postprocessing avoids split legacy observation fields."""
    model_spec, output_spec, run_spec = _minimal_output_specs()
    inv_inputs = _minimal_output_inv_inputs()
    inv_inputs["mf_prior_factor"] = ("nmeasure", [0.2])
    inv_inputs["mf_prior_upper_level_factor"] = ("nmeasure", [0.3])
    prepared = RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=_fake_basis_functions(),
        site_metadata=_prepared_site_metadata(),
    )
    bundle = rhime_outputs.make_standard_output_bundle(
        output_spec=output_spec,
        run_spec=run_spec,
        model_spec=model_spec,
        idata=_minimal_output_idata(),
        prepared=prepared,
        country_file=None,
    )
    assert bundle.inv_out is not None

    obs_inputs = observation_inputs_for_outputs(bundle.inv_out)

    assert not hasattr(bundle.inv_out, "obs")
    assert not hasattr(bundle.inv_out, "obs_err")
    assert set(obs_inputs.data_vars) == {
        "y_obs",
        "y_obs_error",
        "y_obs_prior_factor",
        "y_obs_prior_upper_level_factor",
        "y_obs_repeatability",
        "y_obs_variability",
    }
    assert isinstance(obs_inputs.indexes["nmeasure"], pd.MultiIndex)
    assert obs_inputs.indexes["nmeasure"].names == ["site", "time"]


def test_standard_postprocessing_rejects_multisector_outputs() -> None:
    """Standard postprocessing does not silently accept multisector modern outputs."""
    from openghg_inversions.postprocessing.make_outputs import basic_output

    inv_out = InversionOutput(
        trace=_minimal_output_idata(),
        inv_inputs=_minimal_output_inv_inputs(),
        basis_functions=_fake_basis_functions(),
        run_metadata={
            "start_date": "2019-01-01",
            "end_date": "2019-01-02",
            "sites": ["TAC"],
            "split_by_sectors": True,
        },
        model_metadata={"species": "ch4", "domain": "EUROPE"},
    )

    with pytest.raises(ValueError, match="single-sector"):
        basic_output(inv_out)


def test_basic_output_processes_modern_output(europe_country_file: Path) -> None:
    """Real basic postprocessing accepts modern output directly."""
    from openghg_inversions.postprocessing.make_outputs import basic_output

    outputs = basic_output(
        _modern_postprocessing_inv_out(europe_country_file), country_file=europe_country_file
    )

    assert "y_obs" in outputs
    assert "model_error" in outputs
    assert "y_posterior_predictive_mean" in outputs
    assert "flux_posterior_mean" in outputs
    assert "country_posterior_mean" in outputs


def test_basic_output_uses_variable_roles_for_renamed_model_variables(
    europe_country_file: Path,
) -> None:
    """Product code selects modern variables by semantic role, not hard-coded names."""
    from openghg_inversions.postprocessing.make_outputs import basic_output

    base = _modern_postprocessing_inv_out(europe_country_file)
    inv_inputs = base.inv_inputs.rename(
        {
            "mf": "mole_fraction",
            "mf_error": "mole_fraction_error",
            "mf_repeatability": "repeatability",
            "mf_variability": "variability",
        }
    )
    trace = _rename_idata_data_vars(
        base.trace,
        {
            "x": "scale_factor",
            "epsilon": "total_mismatch",
            "y": "modelled_mole_fraction",
            "mu_bc": "background",
        },
    )

    inv_out = InversionOutput(
        trace=trace,
        inv_inputs=inv_inputs,
        basis_functions=base.basis_functions,
        run_metadata=base.run_metadata,
        model_metadata={
            **base.model_metadata,
            "variable_roles": {
                "observation": "mole_fraction",
                "observation_error": "mole_fraction_error",
                "observation_repeatability": "repeatability",
                "observation_variability": "variability",
                "flux_scale": "scale_factor",
                "model_error": "total_mismatch",
                "concentration": "modelled_mole_fraction",
                "baseline": "background",
            },
        },
        output_metadata=base.output_metadata,
        provenance=base.provenance,
    )

    outputs = basic_output(inv_out, country_file=europe_country_file)

    assert "y_obs" in outputs
    assert "model_error" in outputs
    assert "flux_posterior_mean" in outputs
    assert "country_posterior_mean" in outputs
    assert "y_posterior_predictive_mean" in outputs


def test_paris_output_processes_modern_output(europe_country_file: Path) -> None:
    """Real PARIS postprocessing accepts modern output directly."""
    from openghg_inversions.postprocessing.make_paris_outputs import make_paris_outputs

    flux_outputs, conc_outputs = make_paris_outputs(
        _modern_postprocessing_inv_out(europe_country_file),
        country_file=europe_country_file,
        obs_avg_period="1h",
        domain="europe",
        inversion_grid=False,
    )

    assert "Yobs" in conc_outputs
    assert "Yapost" in conc_outputs
    assert "flux_total_posterior" in flux_outputs
    assert "country_flux_total_posterior" in flux_outputs
    for name in ("uYtotal", "YapostBC", "Yapost", "YaprioriBC", "Yapriori"):
        assert conc_outputs[name].dtype == np.dtype("float32")
    for name in ("flux_total_posterior", "country_flux_total_posterior"):
        assert flux_outputs[name].dtype == np.dtype("float32")


def test_paris_concentration_without_column_prior_factors_leaves_bc_unchanged(
    europe_country_file: Path,
) -> None:
    """Site-like outputs without column prior factors keep existing concentration values."""
    from openghg_inversions.postprocessing.make_paris_outputs import paris_concentration_outputs

    conc_outputs = paris_concentration_outputs(
        _modern_postprocessing_inv_out(europe_country_file),
        obs_avg_period="1h",
    )

    units = 1e-9
    assert "Yobs_prior_factor" not in conc_outputs
    assert "Yobs_prior_upper_level_factor" not in conc_outputs
    assert float(conc_outputs["Yobs"].squeeze()) == pytest.approx(10.0 * units)
    assert float(conc_outputs["Yapriori"].squeeze()) == pytest.approx(9.0 * units)
    assert float(conc_outputs["Yapost"].squeeze()) == pytest.approx(10.0 * units)
    assert float(conc_outputs["YaprioriBC"].squeeze()) == pytest.approx(0.05 * units)
    assert float(conc_outputs["YapostBC"].squeeze()) == pytest.approx(0.1 * units)


def test_paris_concentration_column_prior_factor_is_added_to_totals_not_bc(europe_country_file: Path) -> None:
    """Column prior correction belongs in totals, not boundary-condition fields."""
    from openghg_inversions.postprocessing.make_paris_outputs import paris_concentration_outputs

    conc_outputs = paris_concentration_outputs(
        _with_column_prior_factors(_modern_postprocessing_inv_out(europe_country_file)),
        obs_avg_period="1h",
    )

    units = 1e-9
    assert float(conc_outputs["Yobs"].squeeze()) == pytest.approx(10.5 * units)
    assert float(conc_outputs["Yapriori"].squeeze()) == pytest.approx(9.5 * units)
    assert float(conc_outputs["Yapost"].squeeze()) == pytest.approx(10.5 * units)
    assert float(conc_outputs["YaprioriBC"].squeeze()) == pytest.approx(0.05 * units)
    assert float(conc_outputs["YapostBC"].squeeze()) == pytest.approx(0.1 * units)
    assert float(conc_outputs["Yobs_prior_factor"].squeeze()) == pytest.approx(0.2 * units)
    assert float(conc_outputs["Yobs_prior_upper_level_factor"].squeeze()) == pytest.approx(0.3 * units)


def test_latest_paris_output_processes_modern_output(europe_country_file: Path, tmp_path: Path) -> None:
    """Explicit latest PARIS output uses the new concentration and flux templates."""
    from openghg_inversions.postprocessing.make_paris_outputs import (
        make_paris_outputs,
    )

    flux_outputs, conc_outputs = make_paris_outputs(
        _modern_postprocessing_inv_out(europe_country_file),
        country_file=europe_country_file,
        obs_avg_period="1h",
        domain="europe",
        inversion_grid=False,
        template_version="latest",
    )

    assert conc_outputs.attrs["paris_concentration_template_version"] == "v04"
    assert "index" in conc_outputs.dims
    assert "platform" in conc_outputs.coords
    assert "time_bnds" in conc_outputs
    assert "number_of_identifier" in conc_outputs
    assert "assimilation_flag" in conc_outputs
    assert "mf_observed" in conc_outputs
    assert "mf_posterior" in conc_outputs
    assert "percentile_mf_posterior" in conc_outputs
    assert "Yobs" not in conc_outputs
    assert conc_outputs["mf_posterior"].dtype == np.dtype("float32")
    assert conc_outputs["mf_bc_prior"].dtype == np.dtype("float32")
    assert conc_outputs["time_bnds"].dtype == np.dtype("float64")
    assert conc_outputs["longitude"].dtype == np.dtype("float64")
    assert conc_outputs["number_of_identifier"].dtype == np.dtype("int16")

    assert flux_outputs.attrs["paris_flux_template_version"] == "v03"
    assert "time_bnds" in flux_outputs
    assert "cell_area" in flux_outputs
    assert "country_fraction" in flux_outputs
    assert "flux_total_posterior" in flux_outputs
    assert "flux_total_posterior_country" in flux_outputs
    assert "stdev_flux_total_posterior" in flux_outputs
    assert "covariance_flux_total_posterior_country" in flux_outputs
    assert tuple(flux_outputs.country.values) == PARIS_LATEST_COUNTRIES
    assert flux_outputs["covariance_flux_total_posterior_country"].shape == (
        flux_outputs.sizes["time"],
        flux_outputs.sizes["country"],
        flux_outputs.sizes["country"],
    )
    np.testing.assert_allclose(
        np.diagonal(
            flux_outputs["covariance_flux_total_posterior_country"].values,
            axis1=1,
            axis2=2,
        ),
        flux_outputs["stdev_flux_total_posterior_country"].values ** 2,
        rtol=1e-6,
    )
    assert "country_flux_total_posterior" not in flux_outputs
    assert flux_outputs["flux_total_posterior"].dtype == np.dtype("float32")
    assert flux_outputs["stdev_flux_total_posterior_country"].dtype == np.dtype("float32")
    assert flux_outputs["covariance_flux_total_posterior_country"].dtype == np.dtype("float32")
    assert flux_outputs["time_bnds"].dtype == np.dtype("float64")

    conc_file = tmp_path / "latest_conc.nc"
    flux_file = tmp_path / "latest_flux.nc"
    conc_outputs.to_netcdf(conc_file)
    flux_outputs.to_netcdf(flux_file)
    with xr.open_dataset(conc_file, decode_times=False) as reloaded_conc:
        assert reloaded_conc["mf_posterior"].dtype == np.dtype("float32")
        assert reloaded_conc["time_bnds"].dtype == np.dtype("float64")
        assert reloaded_conc["longitude"].dtype == np.dtype("float64")
    with xr.open_dataset(flux_file, decode_times=False) as reloaded_flux:
        assert reloaded_flux["flux_total_posterior"].dtype == np.dtype("float32")
        assert reloaded_flux["covariance_flux_total_posterior_country"].dtype == np.dtype("float32")
        assert reloaded_flux["time_bnds"].dtype == np.dtype("float64")


def test_latest_paris_concentration_without_column_prior_factors_leaves_bc_unchanged(
    europe_country_file: Path,
) -> None:
    """Latest site-like outputs without column prior factors keep existing concentration values."""
    from openghg_inversions.postprocessing.make_paris_outputs import paris_concentration_outputs

    conc_outputs = paris_concentration_outputs(
        _modern_postprocessing_inv_out(europe_country_file),
        obs_avg_period="1h",
        template_version="latest",
    )

    units = 1e-9
    assert "y_obs_prior_factor" not in conc_outputs
    assert "y_obs_prior_upper_level_factor" not in conc_outputs
    assert float(conc_outputs["mf_observed"].squeeze()) == pytest.approx(10.0 * units)
    assert float(conc_outputs["mf_prior"].squeeze()) == pytest.approx(9.0 * units)
    assert float(conc_outputs["mf_posterior"].squeeze()) == pytest.approx(10.0 * units)
    assert float(conc_outputs["mf_bc_prior"].squeeze()) == pytest.approx(0.05 * units)
    assert float(conc_outputs["mf_bc_posterior"].squeeze()) == pytest.approx(0.1 * units)


def test_latest_paris_concentration_column_prior_factor_is_added_to_totals_not_bc(
    europe_country_file: Path,
) -> None:
    """Latest PARIS concentration keeps column prior correction out of BC fields."""
    from openghg_inversions.postprocessing.make_paris_outputs import paris_concentration_outputs

    conc_outputs = paris_concentration_outputs(
        _with_column_prior_factors(_modern_postprocessing_inv_out(europe_country_file)),
        obs_avg_period="1h",
        template_version="latest",
    )

    units = 1e-9
    assert float(conc_outputs["mf_observed"].squeeze()) == pytest.approx(10.5 * units)
    assert float(conc_outputs["mf_prior"].squeeze()) == pytest.approx(9.5 * units)
    assert float(conc_outputs["mf_posterior"].squeeze()) == pytest.approx(10.5 * units)
    assert float(conc_outputs["mf_bc_prior"].squeeze()) == pytest.approx(0.05 * units)
    assert float(conc_outputs["mf_bc_posterior"].squeeze()) == pytest.approx(0.1 * units)


def test_latest_paris_concentration_fills_missing_bc_with_nan(europe_country_file: Path) -> None:
    """Latest PARIS concentration keeps mandatory BC fields when no baseline trace exists."""
    from openghg_inversions.postprocessing.make_paris_outputs import paris_concentration_outputs

    base = _modern_postprocessing_inv_out(europe_country_file)
    groups = {}
    for group in base.trace.groups():
        ds = base.trace[group]
        groups[group] = ds.drop_vars([name for name in ("mu_bc", "hbc") if name in ds], errors="ignore")
    trace = cast(Any, az.InferenceData)(**groups)
    inv_out = InversionOutput(
        trace=trace,
        inv_inputs=base.inv_inputs,
        basis_functions=base.basis_functions,
        run_metadata=base.run_metadata,
        model_metadata={**base.model_metadata, "use_bc": False},
        output_metadata=base.output_metadata,
        provenance=base.provenance,
    )

    conc_outputs = paris_concentration_outputs(
        inv_out,
        obs_avg_period="1h",
        template_version="latest",
    )

    for name in ("mf_bc_prior", "mf_bc_posterior"):
        assert name in conc_outputs
        assert conc_outputs[name].dims == ("index",)
        assert conc_outputs[name].dtype == np.dtype("float32")
        assert np.isnan(conc_outputs[name].values).all()


def test_standard_basic_output_uses_modern_postprocessing_without_legacy_adapter(monkeypatch) -> None:
    """RHIME basic postprocessing consumes modern output without legacy adapters."""
    model_spec, output_spec, run_spec = _minimal_output_specs(output_format="basic")
    prepared = RhimePreparedInputs(
        inv_inputs=_minimal_output_inv_inputs(),
        basis_functions=_fake_basis_functions(),
        site_metadata=_prepared_site_metadata(),
    )
    captured: dict[str, Any] = {}

    def fail_inferpymc_postprocessouts(**kwargs: Any) -> None:
        raise AssertionError("run_rhime output helpers must not call inferpymc_postprocessouts")

    def fake_basic_output(inv_out: InversionOutput, country_file: str | None = None) -> xr.Dataset:
        captured["inv_out"] = inv_out
        captured["country_file"] = country_file
        return xr.Dataset({"ok": ((), 1)})

    monkeypatch.setattr(legacy_mcmc, "inferpymc_postprocessouts", fail_inferpymc_postprocessouts)
    monkeypatch.setattr("openghg_inversions.postprocessing.make_outputs.basic_output", fake_basic_output)

    bundle = rhime_outputs.make_standard_output_bundle(
        output_spec=output_spec,
        run_spec=run_spec,
        model_spec=model_spec,
        idata=_minimal_output_idata(),
        prepared=prepared,
        country_file="countries.json",
    )

    assert isinstance(bundle.inv_out, InversionOutput)
    assert captured["inv_out"] is bundle.inv_out
    assert captured["country_file"] == "countries.json"
    assert bundle.output_metadata["inversion_output_contract"] == "modern"
    assert bundle.output_metadata["postprocessing_input_contract"] == "modern_inversion_output"
    assert "basic" in bundle.outputs


def test_standard_paris_output_uses_modern_postprocessing_without_legacy_adapter(monkeypatch) -> None:
    """RHIME PARIS postprocessing consumes modern output without legacy adapters."""
    model_spec, output_spec, run_spec = _minimal_output_specs(output_format="paris")
    prepared = RhimePreparedInputs(
        inv_inputs=_minimal_output_inv_inputs(),
        basis_functions=_fake_basis_functions(),
        site_metadata=_prepared_site_metadata(),
    )
    captured: dict[str, Any] = {}

    def fail_inferpymc_postprocessouts(**kwargs: Any) -> None:
        raise AssertionError("run_rhime output helpers must not call inferpymc_postprocessouts")

    def fake_make_paris_outputs(
        inv_out: InversionOutput,
        country_file: str | None = None,
        domain: str | None = None,
        obs_avg_period: str = "4h",
        **kwargs: Any,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        captured["inv_out"] = inv_out
        captured["country_file"] = country_file
        captured["domain"] = domain
        captured["obs_avg_period"] = obs_avg_period
        return xr.Dataset({"flux": ((), 1)}), xr.Dataset({"conc": ((), 1)})

    monkeypatch.setattr(legacy_mcmc, "inferpymc_postprocessouts", fail_inferpymc_postprocessouts)
    monkeypatch.setattr(
        "openghg_inversions.postprocessing.make_paris_outputs.make_paris_outputs",
        fake_make_paris_outputs,
    )

    bundle = rhime_outputs.make_standard_output_bundle(
        output_spec=output_spec,
        run_spec=run_spec,
        model_spec=model_spec,
        idata=_minimal_output_idata(),
        prepared=prepared,
        country_file="countries.json",
    )

    assert isinstance(bundle.inv_out, InversionOutput)
    assert captured["inv_out"] is bundle.inv_out
    assert captured["country_file"] == "countries.json"
    assert captured["domain"] == "EUROPE"
    assert captured["obs_avg_period"] == "1h"
    assert bundle.output_metadata["inversion_output_contract"] == "modern"
    assert bundle.output_metadata["postprocessing_input_contract"] == "modern_inversion_output"
    assert "paris_flux" in bundle.outputs
    assert "paris_concentration" in bundle.outputs


def test_standard_legacy_output_uses_modern_inversion_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """RHIME legacy output is formatted from modern InversionOutput."""
    model_spec, output_spec, run_spec = _minimal_output_specs(output_format="legacy")
    output_spec = RhimeOutputSpec(
        output_format="legacy",
        output_path=str(tmp_path),
        output_name="legacy_test",
        save_inversion_output=False,
    )
    run_spec = RhimeRunSpec(
        run_spec.start_date,
        run_spec.end_date,
        run_spec.sites,
        run_spec.averaging_period,
        model_spec,
        output_spec,
    )
    prepared = RhimePreparedInputs(
        inv_inputs=_minimal_output_inv_inputs(),
        basis_functions=_fake_basis_functions(),
        site_metadata=_prepared_site_metadata(),
    )
    captured: dict[str, Any] = {}

    def fail_inferpymc_postprocessouts(**kwargs: Any) -> None:
        raise AssertionError("run_rhime legacy output must not call inferpymc_postprocessouts")

    def fake_make_legacy_hbmcmc_output(
        inv_out: InversionOutput,
        country_file: str | None = None,
        use_bc: bool = False,
    ) -> xr.Dataset:
        captured["inv_out"] = inv_out
        captured["country_file"] = country_file
        captured["use_bc"] = use_bc
        return xr.Dataset({"legacy": ((), 1)})

    monkeypatch.setattr(legacy_mcmc, "inferpymc_postprocessouts", fail_inferpymc_postprocessouts)
    monkeypatch.setattr(
        "openghg_inversions.postprocessing.legacy_outputs.make_legacy_hbmcmc_output",
        fake_make_legacy_hbmcmc_output,
    )

    bundle = rhime_outputs.make_standard_output_bundle(
        output_spec=output_spec,
        run_spec=run_spec,
        model_spec=model_spec,
        idata=_minimal_output_idata(),
        prepared=prepared,
        country_file="countries.json",
    )

    assert isinstance(bundle.inv_out, InversionOutput)
    assert captured["inv_out"] is bundle.inv_out
    assert captured["country_file"] == "countries.json"
    assert captured["use_bc"] is True
    assert bundle.output_metadata["postprocessing_input_contract"] == "modern_inversion_output"
    assert "legacy" in bundle.outputs
    expected_path = tmp_path / "legacy_test_ch4_EUROPE_2019-01-01.nc"
    assert expected_path.exists()
    assert bundle.output_metadata["legacy_output_path"] == str(expected_path)


def test_save_inferencedata_prefers_h5netcdf(tmp_path: Path) -> None:
    class FakeInferenceData:
        def __init__(self) -> None:
            self.calls = []

        def to_netcdf(self, path, **kwargs):
            self.calls.append((path, kwargs))

    idata = FakeInferenceData()
    path = tmp_path / "trace.nc"

    rhime_outputs._save_inferencedata(idata, path)  # type: ignore[reportArgumentType]

    assert idata.calls == [(str(path), {"engine": "h5netcdf", "compress": True})]


def test_save_inferencedata_falls_back_after_h5netcdf_failure(tmp_path: Path) -> None:
    class FakeInferenceData:
        def __init__(self) -> None:
            self.calls = []

        def to_netcdf(self, path, **kwargs):
            self.calls.append((path, kwargs))
            if kwargs.get("engine") == "h5netcdf":
                raise ValueError("h5netcdf unavailable")

    idata = FakeInferenceData()
    path = tmp_path / "trace.nc"

    rhime_outputs._save_inferencedata(idata, path)  # type: ignore[reportArgumentType]

    assert idata.calls == [
        (str(path), {"engine": "h5netcdf", "compress": True}),
        (str(path), {"compress": True}),
    ]


def test_save_inferencedata_preserves_burn_attrs_and_resets_multiindex_coords(tmp_path: Path) -> None:
    """Standalone trace saving preserves burn metadata and serializable coordinates."""
    nmeasure_index = pd.MultiIndex.from_arrays(
        [["TAC"], pd.to_datetime(["2019-01-01"])],
        names=["site", "time"],
    )
    posterior_predictive = xr.Dataset(
        {"y": (("chain", "draw", "nmeasure"), np.ones((1, 1, 1)))},
        coords={
            "chain": [0],
            "draw": [0],
            **xr.Coordinates.from_pandas_multiindex(nmeasure_index, "nmeasure"),
        },
    )
    path = tmp_path / "trace.nc"

    idata = az.InferenceData(posterior_predictive=posterior_predictive)
    idata.attrs["burn"] = 1000
    cast(Any, idata).posterior_predictive.attrs["burn"] = 1000

    rhime_outputs._save_inferencedata(idata, path)
    reloaded = az.from_netcdf(path)

    reloaded_posterior_predictive = cast(Any, reloaded).posterior_predictive
    assert reloaded.attrs["burn"] == 1000
    assert reloaded_posterior_predictive.attrs["burn"] == 1000
    assert "site" in reloaded_posterior_predictive.coords
    assert "time" in reloaded_posterior_predictive.coords
    assert not isinstance(reloaded_posterior_predictive.indexes.get("nmeasure"), pd.MultiIndex)


def test_supported_parameter_validation_accepts_sigma_per_site(tmp_path: Path) -> None:
    args = {
        "species": "ch4",
        "sites": ["TAC"],
        "averaging_period": ["1h"],
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "flux_sources": ["total-ukghg-edgar7"],
        "output_path": str(tmp_path),
        "output_name": "test",
        "sigma_per_site": False,
    }

    rhime_params.validate_supported_params(
        args,
        data_params=set(inspect.signature(prepare_rhime_inputs).parameters),
    )


def test_run_rhime_rejects_multiple_flux_sources(tac_ch4_data_args, tmp_path: Path) -> None:
    args = tac_ch4_data_args.copy()
    args.update(
        {
            "flux_sources": ["a", "b"],
            "output_path": str(tmp_path),
            "output_name": "test",
        }
    )
    args.pop("emissions_name")

    with pytest.raises(ValueError, match="exactly one flux source"):
        run_rhime(**args)


def test_run_rhime_multisector_rejects_single_flux_source(tac_ch4_data_args, tmp_path: Path) -> None:
    args = tac_ch4_data_args.copy()
    args.update(
        {
            "flux_sources": ["total-ukghg-edgar7"],
            "output_path": str(tmp_path),
            "output_name": "test",
        }
    )
    args.pop("emissions_name")

    with pytest.raises(ValueError, match="at least two flux sources"):
        run_rhime_multisector(**args)


def test_run_rhime_api_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tac_ch4_data_args: dict[str, Any],
    tmp_path: Path,
    default_bc_basis_directory: Path,
) -> None:
    """Run the single-sector API with real preparation, outputs, and mocked sampling."""
    args = tac_ch4_data_args.copy()
    args.update(
        {
            "flux_sources": args.pop("emissions_name"),
            "output_name": "rhime_test",
            "output_path": str(tmp_path),
            "basis_algorithm": "quadtree",
            "basis_output_path": str(tmp_path),
            "bc_basis_directory": default_bc_basis_directory,
            "nbasis": 4,
            "draws": 1,
            "burn": 0,
            "tune": 0,
            "chains": 1,
            "reload_merged_data": False,
            "x_prior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
            "bc_prior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
            "sigma_prior": {"pdf": "uniform", "lower": 0.1, "upper": 10.0},
            "sample_kwargs": {"random_seed": 123, "compute_convergence_checks": False},
        }
    )

    def fake_sample(self: RhimeSampler, model: pm.Model) -> az.InferenceData:
        """Return deterministic scale and modeled-concentration posteriors."""
        assert self.draws == 1
        return _posterior_only_idata(model, ("x", "mu"))

    monkeypatch.setattr(RhimeSampler, "sample", fake_sample)

    result = run_rhime(**args)

    assert isinstance(result, RhimeResult)
    assert isinstance(result.basis_functions, BasisFunctions)
    assert not hasattr(result, "basis_objects")
    posterior = cast(Any, result.idata).posterior
    assert "x" in posterior
    assert "mu" in posterior
    assert result.run_spec.split_by_sectors is False
    assert "inversion_output" in result.outputs
    assert isinstance(result.inv_out, InversionOutput)
    assert result.output_metadata["inversion_output_contract"] == "modern"
    inv_input_long_names = [
        result.inv_inputs.mf.attrs.get("long_name", ""),
        result.inv_inputs.mf_error.attrs.get("long_name", ""),
        result.inv_inputs.mf_repeatability.attrs.get("long_name", ""),
        result.inv_inputs.mf_variability.attrs.get("long_name", ""),
    ]
    assert all("number_of_observations" not in long_name for long_name in inv_input_long_names)
    output_file = tmp_path / "rhime_test2019-01-01_inversion_output.nc"
    assert output_file.exists()
    reloaded = InversionOutput.load(output_file)
    assert reloaded.species == "ch4"
    assert reloaded.domain == args["domain"]
    assert isinstance(reloaded.basis_functions, BasisFunctions)
    xr.testing.assert_identical(reloaded.inv_inputs, result.inv_inputs)
    obs_long_names = [
        reloaded.inv_inputs.mf.attrs.get("long_name", ""),
        reloaded.inv_inputs.mf_error.attrs.get("long_name", ""),
        reloaded.inv_inputs.mf_repeatability.attrs.get("long_name", ""),
        reloaded.inv_inputs.mf_variability.attrs.get("long_name", ""),
    ]
    assert all("number_of_observations" not in long_name for long_name in obs_long_names)


def test_run_rhime_multisector_api_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tac_ch4_data_args: dict[str, Any],
    tmp_path: Path,
    default_bc_basis_directory: Path,
) -> None:
    """Run the multisector API with real diagnostics and mocked sampling."""
    args = tac_ch4_data_args.copy()
    args.update(
        {
            "flux_sources": ["total-ukghg-edgar7", "total-ukghg-edgar7-shuffled"],
            "sector_sources": {
                "FF": "total-ukghg-edgar7",
                "ocean": "total-ukghg-edgar7-shuffled",
            },
            "output_name": "rhime_multisector_test",
            "output_path": str(tmp_path),
            "basis_algorithm": "quadtree",
            "basis_output_path": str(tmp_path),
            "bc_basis_directory": default_bc_basis_directory,
            "nbasis": 4,
            "draws": 1,
            "burn": 0,
            "tune": 0,
            "chains": 1,
            "reload_merged_data": False,
            "output_format": "none",
            "sector_priors": {
                "FF": {"pdf": "uniform", "lower": 0.8, "upper": 1.0},
                "ocean": {"pdf": "uniform", "lower": 1.1, "upper": 1.3},
            },
            "bc_prior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
            "sigma_prior": {"pdf": "uniform", "lower": 0.1, "upper": 10.0},
            "sample_kwargs": {"random_seed": 123, "compute_convergence_checks": False},
        }
    )
    args.pop("emissions_name")

    def fake_sample(self: RhimeSampler, model: pm.Model) -> az.InferenceData:
        """Return deterministic posterior scale factors for both sectors."""
        assert self.draws == 1
        return _posterior_only_idata(model, ("x_ff", "x_ocean"))

    monkeypatch.setattr(RhimeSampler, "sample", fake_sample)

    result = run_rhime_multisector(**args)

    assert isinstance(result, RhimeResult)
    assert isinstance(result.basis_functions, BasisFunctions)
    assert not hasattr(result, "basis_objects")
    assert result.run_spec.split_by_sectors is True
    assert [sector.name for sector in result.model_spec.sectors] == ["FF", "ocean"]
    assert result.model_spec.sectors[0].x_prior == {
        "pdf": "uniform",
        "lower": 0.8,
        "upper": 1.0,
    }
    assert result.model_spec.sectors[1].x_prior == {
        "pdf": "uniform",
        "lower": 1.1,
        "upper": 1.3,
    }
    posterior = cast(Any, result.idata).posterior
    assert "x_ff" in posterior
    assert "x_ocean" in posterior
    assert "sector_flux_diagnostics" in result.outputs
    assert "flux_ff_posterior_mean" in result.outputs["sector_flux_diagnostics"]
    assert "flux_ocean_posterior_mean" in result.outputs["sector_flux_diagnostics"]
    assert "flux_total_posterior_mean" in result.outputs["sector_flux_diagnostics"]


def test_cli_run_rhime_passes_config_and_overrides(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "rhime.ini"
    config_file.write_text('[RHIME.OUTPUT]\noutput_name = "test"\n', encoding="utf-8")
    seen = {}

    def fake_run_rhime(*, config_file, **kwargs):
        seen["config_file"] = config_file
        seen["kwargs"] = kwargs

    monkeypatch.setattr("openghg_inversions.rhime.run_rhime", fake_run_rhime)

    main(
        [
            "run-rhime",
            "2019-01-01",
            "2019-01-02",
            "-c",
            str(config_file),
            "--output-path",
            str(tmp_path),
            "--kwargs",
            '{"draws": 1}',
        ]
    )

    assert seen["config_file"] == str(config_file)
    assert seen["kwargs"]["start_date"] == "2019-01-01"
    assert seen["kwargs"]["end_date"] == "2019-01-02"
    assert seen["kwargs"]["output_path"] == str(tmp_path)
    assert seen["kwargs"]["draws"] == 1


def test_cli_run_rhime_multisector_passes_config(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "rhime.ini"
    config_file.write_text('[RHIME.OUTPUT]\noutput_name = "test"\n', encoding="utf-8")
    seen = {}

    def fake_run_rhime_multisector(*, config_file, **kwargs):
        seen["config_file"] = config_file
        seen["kwargs"] = kwargs

    monkeypatch.setattr("openghg_inversions.rhime.run_rhime_multisector", fake_run_rhime_multisector)

    main(["run-rhime-multisector", "-c", str(config_file)])

    assert seen["config_file"] == str(config_file)
    assert seen["kwargs"] == {}


def test_safe_pymc_name_sanitizes_source_names() -> None:
    assert safe_pymc_name("total-ukghg-edgar7") == "total_ukghg_edgar7"
    assert safe_pymc_name("Sector 2") == "sector_2"
