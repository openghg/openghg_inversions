"""Toy oracles for border-preserving country checks and whole-region country totals."""

import json

import numpy as np
import pytest
import xarray as xr
from openghg.util import molar_mass  # pyright: ignore[reportPrivateImportUsage]

from openghg_inversions.basis.affine_flux_map import AffineFluxMap
from openghg_inversions.basis.affine_flux_map_io import AffineFluxMapArtifact
from openghg_inversions.basis.algorithms import (
    country_border_classes,
    intersect_region_class_layers,
    region_constrained_basis,
)
from openghg_inversions.basis.covariance_products import absolute_weighted_regional_mean_restriction
from openghg_inversions.basis.operators import BucketBasisOperator, MultiSourceBucketBasisOperator
from openghg_inversions.postprocessing.border_countries import (
    EXACT_REGION_SUM,
    REGION_SUM_RETAINED_CONDITIONAL,
    RETAINED_STATE_FUNCTIONAL,
    check_border_preserving_countries,
    co2_region_sum_country_outputs,
    region_class_membership,
    region_sum_membership,
)
from openghg_inversions.postprocessing.co2_flux_outputs import co2_country_flux_outputs
from openghg_inversions.postprocessing.countries import Countries
from openghg_inversions.rhime.co2.co2_affine_output import BoundCo2AffineFluxMap
from openghg_inversions.rhime.co2.co2_outputs import Co2OutputOptions, _affine_products

# Rows 0-2 are land: FRANCE (lon 0-1), GERMANY (lon 2-3), SPAIN (lon 4-5). Row 3 is sea.
COUNTRY_CODES = np.array([[0, 0, 1, 1, 2, 2]] * 3 + [[3] * 6])
COORDS = {"lat": [50.0, 51.0, 52.0, 53.0], "lon": np.arange(6.0)}
SECONDS_PER_YEAR = 365 * 24 * 3600


def _grid(values, name="labels"):
    return xr.DataArray(np.asarray(values), dims=("lat", "lon"), coords=COORDS, name=name)


def _membership():
    """Categorical land membership; sea cells belong to no country."""
    rows = [(COUNTRY_CODES == code).astype(float) for code in range(3)]
    return xr.DataArray(
        np.stack(rows),
        dims=("country", "lat", "lon"),
        coords={"country": ["FRANCE", "GERMANY", "SPAIN"], **COORDS},
    )


def _countries():
    dataset = xr.Dataset(
        {
            "country": (("lat", "lon"), COUNTRY_CODES),
            "name": ("ncountries", ["FRANCE", "GERMANY", "SPAIN", "OCEAN"]),
        },
        coords=COORDS,
    )
    return Countries(dataset)


# Fossil regions follow every border. Bio region 1 spans FRANCE and one GERMANY column.
FOSSIL = _grid([[1, 1, 2, 2, 3, 3]] * 3 + [[4] * 6])
BIO_CROSSING = _grid([[1, 1, 1, 2, 3, 3]] * 3 + [[4] * 6])
# A different bio basis that still follows every border: FRANCE is two regions.
BIO_PRESERVING = _grid([[1, 2, 3, 3, 4, 4]] * 3 + [[5] * 6])
# Ocean regions: land is one region (label 0) that crosses every border, sea is split in two.
OCEAN = _grid([[0] * 6] * 3 + [[1, 1, 1, 2, 2, 2]])
OCEAN_SUPPORT = _grid([[0.0] * 6] * 3 + [[1.0] * 6], name="support")


def test_multisector_crossing_rejects_country_and_names_the_sector():
    labels = {"fossil": FOSSIL, "bio": BIO_CROSSING}
    check = check_border_preserving_countries(labels, _membership())

    assert check.qualifying == ("SPAIN",)
    assert check.non_qualifying == ("FRANCE", "GERMANY")
    assert check.failing_sources("FRANCE") == ("bio",)
    assert check.failing_sources("GERMANY") == ("bio",)
    (france,) = check.failures["FRANCE"]
    # Region 1 has 9 cells, 6 in FRANCE: it leaks a third and carries all of FRANCE.
    assert (france.source, france.region) == ("bio", 1)
    assert france.leak_fraction == pytest.approx(1 / 3)
    assert france.country_fraction == pytest.approx(1.0)
    (germany,) = check.failures["GERMANY"]
    assert germany.leak_fraction == pytest.approx(2 / 3)
    assert germany.country_fraction == pytest.approx(0.5)
    # The fossil sector alone preserves every border.
    assert check.regions["FRANCE"]["fossil"] == (1,)
    assert check.regions["SPAIN"] == {"fossil": (3,), "bio": (3,)}
    attrs = check.to_attrs()
    assert json.loads(attrs["non_border_preserving_countries"]) == ["FRANCE", "GERMANY"]
    details = json.loads(attrs["border_check_details"])["FRANCE"]
    assert details["bio"]["status"] == "fails"
    assert details["bio"]["crossing_regions"][0]["region"] == 1
    assert details["fossil"]["status"] == "preserved"


def test_ocean_only_sector_does_not_disqualify_coastal_country():
    labels = {"fossil": FOSSIL, "ocean": OCEAN}
    support = {"fossil": _grid(np.ones((4, 6)), name="support"), "ocean": OCEAN_SUPPORT}
    check = check_border_preserving_countries(labels, _membership(), support)

    assert check.qualifying == ("FRANCE", "GERMANY", "SPAIN")
    assert check.exempt_sources["FRANCE"] == ("ocean",)
    assert check.regions["FRANCE"] == {"fossil": (1,), "ocean": ()}

    # The exemption follows labels and support, not the source name. Ocean flux on one
    # coastal FRANCE cell makes FRANCE depend on the land-wide ocean region 0, whose
    # support then lies wholly in FRANCE: still a whole-region sum.
    one_coast = OCEAN_SUPPORT.copy()
    one_coast[2, 0] = 1.0
    one_check = check_border_preserving_countries(labels, _membership(), {**support, "ocean": one_coast})
    assert one_check.qualifying == ("FRANCE", "GERMANY", "SPAIN")
    assert one_check.regions["FRANCE"]["ocean"] == (0,)
    # Ocean flux on coastal FRANCE and GERMANY cells makes region 0 cross their border.
    two_coasts = one_coast.copy()
    two_coasts[2, 2] = 1.0
    two_check = check_border_preserving_countries(labels, _membership(), {**support, "ocean": two_coasts})
    assert two_check.failing_sources("FRANCE") == ("ocean",)
    assert two_check.failing_sources("GERMANY") == ("ocean",)
    assert two_check.qualifying == ("SPAIN",)


def test_fractional_border_cells_pass_only_within_tolerance():
    membership = _membership()
    # One FRANCE cell is 4% GERMANY.
    membership.loc[{"country": "FRANCE", "lat": 50.0, "lon": 1.0}] = 0.96
    membership.loc[{"country": "GERMANY", "lat": 50.0, "lon": 1.0}] = 0.04
    labels = {"fossil": FOSSIL}

    strict = check_border_preserving_countries(labels, membership)
    assert strict.qualifying == ("SPAIN",)
    loose = check_border_preserving_countries(labels, membership, tolerance=0.05)
    assert loose.qualifying == ("FRANCE", "GERMANY", "SPAIN")
    # GERMANY's sliver in FRANCE's region is left out of its region sum.
    assert loose.regions["GERMANY"]["fossil"] == (2,)


def _affine_fixture(kind):
    """CO2 map with draws: explicit random U* (two sources), a one-source bucket, or a multisource bucket."""
    rng = np.random.default_rng(256)
    sources = ["fossil", "bio"]
    dims = ("lat", "lon") if kind == "bucket" else ("native_source", "lat", "lon")
    coords = {**COORDS} if kind == "bucket" else {"native_source": sources, **COORDS}
    shape = tuple(len(coords[dim]) for dim in dims)
    mean = xr.DataArray(rng.uniform(0.5, 1.5, shape), dims=dims, coords=coords, attrs={"units": "1"})
    flux = mean.copy(data=rng.uniform(-3, 3, shape))
    flux.attrs = {"units": "mol m-2 s-1"}
    if kind == "explicit":
        state = xr.Coordinates({"region": np.arange(5)})
        prolongation = xr.DataArray(
            rng.normal(size=(*shape, 5)),
            dims=(*dims, "region"),
            coords={**coords, **state},
            attrs={"units": "1"},
        )
    else:
        if kind == "bucket":
            prolongation = BucketBasisOperator(FOSSIL, state_dim="region")
        else:
            prolongation = MultiSourceBucketBasisOperator(
                {"fossil": FOSSIL, "bio": BIO_PRESERVING}, state_dim="region"
            )
        mean = mean.copy(data=np.ones(shape))  # Native scaling equals the bucket state.
        state = prolongation.basis_matrix.coords["region"].to_index()
        state = (
            xr.Coordinates.from_pandas_multiindex(state, "region")
            if kind != "bucket"
            else xr.Coordinates({"region": state})
        )
    nstate = len(state["region"])
    reference = xr.DataArray(np.ones(nstate), dims="region", coords=state, attrs={"units": "1"})
    mapping = AffineFluxMap(mean, flux, prolongation, state_dim="region")
    bound = BoundCo2AffineFluxMap(
        AffineFluxMapArtifact(mapping, "fixture", {"projection_strategy": "fixture"}, {}), reference
    )
    groups = {}
    for group, chains in {"prior": 1, "posterior": 2}.items():
        groups[group] = xr.Dataset(
            {
                "x": xr.DataArray(
                    rng.uniform(0.2, 2.0, (chains, 5, nstate)),
                    dims=("chain", "draw", "region"),
                    coords={"chain": np.arange(chains), "draw": np.arange(5), **state},
                    attrs={"units": "1"},
                )
            }
        )
    groups["/"] = xr.Dataset(
        attrs={
            "rhime_model_metadata": json.dumps({"recipe": "co2"}),
            "rhime_variable_roles": json.dumps({"flux_scale": "x"}),
        }
    )
    return bound, xr.DataTree.from_dict(groups)


def _summary(draws):
    """Pooled mean, population stdev and 0.159/0.841 quantiles over (chain, draw, ...)."""
    samples = draws.reshape((-1, *draws.shape[2:]))
    return samples.mean(0), samples.std(0), np.quantile(samples, [0.159, 0.841], axis=0)


def _assert_matches(result, name, draws, dims):
    mean, stdev, quantiles = _summary(draws)
    np.testing.assert_allclose(result[f"{name}_mean"].transpose(*dims), mean, rtol=1e-6)
    np.testing.assert_allclose(result[f"{name}_stdev"].transpose(*dims), stdev, rtol=1e-6)
    np.testing.assert_allclose(result[f"{name}_quantile"].transpose("quantile", *dims), quantiles, rtol=1e-6)


def _explicit_oracle(bound, state, membership, area):
    """Country draws (chain, draw, country, source) from F (m + U* (alpha - alpha_ref)), in NumPy."""
    mapping = bound.affine_map
    delta = state.values - bound.reference_state.values
    native = mapping.native_mean.values + np.einsum("slir,cdr->cdsli", mapping.prolongation.values, delta)
    weights = mapping.flux.values * area * molar_mass("co2") * SECONDS_PER_YEAR
    return np.einsum("cdsli,sli,ksli->cdks", native, weights, membership)


def test_region_sums_match_an_independent_multisource_oracle():
    bound, trace = _affine_fixture("explicit")
    countries = _countries()
    labels = {"fossil": FOSSIL, "bio": BIO_CROSSING}

    result = co2_region_sum_country_outputs(trace, bound, countries, labels)

    # The sea row is one region in both sources, so OCEAN is preserved too.
    assert result.country.values.tolist() == ["SPAIN", "OCEAN"]
    # Without the restriction an explicit map cannot certify completeness.
    assert result.country_total_method.values.tolist() == [REGION_SUM_RETAINED_CONDITIONAL] * 2
    assert np.isnan(result.restriction_residual).all()
    assert result.border_leak_fraction.values.tolist() == [0.0, 0.0]
    # Independent oracle: whole regions per source, read from the labels directly.
    union = np.stack(
        [
            np.stack([np.isin(FOSSIL.values, [3]), np.isin(BIO_CROSSING.values, [3])]),
            np.stack([np.isin(FOSSIL.values, [4]), np.isin(BIO_CROSSING.values, [4])]),
        ]
    ).astype(float)
    area = countries.area_grid.values
    for group in ("prior", "posterior"):
        draws = _explicit_oracle(bound, trace[group]["x"], union, area)
        _assert_matches(result, f"country_{group}", draws, ("country", "native_source"))
        _assert_matches(result, f"country_total_{group}", draws.sum(-1), ("country",))
    # With exact categorical borders the region sums also equal the ordinary functional.
    native = co2_country_flux_outputs(trace, bound, countries).sel(country=["SPAIN", "OCEAN"])
    xr.testing.assert_allclose(result.reset_coords(drop=True), native.reset_coords(drop=True), rtol=1e-6)
    assert json.loads(result.attrs["non_border_preserving_countries"]) == ["FRANCE", "GERMANY"]
    assert result.attrs["restriction_residual_basis"].startswith("not computed")
    assert result.attrs["uncertainty_scope"] == "retained_state_conditional"
    assert result["country_total_posterior_mean"].attrs["units"] == "g yr-1"


@pytest.mark.parametrize("kind", ["bucket", "multisource"])
def test_bucket_labels_are_read_from_prolongation(kind):
    bound, trace = _affine_fixture(kind)
    countries = _countries()

    exact = co2_region_sum_country_outputs(trace, bound, countries)
    native = co2_country_flux_outputs(trace, bound, countries)

    # Both bucket bases follow every border; OCEAN is the sea region.
    assert exact.country.values.tolist() == ["FRANCE", "GERMANY", "SPAIN", "OCEAN"]
    # A bucket reconstruction has no unresolved component: every row is exact.
    assert set(exact.country_total_method.values) == {EXACT_REGION_SUM}
    assert (exact.restriction_residual == 0).all()
    xr.testing.assert_allclose(exact.reset_coords(drop=True), native.reset_coords(drop=True), rtol=1e-6)
    # A bucket region total is the region's flux total times its scaling draw.
    france = FOSSIL.values == 1
    weights = bound.affine_map.flux.values * countries.area_grid.values * molar_mass("co2") * 365 * 24 * 3600
    draws = trace["posterior"]["x"].isel(region=0).values  # FRANCE is the first region of each source.
    if kind == "bucket":
        expected = (weights[france].sum() * draws).mean()
    else:
        # In the bio basis FRANCE is two regions, the first and second bio states.
        fossil = weights[0][france].sum() * draws
        bio_states = trace["posterior"]["x"].sel(source="bio").values
        bio_regions = BIO_PRESERVING.values
        bio = sum(weights[1][bio_regions == label].sum() * bio_states[..., label - 1] for label in (1, 2))
        expected = (fossil + bio).mean()
    np.testing.assert_allclose(
        exact["country_total_posterior_mean"].sel(country="FRANCE"), expected, rtol=1e-6
    )


def test_all_option_keeps_every_country_and_labels_its_method():
    bound, trace = _affine_fixture("explicit")
    countries = _countries()
    labels = {"fossil": FOSSIL, "bio": BIO_CROSSING}

    result = co2_region_sum_country_outputs(trace, bound, countries, labels, country_outputs="all")
    native = co2_country_flux_outputs(trace, bound, countries)

    assert sorted(result.country.values.tolist()) == ["FRANCE", "GERMANY", "OCEAN", "SPAIN"]
    methods = dict(zip(result.country.values, result.country_total_method.values))
    assert methods["SPAIN"] == REGION_SUM_RETAINED_CONDITIONAL
    assert methods["FRANCE"] == RETAINED_STATE_FUNCTIONAL
    assert result.attrs["country_outputs"] == "all"
    assert np.isnan(result.border_leak_fraction.sel(country="FRANCE"))
    xr.testing.assert_allclose(
        result.reset_coords(drop=True).sel(country=native.country.values),
        native.reset_coords(drop=True),
        rtol=1e-6,
    )


def test_border_preserving_only_without_any_qualifying_country_fails_with_reasons():
    bound, trace = _affine_fixture("explicit")
    everything = _grid(np.zeros((4, 6), dtype=int))
    with pytest.raises(ValueError, match="No country is a sum of whole basis regions"):
        co2_region_sum_country_outputs(trace, bound, _countries(), {"fossil": everything, "bio": everything})


def test_region_classes_used_to_build_the_basis_all_qualify():
    rng = np.random.default_rng(1)
    landsea = _grid(np.where(COUNTRY_CODES == 3, "sea", "land"), name="landsea")
    border = country_border_classes(_membership(), ["FRANCE", "GERMANY"])
    assert border.values[0].tolist() == ["FRANCE"] * 2 + ["GERMANY"] * 2 + ["OTHER"] * 2
    assert set(border.values[3]) == {"OTHER"}
    classes = intersect_region_class_layers({"landsea": landsea, "border": border})

    labels = {
        source: region_constrained_basis(_grid(rng.uniform(size=(4, 6)), name="weights"), classes, 8)
        for source in ("fossil", "bio")
    }
    assert not labels["fossil"].equals(labels["bio"])

    by_class = check_border_preserving_countries(labels, region_class_membership(classes))
    assert len(by_class.qualifying) == 4 and not by_class.failures
    by_country = check_border_preserving_countries(labels, _membership())
    assert by_country.qualifying[:2] == ("FRANCE", "GERMANY")


def test_inactive_cells_are_not_regions_but_count_against_a_country():
    # One FRANCE cell is inactive (-1), as in a restriction without a row there; the
    # sea row is inactive too but carries no support.
    fossil = FOSSIL.copy(data=np.where(COUNTRY_CODES == 3, -1, FOSSIL.values))
    fossil[0, 0] = -1
    support = _grid(np.where(COUNTRY_CODES == 3, 0.0, 1.0), name="support")
    check = check_border_preserving_countries({"fossil": fossil}, _membership(), {"fossil": support})

    assert check.qualifying == ("GERMANY", "SPAIN")
    assert check.failing_sources("FRANCE") == ("fossil",)
    france = check.sources["FRANCE"]["fossil"]
    assert france.inactive_fraction == pytest.approx(1 / 6)
    assert france.missed_fraction == 0.0
    # Inactive cells are never reported as a region, and FRANCE keeps its whole region.
    assert "FRANCE" not in check.failures
    assert check.regions["FRANCE"]["fossil"] == (1,)
    assert all(-1 not in regions for borders in check.regions.values() for regions in borders.values())
    details = json.loads(check.to_attrs()["border_check_details"])["FRANCE"]["fossil"]
    assert details["inactive_fraction"] == pytest.approx(1 / 6)
    assert details["crossing_regions"] == []
    # The whole-region membership leaves inactive cells out.
    union = region_sum_membership(check, {"fossil": fossil}, countries=["FRANCE"])
    assert float(union.sel(country="FRANCE").sum()) == 5.0


def test_negligible_sector_is_opt_in_reported_and_never_exact():
    labels = {"fossil": FOSSIL, "ocean": OCEAN}
    tiny = OCEAN_SUPPORT.copy()
    tiny[2, 0] = tiny[2, 2] = 1e-9  # Ocean flux in one coastal FRANCE and one GERMANY cell.
    support = {"fossil": _grid(np.ones((4, 6)), name="support"), "ocean": tiny}

    strict = check_border_preserving_countries(labels, _membership(), support)
    assert strict.qualifying == ("SPAIN",)
    assert strict.failing_sources("FRANCE") == ("ocean",)
    # The failure report shows how small the failing source is.
    assert strict.sources["FRANCE"]["ocean"].share == pytest.approx(1e-9 / 6)

    relaxed = check_border_preserving_countries(labels, _membership(), support, negligible_source_share=1e-6)
    assert relaxed.qualifying == ("FRANCE", "GERMANY", "SPAIN")
    assert relaxed.negligible_sources("FRANCE") == ("ocean",)
    # The ocean region is left out of FRANCE's region sum, so FRANCE leaks a little.
    assert relaxed.leak_fraction("FRANCE") == pytest.approx(1e-9 / 6, rel=1e-6)
    attrs = relaxed.to_attrs()
    assert attrs["border_check_negligible_source_share"] == "1e-06"
    assert json.loads(attrs["border_check_details"])["FRANCE"]["ocean"]["status"] == "negligible"
    assert json.loads(attrs["border_leak_fractions"])["SPAIN"] == 0.0


def test_region_sums_under_a_positive_tolerance_record_their_leak():
    bound, trace = _affine_fixture("bucket")
    countries = _countries()
    membership = _membership()
    # One FRANCE cell is 4% GERMANY.
    membership.loc[{"country": "FRANCE", "lat": 50.0, "lon": 1.0}] = 0.96
    membership.loc[{"country": "GERMANY", "lat": 50.0, "lon": 1.0}] = 0.04
    area = countries.area_grid
    support = {"total": area.astype("float64")}

    with pytest.raises(ValueError, match="No country"):
        co2_region_sum_country_outputs(
            trace, bound, countries, membership=membership.sel(country=["GERMANY"]), support=support
        )
    result = co2_region_sum_country_outputs(
        trace, bound, countries, membership=membership, support=support, tolerance=0.05
    )

    methods = dict(zip(result.country.values, result.country_total_method.values))
    assert methods == {
        "FRANCE": REGION_SUM_RETAINED_CONDITIONAL,
        "GERMANY": REGION_SUM_RETAINED_CONDITIONAL,
        "SPAIN": EXACT_REGION_SUM,
    }
    cell = area.values[0, 1]
    for country in ("FRANCE", "GERMANY"):
        country_area = float((membership.sel(country=country) * area).sum())
        assert float(result.border_leak_fraction.sel(country=country)) == pytest.approx(
            0.04 * cell / country_area
        )
    # Totals differ from the fractional native functional by exactly the 4% sliver.
    flux = bound.affine_map.flux.values * area.values * molar_mass("co2") * SECONDS_PER_YEAR
    france_draws = trace["posterior"]["x"].isel(region=0).values  # Bucket region 1 is FRANCE.
    states = trace["posterior"]["x"].values
    native_scaling = states[..., FOSSIL.values - 1]  # (chain, draw, lat, lon)
    for country, sign in (("FRANCE", 1.0), ("GERMANY", -1.0)):
        native = np.einsum("cdli,li->cd", native_scaling, flux * membership.sel(country=country).values)
        expected = native + sign * 0.04 * flux[0, 1] * france_draws
        np.testing.assert_allclose(
            result["country_total_posterior_mean"].sel(country=country), expected.mean(), rtol=1e-6
        )


def _coherent_fixture(case):
    """Dense native Gaussian coherent reduction with an absolute-flux-weighted restriction.

    Fossil regions follow every border. Bio regions too, except that the sea row is
    inactive for bio, so OCEAN never qualifies.
    """
    rng = np.random.default_rng(7)
    sources = ["fossil", "bio"]
    labels = {
        "fossil": FOSSIL - 1,
        "bio": BIO_PRESERVING.copy(data=np.where(COUNTRY_CODES == 3, -1, BIO_PRESERVING.values - 1)),
    }
    area = _countries().area_grid.astype("float64")
    shape = (2, 4, 6)
    if case == "same_sign":
        flux_values = rng.uniform(0.5, 3.0, shape)
    elif case == "mixed_sign":
        flux_values = rng.uniform(-3.0, 3.0, shape)
    else:  # Same sign, but two times with different spatial patterns.
        flux_values = rng.uniform(0.5, 3.0, (*shape, 2))
    dims = ("native_source", "lat", "lon", "time")[: flux_values.ndim]
    coords = {"native_source": sources, **COORDS}
    if flux_values.ndim == 4:
        coords["time"] = np.array(["2020-01-01", "2020-07-01"], dtype="datetime64[ns]")
    flux = xr.DataArray(flux_values, dims=dims, coords=coords, attrs={"units": "mol m-2 s-1"})
    q = (flux.mean("time") if "time" in flux.dims else flux) * area
    restriction = absolute_weighted_regional_mean_restriction(labels, q, state_dim="region")
    pi = np.asarray(restriction.data.todense()).reshape(restriction.sizes["region"], -1)
    nstate = pi.shape[0]

    lat, lon = np.meshgrid(np.arange(4), np.arange(6), indexing="ij")
    cells = np.stack([lat.ravel(), lon.ravel()], axis=1)
    correlation = np.exp(-np.linalg.norm(cells[:, None] - cells[None], axis=-1) / 1.5)
    covariance = np.kron(np.eye(2), correlation) + 1e-3 * np.eye(48)
    gram = pi @ covariance @ pi.T
    lift = covariance @ pi.T @ np.linalg.inv(gram)
    unresolved = covariance - lift @ pi @ covariance

    state = {"region": np.arange(nstate)}
    native_coords = {"native_source": sources, **COORDS}
    mean = xr.DataArray(np.ones(shape), dims=dims[:3], coords=native_coords, attrs={"units": "1"})
    prolongation = xr.DataArray(
        lift.reshape(*shape, nstate),
        dims=(*dims[:3], "region"),
        coords={**native_coords, **state},
        attrs={"units": "1"},
    )
    # Rows sum to one, so alpha_ref = Pi m = 1.
    reference = xr.DataArray(np.ones(nstate), dims="region", coords=state, attrs={"units": "1"})
    mapping = AffineFluxMap(mean, flux, prolongation, state_dim="region")
    bound = BoundCo2AffineFluxMap(
        AffineFluxMapArtifact(mapping, "fixture", {"projection_strategy": "supplied_restriction"}, {}),
        reference,
    )
    groups = {
        group: xr.Dataset(
            {
                "x": xr.DataArray(
                    rng.uniform(0.5, 1.5, (1, 6, nstate)),
                    dims=("chain", "draw", "region"),
                    coords={"chain": [0], "draw": np.arange(6), **state},
                    attrs={"units": "1"},
                )
            }
        )
        for group in ("prior", "posterior")
    }
    groups["/"] = xr.Dataset(
        attrs={
            "rhime_model_metadata": json.dumps({"recipe": "co2"}),
            "rhime_variable_roles": json.dumps({"flux_scale": "x"}),
        }
    )
    # The restriction keeps its sparse payload; its MultiIndex state labels play no part.
    restriction = restriction.drop_vars(["region", "source", "region_in_source"], errors="ignore")
    return bound, xr.DataTree.from_dict(groups), labels, restriction, covariance, unresolved


@pytest.mark.parametrize("case", ["same_sign", "mixed_sign", "time_varying"])
def test_coherent_region_sums_are_exact_only_inside_the_restriction_row_space(case):
    bound, trace, labels, restriction, covariance, unresolved = _coherent_fixture(case)
    countries = _countries()

    result = co2_region_sum_country_outputs(trace, bound, countries, labels, restriction=restriction)

    land = ["FRANCE", "GERMANY", "SPAIN"]
    assert result.country.values.tolist() == land
    details = json.loads(result.attrs["border_check_details"])
    assert details["OCEAN"]["bio"]["inactive_fraction"] == pytest.approx(1.0)
    # Oracle: unresolved native variance Q A Q^T of each reported region-sum functional.
    union = region_sum_membership(
        check_border_preserving_countries(labels, countries.matrix), labels, source_dim="native_source"
    ).sel(country=land)
    flux = bound.affine_map.flux
    functional = flux * union * countries.area_grid
    extra = [dim for dim in functional.dims if dim not in ("country", "native_source", "lat", "lon")]
    q = functional.transpose("country", *extra, "native_source", "lat", "lon").values.reshape(3, -1, 48)
    ratio = np.einsum("kti,ij,ktj->kt", q, unresolved, q) / np.einsum("kti,ij,ktj->kt", q, covariance, q)
    residual = result.restriction_residual.values
    if case == "same_sign":
        assert set(result.country_total_method.values) == {EXACT_REGION_SUM}
        assert residual.max() < 1e-10
        assert ratio.max() < 1e-10
        # Nothing unresolved, so the complete posterior total equals the reported region sum.
        delta = trace["posterior"]["x"].values - 1.0
        native = 1.0 + np.einsum("slir,cdr->cdsli", bound.affine_map.prolongation.values, delta)
        expected = (
            np.einsum(
                "cdsli,ksli->cdk",
                native,
                functional.transpose("country", "native_source", "lat", "lon").values,
            )
            * molar_mass("co2")
            * SECONDS_PER_YEAR
        )
        np.testing.assert_allclose(result["country_total_posterior_mean"], expected.mean((0, 1)), rtol=1e-6)
    else:
        assert set(result.country_total_method.values) == {REGION_SUM_RETAINED_CONDITIONAL}
        assert residual.min() > 1e-2
        assert ratio.min() > 1e-4
    residuals = json.loads(result.attrs["restriction_residuals"])
    assert set(residuals) == set(land) and set(residuals["FRANCE"]) == {"fossil", "bio"}


def test_co2_output_option_writes_only_border_preserving_countries_for_every_sector(tmp_path):
    options = Co2OutputOptions(
        country_file=tmp_path / "countries.nc", country_outputs="border_preserving_only"
    )
    assert options.country_outputs == "border_preserving_only"
    assert Co2OutputOptions().country_outputs == "all"
    with pytest.raises(ValueError, match="basic output and a country_file"):
        Co2OutputOptions(country_outputs="border_preserving_only")
    with pytest.raises(ValueError, match="supports only"):
        Co2OutputOptions(country_file=tmp_path / "countries.nc", country_outputs="constrained")

    bound, trace = _affine_fixture("explicit")
    countries = _countries()
    labels = {"fossil": FOSSIL, "bio": BIO_CROSSING}
    sectors = {"fossil": "ff", "bio": "bio"}
    _, country = _affine_products(
        trace, bound, countries, sectors, country_outputs="border_preserving_only", labels=labels
    )
    assert country.country.values.tolist() == ["SPAIN", "OCEAN"]
    assert country.country_total_method.values.tolist() == [REGION_SUM_RETAINED_CONDITIONAL] * 2
    np.testing.assert_allclose(
        country["country_ff_posterior_mean"], country["country_posterior_mean"].sel(native_source="fossil")
    )
    _, everything = _affine_products(trace, bound, countries, sectors)
    assert everything.country.values.tolist() == ["FRANCE", "GERMANY", "SPAIN", "OCEAN"]
    assert "country_total_method" not in everything.coords
