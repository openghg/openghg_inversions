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
from openghg_inversions.basis.operators import BucketBasisOperator, MultiSourceBucketBasisOperator
from openghg_inversions.postprocessing.border_countries import (
    EXACT_REGION_SUM,
    RETAINED_STATE_FUNCTIONAL,
    check_border_preserving_countries,
    co2_region_sum_country_outputs,
    region_class_membership,
)
from openghg_inversions.postprocessing.co2_flux_outputs import co2_country_flux_outputs
from openghg_inversions.postprocessing.countries import Countries
from openghg_inversions.rhime.co2.co2_affine_output import BoundCo2AffineFluxMap

# Rows 0-2 are land: FRANCE (lon 0-1), GERMANY (lon 2-3), SPAIN (lon 4-5). Row 3 is sea.
COUNTRY_CODES = np.array([[0, 0, 1, 1, 2, 2]] * 3 + [[3] * 6])
COORDS = {"lat": [50.0, 51.0, 52.0, 53.0], "lon": np.arange(6.0)}
SEA = -1


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
    assert json.loads(attrs["border_check_failures"])["FRANCE"][0]["source"] == "bio"


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


def test_region_sums_equal_native_country_functional_when_borders_are_preserved():
    bound, trace = _affine_fixture("explicit")
    countries = _countries()
    labels = {"fossil": FOSSIL, "bio": BIO_CROSSING}

    exact = co2_region_sum_country_outputs(trace, bound, countries, labels)
    native = co2_country_flux_outputs(trace, bound, countries)

    # The sea row is one region in both sources, so OCEAN is preserved too.
    assert exact.country.values.tolist() == ["SPAIN", "OCEAN"]
    assert exact.country_total_method.values.tolist() == [EXACT_REGION_SUM] * 2
    # Explicit U* spreads every state over the grid; the identity holds for any reconstruction.
    xr.testing.assert_allclose(
        exact.drop_vars("country_total_method"),
        native.sel(country=["SPAIN", "OCEAN"]),
        rtol=1e-6,
    )
    assert exact.attrs["country_total_method"] == EXACT_REGION_SUM
    assert json.loads(exact.attrs["non_border_preserving_countries"]) == ["FRANCE", "GERMANY"]
    assert json.loads(exact.attrs["border_check_failures"])["FRANCE"][0]["source"] == "bio"
    assert exact.attrs["uncertainty_scope"] == "retained_state_conditional"
    assert exact["country_total_posterior_mean"].attrs["units"] == "g yr-1"


@pytest.mark.parametrize("kind", ["bucket", "multisource"])
def test_bucket_labels_are_read_from_prolongation(kind):
    bound, trace = _affine_fixture(kind)
    countries = _countries()

    exact = co2_region_sum_country_outputs(trace, bound, countries)
    native = co2_country_flux_outputs(trace, bound, countries)

    # Both bucket bases follow every border; OCEAN is the sea region.
    assert exact.country.values.tolist() == ["FRANCE", "GERMANY", "SPAIN", "OCEAN"]
    xr.testing.assert_allclose(exact.drop_vars("country_total_method"), native, rtol=1e-6)
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
    assert methods["SPAIN"] == EXACT_REGION_SUM
    assert methods["FRANCE"] == RETAINED_STATE_FUNCTIONAL
    assert "country_total_method" not in result.attrs
    xr.testing.assert_allclose(
        result.drop_vars("country_total_method").sel(country=native.country.values), native, rtol=1e-6
    )


def test_constrained_only_without_any_qualifying_country_fails_with_reasons():
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
