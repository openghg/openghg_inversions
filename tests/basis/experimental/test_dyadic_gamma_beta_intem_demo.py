"""End-to-end tests for the InTEM land/ocean Gamma--Beta demonstration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest


@pytest.fixture(scope="module")
def demo_module() -> ModuleType:
    """Load the example script as a module without changing Python paths."""
    repository_root = Path(__file__).parents[3]
    script = repository_root / "examples/basis/dyadic_gamma_beta_intem_demo.py"
    spec = importlib.util.spec_from_file_location("_test_dyadic_gamma_beta_intem_demo", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load demonstration module from {script}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def small_case(demo_module: ModuleType):
    """Build one shallow, inexpensive case from committed OGI fixtures."""
    return demo_module.build_case(draws=64, inner_regions=8, max_depth=2, seed=17)


def test_realistic_case_uses_fixed_outer_and_inner_land_ocean_groups(
    small_case,
) -> None:
    """Packaged masks produce six fixed outer and two refinable inner groups."""
    case = small_case
    forest = case.forest
    group_names = tuple(group.name for group in forest.groups)

    assert group_names[:6] == tuple(f"intem_outer_{index}" for index in range(6))
    assert group_names[6:] == ("inner_ocean", "inner_land")
    assert np.all(case.group_labels[case.intem_classes < 6] == case.intem_classes[case.intem_classes < 6])
    assert set(np.unique(case.group_labels[case.intem_classes == 6])) == {6, 7}

    for outer_index in range(6):
        outer_leaves = [
            node_id for node_id in forest.leaf_ids if forest.nodes[node_id].group_index == outer_index
        ]
        assert len(outer_leaves) == 1
        assert not forest.nodes[outer_leaves[0]].child_ids


def test_terminal_labels_do_not_cross_semantic_groups(small_case) -> None:
    """Every terminal forest label remains inside one declared hard group."""
    labels = small_case.forest.leaf_labels()

    assert np.all(labels > 0)
    for label in np.unique(labels):
        assert np.unique(small_case.group_labels[labels == label]).size == 1


def test_case_is_seeded_and_conserves_expected_flux(demo_module: ModuleType, small_case) -> None:
    """Repeated construction reproduces draws and conserves every sampled split."""
    repeated = demo_module.build_case(draws=64, inner_regions=8, max_depth=2, seed=17)

    np.testing.assert_array_equal(small_case.samples.node_scalings, repeated.samples.node_scalings)
    assert small_case.samples.maximum_conservation_error() < 1.0e-8


def test_summary_and_report_artifacts_are_complete(
    demo_module: ModuleType,
    small_case,
    tmp_path: Path,
) -> None:
    """The executable example emits its figure, report, and numeric summary."""
    summary = demo_module.summarize_case(small_case)
    demo_module.write_report(small_case, summary, tmp_path)

    assert summary.fixed_outer_region_count == 6
    assert summary.group_count == 8
    assert summary.inner_land_components >= 1
    assert summary.inner_ocean_components >= 1
    assert summary.inner_region_budget == 8
    assert summary.inner_land_regions + summary.inner_ocean_regions == 8
    assert summary.inner_land_regions > summary.inner_ocean_regions
    assert summary.inner_land_partition_weight > summary.inner_ocean_partition_weight
    assert summary.leaf_count == 14
    assert summary.stochastic_coordinate_count == 11
    assert summary.minimum_kappa == pytest.approx(2.0)
    assert summary.maximum_kappa == pytest.approx(4.0)
    assert summary.minimum_leaf_variance > 0.0
    assert summary.maximum_leaf_variance >= summary.median_leaf_variance
    assert -1.0 <= summary.median_inner_land_correlation <= 1.0
    assert summary.terminal_covariance_rank == summary.stochastic_coordinate_count
    assert summary.maximum_conservation_error < 1.0e-8
    assert (tmp_path / "intem_gamma_beta_summary.json").is_file()
    assert (tmp_path / "intem_gamma_beta_summary.png").is_file()
    assert (tmp_path / "intem_gamma_beta_covariance_matrix.png").is_file()
    assert (tmp_path / "intem_gamma_beta_covariance_maps.png").is_file()
    assert (tmp_path / "intem_gamma_beta_correlation_maps.png").is_file()
    assert (tmp_path / "intem_gamma_beta_distance_fit.json").is_file()
    assert (tmp_path / "intem_gamma_beta_distance_fit_matrices.png").is_file()
    assert (tmp_path / "intem_gamma_beta_distance_fit_maps.png").is_file()
    report = (tmp_path / "intem_gamma_beta_report.md").read_text()
    assert "InTEM land/ocean Gamma--Beta prior prototype" in report
    assert "Exact terminal-state covariance" in report
    assert "Exponential distance-covariance comparison" in report
    assert "This prototype does not yet infer the active partition" in report


def test_inner_budget_uses_standard_basis_weight(demo_module: ModuleType, small_case) -> None:
    """The partition weight is mean absolute footprint-times-flux sensitivity."""
    data = demo_module.load_tac_mhd_week_demo_data(Path("tests/data"))

    np.testing.assert_allclose(
        small_case.partition_weight,
        np.mean(np.abs(data.G), axis=0, dtype=np.float64),
    )
    assert small_case.inner_region_targets == (3, 5)


def test_flat_topology_mode_uses_area_allocation(demo_module: ModuleType) -> None:
    """Flat mode assigns equal topology weight to every mapped grid cell."""
    case = demo_module.build_case(
        draws=2,
        inner_regions=8,
        max_depth=2,
        topology_weight_mode="flat",
        seed=17,
    )

    np.testing.assert_array_equal(case.partition_weight, np.ones(case.forest.shape))
    assert case.topology_weight_mode == "flat"
    assert sum(case.inner_region_targets) == 8


def test_inner_land_and_ocean_root_variances_can_differ(demo_module: ModuleType) -> None:
    """Group-specific overrides avoid coupling country and ocean uncertainty."""
    case = demo_module.build_case(
        draws=2,
        inner_regions=8,
        max_depth=2,
        inner_land_root_variance=0.04,
        inner_ocean_root_variance=0.25,
        seed=17,
    )
    variance_by_group = {group.name: group.root_variance for group in case.forest.groups}

    assert variance_by_group["inner_land"] == pytest.approx(0.04)
    assert variance_by_group["inner_ocean"] == pytest.approx(0.25)


def test_country_mask_loader_selects_aligned_uk_fixture(
    demo_module: ModuleType,
    small_case,
) -> None:
    """The calibration country loader finds the aligned UK support."""
    mask = demo_module.load_country_mask(
        small_case,
        Path("tests/data"),
        "UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND",
    )

    assert mask.shape == small_case.forest.shape
    assert np.count_nonzero(mask) == 392
    assert not mask.flags.writeable


def test_covariance_to_correlation_preserves_sign_and_unit_diagonal(
    demo_module: ModuleType,
) -> None:
    """Correlation normalization preserves covariance sign and unit variance."""
    covariance = np.array([[4.0, -1.0], [-1.0, 9.0]])

    correlation = demo_module._covariance_to_correlation(covariance)

    np.testing.assert_allclose(correlation, np.array([[1.0, -1.0 / 6.0], [-1.0 / 6.0, 1.0]]))


def test_regional_values_use_exact_terminal_label_supports(demo_module: ModuleType) -> None:
    """Projected values remain constant inside the supplied terminal regions."""
    labels = np.array([[1, 1, 2], [1, 3, 3]], dtype=np.int64)
    regional_values = np.array([0.2, 0.7, 0.4])

    grid = demo_module._regional_values_to_grid(labels, regional_values)

    np.testing.assert_allclose(grid, np.array([[0.2, 0.2, 0.7], [0.2, 0.4, 0.4]]))


def test_distance_comparison_projects_native_covariance_to_terminal_regions(
    demo_module: ModuleType,
    small_case,
) -> None:
    """The distance benchmark uses the exact expected-mass restriction."""
    comparison = demo_module.build_distance_covariance_comparison(small_case)

    assert comparison.covariance_fit.converged
    assert comparison.correlation_fit.converged
    assert comparison.covariance_fit.pair_count == 10
    assert comparison.correlation_fit.pair_count == 10
    np.testing.assert_allclose(comparison.restriction_transpose.sum(axis=0), 1.0)
    labels = small_case.forest.leaf_labels().reshape(-1) - 1
    prolongation = np.eye(len(small_case.forest.leaf_ids), dtype=np.float64)[labels]
    np.testing.assert_allclose(
        comparison.restriction_transpose.T @ prolongation,
        np.eye(len(small_case.forest.leaf_ids)),
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        np.diag(comparison.same_sigma_covariance),
        np.diag(comparison.current_covariance),
    )
    assert np.max(np.diag(comparison.group_scale_covariance)) == pytest.approx(1.0)
    assert np.all(
        comparison.group_scale_covariance[comparison.group_labels[:, None] != comparison.group_labels] == 0.0
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"draws": 1}, "draws must be at least 2"),
        ({"max_depth": 0}, "max_depth must be at least 1"),
    ],
)
def test_demo_rejects_configs_without_empirical_split_diagnostics(
    demo_module: ModuleType,
    kwargs: dict[str, int],
    message: str,
) -> None:
    """The demo rejects configurations unsupported by its summary plots."""
    with pytest.raises(ValueError, match=message):
        demo_module.build_case(**kwargs)


@pytest.mark.parametrize(
    ("values", "name"),
    [
        (np.array([[0.9, 1.0]]), "InTEM"),
        (np.array([[0.0, 1.9]]), "land/ocean"),
    ],
)
def test_mask_validation_rejects_fractional_labels_before_casting(
    demo_module: ModuleType,
    values: np.ndarray,
    name: str,
) -> None:
    """Fractional labels cannot silently truncate into valid semantic groups."""
    with pytest.raises(ValueError, match="must be integers before conversion"):
        demo_module._validated_integer_mask(
            values,
            name=name,
            allowed=np.array([0, 1]),
        )
