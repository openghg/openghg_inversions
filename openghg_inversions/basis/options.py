"""Resolved choices shared by recipes that construct flux bases."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class FluxBasisOptions:
    """Select a saved flux basis or configure generation on a transport domain.

    Attributes:
        domain: Domain for selecting packaged support maps and labelling the basis.
        basis_path: Exact retained operator artifact.
        basis_partition_path: File containing a labelled grid partition.
        basis_partition_paths: Files containing a time-varying grid partition.
        landsea_path: Exact land/sea class map, overriding the packaged map.
        outer_regions_path: Exact outer-region map, overriding the packaged map.
        basis_algorithm: Algorithm used when no saved basis or partition is selected.
        nbasis: Target number of generated regions, counting inner regions only
            when the outer regions are fixed.
        basis_seed: Optional seed for reproducible quadtree generation. Saved
            artifacts and partitions ignore the seed.
        fix_basis_outer_regions: Use fixed outer regions during generation.
    """

    domain: str
    basis_path: Path | None = None
    basis_partition_path: Path | None = None
    basis_partition_paths: tuple[Path, ...] | None = None
    landsea_path: Path | None = None
    outer_regions_path: Path | None = None
    basis_algorithm: str | None = "weighted"
    nbasis: int = 100
    basis_seed: int | None = None
    fix_basis_outer_regions: bool = False
