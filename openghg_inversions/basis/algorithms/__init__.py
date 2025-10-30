"""Algorithms for computing basis functions."""
from ._quadtree import get_quadtree_basis as quadtree_algorithm
from ._weighted import nregion_landsea_basis as weighted_algorithm

__all__ = ["quadtree_algorithm", "weighted_algorithm"]
