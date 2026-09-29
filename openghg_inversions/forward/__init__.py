"""Backend-neutral operations on labelled forward responses and domain support."""

from .domain_support import rectangular_extent_mask, remove_domain_overlap

__all__ = ["rectangular_extent_mask", "remove_domain_overlap"]
