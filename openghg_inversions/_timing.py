"""Small helpers for grep-friendly runtime instrumentation."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterator
import resource
import sys


def _maxrss_kb() -> int:
    """Return peak resident set size in KiB where available."""
    maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return maxrss // 1024
    return maxrss


def _format_value(value: Any) -> str:
    """Format a timing field value without spaces."""
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "none"
    if isinstance(value, (list, tuple, set, frozenset)):
        return ",".join(_format_value(item) for item in value)
    return str(value).replace(" ", "_")


def log_timing(label: str, seconds: float, **fields: Any) -> None:
    """Print one grep-friendly timing line."""
    field_text = " ".join(
        f"{name}={_format_value(value)}" for name, value in fields.items() if value is not None
    )
    suffix = f" {field_text}" if field_text else ""
    print(f"TIMING {label} seconds={seconds:.6f} maxrss_kb={_maxrss_kb()}{suffix}")


def timer_start() -> float:
    """Return a timestamp for later ``log_timing`` calls."""
    return perf_counter()


def timer_seconds(start: float) -> float:
    """Return elapsed seconds since ``start``."""
    return perf_counter() - start


@contextmanager
def timed(label: str, **fields: Any) -> Iterator[None]:
    """Context manager that logs elapsed time for a code block."""
    start = timer_start()
    try:
        yield
    finally:
        log_timing(label, timer_seconds(start), **fields)
