"""Select PyTensor's Python linker for pytest unless the caller chose a linker.

Broad test runs check backend-independent rules, so they use the Python linker
to avoid native compilation, as CI does. Compiled coverage stays explicit: set
``linker`` in ``PYTENSOR_FLAGS`` (for example ``linker=numba`` or
``linker=auto``) and that choice is kept. This module must be imported before
PyTensor, which reads ``PYTENSOR_FLAGS`` once at import.
"""

import os


def _default_python_linker() -> None:
    """Append ``linker=py`` to ``PYTENSOR_FLAGS`` when no linker is set."""
    flags = [flag for flag in os.environ.get("PYTENSOR_FLAGS", "").split(",") if flag.strip()]
    if not any(flag.partition("=")[0].strip() == "linker" for flag in flags):
        os.environ["PYTENSOR_FLAGS"] = ",".join([*flags, "linker=py"])


_default_python_linker()
