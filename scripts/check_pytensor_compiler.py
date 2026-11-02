"""Check the resolved tox interpreter and PyTensor compiler before running tests."""

from pathlib import Path
import subprocess
import sys
import sysconfig


def main() -> None:
    """Fail before pytest when the selected Python cannot compile Python.h."""
    import pytensor

    compiler = pytensor.config.cxx
    if not compiler:
        print(f"Compiler preflight: {sys.executable}; PyTensor cxx is disabled, skipping compilation.")
        return

    include_dirs = sorted({sysconfig.get_path("include"), sysconfig.get_path("platinclude")})
    header = Path(sysconfig.get_path("include")) / "Python.h"
    context = f"Python {sys.version.split()[0]} ({sys.executable}); Python.h {header}; compiler {compiler}"
    if not header.is_file():
        raise SystemExit(
            f"Compiler preflight failed: {context}. Install matching Python development headers "
            "or select a managed interpreter with UV_PYTHON_PREFERENCE=only-managed."
        )
    try:
        subprocess.run(
            [compiler, "-x", "c++", "-fsyntax-only", *(f"-I{path}" for path in include_dirs), "-"],
            input="#include <Python.h>\n",
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)
        raise SystemExit(f"Compiler preflight failed: {context}.\n{detail}") from error
    print(f"Compiler preflight passed: {context}")


if __name__ == "__main__":
    main()
