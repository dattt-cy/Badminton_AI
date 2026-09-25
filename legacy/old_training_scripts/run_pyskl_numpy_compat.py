"""Run a PySKL entry point while loading NumPy 2-generated pickle files on NumPy 1."""

from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path

import numpy as np


def install_numpy_pickle_aliases() -> None:
    """Alias NumPy 2 private module names to their NumPy 1 equivalents."""
    sys.modules.setdefault("numpy._core", np.core)
    for module_name in ("multiarray", "numeric", "umath", "_multiarray_umath"):
        module = importlib.import_module(f"numpy.core.{module_name}")
        sys.modules.setdefault(f"numpy._core.{module_name}", module)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_pyskl_numpy_compat.py <pyskl-script> [args ...]")

    install_numpy_pickle_aliases()
    entrypoint = Path(sys.argv.pop(1)).resolve()
    sys.argv[0] = str(entrypoint)
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
