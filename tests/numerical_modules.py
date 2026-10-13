"""Load optional numerical modules without importing the unrelated EHR stack."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def numerical_module(name: str) -> ModuleType:
    root = Path(__file__).resolve().parents[1] / "synthlab"
    if not (root / f"{name}.py").is_file():
        raise ModuleNotFoundError(f"synthlab.{name} is not implemented")
    namespace = "_synthlab_optional_numerical"
    if namespace not in sys.modules:
        package = ModuleType(namespace)
        package.__path__ = [str(root)]
        sys.modules[namespace] = package
    module = importlib.import_module(f"{namespace}.{name}")
    assert Path(module.__file__).resolve() == root / f"{name}.py"
    return module
