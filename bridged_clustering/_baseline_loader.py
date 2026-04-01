"""Helpers for importing heavyweight baseline regressors on demand."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def load_baseline_regressors(names: tuple[str, ...]) -> dict[str, Any]:
    """Load selected regressors from `baseline.py` with a clearer error surface."""
    try:
        baseline_module = import_module("baseline")
    except ModuleNotFoundError as exc:
        missing_name = f" '{exc.name}'" if getattr(exc, "name", None) else ""
        raise ModuleNotFoundError(
            "Missing dependency"
            f"{missing_name} while loading Bridged Clustering baselines. "
            "Install packages from requirements.txt before running experiments.",
        ) from exc

    missing_functions = [name for name in names if not hasattr(baseline_module, name)]
    if missing_functions:
        missing_str = ", ".join(sorted(missing_functions))
        raise AttributeError(f"baseline.py is missing expected regressors: {missing_str}")

    return {name: getattr(baseline_module, name) for name in names}
