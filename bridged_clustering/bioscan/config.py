"""Configuration objects for the BIOSCAN experiment stack."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BIOSCAN_MODEL_NAMES: tuple[str, ...] = (
    "BKM",
    "KNN",
    "FixMatch",
    "Laplacian RLS",
    "TSVR",
    "TNNR",
    "UCVME",
    "GCN",
    "Kernel Mean Matching",
    "EM",
    "EOT",
    "GW",
)


@dataclass(frozen=True)
class BioscanPaths:
    csv_path: str = "data/bioscan_data.csv"
    image_folder: str = "data/bioscan_images"


@dataclass(frozen=True)
class BioscanGridSpec:
    n_families_values: tuple[int, ...] = (3, 4, 5, 6, 7)
    n_samples_values: tuple[int, ...] = (200,)
    supervised_values: tuple[float, ...] = (0.005, 0.01, 0.015, 0.02)
    out_only_values: tuple[float, ...] = (0.1,)
    n_trials: int = 30
    model_names: tuple[str, ...] = BIOSCAN_MODEL_NAMES


def ensure_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Accept either an existing generator or a seed-like value."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)
