"""Lightweight data structures used across the experiment package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


MODEL_ORDER: tuple[str, ...] = (
    "BKM",
    "KNN",
    "FixMatch",
    "Laplacian RLS",
    "TSVR",
    "TNNR",
    "UCVME",
    "GCN",
    "KMM",
    "EM",
    "EOT",
    "GW",
)


CandidateMap = dict[int, dict[str, np.ndarray | list[str]]]


@dataclass(frozen=True)
class PreparedTextCorpus:
    """Dataset-specific view used by the shared text experiment pipeline."""

    name: str
    frame: pd.DataFrame
    cluster_sizes: pd.Series
    candidate_map: CandidateMap | None = None
    candidate_id_column: str | None = None

    @property
    def uses_candidate_alignment(self) -> bool:
        return self.candidate_map is not None and self.candidate_id_column is not None


@dataclass(frozen=True)
class TransportSuiteSpec:
    """Hyperparameter bundle for the unmatched-regression baselines."""

    kmm: dict[str, Any] = field(default_factory=dict)
    em: dict[str, Any] = field(default_factory=dict)
    eot: dict[str, Any] = field(default_factory=dict)
    gw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextExperimentSpec:
    """All dataset-specific knobs for a text experiment family."""

    name: str
    forward_transport: TransportSuiteSpec
    reverse_transport: TransportSuiteSpec
    reverse_text_kmeans: dict[str, Any] = field(default_factory=dict)
    reverse_image_kmeans: dict[str, Any] = field(default_factory=dict)
