"""Shared dataset sampling utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sample_cluster_subset(
    df: pd.DataFrame,
    eligible_clusters: np.ndarray,
    n_clusters: int,
    cluster_size: int,
    seed: int,
) -> pd.DataFrame:
    """Sample a balanced subset with `cluster_size` rows from each chosen cluster."""
    rng = np.random.default_rng(seed)
    chosen_clusters = rng.choice(eligible_clusters, size=n_clusters, replace=False)

    sampled_blocks: list[pd.DataFrame] = []
    for cluster_id in chosen_clusters:
        cluster_rows = df[df["cluster"] == cluster_id]
        sampled_blocks.append(
            cluster_rows.sample(cluster_size, random_state=int(rng.integers(0, 2**32))),
        )
    return pd.concat(sampled_blocks, ignore_index=True)
