"""Algorithmic core of Bridged Clustering.

If you want to inspect the method itself, start here. This module contains the
cluster-wise split, balanced clustering, bridge estimation, oracle comparison,
and centroid-based inference reused by the experiment scripts.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, pairwise_distances, silhouette_score

try:
    from k_means_constrained import KMeansConstrained
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in dependency-light environments
    KMeansConstrained = None
    _KMEANS_CONSTRAINED_IMPORT_ERROR = exc
else:
    _KMEANS_CONSTRAINED_IMPORT_ERROR = None


def _require_kmeans_constrained() -> None:
    if KMeansConstrained is None:
        raise ModuleNotFoundError(
            "k_means_constrained is required for Bridged Clustering. "
            "Install packages from requirements.txt before running experiments.",
        ) from _KMEANS_CONSTRAINED_IMPORT_ERROR


def split_by_cluster(
    df: pd.DataFrame,
    supervised_ratio: float,
    output_only_ratio: float,
    K: int | None = None,
    seed: int | None = None,
    mode: str = "transductive",
    test_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a labelled semantic table into BC training and evaluation blocks.

    The input DataFrame is expected to contain a semantic `cluster` column plus
    the modality columns consumed downstream by the experiment scripts. The
    return value is `(supervised, input_only, output_only, test, input_pool,
    output_pool)`, where the last two pools append the supervised pairs to the
    unmatched marginals. In transductive mode, the same remainder is used for
    both inference and evaluation; in inductive mode, a held-out test slice is
    carved from that remainder.
    """
    del K  # kept for signature compatibility with the original scripts
    if mode not in {"transductive", "inductive"}:
        raise ValueError(f"Unsupported mode: {mode}")

    rng = np.random.default_rng(seed)
    supervised_blocks: list[pd.DataFrame] = []
    input_only_blocks: list[pd.DataFrame] = []
    output_only_blocks: list[pd.DataFrame] = []
    test_blocks: list[pd.DataFrame] = []

    for _, group in df.groupby("cluster"):
        n_samples = len(group)
        n_supervised = max(1, int(np.floor(supervised_ratio * n_samples)))
        n_output_only = int(np.floor(output_only_ratio * n_samples))
        if n_supervised + n_output_only >= n_samples:
            n_supervised = max(1, n_samples - 2)
            n_output_only = 1

        shuffled = group.sample(
            frac=1,
            random_state=int(rng.integers(0, 2**32)),
        ).reset_index(drop=True)

        supervised = shuffled.iloc[:n_supervised]
        output_only = shuffled.iloc[n_supervised : n_supervised + n_output_only]
        remainder = shuffled.iloc[n_supervised + n_output_only :]

        if mode == "transductive":
            # The transductive setting evaluates on the same pool that remains available at inference time.
            input_only = remainder
            test = remainder
        else:
            n_test = max(1, int(np.floor(test_frac * len(remainder)))) if len(remainder) else 0
            test = remainder.iloc[:n_test]
            input_only = remainder.iloc[n_test:]

        supervised_blocks.append(supervised)
        output_only_blocks.append(output_only)
        input_only_blocks.append(input_only)
        test_blocks.append(test)

    supervised_df = pd.concat(supervised_blocks, ignore_index=True)
    input_only_df = pd.concat(input_only_blocks, ignore_index=True)
    output_only_df = pd.concat(output_only_blocks, ignore_index=True)
    test_df = pd.concat(test_blocks, ignore_index=True)

    input_pool = pd.concat([input_only_df, supervised_df], ignore_index=True)
    output_pool = pd.concat([output_only_df, supervised_df], ignore_index=True)
    return supervised_df, input_only_df, output_only_df, test_df, input_pool, output_pool


def fit_constrained_kmeans(
    values: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    **kwargs,
) -> KMeansConstrained:
    """Fit balanced KMeans on an embedding matrix.

    `values` is a two-dimensional array of feature vectors. The cluster sizes
    are constrained to differ by at most one sample, matching the balanced
    clustering assumption used throughout the BC sweeps.
    """
    _require_kmeans_constrained()
    n_samples = values.shape[0]
    size_min = n_samples // n_clusters
    size_max = int(np.ceil(n_samples / n_clusters))
    return KMeansConstrained(
        n_clusters=n_clusters,
        size_min=size_min,
        size_max=size_max,
        random_state=random_state,
        **kwargs,
    ).fit(values)


def perform_size_constrained_clustering(
    input_pool: pd.DataFrame,
    output_pool: pd.DataFrame,
    n_clusters: int,
    input_column: str = "x",
    output_column: str = "yv",
    random_state: int = 42,
    input_kmeans_kwargs: dict | None = None,
    output_kmeans_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, KMeansConstrained, KMeansConstrained]:
    """Cluster the input and output marginals independently.

    `input_pool` and `output_pool` are the two BC marginals after supervised
    pairs have been appended. The function returns input labels, output labels,
    and the two fitted constrained KMeans objects.
    """
    input_kmeans_kwargs = input_kmeans_kwargs or {}
    output_kmeans_kwargs = output_kmeans_kwargs or {}

    input_values = np.vstack(input_pool[input_column].values)
    output_values = np.vstack(output_pool[output_column].values)

    input_kmeans = fit_constrained_kmeans(
        input_values,
        n_clusters=n_clusters,
        random_state=random_state,
        **input_kmeans_kwargs,
    )
    output_kmeans = fit_constrained_kmeans(
        output_values,
        n_clusters=n_clusters,
        random_state=random_state,
        **output_kmeans_kwargs,
    )

    return input_kmeans.labels_, output_kmeans.labels_, input_kmeans, output_kmeans


def assign_by_centroids(values: np.ndarray, kmeans: KMeansConstrained) -> np.ndarray:
    """Assign each point to its nearest fitted centroid."""
    if len(values) == 0:
        return np.array([], dtype=int)
    distances = pairwise_distances(values, kmeans.cluster_centers_)
    return distances.argmin(axis=1)


def clustering_quality_metrics(
    input_pool: pd.DataFrame,
    input_clusters: np.ndarray,
    output_pool: pd.DataFrame,
    output_clusters: np.ndarray,
    input_column: str = "x",
    output_column: str = "yv",
) -> dict[str, float]:
    """Compute the standard clustering metrics reported in the experiments."""
    input_values = np.vstack(input_pool[input_column].values)
    output_values = np.vstack(output_pool[output_column].values)
    return {
        "input_silhouette": silhouette_score(input_values, input_clusters),
        "input_davies_bouldin": davies_bouldin_score(input_values, input_clusters),
        "output_silhouette": silhouette_score(output_values, output_clusters),
        "output_davies_bouldin": davies_bouldin_score(output_values, output_clusters),
    }


def decision_vector(sample: pd.DataFrame, x_column: str, y_column: str, dim: int = 5) -> np.ndarray:
    """Return the majority-vote bridge from input-cluster ids to output-cluster ids."""
    association = np.zeros((dim, dim), dtype=int)
    for i in range(dim):
        for j in range(dim):
            association[i, j] = np.sum((sample[x_column] == i) & (sample[y_column] == j))

    decision = np.zeros(dim, dtype=int)
    for i in range(dim):
        decision[i] = np.argmax(association[i, :])
    return decision


def build_decision_matrix(
    supervised_samples: pd.DataFrame,
    input_clusters: np.ndarray,
    output_clusters: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Estimate the BC bridge from the supervised pairs only.

    The experiment code appends the supervised pairs to the end of both
    clustering pools, so the final `n_supervised` labels correspond exactly to
    those paired examples.
    """
    n_supervised = len(supervised_samples)
    supervised_block = supervised_samples.copy()
    supervised_block["x_cluster"] = input_clusters[-n_supervised:]
    supervised_block["y_cluster"] = output_clusters[-n_supervised:]
    return decision_vector(supervised_block, x_column="x_cluster", y_column="y_cluster", dim=n_clusters)


def build_true_decision_vector(
    input_pool: pd.DataFrame,
    output_pool: pd.DataFrame,
    input_clusters: np.ndarray,
    output_clusters: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Build the oracle bridge by majority-voting with the full labelled pools."""
    input_annotated = input_pool.copy()
    input_annotated["x_cluster"] = input_clusters
    output_annotated = output_pool.copy()
    output_annotated["y_cluster"] = output_clusters

    input_to_semantic = (
        input_annotated.groupby("x_cluster")["cluster"]
        .agg(lambda values: values.value_counts().idxmax())
        .to_dict()
    )
    output_to_semantic = (
        output_annotated.groupby("y_cluster")["cluster"]
        .agg(lambda values: values.value_counts().idxmax())
        .to_dict()
    )
    semantic_to_output = {semantic_cluster: output_cluster for output_cluster, semantic_cluster in output_to_semantic.items()}

    oracle = np.full(n_clusters, -1, dtype=int)
    for cluster_id in range(n_clusters):
        semantic_cluster = input_to_semantic.get(cluster_id)
        if semantic_cluster is not None:
            oracle[cluster_id] = semantic_to_output.get(semantic_cluster, -1)
    return oracle


def compute_cluster_centroids(
    output_pool: pd.DataFrame,
    output_clusters: np.ndarray,
    n_clusters: int,
    vector_column: str = "yv",
    text_column: str = "y",
) -> tuple[np.ndarray, list[str]]:
    """Compute output-cluster centroids and nearest prototype texts.

    The returned centroid matrix is the object used for BC prediction in the
    output space; the companion text list provides a readable prototype for each
    output cluster when the target modality is text.
    """
    if output_pool.empty:
        raise ValueError("output_pool must contain at least one row")

    annotated = output_pool.copy()
    annotated["cluster_id"] = output_clusters
    vector_shape = annotated[vector_column].iloc[0].shape

    centroids: list[np.ndarray] = []
    text_prototypes: list[str] = []
    for cluster_id in range(n_clusters):
        cluster_rows = annotated[annotated["cluster_id"] == cluster_id]
        if len(cluster_rows):
            vectors = np.stack(cluster_rows[vector_column].values)
            centroid = vectors.mean(axis=0)
            prototype_index = np.linalg.norm(vectors - centroid, axis=1).argmin()
            prototype_text = cluster_rows[text_column].values[prototype_index]
        else:
            centroid = np.zeros(vector_shape)
            prototype_text = ""
        centroids.append(centroid)
        text_prototypes.append(prototype_text)

    return np.asarray(centroids), text_prototypes


def perform_bridge_inference(
    inference_samples: pd.DataFrame,
    input_clusters: Sequence[int] | np.ndarray,
    decision: np.ndarray,
    centroids: np.ndarray,
    text_prototypes: Sequence[str],
    input_cluster_column: str = "x_cluster",
    predicted_cluster_column: str = "predicted_y_cluster",
    predicted_vector_column: str = "predicted_yv",
    predicted_text_column: str = "predicted_text",
) -> pd.DataFrame:
    """Apply a learned bridge to inference samples.

    `input_clusters` are the inferred cluster ids for `inference_samples`,
    `decision` maps those ids into output-cluster ids, and `centroids` /
    `text_prototypes` define the predicted output representatives.
    """
    inference = inference_samples.copy()
    input_clusters = np.asarray(input_clusters, dtype=int)
    predicted_clusters = decision[input_clusters]

    inference[input_cluster_column] = input_clusters
    inference[predicted_cluster_column] = predicted_clusters
    inference[predicted_vector_column] = list(centroids[predicted_clusters]) if len(predicted_clusters) else []

    text_array = np.asarray(text_prototypes, dtype=object)
    inference[predicted_text_column] = text_array[predicted_clusters].tolist() if len(predicted_clusters) else []
    return inference
