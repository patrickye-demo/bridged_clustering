"""Bridged Clustering helpers for the BIOSCAN modality pair."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

from .encoders import encode_genes, encode_images

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
            "k_means_constrained is required for BIOSCAN clustering. "
            "Install packages from requirements.txt before running experiments.",
        ) from _KMEANS_CONSTRAINED_IMPORT_ERROR


def _cluster_size_bounds(n_samples: int, n_clusters: int) -> tuple[int, int]:
    base_size = n_samples // n_clusters
    return base_size, base_size + (1 if n_samples % n_clusters else 0)


def perform_clustering(
    image_samples: pd.DataFrame,
    gene_samples: pd.DataFrame,
    image_paths: dict[str, str],
    image_model: Any,
    image_transform: Any,
    barcode_tokenizer: Any,
    barcode_model: Any,
    n_families: int,
) -> tuple[KMeansConstrained, KMeansConstrained, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit balanced KMeans on image and gene encodings."""
    _require_kmeans_constrained()
    image_features = encode_images(image_samples["processid"].values, image_paths, image_model, image_transform)
    size_min, size_max = _cluster_size_bounds(len(image_samples), n_families)
    image_kmeans = KMeansConstrained(
        n_clusters=n_families,
        size_min=size_min,
        size_max=size_max,
        random_state=42,
        max_iter=100,
    ).fit(image_features)
    image_clusters = image_kmeans.labels_

    gene_features = encode_genes(gene_samples["dna_barcode"].values, barcode_tokenizer, barcode_model)
    gene_size_min, gene_size_max = _cluster_size_bounds(len(gene_samples), n_families)
    gene_kmeans = KMeansConstrained(
        n_clusters=n_families,
        size_min=gene_size_min,
        size_max=gene_size_max,
        random_state=42,
        max_iter=100,
    ).fit(gene_features)
    gene_clusters = gene_kmeans.labels_

    return image_kmeans, gene_kmeans, image_features, gene_features, image_clusters, gene_clusters


def decisionVector(
    sample: pd.DataFrame,
    morph_column: str = "morph_cluster",
    gene_column: str = "gene_cluster",
    dim: int = 5,
) -> np.ndarray:
    """Compatibility wrapper for the original BIOSCAN decision vector implementation."""
    if morph_column not in sample.columns:
        raise KeyError(f"Column '{morph_column}' not found in the DataFrame.")
    if gene_column not in sample.columns:
        raise KeyError(f"Column '{gene_column}' not found in the DataFrame.")

    association = np.zeros((dim, dim), dtype=int)
    for i in range(dim):
        for j in range(dim):
            association[i, j] = np.sum((sample[morph_column] == i) & (sample[gene_column] == j))

    decision = np.zeros(dim, dtype=int)
    for i in range(dim):
        decision[i] = np.argmax(association[i, :])
    return decision


def build_true_decision_vector(img_df: pd.DataFrame, gene_df: pd.DataFrame, dim: int) -> np.ndarray:
    """Build the oracle image-to-gene cluster bridge using family labels."""
    image_to_family = (
        img_df.groupby("image_cluster")["family"]
        .agg(lambda values: values.value_counts().idxmax())
        .to_dict()
    )
    gene_to_family = (
        gene_df.groupby("gene_cluster")["family"]
        .agg(lambda values: values.value_counts().idxmax())
        .to_dict()
    )
    family_to_gene = {family: cluster_id for cluster_id, family in gene_to_family.items()}

    decision = np.full(dim, -1, dtype=int)
    for cluster_id in range(dim):
        family = image_to_family.get(cluster_id)
        decision[cluster_id] = family_to_gene.get(family, -1)
    return decision


def build_decision_matrix(
    supervised_samples: pd.DataFrame,
    image_clusters: np.ndarray,
    gene_clusters: np.ndarray,
    n_families: int,
) -> np.ndarray:
    """Build the learned bridge from the supervised BIOSCAN pairs."""
    n_supervised = len(supervised_samples)
    supervised_block = supervised_samples.copy()
    supervised_block["image_cluster"] = image_clusters[-n_supervised:]
    supervised_block["gene_cluster"] = gene_clusters[-n_supervised:]
    return decisionVector(
        supervised_block,
        morph_column="image_cluster",
        gene_column="gene_cluster",
        dim=n_families,
    )


def compute_gene_centroids(
    gene_samples: pd.DataFrame,
    gene_features: np.ndarray,
    gene_clusters: np.ndarray,
    n_families: int,
) -> np.ndarray:
    """Compute gene-cluster centroids used for bridged inference."""
    annotated = gene_samples.copy()
    annotated["gene_cluster"] = gene_clusters[: len(annotated)]
    annotated["gene_coordinates"] = gene_features.tolist()

    centroids: list[np.ndarray] = []
    for cluster_id in range(n_families):
        cluster_rows = annotated[annotated["gene_cluster"] == cluster_id]
        if len(cluster_rows):
            centroid = np.mean(np.stack(cluster_rows["gene_coordinates"].values), axis=0)
        else:
            centroid = np.zeros(gene_features.shape[1])
        centroids.append(centroid)
    return np.array(centroids)


def perform_inference(
    inference_samples: pd.DataFrame,
    image_clusters: np.ndarray,
    barcode_tokenizer: Any,
    barcode_model: Any,
    image_kmeans: Any,
    decision_matrix: np.ndarray,
    centroids: np.ndarray,
) -> pd.DataFrame:
    """Assign image clusters and emit bridged gene predictions."""
    del image_kmeans
    inference = inference_samples.copy()
    inference["image_cluster"] = image_clusters[: len(inference)]
    inference["predicted_gene_cluster"] = inference["image_cluster"].apply(lambda cluster_id: decision_matrix[cluster_id])
    inference["predicted_gene_coordinates"] = inference["predicted_gene_cluster"].apply(lambda cluster_id: centroids[cluster_id])
    inferred_gene_features = encode_genes(inference["dna_barcode"].values, barcode_tokenizer, barcode_model)
    inference["gene_coordinates"] = inferred_gene_features.tolist()
    return inference


def bkm_regression(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return bridged predictions and actual gene coordinates."""
    return np.array(df["predicted_gene_coordinates"].tolist()), np.array(df["gene_coordinates"].tolist())


def knn_regression(
    supervised_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_neighbors: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """KNN baseline for the forward BIOSCAN direction."""
    x_train = np.array(supervised_df["morph_coordinates"].tolist())
    y_train = np.array(supervised_df["gene_coordinates"].tolist())
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    predictions = knn.predict(np.array(test_df["morph_coordinates"].tolist()))
    actuals = np.array(test_df["gene_coordinates"].tolist())
    return predictions, actuals


def evaluate_loss(predictions: np.ndarray, actuals: np.ndarray) -> tuple[float, float]:
    """Return MAE and MSE for BIOSCAN predictions."""
    mae = mean_absolute_error(predictions, actuals)
    mse = mean_squared_error(predictions, actuals)
    return mae, mse
