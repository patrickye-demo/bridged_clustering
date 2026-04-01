"""Shared helpers for the Bridged Clustering experiments."""

from .core import (
    assign_by_centroids,
    build_decision_matrix,
    build_true_decision_vector,
    clustering_quality_metrics,
    compute_cluster_centroids,
    decision_vector,
    fit_constrained_kmeans,
    perform_bridge_inference,
    perform_size_constrained_clustering,
    split_by_cluster,
)
from .result_store import MetricCube
from .structures import MODEL_ORDER, PreparedTextCorpus, TextExperimentSpec, TransportSuiteSpec
from .text import (
    build_candidate_map,
    embeddings_to_nearest_texts,
    evaluate_candidate_predictions,
    evaluate_regression_loss,
    knn_text_regression,
    wrap_text_baseline,
)

__all__ = [
    "assign_by_centroids",
    "build_candidate_map",
    "build_decision_matrix",
    "build_true_decision_vector",
    "clustering_quality_metrics",
    "compute_cluster_centroids",
    "decision_vector",
    "embeddings_to_nearest_texts",
    "evaluate_candidate_predictions",
    "evaluate_regression_loss",
    "fit_constrained_kmeans",
    "knn_text_regression",
    "MODEL_ORDER",
    "MetricCube",
    "PreparedTextCorpus",
    "perform_bridge_inference",
    "perform_size_constrained_clustering",
    "split_by_cluster",
    "TextExperimentSpec",
    "TransportSuiteSpec",
    "wrap_text_baseline",
]
