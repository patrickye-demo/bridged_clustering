"""BIOSCAN-specific experiment stack."""

from .bridge import (
    bkm_regression,
    build_decision_matrix,
    build_true_decision_vector,
    compute_gene_centroids,
    decisionVector,
    evaluate_loss,
    knn_regression,
    perform_clustering,
    perform_inference,
)
from .config import BIOSCAN_MODEL_NAMES, BioscanGridSpec, BioscanPaths
from .data import get_data_splits, load_dataset, split_family_samples
from .encoders import (
    DEVICE,
    EncoderSuite,
    encode_genes,
    encode_genes_for_samples,
    encode_images,
    encode_images_for_samples,
    load_encoder_suite,
    load_pretrained_models,
)
from .experiments import run_experiment, run_reversed_experiment
from .grid import build_parser, main

__all__ = [
    "BIOSCAN_MODEL_NAMES",
    "DEVICE",
    "EncoderSuite",
    "BioscanGridSpec",
    "BioscanPaths",
    "bkm_regression",
    "build_decision_matrix",
    "build_parser",
    "build_true_decision_vector",
    "compute_gene_centroids",
    "decisionVector",
    "encode_genes",
    "encode_genes_for_samples",
    "encode_images",
    "encode_images_for_samples",
    "evaluate_loss",
    "get_data_splits",
    "knn_regression",
    "load_dataset",
    "load_encoder_suite",
    "load_pretrained_models",
    "main",
    "perform_clustering",
    "perform_inference",
    "run_experiment",
    "run_reversed_experiment",
    "split_family_samples",
]
