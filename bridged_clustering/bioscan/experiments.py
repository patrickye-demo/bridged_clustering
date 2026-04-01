"""Forward and reversed BIOSCAN experiment routines.

These functions preserve the original BIOSCAN experiment semantics while
delegating bridge construction, encoder use, and baseline evaluation to nearby
helpers.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

from bridged_clustering.core import assign_by_centroids

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
from .config import ensure_rng
from .data import get_data_splits, load_dataset
from .encoders import EncoderSuite, encode_genes_for_samples, encode_images_for_samples, load_encoder_suite


_BASELINE_FUNCTION_NAMES: tuple[str, ...] = (
    "em_regression",
    "eot_barycentric_regression",
    "fixmatch_regression",
    "gcn_regression",
    "gw_metric_alignment_regression",
    "kernel_mean_matching_regression",
    "laprls_regression",
    "reversed_em_regression",
    "reversed_eot_barycentric_regression",
    "reversed_gw_metric_alignment_regression",
    "reversed_kernel_mean_matching_regression",
    "tnnr_regression",
    "tsvr_regression",
    "ucvme_regression",
)


def _resolve_encoder_suite(encoder_suite: EncoderSuite | None) -> EncoderSuite:
    return encoder_suite if encoder_suite is not None else load_encoder_suite()


def _load_baseline_regressors() -> dict[str, Any]:
    """Import the heavyweight baseline module only when an experiment actually runs."""
    try:
        baseline_module = import_module("baseline")
    except ModuleNotFoundError as exc:
        missing_name = f" '{exc.name}'" if getattr(exc, "name", None) else ""
        raise ModuleNotFoundError(
            "Missing dependency"
            f"{missing_name} while loading BIOSCAN baselines. "
            "Install packages from requirements.txt before running experiments.",
        ) from exc

    missing_functions = [name for name in _BASELINE_FUNCTION_NAMES if not hasattr(baseline_module, name)]
    if missing_functions:
        missing_str = ", ".join(sorted(missing_functions))
        raise AttributeError(f"baseline.py is missing expected BIOSCAN regressors: {missing_str}")

    return {name: getattr(baseline_module, name) for name in _BASELINE_FUNCTION_NAMES}


def _swap_coordinate_columns(df: pd.DataFrame) -> pd.DataFrame:
    swapped = df.copy()
    swapped["tmp"] = swapped["morph_coordinates"]
    swapped["morph_coordinates"] = swapped["gene_coordinates"]
    swapped["gene_coordinates"] = swapped.pop("tmp")
    return swapped


def run_experiment(
    csv_path: str,
    image_folder: str,
    n_families: int,
    n_samples: int = 50,
    supervised: float = 0.05,
    out_only: float = 0.5,
    rng: np.random.Generator | int | None = 42,
    mode: str = "transductive",
    test_frac: float = 0.2,
    *,
    encoder_suite: EncoderSuite | None = None,
) -> tuple[dict[str, float], dict[str, float], float, float, float]:
    """Run the forward BIOSCAN experiment: image -> DNA."""
    rng = ensure_rng(rng)
    encoders = _resolve_encoder_suite(encoder_suite)
    baselines = _load_baseline_regressors()

    df, images = load_dataset(csv_path, image_folder, n_families, n_samples, rng=rng)
    supervised_samples, input_only_df, gene_only_df, test_df = get_data_splits(
        df,
        supervised=supervised,
        out_only=out_only,
        rng=rng,
        mode=mode,
        test_frac=test_frac,
    )

    supervised_samples = encode_genes_for_samples(supervised_samples, encoders.barcode_tokenizer, encoders.barcode_model)
    gene_only_df = encode_genes_for_samples(gene_only_df, encoders.barcode_tokenizer, encoders.barcode_model)
    test_df = encode_genes_for_samples(test_df, encoders.barcode_tokenizer, encoders.barcode_model)
    gene_plus_supervised = pd.concat([gene_only_df, supervised_samples], axis=0)

    supervised_samples = encode_images_for_samples(supervised_samples, images, encoders.image_model, encoders.image_transform)
    input_only_df = encode_images_for_samples(input_only_df, images, encoders.image_model, encoders.image_transform)
    test_df = encode_images_for_samples(test_df, images, encoders.image_model, encoders.image_transform)
    input_plus_supervised = pd.concat([input_only_df, supervised_samples], axis=0)

    image_kmeans, gene_kmeans, _, gene_features, image_clusters, gene_clusters = perform_clustering(
        input_plus_supervised,
        gene_plus_supervised,
        images,
        encoders.image_model,
        encoders.image_transform,
        encoders.barcode_tokenizer,
        encoders.barcode_model,
        n_families,
    )

    ami_image = adjusted_mutual_info_score(image_clusters, input_plus_supervised["family"].values)
    ami_gene = adjusted_mutual_info_score(gene_clusters, gene_plus_supervised["family"].values)
    print(f"Adjusted Mutual Information (Image): {ami_image}")
    print(f"Adjusted Mutual Information (Gene): {ami_gene}")

    input_plus_supervised = input_plus_supervised.copy()
    gene_plus_supervised = gene_plus_supervised.copy()
    input_plus_supervised["image_cluster"] = image_clusters
    gene_plus_supervised["gene_cluster"] = gene_clusters

    true_decision = build_true_decision_vector(input_plus_supervised, gene_plus_supervised, n_families)
    decision_matrix = build_decision_matrix(supervised_samples, image_clusters, gene_clusters, n_families)
    decision_accuracy = np.mean(true_decision == decision_matrix)
    print(f"Decision: {decision_matrix}")
    print(f"Oracle Decision: {true_decision}")
    print(f"Decision Accuracy: {decision_accuracy}")

    centroids = compute_gene_centroids(gene_plus_supervised, gene_features, gene_clusters, n_families)
    test_image_features = (
        np.vstack(test_df["morph_coordinates"].values)
        if len(test_df)
        else np.zeros((0, len(supervised_samples["morph_coordinates"].iloc[0])))
    )
    test_image_clusters = assign_by_centroids(test_image_features, image_kmeans)

    inference_samples_bc = perform_inference(
        test_df.copy(),
        test_image_clusters,
        encoders.barcode_tokenizer,
        encoders.barcode_model,
        image_kmeans,
        decision_matrix,
        centroids,
    )
    bkm_predictions, bkm_actuals = bkm_regression(inference_samples_bc)

    knn_predictions, knn_actuals = knn_regression(
        supervised_samples,
        test_df,
        n_neighbors=max(1, int(n_samples * supervised)),
    )
    print("starting fixmatch")
    fixmatch_predictions, fixmatch_actuals = baselines["fixmatch_regression"](
        supervised_samples,
        input_only_df,
        test_df,
        batch_size=32,
        lr=1e-3,
        alpha_ema=0.99,
        lambda_u_max=0.5,
        rampup_length=10,
        conf_threshold=0.1,
    )
    print("starting laprls")
    lap_predictions, lap_actuals = baselines["laprls_regression"](
        supervised_samples,
        input_only_df,
        test_df,
        lam=0.1,
        gamma=0.001,
        k=20,
        sigma=2.0,
    )
    print("starting tsvr")
    tsvr_predictions, tsvr_actuals = baselines["tsvr_regression"](
        supervised_samples,
        input_only_df,
        test_df,
        C=1.0,
        epsilon=0.01,
        self_training_frac=0.2,
        gamma="scale",
    )
    print("starting tnnr")
    tnnr_predictions, tnnr_actuals = baselines["tnnr_regression"](
        supervised_samples,
        input_only_df,
        test_df,
        rep_dim=128,
        beta=1.0,
        lr=0.0001,
    )
    print("starting ucvme")
    ucv_predictions, ucv_actuals = baselines["ucvme_regression"](
        supervised_samples,
        input_only_df,
        test_df,
        mc_T=5,
        lr=0.001,
        w_unl=10,
    )
    print("starting gcn")
    gcn_predictions, gcn_actuals = baselines["gcn_regression"](
        supervised_samples,
        input_only_df,
        test_df,
        hidden=32,
        dropout=0.1,
        lr=0.003,
    )
    print("starting kernel mean matching baseline")
    kmm_predictions, kmm_actuals = baselines["kernel_mean_matching_regression"](
        image_df=input_plus_supervised,
        gene_df=gene_plus_supervised,
        supervised_df=supervised_samples,
        inference_df=test_df,
        alpha=0.1,
        kmm_B=100,
        kmm_eps=0.001,
        sigma=1.0,
    )
    print("starting em regression")
    em_predictions, em_actuals = baselines["em_regression"](
        supervised_df=supervised_samples,
        image_df=input_plus_supervised,
        gene_df=gene_plus_supervised,
        inference_df=test_df,
        n_components=n_families,
        eps=0.0001,
        max_iter=2000,
        tol=0.0001,
    )
    print("starting eot barycentric regression")
    eot_predictions, eot_actuals = baselines["eot_barycentric_regression"](
        supervised_df=supervised_samples,
        image_df=input_plus_supervised,
        gene_df=gene_plus_supervised,
        inference_df=test_df,
        max_iter=2000,
        ridge_alpha=0.01,
        eps=1,
        tol=1e-07,
    )
    print("starting gw metric alignment regression")
    gw_predictions, gw_actuals = baselines["gw_metric_alignment_regression"](
        supervised_df=supervised_samples,
        image_df=input_plus_supervised,
        gene_df=gene_plus_supervised,
        inference_df=test_df,
        max_iter=2000,
        tol=1e-07,
    )

    errors = {
        "BKM": evaluate_loss(bkm_predictions, bkm_actuals)[0],
        "KNN": evaluate_loss(knn_predictions, knn_actuals)[0],
        "FixMatch": evaluate_loss(fixmatch_predictions, fixmatch_actuals)[0],
        "Laplacian RLS": evaluate_loss(lap_predictions, lap_actuals)[0],
        "TSVR": evaluate_loss(tsvr_predictions, tsvr_actuals)[0],
        "TNNR": evaluate_loss(tnnr_predictions, tnnr_actuals)[0],
        "UCVME": evaluate_loss(ucv_predictions, ucv_actuals)[0],
        "GCN": evaluate_loss(gcn_predictions, gcn_actuals)[0],
        "Kernel Mean Matching": evaluate_loss(kmm_predictions, kmm_actuals)[0],
        "EM": evaluate_loss(em_predictions, em_actuals)[0],
        "EOT": evaluate_loss(eot_predictions, eot_actuals)[0],
        "GW": evaluate_loss(gw_predictions, gw_actuals)[0],
    }
    mses = {
        "BKM": evaluate_loss(bkm_predictions, bkm_actuals)[1],
        "KNN": evaluate_loss(knn_predictions, knn_actuals)[1],
        "FixMatch": evaluate_loss(fixmatch_predictions, fixmatch_actuals)[1],
        "Laplacian RLS": evaluate_loss(lap_predictions, lap_actuals)[1],
        "TSVR": evaluate_loss(tsvr_predictions, tsvr_actuals)[1],
        "TNNR": evaluate_loss(tnnr_predictions, tnnr_actuals)[1],
        "UCVME": evaluate_loss(ucv_predictions, ucv_actuals)[1],
        "GCN": evaluate_loss(gcn_predictions, gcn_actuals)[1],
        "Kernel Mean Matching": evaluate_loss(kmm_predictions, kmm_actuals)[1],
        "EM": evaluate_loss(em_predictions, em_actuals)[1],
        "EOT": evaluate_loss(eot_predictions, eot_actuals)[1],
        "GW": evaluate_loss(gw_predictions, gw_actuals)[1],
    }

    for label, value in errors.items():
        print(f"{label} Error: {value}")

    return errors, mses, ami_image, ami_gene, decision_accuracy


def run_reversed_experiment(
    csv_path: str,
    image_folder: str,
    n_families: int,
    n_samples: int = 50,
    supervised: float = 0.05,
    out_only: float = 0.5,
    knn_neighbors: int | None = None,
    rng: np.random.Generator | int | None = 42,
    mode: str = "transductive",
    test_frac: float = 0.2,
    *,
    encoder_suite: EncoderSuite | None = None,
) -> tuple[dict[str, float], dict[str, float], float, float, float]:
    """Run the reversed BIOSCAN experiment: DNA -> image."""
    rng = ensure_rng(rng)
    encoders = _resolve_encoder_suite(encoder_suite)
    baselines = _load_baseline_regressors()
    if knn_neighbors is None:
        knn_neighbors = max(1, int(n_samples * supervised))

    df, images = load_dataset(csv_path, image_folder, n_families, n_samples, rng=rng)
    supervised_df, input_only_df, gene_only_df, test_df = get_data_splits(
        df,
        supervised=supervised,
        out_only=out_only,
        rng=rng,
        mode=mode,
        test_frac=test_frac,
    )

    supervised_df = encode_genes_for_samples(supervised_df, encoders.barcode_tokenizer, encoders.barcode_model)
    gene_only_df = encode_genes_for_samples(gene_only_df, encoders.barcode_tokenizer, encoders.barcode_model)
    input_only_df = encode_genes_for_samples(input_only_df, encoders.barcode_tokenizer, encoders.barcode_model)
    test_df = encode_genes_for_samples(test_df, encoders.barcode_tokenizer, encoders.barcode_model)

    supervised_df = encode_images_for_samples(supervised_df, images, encoders.image_model, encoders.image_transform)
    input_only_df = encode_images_for_samples(input_only_df, images, encoders.image_model, encoders.image_transform)
    test_df = encode_images_for_samples(test_df, images, encoders.image_model, encoders.image_transform)

    gene_plus_supervised = pd.concat([gene_only_df, supervised_df], axis=0)
    image_plus_supervised = pd.concat([input_only_df, supervised_df], axis=0)

    image_kmeans, gene_kmeans, image_features, _, image_clusters, gene_clusters = perform_clustering(
        image_plus_supervised,
        gene_plus_supervised,
        images,
        encoders.image_model,
        encoders.image_transform,
        encoders.barcode_tokenizer,
        encoders.barcode_model,
        n_families,
    )

    ami_gene = adjusted_mutual_info_score(gene_clusters, gene_plus_supervised["family"].values)
    ami_image = adjusted_mutual_info_score(image_clusters, image_plus_supervised["family"].values)

    gene_plus_supervised = gene_plus_supervised.copy()
    image_plus_supervised = image_plus_supervised.copy()
    gene_plus_supervised["gene_cluster"] = gene_clusters
    image_plus_supervised["image_cluster"] = image_clusters

    supervised_block = supervised_df.copy()
    supervised_block["gene_cluster"] = gene_clusters[-len(supervised_df):]
    supervised_block["image_cluster"] = image_clusters[-len(supervised_df):]
    decision_vector = decisionVector(
        supervised_block,
        morph_column="gene_cluster",
        gene_column="image_cluster",
        dim=n_families,
    )

    true_vector = build_true_decision_vector(image_plus_supervised, gene_plus_supervised, n_families)
    oracle_reverse = np.full(n_families, -1, dtype=int)
    for image_cluster, gene_cluster in enumerate(true_vector):
        if gene_cluster >= 0:
            oracle_reverse[gene_cluster] = image_cluster
    decision_accuracy = (decision_vector == oracle_reverse).mean()

    image_plus_supervised["morph_coordinates"] = image_features.tolist()
    image_centroids: list[np.ndarray] = []
    for cluster_id in range(n_families):
        cluster_rows = image_plus_supervised[image_plus_supervised["image_cluster"] == cluster_id]
        points = (
            np.stack(cluster_rows["morph_coordinates"].values)
            if (image_plus_supervised["image_cluster"] == cluster_id).any()
            else np.zeros(image_features.shape[1])
        )
        image_centroids.append(points.mean(axis=0))
    image_centroids = np.vstack(image_centroids)

    test_gene_features = (
        np.vstack(test_df["gene_coordinates"].values)
        if len(test_df)
        else np.zeros((0, gene_kmeans.cluster_centers_.shape[1]))
    )
    test_gene_clusters = assign_by_centroids(test_gene_features, gene_kmeans)

    test_df = test_df.copy()
    test_df["gene_cluster"] = test_gene_clusters
    test_df["pred_image_cluster"] = test_df["gene_cluster"].apply(
        lambda cluster_id: decision_vector[cluster_id] if cluster_id >= 0 else -1,
    )
    test_df["pred_morph_coordinates"] = test_df["pred_image_cluster"].apply(
        lambda cluster_id: image_centroids[cluster_id] if cluster_id >= 0 else np.zeros(image_centroids.shape[1]),
    )

    bridged_predictions = (
        np.vstack(test_df["pred_morph_coordinates"].values)
        if len(test_df)
        else np.zeros((0, image_features.shape[1]))
    )
    bridged_actuals = (
        np.vstack(test_df["morph_coordinates"].values)
        if len(test_df)
        else np.zeros((0, image_features.shape[1]))
    )

    x_train = np.vstack(supervised_df["gene_coordinates"].values)
    y_train = np.vstack(supervised_df["morph_coordinates"].values)
    x_test = np.vstack(test_df["gene_coordinates"].values)
    y_test = np.vstack(test_df["morph_coordinates"].values)

    knn = KNeighborsRegressor(n_neighbors=knn_neighbors)
    knn.fit(x_train, y_train)
    knn_predictions = knn.predict(x_test)

    kmm_predictions, kmm_actuals = baselines["reversed_kernel_mean_matching_regression"](
        gene_df=gene_plus_supervised,
        image_df=image_plus_supervised,
        supervised_df=supervised_df,
        inference_df=test_df,
        alpha=0.1,
        kmm_B=100,
        kmm_eps=0.001,
        sigma=1.0,
    )
    em_predictions, em_actuals = baselines["reversed_em_regression"](
        gene_df=gene_plus_supervised,
        image_df=image_plus_supervised,
        supervised_df=supervised_df,
        inference_df=test_df,
        n_components=n_families,
        eps=0.0001,
        max_iter=2000,
        tol=0.0001,
    )
    eot_predictions, eot_actuals = baselines["reversed_eot_barycentric_regression"](
        gene_df=gene_plus_supervised,
        image_df=image_plus_supervised,
        supervised_df=supervised_df,
        inference_df=test_df,
        max_iter=2000,
        eps=1,
        ridge_alpha=0.01,
        tol=1e-07,
    )
    gw_predictions, gw_actuals = baselines["reversed_gw_metric_alignment_regression"](
        gene_df=gene_plus_supervised,
        image_df=image_plus_supervised,
        supervised_df=supervised_df,
        inference_df=test_df,
        max_iter=2000,
        tol=1e-07,
    )

    supervised_reverse = _swap_coordinate_columns(supervised_df)
    input_only_reverse = _swap_coordinate_columns(input_only_df)
    test_reverse = _swap_coordinate_columns(test_df)

    fixmatch_predictions, fixmatch_actuals = baselines["fixmatch_regression"](
        supervised_reverse,
        input_only_reverse,
        test_reverse,
        alpha_ema=0.99,
        batch_size=32,
        conf_threshold=0.05,
        lambda_u_max=0.5,
        lr=1e-3,
        rampup_length=30,
    )
    lap_predictions, lap_actuals = baselines["laprls_regression"](
        supervised_reverse,
        input_only_reverse,
        test_reverse,
        lam=0.1,
        gamma=0.1,
        k=20,
        sigma=2.0,
    )
    tsvr_predictions, tsvr_actuals = baselines["tsvr_regression"](
        supervised_reverse,
        input_only_reverse,
        test_reverse,
        C=1.0,
        epsilon=0.01,
        self_training_frac=0.5,
        gamma="scale",
    )
    tnnr_predictions, tnnr_actuals = baselines["tnnr_regression"](
        supervised_reverse,
        input_only_reverse,
        test_reverse,
        rep_dim=128,
        beta=0.1,
        lr=0.0001,
    )
    ucv_predictions, ucv_actuals = baselines["ucvme_regression"](
        supervised_reverse,
        input_only_reverse,
        test_reverse,
        mc_T=5,
        lr=0.0003,
        w_unl=10,
    )
    gcn_predictions, gcn_actuals = baselines["gcn_regression"](
        supervised_reverse,
        input_only_reverse,
        test_reverse,
        hidden=32,
        dropout=0.0,
        lr=0.001,
    )

    def evals(predictions: np.ndarray, actuals: np.ndarray) -> tuple[float, float]:
        return mean_absolute_error(predictions, actuals), mean_squared_error(predictions, actuals)

    errors = {
        "BKM": evals(bridged_predictions, bridged_actuals)[0],
        "KNN": evals(knn_predictions, y_test)[0],
        "FixMatch": evals(fixmatch_predictions, fixmatch_actuals)[0],
        "Laplacian RLS": evals(lap_predictions, lap_actuals)[0],
        "TSVR": evals(tsvr_predictions, tsvr_actuals)[0],
        "TNNR": evals(tnnr_predictions, tnnr_actuals)[0],
        "UCVME": evals(ucv_predictions, ucv_actuals)[0],
        "GCN": evals(gcn_predictions, gcn_actuals)[0],
        "Kernel Mean Matching": evals(kmm_predictions, kmm_actuals)[0],
        "EM": evals(em_predictions, em_actuals)[0],
        "EOT": evals(eot_predictions, eot_actuals)[0],
        "GW": evals(gw_predictions, gw_actuals)[0],
    }
    mses = {
        "BKM": evals(bridged_predictions, bridged_actuals)[1],
        "KNN": evals(knn_predictions, y_test)[1],
        "FixMatch": evals(fixmatch_predictions, fixmatch_actuals)[1],
        "Laplacian RLS": evals(lap_predictions, lap_actuals)[1],
        "TSVR": evals(tsvr_predictions, tsvr_actuals)[1],
        "TNNR": evals(tnnr_predictions, tnnr_actuals)[1],
        "UCVME": evals(ucv_predictions, ucv_actuals)[1],
        "GCN": evals(gcn_predictions, gcn_actuals)[1],
        "Kernel Mean Matching": evals(kmm_predictions, kmm_actuals)[1],
        "EM": evals(em_predictions, em_actuals)[1],
        "EOT": evals(eot_predictions, eot_actuals)[1],
        "GW": evals(gw_predictions, gw_actuals)[1],
    }

    print(f"[Reversed] MSEs: {mses} | Decision Acc: {decision_accuracy:.2%}")
    return errors, mses, ami_gene, ami_image, decision_accuracy
