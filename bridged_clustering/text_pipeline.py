"""Shared image-text experiment runner.

This module keeps the dataset scripts thin: it calls the core Bridged
Clustering primitives, evaluates the baseline regressors, and returns metrics in
the legacy format expected by the analysis notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

from bridged_clustering._baseline_loader import load_baseline_regressors
from bridged_clustering.core import (
    assign_by_centroids,
    build_decision_matrix,
    build_true_decision_vector,
    compute_cluster_centroids,
    decision_vector,
    fit_constrained_kmeans,
    perform_bridge_inference,
    perform_size_constrained_clustering,
    split_by_cluster,
)
from bridged_clustering.structures import PreparedTextCorpus, TextExperimentSpec
from bridged_clustering.text import (
    embeddings_to_nearest_texts,
    evaluate_candidate_predictions,
    evaluate_regression_loss,
    knn_text_regression,
    wrap_text_baseline,
)

_TEXT_PIPELINE_BASELINE_NAMES: tuple[str, ...] = (
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


def _score_forward_predictions(
    predictions: np.ndarray,
    predicted_texts: list[str],
    test_df: pd.DataFrame,
    corpus: PreparedTextCorpus,
) -> tuple[float, float]:
    del predicted_texts
    if corpus.uses_candidate_alignment:
        assert corpus.candidate_map is not None
        assert corpus.candidate_id_column is not None
        return evaluate_candidate_predictions(
            predictions,
            test_df[corpus.candidate_id_column].values,
            corpus.candidate_map,
        )
    actuals = np.vstack(test_df["yv"].values)
    return evaluate_regression_loss(predictions, actuals)


def _prepare_transport_frames(
    supervised_df: pd.DataFrame,
    input_only_df: pd.DataFrame,
    output_only_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    input_pool = pd.concat([input_only_df, supervised_df], ignore_index=True).rename(
        columns={"x": "morph_coordinates"},
    )
    output_pool = pd.concat([output_only_df, supervised_df], ignore_index=True).rename(
        columns={"yv": "gene_coordinates"},
    )
    supervised = supervised_df.rename(columns={"x": "morph_coordinates", "yv": "gene_coordinates"})
    test = test_df.rename(columns={"x": "morph_coordinates", "yv": "gene_coordinates"})
    return input_pool, output_pool, supervised, test


def run_forward_text_experiment(
    df: pd.DataFrame,
    *,
    corpus: PreparedTextCorpus,
    spec: TextExperimentSpec,
    supervised_ratio: float = 0.05,
    output_only_ratio: float = 0.5,
    K: int = 100,
    knn_neighbors: int = 10,
    seed: int | None = None,
    mode: str = "transductive",
) -> dict[str, dict]:
    baselines = load_baseline_regressors(_TEXT_PIPELINE_BASELINE_NAMES)
    supervised_df, input_only_df, output_only_df, test_df, input_pool, output_pool = split_by_cluster(
        df,
        supervised_ratio,
        output_only_ratio,
        K,
        seed,
        mode=mode,
        test_frac=0.2,
    )

    input_clusters, output_clusters, input_kmeans, _ = perform_size_constrained_clustering(
        input_pool,
        output_pool,
        K,
    )
    decision = build_decision_matrix(supervised_df, input_clusters, output_clusters, K)
    oracle = build_true_decision_vector(input_pool, output_pool, input_clusters, output_clusters, K)
    centroids, prototype_texts = compute_cluster_centroids(output_pool, output_clusters, K)

    test_inputs = (
        np.vstack(test_df["x"].values)
        if len(test_df)
        else np.zeros((0, np.vstack(input_pool["x"].values).shape[1]))
    )
    test_input_clusters = assign_by_centroids(test_inputs, input_kmeans)
    bridge_outputs = perform_bridge_inference(
        test_df,
        test_input_clusters,
        decision,
        centroids,
        prototype_texts,
    )

    bkm_predictions = np.vstack(bridge_outputs["predicted_yv"].values)
    bkm_mae, bkm_mse = _score_forward_predictions(
        bkm_predictions,
        bridge_outputs["predicted_text"].tolist(),
        test_df,
        corpus,
    )

    knn_predictions, _, knn_texts, _ = knn_text_regression(supervised_df, test_df, knn_neighbors)
    knn_mae, knn_mse = _score_forward_predictions(knn_predictions, knn_texts, test_df, corpus)

    baseline_metrics: dict[str, dict[str, float]] = {
        "BKM": {"MAE": bkm_mae, "MSE": bkm_mse},
        "KNN": {"MAE": knn_mae, "MSE": knn_mse},
    }

    for model_name, baseline_fn in (
        ("FixMatch", baselines["fixmatch_regression"]),
        ("Laplacian RLS", baselines["laprls_regression"]),
        ("TSVR", baselines["tsvr_regression"]),
        ("TNNR", baselines["tnnr_regression"]),
        ("UCVME", baselines["ucvme_regression"]),
        ("GCN", baselines["gcn_regression"]),
    ):
        predictions, _, actual_texts, predicted_texts = wrap_text_baseline(
            baseline_fn,
            supervised_df,
            input_only_df,
            test_df,
        )
        del actual_texts
        mae, mse = _score_forward_predictions(predictions, predicted_texts, test_df, corpus)
        baseline_metrics[model_name] = {"MAE": mae, "MSE": mse}

    input_transport, output_transport, supervised_transport, test_transport = _prepare_transport_frames(
        supervised_df,
        input_only_df,
        output_only_df,
        test_df,
    )
    reference_embeddings = np.vstack(supervised_df["yv"])
    reference_texts = supervised_df["y"].tolist()

    kmm_predictions, _ = baselines["kernel_mean_matching_regression"](
        image_df=input_transport,
        gene_df=output_transport,
        supervised_df=supervised_transport,
        inference_df=test_transport,
        **spec.forward_transport.kmm,
    )
    kmm_texts = embeddings_to_nearest_texts(kmm_predictions, reference_embeddings, reference_texts)
    kmm_mae, kmm_mse = _score_forward_predictions(kmm_predictions, kmm_texts, test_df, corpus)
    baseline_metrics["KMM"] = {"MAE": kmm_mae, "MSE": kmm_mse}

    em_predictions, _ = baselines["em_regression"](
        supervised_df=supervised_transport,
        image_df=input_transport,
        gene_df=output_transport,
        inference_df=test_transport,
        n_components=K,
        **spec.forward_transport.em,
    )
    em_texts = embeddings_to_nearest_texts(em_predictions, reference_embeddings, reference_texts)
    em_mae, em_mse = _score_forward_predictions(em_predictions, em_texts, test_df, corpus)
    baseline_metrics["EM"] = {"MAE": em_mae, "MSE": em_mse}

    eot_predictions, _ = baselines["eot_barycentric_regression"](
        supervised_df=supervised_transport,
        image_df=input_transport,
        gene_df=output_transport,
        inference_df=test_transport,
        **spec.forward_transport.eot,
    )
    eot_texts = embeddings_to_nearest_texts(eot_predictions, reference_embeddings, reference_texts)
    eot_mae, eot_mse = _score_forward_predictions(eot_predictions, eot_texts, test_df, corpus)
    baseline_metrics["EOT"] = {"MAE": eot_mae, "MSE": eot_mse}

    gw_predictions, _ = baselines["gw_metric_alignment_regression"](
        supervised_df=supervised_transport,
        image_df=input_transport,
        gene_df=output_transport,
        inference_df=test_transport,
        **spec.forward_transport.gw,
    )
    gw_texts = embeddings_to_nearest_texts(gw_predictions, reference_embeddings, reference_texts)
    gw_mae, gw_mse = _score_forward_predictions(gw_predictions, gw_texts, test_df, corpus)
    baseline_metrics["GW"] = {"MAE": gw_mae, "MSE": gw_mse}

    return {
        "clustering": {
            "AMI_X": adjusted_mutual_info_score(input_pool["cluster"], input_clusters),
            "AMI_Y": adjusted_mutual_info_score(output_pool["cluster"], output_clusters),
            "Bridging Accuracy": np.mean(decision == oracle),
        },
        "regression": baseline_metrics,
    }


def run_reversed_text_experiment(
    df: pd.DataFrame,
    *,
    spec: TextExperimentSpec,
    supervised_ratio: float = 0.05,
    output_only_ratio: float = 0.5,
    K: int = 100,
    knn_neighbors: int = 10,
    seed: int | None = None,
    mode: str = "transductive",
) -> dict[str, dict]:
    baselines = load_baseline_regressors(_TEXT_PIPELINE_BASELINE_NAMES)
    supervised_df, input_only_df, output_only_df, test_df, input_pool, output_pool = split_by_cluster(
        df,
        supervised_ratio,
        output_only_ratio,
        K,
        seed,
        mode=mode,
        test_frac=0.2,
    )

    text_values = np.vstack(input_pool["yv"].values)
    text_kmeans = fit_constrained_kmeans(text_values, K, random_state=42, **spec.reverse_text_kmeans)
    text_clusters = text_kmeans.labels_

    image_values = np.vstack(output_pool["x"].values)
    image_kmeans = fit_constrained_kmeans(image_values, K, random_state=42, **spec.reverse_image_kmeans)
    image_clusters = image_kmeans.labels_

    supervised_block = supervised_df.copy()
    supervised_block["text_cluster"] = text_clusters[-len(supervised_df):]
    supervised_block["image_cluster"] = image_clusters[-len(supervised_df):]
    decision = decision_vector(supervised_block, "text_cluster", "image_cluster", dim=K)

    oracle = build_true_decision_vector(input_pool, output_pool, text_clusters, image_clusters, K)
    reversed_oracle = np.full(K, -1, dtype=int)
    for image_cluster, text_cluster in enumerate(oracle):
        if text_cluster >= 0:
            reversed_oracle[text_cluster] = image_cluster

    image_centroids = []
    annotated_output = output_pool.copy()
    annotated_output["image_cluster"] = image_clusters
    for cluster_id in range(K):
        cluster_rows = annotated_output[annotated_output["image_cluster"] == cluster_id]
        points = (
            np.stack(cluster_rows["x"].values)
            if len(cluster_rows)
            else np.zeros(image_values.shape[1])
        )
        image_centroids.append(points.mean(axis=0))
    image_centroids = np.vstack(image_centroids)

    if len(test_df):
        test_text_clusters = assign_by_centroids(np.vstack(test_df["yv"].values), text_kmeans)
    else:
        test_text_clusters = np.array([], dtype=int)

    inference = test_df.copy()
    inference["text_cluster"] = test_text_clusters
    inference["predicted_image_cluster"] = inference["text_cluster"].map(
        lambda cluster_id: decision[cluster_id] if cluster_id >= 0 else -1,
    )
    inference["pred_x"] = inference["predicted_image_cluster"].map(
        lambda cluster_id: image_centroids[cluster_id] if cluster_id >= 0 else np.zeros(image_centroids.shape[1]),
    )

    bridged_predictions = (
        np.vstack(inference["pred_x"].values)
        if len(inference)
        else np.zeros((0, image_centroids.shape[1]))
    )
    bridged_actuals = (
        np.vstack(inference["x"].values)
        if len(inference)
        else np.zeros((0, image_centroids.shape[1]))
    )

    supervised_reverse = supervised_df.rename(columns={"yv": "morph_coordinates", "x": "gene_coordinates"}).copy()
    input_only_reverse = input_only_df.rename(columns={"yv": "morph_coordinates", "x": "gene_coordinates"}).copy()
    test_reverse = test_df.rename(columns={"yv": "morph_coordinates", "x": "gene_coordinates"}).copy()

    if len(supervised_reverse) and len(test_reverse):
        knn = KNeighborsRegressor(n_neighbors=knn_neighbors)
        knn.fit(
            np.vstack(supervised_reverse["morph_coordinates"]),
            np.vstack(supervised_reverse["gene_coordinates"]),
        )
        knn_predictions = knn.predict(np.vstack(test_reverse["morph_coordinates"]))
    else:
        knn_predictions = np.zeros(
            (
                0,
                np.vstack(supervised_reverse["gene_coordinates"]).shape[1] if len(supervised_reverse) else 0,
            ),
        )
    test_actuals = np.vstack(test_reverse["gene_coordinates"]) if len(test_reverse) else np.zeros_like(knn_predictions)

    numeric_predictions = {
        "FixMatch": baselines["fixmatch_regression"](
            supervised_reverse,
            input_only_reverse,
            test_reverse,
            alpha_ema=0.999,
            batch_size=32,
            conf_threshold=0.05,
            lambda_u_max=0.5,
            lr=3e-4,
            rampup_length=10,
        ),
        "Laplacian RLS": baselines["laprls_regression"](
            supervised_reverse,
            input_only_reverse,
            test_reverse,
            gamma=0.1,
            k=20,
            lam=0.001,
            sigma=2.0,
        ),
        "TSVR": baselines["tsvr_regression"](
            supervised_reverse,
            input_only_reverse,
            test_reverse,
            C=10,
            epsilon=0.01,
            gamma="scale",
            self_training_frac=0.5,
        ),
        "TNNR": baselines["tnnr_regression"](
            supervised_reverse,
            input_only_reverse,
            test_reverse,
            beta=1.0,
            lr=0.001,
            rep_dim=128,
        ),
        "UCVME": baselines["ucvme_regression"](
            supervised_reverse,
            input_only_reverse,
            test_reverse,
            lr=3e-4,
            mc_T=5,
            w_unl=1.0,
        ),
        "GCN": baselines["gcn_regression"](
            supervised_reverse,
            input_only_reverse,
            test_reverse,
            hidden=32,
            dropout=0.1,
            lr=0.001,
        ),
    }

    reverse_gene_pool = input_pool.rename(columns={"yv": "gene_coordinates"}).copy()
    reverse_image_pool = output_pool.rename(columns={"x": "morph_coordinates"}).copy()
    reverse_supervised = supervised_df.rename(columns={"yv": "gene_coordinates", "x": "morph_coordinates"}).copy()
    reverse_test = test_df.rename(columns={"yv": "gene_coordinates", "x": "morph_coordinates"}).copy()

    kmm_predictions, kmm_actuals = baselines["reversed_kernel_mean_matching_regression"](
        gene_df=reverse_gene_pool,
        image_df=reverse_image_pool,
        supervised_df=reverse_supervised,
        inference_df=reverse_test,
        **spec.reverse_transport.kmm,
    )
    em_predictions, em_actuals = baselines["reversed_em_regression"](
        gene_df=reverse_gene_pool,
        image_df=reverse_image_pool,
        supervised_df=reverse_supervised,
        inference_df=reverse_test,
        n_components=K,
        **spec.reverse_transport.em,
    )
    eot_predictions, eot_actuals = baselines["reversed_eot_barycentric_regression"](
        gene_df=reverse_gene_pool,
        image_df=reverse_image_pool,
        supervised_df=reverse_supervised,
        inference_df=reverse_test,
        **spec.reverse_transport.eot,
    )
    gw_predictions, gw_actuals = baselines["reversed_gw_metric_alignment_regression"](
        gene_df=reverse_gene_pool,
        image_df=reverse_image_pool,
        supervised_df=reverse_supervised,
        inference_df=reverse_test,
        **spec.reverse_transport.gw,
    )

    def direct_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict[str, float]:
        return {
            "MAE": mean_absolute_error(actuals, predictions),
            "MSE": mean_squared_error(actuals, predictions),
        }

    baseline_metrics = {
        "BKM": direct_metrics(bridged_predictions, bridged_actuals),
        "KNN": direct_metrics(knn_predictions, test_actuals),
    }
    for model_name, (predictions, actuals) in numeric_predictions.items():
        baseline_metrics[model_name] = direct_metrics(predictions, actuals)
    baseline_metrics["KMM"] = direct_metrics(kmm_predictions, kmm_actuals)
    baseline_metrics["EM"] = direct_metrics(em_predictions, em_actuals)
    baseline_metrics["EOT"] = direct_metrics(eot_predictions, eot_actuals)
    baseline_metrics["GW"] = direct_metrics(gw_predictions, gw_actuals)

    return {
        "clustering": {
            "AMI_X": adjusted_mutual_info_score(input_pool["cluster"], text_clusters),
            "AMI_Y": adjusted_mutual_info_score(output_pool["cluster"], image_clusters),
            "Bridging Accuracy": np.mean(decision == reversed_oracle),
        },
        "regression": baseline_metrics,
    }
