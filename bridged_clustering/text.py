"""Shared helpers for the image-text Bridged Clustering experiments."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

from bridged_clustering._baseline_loader import load_baseline_regressors


TextCandidateMap = dict[int, dict[str, np.ndarray | list[str]]]

_TEXT_BASELINE_FUNCTION_NAMES: tuple[str, ...] = (
    "fixmatch_regression",
    "gcn_regression",
    "laprls_regression",
    "tnnr_regression",
    "tsvr_regression",
    "ucvme_regression",
)


def _text_baseline_runners() -> dict[str, Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], tuple[np.ndarray, np.ndarray]]]:
    baselines = load_baseline_regressors(_TEXT_BASELINE_FUNCTION_NAMES)
    return {
        "gcn_regression": lambda sup, ino, tst: baselines["gcn_regression"](
            sup,
            ino,
            tst,
            dropout=0.1,
            hidden=32,
            lr=0.001,
        ),
        "fixmatch_regression": lambda sup, ino, tst: baselines["fixmatch_regression"](
            sup,
            ino,
            tst,
            alpha_ema=0.999,
            batch_size=64,
            conf_threshold=0.1,
            lambda_u_max=0.5,
            lr=0.0003,
            rampup_length=30,
        ),
        "laprls_regression": lambda sup, ino, tst: baselines["laprls_regression"](
            sup,
            ino,
            tst,
            gamma=0.001,
            k=5,
            lam=0.1,
            sigma=2.0,
        ),
        "tsvr_regression": lambda sup, ino, tst: baselines["tsvr_regression"](
            sup,
            ino,
            tst,
            C=10,
            epsilon=0.01,
            gamma="scale",
            self_training_frac=0.1,
        ),
        "tnnr_regression": lambda sup, ino, tst: baselines["tnnr_regression"](
            sup,
            ino,
            tst,
            beta=0.1,
            lr=0.001,
            rep_dim=128,
        ),
        "ucvme_regression": lambda sup, ino, tst: baselines["ucvme_regression"](
            sup,
            ino,
            tst,
            lr=0.001,
            mc_T=5,
            w_unl=10,
        ),
    }


def build_candidate_map(
    df: pd.DataFrame,
    id_column: str,
    embedding_column: str = "yv",
    text_column: str = "y",
) -> TextCandidateMap:
    """Group multiple candidate texts per item for datasets like COCO and Flickr30k."""
    candidate_map: TextCandidateMap = {}
    for item_id, group in df.groupby(id_column):
        candidate_map[item_id] = {
            "embs": np.stack(group[embedding_column].apply(np.asarray)),
            "texts": group[text_column].tolist(),
        }
    return candidate_map


def align_predictions_to_candidates(
    predicted_embeddings: np.ndarray,
    item_ids: Sequence[int] | np.ndarray,
    candidate_map: TextCandidateMap,
) -> tuple[np.ndarray, list[str]]:
    """Align each predicted embedding to the closest candidate embedding for that item."""
    aligned_embeddings: list[np.ndarray] = []
    aligned_texts: list[str] = []

    for embedding, item_id in zip(predicted_embeddings, item_ids):
        entry = candidate_map[item_id]
        candidates = entry["embs"]
        candidate_index = np.linalg.norm(candidates - embedding, axis=1).argmin()
        aligned_embeddings.append(candidates[candidate_index])
        aligned_texts.append(entry["texts"][candidate_index])

    return np.vstack(aligned_embeddings), aligned_texts


def embeddings_to_nearest_texts(
    predicted_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    reference_texts: Sequence[str],
) -> list[str]:
    """Recover text prototypes by nearest-neighbor lookup in embedding space."""
    predicted_texts: list[str] = []
    for embedding in predicted_embeddings:
        nearest_index = np.linalg.norm(reference_embeddings - embedding, axis=1).argmin()
        predicted_texts.append(reference_texts[nearest_index])
    return predicted_texts


def evaluate_regression_loss(predictions: np.ndarray, actuals: np.ndarray) -> tuple[float, float]:
    """Return MAE and MSE using the original experiment convention."""
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    return mae, mse


def evaluate_candidate_predictions(
    predicted_embeddings: np.ndarray,
    item_ids: Sequence[int] | np.ndarray,
    candidate_map: TextCandidateMap,
) -> tuple[float, float]:
    """Evaluate predictions against the closest valid candidate for each example."""
    aligned_embeddings, _ = align_predictions_to_candidates(predicted_embeddings, item_ids, candidate_map)
    return evaluate_regression_loss(predicted_embeddings, aligned_embeddings)


def knn_text_regression(
    supervised_df: pd.DataFrame,
    inference_df: pd.DataFrame,
    n_neighbors: int = 10,
    input_column: str = "x",
    output_column: str = "yv",
    text_column: str = "y",
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """KNN baseline shared by the text experiments."""
    x_train = np.vstack(supervised_df[input_column].values)
    y_train = np.vstack(supervised_df[output_column].values)
    train_texts = supervised_df[text_column].tolist()

    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)

    x_test = np.vstack(inference_df[input_column].values)
    predicted_embeddings = knn.predict(x_test)
    actual_embeddings = np.vstack(inference_df[output_column].values)
    actual_texts = inference_df[text_column].tolist()
    predicted_texts = embeddings_to_nearest_texts(predicted_embeddings, y_train, train_texts)
    return predicted_embeddings, actual_embeddings, predicted_texts, actual_texts


def wrap_text_baseline(
    baseline_fn: Callable,
    supervised_df: pd.DataFrame,
    input_only_df: pd.DataFrame,
    test_df: pd.DataFrame,
    input_column: str = "x",
    output_column: str = "yv",
    text_column: str = "y",
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Run a numeric baseline on embeddings and recover text outputs via nearest-neighbor lookup."""
    supervised = supervised_df.rename(columns={input_column: "morph_coordinates", output_column: "gene_coordinates"})
    input_only = input_only_df.rename(columns={input_column: "morph_coordinates", output_column: "gene_coordinates"})
    test = test_df.rename(columns={input_column: "morph_coordinates", output_column: "gene_coordinates"})

    runner = _text_baseline_runners().get(baseline_fn.__name__)
    if runner is None:
        raise ValueError(f"Unknown baseline function: {baseline_fn.__name__}")

    predictions, actuals = runner(supervised, input_only, test)
    reference_embeddings = np.vstack(supervised["gene_coordinates"])
    reference_texts = supervised_df[text_column].tolist()
    predicted_texts = embeddings_to_nearest_texts(predictions, reference_embeddings, reference_texts)
    actual_texts = test_df[text_column].tolist()
    return predictions, actuals, actual_texts, predicted_texts
