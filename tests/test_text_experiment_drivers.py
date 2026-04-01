from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bridged_clustering.structures import MODEL_ORDER, PreparedTextCorpus


def _synthetic_text_frame(num_clusters: int = 5, rows_per_cluster: int = 4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster in range(num_clusters):
        image_embedding = np.array([float(cluster * 10), float(cluster * 10 + 1)])
        text_embedding = np.array([float(cluster * 100 + 10), float(cluster * 100 + 11)])
        for row_index in range(rows_per_cluster):
            item_id = cluster * 100 + row_index
            rows.append(
                {
                    "cluster": cluster,
                    "x": image_embedding.copy(),
                    "yv": text_embedding.copy(),
                    "y": f"cluster-{cluster}-text-{row_index}",
                    "image_id": item_id,
                    "img_id": item_id,
                },
            )
    return pd.DataFrame(rows)


def _build_corpus(candidate_id_column: str | None) -> PreparedTextCorpus:
    frame = _synthetic_text_frame()
    kwargs: dict[str, object] = {}
    if candidate_id_column is not None:
        candidate_map = {
            int(row[candidate_id_column]): {
                "embs": np.stack([np.asarray(row["yv"])]),
                "texts": [str(row["y"])],
            }
            for _, row in frame.iterrows()
        }
        kwargs["candidate_map"] = candidate_map
        kwargs["candidate_id_column"] = candidate_id_column

    return PreparedTextCorpus(
        name="synthetic",
        frame=frame,
        cluster_sizes=frame["cluster"].value_counts(),
        **kwargs,
    )


def _contiguous_cluster_labels(cluster_values: pd.Series | np.ndarray) -> np.ndarray:
    unique_clusters = sorted(pd.Series(cluster_values).unique().tolist())
    mapping = {cluster_id: index for index, cluster_id in enumerate(unique_clusters)}
    return pd.Series(cluster_values).map(mapping).to_numpy(dtype=int)


def _make_semisupervised_baseline(name: str):
    def baseline(
        _supervised_df: pd.DataFrame,
        _input_only_df: pd.DataFrame,
        test_df: pd.DataFrame,
        **_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        actuals = np.vstack(test_df["gene_coordinates"].values)
        return actuals.copy(), actuals

    baseline.__name__ = name
    return baseline


def _make_forward_transport_baseline(name: str):
    def baseline(*, inference_df: pd.DataFrame, **_kwargs) -> tuple[np.ndarray, np.ndarray]:
        actuals = np.vstack(inference_df["gene_coordinates"].values)
        return actuals.copy(), actuals

    baseline.__name__ = name
    return baseline


def _make_reversed_transport_baseline(name: str):
    def baseline(*, inference_df: pd.DataFrame, **_kwargs) -> tuple[np.ndarray, np.ndarray]:
        actuals = np.vstack(inference_df["morph_coordinates"].values)
        return actuals.copy(), actuals

    baseline.__name__ = name
    return baseline


def _patch_text_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    text_pipeline_module = importlib.import_module("bridged_clustering.text_pipeline")
    text_module = importlib.import_module("bridged_clustering.text")

    baseline_mapping = {
        "em_regression": _make_forward_transport_baseline("em_regression"),
        "eot_barycentric_regression": _make_forward_transport_baseline("eot_barycentric_regression"),
        "fixmatch_regression": _make_semisupervised_baseline("fixmatch_regression"),
        "gcn_regression": _make_semisupervised_baseline("gcn_regression"),
        "gw_metric_alignment_regression": _make_forward_transport_baseline("gw_metric_alignment_regression"),
        "kernel_mean_matching_regression": _make_forward_transport_baseline("kernel_mean_matching_regression"),
        "laprls_regression": _make_semisupervised_baseline("laprls_regression"),
        "reversed_em_regression": _make_reversed_transport_baseline("reversed_em_regression"),
        "reversed_eot_barycentric_regression": _make_reversed_transport_baseline("reversed_eot_barycentric_regression"),
        "reversed_gw_metric_alignment_regression": _make_reversed_transport_baseline(
            "reversed_gw_metric_alignment_regression",
        ),
        "reversed_kernel_mean_matching_regression": _make_reversed_transport_baseline(
            "reversed_kernel_mean_matching_regression",
        ),
        "tnnr_regression": _make_semisupervised_baseline("tnnr_regression"),
        "tsvr_regression": _make_semisupervised_baseline("tsvr_regression"),
        "ucvme_regression": _make_semisupervised_baseline("ucvme_regression"),
    }

    monkeypatch.setattr(
        text_pipeline_module,
        "load_baseline_regressors",
        lambda *_args, **_kwargs: baseline_mapping,
    )
    monkeypatch.setattr(
        text_module,
        "load_baseline_regressors",
        lambda *_args, **_kwargs: baseline_mapping,
    )

    def fake_perform_size_constrained_clustering(
        input_pool: pd.DataFrame,
        output_pool: pd.DataFrame,
        n_clusters: int,
        *_args,
        **_kwargs,
    ) -> tuple[np.ndarray, np.ndarray, SimpleNamespace, SimpleNamespace]:
        input_clusters = _contiguous_cluster_labels(input_pool["cluster"])
        output_clusters = _contiguous_cluster_labels(output_pool["cluster"])
        input_centers = np.vstack(
            [
                np.vstack(input_pool[input_pool["cluster"] == cluster_id]["x"].values).mean(axis=0)
                for cluster_id in sorted(input_pool["cluster"].unique().tolist())
            ],
        )
        output_centers = np.vstack(
            [
                np.vstack(output_pool[output_pool["cluster"] == cluster_id]["yv"].values).mean(axis=0)
                for cluster_id in sorted(output_pool["cluster"].unique().tolist())
            ],
        )
        assert len(input_centers) == n_clusters
        assert len(output_centers) == n_clusters
        return (
            input_clusters,
            output_clusters,
            SimpleNamespace(cluster_centers_=input_centers, labels_=input_clusters),
            SimpleNamespace(cluster_centers_=output_centers, labels_=output_clusters),
        )

    def fake_fit_constrained_kmeans(
        values: np.ndarray,
        n_clusters: int,
        *_args,
        **_kwargs,
    ) -> SimpleNamespace:
        first_coordinate = np.asarray(values)[:, 0]
        unique_values = sorted(np.unique(first_coordinate).tolist())
        assert len(unique_values) == n_clusters
        mapping = {value: index for index, value in enumerate(unique_values)}
        labels = np.asarray([mapping[value] for value in first_coordinate], dtype=int)
        centers = np.vstack([np.asarray(values)[labels == index].mean(axis=0) for index in range(n_clusters)])
        return SimpleNamespace(cluster_centers_=centers, labels_=labels)

    monkeypatch.setattr(
        text_pipeline_module,
        "perform_size_constrained_clustering",
        fake_perform_size_constrained_clustering,
    )
    monkeypatch.setattr(text_pipeline_module, "fit_constrained_kmeans", fake_fit_constrained_kmeans)


@pytest.mark.parametrize(
    ("module_name", "grid_function_name", "candidate_id_column", "reversed_direction", "expected_output_dir"),
    [
        ("coco", "run_coco_grid", "image_id", False, "results/100_coco_tran"),
        ("coco", "run_coco_grid", "image_id", True, "results/101_coco_rev_tran"),
        ("flick", "run_flickr_grid", "img_id", False, "results/104_flick_2_tran"),
        ("flick", "run_flickr_grid", "img_id", True, "results/105_flick_rev_2_tran"),
        ("wiki", "run_wiki_grid", None, False, "results/106_wiki_tran"),
        ("wiki", "run_wiki_grid", None, True, "results/107_wiki_reversed_tran"),
    ],
)
def test_text_experiment_grids_run_on_synthetic_corpora(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module_name: str,
    grid_function_name: str,
    candidate_id_column: str | None,
    reversed_direction: bool,
    expected_output_dir: str,
) -> None:
    _patch_text_pipeline(monkeypatch)
    module = importlib.import_module(module_name)
    corpus = _build_corpus(candidate_id_column)

    monkeypatch.chdir(tmp_path)
    output_dir = getattr(module, grid_function_name)(
        mode="transductive",
        reversed_direction=reversed_direction,
        corpus=corpus,
        k_values=(2,),
        supervision_per_cluster=(1,),
        output_only_ratio=0.25,
        cluster_size=4,
        seeds=(0, 1),
    )

    assert output_dir == Path(expected_output_dir)
    mae = np.load(tmp_path / output_dir / "mae.npy")
    mse = np.load(tmp_path / output_dir / "mse.npy")
    ami_x = np.load(tmp_path / output_dir / "ami_x.npy")
    ami_y = np.load(tmp_path / output_dir / "ami_y.npy")
    accuracy = np.load(tmp_path / output_dir / "accuracy.npy")

    assert mae.shape == (1, 1, len(MODEL_ORDER), 2)
    assert mse.shape == (1, 1, len(MODEL_ORDER), 2)
    assert ami_x.shape == (1, 1, 2)
    assert ami_y.shape == (1, 1, 2)
    assert accuracy.shape == (1, 1, 2)
    assert np.allclose(mae, 0.0)
    assert np.allclose(mse, 0.0)
    assert np.allclose(ami_x, 1.0)
    assert np.allclose(ami_y, 1.0)
    assert np.allclose(accuracy, 1.0)
