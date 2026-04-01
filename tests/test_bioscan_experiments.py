from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bridged_clustering.bioscan.config import BIOSCAN_MODEL_NAMES, BioscanGridSpec, BioscanPaths


FAMILY_TO_CLUSTER = {"alpha": 0, "beta": 1}
FAMILY_TO_MORPH = {
    "alpha": np.array([0.0, 0.0]),
    "beta": np.array([10.0, 10.0]),
}
FAMILY_TO_GENE = {
    "alpha": np.array([100.0, 100.0]),
    "beta": np.array([200.0, 200.0]),
}


def _synthetic_experiment_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family in ("alpha", "beta"):
        for index in range(4):
            rows.append(
                {
                    "processid": f"{family}_{index}",
                    "family": family,
                    "dna_barcode": f"{family}_{index}_barcode",
                },
            )
    return pd.DataFrame(rows)


def _encode_coordinates(df: pd.DataFrame, column: str, mapping: dict[str, np.ndarray]) -> pd.DataFrame:
    encoded = df.copy()
    encoded[column] = [mapping[family].copy() for family in encoded["family"]]
    return encoded


def _forward_semisupervised_baseline(
    _supervised_df: pd.DataFrame,
    _input_only_df: pd.DataFrame,
    test_df: pd.DataFrame,
    **_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    actuals = np.vstack(test_df["gene_coordinates"].values)
    return actuals.copy(), actuals


def _forward_named_baseline(*, inference_df: pd.DataFrame, **_kwargs) -> tuple[np.ndarray, np.ndarray]:
    actuals = np.vstack(inference_df["gene_coordinates"].values)
    return actuals.copy(), actuals


def _reversed_named_baseline(*, inference_df: pd.DataFrame, **_kwargs) -> tuple[np.ndarray, np.ndarray]:
    actuals = np.vstack(inference_df["morph_coordinates"].values)
    return actuals.copy(), actuals


def _baseline_stubs() -> dict[str, object]:
    return {
        "em_regression": _forward_named_baseline,
        "eot_barycentric_regression": _forward_named_baseline,
        "fixmatch_regression": _forward_semisupervised_baseline,
        "gcn_regression": _forward_semisupervised_baseline,
        "gw_metric_alignment_regression": _forward_named_baseline,
        "kernel_mean_matching_regression": _forward_named_baseline,
        "laprls_regression": _forward_semisupervised_baseline,
        "reversed_em_regression": _reversed_named_baseline,
        "reversed_eot_barycentric_regression": _reversed_named_baseline,
        "reversed_gw_metric_alignment_regression": _reversed_named_baseline,
        "reversed_kernel_mean_matching_regression": _reversed_named_baseline,
        "tnnr_regression": _forward_semisupervised_baseline,
        "tsvr_regression": _forward_semisupervised_baseline,
        "ucvme_regression": _forward_semisupervised_baseline,
    }


def _patch_synthetic_pipeline(monkeypatch: pytest.MonkeyPatch, experiments_module) -> pd.DataFrame:
    synthetic_df = _synthetic_experiment_rows()

    monkeypatch.setattr(
        experiments_module,
        "load_dataset",
        lambda *_args, **_kwargs: (
            synthetic_df.copy(),
            {process_id: f"/synthetic/{process_id}.jpg" for process_id in synthetic_df["processid"]},
        ),
    )
    monkeypatch.setattr(
        experiments_module,
        "encode_images_for_samples",
        lambda df, *_args, **_kwargs: _encode_coordinates(df, "morph_coordinates", FAMILY_TO_MORPH),
    )
    monkeypatch.setattr(
        experiments_module,
        "encode_genes_for_samples",
        lambda df, *_args, **_kwargs: _encode_coordinates(df, "gene_coordinates", FAMILY_TO_GENE),
    )
    monkeypatch.setattr(experiments_module, "_load_baseline_regressors", _baseline_stubs)

    def fake_perform_clustering(
        image_samples: pd.DataFrame,
        gene_samples: pd.DataFrame,
        *_args,
        **_kwargs,
    ) -> tuple[SimpleNamespace, SimpleNamespace, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        image_features = np.vstack(image_samples["morph_coordinates"].values)
        gene_features = np.vstack(gene_samples["gene_coordinates"].values)
        image_clusters = image_samples["family"].map(FAMILY_TO_CLUSTER).to_numpy()
        gene_clusters = gene_samples["family"].map(FAMILY_TO_CLUSTER).to_numpy()
        image_kmeans = SimpleNamespace(
            cluster_centers_=np.vstack([FAMILY_TO_MORPH["alpha"], FAMILY_TO_MORPH["beta"]]),
            labels_=image_clusters,
        )
        gene_kmeans = SimpleNamespace(
            cluster_centers_=np.vstack([FAMILY_TO_GENE["alpha"], FAMILY_TO_GENE["beta"]]),
            labels_=gene_clusters,
        )
        return image_kmeans, gene_kmeans, image_features, gene_features, image_clusters, gene_clusters

    def fake_perform_inference(
        inference_samples: pd.DataFrame,
        image_clusters: np.ndarray,
        *_args,
        **_kwargs,
    ) -> pd.DataFrame:
        inference = inference_samples.copy()
        inference["image_cluster"] = image_clusters
        inference["predicted_gene_cluster"] = inference["image_cluster"]
        inference["predicted_gene_coordinates"] = inference["gene_coordinates"]
        return inference

    monkeypatch.setattr(experiments_module, "perform_clustering", fake_perform_clustering)
    monkeypatch.setattr(experiments_module, "perform_inference", fake_perform_inference)
    return synthetic_df


def _dummy_encoder_suite(experiments_module):
    return experiments_module.EncoderSuite(
        barcode_tokenizer=object(),
        barcode_model=object(),
        image_model=object(),
        image_transform=object(),
    )


def test_experiments_module_imports_without_loading_baselines() -> None:
    module = importlib.import_module("bridged_clustering.bioscan.experiments")
    assert hasattr(module, "run_experiment")
    assert hasattr(module, "run_reversed_experiment")


def test_run_experiment_succeeds_on_synthetic_data(monkeypatch: pytest.MonkeyPatch) -> None:
    experiments_module = importlib.import_module("bridged_clustering.bioscan.experiments")
    _patch_synthetic_pipeline(monkeypatch, experiments_module)

    errors, mses, ami_image, ami_gene, accuracy = experiments_module.run_experiment(
        "unused.csv",
        "unused_images",
        n_families=2,
        n_samples=4,
        supervised=0.25,
        out_only=0.25,
        rng=np.random.default_rng(0),
        mode="transductive",
        encoder_suite=_dummy_encoder_suite(experiments_module),
    )

    assert set(errors) == set(BIOSCAN_MODEL_NAMES)
    assert set(mses) == set(BIOSCAN_MODEL_NAMES)
    assert errors == pytest.approx({name: 0.0 for name in BIOSCAN_MODEL_NAMES})
    assert mses == pytest.approx({name: 0.0 for name in BIOSCAN_MODEL_NAMES})
    assert ami_image == pytest.approx(1.0)
    assert ami_gene == pytest.approx(1.0)
    assert accuracy == pytest.approx(1.0)


def test_run_reversed_experiment_succeeds_on_synthetic_data(monkeypatch: pytest.MonkeyPatch) -> None:
    experiments_module = importlib.import_module("bridged_clustering.bioscan.experiments")
    _patch_synthetic_pipeline(monkeypatch, experiments_module)

    errors, mses, ami_image, ami_gene, accuracy = experiments_module.run_reversed_experiment(
        "unused.csv",
        "unused_images",
        n_families=2,
        n_samples=4,
        supervised=0.25,
        out_only=0.25,
        knn_neighbors=1,
        rng=np.random.default_rng(0),
        mode="transductive",
        encoder_suite=_dummy_encoder_suite(experiments_module),
    )

    assert set(errors) == set(BIOSCAN_MODEL_NAMES)
    assert set(mses) == set(BIOSCAN_MODEL_NAMES)
    assert errors == pytest.approx({name: 0.0 for name in BIOSCAN_MODEL_NAMES})
    assert mses == pytest.approx({name: 0.0 for name in BIOSCAN_MODEL_NAMES})
    assert ami_image == pytest.approx(1.0)
    assert ami_gene == pytest.approx(1.0)
    assert accuracy == pytest.approx(1.0)


def test_run_bioscan_grid_writes_result_tensors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    grid_module = importlib.import_module("bridged_clustering.bioscan.grid")
    encoders_module = importlib.import_module("bridged_clustering.bioscan.encoders")
    experiments_module = importlib.import_module("bridged_clustering.bioscan.experiments")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(encoders_module, "load_encoder_suite", lambda: _dummy_encoder_suite(experiments_module))
    monkeypatch.setattr(
        experiments_module,
        "run_experiment",
        lambda *_args, **_kwargs: (
            {"BKM": 1.0, "KNN": 2.0},
            {"BKM": 3.0, "KNN": 4.0},
            0.5,
            0.75,
            1.0,
        ),
    )

    output_dir = grid_module.run_bioscan_grid(
        mode="transductive",
        reversed_direction=False,
        paths=BioscanPaths(csv_path="unused.csv", image_folder="unused_images"),
        grid=BioscanGridSpec(
            n_families_values=(2,),
            n_samples_values=(4,),
            supervised_values=(0.25,),
            out_only_values=(0.25,),
            n_trials=2,
            model_names=("BKM", "KNN"),
        ),
    )

    assert output_dir == Path("results/101_bioscan_tran")
    mae = np.load(tmp_path / output_dir / "mae.npy")
    mse = np.load(tmp_path / output_dir / "mse.npy")
    ami_x = np.load(tmp_path / output_dir / "ami_x.npy")
    ami_y = np.load(tmp_path / output_dir / "ami_y.npy")
    accuracy = np.load(tmp_path / output_dir / "accuracy.npy")

    assert mae.shape == (1, 1, 2, 2)
    assert mse.shape == (1, 1, 2, 2)
    assert ami_x.shape == (1, 1, 2)
    assert ami_y.shape == (1, 1, 2)
    assert accuracy.shape == (1, 1, 2)
    assert np.all(mae == np.array([[[[1.0, 1.0], [2.0, 2.0]]]]))
    assert np.all(mse == np.array([[[[3.0, 3.0], [4.0, 4.0]]]]))
    assert np.all(ami_x == 0.5)
    assert np.all(ami_y == 0.75)
    assert np.all(accuracy == 1.0)
