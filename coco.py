"""COCO image-text Bridged Clustering experiments."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from bridged_clustering.datasets import load_coco_corpus
from bridged_clustering.result_store import MetricCube
from bridged_clustering.structures import MODEL_ORDER, TextExperimentSpec, TransportSuiteSpec
from bridged_clustering.text_pipeline import (
    run_forward_text_experiment,
    run_reversed_text_experiment,
)


SPEC = TextExperimentSpec(
    name="coco",
    forward_transport=TransportSuiteSpec(
        kmm={"alpha": 0.1, "kmm_B": 1000, "kmm_eps": 0.001, "sigma": 0.5},
        em={"eps": 0.001, "max_iter": 2000, "tol": 0.0001},
        eot={"max_iter": 2000, "eps": 10, "ridge_alpha": 0.01, "tol": 1e-09},
        gw={"max_iter": 2000, "tol": 1e-09},
    ),
    reverse_transport=TransportSuiteSpec(
        kmm={"alpha": 0.1, "kmm_B": 1000, "kmm_eps": 0.001, "sigma": 0.5},
        em={"eps": 0.001, "max_iter": 2000, "tol": 0.0001},
        eot={"max_iter": 2000, "eps": 10, "ridge_alpha": 0.1, "tol": 1e-09},
        gw={"max_iter": 2000, "tol": 1e-09},
    ),
    reverse_text_kmeans={"n_init": 1, "max_iter": 30},
    reverse_image_kmeans={"n_init": 1, "max_iter": 30},
)


@lru_cache(maxsize=1)
def get_corpus():
    return load_coco_corpus()


def run_experiment(
    df: pd.DataFrame,
    supervised_ratio: float = 0.05,
    output_only_ratio: float = 0.5,
    K: int = 100,
    knn_neighbors: int = 10,
    seed: int | None = None,
    mode: str = "transductive",
) -> dict[str, dict]:
    return run_forward_text_experiment(
        df,
        corpus=get_corpus(),
        spec=SPEC,
        supervised_ratio=supervised_ratio,
        output_only_ratio=output_only_ratio,
        K=K,
        knn_neighbors=knn_neighbors,
        seed=seed,
        mode=mode,
    )


def run_reversed_experiment(
    df: pd.DataFrame,
    supervised_ratio: float = 0.05,
    output_only_ratio: float = 0.5,
    K: int = 100,
    knn_neighbors: int = 10,
    seed: int | None = None,
    mode: str = "transductive",
) -> dict[str, dict]:
    return run_reversed_text_experiment(
        df,
        spec=SPEC,
        supervised_ratio=supervised_ratio,
        output_only_ratio=output_only_ratio,
        K=K,
        knn_neighbors=knn_neighbors,
        seed=seed,
        mode=mode,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run COCO Bridged Clustering experiments")
    parser.add_argument(
        "--mode",
        choices=["transductive", "inductive"],
        default="transductive",
        help="Data usage mode.",
    )
    parser.add_argument(
        "--reversed",
        action="store_true",
        help="Run text-to-image experiments instead of image-to-text.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    corpus = get_corpus()
    runner = run_reversed_experiment if args.reversed else run_experiment
    prefix = "308_coco_rev" if args.reversed else "307_coco"
    experiment_key = f"{prefix}_tran" if args.mode == "transductive" else f"{prefix}_ind"

    k_values = [3, 4, 5, 6, 7]
    supervision_per_cluster = [1, 2, 3, 4]
    output_only_ratio = 0.1
    cluster_size = 200
    seeds = list(range(30))
    eligible_clusters = corpus.frame["cluster"].unique()

    metrics = MetricCube.allocate(
        n_k=len(k_values),
        n_supervision=len(supervision_per_cluster),
        n_trials=len(seeds),
        model_names=MODEL_ORDER,
    )

    for k_index, n_clusters in enumerate(k_values):
        for sup_index, supervised_points in enumerate(supervision_per_cluster):
            for trial_index, base_seed in enumerate(seeds):
                rng = np.random.default_rng(base_seed + k_index * 1000 + sup_index * 100 + trial_index)
                chosen_categories = rng.choice(eligible_clusters, size=n_clusters, replace=False)
                sample = corpus.frame[corpus.frame["cluster"].isin(chosen_categories)].copy()
                trial_metrics = runner(
                    sample,
                    supervised_ratio=supervised_points / cluster_size,
                    output_only_ratio=output_only_ratio,
                    K=n_clusters,
                    knn_neighbors=supervised_points,
                    seed=int(rng.integers(0, 2**32)),
                    mode=args.mode,
                )
                metrics.record(k_index, sup_index, trial_index, trial_metrics)
            print(f"Finished grid K={n_clusters}, sup={supervised_points}")

    output_dir = Path("results") / experiment_key
    metrics.save(output_dir)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
