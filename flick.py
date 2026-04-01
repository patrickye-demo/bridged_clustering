"""Flickr30k Bridged Clustering experiments."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from bridged_clustering.datasets import load_flickr_corpus, sample_cluster_subset
from bridged_clustering.result_store import MetricCube
from bridged_clustering.structures import MODEL_ORDER, TextExperimentSpec, TransportSuiteSpec
from bridged_clustering.text_pipeline import (
    run_forward_text_experiment,
    run_reversed_text_experiment,
)


SPEC = TextExperimentSpec(
    name="flickr30k",
    forward_transport=TransportSuiteSpec(
        kmm={"alpha": 0.1, "kmm_B": 100, "kmm_eps": 0.001, "sigma": 1.0},
        em={"eps": 0.001, "max_iter": 2000, "tol": 0.0001},
        eot={"max_iter": 2000, "eps": 10, "ridge_alpha": 0.01, "tol": 1e-09},
        gw={"max_iter": 2000, "tol": 1e-07},
    ),
    reverse_transport=TransportSuiteSpec(
        kmm={"alpha": 0.1, "kmm_B": 100, "kmm_eps": 0.001, "sigma": 1.0},
        em={"eps": 0.001, "max_iter": 2000, "tol": 0.0001},
        eot={"max_iter": 2000, "eps": 10, "ridge_alpha": 0.01, "tol": 1e-09},
        gw={"max_iter": 2000, "tol": 1e-09},
    ),
)


@lru_cache(maxsize=1)
def get_corpus():
    return load_flickr_corpus()


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
    parser = argparse.ArgumentParser(description="Run Flickr30k Bridged Clustering experiments")
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
    prefix = "306_flick_rev" if args.reversed else "305_flick"
    experiment_key = f"{prefix}_tran" if args.mode == "transductive" else f"{prefix}_ind"

    k_values = [3, 4, 5, 6, 7]
    supervision_per_cluster = [1, 2, 3, 4]
    output_only_ratio = 0.2
    cluster_size = 25
    seeds = list(range(30))

    eligible_clusters = corpus.cluster_sizes[corpus.cluster_sizes >= cluster_size].index.to_numpy()[3:]
    metrics = MetricCube.allocate(
        n_k=len(k_values),
        n_supervision=len(supervision_per_cluster),
        n_trials=len(seeds),
        model_names=MODEL_ORDER,
    )

    for k_index, n_clusters in enumerate(k_values):
        for sup_index, supervised_points in enumerate(supervision_per_cluster):
            for trial_index, trial_seed in enumerate(seeds):
                seed = trial_seed + k_index * len(k_values) + sup_index * len(k_values) * len(supervision_per_cluster)
                sample = sample_cluster_subset(corpus.frame, eligible_clusters, n_clusters, cluster_size, seed)
                trial_metrics = runner(
                    sample,
                    supervised_ratio=supervised_points / cluster_size,
                    output_only_ratio=output_only_ratio,
                    K=n_clusters,
                    knn_neighbors=supervised_points,
                    seed=seed,
                    mode=args.mode,
                )
                metrics.record(k_index, sup_index, trial_index, trial_metrics)
            print(f"Finished K={n_clusters}, sup={supervised_points}")

    output_dir = Path("results") / experiment_key
    metrics.save(output_dir)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
