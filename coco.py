"""Canonical COCO experiment driver for Bridged Clustering.

Runs the COCO image-text or text-image sweep in the transductive or inductive
setting and writes `ami_x.npy`, `ami_y.npy`, `accuracy.npy`, `mae.npy`, and
`mse.npy` under `results/100_coco_*` or `results/101_coco_rev_*`.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from bridged_clustering.result_store import MetricCube
from bridged_clustering.structures import MODEL_ORDER, TextExperimentSpec, TransportSuiteSpec


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

DEFAULT_K_VALUES: tuple[int, ...] = (3, 4, 5, 6, 7)
DEFAULT_SUPERVISION_PER_CLUSTER: tuple[int, ...] = (1, 2, 3, 4)
DEFAULT_OUTPUT_ONLY_RATIO = 0.1
DEFAULT_CLUSTER_SIZE = 200
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(30))


@lru_cache(maxsize=1)
def get_corpus():
    try:
        from bridged_clustering.datasets.coco import load_coco_corpus
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency while loading the COCO corpus. Install packages from requirements.txt.",
        ) from exc
    return load_coco_corpus()


def run_experiment(
    df: pd.DataFrame,
    supervised_ratio: float = 0.05,
    output_only_ratio: float = 0.5,
    K: int = 100,
    knn_neighbors: int = 10,
    seed: int | None = None,
    mode: str = "transductive",
    *,
    corpus=None,
) -> dict[str, dict]:
    try:
        from bridged_clustering.text_pipeline import run_forward_text_experiment
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency while running the COCO experiment. Install packages from requirements.txt.",
        ) from exc
    return run_forward_text_experiment(
        df,
        corpus=corpus or get_corpus(),
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
    try:
        from bridged_clustering.text_pipeline import run_reversed_text_experiment
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency while running the reversed COCO experiment. Install packages from requirements.txt.",
        ) from exc
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


def run_coco_grid(
    *,
    mode: str = "transductive",
    reversed_direction: bool = False,
    corpus=None,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    supervision_per_cluster: tuple[int, ...] = DEFAULT_SUPERVISION_PER_CLUSTER,
    output_only_ratio: float = DEFAULT_OUTPUT_ONLY_RATIO,
    cluster_size: int = DEFAULT_CLUSTER_SIZE,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> Path:
    corpus = corpus or get_corpus()
    runner = run_reversed_experiment if reversed_direction else run_experiment
    prefix = "101_coco_rev" if reversed_direction else "100_coco"
    experiment_key = f"{prefix}_tran" if mode == "transductive" else f"{prefix}_ind"

    eligible_clusters = corpus.frame["cluster"].unique()
    if len(eligible_clusters) < max(k_values):
        raise ValueError(
            f"COCO needs at least {max(k_values)} eligible categories; found {len(eligible_clusters)}.",
        )

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
                runner_kwargs = dict(
                    supervised_ratio=supervised_points / cluster_size,
                    output_only_ratio=output_only_ratio,
                    K=n_clusters,
                    knn_neighbors=supervised_points,
                    seed=int(rng.integers(0, 2**32)),
                    mode=mode,
                )
                if not reversed_direction:
                    runner_kwargs["corpus"] = corpus
                trial_metrics = runner(
                    sample,
                    **runner_kwargs,
                )
                metrics.record(k_index, sup_index, trial_index, trial_metrics)
            print(f"Finished grid K={n_clusters}, sup={supervised_points}")

    output_dir = Path("results") / experiment_key
    metrics.save(output_dir)
    return output_dir


def main() -> None:
    args = build_parser().parse_args()
    output_dir = run_coco_grid(mode=args.mode, reversed_direction=args.reversed)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
