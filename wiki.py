"""Canonical Wikipedia experiment driver for Bridged Clustering.

Runs the Wikipedia image-text or text-image sweep in the transductive or
inductive setting and writes `ami_x.npy`, `ami_y.npy`, `accuracy.npy`,
`mae.npy`, and `mse.npy` under `results/106_wiki_*` or
`results/107_wiki_reversed_*`.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import pandas as pd

from bridged_clustering.result_store import MetricCube
from bridged_clustering.structures import MODEL_ORDER, TextExperimentSpec, TransportSuiteSpec


SPEC = TextExperimentSpec(
    name="wiki",
    forward_transport=TransportSuiteSpec(
        kmm={"alpha": 0.01, "kmm_B": 100, "kmm_eps": 0.001, "sigma": 0.5},
        em={"eps": 0.001, "max_iter": 2000, "tol": 1e-4},
        eot={"max_iter": 2000, "eps": 10, "ridge_alpha": 0.1, "tol": 1e-09},
        gw={"max_iter": 2000, "tol": 1e-07},
    ),
    reverse_transport=TransportSuiteSpec(
        kmm={"random_state": 0, "alpha": 0.01, "kmm_B": 100, "kmm_eps": 0.001, "sigma": 0.5},
        em={"eps": 0.001, "max_iter": 2000, "tol": 1e-4},
        eot={"max_iter": 2000, "ridge_alpha": 0.01, "tol": 1e-09},
        gw={"max_iter": 2000, "tol": 1e-07},
    ),
)

DEFAULT_K_VALUES: tuple[int, ...] = (3, 4, 5, 6, 7)
DEFAULT_SUPERVISION_PER_CLUSTER: tuple[int, ...] = (1, 2, 3, 4)
DEFAULT_OUTPUT_ONLY_RATIO = 0.2
DEFAULT_CLUSTER_SIZE = 25
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(30))


@lru_cache(maxsize=1)
def get_corpus():
    try:
        from bridged_clustering.datasets.wiki import load_wiki_corpus
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency while loading the Wikipedia corpus. Install packages from requirements.txt.",
        ) from exc
    return load_wiki_corpus()


def _grid_seed(
    k_index: int,
    sup_index: int,
    trial_seed: int,
    *,
    n_trials: int,
    n_supervision: int,
) -> int:
    return trial_seed + n_trials * (sup_index + n_supervision * k_index)


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
            "Missing dependency while running the Wikipedia experiment. Install packages from requirements.txt.",
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
            "Missing dependency while running the reversed Wikipedia experiment. Install packages from requirements.txt.",
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
    parser = argparse.ArgumentParser(description="Run Wikipedia Bridged Clustering experiments")
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


def run_wiki_grid(
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
    from bridged_clustering.datasets.common import sample_cluster_subset

    corpus = corpus or get_corpus()
    runner = run_reversed_experiment if reversed_direction else run_experiment
    prefix = "107_wiki_reversed" if reversed_direction else "106_wiki"
    experiment_key = f"{prefix}_tran" if mode == "transductive" else f"{prefix}_ind"

    grid_seeds = [
        _grid_seed(
            k_index,
            sup_index,
            trial_seed,
            n_trials=len(seeds),
            n_supervision=len(supervision_per_cluster),
        )
        for k_index, _ in enumerate(k_values)
        for sup_index, _ in enumerate(supervision_per_cluster)
        for trial_seed in seeds
    ]
    assert len(grid_seeds) == len(set(grid_seeds))

    eligible_clusters = corpus.cluster_sizes[corpus.cluster_sizes >= cluster_size].index.to_numpy()
    if len(eligible_clusters) < max(k_values):
        raise ValueError(
            f"Wikipedia needs at least {max(k_values)} eligible clusters of size {cluster_size}; "
            f"found {len(eligible_clusters)}.",
        )
    metrics = MetricCube.allocate(
        n_k=len(k_values),
        n_supervision=len(supervision_per_cluster),
        n_trials=len(seeds),
        model_names=MODEL_ORDER,
    )

    for k_index, n_clusters in enumerate(k_values):
        for sup_index, supervised_points in enumerate(supervision_per_cluster):
            for trial_index, trial_seed in enumerate(seeds):
                seed = _grid_seed(
                    k_index,
                    sup_index,
                    trial_seed,
                    n_trials=len(seeds),
                    n_supervision=len(supervision_per_cluster),
                )
                sample = sample_cluster_subset(corpus.frame, eligible_clusters, n_clusters, cluster_size, seed)
                runner_kwargs = dict(
                    supervised_ratio=supervised_points / cluster_size,
                    output_only_ratio=output_only_ratio,
                    K=n_clusters,
                    knn_neighbors=supervised_points,
                    seed=seed,
                    mode=mode,
                )
                if not reversed_direction:
                    runner_kwargs["corpus"] = corpus
                trial_metrics = runner(
                    sample,
                    **runner_kwargs,
                )
                metrics.record(k_index, sup_index, trial_index, trial_metrics)
            print(f"Finished K={n_clusters}, sup={supervised_points}")

    output_dir = Path("results") / experiment_key
    metrics.save(output_dir)
    return output_dir


def main() -> None:
    args = build_parser().parse_args()
    output_dir = run_wiki_grid(mode=args.mode, reversed_direction=args.reversed)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
