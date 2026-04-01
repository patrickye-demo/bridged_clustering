"""CLI and paper-style grid runner for BIOSCAN.

This is the canonical BIOSCAN entrypoint behind `python bioscan.py ...`. It
loads the encoders once, executes the legacy sweep, and writes the legacy
result tensors expected by the analysis notebook.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import BioscanGridSpec, BioscanPaths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BIOSCAN Bridged Clustering experiments")
    parser.add_argument(
        "--mode",
        choices=["transductive", "inductive"],
        default="transductive",
        help="Data usage mode. In inductive, a held-out test split is carved from inference.",
    )
    parser.add_argument(
        "--reversed",
        action="store_true",
        help="Run DNA-to-image experiments instead of image-to-DNA.",
    )
    return parser


def _grid_seed(
    family_index: int,
    sample_index: int,
    supervision_index: int,
    out_only_index: int,
    trial_index: int,
    *,
    grid: BioscanGridSpec,
) -> int:
    return trial_index + grid.n_trials * (
        out_only_index
        + len(grid.out_only_values)
        * (
            supervision_index
            + len(grid.supervised_values) * (sample_index + len(grid.n_samples_values) * family_index)
        )
    )


def run_bioscan_grid(
    *,
    mode: str = "transductive",
    reversed_direction: bool = False,
    paths: BioscanPaths | None = None,
    grid: BioscanGridSpec | None = None,
) -> Path:
    try:
        from .encoders import load_encoder_suite
        from .experiments import run_experiment, run_reversed_experiment
        from .results import BioscanMetricStore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency while running BIOSCAN experiments. Install packages from requirements.txt.",
        ) from exc

    paths = paths or BioscanPaths()
    grid = grid or BioscanGridSpec()

    runner = run_reversed_experiment if reversed_direction else run_experiment
    prefix = "102_bioscan_rev" if reversed_direction else "101_bioscan"
    experiment_key = f"{prefix}_tran" if mode == "transductive" else f"{prefix}_ind"
    output_dir = Path("results") / experiment_key

    encoder_suite = load_encoder_suite()
    results = BioscanMetricStore.allocate(grid)
    grid_seeds = [
        _grid_seed(
            family_index,
            sample_index,
            supervision_index,
            out_only_index,
            trial_index,
            grid=grid,
        )
        for family_index, _ in enumerate(grid.n_families_values)
        for sample_index, _ in enumerate(grid.n_samples_values)
        for supervision_index, _ in enumerate(grid.supervised_values)
        for out_only_index, _ in enumerate(grid.out_only_values)
        for trial_index in range(grid.n_trials)
    ]
    assert len(grid_seeds) == len(set(grid_seeds))

    for family_index, n_families in enumerate(grid.n_families_values):
        for sample_index, n_samples in enumerate(grid.n_samples_values):
            for supervision_index, supervised in enumerate(grid.supervised_values):
                for out_only_index, out_only in enumerate(grid.out_only_values):
                    for trial in range(grid.n_trials):
                        seed = _grid_seed(
                            family_index,
                            sample_index,
                            supervision_index,
                            out_only_index,
                            trial,
                            grid=grid,
                        )
                        rng = np.random.default_rng(seed)
                        print(
                            f"Running trial {trial + 1} for n_families={n_families}, "
                            f"n_samples={n_samples}, supervised={supervised}, out_only={out_only}",
                        )
                        errors, mses, ami_x, ami_y, accuracy = runner(
                            paths.csv_path,
                            paths.image_folder,
                            n_families=n_families,
                            n_samples=n_samples,
                            supervised=supervised,
                            out_only=out_only,
                            rng=rng,
                            mode=mode,
                            test_frac=0.2,
                            encoder_suite=encoder_suite,
                        )
                        results.record(
                            family_index,
                            sample_index,
                            supervision_index,
                            out_only_index,
                            trial,
                            errors,
                            mses,
                            ami_x,
                            ami_y,
                            accuracy,
                        )
                        results.save(output_dir)

    results.save(output_dir)
    return output_dir


def main() -> None:
    args = build_parser().parse_args()
    output_dir = run_bioscan_grid(mode=args.mode, reversed_direction=args.reversed)
    print(f"Experiment completed. Results written to {output_dir}")
