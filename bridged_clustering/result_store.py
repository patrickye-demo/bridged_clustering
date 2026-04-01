"""Result buffers for grid-based experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class MetricCube:
    """Preallocated storage for the experiment grids used in the paper scripts."""

    model_names: Sequence[str]
    ami_x: np.ndarray
    ami_y: np.ndarray
    accuracy: np.ndarray
    mae: np.ndarray
    mse: np.ndarray

    @classmethod
    def allocate(
        cls,
        n_k: int,
        n_supervision: int,
        n_trials: int,
        model_names: Sequence[str],
    ) -> "MetricCube":
        n_models = len(model_names)
        return cls(
            model_names=model_names,
            ami_x=np.empty((n_k, n_supervision, n_trials)),
            ami_y=np.empty((n_k, n_supervision, n_trials)),
            accuracy=np.empty((n_k, n_supervision, n_trials)),
            mae=np.empty((n_k, n_supervision, n_models, n_trials)),
            mse=np.empty((n_k, n_supervision, n_models, n_trials)),
        )

    def record(self, k_index: int, sup_index: int, trial_index: int, metrics: dict[str, dict]) -> None:
        self.ami_x[k_index, sup_index, trial_index] = metrics["clustering"]["AMI_X"]
        self.ami_y[k_index, sup_index, trial_index] = metrics["clustering"]["AMI_Y"]
        self.accuracy[k_index, sup_index, trial_index] = metrics["clustering"]["Bridging Accuracy"]

        for model_index, model_name in enumerate(self.model_names):
            regression = metrics["regression"][model_name]
            self.mae[k_index, sup_index, model_index, trial_index] = regression["MAE"]
            self.mse[k_index, sup_index, model_index, trial_index] = regression["MSE"]

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "ami_x.npy", self.ami_x)
        np.save(output_dir / "ami_y.npy", self.ami_y)
        np.save(output_dir / "accuracy.npy", self.accuracy)
        np.save(output_dir / "mae.npy", self.mae)
        np.save(output_dir / "mse.npy", self.mse)
