"""Result storage for BIOSCAN grid sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import BioscanGridSpec


@dataclass
class BioscanMetricStore:
    model_names: Sequence[str]
    mae: np.ndarray
    mse: np.ndarray
    ami_x: np.ndarray
    ami_y: np.ndarray
    accuracy: np.ndarray

    @staticmethod
    def _legacy_view(array: np.ndarray) -> np.ndarray:
        if array.ndim == 6 and array.shape[1] == 1 and array.shape[3] == 1:
            return array[:, 0, :, 0, :, :]
        if array.ndim == 5 and array.shape[1] == 1 and array.shape[3] == 1:
            return array[:, 0, :, 0, :]
        return array

    @classmethod
    def allocate(cls, grid: BioscanGridSpec) -> "BioscanMetricStore":
        shape_prefix = (
            len(grid.n_families_values),
            len(grid.n_samples_values),
            len(grid.supervised_values),
            len(grid.out_only_values),
        )
        n_models = len(grid.model_names)
        n_trials = grid.n_trials
        return cls(
            model_names=grid.model_names,
            mae=np.empty(shape_prefix + (n_models, n_trials)),
            mse=np.empty(shape_prefix + (n_models, n_trials)),
            ami_x=np.empty(shape_prefix + (n_trials,)),
            ami_y=np.empty(shape_prefix + (n_trials,)),
            accuracy=np.empty(shape_prefix + (n_trials,)),
        )

    def record(
        self,
        family_index: int,
        sample_index: int,
        supervision_index: int,
        out_only_index: int,
        trial_index: int,
        errors: dict[str, float],
        mses: dict[str, float],
        ami_x: float,
        ami_y: float,
        accuracy: float,
    ) -> None:
        self.ami_x[family_index, sample_index, supervision_index, out_only_index, trial_index] = ami_x
        self.ami_y[family_index, sample_index, supervision_index, out_only_index, trial_index] = ami_y
        self.accuracy[family_index, sample_index, supervision_index, out_only_index, trial_index] = accuracy

        for model_index, model_name in enumerate(self.model_names):
            self.mae[family_index, sample_index, supervision_index, out_only_index, model_index, trial_index] = errors[model_name]
            self.mse[family_index, sample_index, supervision_index, out_only_index, model_index, trial_index] = mses[model_name]

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "mae.npy", self._legacy_view(self.mae))
        np.save(output_dir / "mse.npy", self._legacy_view(self.mse))
        np.save(output_dir / "ami_x.npy", self._legacy_view(self.ami_x))
        np.save(output_dir / "ami_y.npy", self._legacy_view(self.ami_y))
        np.save(output_dir / "accuracy.npy", self._legacy_view(self.accuracy))
