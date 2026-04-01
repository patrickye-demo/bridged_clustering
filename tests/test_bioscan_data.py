from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bridged_clustering.bioscan.data import get_data_splits, load_dataset


def _synthetic_bioscan_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    families = {
        "Formicidae": ("Hymenoptera", "Formica", "Formicinae"),
        "Apidae": ("Hymenoptera", "Apis", "Apinae"),
    }
    for family, (order_value, genus, subfamily) in families.items():
        for index in range(4):
            rows.append(
                {
                    "processid": f"{family.lower()}_{index}",
                    "class": "Insecta",
                    "order": order_value,
                    "family": family,
                    "species": f"{genus.lower()}_species",
                    "genus": genus,
                    "subfamily": subfamily,
                    "dna_barcode": f"{family[:3]}{index}",
                },
            )
    return pd.DataFrame(rows)


def _write_dataset(tmp_path: Path) -> Path:
    csv_path = tmp_path / "bioscan.csv"
    _synthetic_bioscan_rows().to_csv(csv_path, index=False)
    return csv_path


def test_load_dataset_selects_homogeneous_families_and_builds_image_paths(tmp_path: Path) -> None:
    csv_path = _write_dataset(tmp_path)
    image_folder = tmp_path / "images"

    sampled_df, images = load_dataset(
        str(csv_path),
        str(image_folder),
        n_families=2,
        n_samples=4,
        rng=np.random.default_rng(0),
    )

    assert len(sampled_df) == 8
    assert set(sampled_df["family"]) == {"Formicidae", "Apidae"}
    assert set(images) == set(sampled_df["processid"])
    assert all(path.startswith(str(image_folder)) for path in images.values())
    assert all(path.endswith(".jpg") for path in images.values())


def test_get_data_splits_handles_transductive_and_inductive_modes() -> None:
    df = _synthetic_bioscan_rows()

    supervised_df, input_only_df, gene_only_df, test_df = get_data_splits(
        df,
        supervised=0.25,
        out_only=0.25,
        rng=np.random.default_rng(1),
        mode="transductive",
    )
    assert len(supervised_df) == 2
    assert len(input_only_df) == 4
    assert len(gene_only_df) == 2
    assert len(test_df) == 4
    assert set(input_only_df["processid"]) == set(test_df["processid"])

    supervised_df, input_only_df, gene_only_df, test_df = get_data_splits(
        df,
        supervised=0.25,
        out_only=0.25,
        rng=np.random.default_rng(1),
        mode="inductive",
        test_frac=0.5,
    )
    assert len(supervised_df) == 2
    assert len(input_only_df) == 2
    assert len(gene_only_df) == 2
    assert len(test_df) == 2
    assert set(input_only_df["processid"]).isdisjoint(set(test_df["processid"]))
