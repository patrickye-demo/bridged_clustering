"""Data selection and split construction for BIOSCAN."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .config import ensure_rng


def _choose_homogeneous_group(family_data: pd.DataFrame, n_samples: int) -> pd.DataFrame | None:
    for group_column in ("species", "genus", "subfamily"):
        groups = family_data.groupby(group_column)
        valid_groups = [(name, group_df) for name, group_df in groups if len(group_df) >= n_samples]
        if valid_groups:
            valid_groups.sort(key=lambda item: len(item[1]), reverse=True)
            return valid_groups[0][1]
    return None


def load_dataset(
    csv_path: str,
    image_folder: str,
    n_families: int = 5,
    n_samples: int = 50,
    rng: np.random.Generator | int | None = 42,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Sample homogeneous family blocks and resolve image paths."""
    rng = ensure_rng(rng)
    df = pd.read_csv(csv_path)
    df = df[df["class"] == "Insecta"]

    family_counts = df["family"].value_counts()
    eligible_families = family_counts[family_counts >= n_samples].index.tolist()
    if len(eligible_families) < n_families:
        raise ValueError(f"Not enough families with at least {n_samples} samples.")

    family_info = df[["family", "class", "order"]].drop_duplicates().set_index("family")

    class_to_families: dict[str, list[str]] = {}
    for family in eligible_families:
        family_class = family_info.loc[family, "class"]
        class_to_families.setdefault(family_class, []).append(family)

    selected_families: list[str] = []
    classes = rng.permutation(list(class_to_families.keys())).tolist()
    for family_class in classes:
        family_list = rng.permutation(class_to_families[family_class]).tolist()
        selected_families.append(family_list[0])
        if len(selected_families) == n_families:
            break

    if len(selected_families) < n_families:
        order_to_families: dict[str, list[str]] = {}
        for family in eligible_families:
            order_value = family_info.loc[family, "order"]
            order_to_families.setdefault(order_value, []).append(family)
        orders = rng.permutation(list(order_to_families.keys())).tolist()
        for order_value in orders:
            candidates = [family for family in order_to_families[order_value] if family not in selected_families]
            if candidates:
                selected_families.append(rng.permutation(candidates).tolist()[0])
                if len(selected_families) == n_families:
                    break

    if len(selected_families) < n_families:
        remaining = [family for family in eligible_families if family not in selected_families]
        selected_families.extend(rng.permutation(remaining).tolist()[: n_families - len(selected_families)])

    valid_family_samples: list[pd.DataFrame] = []
    failed_families: set[str] = set()
    for family in selected_families:
        family_data = df[df["family"] == family]
        homogeneous_group = _choose_homogeneous_group(family_data, n_samples)
        if homogeneous_group is None:
            print(f"Family {family} does not have a homogeneous group with at least {n_samples} samples. Skipping.")
            failed_families.add(family)
            continue
        valid_family_samples.append(
            homogeneous_group.sample(n=n_samples, random_state=int(rng.integers(0, 2**32))),
        )
        if len(valid_family_samples) == n_families:
            break

    if len(valid_family_samples) < n_families:
        remaining_families = [
            family
            for family in eligible_families
            if family not in set(selected_families).union(failed_families)
        ]
        for family in rng.permutation(remaining_families).tolist():
            family_data = df[df["family"] == family]
            homogeneous_group = _choose_homogeneous_group(family_data, n_samples)
            if homogeneous_group is None:
                continue
            valid_family_samples.append(
                homogeneous_group.sample(n=n_samples, random_state=int(rng.integers(0, 2**32))),
            )
            if len(valid_family_samples) == n_families:
                break

    if len(valid_family_samples) < n_families:
        raise ValueError(
            f"Could not find {n_families} families with a valid homogeneous group of at least {n_samples} samples.",
        )

    final_df = pd.concat(valid_family_samples)
    print("Selected families:", list(final_df["family"].unique()))

    images = {
        row["processid"]: os.path.join(image_folder, f"{row['processid']}.jpg")
        for _, row in final_df.iterrows()
    }
    return final_df, images


def split_family_samples(
    family_data: pd.DataFrame,
    supervised: float,
    out_only: float,
    rng: np.random.Generator | int | None,
    mode: str = "transductive",
    test_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a single family into supervised, input-only, gene-only, and test blocks."""
    rng = ensure_rng(rng)
    n = len(family_data)
    n_gene = int(out_only * n)
    n_supervised = max(int(supervised * n), 1)
    permutation = rng.permutation(n)

    gene_indices = permutation[:n_gene]
    supervised_indices = permutation[n_gene : n_gene + n_supervised]
    rest_indices = permutation[n_gene + n_supervised :]

    if mode == "transductive":
        input_indices = rest_indices
        test_indices = rest_indices
    else:
        n_test = max(1, int(test_frac * len(rest_indices))) if len(rest_indices) else 0
        test_indices = rest_indices[:n_test]
        input_indices = rest_indices[n_test:]

    supervised_df = family_data.iloc[supervised_indices]
    input_only_df = family_data.iloc[input_indices]
    gene_only_df = family_data.iloc[gene_indices]
    test_df = family_data.iloc[test_indices]
    return supervised_df, input_only_df, gene_only_df, test_df


def get_data_splits(
    df: pd.DataFrame,
    supervised: float,
    out_only: float,
    rng: np.random.Generator | int | None,
    mode: str = "transductive",
    test_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the BIOSCAN four-way split independently to each family."""
    rng = ensure_rng(rng)
    supervised_blocks: list[pd.DataFrame] = []
    input_only_blocks: list[pd.DataFrame] = []
    gene_only_blocks: list[pd.DataFrame] = []
    test_blocks: list[pd.DataFrame] = []

    for family in df["family"].unique():
        family_rows = df[df["family"] == family]
        sup, input_only, gene_only, test = split_family_samples(
            family_rows,
            supervised=supervised,
            out_only=out_only,
            rng=rng,
            mode=mode,
            test_frac=test_frac,
        )
        supervised_blocks.append(sup)
        input_only_blocks.append(input_only)
        gene_only_blocks.append(gene_only)
        test_blocks.append(test)

    empty = df.iloc[0:0].copy()
    supervised_df = pd.concat(supervised_blocks) if supervised_blocks else empty
    input_only_df = pd.concat(input_only_blocks) if input_only_blocks else empty
    gene_only_df = pd.concat(gene_only_blocks) if gene_only_blocks else empty
    test_df = pd.concat(test_blocks) if test_blocks else empty
    return supervised_df, input_only_df, gene_only_df, test_df
