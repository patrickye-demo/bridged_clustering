"""COCO corpus preparation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from bridged_clustering.structures import PreparedTextCorpus
from bridged_clustering.text import build_candidate_map


DATA_CSV = "data/coco_2.csv"


def load_coco_corpus(csv_path: str = DATA_CSV) -> PreparedTextCorpus:
    df = pd.read_csv(csv_path)
    for column in ("x", "yv"):
        df[column] = df[column].apply(lambda value: np.asarray(json.loads(value)))

    df["cluster"] = df["cat"].astype("category").cat.codes
    counts = df["cat"].value_counts()
    assert (counts == 200).all(), "Each category must have exactly 200 rows"

    return PreparedTextCorpus(
        name="coco",
        frame=df,
        cluster_sizes=df["cluster"].value_counts(),
        candidate_map=build_candidate_map(df, id_column="image_id"),
        candidate_id_column="image_id",
    )
