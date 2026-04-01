"""Wikipedia image-text corpus preparation."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from bridged_clustering.structures import PreparedTextCorpus


WIKI_CSV = "data/wiki_df.csv"

_STOP_WORDS = {
    "on",
    "in",
    "of",
    "to",
    "for",
    "with",
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "is",
    "are",
    "be",
    "as",
    "by",
    "at",
    "from",
    "that",
    "this",
    "these",
    "those",
}


def load_wiki_corpus(csv_path: str = WIKI_CSV) -> PreparedTextCorpus:
    df = pd.read_csv(csv_path)
    df["x"] = df["x"].apply(lambda value: np.fromstring(value[1:-1], sep=",")).tolist()
    df["yv"] = df["yv"].apply(lambda value: np.fromstring(value[1:-1], sep=",")).tolist()
    df["zv"] = df["zv"].apply(lambda value: np.fromstring(value[1:-1], sep=",")).tolist()

    dbscan = DBSCAN(eps=0.36, min_samples=12, metric="cosine").fit(np.vstack(df["zv"].values))
    df = df.assign(cluster=dbscan.labels_)

    valid_rows = df[df["cluster"] != -1].copy()
    eligible_clusters = valid_rows["cluster"].value_counts()
    eligible_clusters = eligible_clusters[eligible_clusters >= 12].index

    pruned_clusters: list[pd.DataFrame] = []
    for cluster_id in eligible_clusters:
        cluster_rows = valid_rows[valid_rows["cluster"] == cluster_id]
        word_lists = cluster_rows["z"].str.split(",").apply(
            lambda shards: [
                word.lower()
                for shard in shards
                for word in shard.strip().split()
                if word.isalpha() and word.lower() not in _STOP_WORDS
            ],
        )
        word_counts = Counter(word for words in word_lists for word in words)
        if not word_counts:
            continue

        dominant_word, _ = word_counts.most_common(1)[0]
        mask = word_lists.apply(lambda words: dominant_word in words)
        pruned_rows = cluster_rows[mask]
        if len(pruned_rows) >= 12:
            pruned_clusters.append(pruned_rows)

    frame = pd.concat(pruned_clusters, ignore_index=True)
    return PreparedTextCorpus(
        name="wiki",
        frame=frame,
        cluster_sizes=frame["cluster"].value_counts(),
    )
