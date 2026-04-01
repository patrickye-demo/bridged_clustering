"""Flickr30k corpus preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

from bridged_clustering.structures import PreparedTextCorpus
from bridged_clustering.text import build_candidate_map


FLICKR_PARQUET = "data/flickr30k.parquet"


def _tfidf_encode(
    captions: pd.Series,
    max_df: float = 0.9,
    min_df: int = 3,
    stop_words: str = "english",
):
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words)
    return vectorizer.fit_transform(captions), vectorizer


def load_flickr_corpus(parquet_path: str = FLICKR_PARQUET) -> PreparedTextCorpus:
    df = pd.read_parquet(parquet_path).reset_index(drop=True)
    df = df.drop(columns=["sentids", "split", "filename"])
    df["_pair"] = df.apply(lambda row: list(zip(row["caption"], row["caption_embs"])), axis=1)
    df = df.explode("_pair").reset_index(drop=True)
    df[["caption", "caption_emb"]] = pd.DataFrame(df["_pair"].tolist(), index=df.index)
    df = df.drop(columns=["caption_embs", "_pair"])

    tfidf, _ = _tfidf_encode(df["caption"])
    dbscan = DBSCAN(eps=0.6, min_samples=12, metric="euclidean")
    df["cluster"] = dbscan.fit_predict(tfidf)

    valid_rows = df[df["cluster"] != -1].copy()
    eligible_clusters = valid_rows["cluster"].value_counts()
    eligible_clusters = eligible_clusters[eligible_clusters >= 25].index
    pruned_rows = valid_rows[valid_rows["cluster"].isin(eligible_clusters)].copy()

    pruned_rows["x"] = pruned_rows["image_emb"].apply(np.array)
    pruned_rows["yv"] = pruned_rows["caption_emb"].apply(np.array)
    pruned_rows["y"] = pruned_rows["caption"]

    return PreparedTextCorpus(
        name="flickr30k",
        frame=pruned_rows,
        cluster_sizes=pruned_rows["cluster"].value_counts(),
        candidate_map=build_candidate_map(pruned_rows, id_column="img_id"),
        candidate_id_column="img_id",
    )
