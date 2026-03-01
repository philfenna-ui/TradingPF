from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(slots=True)
class NarrativeClusterResult:
    labels: list[int]
    mention_acceleration: float
    sentiment_magnitude: float
    catalyst_strength: float


class NewsEmbeddingEngine:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)

    def cluster(self, headlines: list[str], sentiments: list[float]) -> NarrativeClusterResult:
        if not headlines:
            return NarrativeClusterResult(labels=[], mention_acceleration=0.0, sentiment_magnitude=0.0, catalyst_strength=0.0)
        emb = self.vectorizer.fit_transform(headlines)
        labels = DBSCAN(eps=1.2, min_samples=1, metric="cosine").fit_predict(emb)
        counts = np.bincount(labels + abs(labels.min()))
        accel = float(np.max(counts) / max(len(headlines), 1))
        mag = float(np.mean(np.abs(sentiments))) if sentiments else 0.0
        strength = float(np.clip(10 * (0.5 * accel + 0.5 * mag), 0, 10))
        return NarrativeClusterResult(
            labels=labels.tolist(),
            mention_acceleration=accel,
            sentiment_magnitude=mag,
            catalyst_strength=strength,
        )

