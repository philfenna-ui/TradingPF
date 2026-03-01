from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from core.base import BaseModule
from core.models import ModuleScore
from news.embeddings import NewsEmbeddingEngine


class NewsCatalystModule(BaseModule):
    def __init__(self) -> None:
        super().__init__(name="catalyst")
        self.embedding_engine = NewsEmbeddingEngine()

    def evaluate(self, ticker: str, bundle: dict[str, Any]) -> ModuleScore:
        with self._lock:
            news = bundle.get("news", [])
            news = news + bundle.get("macro_news", [])
            if not news:
                return ModuleScore(module=self.name, value=4.5, confidence=0.25, metadata={"reason": "no_news"})
            headlines = [n["headline"] for n in news]
            sentiments = np.array([float(n.get("sentiment", 0.0)) for n in news], dtype=float)
            cluster = self.embedding_engine.cluster(headlines=headlines, sentiments=sentiments.tolist())
            labels = np.array(cluster.labels)
            cluster_counts = Counter(labels.tolist())
            emerging = max(cluster_counts.values()) / len(headlines)
            sentiment_magnitude = cluster.sentiment_magnitude
            accel_mentions = cluster.mention_acceleration
            catalyst_strength = float(np.clip(10 * (0.4 * sentiment_magnitude + 0.3 * emerging + 0.3 * accel_mentions), 0, 10))
            confidence = float(np.clip(0.4 + 0.45 * emerging, 0.35, 0.92))
            sectors = sorted({n.get("sector", "unknown") for n in news})
            risk_flag = bool(sentiments.min() < -0.6)
            text = " ".join(headlines).lower()
            themes = []
            if any(k in text for k in ["war", "missile", "conflict", "military", "geopolitical"]):
                themes.append("geopolitical_conflict")
            if any(k in text for k in ["defense", "aerospace", "pentagon"]):
                themes.append("defense_spending")
            if any(k in text for k in ["oil", "energy", "opec"]):
                themes.append("energy_supply")
            if any(k in text for k in ["rate", "fed", "inflation"]):
                themes.append("rates_inflation")
            return ModuleScore(
                module=self.name,
                value=catalyst_strength,
                confidence=confidence,
                metadata={
                    "catalyst_strength_score": catalyst_strength,
                    "sector_beneficiaries": sectors,
                    "risk_flags": ["negative_narrative_cluster"] if risk_flag else [],
                    "cluster_labels": labels.tolist(),
                    "themes": themes,
                },
            )
