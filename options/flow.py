from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.base import BaseModule
from core.models import ModuleScore


class OptionsFlowModule(BaseModule):
    def __init__(self) -> None:
        super().__init__(name="options_flow")

    def evaluate(self, ticker: str, bundle: dict[str, Any]) -> ModuleScore:
        with self._lock:
            chain: pd.DataFrame = bundle["options_chain"]
            unusual = (chain["volume"] / (chain["open_interest"] + 1)).clip(0, 10)
            block = float((chain["volume"] > chain["volume"].quantile(0.95)).mean())
            call_prem = chain.loc[chain["type"] == "C", "premium"].sum()
            put_prem = chain.loc[chain["type"] == "P", "premium"].sum()
            net_call_ratio = float(call_prem / max(put_prem, 1.0))
            iv_rank = float(np.clip((chain["iv"].mean() - chain["iv"].min()) / (chain["iv"].max() - chain["iv"].min() + 1e-6), 0, 1))
            gex = float((chain["open_interest"] * (chain["strike"] - chain["strike"].mean())).sum() / 1e6)

            bullish = float(np.clip(4.5 * np.tanh(net_call_ratio - 1) + 2 * unusual.mean() + 2 * block, 0, 10))
            bearish = float(np.clip(4.5 * np.tanh(1 - net_call_ratio) + 2 * unusual.mean() + 2 * block, 0, 10))
            conviction = float(np.clip(0.5 + 0.4 * unusual.mean() + 0.2 * block, 0.2, 1.5))
            score = float(np.clip((bullish - bearish + 10) / 2, 0, 10))
            confidence = float(np.clip(0.45 + 0.25 * conviction, 0.35, 0.9))

            return ModuleScore(
                module=self.name,
                value=score,
                confidence=confidence,
                metadata={
                    "bullish_flow_score": bullish,
                    "bearish_flow_score": bearish,
                    "conviction_multiplier": conviction,
                    "iv_rank": iv_rank,
                    "net_call_put_premium_ratio": net_call_ratio,
                    "gamma_exposure_estimate": gex,
                },
            )

