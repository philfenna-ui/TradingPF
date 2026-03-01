from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class BacktestMetrics:
    win_rate: float
    sharpe: float
    sortino: float
    cagr: float
    max_drawdown: float
    expectancy: float
    signal_decay: float


class WalkForwardBacktester:
    def run(self, returns: np.ndarray) -> BacktestMetrics:
        if len(returns) < 30:
            return BacktestMetrics(0, 0, 0, 0, 0, 0, 0)
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        win_rate = float(len(wins) / len(returns))
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252))
        downside = np.std(losses) if len(losses) else 1e-9
        sortino = float(np.mean(returns) / (downside + 1e-9) * np.sqrt(252))
        equity = np.cumprod(1 + returns)
        years = max(len(returns) / 252, 1e-6)
        cagr = float(equity[-1] ** (1 / years) - 1)
        running_max = np.maximum.accumulate(equity)
        dd = float(np.max((running_max - equity) / np.maximum(running_max, 1e-9)))
        expectancy = float(np.mean(returns))
        signal_decay = float(np.clip(np.corrcoef(returns[:-1], returns[1:])[0, 1], -1, 1)) if len(returns) > 5 else 0.0
        return BacktestMetrics(
            win_rate=win_rate,
            sharpe=sharpe,
            sortino=sortino,
            cagr=cagr,
            max_drawdown=dd,
            expectancy=expectancy,
            signal_decay=signal_decay,
        )

