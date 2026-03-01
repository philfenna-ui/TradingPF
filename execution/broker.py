from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

from core.exceptions import ExecutionError
from core.logging_utils import append_jsonl


@dataclass(slots=True)
class OrderTicket:
    ticker: str
    side: str
    qty: float
    limit_price: float
    stop_loss: float
    take_profit: float
    manual_confirmation: bool = True


class PaperBroker:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def submit(self, ticket: OrderTicket, confirm: bool = False) -> dict[str, Any]:
        if ticket.manual_confirmation and not confirm:
            raise ExecutionError("Manual confirmation required before execution.")
        slippage = float(self.cfg.get("default_slippage_bps", 5)) / 10_000
        fill_price = ticket.limit_price * (1 + slippage if ticket.side == "BUY" else 1 - slippage)
        receipt = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "broker": "paper",
            "status": "FILLED",
            "ticker": ticket.ticker,
            "side": ticket.side,
            "qty": ticket.qty,
            "limit_price": ticket.limit_price,
            "fill_price": fill_price,
            "stop_loss": ticket.stop_loss,
            "take_profit": ticket.take_profit,
            "slippage_bps": self.cfg.get("default_slippage_bps", 5),
        }
        append_jsonl("logs/orders.jsonl", receipt)
        return receipt

    def build_ticket(self, ticker: str, side: str, qty: float, price: float, atr: float) -> OrderTicket:
        if qty <= 0:
            raise ExecutionError("Order quantity must be positive.")
        stop_mult = float(self.cfg.get("stop_loss_atr_mult", 1.5))
        tp_mult = float(self.cfg.get("take_profit_atr_mult", 3.0))
        stop = price - stop_mult * atr if side == "BUY" else price + stop_mult * atr
        take = price + tp_mult * atr if side == "BUY" else price - tp_mult * atr
        return OrderTicket(
            ticker=ticker,
            side=side,
            qty=qty,
            limit_price=price,
            stop_loss=float(stop),
            take_profit=float(take),
            manual_confirmation=bool(self.cfg.get("manual_confirmation", True)),
        )

