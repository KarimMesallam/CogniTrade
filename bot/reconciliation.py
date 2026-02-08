from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class ReconciliationReport:
    """Summary of a reconciliation pass."""

    symbol: str
    trade_mode: str
    reconciled_at: str
    local_active_count: int
    exchange_open_count: int
    stale_local_order_ids: List[str] = field(default_factory=list)
    missing_local_order_ids: List[str] = field(default_factory=list)
    synced_local_order_ids: List[str] = field(default_factory=list)
    position_mismatch: bool = False
    position_delta: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "trade_mode": self.trade_mode,
            "reconciled_at": self.reconciled_at,
            "local_active_count": self.local_active_count,
            "exchange_open_count": self.exchange_open_count,
            "stale_local_order_ids": list(self.stale_local_order_ids),
            "missing_local_order_ids": list(self.missing_local_order_ids),
            "synced_local_order_ids": list(self.synced_local_order_ids),
            "position_mismatch": bool(self.position_mismatch),
            "position_delta": float(self.position_delta),
        }


def _normalize_order_id(order_id: Any) -> Optional[str]:
    if order_id is None:
        return None
    return str(order_id)


class ExchangeReconciler:
    """Pure reconciliation logic for local-vs-exchange state convergence."""

    def __init__(self, symbol: str, trade_mode: str = "SPOT"):
        self.symbol = symbol
        self.trade_mode = str(trade_mode or "SPOT").upper()

    def reconcile_orders(
        self,
        *,
        local_active_orders: Dict[Any, Dict[str, Any]],
        exchange_open_orders: Iterable[Dict[str, Any]],
    ) -> ReconciliationReport:
        local_by_id: Dict[str, Dict[str, Any]] = {}
        for key, payload in local_active_orders.items():
            oid = _normalize_order_id(payload.get("order_id", key))
            if oid is None:
                continue
            local_by_id[oid] = payload

        exchange_by_id: Dict[str, Dict[str, Any]] = {}
        for order in exchange_open_orders:
            oid = _normalize_order_id(order.get("orderId"))
            if oid is None:
                continue
            exchange_by_id[oid] = order

        local_ids = set(local_by_id.keys())
        exchange_ids = set(exchange_by_id.keys())

        stale_local = sorted(local_ids - exchange_ids)
        missing_local = sorted(exchange_ids - local_ids)
        synced = sorted(local_ids & exchange_ids)

        return ReconciliationReport(
            symbol=self.symbol,
            trade_mode=self.trade_mode,
            reconciled_at=datetime.utcnow().isoformat(),
            local_active_count=len(local_ids),
            exchange_open_count=len(exchange_ids),
            stale_local_order_ids=stale_local,
            missing_local_order_ids=missing_local,
            synced_local_order_ids=synced,
        )

    def reconcile_position(
        self,
        *,
        local_position_qty: Optional[float],
        exchange_position_qty: Optional[float],
        tolerance: float = 1e-8,
    ) -> Tuple[bool, float]:
        if local_position_qty is None or exchange_position_qty is None:
            return False, 0.0

        local_dec = Decimal(str(local_position_qty))
        exchange_dec = Decimal(str(exchange_position_qty))
        delta = exchange_dec - local_dec
        mismatch = abs(delta) > Decimal(str(tolerance))
        return mismatch, float(delta)

