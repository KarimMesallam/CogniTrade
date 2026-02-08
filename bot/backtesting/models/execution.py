"""
Execution simulation models for realistic backtest fills.
"""
from dataclasses import dataclass
from decimal import Decimal, getcontext
from typing import Optional

getcontext().prec = 28


@dataclass
class ExecutionSimulationConfig:
    """Configuration for execution realism in backtests."""
    enabled: bool = False
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    latency_bps: float = 0.0
    max_volume_participation: float = 1.0
    min_partial_fill_ratio: float = 0.0
    funding_rate_per_8h: float = 0.0


@dataclass
class ExecutionFill:
    """Represents a simulated order fill."""
    side: str
    requested_quantity: Decimal
    executed_quantity: Decimal
    reference_price: Decimal
    executed_price: Decimal
    fill_ratio: Decimal
    spread_cost: Decimal
    slippage_cost: Decimal
    latency_cost: Decimal

    @property
    def notional(self) -> Decimal:
        return self.executed_price * self.executed_quantity


class ExecutionSimulator:
    """Simple deterministic execution simulator for backtests."""

    def __init__(self, config: Optional[ExecutionSimulationConfig] = None):
        self.config = config or ExecutionSimulationConfig()

    def simulate_fill(
        self,
        side: str,
        reference_price: Decimal,
        requested_quantity: Decimal,
        candle_volume: Optional[Decimal] = None,
    ) -> ExecutionFill:
        """
        Simulate fill price/quantity from a reference price and order size.
        """
        side = side.upper()
        if requested_quantity <= 0:
            return ExecutionFill(
                side=side,
                requested_quantity=requested_quantity,
                executed_quantity=Decimal("0"),
                reference_price=reference_price,
                executed_price=reference_price,
                fill_ratio=Decimal("0"),
                spread_cost=Decimal("0"),
                slippage_cost=Decimal("0"),
                latency_cost=Decimal("0"),
            )

        if not self.config.enabled:
            return ExecutionFill(
                side=side,
                requested_quantity=requested_quantity,
                executed_quantity=requested_quantity,
                reference_price=reference_price,
                executed_price=reference_price,
                fill_ratio=Decimal("1"),
                spread_cost=Decimal("0"),
                slippage_cost=Decimal("0"),
                latency_cost=Decimal("0"),
            )

        fill_ratio = self._calculate_fill_ratio(requested_quantity, candle_volume)
        executed_quantity = requested_quantity * fill_ratio
        if executed_quantity <= 0:
            return ExecutionFill(
                side=side,
                requested_quantity=requested_quantity,
                executed_quantity=Decimal("0"),
                reference_price=reference_price,
                executed_price=reference_price,
                fill_ratio=Decimal("0"),
                spread_cost=Decimal("0"),
                slippage_cost=Decimal("0"),
                latency_cost=Decimal("0"),
            )

        half_spread_bps = Decimal(str(self.config.spread_bps)) / Decimal("2")
        slippage_bps = Decimal(str(self.config.slippage_bps))
        latency_bps = Decimal(str(self.config.latency_bps))
        total_bps = half_spread_bps + slippage_bps + latency_bps
        total_multiplier = total_bps / Decimal("10000")

        if side == "BUY":
            executed_price = reference_price * (Decimal("1") + total_multiplier)
        else:
            executed_price = reference_price * (Decimal("1") - total_multiplier)
            if executed_price <= 0:
                executed_price = Decimal("0.00000001")

        spread_component = (reference_price * (half_spread_bps / Decimal("10000"))) * executed_quantity
        slippage_component = (reference_price * (slippage_bps / Decimal("10000"))) * executed_quantity
        latency_component = (reference_price * (latency_bps / Decimal("10000"))) * executed_quantity

        return ExecutionFill(
            side=side,
            requested_quantity=requested_quantity,
            executed_quantity=executed_quantity,
            reference_price=reference_price,
            executed_price=executed_price,
            fill_ratio=fill_ratio,
            spread_cost=spread_component,
            slippage_cost=slippage_component,
            latency_cost=latency_component,
        )

    def estimate_funding_cost(
        self,
        notional: Decimal,
        holding_hours: float,
        position_side: str = "LONG",
    ) -> Decimal:
        """
        Estimate perpetual funding impact for held position notional.
        Positive value means a cost; negative means a credit.
        """
        if not self.config.enabled:
            return Decimal("0")

        rate_per_8h = Decimal(str(self.config.funding_rate_per_8h))
        if rate_per_8h == 0 or holding_hours <= 0:
            return Decimal("0")

        intervals = Decimal(str(holding_hours)) / Decimal("8")
        funding = notional * rate_per_8h * intervals

        side = position_side.upper()
        if side == "LONG":
            return funding
        return -funding

    def _calculate_fill_ratio(
        self,
        requested_quantity: Decimal,
        candle_volume: Optional[Decimal],
    ) -> Decimal:
        """
        Calculate deterministic partial-fill ratio from volume participation.
        """
        participation_cap = Decimal(str(self.config.max_volume_participation))
        min_fill = Decimal(str(self.config.min_partial_fill_ratio))

        if participation_cap <= 0:
            return Decimal("0")

        if candle_volume is None or candle_volume <= 0:
            return Decimal("1")

        participation = requested_quantity / candle_volume
        if participation <= participation_cap:
            return Decimal("1")

        ratio = participation_cap / participation
        if ratio < min_fill:
            ratio = min_fill
        if ratio > Decimal("1"):
            ratio = Decimal("1")
        if ratio < Decimal("0"):
            ratio = Decimal("0")
        return ratio
