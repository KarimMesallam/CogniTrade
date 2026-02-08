from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class RiskSnapshot:
    """Current live portfolio risk state."""

    timestamp: str
    equity_usd: float
    peak_equity_usd: float
    gross_exposure_usd: float
    drawdown_pct: float
    daily_pnl_usd: float
    kill_switch_active: bool
    kill_switch_reason: Optional[str]
    kill_switch_triggered: bool = False


@dataclass
class RiskCheckResult:
    """Result of a pre-trade hard-risk check."""

    allowed: bool
    reason: Optional[str]
    projected_exposure_usd: float
    kill_switch_active: bool


class LiveRiskEngine:
    """
    Hard live risk controls for production trading.

    The engine enforces:
    - max drawdown cap (vs. peak equity)
    - max gross exposure cap
    - daily loss cap
    - kill-switch automation
    """

    def __init__(
        self,
        *,
        max_drawdown_pct: float = 20.0,
        max_gross_exposure_usd: float = 0.0,
        daily_loss_limit_usd: float = 0.0,
        kill_switch_enabled: bool = True,
        allow_risk_reducing_orders: bool = True,
    ):
        self.max_drawdown_pct = max(0.0, float(max_drawdown_pct))
        self.max_gross_exposure_usd = max(0.0, float(max_gross_exposure_usd))
        self.daily_loss_limit_usd = max(0.0, float(daily_loss_limit_usd))
        self.kill_switch_enabled = bool(kill_switch_enabled)
        self.allow_risk_reducing_orders = bool(allow_risk_reducing_orders)

        self.current_equity_usd: float = 0.0
        self.peak_equity_usd: float = 0.0
        self.current_gross_exposure_usd: float = 0.0

        self._daily_anchor_date: Optional[str] = None
        self._daily_anchor_equity_usd: Optional[float] = None

        self.kill_switch_active: bool = False
        self.kill_switch_reason: Optional[str] = None

    def _normalize_timestamp(self, timestamp: Optional[str]) -> str:
        return timestamp or datetime.utcnow().isoformat()

    def _daily_rollover_if_needed(self, ts: datetime, equity_usd: float) -> None:
        anchor_date = ts.date().isoformat()
        if self._daily_anchor_date != anchor_date:
            self._daily_anchor_date = anchor_date
            self._daily_anchor_equity_usd = float(equity_usd)

    def _compute_drawdown_pct(self) -> float:
        if self.peak_equity_usd <= 0:
            return 0.0
        drawdown = max(0.0, self.peak_equity_usd - self.current_equity_usd)
        return (drawdown / self.peak_equity_usd) * 100.0

    def _compute_daily_pnl_usd(self) -> float:
        if self._daily_anchor_equity_usd is None:
            return 0.0
        return float(self.current_equity_usd - self._daily_anchor_equity_usd)

    def activate_kill_switch(self, reason: str) -> None:
        self.kill_switch_active = True
        self.kill_switch_reason = str(reason)

    def reset_kill_switch(self) -> None:
        self.kill_switch_active = False
        self.kill_switch_reason = None

    def update_portfolio_state(
        self,
        *,
        equity_usd: float,
        gross_exposure_usd: float,
        timestamp: Optional[str] = None,
    ) -> RiskSnapshot:
        """
        Update portfolio risk state and trigger kill-switch when hard limits are breached.
        """
        ts_str = self._normalize_timestamp(timestamp)
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

        self.current_equity_usd = float(equity_usd)
        self.current_gross_exposure_usd = max(0.0, float(gross_exposure_usd))
        self.peak_equity_usd = max(self.peak_equity_usd, self.current_equity_usd)
        if self.peak_equity_usd <= 0 and self.current_equity_usd > 0:
            self.peak_equity_usd = self.current_equity_usd

        self._daily_rollover_if_needed(ts, self.current_equity_usd)

        drawdown_pct = self._compute_drawdown_pct()
        daily_pnl_usd = self._compute_daily_pnl_usd()

        triggered = False
        if self.kill_switch_enabled:
            if self.max_drawdown_pct > 0 and drawdown_pct >= self.max_drawdown_pct:
                self.activate_kill_switch(
                    f"max_drawdown_breached:{drawdown_pct:.4f}% >= {self.max_drawdown_pct:.4f}%"
                )
                triggered = True

            if self.daily_loss_limit_usd > 0 and daily_pnl_usd <= -self.daily_loss_limit_usd:
                self.activate_kill_switch(
                    f"daily_loss_breached:{daily_pnl_usd:.4f} <= -{self.daily_loss_limit_usd:.4f}"
                )
                triggered = True

            if (
                self.max_gross_exposure_usd > 0
                and self.current_gross_exposure_usd > self.max_gross_exposure_usd
            ):
                self.activate_kill_switch(
                    "exposure_breached:"
                    f"{self.current_gross_exposure_usd:.4f} > {self.max_gross_exposure_usd:.4f}"
                )
                triggered = True

        return RiskSnapshot(
            timestamp=ts_str,
            equity_usd=float(self.current_equity_usd),
            peak_equity_usd=float(self.peak_equity_usd),
            gross_exposure_usd=float(self.current_gross_exposure_usd),
            drawdown_pct=float(drawdown_pct),
            daily_pnl_usd=float(daily_pnl_usd),
            kill_switch_active=bool(self.kill_switch_active),
            kill_switch_reason=self.kill_switch_reason,
            kill_switch_triggered=triggered,
        )

    def pre_trade_check(
        self,
        *,
        order_notional_usd: float,
        reduces_exposure: bool = False,
    ) -> RiskCheckResult:
        """
        Enforce hard limits before every order.
        """
        notional = max(0.0, float(order_notional_usd))
        current_exposure = float(self.current_gross_exposure_usd)
        projected = max(0.0, current_exposure - notional) if reduces_exposure else current_exposure + notional

        if self.kill_switch_active:
            if reduces_exposure and self.allow_risk_reducing_orders:
                return RiskCheckResult(
                    allowed=True,
                    reason=None,
                    projected_exposure_usd=projected,
                    kill_switch_active=True,
                )
            return RiskCheckResult(
                allowed=False,
                reason=f"kill_switch_active:{self.kill_switch_reason or 'manual'}",
                projected_exposure_usd=projected,
                kill_switch_active=True,
            )

        if (
            self.max_gross_exposure_usd > 0
            and not reduces_exposure
            and projected > self.max_gross_exposure_usd
        ):
            return RiskCheckResult(
                allowed=False,
                reason=(
                    "max_gross_exposure_breached:"
                    f"{projected:.4f} > {self.max_gross_exposure_usd:.4f}"
                ),
                projected_exposure_usd=projected,
                kill_switch_active=False,
            )

        return RiskCheckResult(
            allowed=True,
            reason=None,
            projected_exposure_usd=projected,
            kill_switch_active=False,
        )

