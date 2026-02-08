import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.risk_engine import LiveRiskEngine


def test_drawdown_breach_triggers_kill_switch():
    engine = LiveRiskEngine(
        max_drawdown_pct=10.0,
        max_gross_exposure_usd=1000.0,
        daily_loss_limit_usd=500.0,
        kill_switch_enabled=True,
    )
    t0 = datetime(2026, 2, 8, 0, 0, 0)

    snap1 = engine.update_portfolio_state(
        equity_usd=1000.0,
        gross_exposure_usd=200.0,
        timestamp=t0.isoformat(),
    )
    assert snap1.kill_switch_active is False
    assert snap1.drawdown_pct == 0.0

    snap2 = engine.update_portfolio_state(
        equity_usd=880.0,
        gross_exposure_usd=200.0,
        timestamp=(t0 + timedelta(minutes=1)).isoformat(),
    )
    assert snap2.kill_switch_active is True
    assert "max_drawdown_breached" in (snap2.kill_switch_reason or "")


def test_daily_loss_limit_triggers_kill_switch():
    engine = LiveRiskEngine(
        max_drawdown_pct=90.0,
        max_gross_exposure_usd=1000.0,
        daily_loss_limit_usd=50.0,
        kill_switch_enabled=True,
    )
    t0 = datetime(2026, 2, 8, 0, 0, 0)
    engine.update_portfolio_state(equity_usd=500.0, gross_exposure_usd=100.0, timestamp=t0.isoformat())

    snap = engine.update_portfolio_state(
        equity_usd=440.0,
        gross_exposure_usd=100.0,
        timestamp=(t0 + timedelta(hours=1)).isoformat(),
    )
    assert snap.kill_switch_active is True
    assert "daily_loss_breached" in (snap.kill_switch_reason or "")


def test_pre_trade_blocks_projected_exposure_breach():
    engine = LiveRiskEngine(
        max_drawdown_pct=50.0,
        max_gross_exposure_usd=250.0,
        daily_loss_limit_usd=0.0,
        kill_switch_enabled=True,
    )
    engine.update_portfolio_state(
        equity_usd=1000.0,
        gross_exposure_usd=200.0,
        timestamp=datetime.utcnow().isoformat(),
    )

    check = engine.pre_trade_check(order_notional_usd=60.0, reduces_exposure=False)
    assert check.allowed is False
    assert "max_gross_exposure_breached" in (check.reason or "")


def test_kill_switch_still_allows_risk_reducing_orders():
    engine = LiveRiskEngine(
        max_drawdown_pct=10.0,
        max_gross_exposure_usd=1000.0,
        daily_loss_limit_usd=500.0,
        kill_switch_enabled=True,
        allow_risk_reducing_orders=True,
    )
    t0 = datetime(2026, 2, 8, 0, 0, 0)
    engine.update_portfolio_state(equity_usd=1000.0, gross_exposure_usd=300.0, timestamp=t0.isoformat())
    engine.update_portfolio_state(
        equity_usd=850.0,
        gross_exposure_usd=300.0,
        timestamp=(t0 + timedelta(minutes=1)).isoformat(),
    )
    assert engine.kill_switch_active is True

    reducing = engine.pre_trade_check(order_notional_usd=50.0, reduces_exposure=True)
    assert reducing.allowed is True

    increasing = engine.pre_trade_check(order_notional_usd=50.0, reduces_exposure=False)
    assert increasing.allowed is False
    assert "kill_switch_active" in (increasing.reason or "")

