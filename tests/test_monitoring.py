import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.monitoring import EdgeDecayMonitor


def test_edge_monitor_derisks_strategy_on_mild_decay():
    monitor = EdgeDecayMonitor(
        window_size=20,
        min_samples=5,
        derisk_hit_rate_threshold=0.60,
        disable_hit_rate_threshold=0.20,
        derisk_mean_return_threshold=-0.0001,
        derisk_size_multiplier=0.4,
        disable_sticky=True,
    )

    for realized_return in [-0.010, -0.005, 0.002, -0.003, 0.001]:
        status_map = monitor.record_outcomes({"simple": "BUY"}, realized_return)

    simple_status = status_map["simple"]
    assert simple_status.sample_count == 5
    assert simple_status.state == "derisked"
    assert simple_status.disable_trading is False
    assert simple_status.size_multiplier == 0.4


def test_edge_monitor_disables_strategy_on_severe_decay():
    monitor = EdgeDecayMonitor(
        window_size=10,
        min_samples=5,
        derisk_hit_rate_threshold=0.50,
        disable_hit_rate_threshold=0.30,
        derisk_mean_return_threshold=-0.0001,
        derisk_size_multiplier=0.5,
        disable_sticky=True,
    )

    for realized_return in [-0.01, -0.02, -0.015, -0.008, -0.012]:
        status_map = monitor.record_outcomes({"technical": "BUY"}, realized_return)

    tech_status = status_map["technical"]
    assert tech_status.sample_count == 5
    assert tech_status.state == "disabled"
    assert tech_status.disable_trading is True
    assert tech_status.size_multiplier == 0.0


def test_sell_signals_score_correctly_in_down_moves():
    monitor = EdgeDecayMonitor(
        window_size=10,
        min_samples=3,
        derisk_hit_rate_threshold=0.40,
        disable_hit_rate_threshold=0.20,
        derisk_mean_return_threshold=-0.0001,
        derisk_size_multiplier=0.5,
        disable_sticky=True,
    )

    for realized_return in [-0.01, -0.02, -0.005]:
        status_map = monitor.record_outcomes({"custom": "SELL"}, realized_return)

    custom_status = status_map["custom"]
    assert custom_status.sample_count == 3
    assert custom_status.win_rate == 1.0
    assert custom_status.state == "healthy"
    assert custom_status.disable_trading is False


def test_disable_state_is_sticky_when_configured():
    monitor = EdgeDecayMonitor(
        window_size=20,
        min_samples=4,
        derisk_hit_rate_threshold=0.60,
        disable_hit_rate_threshold=0.40,
        derisk_mean_return_threshold=-0.0001,
        derisk_size_multiplier=0.5,
        disable_sticky=True,
    )

    for realized_return in [-0.01, -0.02, -0.01, -0.02]:
        monitor.record_outcomes({"simple": "BUY"}, realized_return)

    disabled_status = monitor.get_status("simple")
    assert disabled_status.state == "disabled"

    # Even with later positive outcomes, sticky disable keeps state disabled.
    for realized_return in [0.02, 0.02, 0.02, 0.02]:
        monitor.record_outcomes({"simple": "BUY"}, realized_return)

    still_disabled = monitor.get_status("simple")
    assert still_disabled.state == "disabled"
    assert still_disabled.disable_trading is True


def test_clear_strategy_removes_history_and_state():
    monitor = EdgeDecayMonitor(min_samples=2)
    monitor.record_outcomes({"simple": "BUY"}, -0.01)
    assert "simple" in monitor.get_all_statuses()

    monitor.clear_strategy("simple")
    assert "simple" not in monitor.get_all_statuses()
