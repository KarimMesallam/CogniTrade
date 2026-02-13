import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from binance.exceptions import BinanceAPIException, BinanceRequestException
import pandas as pd
from bot.config import (
    SYMBOL, TESTNET, TRADING_CONFIG,
    get_trading_parameter, get_loop_interval,
    get_consensus_method, is_llm_agreement_required, is_live_trading_enabled,
    get_trade_mode, is_futures_short_enabled, get_regime_config, get_policy_config,
    get_data_pipeline_config, get_risk_engine_config, get_reconciliation_config,
    get_observability_config, get_monitoring_config, get_rollout_config,
    get_notification_config,
)
from bot.notifications import get_notifier
from bot.strategy import get_all_strategy_signals, simple_signal, technical_analysis_signal
from bot.binance_api import (
    place_market_buy,
    place_market_sell,
    client,
    get_recent_closes,
    synchronize_time,
    get_account_balance,
    get_futures_mark_price,
)
from bot.llm_manager import get_decision_from_llm, log_decision_with_context, LLMManager
from bot.order_manager import OrderManager
from bot.db_integration import DatabaseIntegration
from bot.portfolio import PortfolioOptimizer, PortfolioConstraints, PortfolioAllocationResult
from bot.regime import MarketRegimeDetector, RegimeState, REGIME_UNKNOWN
from bot.policy import RegimePolicyEngine, PolicyDecision
from bot.data_pipeline import PointInTimeDataPipeline, find_lookahead_violations
from bot.risk_engine import LiveRiskEngine
from bot.observability import get_observability_manager
from bot.monitoring import EdgeDecayMonitor
from bot.deploy_policy import ShadowCanaryRolloutGate, thresholds_from_config
import json
import uuid
import asyncio
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cognitrade.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_bot")

# Set this to True when running tests to bypass sleeps
TESTING_MODE = os.environ.get('TESTING_MODE', '0') == '1'


def optimize_portfolio_from_returns(
    asset_returns: Dict[str, list],
    method: str = "risk_parity",
    risk_budgets: Optional[Dict[str, float]] = None,
    constraints: Optional[Dict[str, Any]] = None,
) -> PortfolioAllocationResult:
    """
    Build a constrained multi-asset allocation from historical return series.
    """
    returns_df = pd.DataFrame(asset_returns)
    optimizer = PortfolioOptimizer(
        PortfolioConstraints(**(constraints or {}))
    )
    return optimizer.optimize(
        returns=returns_df,
        method=method,
        risk_budgets=risk_budgets,
    )


def build_portfolio_targets(
    total_capital: float,
    allocation: PortfolioAllocationResult,
    latest_prices: Dict[str, float],
    min_notional: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    """
    Convert optimized weights into per-asset notional and unit targets.
    """
    optimizer = PortfolioOptimizer()
    return optimizer.allocate_units(
        total_capital=total_capital,
        weights=allocation.weights,
        prices=latest_prices,
        min_notional=min_notional,
    )

def handle_testnet_balance(symbol='BTC'):
    """
    Check if we're using testnet and if balances are too low.
    
    For Binance Testnet, if balance is zero, we inform the user about how to get more test funds.
    Testnet accounts are automatically funded upon creation but may be reset periodically.
    
    Args:
        symbol: Asset symbol to check (default: BTC)
        
    Returns:
        bool: True if balance is sufficient or handled successfully
    """
    if not TESTNET:
        return True
    
    try:
        # Check balance for the specific asset
        balance = get_account_balance(symbol)
        if balance and balance.get('free', 0) > 0:
            logger.info(f"Testnet {symbol} balance: {balance['free']}")
            return True
        
        # If balance is zero or not found, provide instructions to the user
        logger.warning(f"Your Testnet {symbol} balance is zero or insufficient.")
        logger.info("===== BINANCE TESTNET FUND INFORMATION =====")
        logger.info("The Binance Testnet resets periodically (typically once a month).")
        logger.info("After a reset, your test funds should be automatically replenished.")
        logger.info("Options to get more test funds:")
        logger.info("1. Visit https://testnet.binance.vision/ and login with GitHub")
        logger.info("2. Wait for the next scheduled reset of the testnet")
        logger.info("3. Create a new API key which may trigger a balance refresh")
        logger.info("Note: There is no direct API method to request more test funds.")
        logger.info("==============================================")
        
        # We return True since this is just informational and shouldn't stop the bot
        return True
    except Exception as e:
        logger.error(f"Error checking testnet balance: {e}")
        return False


def _extract_last_price_from_market_data(market_data: Dict[str, Any], symbol: str) -> Optional[float]:
    """Best-effort last-trade/close price extraction from market payload."""
    candles = (market_data or {}).get("candles") or []
    if candles:
        last = candles[-1]
        try:
            if isinstance(last, (list, tuple)) and len(last) > 4:
                return float(last[4])
            if isinstance(last, dict) and "close" in last:
                return float(last["close"])
        except Exception:
            pass

    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        if ticker and "price" in ticker:
            return float(ticker["price"])
    except Exception:
        pass

    return None


_last_good_portfolio_state: Optional[Dict[str, float]] = None


def _fetch_balance_with_retry(asset: str) -> Optional[Dict[str, float]]:
    """Fetch balance for an asset, retrying once on failure."""
    result = get_account_balance(asset)
    if result is not None:
        return result
    time.sleep(0.5)
    return get_account_balance(asset)


def estimate_portfolio_risk_state(
    symbol: str,
    market_data: Dict[str, Any],
    order_manager: OrderManager,
    trade_mode: str,
) -> Dict[str, float]:
    """
    Estimate live portfolio equity and gross exposure for hard risk checks.
    """
    global _last_good_portfolio_state
    price = _extract_last_price_from_market_data(market_data, symbol) or 0.0
    quote_asset = getattr(order_manager, "quote_asset", "USDT") or "USDT"
    base_asset = getattr(order_manager, "base_asset", symbol.replace("USDT", ""))

    if trade_mode == "FUTURES":
        quote_balance = _fetch_balance_with_retry(quote_asset) or _fetch_balance_with_retry("USDT") or {}
        equity_usd = float(quote_balance.get("total", quote_balance.get("free", 0.0)) or 0.0)
        position_qty = order_manager.get_position_quantity()
        mark_price = get_futures_mark_price(symbol) or price or 0.0
        gross_exposure_usd = abs(float(position_qty or 0.0)) * float(mark_price)
        result = {
            "equity_usd": float(equity_usd),
            "gross_exposure_usd": float(gross_exposure_usd),
            "reference_price": float(mark_price),
        }
        if equity_usd > 0:
            _last_good_portfolio_state = result
        elif _last_good_portfolio_state is not None:
            logger.warning("Balance read returned zero equity; using last known good state")
            return _last_good_portfolio_state
        return result

    quote_balance = _fetch_balance_with_retry(quote_asset) or {}
    base_balance = _fetch_balance_with_retry(base_asset) or {}
    quote_total = float(quote_balance.get("total", quote_balance.get("free", 0.0)) or 0.0)
    base_total = float(base_balance.get("total", base_balance.get("free", 0.0)) or 0.0)
    equity_usd = quote_total + (base_total * float(price))
    gross_exposure_usd = abs(base_total * float(price))
    result = {
        "equity_usd": float(equity_usd),
        "gross_exposure_usd": float(gross_exposure_usd),
        "reference_price": float(price),
    }
    if equity_usd > 0:
        _last_good_portfolio_state = result
    elif _last_good_portfolio_state is not None:
        logger.warning("Balance read returned zero equity; using last known good state")
        return _last_good_portfolio_state
    return result

def enforce_live_trading_safety_gate():
    """
    Enforce explicit opt-in for live trading.

    Returns:
        bool: True when trading mode is allowed, otherwise False.
    """
    if TESTNET:
        return True

    if is_live_trading_enabled():
        logger.warning("Live trading explicitly enabled (TESTNET=False and ENABLE_LIVE_TRADING=True).")
        return True

    logger.critical(
        "Live trading safety gate blocked startup: TESTNET=False while ENABLE_LIVE_TRADING is not enabled."
    )
    logger.critical("Set ENABLE_LIVE_TRADING=True only after completing production readiness checks.")
    return False


def enforce_rollout_production_gate():
    """
    Enforce mandatory rollout-gate approval before runtime trading when configured.
    """
    rollout_config = get_rollout_config()
    if not rollout_config.get("enabled", True):
        return True
    if not rollout_config.get("enforce_production_gate", False):
        return True

    required_rollout_id = str(rollout_config.get("required_rollout_id", "")).strip()
    if not required_rollout_id:
        logger.critical("Rollout gate enforcement enabled but ROLLOUT_REQUIRED_ID is empty.")
        return False

    gate = ShadowCanaryRolloutGate(thresholds=thresholds_from_config(rollout_config))
    state_store_path = str(rollout_config.get("state_store_path", "data/rollout_gate_state.json"))
    try:
        gate.load_from_file(state_store_path)
    except Exception as exc:
        logger.critical("Failed to load rollout state from %s: %s", state_store_path, exc)
        return False

    status = gate.get_rollout_status(required_rollout_id)
    if not bool(status.get("production_passed", False)):
        logger.critical(
            "Production rollout gate blocked startup for rollout_id=%s. "
            "Run shadow/canary/production gate workflow first.",
            required_rollout_id,
        )
        return False
    return True


def initialize_bot():
    """Initialize the trading bot and verify connectivity."""
    try:
        if not enforce_live_trading_safety_gate():
            return False
        if not enforce_rollout_production_gate():
            return False

        # Synchronize time with Binance server
        synchronize_time()
        
        # Test API connection
        server_time = client.get_server_time()
        logger.info(f"Connected to Binance {'Testnet' if TESTNET else 'Live'} successfully")
        logger.info(f"Server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
        
        # Get account information
        balances = get_account_balance()
        if balances:
            non_zero_assets = [
                asset
                for asset, payload in balances.items()
                if float(payload.get("total", payload.get("free", 0.0)) or 0.0) > 0
            ]
            preview = sorted(non_zero_assets)[:10]
            logger.info(
                "Account balance snapshot received | non_zero_assets=%s preview=%s",
                len(non_zero_assets),
                preview,
            )
        else:
            logger.warning("Could not retrieve account balances")
        
        # Check if trading the configured symbol is possible
        symbol_info = client.get_symbol_info(SYMBOL)
        if not symbol_info:
            logger.error(f"Symbol {SYMBOL} not found or not available for trading")
            return False
        
        # Check and handle testnet balance if necessary
        base_asset = SYMBOL.replace('USDT', '')  # Extract BTC from BTCUSDT
        if TESTNET:
            handle_testnet_balance(base_asset)
        
        # Initialize database
        try:
            db = DatabaseIntegration()
            db.add_system_alert(
                message=f"Bot initialized for trading {SYMBOL} on Binance {'Testnet' if TESTNET else 'Live'}",
                alert_type="info",
                severity="low"
            )
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            logger.warning("Continuing without database support")
        
        logger.info(f"Bot initialized successfully for trading {SYMBOL}")
        return True
    except (BinanceAPIException, BinanceRequestException) as e:
        logger.error(f"Failed to initialize bot: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        return False

def get_market_data(symbol, interval='1m'):
    """Fetch current market data for analysis."""
    try:
        # Get recent candles
        candles = client.get_klines(symbol=symbol, interval=interval, limit=20)
        
        # Get order book
        order_book = client.get_order_book(symbol=symbol, limit=5)
        
        # Get recent trades
        recent_trades = client.get_recent_trades(symbol=symbol, limit=5)
        
        market_data = {
            'symbol': symbol,
            'candles': candles,
            'order_book': order_book,
            'recent_trades': recent_trades,
            'timestamp': datetime.now().isoformat()
        }
        
        return market_data
    except (BinanceAPIException, BinanceRequestException) as e:
        logger.error(f"Error fetching market data: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching market data: {e}")
        return None

def get_signal_consensus(signals, strategy_weights: Optional[Dict[str, float]] = None):
    """
    Determine a consensus signal from multiple strategies.
    
    Args:
        signals: Dictionary of strategy name -> signal value
        
    Returns:
        String: 'BUY', 'SELL', or 'HOLD'
    """
    if not signals:
        return "HOLD"

    # Get consensus method from config
    consensus_method = get_consensus_method()
    
    if consensus_method == "simple_majority":
        # Simple majority vote
        buy_count = sum(1 for signal in signals.values() if signal == "BUY")
        sell_count = sum(1 for signal in signals.values() if signal == "SELL")
        
        # Simple majority rule with bias toward HOLD if tied
        if buy_count > sell_count:
            return "BUY"
        elif sell_count > buy_count:
            return "SELL"
        else:
            return "HOLD"
            
    elif consensus_method == "weighted_majority":
        # Weighted voting based on strategy weights
        buy_weight = 0
        sell_weight = 0
        hold_weight = 0
        
        for strategy_name, signal in signals.items():
            # Allow policy to provide dynamic weight overrides per strategy.
            if strategy_weights and strategy_name in strategy_weights:
                weight = float(strategy_weights.get(strategy_name, 1.0))
            else:
                strategy_config = TRADING_CONFIG["strategies"].get(strategy_name, {})
                weight = float(strategy_config.get("weight", 1.0))
            
            if signal == "BUY":
                buy_weight += weight
            elif signal == "SELL":
                sell_weight += weight
            else:
                hold_weight += weight
        
        # Determine the consensus based on weighted votes
        if buy_weight > sell_weight and buy_weight > hold_weight:
            return "BUY"
        elif sell_weight > buy_weight and sell_weight > hold_weight:
            return "SELL"
        else:
            return "HOLD"
            
    elif consensus_method == "unanimous":
        # All signals must agree for action
        if all(signal == "BUY" for signal in signals.values()):
            return "BUY"
        elif all(signal == "SELL" for signal in signals.values()):
            return "SELL"
        else:
            return "HOLD"
    
    else:
        # Default to simple majority if method is not recognized
        logger.warning(f"Unrecognized consensus method '{consensus_method}', using simple majority")
        buy_count = sum(1 for signal in signals.values() if signal == "BUY")
        sell_count = sum(1 for signal in signals.values() if signal == "SELL")
        
        if buy_count > sell_count:
            return "BUY"
        elif sell_count > buy_count:
            return "SELL"
        else:
            return "HOLD"

def execute_trade(
    signals,
    llm_decision,
    symbol,
    market_data,
    order_manager,
    db_integration=None,
    strategy_weights: Optional[Dict[str, float]] = None,
    position_size_multiplier: float = 1.0,
    risk_engine: Optional[LiveRiskEngine] = None,
):
    """Execute a trade based on signals and decisions."""
    try:
        # Get consensus from signals
        signal_consensus = get_signal_consensus(signals, strategy_weights=strategy_weights)
        logger.info(f"Signal consensus: {signal_consensus}")
        
        # Log the decision with full context for later analysis
        log_decision_with_context(llm_decision, signals, market_data)
        
        # Save signals to database if available
        signal_ids = {}
        if db_integration:
            for strategy_name, signal_value in signals.items():
                # Get timeframe from the strategy configuration
                strategy_config = TRADING_CONFIG["strategies"].get(strategy_name, {})
                timeframe = strategy_config.get("timeframe", "1m")
                
                # Extract LLM model responses if available in market_data
                llm_data = None
                if 'llm_result' in market_data:
                    llm_data = {
                        'primary_model_response': market_data['llm_result'].get('primary_model_response', ''),
                        'secondary_model_response': market_data['llm_result'].get('secondary_model_response', '')
                    }
                
                signal_ids[strategy_name] = db_integration.save_signal(
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy=strategy_name,
                    signal=signal_value,
                    llm_decision=llm_decision,
                    llm_data=llm_data
                )
        
        # Check if LLM agreement is required from config
        llm_required = is_llm_agreement_required()
        
        # Determine if we should execute a trade
        llm_upper = (llm_decision or "").upper()
        execute_buy = signal_consensus == "BUY" and (not llm_required or llm_upper == "BUY")
        execute_sell = signal_consensus == "SELL" and (not llm_required or llm_upper == "SELL")
        
        # Get execution mode and position sizing config.
        trade_mode = get_trade_mode()
        futures_shorts_enabled = is_futures_short_enabled()
        default_order_amount = float(get_trading_parameter("default_order_amount_usd", 10.0))
        multiplier = max(0.0, float(position_size_multiplier))
        effective_order_amount = default_order_amount * multiplier
        logger.info(
            "Effective order amount: base=%.4f multiplier=%.4f effective=%.4f",
            default_order_amount,
            multiplier,
            effective_order_amount,
        )

        if execute_buy and effective_order_amount <= 0:
            logger.info("Skipping BUY execution because effective order amount is zero after policy sizing.")
            return None
        if execute_sell and trade_mode == "FUTURES" and effective_order_amount <= 0:
            logger.info("Skipping futures short execution because effective order amount is zero after policy sizing.")
            return None

        market_price = _extract_last_price_from_market_data(market_data or {}, symbol) or 0.0

        def _check_risk(order_notional_usd: float, *, reduces_exposure: bool) -> bool:
            if not risk_engine:
                return True
            result = risk_engine.pre_trade_check(
                order_notional_usd=float(order_notional_usd),
                reduces_exposure=reduces_exposure,
            )
            if result.allowed:
                return True
            logger.warning(
                "Risk engine rejected order for %s: %s (projected exposure %.4f)",
                symbol,
                result.reason,
                float(result.projected_exposure_usd),
            )
            if db_integration:
                db_integration.add_system_alert(
                    message=f"Risk engine rejected order for {symbol}: {result.reason}",
                    alert_type="warning",
                    severity="high",
                    data={
                        "order_notional_usd": float(order_notional_usd),
                        "projected_exposure_usd": float(result.projected_exposure_usd),
                        "kill_switch_active": bool(result.kill_switch_active),
                        "reduces_exposure": bool(reduces_exposure),
                    },
                )
            return False

        def _link_signals_to_trade(order_payload):
            if db_integration and order_payload and 'trade_id' in order_payload and signal_ids:
                for signal_id in signal_ids.values():
                    if signal_id > 0:
                        db_integration.link_signal_to_trade(signal_id, order_payload['trade_id'])

        if trade_mode == "FUTURES":
            default_futures_leverage = get_trading_parameter("default_futures_leverage", 2.0)

            current_position = order_manager.get_position_quantity()
            if current_position is None:
                logger.warning("Unable to determine current futures position; skipping trade.")
                return None

            if execute_sell:
                # Close long if currently long
                if current_position > 0:
                    close_notional = float(current_position) * float(market_price)
                    if not _check_risk(close_notional, reduces_exposure=True):
                        return None
                    close_order = order_manager.execute_market_close_long(abs(current_position))
                    if not close_order:
                        reject_reason = getattr(order_manager, "last_reject_reason", None)
                        if reject_reason:
                            logger.warning("Futures close long rejected: %s", reject_reason)
                        logger.warning("Futures close long execution failed")
                        return None
                    logger.info("Futures close long executed: %s", close_order.get('orderId'))
                    _link_signals_to_trade(close_order)
                    # Fall through to open short

                # Skip if already short
                if current_position < 0:
                    logger.info("Already short for %s; skipping SELL signal.", symbol)
                    return None

                # Open short
                if not futures_shorts_enabled:
                    logger.warning(
                        "TRADE_MODE is FUTURES but ENABLE_FUTURES_SHORTS is false; skipping short entry."
                    )
                    # Return close order if we closed a long above, else None
                    return locals().get("close_order")
                if not _check_risk(effective_order_amount, reduces_exposure=False):
                    return locals().get("close_order")

                order = order_manager.execute_market_short(
                    quote_amount=effective_order_amount,
                    leverage=default_futures_leverage,
                )
                if order:
                    logger.info("Futures short executed successfully: %s", order.get('orderId'))
                    _link_signals_to_trade(order)
                    return order

                reject_reason = getattr(order_manager, "last_reject_reason", None)
                if reject_reason:
                    logger.warning("Futures short rejected by risk checks: %s", reject_reason)
                logger.warning("Futures short execution failed")
                # If we closed a long but failed to open short, return the close order (safe flat state)
                return locals().get("close_order")

            elif execute_buy:
                # Cover short if currently short
                if current_position < 0:
                    cover_notional = abs(float(current_position)) * float(market_price)
                    if not _check_risk(cover_notional, reduces_exposure=True):
                        return None
                    cover_order = order_manager.execute_market_cover(abs(current_position))
                    if not cover_order:
                        reject_reason = getattr(order_manager, "last_reject_reason", None)
                        if reject_reason:
                            logger.warning("Futures short cover rejected: %s", reject_reason)
                        logger.warning("Futures short cover execution failed")
                        return None
                    logger.info("Futures short cover executed: %s", cover_order.get('orderId'))
                    _link_signals_to_trade(cover_order)
                    # Fall through to open long

                # Skip if already long
                if current_position > 0:
                    logger.info("Already long for %s; skipping BUY signal.", symbol)
                    return None

                # Open long
                if not _check_risk(effective_order_amount, reduces_exposure=False):
                    return locals().get("cover_order")

                order = order_manager.execute_market_long(
                    quote_amount=effective_order_amount,
                    leverage=default_futures_leverage,
                )
                if order:
                    logger.info("Futures long executed successfully: %s", order.get('orderId'))
                    _link_signals_to_trade(order)
                    return order

                reject_reason = getattr(order_manager, "last_reject_reason", None)
                if reject_reason:
                    logger.warning("Futures long rejected by risk checks: %s", reject_reason)
                logger.warning("Futures long execution failed")
                # If we covered a short but failed to open long, return cover order (safe flat state)
                return locals().get("cover_order")

            else:
                logger.info(
                    "No trade executed. Signal consensus: %s, LLM decision: %s",
                    signal_consensus,
                    llm_decision,
                )
        else:
            if execute_buy:
                logger.info(f"Executing BUY for {symbol}")
                if not _check_risk(effective_order_amount, reduces_exposure=False):
                    return None
                order = order_manager.execute_market_buy(quote_amount=effective_order_amount)
                if order:
                    logger.info(f"Buy order executed successfully: {order['orderId']}")
                    _link_signals_to_trade(order)
                    return order

                reject_reason = getattr(order_manager, "last_reject_reason", None)
                if reject_reason:
                    logger.warning(f"Buy order rejected by risk checks: {reject_reason}")
                logger.warning("Buy order execution failed")

            elif execute_sell:
                logger.info(f"Executing SELL for {symbol}")
                balance = get_account_balance(symbol.replace('USDT', ''))

                if balance and balance.get('free', 0) > 0:
                    quantity = balance['free']
                    sell_notional = float(quantity) * float(market_price)
                    if not _check_risk(sell_notional, reduces_exposure=True):
                        return None
                    order = order_manager.execute_market_sell(quantity)
                    if order:
                        logger.info(f"Sell order executed successfully: {order['orderId']}")
                        _link_signals_to_trade(order)
                        return order

                    reject_reason = getattr(order_manager, "last_reject_reason", None)
                    if reject_reason:
                        logger.warning(f"Sell order rejected by risk checks: {reject_reason}")
                    logger.warning("Sell order execution failed")
                else:
                    logger.warning(f"No balance available to sell for {symbol.replace('USDT', '')}")

            else:
                logger.info(
                    "No trade executed. Signal consensus: %s, LLM decision: %s",
                    signal_consensus,
                    llm_decision,
                )
        
        return None
    except (BinanceAPIException, BinanceRequestException) as e:
        logger.error(f"Error executing trade: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during trade execution: {e}")
        return None

def trading_loop():
    """Main trading loop."""
    if not enforce_live_trading_safety_gate():
        raise RuntimeError("Live trading safety gate blocked trading loop startup")
    if not enforce_rollout_production_gate():
        raise RuntimeError("Rollout production gate blocked trading loop startup")

    logger.info("Starting trading loop...")
    
    # Initialize database integration
    db_integration = None
    try:
        db_integration = DatabaseIntegration()
        logger.info("Database integration initialized for trading loop")
        
        # Log start of trading session
        db_integration.add_system_alert(
            message="Trading session started",
            alert_type="info",
            severity="low",
            data={"symbol": SYMBOL, "testnet": TESTNET}
        )
    except Exception as e:
        logger.error(f"Failed to initialize database integration: {e}")
        logger.warning("Continuing without database support")
    
    # Initialize order manager with risk percentage from config
    risk_percentage = get_trading_parameter("risk_percentage", 1.0)
    max_order_notional_usd = get_trading_parameter(
        "max_order_notional_usd",
        get_trading_parameter("max_order_amount_usd", 100.0)
    )
    max_position_exposure_usd = get_trading_parameter("max_position_exposure_usd", 250.0)
    trade_mode = get_trade_mode()
    enable_futures_shorts = is_futures_short_enabled()
    max_short_notional_usd = get_trading_parameter("max_short_notional_usd", max_order_notional_usd)
    default_futures_leverage = get_trading_parameter("default_futures_leverage", 2.0)
    max_short_leverage = get_trading_parameter("max_short_leverage", 3.0)
    min_short_liquidation_buffer_pct = get_trading_parameter("min_short_liquidation_buffer_pct", 20.0)
    order_manager = OrderManager(
        SYMBOL,
        risk_percentage=risk_percentage,
        use_database=db_integration is not None,
        max_order_notional_usd=max_order_notional_usd,
        max_position_exposure_usd=max_position_exposure_usd,
        trade_mode=trade_mode,
        enable_futures_shorts=enable_futures_shorts,
        max_short_notional_usd=max_short_notional_usd,
        default_futures_leverage=default_futures_leverage,
        max_short_leverage=max_short_leverage,
        min_short_liquidation_buffer_pct=min_short_liquidation_buffer_pct,
    )
    logger.info(
        "Order manager initialized for %s | mode=%s | futures_shorts=%s | risk=%s%% | max order notional %.2f | max exposure %.2f | max short notional %.2f | max short leverage %.2f | min short liq buffer %.2f%%",
        SYMBOL,
        trade_mode,
        "enabled" if enable_futures_shorts else "disabled",
        risk_percentage,
        float(max_order_notional_usd),
        float(max_position_exposure_usd),
        float(max_short_notional_usd),
        float(max_short_leverage),
        float(min_short_liquidation_buffer_pct),
    )
    
    # Initialize LLM manager
    llm_manager = LLMManager()

    # Initialize regime detector and regime-policy engine.
    regime_config = get_regime_config()
    regime_detector = MarketRegimeDetector(
        lookback_candles=regime_config.get("lookback_candles", 50),
        trend_threshold_pct=regime_config.get("trend_threshold_pct", 0.02),
        sideways_threshold_pct=regime_config.get("sideways_threshold_pct", 0.01),
        high_volatility_threshold_pct=regime_config.get("high_volatility_threshold_pct", 0.015),
    )
    policy_engine = RegimePolicyEngine(
        strategy_config=TRADING_CONFIG.get("strategies", {}),
        policy_config=get_policy_config(),
    )

    data_pipeline_config = get_data_pipeline_config()
    pit_pipeline = None
    if data_pipeline_config.get("enabled", True):
        pit_pipeline = PointInTimeDataPipeline(
            require_monotonic_timestamps=data_pipeline_config.get("require_monotonic_timestamps", True),
            source=data_pipeline_config.get("feature_source", "exchange_ohlcv"),
        )

    risk_engine_config = get_risk_engine_config()
    risk_engine = None
    if risk_engine_config.get("enabled", True):
        risk_engine = LiveRiskEngine(
            max_drawdown_pct=risk_engine_config.get("max_drawdown_pct", 20.0),
            max_gross_exposure_usd=risk_engine_config.get("max_gross_exposure_usd", 0.0),
            daily_loss_limit_usd=risk_engine_config.get("daily_loss_limit_usd", 0.0),
            kill_switch_enabled=risk_engine_config.get("kill_switch_enabled", True),
            allow_risk_reducing_orders=risk_engine_config.get("allow_risk_reducing_orders", True),
        )
        logger.info(
            "Risk engine enabled | max_drawdown=%.2f%% max_exposure=%.2f daily_loss=%.2f kill_switch=%s",
            float(risk_engine.max_drawdown_pct),
            float(risk_engine.max_gross_exposure_usd),
            float(risk_engine.daily_loss_limit_usd),
            "enabled" if risk_engine.kill_switch_enabled else "disabled",
        )

    reconciliation_config = get_reconciliation_config()
    reconciliation_enabled = reconciliation_config.get("enabled", True)
    reconciliation_interval_loops = max(1, int(reconciliation_config.get("interval_loops", 5)))
    reconciliation_position_tolerance = float(reconciliation_config.get("position_tolerance", 0.000001))
    if reconciliation_enabled and reconciliation_config.get("run_on_startup", True):
        startup_report = order_manager.recover_state_from_exchange()
        logger.info("Startup reconciliation report: %s", startup_report)
        if db_integration:
            db_integration.save_reconciliation_event(
                {
                    "symbol": startup_report.get("symbol", SYMBOL),
                    "trade_mode": startup_report.get("trade_mode", trade_mode),
                    "reconciled_at": startup_report.get("reconciled_at", datetime.utcnow().isoformat()),
                    "local_active_count": len(order_manager.get_active_orders()),
                    "exchange_open_count": startup_report.get("exchange_open_count", 0),
                    "stale_local_order_ids": [],
                    "missing_local_order_ids": [],
                    "synced_local_order_ids": [],
                    "position_mismatch": False,
                    "position_delta": 0.0,
                    "details": startup_report,
                }
            )

    observability_config = get_observability_config()
    observability = None
    if observability_config.get("enabled", True):
        observability = get_observability_manager(
            max_events=observability_config.get("max_events", 4000),
            latency_alert_ms=observability_config.get("latency_alert_ms", 2500.0),
            error_rate_alert_threshold=observability_config.get("error_rate_alert_threshold", 0.25),
            error_rate_min_events=observability_config.get("error_rate_min_events", 20),
            persistence_enabled=observability_config.get("persistence_enabled", True),
            persistence_db_url=observability_config.get("persistence_db_url", "sqlite:///data/observability.db"),
        )
        logger.info(
            "Observability enabled | max_events=%s latency_alert_ms=%.2f error_rate_threshold=%.2f min_events=%s",
            int(observability_config.get("max_events", 4000)),
            float(observability_config.get("latency_alert_ms", 2500.0)),
            float(observability_config.get("error_rate_alert_threshold", 0.25)),
            int(observability_config.get("error_rate_min_events", 20)),
        )

    monitoring_config = get_monitoring_config()
    edge_monitor = None
    if monitoring_config.get("enabled", True):
        edge_monitor = EdgeDecayMonitor(
            window_size=monitoring_config.get("window_size", 50),
            min_samples=monitoring_config.get("min_samples", 20),
            derisk_hit_rate_threshold=monitoring_config.get("derisk_hit_rate_threshold", 0.45),
            disable_hit_rate_threshold=monitoring_config.get("disable_hit_rate_threshold", 0.35),
            derisk_mean_return_threshold=monitoring_config.get("derisk_mean_return_threshold", -0.0002),
            derisk_size_multiplier=monitoring_config.get("derisk_size_multiplier", 0.5),
            disable_sticky=monitoring_config.get("disable_sticky", True),
        )
        logger.info(
            "Edge monitor enabled | window=%s min_samples=%s derisk<=%.2f disable<=%.2f derisk_multiplier=%.2f",
            int(monitoring_config.get("window_size", 50)),
            int(monitoring_config.get("min_samples", 20)),
            float(monitoring_config.get("derisk_hit_rate_threshold", 0.45)),
            float(monitoring_config.get("disable_hit_rate_threshold", 0.35)),
            float(monitoring_config.get("derisk_size_multiplier", 0.5)),
        )

    notification_config = get_notification_config()
    notifier = get_notifier(notification_config)
    notify_severities = {"critical", "high"}
    min_sev = str(notification_config.get("min_alert_severity", "high")).lower()
    if min_sev == "medium":
        notify_severities = {"critical", "high", "medium"}
    elif min_sev == "low":
        notify_severities = {"critical", "high", "medium", "low"}
    if notifier.enabled:
        notifier.send_lifecycle("started")
        logger.info("Telegram notifications enabled (min_severity=%s)", min_sev)

    def _flush_observability_alerts() -> None:
        if not observability or not db_integration:
            return
        for alert in observability.drain_pending_alerts(limit=50):
            severity = str(alert.get("severity", "medium")).lower()
            if severity not in {"low", "medium", "high", "critical"}:
                severity = "medium"
            db_integration.add_system_alert(
                message=f"[observability] {alert.get('message', 'telemetry alert')}",
                alert_type=str(alert.get("alert_type", "warning")),
                severity=severity,
                data=alert.get("details", {}),
            )
            if severity in notify_severities:
                notifier.send_alert(alert)

    previous_policy_signals: Dict[str, str] = {}
    previous_reference_price: Optional[float] = None
    last_edge_states: Dict[str, str] = {}
    
    # Track consecutive errors to implement exponential backoff
    consecutive_errors = 0
    max_consecutive_errors = TRADING_CONFIG["operation"].get("max_consecutive_errors", 5)
    max_backoff_seconds = TRADING_CONFIG["operation"].get("max_backoff_seconds", 3600)
    
    # Get loop interval from config
    loop_interval = get_loop_interval()
    loop_count = 0
    
    while True:
        loop_trace = None
        try:
            loop_count += 1
            if observability:
                loop_trace = observability.start_trace(
                    component="trading_loop",
                    operation="iteration",
                    metadata={"symbol": SYMBOL, "loop_count": loop_count},
                )
            if reconciliation_enabled and loop_count % reconciliation_interval_loops == 0:
                reconciliation_start = time.perf_counter()
                reconciliation_report = order_manager.reconcile_with_exchange(
                    position_tolerance=reconciliation_position_tolerance,
                )
                reconciliation_latency_ms = (time.perf_counter() - reconciliation_start) * 1000.0
                if observability:
                    observability.record_latency(
                        component="reconciliation",
                        operation="loop_reconcile",
                        latency_ms=reconciliation_latency_ms,
                        success="error" not in reconciliation_report,
                        trace_id=loop_trace.trace_id if loop_trace else None,
                    )
                logger.info("Periodic reconciliation report: %s", reconciliation_report)
                if db_integration and "error" not in reconciliation_report:
                    db_integration.save_reconciliation_event(reconciliation_report)
                if reconciliation_report.get("position_mismatch") and db_integration:
                    db_integration.add_system_alert(
                        message=(
                            f"Position mismatch detected for {SYMBOL}: "
                            f"delta={reconciliation_report.get('position_delta')}"
                        ),
                        alert_type="warning",
                        severity="medium",
                        data=reconciliation_report,
                    )
                if observability and reconciliation_report.get("position_mismatch"):
                    observability.record_error(
                        component="reconciliation",
                        error_type="position_mismatch",
                        message=(
                            f"{SYMBOL} position delta={reconciliation_report.get('position_delta')}"
                        ),
                        severity="medium",
                        metadata={"report": reconciliation_report},
                    )

            # Update statuses of existing orders
            updated_orders = order_manager.update_order_statuses()
            if updated_orders:
                logger.info(f"Updated {len(updated_orders)} order statuses")
            
            # Get current market data
            primary_timeframe = TRADING_CONFIG["timeframes"].get("primary", "1m")
            market_data_start = time.perf_counter()
            market_data = get_market_data(SYMBOL, primary_timeframe)
            if observability:
                observability.record_latency(
                    component="market_data",
                    operation="fetch",
                    latency_ms=(time.perf_counter() - market_data_start) * 1000.0,
                    success=market_data is not None,
                    trace_id=loop_trace.trace_id if loop_trace else None,
                )
            if not market_data:
                raise Exception("Failed to get market data")

            current_reference_price = _extract_last_price_from_market_data(market_data, SYMBOL) or 0.0
            edge_status_by_strategy = {}
            if edge_monitor and previous_policy_signals and previous_reference_price and previous_reference_price > 0:
                realized_return = (float(current_reference_price) / float(previous_reference_price)) - 1.0
                edge_status_by_strategy = edge_monitor.record_outcomes(
                    previous_policy_signals,
                    realized_return,
                    timestamp=market_data.get("timestamp"),
                )
                if observability:
                    observability.record_trade_decision(
                        symbol=SYMBOL,
                        signal_consensus="N/A",
                        llm_decision="N/A",
                        executed=False,
                        trade_mode=trade_mode,
                        strategies=list(previous_policy_signals.keys()),
                        trace_id=loop_trace.trace_id if loop_trace else None,
                        metadata={
                            "monitoring_update_only": True,
                            "realized_return": float(realized_return),
                        },
                    )
            elif edge_monitor:
                edge_status_by_strategy = edge_monitor.get_all_statuses(
                    timestamp=market_data.get("timestamp"),
                )
            
            # Save market data to database if available
            if db_integration:
                db_integration.save_market_data(market_data, SYMBOL, primary_timeframe)

            if pit_pipeline:
                try:
                    pit_rows = pit_pipeline.build_snapshot(
                        symbol=SYMBOL,
                        timeframe=primary_timeframe,
                        candles=market_data.get("candles", []),
                        decision_timestamp=market_data.get("timestamp"),
                    )
                    leakage_rows = find_lookahead_violations(pit_rows)
                    if leakage_rows:
                        raise RuntimeError(
                            f"Detected {len(leakage_rows)} PIT leakage rows for {SYMBOL} {primary_timeframe}"
                        )
                    if db_integration:
                        db_integration.save_feature_snapshots(pit_rows)
                        persisted_leaks = db_integration.get_feature_leakage_violations(
                            symbol=SYMBOL,
                            timeframe=primary_timeframe,
                        )
                        if persisted_leaks:
                            raise RuntimeError(
                                f"Persisted PIT leakage check failed with {len(persisted_leaks)} violating rows."
                            )
                    market_data["pit_features"] = {
                        row["feature_name"]: row["feature_value"] for row in pit_rows
                    }
                except Exception as pit_error:
                    logger.error("PIT pipeline failed; skipping trade iteration: %s", pit_error)
                    if db_integration:
                        db_integration.add_system_alert(
                            message=f"PIT pipeline failure for {SYMBOL}: {pit_error}",
                            alert_type="error",
                            severity="high",
                            data={"symbol": SYMBOL, "timeframe": primary_timeframe},
                        )
                    continue

            if risk_engine:
                portfolio_state = estimate_portfolio_risk_state(
                    symbol=SYMBOL,
                    market_data=market_data,
                    order_manager=order_manager,
                    trade_mode=trade_mode,
                )
                risk_snapshot = risk_engine.update_portfolio_state(
                    equity_usd=portfolio_state["equity_usd"],
                    gross_exposure_usd=portfolio_state["gross_exposure_usd"],
                    timestamp=market_data.get("timestamp"),
                )
                logger.info(
                    "Risk snapshot | equity=%.4f peak=%.4f exposure=%.4f drawdown=%.4f%% daily_pnl=%.4f kill_switch=%s",
                    float(risk_snapshot.equity_usd),
                    float(risk_snapshot.peak_equity_usd),
                    float(risk_snapshot.gross_exposure_usd),
                    float(risk_snapshot.drawdown_pct),
                    float(risk_snapshot.daily_pnl_usd),
                    "active" if risk_snapshot.kill_switch_active else "inactive",
                )
                if db_integration:
                    db_integration.save_risk_state(
                        {
                            "timestamp": risk_snapshot.timestamp,
                            "equity_usd": risk_snapshot.equity_usd,
                            "peak_equity_usd": risk_snapshot.peak_equity_usd,
                            "gross_exposure_usd": risk_snapshot.gross_exposure_usd,
                            "drawdown_pct": risk_snapshot.drawdown_pct,
                            "daily_pnl_usd": risk_snapshot.daily_pnl_usd,
                            "kill_switch_active": risk_snapshot.kill_switch_active,
                            "kill_switch_reason": risk_snapshot.kill_switch_reason,
                        }
                    )
                    if risk_snapshot.kill_switch_triggered:
                        db_integration.add_system_alert(
                            message=f"Risk kill-switch activated: {risk_snapshot.kill_switch_reason}",
                            alert_type="error",
                            severity="critical",
                            data={
                                "equity_usd": risk_snapshot.equity_usd,
                                "peak_equity_usd": risk_snapshot.peak_equity_usd,
                                "drawdown_pct": risk_snapshot.drawdown_pct,
                                "gross_exposure_usd": risk_snapshot.gross_exposure_usd,
                                "daily_pnl_usd": risk_snapshot.daily_pnl_usd,
                            },
                        )
                        notifier.send_alert({
                            "severity": "critical",
                            "message": f"Risk kill-switch activated: {risk_snapshot.kill_switch_reason}",
                            "alert_type": "error",
                        })
                if risk_snapshot.kill_switch_active:
                    logger.warning("Risk kill-switch active: %s", risk_snapshot.kill_switch_reason)
            
            # Get signals from all enabled strategies
            signals = get_all_strategy_signals(SYMBOL)
            logger.info(f"Strategy signals: {signals}")

            # Detect current market regime from the same candle window used in this loop.
            if regime_config.get("enabled", True):
                regime_state = regime_detector.detect_from_candles(
                    market_data.get("candles", []),
                    timestamp=market_data.get("timestamp"),
                )
            else:
                regime_state = RegimeState(
                    regime=REGIME_UNKNOWN,
                    confidence=0.0,
                    trend_pct=0.0,
                    realized_volatility_pct=0.0,
                    lookback_candles=0,
                    timestamp=market_data.get("timestamp", datetime.utcnow().isoformat()),
                    details={"reason": "regime_detection_disabled"},
                )
            logger.info(
                "Market regime: %s (confidence=%.2f trend=%.2f%% vol=%.2f%% lookback=%s)",
                regime_state.regime,
                float(regime_state.confidence),
                float(regime_state.trend_pct) * 100.0,
                float(regime_state.realized_volatility_pct) * 100.0,
                regime_state.lookback_candles,
            )

            if db_integration:
                db_integration.save_regime_state(
                    symbol=SYMBOL,
                    timeframe=primary_timeframe,
                    regime=regime_state.regime,
                    confidence=regime_state.confidence,
                    trend_pct=regime_state.trend_pct,
                    volatility_pct=regime_state.realized_volatility_pct,
                    lookback_candles=regime_state.lookback_candles,
                    timestamp=regime_state.timestamp,
                    details=regime_state.details,
                )

            # Apply regime policy routing (strategy enable/disable, weights, and sizing).
            policy_decision = policy_engine.evaluate(
                signals=signals,
                regime_state=regime_state,
                now=market_data.get("timestamp"),
            )
            policy_signals = dict(policy_decision.signals)
            effective_size_multiplier = float(policy_decision.size_multiplier)
            edge_size_multiplier = 1.0
            edge_disabled_strategies = []

            if edge_monitor and policy_signals:
                for strategy_name in list(policy_signals.keys()):
                    edge_status = edge_status_by_strategy.get(strategy_name)
                    if edge_status is None:
                        continue
                    previous_state = last_edge_states.get(strategy_name)
                    last_edge_states[strategy_name] = edge_status.state

                    if edge_status.disable_trading:
                        edge_disabled_strategies.append(strategy_name)
                        policy_signals.pop(strategy_name, None)
                    else:
                        edge_size_multiplier = min(
                            edge_size_multiplier,
                            float(edge_status.size_multiplier),
                        )

                    if previous_state != edge_status.state and edge_status.state in {"derisked", "disabled"}:
                        logger.warning(
                            "Edge monitor action for %s: state=%s win_rate=%.2f mean_edge=%.6f",
                            strategy_name,
                            edge_status.state,
                            float(edge_status.win_rate),
                            float(edge_status.mean_edge_return),
                        )
                        if db_integration:
                            severity = "high" if edge_status.state == "disabled" else "medium"
                            db_integration.add_system_alert(
                                message=(
                                    f"Edge monitor {edge_status.state} strategy {strategy_name}: "
                                    f"win_rate={edge_status.win_rate:.2f} mean_edge={edge_status.mean_edge_return:.6f}"
                                ),
                                alert_type="warning",
                                severity=severity,
                                data=edge_status.to_dict(),
                            )
                        if observability:
                            observability.record_error(
                                component="edge_monitor",
                                error_type=f"strategy_{edge_status.state}",
                                message=(
                                    f"{strategy_name} moved to {edge_status.state} "
                                    f"(win_rate={edge_status.win_rate:.2f})"
                                ),
                                severity="medium",
                                metadata=edge_status.to_dict(),
                            )

            effective_size_multiplier = max(
                0.0,
                float(effective_size_multiplier) * float(edge_size_multiplier),
            )
            if not policy_signals and signals:
                logger.warning(
                    "Policy disabled all active strategy signals for regime %s; forcing HOLD behavior.",
                    policy_decision.active_regime,
                )
            logger.info(
                (
                    "Policy routing: detected=%s active=%s enabled=%s "
                    "base_size_multiplier=%.2f edge_multiplier=%.2f edge_disabled=%s "
                    "switch=%s reason=%s"
                ),
                policy_decision.detected_regime,
                policy_decision.active_regime,
                policy_decision.enabled_strategies,
                float(policy_decision.size_multiplier),
                float(edge_size_multiplier),
                edge_disabled_strategies,
                policy_decision.switch_applied,
                policy_decision.switch_reason or "none",
            )

            if db_integration and policy_decision.switch_reason:
                if policy_decision.switch_applied or (
                    "hysteresis" in policy_decision.switch_reason
                    or "cooldown" in policy_decision.switch_reason
                    or "turnover" in policy_decision.switch_reason
                    or "shadow" in policy_decision.switch_reason
                ):
                    db_integration.add_system_alert(
                        message=(
                            f"Regime policy event: detected={policy_decision.detected_regime}, "
                            f"active={policy_decision.active_regime}, reason={policy_decision.switch_reason}"
                        ),
                        alert_type="info",
                        severity="low",
                        data={
                            "detected_regime": policy_decision.detected_regime,
                            "active_regime": policy_decision.active_regime,
                            "switch_applied": policy_decision.switch_applied,
                            "switch_reason": policy_decision.switch_reason,
                            "candidate_regime": policy_decision.candidate_regime,
                            "candidate_count": policy_decision.candidate_count,
                            "turnover_ratio": policy_decision.turnover_ratio,
                            "shadow_mode": policy_decision.shadow_mode,
                        },
                    )
            
            # Create a detailed prompt for the LLM with market context
            prompt = (
                f"Current signals for {SYMBOL}: {policy_signals}.\n"
            )
            
            # Add each strategy's signal and timeframe
            for strategy_name, signal in policy_signals.items():
                strategy_config = TRADING_CONFIG["strategies"].get(strategy_name, {})
                timeframe = strategy_config.get("timeframe", "1m")
                prompt += f"{strategy_name.capitalize()} strategy signal ({timeframe} timeframe): {signal}\n"
            
            prompt += (
                f"Detected market regime: {policy_decision.active_regime} "
                f"(confidence {regime_state.confidence:.2f}). "
                "Given these signals and the latest market data, should we BUY, SELL, or HOLD?"
            )
            
            # Use LLM for decision support (with detailed market data and context)
            llm_start = time.perf_counter()
            llm_result = llm_manager.make_llm_decision(
                market_data=market_data,
                symbol=SYMBOL,
                timeframe=primary_timeframe,
                context=(
                    f"Trading {SYMBOL} with {len(policy_signals)} policy-routed strategies | "
                    f"regime={policy_decision.active_regime} confidence={regime_state.confidence:.2f}"
                ),
                strategy_signals=policy_signals
            )
            if observability:
                observability.record_latency(
                    component="llm",
                    operation="decision",
                    latency_ms=(time.perf_counter() - llm_start) * 1000.0,
                    success=True,
                    trace_id=loop_trace.trace_id if loop_trace else None,
                )
            
            llm_decision = llm_result.get("decision", "HOLD")
            llm_confidence = llm_result.get("confidence", 0.5)
            llm_reasoning = llm_result.get("reasoning", "No reasoning provided")
            
            logger.info(f"LLM decision: {llm_decision} (confidence: {llm_confidence:.2f})")
            logger.info(f"LLM reasoning: {llm_reasoning}")
            
            # Add LLM result to market data for database storage
            market_data['llm_result'] = llm_result
            market_data['regime_state'] = {
                "regime": regime_state.regime,
                "confidence": regime_state.confidence,
                "trend_pct": regime_state.trend_pct,
                "volatility_pct": regime_state.realized_volatility_pct,
            }
            market_data['policy_decision'] = {
                "detected_regime": policy_decision.detected_regime,
                "active_regime": policy_decision.active_regime,
                "enabled_strategies": policy_decision.enabled_strategies,
                "size_multiplier": effective_size_multiplier,
                "switch_applied": policy_decision.switch_applied,
                "switch_reason": policy_decision.switch_reason,
            }
            if edge_monitor:
                market_data["edge_monitoring"] = {
                    strategy_name: status.to_dict()
                    for strategy_name, status in edge_status_by_strategy.items()
                }
            
            # Execute trade if appropriate
            execution_start = time.perf_counter()
            order = execute_trade(
                policy_signals,
                llm_decision,
                SYMBOL,
                market_data,
                order_manager,
                db_integration,
                strategy_weights=policy_decision.strategy_weight_overrides,
                position_size_multiplier=effective_size_multiplier,
                risk_engine=risk_engine,
            )
            if order:
                trade_side = order.get("side", "BUY")
                notifier.send_trade(
                    symbol=SYMBOL,
                    side=trade_side,
                    order=order,
                    regime=policy_decision.active_regime,
                )
            execution_latency_ms = (time.perf_counter() - execution_start) * 1000.0
            signal_consensus = get_signal_consensus(
                policy_signals,
                strategy_weights=policy_decision.strategy_weight_overrides,
            )
            if observability:
                observability.record_latency(
                    component="execution",
                    operation="execute_trade",
                    latency_ms=execution_latency_ms,
                    success=True,
                    trace_id=loop_trace.trace_id if loop_trace else None,
                )
                observability.record_trade_decision(
                    symbol=SYMBOL,
                    signal_consensus=signal_consensus,
                    llm_decision=llm_decision,
                    executed=order is not None,
                    trade_mode=trade_mode,
                    strategies=list(policy_signals.keys()),
                    trace_id=loop_trace.trace_id if loop_trace else None,
                    metadata={
                        "policy_regime": policy_decision.active_regime,
                        "size_multiplier": float(effective_size_multiplier),
                        "edge_multiplier": float(edge_size_multiplier),
                        "edge_disabled_strategies": list(edge_disabled_strategies),
                    },
                )
                if risk_engine:
                    strategy_pnl = {}
                    if policy_signals:
                        per_strategy = float(risk_snapshot.daily_pnl_usd) / max(1, len(policy_signals))
                        strategy_pnl = {
                            strategy_name: float(per_strategy)
                            for strategy_name in policy_signals.keys()
                        }
                    observability.record_pnl_attribution(
                        symbol=SYMBOL,
                        strategy_pnl=strategy_pnl,
                        total_pnl=float(risk_snapshot.daily_pnl_usd),
                        regime=policy_decision.active_regime,
                        timestamp=market_data.get("timestamp"),
                        metadata={"confidence": float(regime_state.confidence)},
                    )
            
            # Reset error counter on success
            consecutive_errors = 0

            if observability and loop_trace:
                observability.end_trace(
                    loop_trace,
                    status="ok",
                    metadata={"trade_executed": bool(order), "loop_count": loop_count},
                )
                _flush_observability_alerts()

            if edge_monitor:
                previous_policy_signals = dict(policy_signals)
                if float(current_reference_price) > 0.0:
                    previous_reference_price = float(current_reference_price)
            
            # Wait before next iteration
            logger.debug(f"Waiting {loop_interval} seconds for next iteration...")
            if not TESTING_MODE:
                time.sleep(loop_interval)
            
        except (BinanceAPIException, BinanceRequestException) as e:
            consecutive_errors += 1
            backoff_time = min(loop_interval * 2 ** min(consecutive_errors, max_consecutive_errors), max_backoff_seconds)
            logger.error(f"Binance API error: {e}")
            logger.info(f"Retrying in {backoff_time} seconds...")
            if observability:
                observability.record_error(
                    component="trading_loop",
                    error_type="binance_api",
                    message=str(e),
                    severity="medium",
                    metadata={"retry_in_seconds": backoff_time, "loop_count": loop_count},
                )
                if loop_trace:
                    observability.end_trace(
                        loop_trace,
                        status="error",
                        error=str(e),
                        metadata={"retry_in_seconds": backoff_time},
                    )
                _flush_observability_alerts()
            
            # Log error to database if available
            if db_integration:
                db_integration.add_system_alert(
                    message=f"Binance API error: {str(e)}",
                    alert_type="error",
                    severity="medium",
                    data={"error_type": "api_error", "retry_in": backoff_time}
                )
            
            if not TESTING_MODE:
                time.sleep(backoff_time)
            
        except Exception as e:
            consecutive_errors += 1
            backoff_time = min(loop_interval * 2 ** min(consecutive_errors, max_consecutive_errors), max_backoff_seconds)
            logger.error(f"Unexpected error in trading loop: {e}")
            logger.info(f"Retrying in {backoff_time} seconds...")
            if observability:
                observability.record_error(
                    component="trading_loop",
                    error_type="system_error",
                    message=str(e),
                    severity="high",
                    metadata={"retry_in_seconds": backoff_time, "loop_count": loop_count},
                )
                if loop_trace:
                    observability.end_trace(
                        loop_trace,
                        status="error",
                        error=str(e),
                        metadata={"retry_in_seconds": backoff_time},
                    )
                _flush_observability_alerts()
            
            # Log error to database if available
            if db_integration:
                db_integration.add_system_alert(
                    message=f"Unexpected error in trading loop: {str(e)}",
                    alert_type="error",
                    severity="high",
                    data={"error_type": "system_error", "retry_in": backoff_time}
                )
            
            if not TESTING_MODE:
                time.sleep(backoff_time)

if __name__ == '__main__':
    logger.info("=== Starting trading bot ===")
    logger.info(f"Trading symbol: {SYMBOL}")
    logger.info(f"Mode: {'TESTNET' if TESTNET else 'LIVE'}")
    logger.info(
        "Live trading explicit enable flag: %s",
        "enabled" if is_live_trading_enabled() else "disabled"
    )
    
    if initialize_bot():
        try:
            trading_loop()
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")

            # Log shutdown to database
            try:
                db = DatabaseIntegration()
                db.add_system_alert(
                    message="Bot shutdown initiated by user",
                    alert_type="info",
                    severity="low"
                )
            except:
                pass
            try:
                _notifier = get_notifier()
                _notifier.send_lifecycle("stopped")
                _notifier.shutdown()
            except:
                pass
        except Exception as e:
            logger.critical(f"Critical error: {e}")

            # Log critical error to database
            try:
                db = DatabaseIntegration()
                db.add_system_alert(
                    message=f"Critical error: {str(e)}",
                    alert_type="error",
                    severity="critical",
                    data={"error_type": "critical_system_error"}
                )
            except:
                pass
            try:
                _notifier = get_notifier()
                _notifier.send_lifecycle("crashed")
                _notifier.shutdown()
            except:
                pass
    else:
        logger.critical("Failed to initialize bot. Exiting.")
