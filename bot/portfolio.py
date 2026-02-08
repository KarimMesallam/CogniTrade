"""
Portfolio allocation utilities for multi-asset trading.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class PortfolioConstraints:
    """Risk and sizing constraints for portfolio allocation."""
    min_weight: float = 0.0
    max_weight: float = 0.6
    max_leverage: float = 1.0
    risk_free_rate: float = 0.0


@dataclass
class PortfolioAllocationResult:
    """Optimizer output with allocation and diagnostics."""
    weights: Dict[str, float]
    expected_return_annual: float
    expected_volatility_annual: float
    sharpe_ratio: float
    risk_contributions: Dict[str, float] = field(default_factory=dict)


class PortfolioOptimizer:
    """Constrained portfolio optimizer."""

    def __init__(self, constraints: Optional[PortfolioConstraints] = None):
        self.constraints = constraints or PortfolioConstraints()

    def optimize(
        self,
        returns: pd.DataFrame,
        method: str = "risk_parity",
        risk_budgets: Optional[Dict[str, float]] = None,
    ) -> PortfolioAllocationResult:
        """
        Optimize portfolio weights from historical returns.

        Args:
            returns: DataFrame of periodic returns with one column per asset.
            method: `risk_parity` or `mean_variance`.
            risk_budgets: Optional per-asset target risk budget for risk parity.
        """
        clean_returns = self._validate_returns(returns)
        assets = list(clean_returns.columns)

        covariance = clean_returns.cov().values
        mean_returns = clean_returns.mean().values

        if method == "risk_parity":
            raw_weights = self._risk_parity_weights(assets, covariance, risk_budgets)
        elif method in ("mean_variance", "max_sharpe"):
            raw_weights = self._mean_variance_weights(mean_returns, covariance)
        else:
            raise ValueError(f"Unsupported optimization method: {method}")

        constrained_weights = self._apply_constraints(raw_weights)
        weights_dict = {
            asset: float(weight)
            for asset, weight in zip(assets, constrained_weights)
        }

        expected_return_annual = float(np.dot(constrained_weights, mean_returns) * 252.0)
        portfolio_var_daily = float(constrained_weights.T @ covariance @ constrained_weights)
        expected_volatility_annual = float(np.sqrt(max(portfolio_var_daily, 0.0) * 252.0))
        if expected_volatility_annual > 0:
            sharpe_ratio = (
                expected_return_annual - self.constraints.risk_free_rate
            ) / expected_volatility_annual
        else:
            sharpe_ratio = 0.0

        risk_contributions = self._risk_contributions_dict(
            assets,
            constrained_weights,
            covariance,
        )

        return PortfolioAllocationResult(
            weights=weights_dict,
            expected_return_annual=expected_return_annual,
            expected_volatility_annual=expected_volatility_annual,
            sharpe_ratio=sharpe_ratio,
            risk_contributions=risk_contributions,
        )

    def allocate_notional(self, total_capital: float, weights: Dict[str, float]) -> Dict[str, float]:
        """Convert portfolio weights to per-asset notional allocations."""
        return {asset: total_capital * float(weight) for asset, weight in weights.items()}

    def allocate_units(
        self,
        total_capital: float,
        weights: Dict[str, float],
        prices: Dict[str, float],
        min_notional: float = 0.0,
    ) -> Dict[str, Dict[str, float]]:
        """
        Convert target weights into notional and unit allocations.
        """
        allocations: Dict[str, Dict[str, float]] = {}
        for asset, weight in weights.items():
            notional = total_capital * float(weight)
            price = float(prices.get(asset, 0.0))
            if price <= 0 or abs(notional) < min_notional:
                units = 0.0
            else:
                units = notional / price

            allocations[asset] = {
                "weight": float(weight),
                "notional": float(notional),
                "units": float(units),
                "price": float(price),
            }
        return allocations

    def _validate_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        if returns is None or not isinstance(returns, pd.DataFrame):
            raise ValueError("returns must be a pandas DataFrame")
        if returns.empty:
            raise ValueError("returns DataFrame must not be empty")
        clean = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        if clean.empty:
            raise ValueError("returns DataFrame has no valid rows after cleaning")
        if clean.shape[1] < 1:
            raise ValueError("returns DataFrame must include at least one asset column")
        return clean

    def _risk_parity_weights(
        self,
        assets: list,
        covariance: np.ndarray,
        risk_budgets: Optional[Dict[str, float]] = None,
        max_iter: int = 500,
        tolerance: float = 1e-6,
    ) -> np.ndarray:
        n_assets = len(assets)
        if n_assets == 1:
            return np.array([1.0], dtype=float)

        if risk_budgets:
            budgets = np.array([max(risk_budgets.get(a, 0.0), 0.0) for a in assets], dtype=float)
            if budgets.sum() <= 0:
                budgets = np.ones(n_assets, dtype=float) / n_assets
            else:
                budgets = budgets / budgets.sum()
        else:
            budgets = np.ones(n_assets, dtype=float) / n_assets

        weights = np.ones(n_assets, dtype=float) / n_assets
        eps = 1e-12

        for _ in range(max_iter):
            portfolio_var = float(weights.T @ covariance @ weights)
            if portfolio_var <= eps:
                break

            marginal = covariance @ weights
            risk_contrib = weights * marginal
            target = budgets * portfolio_var

            step = np.sqrt((target + eps) / (risk_contrib + eps))
            new_weights = weights * step
            new_weights = self._apply_constraints(new_weights)

            delta = np.max(np.abs(new_weights - weights))
            weights = new_weights
            if delta < tolerance:
                break

        return weights

    def _mean_variance_weights(self, mean_returns: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        n_assets = len(mean_returns)
        if n_assets == 1:
            return np.array([1.0], dtype=float)

        jitter = np.eye(n_assets) * 1e-8
        inv_cov = np.linalg.pinv(covariance + jitter)
        raw = inv_cov @ mean_returns
        if not np.isfinite(raw).all() or np.allclose(raw, 0):
            raw = np.ones(n_assets, dtype=float)

        # Long-only baseline before constraints.
        raw = np.maximum(raw, 0.0)
        if raw.sum() <= 0:
            raw = np.ones(n_assets, dtype=float)
        raw = raw / raw.sum()
        return raw

    def _apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        min_w = self.constraints.min_weight
        max_w = self.constraints.max_weight
        leverage = self.constraints.max_leverage

        n_assets = len(weights)
        target_sum = min(leverage, n_assets * max_w)
        target_sum = max(target_sum, n_assets * min_w)

        projected = np.clip(weights, min_w, max_w).astype(float)
        if projected.sum() <= 0:
            projected = np.full(n_assets, 1.0 / n_assets, dtype=float)
            projected = np.clip(projected, min_w, max_w)

        eps = 1e-12
        for _ in range(100):
            current_sum = projected.sum()
            diff = target_sum - current_sum
            if abs(diff) <= 1e-9:
                break

            if diff > 0:
                mask = projected < (max_w - eps)
                if not np.any(mask):
                    break
                capacity = (max_w - projected[mask]).sum()
                if capacity <= eps:
                    break
                step = diff * ((max_w - projected[mask]) / capacity)
                projected[mask] += step
                projected = np.minimum(projected, max_w)
            else:
                mask = projected > (min_w + eps)
                if not np.any(mask):
                    break
                capacity = (projected[mask] - min_w).sum()
                if capacity <= eps:
                    break
                step = (-diff) * ((projected[mask] - min_w) / capacity)
                projected[mask] -= step
                projected = np.maximum(projected, min_w)

        return projected

    def _risk_contributions_dict(
        self,
        assets: list,
        weights: np.ndarray,
        covariance: np.ndarray,
    ) -> Dict[str, float]:
        port_var = float(weights.T @ covariance @ weights)
        if port_var <= 0:
            return {asset: 0.0 for asset in assets}

        marginal = covariance @ weights
        contributions = weights * marginal / port_var
        return {
            asset: float(contrib)
            for asset, contrib in zip(assets, contributions)
        }
