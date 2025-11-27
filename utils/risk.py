"""
utils/risk.py

A lightweight RiskManager used by live_trading.py.
Provides:
- check_trade(symbol, signal, portfolio) -> bool
- calculate_position_size(symbol, signal, portfolio) -> int

Config keys used (should be in config.json):
- position_size (default base size)
- max_position_size
- max_positions
- max_portfolio_value
- max_loss_per_trade (fraction of portfolio e.g., 0.02)
- max_daily_loss (fraction)
- risk_management.stop_loss_pct (used to compute risk per share)
- risk_management.take_profit_pct
"""

from typing import Dict
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, config: Dict):
        self.config = config or {}
        # Top-level defaults
        self.base_size = int(self.config.get("position_size", 1))
        self.max_position_size = int(self.config.get("max_position_size", 100))
        self.max_positions = int(self.config.get("max_positions", 10))
        self.max_portfolio_value = float(self.config.get("max_portfolio_value", 100000.0))
        self.max_loss_per_trade = float(self.config.get("max_loss_per_trade", 0.02))  # 2% default
        self.max_daily_loss = float(self.config.get("max_daily_loss", 0.05))  # 5% default

        # Risk management nested dict
        rm = self.config.get("risk_management", {})
        self.stop_loss_pct = float(rm.get("stop_loss_pct", 0.02))  # 2% stop loss by default
        self.take_profit_pct = float(rm.get("take_profit_pct", 0.04))  # default 4% TP
        self.trailing_stop_enabled = bool(rm.get("trailing_stop_enabled", False))

    def check_trade(self, symbol: str, signal: Dict, portfolio) -> bool:
        """
        Quick yes/no risk checks before attempting a trade:
        - Reject if portfolio already has too many positions
        - Reject if exposure would exceed max_portfolio_value
        - Reject if symbol already has a position in same direction (no pyramiding)
        - Check price validity and confidence threshold should be done earlier
        """
        # Basic config checks
        if portfolio.num_positions >= self.max_positions:
            logger.info(f"RiskManager: max positions reached ({portfolio.num_positions} >= {self.max_positions})")
            return False

        # Avoid adding to same side position (simple policy)
        if portfolio.has_position(symbol):
            logger.info(f"RiskManager: already have an open position in {symbol}. Rejecting add/pyramid.")
            return False

        # Exposure check: approximate exposure after opening new base position
        price = float(signal.get("price", 0.0))
        if price <= 0:
            logger.warning(f"RiskManager: invalid price for {symbol}: {price}")
            return False

        estimated_size = self.calculate_position_size(symbol, signal, portfolio)
        estimated_exposure = portfolio.total_exposure() + (estimated_size * price)
        if estimated_exposure > self.max_portfolio_value:
            logger.info(f"RiskManager: estimated exposure {estimated_exposure:.2f} exceeds max portfolio value {self.max_portfolio_value:.2f}")
            return False

        # Further checks (daily loss, etc) can be added; for demo we allow if above pass
        return True

    def calculate_position_size(self, symbol: str, signal: Dict, portfolio) -> int:
        """
        Determine a safe integer position size given:
         - base_size and max_position_size
         - max_loss_per_trade (fraction of portfolio value)
         - stop_loss_pct to compute per-share risk

        Formula:
            allowed_risk_value = max_loss_per_trade * max_portfolio_value
            per_share_risk = price * stop_loss_pct
            size_by_risk = floor(allowed_risk_value / per_share_risk)

        Final size = min(base_size, size_by_risk, max_position_size)
        (also ensure >= 0)
        """
        price = float(signal.get("price", 0.0))
        if price <= 0:
            return 0

        base = self.base_size
        max_sz = self.max_position_size

        # risk-limited size
        allowed_risk_value = self.max_loss_per_trade * self.max_portfolio_value
        per_share_risk = max(price * self.stop_loss_pct, 1e-9)  # avoid division by zero
        if per_share_risk <= 0:
            size_by_risk = max_sz
        else:
            size_by_risk = int(allowed_risk_value // per_share_risk)

        # Final size selection
        size = min(base, size_by_risk, max_sz)
        if size < 0:
            size = 0

        # Enforce integer
        size = int(size)
        logger.debug(f"RiskManager: size calc for {symbol}: price={price:.2f}, base={base}, size_by_risk={size_by_risk}, final={size}")
        return size

    # Optional helper: you can call this when an order fills to adjust daily loss accounting
    def record_trade(self, order: Dict, portfolio) -> None:
        """
        Update internal tracking after a trade filled.
        For demo purposes this function simply logs trade; in production you'd
        update daily loss counters, adjust risk budgets, etc.
        """
        logger.info(f"RiskManager: Recording trade {order.get('symbol')} {order.get('side')} size={order.get('size')} price={order.get('price')}")

    # Optional check for daily loss limit (demo)
    def check_daily_loss(self, portfolio) -> bool:
        """
        Returns False if today's realized loss exceeds max_daily_loss * max_portfolio_value.
        We assume portfolio.total_pnl is realized P&L.
        """
        realized = portfolio.total_pnl
        if realized < 0 and abs(realized) > (self.max_daily_loss * self.max_portfolio_value):
            logger.info(f"RiskManager: daily loss limit exceeded: realized {realized:.2f}")
            return False
        return True
