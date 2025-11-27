# test_risk.py (quick check)
from utils.risk import RiskManager
from utils.portfolio import PortfolioTracker

cfg = {
    "position_size": 10,
    "max_position_size": 50,
    "max_positions": 5,
    "max_portfolio_value": 100000,
    "max_loss_per_trade": 0.02,
    "risk_management": {"stop_loss_pct": 0.02}
}

rm = RiskManager(cfg)
pf = PortfolioTracker()

signal = {"price": 3500.0, "pred": 1, "confidence": 0.8}
print("Can trade?", rm.check_trade("TCS", signal, pf))
print("Size:", rm.calculate_position_size("TCS", signal, pf))
