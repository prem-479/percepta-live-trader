"""
utils/portfolio.py
Portfolio position tracking and P&L calculation
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class Position:
    """Represents a trading position"""
    
    def __init__(self, symbol: str, side: str, size: int, entry_price: float, 
                 timestamp: str, order_id: Optional[str] = None):
        self.symbol = symbol
        self.side = side  # BUY or SELL
        self.size = size
        self.entry_price = entry_price
        self.timestamp = timestamp
        self.order_id = order_id
        self.exit_price = None
        self.exit_timestamp = None
        self.pnl = 0.0
        self.status = 'open'
    
    def close(self, exit_price: float, timestamp: str):
        """Close the position"""
        self.exit_price = exit_price
        self.exit_timestamp = timestamp
        self.status = 'closed'
        
        # Calculate PnL
        if self.side == 'BUY':
            self.pnl = (exit_price - self.entry_price) * self.size
        else:
            self.pnl = (self.entry_price - exit_price) * self.size
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL"""
        if self.status == 'closed':
            return self.pnl
        
        if self.side == 'BUY':
            return (current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - current_price) * self.size
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'size': self.size,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'timestamp': self.timestamp,
            'exit_timestamp': self.exit_timestamp,
            'pnl': self.pnl,
            'status': self.status,
            'order_id': self.order_id
        }


class PortfolioTracker:
    """Track portfolio positions and P&L"""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.total_pnl = 0.0
    
    @property
    def num_positions(self) -> int:
        """Number of open positions"""
        return len(self.positions)
    
    def has_position(self, symbol: str) -> bool:
        """Check if symbol has open position"""
        return symbol in self.positions
    
    def update_position(self, order: Dict):
        """Update portfolio with new order"""
        symbol = order['symbol']
        side = order['side']
        
        # Check if closing existing position
        if symbol in self.positions:
            pos = self.positions[symbol]
            
            # If opposite side, close position
            if (pos.side == 'BUY' and side == 'SELL') or \
               (pos.side == 'SELL' and side == 'BUY'):
                pos.close(order['price'], order['timestamp'])
                self.closed_positions.append(pos)
                self.total_pnl += pos.pnl
                del self.positions[symbol]
                logger.info(f"Closed {symbol}: PnL = ₹{pos.pnl:.2f}")
            else:
                # Same side - this shouldn't happen with proper risk management
                logger.warning(f"Attempting to add to existing {side} position in {symbol}")
        else:
            # Open new position
            pos = Position(
                symbol=symbol,
                side=side,
                size=order['size'],
                entry_price=order['price'],
                timestamp=order['timestamp'],
                order_id=order.get('order_id')
            )
            self.positions[symbol] = pos
            logger.info(f"Opened {symbol}: {side} {order['size']} @ ₹{order['price']:.2f}")
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        return self.positions.get(symbol)
    
    def total_exposure(self) -> float:
        """Calculate total portfolio exposure"""
        exposure = sum(
            pos.size * pos.entry_price 
            for pos in self.positions.values()
        )
        return exposure
    
    def get_unrealized_pnl(self, current_prices: Dict[str, float]) -> float:
        """Calculate total unrealized PnL"""
        unrealized = 0.0
        for symbol, pos in self.positions.items():
            if symbol in current_prices:
                unrealized += pos.get_unrealized_pnl(current_prices[symbol])
        return unrealized
    
    def get_summary(self) -> Dict:
        """Get portfolio summary"""
        return {
            'num_positions': self.num_positions,
            'total_pnl': self.total_pnl,
            'total_exposure': self.total_exposure(),
            'num_closed_trades': len(self.closed_positions)
        }
    
    def print_summary(self):
        """Print detailed portfolio summary"""
        print("\n" + "=" * 60)
        print("PORTFOLIO SUMMARY")
        print("=" * 60)
        print(f"Open Positions: {self.num_positions}")
        print(f"Total Exposure: ₹{self.total_exposure():,.2f}")
        print(f"Realized P&L: ₹{self.total_pnl:,.2f}")
        print(f"Closed Trades: {len(self.closed_positions)}")
        
        if self.positions:
            print("\nOpen Positions:")
            print("-" * 60)
            for symbol, pos in self.positions.items():
                print(f"  {symbol}: {pos.side} {pos.size} @ ₹{pos.entry_price:.2f}")
        
        if self.closed_positions:
            winning = [p for p in self.closed_positions if p.pnl > 0]
            losing = [p for p in self.closed_positions if p.pnl < 0]
            
            print("\nClosed Trades Stats:")
            print("-" * 60)
            print(f"  Winners: {len(winning)}")
            print(f"  Losers: {len(losing)}")
            if self.closed_positions:
                win_rate = len(winning) / len(self.closed_positions) * 100
                print(f"  Win Rate: {win_rate:.1f}%")
                avg_win = sum(p.pnl for p in winning) / len(winning) if winning else 0
                avg_loss = sum(p.pnl for p in losing) / len(losing) if losing else 0
                print(f"  Avg Win: ₹{avg_win:.2f}")
                print(f"  Avg Loss: ₹{avg_loss:.2f}")
        
        print("=" * 60 + "\n")