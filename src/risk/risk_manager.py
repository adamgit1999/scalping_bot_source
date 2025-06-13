import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal

from src.exceptions import RiskManagerError

logger = logging.getLogger(__name__)

class RiskManager:
    """System for managing trading risk."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize risk manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.max_position_size = Decimal(str(self.config.get('max_position_size', '1.0')))
        self.max_daily_loss = Decimal(str(self.config.get('max_daily_loss', '0.05')))  # 5%
        self.max_drawdown = Decimal(str(self.config.get('max_drawdown', '0.10')))  # 10%
        self.max_leverage = Decimal(str(self.config.get('max_leverage', '3.0')))
        self.min_margin = Decimal(str(self.config.get('min_margin', '0.3')))  # 30%
        self.max_open_positions = self.config.get('max_open_positions', 5)
        self.max_daily_trades = self.config.get('max_daily_trades', 20)
        
        self.daily_stats = {
            'trades': 0,
            'pnl': Decimal('0'),
            'start_balance': None,
            'max_balance': None,
            'min_balance': None
        }
        
    def check_position_size(self, symbol: str, size: Decimal, price: Decimal,
                          balance: Decimal) -> bool:
        """Check if position size is within limits.
        
        Args:
            symbol: Trading pair symbol
            size: Position size
            price: Current price
            balance: Account balance
            
        Returns:
            True if position size is acceptable
            
        Raises:
            RiskManagerError: If position size check fails
        """
        try:
            position_value = size * price
            position_ratio = position_value / balance
            
            if position_ratio > self.max_position_size:
                logger.warning(f"Position size {position_ratio} exceeds maximum {self.max_position_size}")
                return False
                
            return True
            
        except Exception as e:
            raise RiskManagerError(f"Failed to check position size: {e}")
            
    def check_daily_loss(self, pnl: Decimal, balance: Decimal) -> bool:
        """Check if daily loss is within limits.
        
        Args:
            pnl: Current PnL
            balance: Account balance
            
        Returns:
            True if daily loss is acceptable
            
        Raises:
            RiskManagerError: If daily loss check fails
        """
        try:
            if self.daily_stats['start_balance'] is None:
                self.daily_stats['start_balance'] = balance
                self.daily_stats['max_balance'] = balance
                self.daily_stats['min_balance'] = balance
                
            current_balance = self.daily_stats['start_balance'] + pnl
            self.daily_stats['max_balance'] = max(self.daily_stats['max_balance'], current_balance)
            self.daily_stats['min_balance'] = min(self.daily_stats['min_balance'], current_balance)
            
            daily_loss = (self.daily_stats['start_balance'] - current_balance) / self.daily_stats['start_balance']
            
            if daily_loss > self.max_daily_loss:
                logger.warning(f"Daily loss {daily_loss} exceeds maximum {self.max_daily_loss}")
                return False
                
            return True
            
        except Exception as e:
            raise RiskManagerError(f"Failed to check daily loss: {e}")
            
    def check_drawdown(self, current_balance: Decimal) -> bool:
        """Check if drawdown is within limits.
        
        Args:
            current_balance: Current account balance
            
        Returns:
            True if drawdown is acceptable
            
        Raises:
            RiskManagerError: If drawdown check fails
        """
        try:
            if self.daily_stats['max_balance'] is None:
                return True
                
            drawdown = (self.daily_stats['max_balance'] - current_balance) / self.daily_stats['max_balance']
            
            if drawdown > self.max_drawdown:
                logger.warning(f"Drawdown {drawdown} exceeds maximum {self.max_drawdown}")
                return False
                
            return True
            
        except Exception as e:
            raise RiskManagerError(f"Failed to check drawdown: {e}")
            
    def check_leverage(self, position_value: Decimal, margin: Decimal) -> bool:
        """Check if leverage is within limits.
        
        Args:
            position_value: Total position value
            margin: Margin used
            
        Returns:
            True if leverage is acceptable
            
        Raises:
            RiskManagerError: If leverage check fails
        """
        try:
            if margin == 0:
                return True
                
            leverage = position_value / margin
            
            if leverage > self.max_leverage:
                logger.warning(f"Leverage {leverage} exceeds maximum {self.max_leverage}")
                return False
                
            return True
            
        except Exception as e:
            raise RiskManagerError(f"Failed to check leverage: {e}")
            
    def check_margin(self, position_value: Decimal, margin: Decimal,
                    balance: Decimal) -> bool:
        """Check if margin is sufficient.
        
        Args:
            position_value: Total position value
            margin: Margin used
            balance: Account balance
            
        Returns:
            True if margin is sufficient
            
        Raises:
            RiskManagerError: If margin check fails
        """
        try:
            if position_value == 0:
                return True
                
            margin_ratio = margin / position_value
            
            if margin_ratio < self.min_margin:
                logger.warning(f"Margin ratio {margin_ratio} below minimum {self.min_margin}")
                return False
                
            return True
            
        except Exception as e:
            raise RiskManagerError(f"Failed to check margin: {e}")
            
    def check_open_positions(self, current_positions: int) -> bool:
        """Check if number of open positions is within limits.
        
        Args:
            current_positions: Number of current open positions
            
        Returns:
            True if number of positions is acceptable
            
        Raises:
            RiskManagerError: If position count check fails
        """
        try:
            if current_positions >= self.max_open_positions:
                logger.warning(f"Open positions {current_positions} exceeds maximum {self.max_open_positions}")
                return False
                
            return True
            
        except Exception as e:
            raise RiskManagerError(f"Failed to check open positions: {e}")
            
    def check_daily_trades(self) -> bool:
        """Check if daily trade limit is reached.
        
        Returns:
            True if more trades are allowed
            
        Raises:
            RiskManagerError: If trade count check fails
        """
        try:
            if self.daily_stats['trades'] >= self.max_daily_trades:
                logger.warning(f"Daily trades {self.daily_stats['trades']} exceeds maximum {self.max_daily_trades}")
                return False
                
            return True
            
        except Exception as e:
            raise RiskManagerError(f"Failed to check daily trades: {e}")
            
    def record_trade(self, pnl: Decimal) -> None:
        """Record a trade.
        
        Args:
            pnl: Trade PnL
            
        Raises:
            RiskManagerError: If trade recording fails
        """
        try:
            self.daily_stats['trades'] += 1
            self.daily_stats['pnl'] += pnl
            
        except Exception as e:
            raise RiskManagerError(f"Failed to record trade: {e}")
            
    def reset_daily_stats(self) -> None:
        """Reset daily statistics."""
        self.daily_stats = {
            'trades': 0,
            'pnl': Decimal('0'),
            'start_balance': None,
            'max_balance': None,
            'min_balance': None
        }
        
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics.
        
        Returns:
            Dictionary with risk metrics
        """
        return {
            'max_position_size': float(self.max_position_size),
            'max_daily_loss': float(self.max_daily_loss),
            'max_drawdown': float(self.max_drawdown),
            'max_leverage': float(self.max_leverage),
            'min_margin': float(self.min_margin),
            'max_open_positions': self.max_open_positions,
            'max_daily_trades': self.max_daily_trades,
            'daily_trades': self.daily_stats['trades'],
            'daily_pnl': float(self.daily_stats['pnl']),
            'start_balance': float(self.daily_stats['start_balance']) if self.daily_stats['start_balance'] else None,
            'max_balance': float(self.daily_stats['max_balance']) if self.daily_stats['max_balance'] else None,
            'min_balance': float(self.daily_stats['min_balance']) if self.daily_stats['min_balance'] else None
        } 