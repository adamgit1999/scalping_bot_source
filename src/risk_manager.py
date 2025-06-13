from typing import Dict, Optional, Tuple
from decimal import Decimal
from src.exceptions import InvalidOrderError, RiskLimitExceededError
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, max_position_size: float = 0.1, max_drawdown: float = 0.1):
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.current_drawdown = 0.0
        self.initial_balance = Decimal('0')
        self.current_balance = Decimal('0')
        self.position_limits: Dict[str, float] = {}
        self.daily_loss_limits: Dict[str, float] = {}
        self.total_daily_loss = 0.0
        self.max_daily_loss = 0.05  # 5% of portfolio
        
    def update_balance(self, balance: Decimal) -> None:
        """Update current balance and calculate drawdown."""
        if self.initial_balance == 0:
            self.initial_balance = balance
        self.current_balance = balance
        self.current_drawdown = 1 - (float(self.current_balance) / float(self.initial_balance))
        
    def set_position_limit(self, symbol: str, limit: float) -> None:
        """Set position size limit for a symbol."""
        if limit <= 0 or limit > 1.0:
            raise ValueError("Position limit must be between 0 and 1")
        self.position_limits[symbol] = limit
        
    def set_daily_loss_limit(self, symbol: str, limit: float) -> None:
        """Set daily loss limit for a symbol."""
        if limit <= 0 or limit > 1.0:
            raise ValueError("Daily loss limit must be between 0 and 1")
        self.daily_loss_limits[symbol] = limit
        
    def check_order(self, symbol: str, side: str, price: float, amount: float) -> Tuple[bool, str]:
        """
        Check if an order meets risk management criteria.
        
        Returns:
            Tuple[bool, str]: (is_allowed, reason)
        """
        try:
            # Validate inputs
            if side.upper() not in ['BUY', 'SELL']:
                raise InvalidOrderError(f"Invalid order side: {side}")
            if price <= 0:
                raise InvalidOrderError(f"Invalid price: {price}")
            if amount <= 0:
                raise InvalidOrderError(f"Invalid amount: {amount}")
                
            # Check position size
            position_value = price * amount
            max_position = self.max_position_size * float(self.current_balance)
            
            if position_value > max_position:
                return False, f"Position size {position_value} exceeds maximum {max_position}"
                
            # Check symbol-specific position limit
            if symbol in self.position_limits:
                symbol_limit = self.position_limits[symbol] * float(self.current_balance)
                if position_value > symbol_limit:
                    return False, f"Position size {position_value} exceeds symbol limit {symbol_limit}"
                
            # Check drawdown
            if self.current_drawdown >= self.max_drawdown:
                return False, f"Current drawdown {self.current_drawdown:.2%} exceeds maximum {self.max_drawdown:.2%}"
                
            # Check daily loss limit
            if symbol in self.daily_loss_limits:
                symbol_daily_limit = self.daily_loss_limits[symbol] * float(self.current_balance)
                if self.total_daily_loss + position_value > symbol_daily_limit:
                    return False, f"Order would exceed daily loss limit for {symbol}"
                    
            # Check total daily loss
            if self.total_daily_loss + position_value > self.max_daily_loss * float(self.current_balance):
                return False, "Order would exceed total daily loss limit"
                
            return True, "Order passed risk checks"
            
        except Exception as e:
            logger.error(f"Risk check failed: {str(e)}")
            return False, str(e)
            
    def update_daily_loss(self, loss: float) -> None:
        """Update total daily loss."""
        self.total_daily_loss += loss
        
    def reset_daily_metrics(self) -> None:
        """Reset daily metrics."""
        self.total_daily_loss = 0.0
        
    def calculate_position_size(self, symbol: str, side: str, price: float) -> Decimal:
        """Calculate the maximum position size based on risk parameters."""
        try:
            # Calculate maximum position value
            max_position_value = self.max_position_size * float(self.current_balance)
            
            # Calculate position size
            position_size = Decimal(str(max_position_value / price))
            
            return position_size
            
        except Exception as e:
            print(f"Position size calculation failed: {str(e)}")
            return Decimal('0') 