"""
Broker factory: returns the right module for configured broker.
"""
import importlib
from src.broker.binance import BinanceBroker
from .coinbase import CoinbaseBroker
from .kraken import KrakenBroker
from .broker import Broker, Order
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, Optional

class BrokerInterface(ABC):
    """Interface for broker implementations."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the broker connection."""
        pass
    
    @abstractmethod
    async def place_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None) -> Dict:
        """Place an order."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """Cancel an order."""
        pass
    
    @abstractmethod
    async def get_balance(self, currency: str) -> float:
        """Get account balance."""
        pass
    
    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        pass

def get_broker(broker_name):
    """Factory function to get broker instance."""
    if broker_name == 'binance':
        from .binance import BinanceBroker
        return BinanceBroker()
    elif broker_name == 'coinbase':
        from .coinbase import CoinbaseBroker
        return CoinbaseBroker()
    elif broker_name == 'kraken':
        from .kraken import KrakenBroker
        return KrakenBroker()
    else:
        raise ValueError(f"Unsupported broker: {broker_name}")

__all__ = ['Broker', 'Order']

