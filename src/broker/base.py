from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from decimal import Decimal

from src.exceptions import BrokerError

class BrokerInterface(ABC):
    """Base interface for broker implementations."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the broker connection."""
        pass
        
    @abstractmethod
    async def place_order(self, symbol: str, side: str, order_type: str,
                         quantity: Decimal, price: Optional[Decimal] = None) -> Dict:
        """Place a new order.
        
        Args:
            symbol: Trading pair symbol
            side: Order side (buy/sell)
            order_type: Order type (market/limit)
            quantity: Order quantity
            price: Order price (required for limit orders)
            
        Returns:
            Order details dictionary
            
        Raises:
            BrokerError: If order placement fails
        """
        pass
        
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if order was cancelled successfully
            
        Raises:
            BrokerError: If order cancellation fails
        """
        pass
        
    @abstractmethod
    async def get_balance(self) -> Dict[str, Decimal]:
        """Get account balance.
        
        Returns:
            Dictionary of asset balances
            
        Raises:
            BrokerError: If balance retrieval fails
        """
        pass
        
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Dict:
        """Get order status.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status dictionary
            
        Raises:
            BrokerError: If order status retrieval fails
        """
        pass
        
    @abstractmethod
    async def get_current_price(self, symbol: str) -> Decimal:
        """Get current price for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Current price
            
        Raises:
            BrokerError: If price retrieval fails
        """
        pass
        
    @abstractmethod
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book for a symbol.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of orders to retrieve
            
        Returns:
            Order book dictionary
            
        Raises:
            BrokerError: If order book retrieval fails
        """
        pass
        
    @abstractmethod
    async def get_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades for a symbol.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of trades to retrieve
            
        Returns:
            List of trade dictionaries
            
        Raises:
            BrokerError: If trade history retrieval fails
        """
        pass 