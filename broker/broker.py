from typing import Dict, List, Optional
from decimal import Decimal
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class Order:
    def __init__(self, symbol: str, quantity: Decimal, price: Decimal, order_type: str):
        self.symbol = symbol
        self.quantity = quantity
        self.price = price
        self.order_type = order_type
        self.status = "PENDING"
        self.timestamp = datetime.now(datetime.UTC)
        self.order_id = None

class Broker:
    def __init__(self, api_key: str, api_secret: str, test_mode: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.test_mode = test_mode
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Decimal] = {}
        self.balance: Decimal = Decimal('0')
        logger.info(f"Initialized broker in {'test' if test_mode else 'live'} mode")

    def place_order(self, symbol: str, quantity: Decimal, price: Decimal, order_type: str) -> str:
        """Place a new order."""
        # Validate order parameters
        if not isinstance(quantity, Decimal) or quantity <= 0:
            raise ValueError("Invalid order quantity")
        if not isinstance(price, Decimal) or price <= 0:
            raise ValueError("Invalid order price")
        if order_type not in ["LIMIT", "MARKET"]:
            raise ValueError("Invalid order type")
        if not isinstance(symbol, str) or "/" not in symbol:
            raise ValueError("Invalid symbol format")

        order = Order(symbol, quantity, price, order_type)
        order.order_id = f"order_{len(self.orders) + 1}"
        self.orders[order.order_id] = order
        
        if not self.test_mode:
            # In live mode, we would make actual API calls here
            pass
        
        logger.info(f"Placed {order_type} order for {quantity} {symbol} at {price}")
        return order.order_id

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        if order_id not in self.orders:
            logger.error(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        order.status = "CANCELLED"
        logger.info(f"Cancelled order {order_id}")
        return True

    def get_order_status(self, order_id: str) -> Optional[str]:
        """Get the status of an order."""
        if order_id not in self.orders:
            return None
        return self.orders[order_id].status

    def get_position(self, symbol: str) -> Decimal:
        """Get current position for a symbol."""
        return self.positions.get(symbol, Decimal('0'))

    def get_balance(self) -> Decimal:
        """Get current account balance."""
        return self.balance

    def update_balance(self, amount: Decimal) -> None:
        """Update account balance."""
        self.balance += amount
        logger.info(f"Updated balance: {self.balance}")

    def update_position(self, symbol: str, quantity: Decimal) -> None:
        """Update position for a symbol by adding to the existing position."""
        current_position = self.positions.get(symbol, Decimal('0'))
        self.positions[symbol] = current_position + quantity
        logger.info(f"Updated position for {symbol}: {self.positions[symbol]}")

    def get_orders(self) -> Dict[str, Order]:
        """Get all orders."""
        return self.orders.copy() 