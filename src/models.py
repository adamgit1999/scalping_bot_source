from dataclasses import dataclass
from decimal import Decimal
from typing import Optional
from datetime import datetime

@dataclass
class Order:
    id: str
    symbol: str
    type: str  # Use str for simplicity in tests; can be Enum in prod
    side: str  # 'buy' or 'sell'
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: str = 'NEW'
    filled_quantity: Decimal = Decimal('0')
    average_price: Decimal = Decimal('0')
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    client_order_id: Optional[str] = None
    post_only: bool = False
    reduce_only: bool = False
    time_in_force: str = 'GTC'
    expire_time: Optional[float] = None
    order_type: str = 'LIMIT'  # Added field to match test expectations

@dataclass
class Position:
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal

@dataclass
class Trade:
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    order_id: str
    profit: Optional[Decimal] = None
    fee: Optional[Decimal] = None 