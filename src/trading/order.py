class Order:
    """Order class representing a trading order."""
    
    def __init__(
        self,
        id: str,
        symbol: str,
        type: OrderType,
        side: OrderSide,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        status: OrderStatus = OrderStatus.NEW,
        filled_quantity: Decimal = Decimal("0"),
        average_price: Decimal = Decimal("0"),
        created_at: float = None,
        updated_at: float = None,
        client_order_id: Optional[str] = None,
        post_only: bool = False,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
        expire_time: Optional[float] = None,
        retry_count: int = 0
    ):
        """Initialize order."""
        self.id = id
        self.symbol = symbol
        self.type = type
        self.side = side
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.status = status
        self.filled_quantity = filled_quantity
        self.average_price = average_price
        self.created_at = created_at or time.time()
        self.updated_at = updated_at or time.time()
        self.client_order_id = client_order_id
        self.post_only = post_only
        self.reduce_only = reduce_only
        self.time_in_force = time_in_force
        self.expire_time = expire_time
        self.retry_count = retry_count 