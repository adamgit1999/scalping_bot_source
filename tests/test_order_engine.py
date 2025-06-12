import pytest
import pytest_asyncio
import sys
import asyncio
import time
from decimal import Decimal
from unittest.mock import Mock, patch
from trading.order_engine import OrderEngine, Order, OrderType, OrderSide, OrderStatus
from trading.exceptions import OrderError, ExecutionError, ValidationError

@pytest.fixture
def event_loop():
    """Create event loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture
async def order_engine():
    """Create order engine."""
    engine = OrderEngine()
    yield engine
    await engine.stop()

@pytest.fixture
def market_order():
    """Create market order."""
    return Order(
        id="test_market",
        symbol="BTC/USD",
        type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=Decimal("1.0"),
        price=None,
        stop_price=None,
        status=OrderStatus.NEW,
        filled_quantity=Decimal("0"),
        average_price=Decimal("0"),
        created_at=time.time(),
        updated_at=time.time(),
        client_order_id=None,
        post_only=False,
        reduce_only=False,
        time_in_force="GTC",
        expire_time=None
    )

@pytest.fixture
def limit_order():
    """Create limit order."""
    return Order(
        id="test_limit",
        symbol="BTC/USD",
        type=OrderType.LIMIT,
        side=OrderSide.SELL,
        quantity=Decimal("1.0"),
        price=Decimal("50000.0"),
        stop_price=None,
        status=OrderStatus.NEW,
        filled_quantity=Decimal("0"),
        average_price=Decimal("0"),
        created_at=time.time(),
        updated_at=time.time(),
        client_order_id=None,
        post_only=False,
        reduce_only=False,
        time_in_force="GTC",
        expire_time=None
    )

@pytest.fixture
def stop_limit_order():
    """Create stop limit order."""
    return Order(
        id="test_stop_limit",
        symbol="BTC/USD",
        type=OrderType.STOP_LIMIT,
        side=OrderSide.BUY,
        quantity=Decimal("1.0"),
        price=Decimal("50000.0"),
        stop_price=Decimal("49000.0"),
        status=OrderStatus.NEW,
        filled_quantity=Decimal("0"),
        average_price=Decimal("0"),
        created_at=time.time(),
        updated_at=time.time(),
        client_order_id=None,
        post_only=False,
        reduce_only=False,
        time_in_force="GTC",
        expire_time=None
    )

@pytest.mark.asyncio
async def test_place_market_order(order_engine, market_order):
    """Test placing market order."""
    await order_engine.start()
    
    # Mock exchange execution
    async def mock_execute(order):
        return {
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side.value,
            'type': order.type.value,
            'quantity': float(order.quantity),
            'price': 50000.0,
            'timestamp': time.time()
        }
    
    order_engine._send_to_exchange = mock_execute
    
    # Place order
    await order_engine.place_order(market_order)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Check order status
    order = order_engine.get_order(market_order.id)
    assert order is not None
    assert order.status == OrderStatus.FILLED
    assert order.filled_quantity == market_order.quantity
    assert order.average_price == Decimal("50000.0")

@pytest.mark.asyncio
async def test_place_limit_order(order_engine, limit_order):
    """Test placing limit order."""
    await order_engine.start()
    
    # Mock exchange execution
    async def mock_execute(order):
        return {
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side.value,
            'type': order.type.value,
            'quantity': float(order.quantity),
            'price': float(order.price),
            'timestamp': time.time()
        }
    
    order_engine._send_to_exchange = mock_execute
    
    # Place order
    await order_engine.place_order(limit_order)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Check order status
    order = order_engine.get_order(limit_order.id)
    assert order is not None
    assert order.status == OrderStatus.FILLED
    assert order.filled_quantity == limit_order.quantity
    assert order.average_price == limit_order.price

@pytest.mark.asyncio
async def test_place_stop_limit_order(order_engine, stop_limit_order):
    """Test placing stop limit order."""
    await order_engine.start()
    
    # Mock exchange execution
    async def mock_execute(order):
        return {
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side.value,
            'type': order.type.value,
            'quantity': float(order.quantity),
            'price': float(order.price),
            'timestamp': time.time()
        }
    
    order_engine._send_to_exchange = mock_execute
    
    # Place order
    await order_engine.place_order(stop_limit_order)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Check order status
    order = order_engine.get_order(stop_limit_order.id)
    assert order is not None
    assert order.status == OrderStatus.NEW  # Should remain NEW until stop price is hit
    
    # Simulate stop price being hit
    await order_engine._process_stop_orders(Decimal("48000.0"))
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Check order was filled
    order = order_engine.get_order(stop_limit_order.id)
    assert order is not None
    assert order.status == OrderStatus.FILLED
    assert order.filled_quantity == stop_limit_order.quantity
    assert order.average_price == stop_limit_order.price

@pytest.mark.asyncio
async def test_cancel_order(order_engine, market_order):
    """Test canceling order."""
    await order_engine.start()
    
    # Place order
    await order_engine.place_order(market_order)
    
    # Cancel order
    await order_engine.cancel_order(market_order.id)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Check order status
    order = order_engine.get_order(market_order.id)
    assert order is not None
    assert order.status == OrderStatus.CANCELED

@pytest.mark.asyncio
async def test_cancel_nonexistent_order(order_engine):
    """Test canceling nonexistent order."""
    await order_engine.start()
    
    # Try to cancel nonexistent order
    with pytest.raises(OrderError):
        await order_engine.cancel_order("nonexistent")

@pytest.mark.asyncio
async def test_cancel_filled_order(order_engine, market_order):
    """Test canceling filled order."""
    await order_engine.start()
    
    # Mock exchange execution
    async def mock_execute(order):
        return {
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side.value,
            'type': order.type.value,
            'quantity': float(order.quantity),
            'price': 50000.0,
            'timestamp': time.time()
        }
    
    order_engine._send_to_exchange = mock_execute
    
    # Place and fill order
    await order_engine.place_order(market_order)
    await asyncio.sleep(0.1)
    
    # Try to cancel filled order
    with pytest.raises(OrderError, match="cannot be canceled"):
        await order_engine.cancel_order(market_order.id)

@pytest.mark.asyncio
async def test_get_orders(order_engine, market_order, limit_order):
    """Test getting orders."""
    await order_engine.start()
    
    # Place orders
    await order_engine.place_order(market_order)
    await order_engine.place_order(limit_order)
    
    # Get all orders
    orders = order_engine.get_orders()
    assert len(orders) == 2
    
    # Get orders by symbol
    orders = order_engine.get_orders(symbol="BTC/USD")
    assert len(orders) == 2
    
    # Get orders by status
    orders = order_engine.get_orders(status=OrderStatus.NEW)
    assert len(orders) == 2
    
    # Get orders by side
    orders = order_engine.get_orders(side=OrderSide.BUY)
    assert len(orders) == 1
    assert orders[0].id == market_order.id

@pytest.mark.asyncio
async def test_order_callbacks(order_engine, market_order):
    """Test order callbacks."""
    await order_engine.start()
    
    # Create mock callbacks
    order_update_callback = Mock()
    execution_callback = Mock()
    
    # Register callbacks
    order_engine.register_callback('order_update', order_update_callback)
    order_engine.register_callback('execution', execution_callback)
    
    # Mock exchange execution
    async def mock_execute(order):
        return {
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side.value,
            'type': order.type.value,
            'quantity': float(order.quantity),
            'price': 50000.0,
            'timestamp': time.time()
        }
    
    order_engine._send_to_exchange = mock_execute
    
    # Place order
    await order_engine.place_order(market_order)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Check callbacks were called
    assert order_update_callback.call_count >= 1
    assert execution_callback.call_count >= 1

@pytest.mark.asyncio
async def test_execution_stats(order_engine, market_order):
    """Test execution statistics."""
    await order_engine.start()
    
    # Mock exchange execution
    async def mock_execute(order):
        await asyncio.sleep(0.1)  # Simulate execution time
        return {
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side.value,
            'type': order.type.value,
            'quantity': float(order.quantity),
            'price': 50000.0,
            'timestamp': time.time()
        }
    
    order_engine._send_to_exchange = mock_execute
    
    # Place order
    await order_engine.place_order(market_order)
    
    # Wait for processing
    await asyncio.sleep(0.2)
    
    # Check execution stats
    stats = order_engine.get_execution_stats()
    assert 'avg_execution_time' in stats
    assert 'min_execution_time' in stats
    assert 'max_execution_time' in stats
    assert stats['avg_execution_time'] >= 0.1

@pytest.mark.asyncio
async def test_invalid_order(order_engine):
    """Test invalid order handling."""
    await order_engine.start()
    
    # Create invalid order
    invalid_order = Order(
        id="test_invalid",
        symbol="",  # Empty symbol
        type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=Decimal("0"),  # Zero quantity
        price=None,
        stop_price=None,
        status=OrderStatus.NEW,
        filled_quantity=Decimal("0"),
        average_price=Decimal("0"),
        created_at=time.time(),
        updated_at=time.time(),
        client_order_id=None,
        post_only=False,
        reduce_only=False,
        time_in_force="GTC",
        expire_time=None
    )
    
    # Place invalid order
    await order_engine.place_order(invalid_order)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Check order was rejected
    order = order_engine.get_order(invalid_order.id)
    assert order is not None
    assert order.status == OrderStatus.REJECTED

@pytest.mark.asyncio
async def test_expired_order(order_engine):
    """Test expired order handling."""
    await order_engine.start()
    
    # Create order with short expiration
    expired_order = Order(
        id="test_expired",
        symbol="BTC/USD",
        type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=Decimal("1.0"),
        price=None,
        stop_price=None,
        status=OrderStatus.NEW,
        filled_quantity=Decimal("0"),
        average_price=Decimal("0"),
        created_at=time.time(),
        updated_at=time.time(),
        client_order_id=None,
        post_only=False,
        reduce_only=False,
        time_in_force="GTC",
        expire_time=time.time() + 0.1  # Expire in 100ms
    )
    
    # Place order
    await order_engine.place_order(expired_order)
    
    # Wait for expiration
    await asyncio.sleep(0.2)
    
    # Check order expired
    order = order_engine.get_order(expired_order.id)
    assert order is not None
    assert order.status == OrderStatus.EXPIRED

@pytest.mark.asyncio
async def test_concurrent_orders(order_engine):
    """Test concurrent order processing."""
    await order_engine.start()
    
    # Create multiple orders
    orders = []
    for i in range(10):
        order = Order(
            id=f"test_concurrent_{i}",
            symbol="BTC/USD",
            type=OrderType.MARKET,
            side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            quantity=Decimal("1.0"),
            price=None,
            stop_price=None,
            status=OrderStatus.NEW,
            filled_quantity=Decimal("0"),
            average_price=Decimal("0"),
            created_at=time.time(),
            updated_at=time.time(),
            client_order_id=None,
            post_only=False,
            reduce_only=False,
            time_in_force="GTC",
            expire_time=None
        )
        orders.append(order)
    
    # Mock exchange execution
    async def mock_execute(order):
        return {
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side.value,
            'type': order.type.value,
            'quantity': float(order.quantity),
            'price': 50000.0,
            'timestamp': time.time()
        }
    
    order_engine._send_to_exchange = mock_execute
    
    # Place orders concurrently
    await asyncio.gather(*[
        order_engine.place_order(order)
        for order in orders
    ])
    
    # Wait for processing
    await asyncio.sleep(0.2)
    
    # Check all orders were processed
    for order in orders:
        processed_order = order_engine.get_order(order.id)
        assert processed_order is not None
        assert processed_order.status == OrderStatus.FILLED

@pytest.mark.asyncio
async def test_order_retry_mechanism(order_engine, market_order):
    """Test order retry mechanism."""
    await order_engine.start()
    
    # Mock exchange to fail twice then succeed
    retry_count = 0
    
    async def mock_execute(order):
        nonlocal retry_count
        retry_count += 1
        if retry_count <= 2:
            raise ExecutionError("Temporary failure")
        return {
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side.value,
            'type': order.type.value,
            'quantity': float(order.quantity),
            'price': 50000.0,
            'timestamp': time.time()
        }
    
    # Replace exchange execution with mock
    order_engine._send_to_exchange = mock_execute
    
    # Place order
    await order_engine.place_order(market_order)
    
    # Wait for processing
    await asyncio.sleep(0.5)
    
    # Check order was eventually filled
    order = order_engine.get_order(market_order.id)
    assert order is not None
    assert order.status == OrderStatus.FILLED
    assert retry_count == 3

@pytest.mark.asyncio
async def test_post_only_order(order_engine):
    """Test post-only order handling."""
    await order_engine.start()
    
    # Create post-only order
    post_only_order = Order(
        id="test_post_only",
        symbol="BTC/USD",
        type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=Decimal("1.0"),
        price=Decimal("50000.0"),
        stop_price=None,
        status=OrderStatus.NEW,
        filled_quantity=Decimal("0"),
        average_price=Decimal("0"),
        created_at=time.time(),
        updated_at=time.time(),
        client_order_id=None,
        post_only=True,
        reduce_only=False,
        time_in_force="GTC",
        expire_time=None
    )
    
    # Mock exchange execution
    async def mock_execute(order):
        if order.post_only:
            raise ExecutionError("Post-only order would have been filled immediately")
        return {
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side.value,
            'type': order.type.value,
            'quantity': float(order.quantity),
            'price': float(order.price),
            'timestamp': time.time()
        }
    
    order_engine._send_to_exchange = mock_execute
    
    # Place order
    await order_engine.place_order(post_only_order)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Check order was rejected
    order = order_engine.get_order(post_only_order.id)
    assert order is not None
    assert order.status == OrderStatus.REJECTED

@pytest.mark.asyncio
async def test_reduce_only_order(order_engine):
    """Test reduce-only order handling."""
    await order_engine.start()
    
    # Create reduce-only order
    reduce_only_order = Order(
        id="test_reduce_only",
        symbol="BTC/USD",
        type=OrderType.MARKET,
        side=OrderSide.SELL,
        quantity=Decimal("1.0"),
        price=None,
        stop_price=None,
        status=OrderStatus.NEW,
        filled_quantity=Decimal("0"),
        average_price=Decimal("0"),
        created_at=time.time(),
        updated_at=time.time(),
        client_order_id=None,
        post_only=False,
        reduce_only=True,
        time_in_force="GTC",
        expire_time=None
    )
    
    # Mock position to disallow reduce-only order
    def mock_get_position(symbol):
        return 0  # No position
    
    order_engine._get_position = mock_get_position
    
    # Place order
    await order_engine.place_order(reduce_only_order)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Check order was rejected
    order = order_engine.get_order(reduce_only_order.id)
    assert order is not None
    assert order.status == OrderStatus.REJECTED

@pytest.mark.asyncio
async def test_reduce_only_order_success(order_engine):
    """Test successful reduce-only order."""
    await order_engine.start()
    
    # Mock position to allow reduce-only order
    def mock_get_position(symbol):
        return 1  # Long position
    
    order_engine._get_position = mock_get_position
    
    # Create reduce-only order
    reduce_only_order = Order(
        id="test_reduce_only_success",
        symbol="BTC/USD",
        type=OrderType.MARKET,
        side=OrderSide.SELL,
        quantity=Decimal("1.0"),
        price=None,
        stop_price=None,
        status=OrderStatus.NEW,
        filled_quantity=Decimal("0"),
        average_price=Decimal("0"),
        created_at=time.time(),
        updated_at=time.time(),
        client_order_id=None,
        post_only=False,
        reduce_only=True,
        time_in_force="GTC",
        expire_time=None
    )
    
    # Mock exchange execution
    async def mock_execute(order):
        return {
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side.value,
            'type': order.type.value,
            'quantity': float(order.quantity),
            'price': 50000.0,
            'timestamp': time.time()
        }
    
    order_engine._send_to_exchange = mock_execute
    
    # Place order
    await order_engine.place_order(reduce_only_order)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Check order was filled
    order = order_engine.get_order(reduce_only_order.id)
    assert order is not None
    assert order.status == OrderStatus.FILLED

@pytest.mark.asyncio
async def test_task_restart(order_engine):
    """Test task restart on failure."""
    await order_engine.start()
    
    # Create task that fails
    async def failing_task():
        raise Exception("Task failed")
    
    # Add task
    task = asyncio.create_task(failing_task())
    order_engine._tasks.add(task)
    
    # Wait for task to fail
    await asyncio.sleep(0.1)
    
    # Check task was removed
    assert task not in order_engine._tasks

@pytest.mark.asyncio
async def test_cleanup_handlers(order_engine):
    """Test cleanup handlers."""
    await order_engine.start()
    
    # Create mock cleanup handler
    cleanup_called = False
    
    async def cleanup_handler():
        nonlocal cleanup_called
        cleanup_called = True
    
    # Register cleanup handler
    order_engine.register_cleanup_handler(cleanup_handler)
    
    # Stop engine
    await order_engine.stop()
    
    # Check cleanup handler was called
    assert cleanup_called

@pytest.mark.asyncio
async def test_signal_handlers(order_engine):
    """Test signal handlers."""
    if sys.platform == 'win32':
        pytest.skip("Signal handlers not supported on Windows")
        
    await order_engine.start()
    
    # Create dummy task
    async def dummy_task():
        while True:
            await asyncio.sleep(0.1)
    
    task = asyncio.create_task(dummy_task())
    
    # Simulate signal handling
    await order_engine.stop()
    
    # Check engine stopped
    assert not order_engine.running
    
    # Check task was cancelled
    assert task.cancelled()

@pytest.mark.asyncio
async def test_partial_fills(order_engine):
    """Test handling of partial fills."""
    await order_engine.start()
    
    # Create order
    order = Order(
        id="test_partial",
        symbol="BTC/USD",
        type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=Decimal("2.0"),
        price=None,
        stop_price=None,
        status=OrderStatus.NEW,
        filled_quantity=Decimal("0"),
        average_price=Decimal("0"),
        created_at=time.time(),
        updated_at=time.time(),
        client_order_id=None,
        post_only=False,
        reduce_only=False,
        time_in_force="GTC",
        expire_time=None
    )
    
    # Mock exchange to return partial fills
    fill_count = 0
    async def mock_execute(order):
        nonlocal fill_count
        fill_count += 1
        if fill_count == 1:
            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'side': order.side.value,
                'type': order.type.value,
                'quantity': float(order.quantity) / 2,
                'price': 50000.0,
                'timestamp': time.time()
            }
        else:
            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'side': order.side.value,
                'type': order.type.value,
                'quantity': float(order.quantity) / 2,
                'price': 50100.0,
                'timestamp': time.time()
            }
    
    order_engine._send_to_exchange = mock_execute
    
    # Place order
    await order_engine.place_order(order)
    
    # Wait for processing
    await asyncio.sleep(0.2)
    
    # Check order status
    order = order_engine.get_order(order.id)
    assert order is not None
    assert order.status == OrderStatus.FILLED
    assert order.filled_quantity == Decimal("2.0")
    assert order.average_price == Decimal("50050.0")  # Average of 50000 and 50100

@pytest.mark.asyncio
async def test_order_priority(order_engine):
    """Test order priority handling."""
    await order_engine.start()
    
    # Create orders with different priorities
    orders = []
    for i in range(3):
        order = Order(
            id=f"test_priority_{i}",
            symbol="BTC/USD",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=None,
            stop_price=None,
            status=OrderStatus.NEW,
            filled_quantity=Decimal("0"),
            average_price=Decimal("0"),
            created_at=time.time(),
            updated_at=time.time(),
            client_order_id=None,
            post_only=False,
            reduce_only=False,
            time_in_force="GTC",
            expire_time=None
        )
        orders.append(order)
    
    # Track execution order
    execution_order = []
    
    async def mock_execute(order):
        execution_order.append(order.id)
        return {
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side.value,
            'type': order.type.value,
            'quantity': float(order.quantity),
            'price': 50000.0,
            'timestamp': time.time()
        }
    
    order_engine._send_to_exchange = mock_execute
    
    # Place orders with different priorities
    await order_engine.place_order(orders[0])  # Lowest priority
    await order_engine.place_order(orders[1])  # Highest priority
    await order_engine.place_order(orders[2])  # Medium priority
    
    # Wait for processing
    await asyncio.sleep(0.2)
    
    # Check execution order
    assert execution_order == [orders[1].id, orders[2].id, orders[0].id]

@pytest.mark.asyncio
async def test_market_data_processing(order_engine):
    """Test market data processing and order updates."""
    await order_engine.start()
    
    # Create limit order
    order = Order(
        id="test_market_data",
        symbol="BTC/USD",
        type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=Decimal("1.0"),
        price=Decimal("50000.0"),
        stop_price=None,
        status=OrderStatus.NEW,
        filled_quantity=Decimal("0"),
        average_price=Decimal("0"),
        created_at=time.time(),
        updated_at=time.time(),
        client_order_id=None,
        post_only=False,
        reduce_only=False,
        time_in_force="GTC",
        expire_time=None
    )
    
    # Track market data updates
    market_data_updates = []
    
    def market_data_callback(data):
        market_data_updates.append(data)
    
    order_engine.register_callback('market_data', market_data_callback)
    
    # Place order
    await order_engine.place_order(order)
    
    # Simulate market data updates
    for price in [51000.0, 50500.0, 50000.0]:
        await order_engine._process_market_data({
            'symbol': 'BTC/USD',
            'price': price,
            'timestamp': time.time()
        })
        await asyncio.sleep(0.1)
    
    # Wait for processing
    await asyncio.sleep(0.2)
    
    # Check market data processing
    assert len(market_data_updates) == 3
    assert market_data_updates[0]['price'] == 51000.0
    assert market_data_updates[1]['price'] == 50500.0
    assert market_data_updates[2]['price'] == 50000.0
    
    # Check order was filled at the right price
    order = order_engine.get_order(order.id)
    assert order is not None
    assert order.status == OrderStatus.FILLED
    assert order.average_price == Decimal("50000.0")

@pytest.mark.asyncio
async def test_error_handling(order_engine):
    """Test error handling and recovery."""
    await order_engine.start()
    
    # Create order
    order = Order(
        id="test_error",
        symbol="BTC/USD",
        type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=Decimal("1.0"),
        price=None,
        stop_price=None,
        status=OrderStatus.NEW,
        filled_quantity=Decimal("0"),
        average_price=Decimal("0"),
        created_at=time.time(),
        updated_at=time.time(),
        client_order_id=None,
        post_only=False,
        reduce_only=False,
        time_in_force="GTC",
        expire_time=None
    )
    
    # Track errors
    errors = []
    
    def error_callback(error):
        errors.append(error)
    
    order_engine.register_callback('error', error_callback)
    
    # Mock exchange to raise different types of errors
    error_count = 0
    async def mock_execute(order):
        nonlocal error_count
        error_count += 1
        if error_count == 1:
            raise ExecutionError("Temporary error")
        elif error_count == 2:
            raise ValidationError("Invalid order")
        elif error_count == 3:
            raise Exception("Unexpected error")
        else:
            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'side': order.side.value,
                'type': order.type.value,
                'quantity': float(order.quantity),
                'price': 50000.0,
                'timestamp': time.time()
            }
    
    order_engine._send_to_exchange = mock_execute
    
    # Place order
    await order_engine.place_order(order)
    
    # Wait for processing
    await asyncio.sleep(0.5)
    
    # Check error handling
    assert len(errors) == 3
    assert isinstance(errors[0], ExecutionError)
    assert isinstance(errors[1], ValidationError)
    assert isinstance(errors[2], Exception)

@pytest.mark.asyncio
async def test_concurrent_cancellations(order_engine):
    """Test concurrent order cancellations."""
    await order_engine.start()
    
    # Create multiple orders
    orders = []
    for i in range(10):
        order = Order(
            id=f"test_cancel_{i}",
            symbol="BTC/USD",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=None,
            stop_price=None,
            status=OrderStatus.NEW,
            filled_quantity=Decimal("0"),
            average_price=Decimal("0"),
            created_at=time.time(),
            updated_at=time.time(),
            client_order_id=None,
            post_only=False,
            reduce_only=False,
            time_in_force="GTC",
            expire_time=None
        )
        orders.append(order)
    
    # Place orders
    for order in orders:
        await order_engine.place_order(order)
    
    # Cancel orders concurrently
    await asyncio.gather(*[
        order_engine.cancel_order(order.id)
        for order in orders
    ])
    
    # Wait for processing
    await asyncio.sleep(0.2)
    
    # Check all orders were canceled
    for order in orders:
        processed_order = order_engine.get_order(order.id)
        assert processed_order is not None
        assert processed_order.status == OrderStatus.CANCELED

@pytest.mark.asyncio
async def test_order_expiration_handling(order_engine):
    """Test handling of order expiration during processing."""
    await order_engine.start()
    
    # Create order with short expiration
    order = Order(
        id="test_expiration",
        symbol="BTC/USD",
        type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=Decimal("1.0"),
        price=None,
        stop_price=None,
        status=OrderStatus.NEW,
        filled_quantity=Decimal("0"),
        average_price=Decimal("0"),
        created_at=time.time(),
        updated_at=time.time(),
        client_order_id=None,
        post_only=False,
        reduce_only=False,
        time_in_force="GTC",
        expire_time=time.time() + 0.1  # Expire in 100ms
    )
    
    # Mock exchange to delay execution
    async def mock_execute(order):
        await asyncio.sleep(0.2)  # Delay longer than expiration
        return {
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side.value,
            'type': order.type.value,
            'quantity': float(order.quantity),
            'price': 50000.0,
            'timestamp': time.time()
        }
    
    order_engine._send_to_exchange = mock_execute
    
    # Place order
    await order_engine.place_order(order)
    
    # Wait for processing
    await asyncio.sleep(0.3)
    
    # Check order expired
    order = order_engine.get_order(order.id)
    assert order is not None
    assert order.status == OrderStatus.EXPIRED

@pytest.mark.asyncio
async def test_memory_management(order_engine):
    """Test memory management and cleanup."""
    await order_engine.start()
    
    # Create and process many orders
    orders = []
    for i in range(1000):
        order = Order(
            id=f"test_memory_{i}",
            symbol="BTC/USD",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=None,
            stop_price=None,
            status=OrderStatus.NEW,
            filled_quantity=Decimal("0"),
            average_price=Decimal("0"),
            created_at=time.time(),
            updated_at=time.time(),
            client_order_id=None,
            post_only=False,
            reduce_only=False,
            time_in_force="GTC",
            expire_time=None
        )
        orders.append(order)
        await order_engine.place_order(order)
    
    # Wait for processing
    await asyncio.sleep(0.5)
    
    # Check memory usage
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    
    # Memory usage should be reasonable
    assert memory_info.rss < 100 * 1024 * 1024  # Less than 100MB
    
    # Clean up old orders
    order_engine._cleanup_old_orders()
    
    # Check orders were cleaned up
    assert len(order_engine.orders) < 1000

@pytest.mark.asyncio
async def test_performance_under_load(order_engine):
    """Test performance under heavy load."""
    await order_engine.start()
    
    # Create many orders
    orders = []
    for i in range(100):
        order = Order(
            id=f"test_performance_{i}",
            symbol="BTC/USD",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=None,
            stop_price=None,
            status=OrderStatus.NEW,
            filled_quantity=Decimal("0"),
            average_price=Decimal("0"),
            created_at=time.time(),
            updated_at=time.time(),
            client_order_id=None,
            post_only=False,
            reduce_only=False,
            time_in_force="GTC",
            expire_time=None
        )
        orders.append(order)
    
    # Mock exchange execution
    async def mock_execute(order):
        return {
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side.value,
            'type': order.type.value,
            'quantity': float(order.quantity),
            'price': 50000.0,
            'timestamp': time.time()
        }
    
    order_engine._send_to_exchange = mock_execute
    
    # Measure execution time
    start_time = time.time()
    
    # Place orders concurrently
    await asyncio.gather(*[
        order_engine.place_order(order)
        for order in orders
    ])
    
    # Wait for processing
    await asyncio.sleep(0.5)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Check performance
    assert execution_time < 1.0  # Should process 100 orders in less than 1 second
    
    # Check all orders were processed
    for order in orders:
        processed_order = order_engine.get_order(order.id)
        assert processed_order is not None
        assert processed_order.status == OrderStatus.FILLED 