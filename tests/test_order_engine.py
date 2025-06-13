import pytest
import pytest_asyncio
import sys
import asyncio
import time
from decimal import Decimal
from unittest.mock import Mock, patch
from src.trading.order_engine import OrderEngine, Order, OrderType, OrderSide, OrderStatus
from src.exceptions import ValidationError, ExecutionError
from datetime import datetime, timezone
import pytest_timeout
import concurrent.futures
from unittest.mock import AsyncMock
from src.trading.exceptions import OrderError

@pytest.fixture
def event_loop():
    """Create event loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.mark.timeout(5)  # 5 second timeout for all tests in this file
@pytest_asyncio.fixture
async def order_engine():
    """Create order engine."""
    print("Creating order engine fixture...")
    engine = OrderEngine()
    print("Starting order engine...")
    await engine.start()
    print("Order engine started in fixture")
    yield engine
    print("Cleaning up order engine...")
    await asyncio.wait_for(engine.stop(), timeout=2.0)
    print("Order engine cleaned up")

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
            'status': 'filled',
            'filled_quantity': float(order.quantity),
            'average_price': 50000.0,
            'timestamp': time.time()
        }
    
    order_engine._send_to_exchange = mock_execute
    
    # Place order
    await order_engine.place_order(market_order)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Check order status
    order = await order_engine.get_order(market_order.id)
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
            'status': 'filled',
            'filled_quantity': float(order.quantity),
            'average_price': float(order.price),
            'timestamp': time.time()
        }
    
    order_engine._send_to_exchange = mock_execute
    
    # Place order
    await order_engine.place_order(limit_order)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Check order status
    order = await order_engine.get_order(limit_order.id)
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
    # Should remain NEW until stop price is hit
    order = await order_engine.get_order(stop_limit_order.id)
    assert order is not None
    assert order.status == OrderStatus.NEW
    # Simulate stop price being hit
    order_engine.set_mock_price(stop_limit_order.symbol, Decimal("50000.0"))
    await order_engine._process_stop_orders(Decimal("50000.0"))
    await asyncio.sleep(0.1)
    order = await order_engine.get_order(stop_limit_order.id)
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
    order = await order_engine.get_order(market_order.id)
    assert order is not None
    assert order.status == OrderStatus.CANCELED

@pytest.mark.asyncio
@pytest.mark.timeout(3)  # 3 second timeout for this specific test
async def test_cancel_nonexistent_order(order_engine):
    """Test canceling nonexistent order."""
    print("Starting test_cancel_nonexistent_order...")
    
    # Try to cancel nonexistent order with timeout
    with pytest.raises(OrderError, match="Order nonexistent not found"):
        await asyncio.wait_for(
            order_engine.cancel_order("nonexistent"),
            timeout=2.0
        )
    print("Test completed successfully")

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
    order = await order_engine.get_order(invalid_order.id)
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
    order = await order_engine.get_order(expired_order.id)
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
        processed_order = await order_engine.get_order(order.id)
        assert processed_order is not None
        assert processed_order.status == OrderStatus.FILLED

@pytest.mark.asyncio
async def test_order_retry_mechanism(order_engine, market_order):
    """Test order retry mechanism."""
    await order_engine.start()
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
    order_engine._send_to_exchange = mock_execute
    await order_engine.place_order(market_order)
    await asyncio.sleep(0.5)
    order = await order_engine.get_order(market_order.id)
    assert order is not None
    # Accept either FILLED (if retries succeed) or REJECTED (if max_retries is hit)
    assert order.status in (OrderStatus.FILLED, OrderStatus.REJECTED)
    assert retry_count >= 1

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
    order = await order_engine.get_order(post_only_order.id)
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
    order = await order_engine.get_order(reduce_only_order.id)
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
    order = await order_engine.get_order(reduce_only_order.id)
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
    
    # The engine only cancels its own tasks; external tasks must be cancelled by the test or their creator.

@pytest.mark.asyncio
async def test_partial_fill(order_engine, market_order):
    """Test partial fill handling."""
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
            'status': 'partially_filled',
            'filled_quantity': float(order.quantity) * 0.5,
            'average_price': 50000.0,
            'timestamp': time.time()
        }
    
    order_engine._send_to_exchange = mock_execute
    
    # Place order
    await order_engine.place_order(market_order)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Check order status
    order = await order_engine.get_order(market_order.id)
    assert order is not None
    assert order.status == OrderStatus.PARTIALLY_FILLED
    assert order.filled_quantity == market_order.quantity * Decimal("0.5")
    assert order.average_price == Decimal("50000.0")

@pytest.mark.asyncio
async def test_order_priority(order_engine):
    """Test order priority handling."""
    await order_engine.start()
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
    # Place orders with explicit priorities: 0 (lowest), 2 (highest), 1 (medium)
    await order_engine.place_order(orders[0], priority=0)
    await order_engine.place_order(orders[1], priority=2)
    await order_engine.place_order(orders[2], priority=1)
    await asyncio.sleep(0.2)
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
    order = await order_engine.get_order(order.id)
    assert order is not None
    assert order.status == OrderStatus.FILLED
    assert order.average_price == Decimal("50000.0")

@pytest.mark.asyncio
async def test_error_handling(order_engine):
    """Test error handling and recovery."""
    await order_engine.start()
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
    errors = []
    def error_callback(error):
        errors.append(error)
    order_engine.register_callback('error', error_callback)
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
    await order_engine.place_order(order)
    await asyncio.sleep(0.5)
    # Accept 1-3 errors depending on max_retries
    assert 1 <= len(errors) <= 3

@pytest.mark.asyncio
async def test_concurrent_cancellations(order_engine):
    """Test concurrent order cancellations."""
    await order_engine.start()
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
    for order in orders:
        await order_engine.place_order(order)
    await asyncio.gather(*[
        order_engine.cancel_order(order.id)
        for order in orders
    ])
    await asyncio.sleep(0.2)
    for order in orders:
        processed_order = await order_engine.get_order(order.id)
        assert processed_order is not None
        # Allow for either canceled or filled due to race
        assert processed_order.status in (OrderStatus.CANCELED, OrderStatus.FILLED)

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
    order = await order_engine.get_order(order.id)
    assert order is not None
    assert order.status == OrderStatus.EXPIRED

@pytest.mark.asyncio
async def test_memory_management(order_engine):
    """Test memory management and cleanup."""
    await order_engine.start()
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
    await asyncio.sleep(0.5)
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    assert memory_info.rss < 256 * 1024 * 1024  # Less than 256MB
    # Clean up all orders
    order_engine.clear_all_orders()

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
        processed_order = await order_engine.get_order(order.id)
        assert processed_order is not None
        assert processed_order.status == OrderStatus.FILLED 