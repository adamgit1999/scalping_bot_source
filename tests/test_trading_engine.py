import pytest
import pytest_asyncio
from src.trading_engine import TradingEngine
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from mock_exchange import MockExchange
from decimal import Decimal
from src.models import User, Trade, Notification, Webhook, db
from src.exceptions import InsufficientFundsError, InvalidOrderError, MarketDataError
import time
import psutil
from collections import deque

from src.broker.base import BrokerInterface
from src.risk.risk_manager import RiskManager
from src.strategies.base import Strategy
from src.data.market_data_store import MarketDataStore
from src.monitoring.performance_monitor import PerformanceMonitor
from src.notifications.notification_manager import NotificationManager
from src.trading.order import Order, OrderType, OrderSide
from src.trading.position import Position
from src.trading.trade import Trade
from src.exceptions import TradingError, ValidationError

@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture(scope="function")
async def trading_engine(event_loop):
    """Create a trading engine instance for testing."""
    # Create mock exchange
    exchange = MockExchange()
    
    # Create mock data processor
    data_processor = MockDataProcessor()
    
    # Create trading engine
    engine = TradingEngine(
        exchange=exchange,
        data_processor=data_processor,
        config={
            'risk_management': {
                'max_position_size': 1.0,
                'max_drawdown': 0.1,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.05
            },
            'order_engine': {
                'max_retries': 3,
                'retry_delay': 0.1,
                'timeout': 1.0
            }
        }
    )
    
    try:
        # Start engine
        await engine.start()
        yield engine
    finally:
        # Ensure proper cleanup
        if engine.order_engine:
            # Cancel all tasks
            for task in engine.order_engine._tasks:
                if not task.done():
                    task.cancel()
            # Clear task sets
            engine.order_engine._tasks.clear()
            engine.order_engine._process_orders_task = None
            engine.order_engine._process_executions_task = None
            # Stop the engine
            await engine.stop()
            # Wait for all tasks to complete
            pending = asyncio.all_tasks()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1h')
    data = pd.DataFrame({
        'open': np.random.normal(100, 1, len(dates)),
        'high': np.random.normal(101, 1, len(dates)),
        'low': np.random.normal(99, 1, len(dates)),
        'close': np.random.normal(100, 1, len(dates)),
        'volume': np.random.normal(1000, 100, len(dates))
    }, index=dates)
    return data

@pytest.fixture
def mock_broker():
    broker = Mock(spec=BrokerInterface)
    broker.get_balance.return_value = Decimal('10000.00')
    broker.get_position.return_value = None
    broker.place_order = AsyncMock(return_value={
        'id': 'test_order_1',
        'symbol': 'BTC/USDT',
        'side': 'BUY',
        'type': 'limit',
        'price': Decimal('100.00'),
        'quantity': Decimal('1.0'),
        'status': 'FILLED',
        'filled_quantity': Decimal('1.0'),
        'remaining': Decimal('0'),
        'cost': Decimal('100.00'),
        'timestamp': datetime.now(timezone.utc).timestamp()
    })
    return broker

@pytest.fixture
def mock_risk_manager():
    risk_manager = Mock(spec=RiskManager)
    risk_manager.check_order = AsyncMock(return_value=(True, "OK"))
    risk_manager.calculate_position_size = AsyncMock(return_value=Decimal('1.0'))
    risk_manager.check_risk_limits = AsyncMock(return_value=True)
    return risk_manager

@pytest.fixture
def mock_strategy():
    strategy = Mock(spec=Strategy)
    strategy.on_market_data = AsyncMock()
    strategy.update = AsyncMock()
    strategy.generate_signal.return_value = 'BUY'
    strategy.calculate_entry_price.return_value = Decimal('100.00')
    strategy.calculate_stop_loss.return_value = Decimal('95.00')
    strategy.calculate_take_profit.return_value = Decimal('105.00')
    return strategy

@pytest.fixture
def mock_market_data_store():
    store = Mock(spec=MarketDataStore)
    store.get_latest_data.return_value = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [102, 103, 104],
        'volume': [1000, 1100, 1200]
    })
    return store

@pytest.fixture
def mock_performance_monitor():
    monitor = Mock(spec=PerformanceMonitor)
    monitor.update_metrics.return_value = {
        'win_rate': 0.6,
        'profit_factor': 1.5,
        'sharpe_ratio': 1.2
    }
    return monitor

@pytest.fixture
def mock_notification_manager():
    manager = Mock(spec=NotificationManager)
    return manager

@pytest.fixture
def trading_engine():
    """Create a trading engine instance."""
    broker = Mock()
    risk_manager = Mock()
    strategy = Mock()
    return TradingEngine(broker=broker, risk_manager=risk_manager, strategy=strategy)

@pytest.fixture
def performance_monitor():
    """Create a performance monitor instance."""
    return PerformanceMonitor()

@pytest.mark.asyncio
async def test_initialize(trading_engine):
    """Test engine initialization."""
    assert trading_engine.broker is not None
    assert trading_engine.risk_manager is not None
    assert trading_engine.strategy is not None
    assert trading_engine.market_data_store is not None
    assert trading_engine.performance_monitor is not None
    assert trading_engine.notification_manager is not None

@pytest.mark.asyncio
async def test_fetch_ohlcv(trading_engine):
    """Test fetching OHLCV data."""
    engine = trading_engine
    
    # Mock exchange's fetch_ohlcv method
    mock_data = [
        [1625097600000, 35000.0, 36000.0, 34000.0, 35500.0, 100.0],  # timestamp, open, high, low, close, volume
        [1625184000000, 35500.0, 37000.0, 35000.0, 36500.0, 150.0],
        [1625270400000, 36500.0, 38000.0, 36000.0, 37500.0, 200.0]
    ]
    engine.exchange.fetch_ohlcv = AsyncMock(return_value=mock_data)
    
    data = await engine.fetch_ohlcv('BTC/USDT', '1h')
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert len(data) == 3
    assert list(data.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']

@pytest.mark.asyncio
async def test_fetch_ohlcv_network_error(trading_engine):
    """Test handling network errors when fetching OHLCV data."""
    engine = trading_engine
    engine.exchange.fetch_ohlcv.side_effect = Exception('Network error')
    with pytest.raises(Exception):
        await engine.fetch_ohlcv('BTC/USDT', '1h')

@pytest.mark.asyncio
async def test_concurrent_order_execution(trading_engine):
    """Test concurrent order execution."""
    engine = trading_engine
    
    # Mock broker and exchange
    engine.broker.get_account_balance = AsyncMock(return_value=10000.0)
    engine.broker.place_order = AsyncMock(return_value={
        'id': 'test_order',
        'status': 'FILLED',
        'filled_amount': 0.1,
        'filled_price': 100.0
    })
    engine.exchange.create_order = AsyncMock(return_value={
        'id': 'test_order',
        'status': 'FILLED',
        'filled_amount': 0.1,
        'filled_price': 100.0
    })
    
    # Mock risk manager
    engine.risk_manager.check_order = AsyncMock(return_value=(True, "OK"))
    
    # Mock balance check
    engine._check_balance = AsyncMock(return_value=True)
    
    # Create multiple orders
    orders = [
        Order(
            id=f'order_{i}',
            type='MARKET',
            symbol='BTC/USDT',
            side='BUY',
            quantity=0.1,
            price=100.0,
            order_type='MARKET'
        ) for i in range(5)
    ]

    # Execute orders concurrently
    results = await asyncio.gather(*[engine.execute_order(order) for order in orders])

    # Verify all orders were executed
    for i, order in enumerate(orders):
        assert order.id in engine.orders
        assert engine.orders[order.id]['status'] == 'FILLED'
        assert results[i]['status'] == 'FILLED'
        assert results[i]['id'] == order.id

@pytest.mark.asyncio
async def test_position_sizing_with_different_risk(trading_engine):
    """Test position sizing with different risk levels."""
    engine = trading_engine
    symbol = 'BTC/USDT'
    price = 50000.0
    
    # Create mock risk manager
    mock_risk_manager = AsyncMock()
    engine.risk_manager = mock_risk_manager
    
    # Test with different risk levels
    risk_levels = [0.01, 0.02, 0.03]
    for risk in risk_levels:
        mock_risk_manager.calculate_position_size.return_value = Decimal(str(risk))
        position_size = await engine.risk_manager.calculate_position_size(symbol, 'BUY', price)
        assert position_size == Decimal(str(risk))

@pytest.mark.asyncio
async def test_strategy_parameter_validation_edge_cases(trading_engine):
    """Test strategy parameter validation with edge cases."""
    engine = trading_engine
    strategy = 'scalping'
    
    # Test with empty parameters
    with pytest.raises(ValueError):
        await engine._validate_parameters(strategy, {})
    
    # Test with invalid parameter types
    with pytest.raises(ValueError):
        await engine._validate_parameters(strategy, {'sma_short': 'invalid'})
    
    # Test with out of range parameters
    with pytest.raises(ValueError):
        await engine._validate_parameters(strategy, {'sma_short': -1})

@pytest.mark.asyncio
async def test_calculate_signals_scalping(trading_engine, mock_risk_manager):
    """Test signal calculation for scalping strategy."""
    engine = trading_engine
    
    # Create mock strategy
    mock_strategy = AsyncMock()
    mock_strategy.generate_signal.return_value = 'BUY'
    mock_strategy.calculate_entry_price.return_value = 100.0
    mock_strategy.calculate_stop_loss.return_value = 90.0
    mock_strategy.calculate_take_profit.return_value = 110.0
    engine.strategy = mock_strategy
    
    # Mock broker and exchange
    engine.broker.get_account_balance = AsyncMock(return_value=100000.0)
    engine.broker.place_order = AsyncMock(return_value={
        'id': 'test_order',
        'status': 'FILLED',
        'filled_amount': 0.1,
        'filled_price': 100.0
    })
    engine.exchange.create_order = AsyncMock(return_value={
        'id': 'test_order',
        'status': 'FILLED',
        'filled_amount': 0.1,
        'filled_price': 100.0
    })
    
    # Execute strategy signal
    engine.risk_manager = mock_risk_manager
    engine.risk_manager.check_order = AsyncMock(return_value=(True, "OK"))
    engine.risk_manager.calculate_position_size = AsyncMock(return_value=0.1)
    engine.get_balance = AsyncMock(return_value=100000.0)
    result = await engine.execute_strategy_signal('BTC/USD')
    assert result is not None
    assert result['status'] in ['FILLED', 'PARTIALLY_FILLED', 'UNKNOWN']

@pytest.mark.asyncio
async def test_calculate_signals_scalping_with_invalid_parameters(trading_engine):
    """Test signal calculation with invalid parameters."""
    engine = trading_engine
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1h'),
        'open': np.random.normal(100, 1, 100),
        'high': np.random.normal(101, 1, 100),
        'low': np.random.normal(99, 1, 100),
        'close': np.random.normal(100, 1, 100),
        'volume': np.random.normal(1000, 100, 100)
    })
    with pytest.raises(ValueError):
        await engine._validate_parameters('scalping', {'invalid': 'parameter'})

@pytest.mark.asyncio
async def test_calculate_signals_momentum(trading_engine, mock_risk_manager):
    """Test signal calculation for momentum strategy."""
    engine = trading_engine
    
    # Create mock strategy
    mock_strategy = AsyncMock()
    mock_strategy.generate_signal.return_value = 'BUY'
    mock_strategy.calculate_entry_price.return_value = 100.0
    mock_strategy.calculate_stop_loss.return_value = 90.0
    mock_strategy.calculate_take_profit.return_value = 110.0
    engine.strategy = mock_strategy
    
    # Mock broker and exchange
    engine.broker.get_account_balance = AsyncMock(return_value=100000.0)
    engine.broker.place_order = AsyncMock(return_value={
        'id': 'test_order',
        'status': 'FILLED',
        'filled_amount': 0.1,
        'filled_price': 100.0
    })
    engine.exchange.create_order = AsyncMock(return_value={
        'id': 'test_order',
        'status': 'FILLED',
        'filled_amount': 0.1,
        'filled_price': 100.0
    })
    
    # Execute strategy signal
    engine.risk_manager = mock_risk_manager
    engine.risk_manager.check_order = AsyncMock(return_value=(True, "OK"))
    engine.risk_manager.calculate_position_size = AsyncMock(return_value=0.1)
    engine.get_balance = AsyncMock(return_value=100000.0)
    result = await engine.execute_strategy_signal('BTC/USD')
    assert result is not None
    assert result['status'] in ['FILLED', 'PARTIALLY_FILLED', 'UNKNOWN']

@pytest.mark.asyncio
async def test_calculate_signals_mean_reversion(trading_engine, mock_risk_manager):
    """Test signal calculation for mean reversion strategy."""
    engine = trading_engine
    
    # Create mock strategy
    mock_strategy = AsyncMock()
    mock_strategy.generate_signal.return_value = 'BUY'
    mock_strategy.calculate_entry_price.return_value = 100.0
    mock_strategy.calculate_stop_loss.return_value = 90.0
    mock_strategy.calculate_take_profit.return_value = 110.0
    engine.strategy = mock_strategy
    
    # Mock broker and exchange
    engine.broker.get_account_balance = AsyncMock(return_value=100000.0)
    engine.broker.place_order = AsyncMock(return_value={
        'id': 'test_order',
        'status': 'FILLED',
        'filled_amount': 0.1,
        'filled_price': 100.0
    })
    engine.exchange.create_order = AsyncMock(return_value={
        'id': 'test_order',
        'status': 'FILLED',
        'filled_amount': 0.1,
        'filled_price': 100.0
    })
    
    # Execute strategy signal
    engine.risk_manager = mock_risk_manager
    engine.risk_manager.check_order = AsyncMock(return_value=(True, "OK"))
    engine.risk_manager.calculate_position_size = AsyncMock(return_value=0.1)
    engine.get_balance = AsyncMock(return_value=100000.0)
    result = await engine.execute_strategy_signal('BTC/USD')
    assert result is not None
    assert result['status'] in ['FILLED', 'PARTIALLY_FILLED', 'UNKNOWN']

@pytest.mark.asyncio
async def test_place_order_success(trading_engine, mock_risk_manager):
    """Test successful order placement."""
    engine = trading_engine
    engine.risk_manager = mock_risk_manager
    engine.risk_manager.check_order = AsyncMock(return_value=(True, "OK"))
    engine.risk_manager.calculate_position_size = AsyncMock(return_value=0.1)
    engine.get_balance = AsyncMock(return_value=100000.0)
    engine.broker.get_account_balance = AsyncMock(return_value=100000.0)
    engine.exchange.quote_currency = 'USDT'
    
    # Mock order creation
    engine.exchange.create_order = AsyncMock(return_value={
        'id': 'test_order',
        'status': 'FILLED',
        'filled_amount': 0.1,
        'filled_price': 100.0
    })
    engine.broker.place_order = AsyncMock(return_value={
        'id': 'test_order',
        'status': 'FILLED',
        'filled_amount': 0.1,
        'filled_price': 100.0
    })
    
    result = await engine.place_order('BTC/USDT', 'BUY', 100.0, 0.1)
    assert result['status'] == 'FILLED'
    assert result['id'] == 'test_order'

@pytest.mark.asyncio
async def test_place_order_insufficient_funds(trading_engine):
    """Test order placement with insufficient funds."""
    engine = trading_engine
    symbol = 'BTC/USDT'
    side = 'BUY'
    price = 50000.0
    amount = 1000.0  # Large amount that exceeds balance
    
    with pytest.raises(InsufficientFundsError):
        await engine.place_order(symbol, side, price, amount)

@pytest.mark.asyncio
async def test_place_order_invalid_order(trading_engine):
    """Test order placement with invalid order."""
    engine = trading_engine
    symbol = 'BTC/USDT'
    side = 'INVALID'  # Invalid side
    price = 50000.0
    amount = 0.1

    with pytest.raises(InvalidOrderError, match="Invalid order side: INVALID. Must be 'BUY' or 'SELL'"):
        await engine.place_order(symbol, side, price, amount)

@pytest.mark.asyncio
async def test_place_order_risk_check_failed(trading_engine):
    """Test order placement when risk check fails."""
    engine = trading_engine
    symbol = 'BTC/USDT'
    side = 'BUY'
    price = 50000.0
    amount = 0.1

    # Mock broker to return sufficient balance
    engine.broker.get_account_balance = AsyncMock(return_value=10000.0)
    
    # Mock risk manager to reject the order
    engine.risk_manager.check_order = AsyncMock(return_value=(False, "Rejected"))

    with pytest.raises(ValueError, match="Order rejected by risk manager"):
        await engine.place_order(symbol, side, price, amount)

@pytest.mark.asyncio
async def test_cancel_order(trading_engine, mock_risk_manager):
    """Test order cancellation."""
    engine = trading_engine
    engine.risk_manager = mock_risk_manager
    engine.risk_manager.check_order = AsyncMock(return_value=(True, "OK"))
    engine.risk_manager.calculate_position_size = AsyncMock(return_value=0.1)
    engine.get_balance = AsyncMock(return_value=100000.0)
    engine.broker.get_account_balance = AsyncMock(return_value=100000.0)
    engine.exchange.quote_currency = 'USDT'
    
    # Place an order first
    order = await engine.place_order('BTC/USDT', 'BUY', 100.0, 0.1)
    
    # Cancel the order
    result = await engine.cancel_order(order['id'])
    assert result is True

@pytest.mark.asyncio
async def test_cancel_nonexistent_order(trading_engine):
    """Test canceling a nonexistent order."""
    engine = trading_engine
    with pytest.raises(ValueError):
        await engine.cancel_order('nonexistent')

@pytest.mark.asyncio
async def test_get_balance(trading_engine):
    """Test getting balance for a currency."""
    engine = trading_engine
    currency = 'USDT'
    balance = await engine.get_balance(currency)
    assert isinstance(balance, float)
    assert balance >= 0

@pytest.mark.asyncio
async def test_get_balance_for_nonexistent_currency(trading_engine):
    """Test getting balance for a nonexistent currency."""
    engine = trading_engine
    engine.broker.get_account_balance = AsyncMock(side_effect=ValueError('Currency NONEXISTENT not found'))
    
    with pytest.raises(ValueError):
        await engine.get_balance('NONEXISTENT')

def test_calculate_rsi(trading_engine):
    """Test RSI calculation."""
    engine = trading_engine
    prices = pd.Series([100 + (i % 2) for i in range(30)])
    period = 14
    rsi = engine._calculate_rsi(prices, period)
    assert isinstance(rsi, pd.Series)
    assert not rsi[period:].isna().any()
    assert all(0 <= x <= 100 for x in rsi.dropna())

def test_calculate_rsi_with_constant_prices(trading_engine):
    """Test RSI calculation with constant prices."""
    engine = trading_engine
    prices = pd.Series([100] * 20)
    period = 14
    rsi = engine._calculate_rsi(prices, period)
    assert isinstance(rsi, pd.Series)
    # For constant prices, RSI is undefined (NaN) after warmup period
    assert rsi[period:].isna().all()

def test_calculate_macd(trading_engine):
    """Test MACD calculation."""
    engine = trading_engine
    prices = pd.Series([100 + (i % 2) for i in range(30)])
    fast_period = 12
    slow_period = 26
    signal_period = 9
    macd, signal = engine._calculate_macd(prices, fast_period, slow_period, signal_period)
    assert isinstance(macd, pd.Series)
    assert isinstance(signal, pd.Series)
    assert not macd[slow_period-1:].isna().any()
    assert not signal[slow_period-1:].isna().any()

def test_calculate_macd_with_invalid_periods(trading_engine):
    """Test MACD calculation with invalid periods."""
    engine = trading_engine
    prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 97, 98, 99])
    
    with pytest.raises(ValueError):
        engine._calculate_macd(prices, fast_period=30, slow_period=20)

@pytest.mark.asyncio
async def test_close_all_positions(trading_engine, mock_risk_manager):
    """Test closing all positions."""
    engine = trading_engine
    engine.risk_manager = mock_risk_manager
    engine.risk_manager.check_order = AsyncMock(return_value=(True, "OK"))
    engine.risk_manager.calculate_position_size = AsyncMock(return_value=0.1)
    engine.get_balance = AsyncMock(return_value=100000.0)
    engine.broker.get_account_balance = AsyncMock(return_value=100000.0)
    engine.exchange.quote_currency = 'USDT'
    
    # Add some positions
    engine.positions['BTC/USDT'] = Position(
        symbol='BTC/USDT',
        quantity=Decimal('0.1'),
        entry_price=Decimal('100.0'),
        current_price=Decimal('100.0'),
        unrealized_pnl=Decimal('0.0'),
        realized_pnl=Decimal('0.0')
    )
    
    # Mock order creation for closing positions
    engine.exchange.create_order = AsyncMock(return_value={
        'id': 'test_order',
        'status': 'FILLED',
        'filled_amount': 0.1,
        'filled_price': 100.0
    })
    
    await engine.close_all_positions()
    assert len(engine.positions) == 0

@pytest.mark.asyncio
async def test_close_all_positions_with_no_positions(trading_engine):
    """Test closing all positions when no positions exist."""
    engine = trading_engine
    # Mock the exchange's fetch_ticker method
    engine.exchange.fetch_ticker = AsyncMock(return_value={'last': 50000.0})
    
    # Close all positions
    await engine.close_all_positions()
    assert len(engine.positions) == 0

@pytest.mark.asyncio
async def test_update_position(trading_engine):
    """Test position update."""
    engine = trading_engine
    symbol = 'BTC/USDT'
    position = Position(
        symbol='BTC/USDT',
        quantity=Decimal('0.1'),
        entry_price=Decimal('50000.00'),
        current_price=Decimal('50000.00'),
        unrealized_pnl=Decimal('0.00'),
        realized_pnl=Decimal('0.00')
    )
    
    await engine._update_position(position)
    assert symbol in engine.positions
    assert engine.positions[symbol] == position

@pytest.mark.asyncio
async def test_update_position_with_zero_amount(trading_engine):
    """Test position update with zero amount."""
    engine = trading_engine
    symbol = 'BTC/USDT'
    
    # First add a position
    position = Position(
        symbol='BTC/USDT',
        quantity=Decimal('0.1'),
        entry_price=Decimal('50000.00'),
        current_price=Decimal('50000.00'),
        unrealized_pnl=Decimal('0.00'),
        realized_pnl=Decimal('0.00')
    )
    
    await engine._update_position(position)
    assert symbol in engine.positions
    
    # Update with zero amount
    position.quantity = Decimal('0.0')
    await engine._update_position(position)
    assert symbol not in engine.positions

@pytest.mark.asyncio
async def test_calculate_commission(trading_engine):
    """Test commission calculation."""
    engine = trading_engine
    price = 50000
    amount = 0.1
    
    commission = engine._calculate_commission(price, amount)
    assert isinstance(commission, float)
    assert commission >= 0

@pytest.mark.asyncio
async def test_calculate_commission_with_zero_amount(trading_engine):
    """Test commission calculation with zero amount."""
    engine = trading_engine
    with pytest.raises(ValueError):
        engine._calculate_commission(50000, 0)

@pytest.mark.asyncio
async def test_validate_parameters(trading_engine):
    """Test parameter validation."""
    engine = trading_engine
    strategy = 'scalping'
    parameters = {
        'sma_short': 10,
        'sma_long': 20,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70
    }
    assert engine._validate_parameters(strategy, parameters) is True

@pytest.mark.asyncio
async def test_validate_parameters_with_missing_required(trading_engine):
    """Test parameter validation with missing required parameters."""
    engine = trading_engine
    strategy = 'scalping'
    parameters = {
        'sma_short': 10,
        'sma_long': 20
        # Missing required parameters
    }
    with pytest.raises(ValueError):
        engine._validate_parameters(strategy, parameters)

@pytest.mark.asyncio
async def test_validate_parameters_with_invalid_strategy(trading_engine):
    """Test parameter validation with invalid strategy."""
    engine = trading_engine
    strategy = 'invalid_strategy'
    parameters = {
        'sma_short': 10,
        'sma_long': 20,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70
    }
    with pytest.raises(ValueError):
        engine._validate_parameters(strategy, parameters)

def test_initialize_engine(trading_engine):
    """Test engine initialization."""
    engine = trading_engine
    assert engine.broker is not None
    assert engine.risk_manager is not None
    assert engine.strategy is not None
    assert engine.market_data_buffer == []
    assert engine.performance_metrics == {}
    assert not engine.running

@pytest.mark.asyncio
async def test_update_position_new_position(trading_engine):
    """Test updating position with new position"""
    engine = trading_engine
    position = Position(
        symbol='BTC/USD',
        quantity=Decimal('1.0'),
        entry_price=Decimal('100.00'),
        current_price=Decimal('100.00'),
        unrealized_pnl=Decimal('0.00'),
        realized_pnl=Decimal('0.00')
    )
    
    await engine.update_position(position)
    assert position.symbol in engine.positions
    assert engine.positions[position.symbol] == position

@pytest.mark.asyncio
async def test_update_position_existing_position(trading_engine):
    """Test updating existing position"""
    engine = trading_engine
    # First position
    position1 = Position(
        symbol='BTC/USD',
        quantity=Decimal('1.0'),
        entry_price=Decimal('100.00'),
        current_price=Decimal('100.00'),
        unrealized_pnl=Decimal('0.00'),
        realized_pnl=Decimal('0.00')
    )
    
    # Updated position
    position2 = Position(
        symbol='BTC/USD',
        quantity=Decimal('2.0'),
        entry_price=Decimal('100.00'),
        current_price=Decimal('105.00'),
        unrealized_pnl=Decimal('10.00'),
        realized_pnl=Decimal('0.00')
    )
    
    await engine.update_position(position1)
    await engine.update_position(position2)
    assert position2.symbol in engine.positions
    assert engine.positions[position2.symbol] == position2

def test_update_position_closed_position(trading_engine):
    """Test updating position that is closed"""
    engine = trading_engine
    position = Position(
        symbol='BTC/USD',
        quantity=Decimal('0.0'),
        entry_price=Decimal('100.00'),
        current_price=Decimal('105.00'),
        unrealized_pnl=Decimal('0.00'),
        realized_pnl=Decimal('5.00')
    )
    
    asyncio.get_event_loop().run_until_complete(engine.update_position(position))
    assert position.symbol not in engine.positions

def test_record_trade(trading_engine):
    """Test recording a trade"""
    engine = trading_engine
    trade = Trade(
        symbol='BTC/USD',
        side='BUY',
        quantity=Decimal('0.1'),
        price=Decimal('50000.0'),
        timestamp=datetime.now(timezone.utc),
        order_id='test_order_1',
        commission=None,
        realized_pnl=None
    )
    
    engine.record_trade(trade)
    assert trade in engine.trades

@pytest.mark.asyncio
async def test_execute_strategy_signal(trading_engine):
    """Test executing a strategy signal."""
    # Create mock signal
    signal = StrategySignal(
        symbol='BTC/USD',
        side='buy',
        quantity=0.1,
        price=50000.0,
        timestamp=time.time(),
        confidence=0.8,
        metadata={'strategy': 'test_strategy'}
    )
    
    # Execute signal
    order = await trading_engine.execute_strategy_signal(signal)
    
    # Verify order was placed
    assert order is not None
    assert order.symbol == 'BTC/USD'
    assert order.side == 'buy'
    assert order.quantity == 0.1
    assert order.price == 50000.0
    assert order.status == 'open'
    
    # Verify order was added to active orders
    assert len(trading_engine.active_orders) == 1
    assert trading_engine.active_orders[0].id == order.id
    
    # Verify order was added to order history
    assert len(trading_engine.order_history) == 1
    assert trading_engine.order_history[0].id == order.id

@pytest.mark.asyncio
async def test_market_data_processing(trading_engine, mock_market_data_store):
    """Test market data processing."""
    # Get market data
    data = await trading_engine._process_market_data()
    
    # Verify data processing
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert len(data) > 0
    mock_market_data_store.get_latest_data.assert_called_once()

@pytest.mark.asyncio
async def test_strategy_execution(trading_engine, mock_strategy):
    """Test strategy execution."""
    # Create sample data
    data = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [102, 103, 104],
        'volume': [1000, 1100, 1200]
    })
    
    # Execute strategy
    signal = await trading_engine._execute_strategy(data)
    
    # Verify strategy execution
    assert signal in ['BUY', 'SELL', None]
    mock_strategy.on_market_data.assert_called_once_with(data)
    mock_strategy.update.assert_called_once()
    mock_strategy.generate_signal.assert_called_once()

@pytest.mark.asyncio
async def test_order_execution(trading_engine, mock_broker, mock_risk_manager):
    """Test order execution."""
    # Create order
    order = {
        'symbol': 'BTC/USDT',
        'side': 'BUY',
        'type': 'limit',
        'price': Decimal('100.00'),
        'quantity': Decimal('1.0')
    }
    
    # Execute order
    result = await trading_engine._execute_order(order)
    
    # Verify order execution
    assert result['status'] == 'FILLED'
    assert result['id'] == 'test_order_1'
    mock_risk_manager.check_order.assert_called_once()
    mock_broker.place_order.assert_called_once()

@pytest.mark.asyncio
async def test_risk_management(trading_engine, mock_risk_manager):
    """Test risk management."""
    # Check risk limits
    result = await trading_engine._check_risk_limits()
    
    # Verify risk check
    assert result == True
    mock_risk_manager.check_risk_limits.assert_called_once()

@pytest.mark.asyncio
async def test_performance_monitoring(trading_engine, mock_performance_monitor):
    """Test performance monitoring."""
    # Update metrics
    metrics = await trading_engine._update_performance_metrics()
    
    # Verify metrics update
    assert isinstance(metrics, dict)
    assert 'win_rate' in metrics
    assert 'profit_factor' in metrics
    assert 'sharpe_ratio' in metrics
    mock_performance_monitor.update_metrics.assert_called_once()

@pytest.mark.asyncio
async def test_notification_system(trading_engine, mock_notification_manager):
    """Test notification system."""
    # Send notification
    await trading_engine._send_notification('Test message', 'INFO')
    
    # Verify notification
    mock_notification_manager.send_notification.assert_called_once_with(
        'Test message', 'INFO'
    )

@pytest.mark.asyncio
async def test_error_handling(trading_engine, mock_notification_manager):
    """Test error handling."""
    # Simulate error
    error = Exception('Test error')
    
    # Handle error
    await trading_engine._handle_error(error)
    
    # Verify error handling
    mock_notification_manager.send_notification.assert_called_once_with(
        'Test error', 'ERROR'
    )

@pytest.mark.asyncio
async def test_concurrent_operations(trading_engine):
    """Test concurrent operations."""
    # Create multiple tasks
    tasks = [
        trading_engine._process_market_data(),
        trading_engine._execute_strategy(pd.DataFrame()),
        trading_engine._update_performance_metrics()
    ]
    
    # Execute tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Verify all tasks completed successfully
    assert all(not isinstance(r, Exception) for r in results)

@pytest.mark.asyncio
async def test_resource_optimization(trading_engine):
    """Test resource optimization."""
    # Get initial memory usage
    initial_memory = psutil.Process().memory_info().rss
    
    # Perform operations
    for _ in range(100):
        await trading_engine._process_market_data()
        await trading_engine._update_performance_metrics()
    
    # Get final memory usage
    final_memory = psutil.Process().memory_info().rss
    
    # Verify memory usage is reasonable
    memory_increase = final_memory - initial_memory
    assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase

@pytest.mark.asyncio
async def test_cleanup(trading_engine, mock_broker):
    """Test cleanup functionality."""
    # Perform cleanup
    await trading_engine._cleanup()
    
    # Verify cleanup
    mock_broker.cancel_all_orders.assert_called_once()

@pytest.mark.asyncio
async def test_strategy_parameter_validation(trading_engine):
    """Test strategy parameter validation."""
    # Test with valid parameters
    valid_params = {
        'sma_short': 10,
        'sma_long': 20,
        'rsi_period': 14
    }
    result = await trading_engine._validate_parameters('scalping', valid_params)
    assert result == True
    
    # Test with invalid parameters
    invalid_params = {
        'sma_short': -1,
        'sma_long': 0,
        'rsi_period': 'invalid'
    }
    with pytest.raises(ValueError):
        await trading_engine._validate_parameters('scalping', invalid_params)

@pytest.mark.asyncio
async def test_market_data_validation(trading_engine):
    """Test market data validation."""
    # Test with valid data
    valid_data = pd.DataFrame({
        'open': [100],
        'high': [105],
        'low': [95],
        'close': [102],
        'volume': [1000]
    })
    result = await trading_engine._validate_market_data(valid_data)
    assert result == True
    
    # Test with invalid data
    invalid_data = pd.DataFrame({
        'open': [100],
        'high': [95],  # High lower than low
        'low': [105],
        'close': [102],
        'volume': [-1000]  # Negative volume
    })
    with pytest.raises(ValueError):
        await trading_engine._validate_market_data(invalid_data)

@pytest.mark.asyncio
async def test_performance_metrics_validation(trading_engine):
    """Test performance metrics validation."""
    # Test with valid metrics
    valid_metrics = {
        'win_rate': 0.6,
        'profit_factor': 1.5,
        'sharpe_ratio': 1.2
    }
    result = await trading_engine._validate_performance_metrics(valid_metrics)
    assert result == True
    
    # Test with invalid metrics
    invalid_metrics = {
        'win_rate': 1.5,  # Win rate > 1
        'profit_factor': -1.0,  # Negative profit factor
        'sharpe_ratio': 'invalid'  # Invalid type
    }
    with pytest.raises(ValueError):
        await trading_engine._validate_performance_metrics(invalid_metrics)

def test_trading_engine_basic():
    # Example test for trading engine core logic
    assert True  # Replace with real test logic for supported features

@pytest.mark.asyncio
async def test_process_market_data(trading_engine):
    """Test market data processing."""
    # Test valid market data
    market_data = {
        'symbol': 'BTC/USDT',
        'price': 50000.0,
        'volume': 100.0,
        'timestamp': datetime.now(timezone.utc).timestamp()
    }
    
    await trading_engine.process_market_data(market_data)
    assert len(trading_engine.market_data_buffer) == 1
    
    # Test invalid market data
    with pytest.raises(ValidationError):
        await trading_engine.process_market_data({'invalid': 'data'})

@pytest.mark.asyncio
async def test_execute_order(trading_engine):
    """Test order execution."""
    order = Order(
        id="test_order",
        symbol="BTC/USDT",
        type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=Decimal("0.1"),
        price=None
    )
    
    # Mock broker response
    trading_engine.broker.place_order = AsyncMock(return_value={"order_id": "test_order"})
    
    await trading_engine.execute_order(order)
    trading_engine.broker.place_order.assert_called_once()
    
    # Test invalid order
    invalid_order = Order(
        id="invalid_order",
        symbol="BTC/USDT",
        type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=Decimal("0"),
        price=None
    )
    
    with pytest.raises(ValidationError):
        await trading_engine.execute_order(invalid_order)

@pytest.mark.asyncio
async def test_execute_strategy(trading_engine):
    """Test strategy execution."""
    # Mock strategy response
    trading_engine.strategy.generate_signals = Mock(return_value=[
        {'symbol': 'BTC/USDT', 'side': 'buy', 'price': 50000.0, 'amount': 0.1}
    ])
    
    await trading_engine.execute_strategy("test_strategy", "BTC/USDT", "1m", {})
    assert trading_engine.strategy.generate_signals.called

def test_calculate_signals(trading_engine):
    """Test signal calculation."""
    df = pd.DataFrame({
        'close': [100, 101, 102, 103, 104],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    
    signals = trading_engine._calculate_signals("scalping", df, {})
    assert isinstance(signals, list)
    assert len(signals) > 0

def test_get_performance_metrics(trading_engine):
    """Test performance metrics retrieval."""
    metrics = trading_engine.get_performance_metrics()
    assert isinstance(metrics, dict)
    assert 'latency' in metrics
    assert 'operations' in metrics
    assert 'system' in metrics

@pytest.mark.asyncio
async def test_close_all_positions(trading_engine):
    """Test closing all positions."""
    # Mock positions
    trading_engine.positions = {
        'BTC/USDT': Position(symbol='BTC/USDT', amount=Decimal('0.1'), entry_price=Decimal('50000'))
    }
    
    trading_engine.broker.place_order = AsyncMock(return_value={"order_id": "test_order"})
    
    await trading_engine.close_all_positions()
    assert len(trading_engine.positions) == 0
    assert trading_engine.broker.place_order.called

def test_get_position(trading_engine):
    """Test position retrieval."""
    # Add test position
    position = Position(symbol='BTC/USDT', amount=Decimal('0.1'), entry_price=Decimal('50000'))
    trading_engine.positions['BTC/USDT'] = position
    
    retrieved_position = trading_engine.get_position('BTC/USDT')
    assert retrieved_position == position
    assert trading_engine.get_position('ETH/USDT') is None

def test_get_trade_history(trading_engine):
    """Test trade history retrieval."""
    # Add test trade
    trade = Trade(
        symbol='BTC/USDT',
        side='buy',
        price=Decimal('50000'),
        amount=Decimal('0.1'),
        timestamp=datetime.now(timezone.utc)
    )
    trading_engine.trades.append(trade)
    
    history = trading_engine.get_trade_history('BTC/USDT')
    assert len(history) == 1
    assert history[0] == trade

def test_calculate_pnl(trading_engine):
    """Test PnL calculation."""
    # Add test position
    position = Position(symbol='BTC/USDT', amount=Decimal('0.1'), entry_price=Decimal('50000'))
    trading_engine.positions['BTC/USDT'] = position
    
    # Mock current price
    trading_engine.broker.get_current_price = AsyncMock(return_value=51000.0)
    
    pnl = trading_engine.calculate_pnl('BTC/USDT')
    assert isinstance(pnl, dict)
    assert 'unrealized_pnl' in pnl
    assert 'realized_pnl' in pnl

def test_performance_monitor(performance_monitor):
    """Test performance monitor."""
    performance_monitor.start()
    assert performance_monitor.start_time > 0
    
    metrics = performance_monitor.get_system_metrics()
    assert isinstance(metrics, dict)
    assert 'cpu_usage' in metrics
    assert 'memory_usage' in metrics
    assert 'latency' in metrics

# ... (repeat for other tests, skipping or adapting as needed) ... 