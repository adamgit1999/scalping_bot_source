import pytest
import time
import asyncio
import psutil
import os
from src.app import app, db, User, Strategy, Trade
from src.trading_engine import TradingEngine
from src.backtest_engine import BacktestEngine
from datetime import datetime, timedelta, timezone
import random
import json
from sqlalchemy import func
from werkzeug.security import generate_password_hash
from src.models import Notification, Webhook
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock
from src.broker.base import BrokerInterface
from src.risk.risk_manager import RiskManager
from src.models import Strategy
from src.data.market_data_store import MarketDataStore
from src.monitoring.performance_monitor import PerformanceMonitor
from src.notification_system import NotificationSystem
from src.websocket.websocket_manager import WebSocketManager

@pytest.fixture
def app_context():
    """Create app context and initialize test database."""
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

@pytest.fixture
def test_user(app_context):
    """Create a test user."""
    user = User(
        username='testuser',
        email='test@example.com',
        password_hash=generate_password_hash('testpassword')
    )
    db.session.add(user)
    db.session.commit()
    return user

@pytest.fixture
def mock_broker():
    """Create a mock broker for testing."""
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
        'status': 'FILLED'
    })
    return broker

@pytest.fixture
def mock_risk_manager():
    """Create a mock risk manager for testing."""
    risk_manager = Mock(spec=RiskManager)
    risk_manager.check_order = AsyncMock(return_value=(True, "OK"))
    risk_manager.calculate_position_size = AsyncMock(return_value=Decimal('1.0'))
    risk_manager.check_risk_limits = AsyncMock(return_value=True)
    return risk_manager

@pytest.fixture
def mock_market_data_store():
    """Create a mock market data store for testing."""
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
    """Create a mock performance monitor for testing."""
    monitor = Mock(spec=PerformanceMonitor)
    monitor.update_metrics.return_value = {
        'win_rate': 0.6,
        'profit_factor': 1.5,
        'sharpe_ratio': 1.2
    }
    return monitor

@pytest.fixture
def mock_notification_manager():
    """Create a mock notification manager for testing."""
    manager = Mock(spec=NotificationSystem)
    return manager

@pytest.fixture
def mock_websocket_manager():
    """Create a mock websocket manager for testing."""
    manager = Mock(spec=WebSocketManager)
    return manager

@pytest.fixture
def trading_engine(mock_broker, mock_risk_manager, mock_market_data_store,
                  mock_performance_monitor, mock_notification_manager,
                  mock_websocket_manager):
    """Create a trading engine instance."""
    return TradingEngine(
        broker=mock_broker,
        risk_manager=mock_risk_manager,
        market_data_store=mock_market_data_store,
        performance_monitor=mock_performance_monitor,
        notification_manager=mock_notification_manager,
        websocket_manager=mock_websocket_manager,
        config={
            'risk_management': {
                'max_position_size': 1.0,
                'max_drawdown': 0.1,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.05
            }
        }
    )

@pytest.fixture
def backtest_engine():
    """Create a backtest engine instance."""
    return BacktestEngine()

@pytest.mark.asyncio
async def test_market_data_processing_performance(trading_engine, mock_market_data_store):
    """Test market data processing performance."""
    # Generate test market data
    market_data = pd.DataFrame({
        'open': [100 + random.uniform(-1, 1) for _ in range(1000)],
        'high': [105 + random.uniform(-1, 1) for _ in range(1000)],
        'low': [95 + random.uniform(-1, 1) for _ in range(1000)],
        'close': [102 + random.uniform(-1, 1) for _ in range(1000)],
        'volume': [1000 + random.uniform(-100, 100) for _ in range(1000)]
    })
    
    # Measure processing time
    start_time = time.time()
    for _ in range(1000):
        await trading_engine._process_market_data()
    end_time = time.time()
    
    # Calculate performance metrics
    processing_time = end_time - start_time
    avg_processing_time = processing_time / 1000
    
    # Assert performance requirements
    assert avg_processing_time < 0.001  # Less than 1ms per data point
    assert processing_time < 1.0  # Total processing time less than 1 second

@pytest.mark.asyncio
async def test_trading_engine_performance(trading_engine, mock_broker, mock_risk_manager):
    """Test trading engine performance."""
    # Generate test orders
    orders = [{
        'symbol': 'BTC/USDT',
        'side': 'BUY',
        'type': 'limit',
        'price': Decimal('100.00'),
        'quantity': Decimal('1.0')
    } for _ in range(100)]
    
    # Measure order execution time
    start_time = time.time()
    for order in orders:
        await trading_engine._execute_order(order)
    end_time = time.time()
    
    # Calculate performance metrics
    execution_time = end_time - start_time
    avg_execution_time = execution_time / len(orders)
    
    # Assert performance requirements
    assert avg_execution_time < 0.01  # Less than 10ms per order
    assert execution_time < 1.0  # Total execution time less than 1 second

@pytest.mark.asyncio
async def test_risk_management_performance(trading_engine, mock_risk_manager):
    """Test risk management performance."""
    # Generate test positions
    positions = [{
        'symbol': 'BTC/USDT',
        'amount': Decimal('1.0'),
        'entry_price': Decimal('100.00'),
        'current_price': Decimal('105.00')
    } for _ in range(100)]
    
    # Measure risk check time
    start_time = time.time()
    for position in positions:
        await trading_engine._check_risk_limits()
    end_time = time.time()
    
    # Calculate performance metrics
    check_time = end_time - start_time
    avg_check_time = check_time / len(positions)
    
    # Assert performance requirements
    assert avg_check_time < 0.001  # Less than 1ms per check
    assert check_time < 0.1  # Total check time less than 100ms

@pytest.mark.asyncio
async def test_performance_monitoring(trading_engine, mock_performance_monitor):
    """Test performance monitoring."""
    # Generate test metrics
    metrics = [{
        'win_rate': random.uniform(0.4, 0.6),
        'profit_factor': random.uniform(1.0, 2.0),
        'sharpe_ratio': random.uniform(0.5, 1.5)
    } for _ in range(100)]
    
    # Measure metrics update time
    start_time = time.time()
    for metric in metrics:
        await trading_engine._update_performance_metrics()
    end_time = time.time()
    
    # Calculate performance metrics
    update_time = end_time - start_time
    avg_update_time = update_time / len(metrics)
    
    # Assert performance requirements
    assert avg_update_time < 0.001  # Less than 1ms per update
    assert update_time < 0.1  # Total update time less than 100ms

@pytest.mark.asyncio
async def test_notification_system_performance(trading_engine, mock_notification_manager):
    """Test notification system performance."""
    # Generate test notifications
    notifications = [
        ('Test notification', 'INFO') for _ in range(100)
    ]
    
    # Measure notification sending time
    start_time = time.time()
    for message, level in notifications:
        await trading_engine._send_notification(message, level)
    end_time = time.time()
    
    # Calculate performance metrics
    sending_time = end_time - start_time
    avg_sending_time = sending_time / len(notifications)
    
    # Assert performance requirements
    assert avg_sending_time < 0.001  # Less than 1ms per notification
    assert sending_time < 0.1  # Total sending time less than 100ms

@pytest.mark.asyncio
async def test_websocket_performance(trading_engine, mock_websocket_manager):
    """Test WebSocket performance."""
    # Generate test messages
    messages = [{
        'type': 'trade',
        'data': {
            'symbol': 'BTC/USDT',
            'price': 100.0,
            'amount': 1.0
        }
    } for _ in range(100)]
    
    # Measure message processing time
    start_time = time.time()
    for message in messages:
        await trading_engine._handle_websocket_message(message)
    end_time = time.time()
    
    # Calculate performance metrics
    processing_time = end_time - start_time
    avg_processing_time = processing_time / len(messages)
    
    # Assert performance requirements
    assert avg_processing_time < 0.001  # Less than 1ms per message
    assert processing_time < 0.1  # Total processing time less than 100ms

def test_memory_usage(trading_engine):
    """Test memory usage under load."""
    # Get initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Generate and process large amount of data
    for _ in range(1000):
        trading_engine._process_market_data()
        trading_engine._update_performance_metrics()
    
    # Get final memory usage
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Assert memory requirements
    assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase

def test_cpu_usage(trading_engine):
    """Test CPU usage under load."""
    # Get initial CPU usage
    process = psutil.Process()
    initial_cpu = process.cpu_percent()
    
    # Generate load
    for _ in range(1000):
        trading_engine._process_market_data()
        trading_engine._update_performance_metrics()
    
    # Get final CPU usage
    final_cpu = process.cpu_percent()
    cpu_increase = final_cpu - initial_cpu
    
    # Assert CPU requirements
    assert cpu_increase < 50  # Less than 50% increase

@pytest.mark.asyncio
async def test_concurrent_operations_performance(trading_engine):
    """Test concurrent operations performance."""
    async def simulate_market_data():
        for _ in range(100):
            await trading_engine._process_market_data()
            await asyncio.sleep(0.001)
    
    async def simulate_trading():
        for _ in range(100):
            await trading_engine._execute_order({
                'symbol': 'BTC/USDT',
                'side': 'BUY',
                'type': 'limit',
                'price': Decimal('100.00'),
                'quantity': Decimal('1.0')
            })
            await asyncio.sleep(0.001)
    
    async def simulate_monitoring():
        for _ in range(100):
            await trading_engine._update_performance_metrics()
            await asyncio.sleep(0.001)
    
    # Run concurrent operations
    start_time = time.time()
    await asyncio.gather(
        simulate_market_data(),
        simulate_trading(),
        simulate_monitoring()
    )
    end_time = time.time()
    
    # Calculate performance metrics
    execution_time = end_time - start_time
    
    # Assert performance requirements
    assert execution_time < 2.0  # Total execution time less than 2 seconds

def test_error_handling_performance(trading_engine):
    """Test error handling performance."""
    # Generate test errors
    errors = [Exception(f'Test error {i}') for i in range(100)]
    
    # Measure error handling time
    start_time = time.time()
    for error in errors:
        trading_engine._handle_error(error)
    end_time = time.time()
    
    # Calculate performance metrics
    handling_time = end_time - start_time
    avg_handling_time = handling_time / len(errors)
    
    # Assert performance requirements
    assert avg_handling_time < 0.001  # Less than 1ms per error
    assert handling_time < 0.1  # Total handling time less than 100ms

def test_database_operation_performance(app_context, test_user):
    """Test database operation performance."""
    # Create a strategy first
    strategy = Strategy(
        name='Performance Test Strategy',
        description='Test strategy for performance testing',
        parameters={},
        user_id=test_user.id
    )
    db.session.add(strategy)
    db.session.commit()
    
    # Measure bulk insert performance
    start_time = time.time()
    trades = []
    for i in range(1000):
        trade = Trade(
            strategy_id=strategy.id,
            symbol='BTC/USDT',
            side='buy',  # Use 'side' instead of 'type'
            price=50000.0 + random.uniform(-100, 100),
            amount=0.1,
            total=5000.0,
            fee=5.0,
            profit=100.0,
            timestamp=datetime.now(timezone.utc)
        )
        trades.append(trade)
    
    # Use bulk insert
    db.session.bulk_save_objects(trades)
    db.session.commit()
    end_time = time.time()
    
    # Calculate performance metrics
    insert_time = end_time - start_time
    assert insert_time < 2.0  # Bulk insert should complete within 2 seconds
    
    # Measure query performance with optimized queries
    start_time = time.time()
    
    # Test indexed queries
    trades = Trade.query.filter(
        Trade.strategy_id == strategy.id,
        Trade.timestamp >= datetime.now(timezone.utc) - timedelta(hours=1)
    ).order_by(Trade.timestamp.desc()).all()
    
    end_time = time.time()
    query_time = end_time - start_time
    assert query_time < 0.1  # Query should complete within 100ms
    
    # Test aggregation queries
    start_time = time.time()
    result = db.session.query(
        Trade.symbol,
        func.count(Trade.id).label('trade_count'),
        func.sum(Trade.profit).label('total_profit')
    ).filter(
        Trade.strategy_id == strategy.id
    ).group_by(Trade.symbol).all()
    
    end_time = time.time()
    agg_time = end_time - start_time
    assert agg_time < 0.1  # Aggregation should complete within 100ms

def test_backtest_performance(app_context, test_user, backtest_engine):
    """Test backtest performance."""
    # Create strategy
    strategy = Strategy(
        name='Backtest Performance Strategy',
        description='Test strategy for backtest performance',
        parameters={},
        user_id=test_user.id
    )
    db.session.add(strategy)
    db.session.commit()
    
    # Generate test data
    backtest_data = {
        'start_date': (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
        'end_date': datetime.now(timezone.utc).isoformat(),
        'initial_balance': 10000.0
    }
    
    # Measure backtest execution time
    start_time = time.time()
    results = backtest_engine.run_backtest(backtest_data, strategy, {})
    end_time = time.time()
    
    # Calculate performance metrics
    execution_time = end_time - start_time
    assert execution_time < 2.0  # Backtest should complete within 2 seconds
    assert hasattr(results, 'total_return')
    assert hasattr(results, 'sharpe_ratio')
    assert hasattr(results, 'max_drawdown')
    assert hasattr(results, 'win_rate')
    assert hasattr(results, 'trades')
    assert hasattr(results, 'equity_curve')

def test_websocket_performance(app_context, test_user):
    """Test WebSocket performance."""
    client = app.test_client()
    
    # Measure connection time
    start_time = time.time()
    response = client.get('/ws')
    end_time = time.time()
    
    connection_time = end_time - start_time
    assert connection_time < 0.1  # Connection should establish within 100ms
    
    # Measure message broadcast performance
    start_time = time.time()
    for _ in range(100):
        client.post('/api/broadcast', json={
            'channel': 'market_data',
            'data': {
                'symbol': 'BTC/USDT',
                'price': 50000.0
            }
        })
    end_time = time.time()
    
    broadcast_time = end_time - start_time
    assert broadcast_time < 1.0  # 100 broadcasts should complete within 1 second

def test_data_persistence_performance(app_context, test_user):
    """Test data persistence performance."""
    # Measure file I/O performance
    start_time = time.time()
    data = {
        'trades': [{
            'id': i,
            'symbol': 'BTC/USDT',
            'price': 50000.0 + random.uniform(-100, 100),
            'amount': 0.1,
            'timestamp': datetime.now(timezone.utc).isoformat()
        } for i in range(1000)]
    }
    
    # Write to file
    with open('test_trades.json', 'w') as f:
        json.dump(data, f)
    
    # Read from file
    with open('test_trades.json', 'r') as f:
        loaded_data = json.load(f)
    
    end_time = time.time()
    
    # Calculate performance metrics
    io_time = end_time - start_time
    assert io_time < 1.0  # File I/O should complete within 1 second
    assert len(loaded_data['trades']) == 1000
    
    # Clean up
    os.remove('test_trades.json')

@pytest.mark.asyncio
async def test_concurrent_operations_performance(app_context, test_user, trading_engine):
    """Test performance under concurrent operations."""
    async def simulate_concurrent_operations():
        tasks = []
        for _ in range(10):
            tasks.append(asyncio.create_task(simulate_market_data_stream()))
        await asyncio.gather(*tasks)
    
    async def simulate_market_data_stream():
        for _ in range(100):
            market_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'symbol': 'BTC/USDT',
                'price': 50000.0 + random.uniform(-100, 100),
                'volume': 100.0 + random.uniform(-10, 10)
            }
            await trading_engine.process_market_data(market_data)
            await asyncio.sleep(0.001)
    
    # Measure concurrent operation performance
    start_time = time.time()
    await simulate_concurrent_operations()
    end_time = time.time()
    
    execution_time = end_time - start_time
    assert execution_time < 2.0  # Concurrent operations should complete within 2 seconds

def test_api_response_time(app_context, test_user):
    """Test API response time."""
    client = app.test_client()
    
    # Test various API endpoints
    endpoints = [
        ('/api/status', 'GET'),
        ('/api/trades', 'GET'),
        ('/api/balance', 'GET'),
        ('/api/strategies', 'GET')
    ]
    
    for endpoint, method in endpoints:
        start_time = time.time()
        if method == 'GET':
            response = client.get(endpoint, follow_redirects=True)
        else:
            response = client.post(endpoint, follow_redirects=True)
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 0.1  # API response should be within 100ms
        assert response.status_code == 200

def test_data_persistence_performance(app_context, test_user):
    """Test data persistence performance."""
    # Measure file I/O performance
    start_time = time.time()
    data = {
        'trades': [{
            'id': i,
            'symbol': 'BTC/USDT',
            'price': 50000.0 + random.uniform(-100, 100),
            'amount': 0.1,
            'timestamp': datetime.now(timezone.utc).isoformat()
        } for i in range(1000)]
    }
    
    # Write to file
    with open('test_trades.json', 'w') as f:
        json.dump(data, f)
    
    # Read from file
    with open('test_trades.json', 'r') as f:
        loaded_data = json.load(f)
    
    end_time = time.time()
    
    # Calculate performance metrics
    io_time = end_time - start_time
    assert io_time < 1.0  # File I/O should complete within 1 second
    assert len(loaded_data['trades']) == 1000
    
    # Clean up
    os.remove('test_trades.json')

def test_error_handling_performance(app_context, test_user, trading_engine):
    """Test error handling performance."""
    # Measure error handling performance
    start_time = time.time()
    for _ in range(100):
        try:
            asyncio.run(trading_engine.process_market_data(None))
        except Exception:
            pass
    
    end_time = time.time()
    
    error_handling_time = end_time - start_time
    assert error_handling_time < 0.1  # Error handling should be within 100ms 