import pytest
import time
import asyncio
import psutil
import os
from app import app, db, User, Strategy, Trade
from trading_engine import TradingEngine
from backtest_engine import BacktestEngine
from datetime import datetime, timedelta, timezone
import random
import json
from sqlalchemy import func
from werkzeug.security import generate_password_hash

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
def trading_engine():
    """Create a trading engine instance."""
    class MockExchange:
        def __init__(self):
            self.markets = {'BTC/USDT': {}}
            
        async def load_markets(self):
            pass
            
        async def create_order(self, symbol, type, side, amount, price):
            return {
                'id': 'test_order_id',
                'symbol': symbol,
                'type': type,
                'side': side,
                'amount': amount,
                'price': price,
                'status': 'open'
            }
            
        async def fetch_ohlcv(self, symbol, timeframe, limit):
            return [
                [int(time.time() * 1000), 50000.0, 50100.0, 49900.0, 50050.0, 100.0]
                for _ in range(limit)
            ]
            
        async def fetch_positions(self):
            return [
                {
                    'symbol': 'BTC/USDT',
                    'contracts': 0.0,
                    'notional': 0.0,
                    'leverage': 1.0,
                    'unrealizedPnl': 0.0,
                    'marginType': 'cross',
                    'liquidationPrice': None,
                    'markPrice': 50000.0,
                    'entryPrice': 0.0,
                    'timestamp': int(time.time() * 1000)
                }
            ]
            
        async def fetch_balance(self):
            return {
                'BTC': {'free': 1.0, 'used': 0.0, 'total': 1.0},
                'USDT': {'free': 10000.0, 'used': 0.0, 'total': 10000.0}
            }
    
    return TradingEngine(exchange=MockExchange())

@pytest.fixture
def backtest_engine():
    """Create a backtest engine instance."""
    return BacktestEngine()

@pytest.mark.asyncio
async def test_market_data_processing_performance(app_context, test_user, trading_engine):
    """Test market data processing performance."""
    # Create strategy
    strategy = Strategy(
        name='Performance Test Strategy',
        description='Test strategy for performance testing',
        parameters={},
        user_id=test_user.id
    )
    db.session.add(strategy)
    db.session.commit()
    
    # Initialize trading engine
    await trading_engine.initialize()
    
    # Generate test market data
    market_data = [{
        'timestamp': (datetime.now(timezone.utc) + timedelta(seconds=i)).isoformat(),
        'symbol': 'BTC/USDT',
        'price': 50000.0 + random.uniform(-100, 100),
        'volume': 100.0 + random.uniform(-10, 10)
    } for i in range(1000)]
    
    # Measure processing time
    start_time = time.time()
    for data in market_data:
        await trading_engine.process_market_data(data)
    end_time = time.time()
    
    # Calculate performance metrics
    processing_time = end_time - start_time
    avg_processing_time = processing_time / len(market_data)
    
    # Assert performance requirements
    assert avg_processing_time < 0.001  # Less than 1ms per data point
    assert processing_time < 1.0  # Total processing time less than 1 second

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

def test_memory_usage(app_context, test_user, trading_engine):
    """Test memory usage under load."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Create and process large dataset
    market_data = [{
        'timestamp': (datetime.now(timezone.utc) + timedelta(seconds=i)).isoformat(),
        'symbol': 'BTC/USDT',
        'price': 50000.0 + random.uniform(-100, 100),
        'volume': 100.0 + random.uniform(-10, 10)
    } for i in range(10000)]
    
    for data in market_data:
        asyncio.run(trading_engine.process_market_data(data))
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Assert memory usage requirements
    assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase

def test_cpu_usage(app_context, test_user, trading_engine):
    """Test CPU usage under load."""
    process = psutil.Process(os.getpid())
    
    # Measure CPU usage during intensive operations
    cpu_percentages = []
    for _ in range(10):
        start_time = time.time()
        while time.time() - start_time < 0.1:  # Measure for 100ms
            asyncio.run(trading_engine.process_market_data({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'symbol': 'BTC/USDT',
                'price': 50000.0 + random.uniform(-100, 100),
                'volume': 100.0 + random.uniform(-10, 10)
            }))
        cpu_percentages.append(process.cpu_percent())
    
    avg_cpu_usage = sum(cpu_percentages) / len(cpu_percentages)
    assert avg_cpu_usage < 50.0  # Average CPU usage should be less than 50%

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