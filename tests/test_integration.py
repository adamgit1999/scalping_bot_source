import pytest
from app import app, db, User, Strategy, Trade, Backtest
from trading_engine import TradingEngine
from backtest_engine import BacktestEngine
from notification_system import NotificationSystem
from websocket_server import WebSocketServer
from datetime import datetime, timedelta
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock

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
        password_hash='hashed_password'
    )
    db.session.add(user)
    db.session.commit()
    return user

@pytest.fixture
def trading_engine():
    """Create a trading engine instance."""
    return TradingEngine()

@pytest.fixture
def backtest_engine():
    """Create a backtest engine instance."""
    return BacktestEngine()

@pytest.fixture
def notification_system():
    """Create a notification system instance."""
    return NotificationSystem()

@pytest.fixture
def websocket_server():
    """Create a WebSocket server instance."""
    return WebSocketServer()

def test_complete_trading_flow(app_context, test_user, trading_engine, notification_system, websocket_server):
    """Test complete trading flow from strategy creation to trade execution."""
    # Create strategy
    strategy = Strategy(
        name='Test Strategy',
        description='Integration test strategy',
        parameters={
            'sma_short': 10,
            'sma_long': 20,
            'rsi_period': 14
        },
        user_id=test_user.id
    )
    db.session.add(strategy)
    db.session.commit()
    
    # Initialize trading engine
    trading_engine.initialize(strategy)
    
    # Mock market data
    market_data = {
        'timestamp': datetime.now(datetime.UTC).isoformat(),
        'symbol': 'BTC/USDT',
        'price': 50000.0,
        'volume': 100.0
    }
    
    # Process market data
    signals = trading_engine.process_market_data(market_data)
    
    # Execute trades based on signals
    for signal in signals:
        trade = trading_engine.execute_trade(signal)
        if trade:
            # Verify trade creation
            assert trade.strategy_id == strategy.id
            assert trade.symbol == market_data['symbol']
            assert trade.price > 0
            assert trade.amount > 0
            
            # Verify notification
            notification = notification_system.send_trade_notification(test_user.email, trade)
            assert notification is True
            
            # Verify WebSocket broadcast
            websocket_server.broadcast_trade(trade)
    
    # Verify strategy performance
    performance = trading_engine.get_performance()
    assert 'total_trades' in performance
    assert 'win_rate' in performance
    assert 'total_profit' in performance

def test_backtest_to_live_transition(app_context, test_user, backtest_engine, trading_engine):
    """Test transitioning from backtest to live trading."""
    # Create strategy
    strategy = Strategy(
        name='Test Strategy',
        description='Backtest to live strategy',
        parameters={
            'sma_short': 10,
            'sma_long': 20
        },
        user_id=test_user.id
    )
    db.session.add(strategy)
    db.session.commit()
    
    # Run backtest
    backtest_data = {
        'start_date': (datetime.now(datetime.UTC) - timedelta(days=30)).isoformat(),
        'end_date': datetime.now(datetime.UTC).isoformat(),
        'initial_balance': 10000.0
    }
    
    backtest_results = backtest_engine.run_backtest(backtest_data, strategy)
    assert backtest_results['total_return'] > 0
    
    # Create backtest record
    backtest = Backtest(
        strategy_id=strategy.id,
        start_date=datetime.fromisoformat(backtest_data['start_date']),
        end_date=datetime.fromisoformat(backtest_data['end_date']),
        initial_balance=backtest_data['initial_balance'],
        final_balance=backtest_results['final_balance'],
        total_trades=backtest_results['total_trades'],
        win_rate=backtest_results['win_rate'],
        profit_factor=backtest_results['profit_factor'],
        max_drawdown=backtest_results['max_drawdown']
    )
    db.session.add(backtest)
    db.session.commit()
    
    # Initialize live trading
    trading_engine.initialize(strategy)
    
    # Verify strategy parameters match
    assert trading_engine.strategy.parameters == strategy.parameters
    
    # Verify initial state
    assert trading_engine.is_running is True
    assert trading_engine.initial_balance == backtest_data['initial_balance']

def test_multi_strategy_execution(app_context, test_user, trading_engine):
    """Test execution of multiple strategies simultaneously."""
    # Create multiple strategies
    strategies = []
    for i in range(3):
        strategy = Strategy(
            name=f'Strategy {i+1}',
            description=f'Test strategy {i+1}',
            parameters={
                'sma_short': 10 + i,
                'sma_long': 20 + i
            },
            user_id=test_user.id
        )
        db.session.add(strategy)
        strategies.append(strategy)
    db.session.commit()
    
    # Initialize trading engine for each strategy
    engines = []
    for strategy in strategies:
        engine = TradingEngine()
        engine.initialize(strategy)
        engines.append(engine)
    
    # Process market data for all strategies
    market_data = {
        'timestamp': datetime.now(datetime.UTC).isoformat(),
        'symbol': 'BTC/USDT',
        'price': 50000.0,
        'volume': 100.0
    }
    
    for engine in engines:
        signals = engine.process_market_data(market_data)
        for signal in signals:
            trade = engine.execute_trade(signal)
            if trade:
                assert trade.strategy_id in [s.id for s in strategies]

def test_error_handling_and_recovery(app_context, test_user, trading_engine, notification_system):
    """Test error handling and recovery mechanisms."""
    # Create strategy
    strategy = Strategy(
        name='Test Strategy',
        description='Error handling test',
        parameters={},
        user_id=test_user.id
    )
    db.session.add(strategy)
    db.session.commit()
    
    # Initialize trading engine
    trading_engine.initialize(strategy)
    
    # Simulate various error conditions
    error_scenarios = [
        {'type': 'market_data_error', 'data': None},
        {'type': 'execution_error', 'data': {'price': -1}},
        {'type': 'connection_error', 'data': {'status': 'disconnected'}}
    ]
    
    for scenario in error_scenarios:
        # Process error scenario
        with pytest.raises(Exception):
            if scenario['type'] == 'market_data_error':
                trading_engine.process_market_data(scenario['data'])
            elif scenario['type'] == 'execution_error':
                trading_engine.execute_trade(scenario['data'])
            elif scenario['type'] == 'connection_error':
                trading_engine.handle_connection_error(scenario['data'])
        
        # Verify error notification
        notification = notification_system.send_error_notification(
            test_user.email,
            f"Error in {scenario['type']}"
        )
        assert notification is True
        
        # Verify recovery
        assert trading_engine.is_running is True
        assert trading_engine.error_count < trading_engine.max_errors

def test_data_persistence_and_recovery(app_context, test_user, trading_engine):
    """Test data persistence and recovery mechanisms."""
    # Create strategy
    strategy = Strategy(
        name='Test Strategy',
        description='Data persistence test',
        parameters={},
        user_id=test_user.id
    )
    db.session.add(strategy)
    db.session.commit()
    
    # Initialize trading engine
    trading_engine.initialize(strategy)
    
    # Execute some trades
    for _ in range(5):
        trade = Trade(
            strategy_id=strategy.id,
            symbol='BTC/USDT',
            side='buy',
            price=50000.0,
            amount=0.1,
            total=5000.0,
            fee=5.0,
            profit=100.0,
            timestamp=datetime.now(datetime.UTC)
        )
        db.session.add(trade)
    db.session.commit()
    
    # Simulate application restart
    trading_engine.stop()
    trading_engine.initialize(strategy)
    
    # Verify data recovery
    assert len(trading_engine.get_trades()) == 5
    assert trading_engine.get_performance()['total_trades'] == 5

def test_concurrent_operations(app_context, test_user, trading_engine, websocket_server):
    """Test concurrent operations handling."""
    # Create strategy
    strategy = Strategy(
        name='Test Strategy',
        description='Concurrent operations test',
        parameters={},
        user_id=test_user.id
    )
    db.session.add(strategy)
    db.session.commit()
    
    # Initialize trading engine
    trading_engine.initialize(strategy)
    
    # Simulate concurrent market data updates
    async def simulate_market_updates():
        for _ in range(10):
            market_data = {
                'timestamp': datetime.now(datetime.UTC).isoformat(),
                'symbol': 'BTC/USDT',
                'price': 50000.0 + (0.1 * _),
                'volume': 100.0
            }
            await websocket_server.broadcast_price(market_data)
            await asyncio.sleep(0.1)
    
    # Run concurrent operations
    asyncio.run(simulate_market_updates())
    
    # Verify data consistency
    trades = trading_engine.get_trades()
    assert len(trades) > 0
    assert all(trade.price > 0 for trade in trades)
    assert all(trade.amount > 0 for trade in trades)

def test_system_health_monitoring(app_context, test_user, trading_engine, notification_system):
    """Test system health monitoring and alerts."""
    # Create strategy
    strategy = Strategy(
        name='Test Strategy',
        description='Health monitoring test',
        parameters={},
        user_id=test_user.id
    )
    db.session.add(strategy)
    db.session.commit()
    
    # Initialize trading engine
    trading_engine.initialize(strategy)
    
    # Simulate various health scenarios
    health_scenarios = [
        {'type': 'high_memory_usage', 'value': 90},
        {'type': 'high_cpu_usage', 'value': 95},
        {'type': 'high_latency', 'value': 1000}
    ]
    
    for scenario in health_scenarios:
        # Process health check
        health_status = trading_engine.check_system_health(scenario)
        
        # Verify health status
        assert 'status' in health_status
        assert 'metrics' in health_status
        
        # Verify alert if necessary
        if health_status['status'] == 'warning':
            notification = notification_system.send_system_alert(
                test_user.email,
                f"System {scenario['type']} alert"
            )
            assert notification is True 