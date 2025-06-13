import pytest
import json
import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock

from src.app import app, db
from src.models import User, Strategy, Trade, Notification, Webhook, Backtest
from src.trading_engine import TradingEngine
from src.backtest_engine import BacktestEngine
from src.broker import BrokerInterface
from src.risk_manager import RiskManager
from src.data_processing import MarketDataStore
from src.performance_monitoring import PerformanceMonitor
from src.notification_system import NotificationSystem
from src.websocket_server import WebSocketManager
from src.indicators import TechnicalIndicators
from src.config import Config

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
def mock_broker():
    """Create a mock broker for testing."""
    broker = Mock(spec=BrokerInterface)
    broker.get_balance = AsyncMock(return_value=Decimal('10000.00'))
    broker.get_position = AsyncMock(return_value=None)
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
    store.get_latest_data = AsyncMock(return_value={
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
    monitor.update_metrics = AsyncMock(return_value={
        'win_rate': 0.6,
        'profit_factor': 1.5,
        'sharpe_ratio': 1.2
    })
    return monitor

@pytest.fixture
def mock_notification_manager():
    """Create a mock notification manager for testing."""
    manager = Mock(spec=NotificationSystem)
    manager.send_notification = AsyncMock(return_value=True)
    return manager

@pytest.fixture
def mock_websocket_manager():
    """Create a mock websocket manager for testing."""
    manager = Mock(spec=WebSocketManager)
    manager.broadcast = AsyncMock(return_value=True)
    return manager

@pytest.fixture
def mock_indicators():
    """Create a mock technical indicators instance."""
    indicators = Mock(spec=TechnicalIndicators)
    indicators.calculate_sma = AsyncMock(return_value=[100, 101, 102])
    indicators.calculate_rsi = AsyncMock(return_value=[50, 51, 52])
    indicators.calculate_macd = AsyncMock(return_value={
        'macd': [1, 2, 3],
        'signal': [0.5, 1, 1.5],
        'histogram': [0.5, 1, 1.5]
    })
    return indicators

@pytest.fixture
def mock_config():
    """Create a mock configuration instance."""
    config = Mock(spec=Config)
    config.get = Mock(return_value={
        'risk_management': {
            'max_position_size': 1.0,
            'max_drawdown': 0.1,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.05
        },
        'trading': {
            'default_leverage': 1,
            'max_leverage': 5,
            'min_volume': 100
        },
        'performance': {
            'metrics_update_interval': 60,
            'max_drawdown_threshold': 0.15
        }
    })
    return config

@pytest.fixture
def trading_engine(mock_broker, mock_risk_manager, mock_market_data_store,
                  mock_performance_monitor, mock_notification_manager,
                  mock_websocket_manager, mock_indicators, mock_config):
    """Create a trading engine instance with updated dependencies."""
    return TradingEngine(
        broker=mock_broker,
        risk_manager=mock_risk_manager,
        market_data_store=mock_market_data_store,
        performance_monitor=mock_performance_monitor,
        notification_manager=mock_notification_manager,
        websocket_manager=mock_websocket_manager,
        indicators=mock_indicators,
        config=mock_config
    )

@pytest.fixture
def backtest_engine():
    """Create a backtest engine instance."""
    return BacktestEngine()

@pytest.mark.asyncio
async def test_complete_trading_flow(app_context, test_user, trading_engine):
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
    await trading_engine.initialize(strategy)
    
    # Process market data
    market_data = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'symbol': 'BTC/USDT',
        'price': 50000.0,
        'volume': 100.0
    }
    
    signals = await trading_engine.process_market_data(market_data)
    
    # Execute trades based on signals
    for signal in signals:
        trade = await trading_engine.execute_trade(signal)
        if trade:
            # Verify trade creation
            assert trade.strategy_id == strategy.id
            assert trade.symbol == market_data['symbol']
            assert trade.price > 0
            assert trade.amount > 0
            
            # Verify risk management
            assert await trading_engine.risk_manager.check_order(trade)
            
            # Verify performance monitoring
            metrics = await trading_engine.performance_monitor.update_metrics()
            assert 'win_rate' in metrics
            assert 'profit_factor' in metrics
            
            # Verify notification
            assert await trading_engine.notification_manager.send_notification(
                f"New trade executed: {trade.symbol}",
                'INFO'
            )
            
            # Verify WebSocket broadcast
            assert await trading_engine.websocket_manager.broadcast({
                'type': 'trade',
                'data': trade.to_dict()
            })

@pytest.mark.asyncio
async def test_backtest_to_live_transition(app_context, test_user, backtest_engine, trading_engine):
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
        'start_date': (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
        'end_date': datetime.now(timezone.utc).isoformat(),
        'initial_balance': 10000.0
    }
    
    backtest_results = await backtest_engine.run_backtest(backtest_data, strategy)
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
    await trading_engine.initialize(strategy)
    
    # Verify strategy parameters match
    assert trading_engine.strategy.parameters == strategy.parameters
    
    # Verify initial state
    assert trading_engine.is_running is True
    assert trading_engine.initial_balance == backtest_data['initial_balance']

@pytest.mark.asyncio
async def test_multi_strategy_execution(app_context, test_user, trading_engine):
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
        engine = TradingEngine(
            broker=trading_engine.broker,
            risk_manager=trading_engine.risk_manager,
            market_data_store=trading_engine.market_data_store,
            performance_monitor=trading_engine.performance_monitor,
            notification_manager=trading_engine.notification_manager,
            websocket_manager=trading_engine.websocket_manager,
            config=trading_engine.config
        )
        await engine.initialize(strategy)
        engines.append(engine)
    
    # Process market data for all strategies
    market_data = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'symbol': 'BTC/USDT',
        'price': 50000.0,
        'volume': 100.0
    }
    
    for engine in engines:
        signals = await engine.process_market_data(market_data)
        for signal in signals:
            trade = await engine.execute_trade(signal)
            if trade:
                assert trade.strategy_id in [s.id for s in strategies]
                assert await engine.risk_manager.check_order(trade)
                assert await engine.performance_monitor.update_metrics()

@pytest.mark.asyncio
async def test_error_handling_and_recovery(app_context, test_user, trading_engine):
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
    await trading_engine.initialize(strategy)
    
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
                await trading_engine.process_market_data(scenario['data'])
            elif scenario['type'] == 'execution_error':
                await trading_engine.execute_trade(scenario['data'])
            elif scenario['type'] == 'connection_error':
                await trading_engine.handle_connection_error(scenario['data'])
        
        # Verify error notification
        assert await trading_engine.notification_manager.send_notification(
            f"Error in {scenario['type']}",
            'ERROR'
        )
        
        # Verify recovery
        assert trading_engine.is_running is True
        assert trading_engine.error_count < trading_engine.max_errors

@pytest.mark.asyncio
async def test_concurrent_operations(app_context, test_user, trading_engine):
    """Test concurrent operations."""
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
    await trading_engine.initialize(strategy)
    
    async def simulate_market_updates():
        for _ in range(10):
            market_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'symbol': 'BTC/USDT',
                'price': 50000.0 + random.uniform(-100, 100),
                'volume': 100.0 + random.uniform(-10, 10)
            }
            await trading_engine.process_market_data(market_data)
            await asyncio.sleep(0.1)
    
    async def simulate_trades():
        for _ in range(10):
            signal = {
                'symbol': 'BTC/USDT',
                'side': 'BUY',
                'type': 'limit',
                'price': 50000.0,
                'quantity': 1.0
            }
            await trading_engine.execute_trade(signal)
            await asyncio.sleep(0.1)
    
    # Run concurrent operations
    await asyncio.gather(
        simulate_market_updates(),
        simulate_trades()
    )
    
    # Verify system state
    assert trading_engine.is_running is True
    assert trading_engine.error_count == 0

@pytest.mark.asyncio
async def test_system_health_monitoring(app_context, test_user, trading_engine):
    """Test system health monitoring."""
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
    await trading_engine.initialize(strategy)
    
    # Monitor system health
    health_metrics = await trading_engine.performance_monitor.get_health_metrics()
    assert 'cpu_usage' in health_metrics
    assert 'memory_usage' in health_metrics
    assert 'error_rate' in health_metrics
    assert 'latency' in health_metrics
    
    # Verify health notifications
    if health_metrics['error_rate'] > 0.1:
        assert await trading_engine.notification_manager.send_notification(
            "High error rate detected",
            'WARNING'
        )
    
    if health_metrics['latency'] > 1000:
        assert await trading_engine.notification_manager.send_notification(
            "High latency detected",
            'WARNING'
        )

@pytest.mark.asyncio
async def test_technical_analysis_integration(app_context, test_user, trading_engine):
    """Test integration of technical analysis with trading decisions."""
    strategy = Strategy(
        name='Technical Analysis Strategy',
        description='Strategy using multiple technical indicators',
        parameters={
            'sma_short': 10,
            'sma_long': 20,
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        },
        user_id=test_user.id
    )
    db.session.add(strategy)
    db.session.commit()
    
    await trading_engine.initialize(strategy)
    
    market_data = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'symbol': 'BTC/USDT',
        'price': 50000.0,
        'volume': 100.0,
        'candles': {
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        }
    }
    
    signals = await trading_engine.process_market_data(market_data)
    
    # Verify technical indicators were calculated
    assert await trading_engine.indicators.calculate_sma.called
    assert await trading_engine.indicators.calculate_rsi.called
    assert await trading_engine.indicators.calculate_macd.called
    
    # Verify trading decisions based on indicators
    if signals:
        for signal in signals:
            assert signal['type'] in ['BUY', 'SELL']
            assert 'price' in signal
            assert 'quantity' in signal
            assert 'reason' in signal

@pytest.mark.asyncio
async def test_risk_management_integration(app_context, test_user, trading_engine):
    """Test integration of risk management with trading operations."""
    strategy = Strategy(
        name='Risk Managed Strategy',
        description='Strategy with comprehensive risk management',
        parameters={
            'max_position_size': 1.0,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.05
        },
        user_id=test_user.id
    )
    db.session.add(strategy)
    db.session.commit()
    
    await trading_engine.initialize(strategy)
    
    # Test position sizing
    position_size = await trading_engine.risk_manager.calculate_position_size(
        price=50000.0,
        balance=10000.0,
        risk_per_trade=0.01
    )
    assert position_size > 0
    assert position_size <= 1.0  # Max position size from config
    
    # Test risk limits
    assert await trading_engine.risk_manager.check_risk_limits(
        current_drawdown=0.05,
        max_drawdown=0.1
    )
    
    # Test order validation
    order = {
        'symbol': 'BTC/USDT',
        'side': 'BUY',
        'type': 'limit',
        'price': 50000.0,
        'quantity': 0.5
    }
    is_valid, message = await trading_engine.risk_manager.check_order(order)
    assert is_valid
    assert message == "OK" 