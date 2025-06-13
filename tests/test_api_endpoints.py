import pytest
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal

from src.models import User, Trade, Notification, Webhook, db
from src.app import app
from src.broker import BrokerInterface
from src.trading_engine import TradingEngine
from src.risk_manager import RiskManager
from src.data_processing import MarketDataStore
from src.performance_monitoring import PerformanceMonitor
from src.notification_system import NotificationSystem
from src.websocket_server import WebSocketServer
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
def mock_trading_engine():
    """Create a mock trading engine for testing."""
    engine = Mock(spec=TradingEngine)
    engine.start = AsyncMock()
    engine.stop = AsyncMock()
    engine.get_performance_metrics = AsyncMock(return_value={
        'win_rate': 0.6,
        'profit_factor': 1.5,
        'sharpe_ratio': 1.2
    })
    return engine

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
def auth_headers(test_user):
    """Create authentication headers."""
    return {
        'Authorization': f'Bearer {test_user.generate_token()}',
        'Content-Type': 'application/json'
    }

@pytest.fixture
def mock_websocket_server():
    """Create a mock websocket server for testing."""
    server = Mock(spec=WebSocketServer)
    server.broadcast_trade = AsyncMock()
    server.broadcast_price = AsyncMock()
    server.broadcast_performance = AsyncMock()
    server.broadcast_bot_status = AsyncMock()
    return server

@pytest.fixture
def mock_config():
    """Create a mock configuration instance."""
    config = Mock(spec=Config)
    config.get = Mock(return_value={
        'api': {
            'rate_limit': 100,
            'timeout': 30,
            'max_retries': 3
        },
        'risk_management': {
            'max_position_size': 1.0,
            'max_drawdown': 0.1,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.05
        }
    })
    return config

def test_user_endpoints(app_context, test_user, auth_headers):
    """Test user-related API endpoints."""
    client = app.test_client()
    
    # Test user registration
    response = client.post('/api/register', json={
        'username': 'newuser',
        'email': 'new@example.com',
        'password': 'password123'
    })
    assert response.status_code == 201
    assert 'user_id' in response.json
    
    # Test user login
    response = client.post('/api/login', json={
        'username': 'newuser',
        'password': 'password123'
    })
    assert response.status_code == 200
    assert 'token' in response.json
    
    # Test user profile
    response = client.get('/api/user/profile', headers=auth_headers)
    assert response.status_code == 200
    assert response.json['username'] == 'testuser'
    
    # Test user update
    response = client.put('/api/user/profile', json={
        'email': 'updated@example.com'
    }, headers=auth_headers)
    assert response.status_code == 200
    assert response.json['email'] == 'updated@example.com'
    
    # Test password change
    response = client.put('/api/user/password', json={
        'current_password': 'testpassword',
        'new_password': 'newpassword123'
    }, headers=auth_headers)
    assert response.status_code == 200

def test_trading_endpoints(app_context, test_user, auth_headers, mock_trading_engine):
    """Test trading-related API endpoints."""
    client = app.test_client()
    
    # Test start trading
    response = client.post('/api/trading/start', json={
        'strategy': 'scalping',
        'symbol': 'BTC/USDT',
        'parameters': {
            'sma_short': 10,
            'sma_long': 20,
            'rsi_period': 14
        }
    }, headers=auth_headers)
    assert response.status_code == 200
    assert response.json['status'] == 'started'
    
    # Test stop trading
    response = client.post('/api/trading/stop', headers=auth_headers)
    assert response.status_code == 200
    assert response.json['status'] == 'stopped'
    
    # Test get trading status
    response = client.get('/api/trading/status', headers=auth_headers)
    assert response.status_code == 200
    assert 'status' in response.json
    
    # Test get performance metrics
    response = client.get('/api/trading/performance', headers=auth_headers)
    assert response.status_code == 200
    assert 'win_rate' in response.json
    assert 'profit_factor' in response.json
    assert 'sharpe_ratio' in response.json

def test_market_data_endpoints(app_context, test_user, auth_headers):
    """Test market data-related API endpoints."""
    client = app.test_client()
    
    # Test market data retrieval
    response = client.get('/api/market/price/BTC/USDT', headers=auth_headers)
    assert response.status_code == 200
    assert 'price' in response.json
    
    # Test historical data
    response = client.get('/api/market/history/BTC/USDT', headers=auth_headers)
    assert response.status_code == 200
    assert 'data' in response.json
    
    # Test order book
    response = client.get('/api/market/orderbook/BTC/USDT', headers=auth_headers)
    assert response.status_code == 200
    assert 'bids' in response.json
    assert 'asks' in response.json
    
    # Test trading pairs
    response = client.get('/api/market/pairs', headers=auth_headers)
    assert response.status_code == 200
    assert len(response.json) > 0
    
    # Test market depth
    response = client.get('/api/market/depth/BTC/USDT', headers=auth_headers)
    assert response.status_code == 200
    assert 'depth' in response.json
    
    # Test recent trades
    response = client.get('/api/market/trades/BTC/USDT', headers=auth_headers)
    assert response.status_code == 200
    assert 'trades' in response.json

def test_risk_management_endpoints(app_context, test_user, auth_headers):
    """Test risk management-related API endpoints."""
    client = app.test_client()
    
    # Test get risk limits
    response = client.get('/api/risk/limits', headers=auth_headers)
    assert response.status_code == 200
    assert 'max_position_size' in response.json
    assert 'max_drawdown' in response.json
    
    # Test update risk limits
    response = client.put('/api/risk/limits', json={
        'max_position_size': 1.0,
        'max_drawdown': 0.1,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.05
    }, headers=auth_headers)
    assert response.status_code == 200
    assert response.json['max_position_size'] == 1.0
    
    # Test get risk metrics
    response = client.get('/api/risk/metrics', headers=auth_headers)
    assert response.status_code == 200
    assert 'current_drawdown' in response.json
    assert 'exposure' in response.json

def test_performance_endpoints(app_context, test_user, auth_headers):
    """Test performance-related API endpoints."""
    client = app.test_client()
    
    # Test get performance metrics
    response = client.get('/api/performance/metrics', headers=auth_headers)
    assert response.status_code == 200
    assert 'win_rate' in response.json
    assert 'profit_factor' in response.json
    assert 'sharpe_ratio' in response.json
    
    # Test get performance history
    response = client.get('/api/performance/history', headers=auth_headers)
    assert response.status_code == 200
    assert 'data' in response.json
    
    # Test get performance report
    response = client.get('/api/performance/report', headers=auth_headers)
    assert response.status_code == 200
    assert 'report' in response.json

def test_notification_endpoints(app_context, test_user, auth_headers):
    """Test notification-related API endpoints."""
    client = app.test_client()
    
    # Test get notifications
    response = client.get('/api/notifications', headers=auth_headers)
    assert response.status_code == 200
    assert isinstance(response.json, list)
    
    # Test create notification
    response = client.post('/api/notifications', json={
        'type': 'INFO',
        'message': 'Test notification'
    }, headers=auth_headers)
    assert response.status_code == 201
    assert response.json['message'] == 'Test notification'
    
    # Test mark notification as read
    notification_id = response.json['id']
    response = client.put(f'/api/notifications/{notification_id}/read', headers=auth_headers)
    assert response.status_code == 200
    assert response.json['read'] == True

def test_error_handling(app_context, test_user, auth_headers):
    """Test API error handling."""
    client = app.test_client()
    
    # Test invalid endpoint
    response = client.get('/api/invalid', headers=auth_headers)
    assert response.status_code == 404
    
    # Test invalid method
    response = client.put('/api/login', headers=auth_headers)
    assert response.status_code == 405
    
    # Test invalid data
    response = client.post('/api/trading/start', json={
        'invalid': 'data'
    }, headers=auth_headers)
    assert response.status_code == 400
    
    # Test unauthorized access
    response = client.get('/api/user/profile')
    assert response.status_code == 401
    
    # Test rate limiting
    for _ in range(100):
        client.get('/api/status', headers=auth_headers)
    response = client.get('/api/status', headers=auth_headers)
    assert response.status_code == 429
    
    # Test validation errors
    response = client.post('/api/trading/start', json={
        'strategy': 'invalid',
        'symbol': 'BTC/USDT',
        'parameters': {}
    }, headers=auth_headers)
    assert response.status_code == 422

def test_pagination(app_context, test_user, auth_headers):
    """Test API pagination."""
    client = app.test_client()
    
    # Create multiple trades
    for i in range(15):
        client.post('/api/trades', json={
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'price': 50000.0,
            'amount': 0.1,
            'total': 5000.0,
            'fee': 5.0
        }, headers=auth_headers)
    
    # Test default pagination
    response = client.get('/api/trades', headers=auth_headers)
    assert response.status_code == 200
    assert len(response.json) == 10  # Default page size
    
    # Test custom page size
    response = client.get('/api/trades?page_size=5', headers=auth_headers)
    assert response.status_code == 200
    assert len(response.json) == 5
    
    # Test page navigation
    response = client.get('/api/trades?page=2', headers=auth_headers)
    assert response.status_code == 200
    assert len(response.json) == 5  # Remaining items

def test_filtering_and_sorting(app_context, test_user, auth_headers):
    """Test API filtering and sorting."""
    client = app.test_client()
    
    # Create trades with different statuses
    for status in ['open', 'closed', 'cancelled']:
        client.post('/api/trades', json={
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'price': 50000.0,
            'amount': 0.1,
            'total': 5000.0,
            'fee': 5.0,
            'status': status
        }, headers=auth_headers)
    
    # Test filtering
    response = client.get('/api/trades?status=open', headers=auth_headers)
    assert response.status_code == 200
    assert all(trade['status'] == 'open' for trade in response.json)
    
    # Test sorting
    response = client.get('/api/trades?sort_by=price&order=desc', headers=auth_headers)
    assert response.status_code == 200
    prices = [trade['price'] for trade in response.json]
    assert prices == sorted(prices, reverse=True)
    
    # Test combined filtering and sorting
    response = client.get('/api/trades?status=closed&sort_by=timestamp&order=asc', headers=auth_headers)
    assert response.status_code == 200
    assert all(trade['status'] == 'closed' for trade in response.json)
    timestamps = [trade['timestamp'] for trade in response.json]
    assert timestamps == sorted(timestamps)

def test_websocket_endpoints(app_context, test_user, auth_headers, mock_websocket_server):
    """Test websocket-related API endpoints."""
    client = app.test_client()
    
    # Test websocket connection
    with patch('src.app.socketio') as mock_socketio:
        response = client.get('/ws', headers=auth_headers)
        assert response.status_code == 200
        mock_socketio.emit.assert_called()
    
    # Test trade subscription
    with patch('src.app.socketio') as mock_socketio:
        response = client.post('/api/ws/subscribe', json={
            'channel': 'trades',
            'symbol': 'BTC/USDT'
        }, headers=auth_headers)
        assert response.status_code == 200
        mock_socketio.emit.assert_called()
    
    # Test price subscription
    with patch('src.app.socketio') as mock_socketio:
        response = client.post('/api/ws/subscribe', json={
            'channel': 'prices',
            'symbol': 'BTC/USDT'
        }, headers=auth_headers)
        assert response.status_code == 200
        mock_socketio.emit.assert_called()

def test_backtest_endpoints(app_context, test_user, auth_headers):
    """Test backtest-related API endpoints."""
    client = app.test_client()
    
    # Test run backtest
    response = client.post('/api/backtest', json={
        'strategy': 'scalping',
        'symbol': 'BTC/USDT',
        'timeframe': '1h',
        'start_date': '2024-01-01',
        'end_date': '2024-01-31',
        'initial_balance': 10000.0,
        'parameters': {
            'sma_short': 10,
            'sma_long': 20,
            'rsi_period': 14
        }
    }, headers=auth_headers)
    assert response.status_code == 200
    assert 'results' in response.json
    
    # Test get backtest results
    response = client.get('/api/backtest/results/1', headers=auth_headers)
    assert response.status_code == 200
    assert 'total_return' in response.json
    assert 'sharpe_ratio' in response.json
    assert 'max_drawdown' in response.json
    
    # Test export backtest results
    response = client.get('/api/backtest/export/1', headers=auth_headers)
    assert response.status_code == 200
    assert response.headers['Content-Type'] == 'application/json'

def test_health_check_endpoints(app_context, test_user, auth_headers):
    """Test health check-related API endpoints."""
    client = app.test_client()
    
    # Test system health
    response = client.get('/api/health', headers=auth_headers)
    assert response.status_code == 200
    assert 'status' in response.json
    assert 'components' in response.json
    
    # Test component health
    response = client.get('/api/health/components', headers=auth_headers)
    assert response.status_code == 200
    assert 'trading_engine' in response.json
    assert 'market_data' in response.json
    assert 'risk_manager' in response.json
    
    # Test performance metrics
    response = client.get('/api/health/performance', headers=auth_headers)
    assert response.status_code == 200
    assert 'cpu_usage' in response.json
    assert 'memory_usage' in response.json
    assert 'latency' in response.json

def test_configuration_endpoints(app_context, test_user, auth_headers, mock_config):
    """Test configuration-related API endpoints."""
    client = app.test_client()
    
    # Test get configuration
    response = client.get('/api/config', headers=auth_headers)
    assert response.status_code == 200
    assert 'api' in response.json
    assert 'risk_management' in response.json
    
    # Test update configuration
    response = client.put('/api/config', json={
        'risk_management': {
            'max_position_size': 2.0,
            'max_drawdown': 0.15
        }
    }, headers=auth_headers)
    assert response.status_code == 200
    assert response.json['risk_management']['max_position_size'] == 2.0
    
    # Test reset configuration
    response = client.post('/api/config/reset', headers=auth_headers)
    assert response.status_code == 200
    assert response.json['status'] == 'reset' 