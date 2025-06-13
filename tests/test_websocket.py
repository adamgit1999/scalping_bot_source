import pytest
import asyncio
from datetime import datetime, timedelta, timezone
import json
from unittest.mock import Mock, AsyncMock, patch
from typing import Any, Dict, List

from src.app import app, db, User, Strategy, Trade
from src.websocket_server import WebSocketServer
from src.trading_engine import TradingEngine
from src.notification_system import NotificationSystem
from src.performance_monitoring import PerformanceMonitor
from src.models import Strategy

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
def mock_trading_engine():
    """Create a mock trading engine."""
    engine = Mock(spec=TradingEngine)
    engine.get_performance_metrics = Mock(return_value={
        'cpu_usage': 50.0,
        'memory_usage': 60.0,
        'latency': 100.0
    })
    return engine

@pytest.fixture
def mock_notification_system():
    """Create a mock notification system."""
    system = Mock(spec=NotificationSystem)
    system.send_notification = AsyncMock()
    return system

@pytest.fixture
def mock_performance_monitor():
    """Create a mock performance monitor."""
    monitor = Mock(spec=PerformanceMonitor)
    monitor.get_metrics = Mock(return_value={
        'cpu': 50.0,
        'memory': 60.0,
        'latency': 100.0
    })
    return monitor

@pytest.fixture
def websocket_server(mock_trading_engine, mock_notification_system, mock_performance_monitor):
    """Create a WebSocket server instance with mocked dependencies."""
    server = WebSocketServer(
        app=app,
        trading_engine=mock_trading_engine
    )
    server.notification_system = mock_notification_system
    server.performance_monitor = mock_performance_monitor
    return server

@pytest.fixture
def mock_socket():
    """Create a mock WebSocket connection."""
    socket = AsyncMock()
    socket.send = AsyncMock()
    socket.close = AsyncMock()
    return socket

def test_websocket_initialization(websocket_server):
    """Test WebSocket server initialization."""
    assert websocket_server.message_queue is not None
    assert websocket_server.is_running is False
    assert websocket_server.clients == {}
    assert websocket_server.subscriptions == {}

def test_client_connection(websocket_server, mock_socket):
    """Test client connection handling."""
    # Test new client connection
    client_id = 'test_client'
    websocket_server.handle_connect(client_id, mock_socket)
    assert client_id in websocket_server.clients
    assert websocket_server.clients[client_id] == mock_socket
    
    # Test existing client connection
    websocket_server.handle_connect(client_id, mock_socket)
    assert len(websocket_server.clients) == 1

def test_client_disconnection(websocket_server, mock_socket):
    """Test client disconnection handling."""
    # Connect client
    client_id = 'test_client'
    websocket_server.handle_connect(client_id, mock_socket)
    
    # Test client disconnection
    websocket_server.handle_disconnect(client_id)
    assert client_id not in websocket_server.clients
    assert client_id not in websocket_server.subscriptions

def test_channel_subscription(websocket_server, mock_socket):
    """Test channel subscription handling."""
    # Connect client
    client_id = 'test_client'
    websocket_server.handle_connect(client_id, mock_socket)
    
    # Test channel subscription
    channel = 'market_data'
    websocket_server.handle_subscribe(client_id, channel)
    assert channel in websocket_server.subscriptions
    assert client_id in websocket_server.subscriptions[channel]
    
    # Test duplicate subscription
    websocket_server.handle_subscribe(client_id, channel)
    assert len(websocket_server.subscriptions[channel]) == 1

def test_channel_unsubscription(websocket_server, mock_socket):
    """Test channel unsubscription handling."""
    # Connect client and subscribe to channel
    client_id = 'test_client'
    channel = 'market_data'
    websocket_server.handle_connect(client_id, mock_socket)
    websocket_server.handle_subscribe(client_id, channel)
    
    # Test channel unsubscription
    websocket_server.handle_unsubscribe(client_id, channel)
    assert channel not in websocket_server.subscriptions
    
    # Test unsubscribing from nonexistent channel
    websocket_server.handle_unsubscribe(client_id, 'nonexistent')
    assert 'nonexistent' not in websocket_server.subscriptions

def test_message_broadcasting(websocket_server, mock_socket):
    """Test message broadcasting functionality."""
    # Connect client and subscribe to channel
    client_id = 'test_client'
    channel = 'market_data'
    websocket_server.handle_connect(client_id, mock_socket)
    websocket_server.handle_subscribe(client_id, channel)
    
    # Test message broadcast
    message = {'type': 'price', 'symbol': 'BTC/USDT', 'price': 50000.0}
    websocket_server.broadcast_message(channel, message)
    mock_socket.send.assert_called_once_with(json.dumps(message))
    
    # Test broadcast to empty channel
    websocket_server.broadcast_message('empty_channel', message)
    assert mock_socket.send.call_count == 1

def test_market_data_broadcasting(websocket_server, mock_socket):
    """Test market data broadcasting."""
    # Connect client and subscribe to market data
    client_id = 'test_client'
    websocket_server.handle_connect(client_id, mock_socket)
    websocket_server.handle_subscribe(client_id, 'market_data')
    
    # Test market data broadcast
    market_data = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'symbol': 'BTC/USDT',
        'price': 50000.0,
        'volume': 100.0
    }
    websocket_server.broadcast_market_data(market_data)
    mock_socket.send.assert_called_once()
    
    # Verify message format
    sent_message = json.loads(mock_socket.send.call_args[0][0])
    assert sent_message['type'] == 'market_data'
    assert sent_message['data'] == market_data

def test_trade_broadcasting(websocket_server, mock_socket):
    """Test trade broadcasting."""
    # Connect client and subscribe to trades
    client_id = 'test_client'
    websocket_server.handle_connect(client_id, mock_socket)
    websocket_server.handle_subscribe(client_id, 'trades')
    
    # Test trade broadcast
    trade = {
        'id': 1,
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'price': 50000.0,
        'amount': 0.1,
        'total': 5000.0,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    websocket_server.broadcast_trade(trade)
    mock_socket.send.assert_called_once()
    
    # Verify message format
    sent_message = json.loads(mock_socket.send.call_args[0][0])
    assert sent_message['type'] == 'trade'
    assert sent_message['data'] == trade

def test_error_broadcasting(websocket_server, mock_socket):
    """Test error broadcasting."""
    # Connect client and subscribe to errors
    client_id = 'test_client'
    websocket_server.handle_connect(client_id, mock_socket)
    websocket_server.handle_subscribe(client_id, 'errors')
    
    # Test error broadcast
    error = {
        'type': 'execution_error',
        'message': 'Failed to execute trade',
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    websocket_server.broadcast_error(error)
    mock_socket.send.assert_called_once()
    
    # Verify message format
    sent_message = json.loads(mock_socket.send.call_args[0][0])
    assert sent_message['type'] == 'error'
    assert sent_message['data'] == error

def test_server_lifecycle(websocket_server):
    """Test server lifecycle management."""
    # Test server start
    websocket_server.start()
    assert websocket_server.is_running is True
    
    # Test server stop
    websocket_server.stop()
    assert websocket_server.is_running is False

def test_message_queue_handling(websocket_server, mock_socket):
    """Test message queue handling."""
    # Connect client
    client_id = 'test_client'
    websocket_server.handle_connect(client_id, mock_socket)
    
    # Add message to queue
    message = {'type': 'test', 'data': 'test_data'}
    websocket_server.message_queue.put(message)
    
    # Process message queue
    websocket_server.process_message_queue()
    mock_socket.send.assert_called_once_with(json.dumps(message))

def test_heartbeat_handling(websocket_server, mock_socket):
    """Test heartbeat message handling."""
    # Connect client
    client_id = 'test_client'
    websocket_server.handle_connect(client_id, mock_socket)
    
    # Send heartbeat
    websocket_server.handle_heartbeat(client_id)
    mock_socket.send.assert_called_once_with(json.dumps({'type': 'pong'}))

def test_custom_message_handling(websocket_server, mock_socket):
    """Test custom message handling."""
    # Connect client
    client_id = 'test_client'
    websocket_server.handle_connect(client_id, mock_socket)
    
    # Send custom message
    message = {
        'type': 'custom',
        'data': {
            'action': 'test_action',
            'parameters': {'param1': 'value1'}
        }
    }
    websocket_server.handle_message(client_id, message)
    mock_socket.send.assert_called_once()

def test_multiple_clients(websocket_server):
    """Test handling multiple clients."""
    # Create multiple mock sockets
    sockets = [AsyncMock() for _ in range(3)]
    client_ids = [f'client_{i}' for i in range(3)]
    
    # Connect clients
    for client_id, socket in zip(client_ids, sockets):
        websocket_server.handle_connect(client_id, socket)
    
    # Subscribe all clients to a channel
    channel = 'market_data'
    for client_id in client_ids:
        websocket_server.handle_subscribe(client_id, channel)
    
    # Broadcast message
    message = {'type': 'test', 'data': 'test_data'}
    websocket_server.broadcast_message(channel, message)
    
    # Verify all clients received the message
    for socket in sockets:
        socket.send.assert_called_once_with(json.dumps(message))

def test_channel_management(websocket_server, mock_socket):
    """Test channel management functionality."""
    # Connect client
    client_id = 'test_client'
    websocket_server.handle_connect(client_id, mock_socket)
    
    # Subscribe to multiple channels
    channels = ['market_data', 'trades', 'errors']
    for channel in channels:
        websocket_server.handle_subscribe(client_id, channel)
    
    # Verify subscriptions
    for channel in channels:
        assert channel in websocket_server.subscriptions
        assert client_id in websocket_server.subscriptions[channel]
    
    # Unsubscribe from one channel
    websocket_server.handle_unsubscribe(client_id, channels[0])
    assert channels[0] not in websocket_server.subscriptions
    assert all(channel in websocket_server.subscriptions for channel in channels[1:])

def test_error_handling(websocket_server, mock_socket):
    """Test error handling in WebSocket server."""
    # Connect client
    client_id = 'test_client'
    websocket_server.handle_connect(client_id, mock_socket)
    
    # Test invalid message format
    invalid_message = 'invalid_json'
    websocket_server.handle_message(client_id, invalid_message)
    mock_socket.send.assert_called_once()
    error_message = json.loads(mock_socket.send.call_args[0][0])
    assert error_message['type'] == 'error'
    
    # Test invalid channel subscription
    websocket_server.handle_subscribe(client_id, '')
    assert '' not in websocket_server.subscriptions
    
    # Test sending to disconnected client
    websocket_server.handle_disconnect(client_id)
    websocket_server.broadcast_message('market_data', {'type': 'test'})
    assert mock_socket.send.call_count == 1  # No additional calls after disconnect 

def test_performance_broadcasting(websocket_server, mock_socket):
    """Test performance metrics broadcasting."""
    # Connect client and subscribe to performance updates
    client_id = 'test_client'
    websocket_server.handle_connect(client_id, mock_socket)
    websocket_server.handle_subscribe(client_id, 'performance')
    
    # Test performance broadcast
    performance_data = {
        'cpu_usage': 50.0,
        'memory_usage': 60.0,
        'latency': 100.0,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    websocket_server.broadcast_performance(performance_data)
    mock_socket.send.assert_called_once()
    
    # Verify message format
    sent_message = json.loads(mock_socket.send.call_args[0][0])
    assert sent_message['type'] == 'performance'
    assert sent_message['data'] == performance_data

def test_bot_status_broadcasting(websocket_server, mock_socket):
    """Test bot status broadcasting."""
    # Connect client and subscribe to bot status
    client_id = 'test_client'
    websocket_server.handle_connect(client_id, mock_socket)
    websocket_server.handle_subscribe(client_id, 'bot_status')
    
    # Test bot status broadcast
    status_data = {
        'status': 'running',
        'active_strategies': ['scalping', 'momentum'],
        'open_positions': 2,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    websocket_server.broadcast_bot_status(status_data)
    mock_socket.send.assert_called_once()
    
    # Verify message format
    sent_message = json.loads(mock_socket.send.call_args[0][0])
    assert sent_message['type'] == 'bot_status'
    assert sent_message['data'] == status_data

def test_notification_broadcasting(websocket_server, mock_socket):
    """Test notification broadcasting."""
    # Connect client and subscribe to notifications
    client_id = 'test_client'
    websocket_server.handle_connect(client_id, mock_socket)
    websocket_server.handle_subscribe(client_id, 'notifications')
    
    # Test notification broadcast
    notification = {
        'type': 'trade',
        'priority': 'high',
        'title': 'Trade Executed',
        'message': 'Buy order executed for BTC/USDT',
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    websocket_server.broadcast_notification(notification)
    mock_socket.send.assert_called_once()
    
    # Verify message format
    sent_message = json.loads(mock_socket.send.call_args[0][0])
    assert sent_message['type'] == 'notification'
    assert sent_message['data'] == notification

def test_connection_limits(websocket_server):
    """Test connection limit handling."""
    # Create maximum number of connections
    max_connections = 100
    sockets = [AsyncMock() for _ in range(max_connections)]
    
    # Connect clients up to limit
    for i in range(max_connections):
        client_id = f'client_{i}'
        websocket_server.handle_connect(client_id, sockets[i])
    
    # Try to connect one more client
    extra_socket = AsyncMock()
    websocket_server.handle_connect('extra_client', extra_socket)
    
    # Verify connection was rejected
    assert 'extra_client' not in websocket_server.clients
    extra_socket.close.assert_called_once()

def test_message_validation(websocket_server, mock_socket):
    """Test message validation."""
    # Connect client
    client_id = 'test_client'
    websocket_server.handle_connect(client_id, mock_socket)
    
    # Test valid message
    valid_message = {
        'type': 'subscribe',
        'channel': 'market_data',
        'symbol': 'BTC/USDT'
    }
    assert websocket_server.validate_message(valid_message) is True
    
    # Test invalid message
    invalid_message = {
        'type': 'invalid_type',
        'data': 'invalid_data'
    }
    assert websocket_server.validate_message(invalid_message) is False

def test_subscription_validation(websocket_server):
    """Test subscription validation."""
    # Test valid channels
    valid_channels = ['market_data', 'trades', 'performance', 'bot_status', 'notifications']
    for channel in valid_channels:
        assert websocket_server.validate_subscription(channel) is True
    
    # Test invalid channel
    assert websocket_server.validate_subscription('invalid_channel') is False 