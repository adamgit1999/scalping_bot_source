import pytest
from websocket_server import WebSocketServer
import asyncio
from datetime import datetime
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from notification.notification_system import NotificationSystem, NotificationType, NotificationPriority

@pytest.fixture
def websocket_server():
    return WebSocketServer()

@pytest.fixture
def sample_trade():
    return {
        'id': '12345',
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'price': 50000.0,
        'amount': 0.1,
        'total': 5000.0,
        'fee': 5.0,
        'profit': 100.0,
        'timestamp': datetime.now().isoformat()
    }

@pytest.fixture
def sample_price():
    return {
        'symbol': 'BTC/USDT',
        'price': 50000.0,
        'timestamp': datetime.now().isoformat()
    }

@pytest.fixture
def sample_bot_status():
    return {
        'status': 'running',
        'strategy': 'scalping',
        'active_trades': 5,
        'timestamp': datetime.now().isoformat()
    }

@pytest.fixture
def sample_performance():
    return {
        'total_profit': 1000.0,
        'win_rate': 0.65,
        'total_trades': 100,
        'active_trades': 5,
        'timestamp': datetime.now().isoformat()
    }

@pytest.fixture
def mock_websocket():
    """Create a mock websocket connection."""
    return Mock()

@pytest.fixture
def notification_system():
    """Create a notification system instance."""
    return NotificationSystem()

@pytest.fixture
def sample_notification():
    """Create a sample notification."""
    return {
        'type': NotificationType.TRADE,
        'priority': NotificationPriority.HIGH,
        'title': 'Trade Executed',
        'message': 'BTC/USDT buy order executed at 50000',
        'timestamp': datetime.now().isoformat(),
        'data': {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'price': 50000,
            'amount': 0.1
        }
    }

@pytest.mark.asyncio
async def test_initialize(websocket_server):
    """Test WebSocket server initialization."""
    await websocket_server.initialize()
    assert websocket_server.sio is not None
    assert websocket_server.connected_clients == set()
    assert websocket_server.subscriptions == {}
    assert websocket_server.message_queue is not None
    assert websocket_server.is_running is False

@pytest.mark.asyncio
async def test_initialize_with_custom_config():
    """Test initialization with custom configuration."""
    config = {
        'CORS_ORIGINS': ['http://localhost:3000'],
        'ASYNC_MODE': 'eventlet',
        'PING_TIMEOUT': 60,
        'PING_INTERVAL': 25
    }
    
    server = WebSocketServer(config=config)
    await server.initialize()
    
    assert server.sio is not None
    assert server.config == config

@pytest.mark.asyncio
async def test_handle_connect(websocket_server):
    """Test handling client connection."""
    sid = 'test_sid'
    environ = {'REMOTE_ADDR': '127.0.0.1'}
    
    await websocket_server.handle_connect(sid, environ)
    
    assert sid in websocket_server.connected_clients
    assert sid in websocket_server.subscriptions
    assert websocket_server.subscriptions[sid] == {
        'trades': False,
        'prices': False,
        'bot_status': False,
        'performance': False
    }

@pytest.mark.asyncio
async def test_handle_connect_with_existing_client(websocket_server):
    """Test handling connection for existing client."""
    sid = 'test_sid'
    environ = {'REMOTE_ADDR': '127.0.0.1'}
    
    # First connection
    await websocket_server.handle_connect(sid, environ)
    
    # Second connection with same sid
    await websocket_server.handle_connect(sid, environ)
    
    assert sid in websocket_server.connected_clients
    assert len(websocket_server.connected_clients) == 1

@pytest.mark.asyncio
async def test_handle_disconnect(websocket_server):
    """Test handling client disconnection."""
    sid = 'test_sid'
    websocket_server.connected_clients.add(sid)
    websocket_server.subscriptions[sid] = {'trades': True}
    
    await websocket_server.handle_disconnect(sid)
    
    assert sid not in websocket_server.connected_clients
    assert sid not in websocket_server.subscriptions

@pytest.mark.asyncio
async def test_handle_disconnect_nonexistent_client(websocket_server):
    """Test handling disconnection for nonexistent client."""
    sid = 'nonexistent_sid'
    
    await websocket_server.handle_disconnect(sid)
    
    assert sid not in websocket_server.connected_clients
    assert sid not in websocket_server.subscriptions

@pytest.mark.asyncio
async def test_handle_subscribe(websocket_server):
    """Test handling subscription request."""
    sid = 'test_sid'
    data = {
        'channels': ['trades', 'prices', 'bot_status', 'performance']
    }
    
    await websocket_server.handle_subscribe(sid, data)
    
    assert sid in websocket_server.subscriptions
    assert websocket_server.subscriptions[sid] == {
        'trades': True,
        'prices': True,
        'bot_status': True,
        'performance': True
    }

@pytest.mark.asyncio
async def test_handle_subscribe_with_invalid_channels(websocket_server):
    """Test handling subscription request with invalid channels."""
    sid = 'test_sid'
    data = {
        'channels': ['invalid_channel']
    }
    
    with pytest.raises(ValueError):
        await websocket_server.handle_subscribe(sid, data)
    
    assert sid not in websocket_server.subscriptions

@pytest.mark.asyncio
async def test_handle_unsubscribe(websocket_server):
    """Test handling unsubscribe request."""
    sid = 'test_sid'
    websocket_server.subscriptions[sid] = {
        'trades': True,
        'prices': True,
        'bot_status': True,
        'performance': True
    }
    
    data = {
        'channels': ['trades', 'prices']
    }
    
    await websocket_server.handle_unsubscribe(sid, data)
    
    assert websocket_server.subscriptions[sid] == {
        'trades': False,
        'prices': False,
        'bot_status': True,
        'performance': True
    }

@pytest.mark.asyncio
async def test_handle_unsubscribe_nonexistent_client(websocket_server):
    """Test handling unsubscribe request for nonexistent client."""
    sid = 'nonexistent_sid'
    data = {
        'channels': ['trades']
    }
    
    with pytest.raises(ValueError):
        await websocket_server.handle_unsubscribe(sid, data)

@pytest.mark.asyncio
async def test_broadcast_trade(websocket_server, sample_trade):
    """Test broadcasting trade update."""
    sid = 'test_sid'
    websocket_server.connected_clients.add(sid)
    websocket_server.subscriptions[sid] = {'trades': True}
    
    mock_sio = AsyncMock()
    websocket_server.sio = mock_sio
    
    await websocket_server.broadcast_trade(sample_trade)
    
    mock_sio.emit.assert_called_once_with('trade', sample_trade, room=sid)

@pytest.mark.asyncio
async def test_broadcast_trade_with_no_subscribers(websocket_server, sample_trade):
    """Test broadcasting trade update with no subscribers."""
    mock_sio = AsyncMock()
    websocket_server.sio = mock_sio
    
    await websocket_server.broadcast_trade(sample_trade)
    
    mock_sio.emit.assert_not_called()

@pytest.mark.asyncio
async def test_broadcast_price(websocket_server, sample_price):
    """Test broadcasting price update."""
    sid = 'test_sid'
    websocket_server.connected_clients.add(sid)
    websocket_server.subscriptions[sid] = {'prices': True}
    
    mock_sio = AsyncMock()
    websocket_server.sio = mock_sio
    
    await websocket_server.broadcast_price(sample_price)
    
    mock_sio.emit.assert_called_once_with('price', sample_price, room=sid)

@pytest.mark.asyncio
async def test_broadcast_bot_status(websocket_server, sample_bot_status):
    """Test broadcasting bot status update."""
    sid = 'test_sid'
    websocket_server.connected_clients.add(sid)
    websocket_server.subscriptions[sid] = {'bot_status': True}
    
    mock_sio = AsyncMock()
    websocket_server.sio = mock_sio
    
    await websocket_server.broadcast_bot_status(sample_bot_status)
    
    mock_sio.emit.assert_called_once_with('bot_status', sample_bot_status, room=sid)

@pytest.mark.asyncio
async def test_broadcast_performance(websocket_server, sample_performance):
    """Test broadcasting performance update."""
    sid = 'test_sid'
    websocket_server.connected_clients.add(sid)
    websocket_server.subscriptions[sid] = {'performance': True}
    
    mock_sio = AsyncMock()
    websocket_server.sio = mock_sio
    
    await websocket_server.broadcast_performance(sample_performance)
    
    mock_sio.emit.assert_called_once_with('performance', sample_performance, room=sid)

def test_validate_subscription(websocket_server):
    """Test subscription validation."""
    valid_channels = ['trades', 'prices', 'bot_status', 'performance']
    invalid_channels = ['invalid_channel']
    
    assert websocket_server._validate_subscription(valid_channels) is True
    assert websocket_server._validate_subscription(invalid_channels) is False

def test_get_subscribed_clients(websocket_server):
    """Test getting subscribed clients for a channel."""
    sid1 = 'test_sid1'
    sid2 = 'test_sid2'
    websocket_server.connected_clients.add(sid1)
    websocket_server.connected_clients.add(sid2)
    websocket_server.subscriptions[sid1] = {'trades': True, 'prices': False}
    websocket_server.subscriptions[sid2] = {'trades': True, 'prices': True}
    
    subscribed_clients = websocket_server._get_subscribed_clients('trades')
    assert sid1 in subscribed_clients
    assert sid2 in subscribed_clients
    
    subscribed_clients = websocket_server._get_subscribed_clients('prices')
    assert sid1 not in subscribed_clients
    assert sid2 in subscribed_clients

@pytest.mark.asyncio
async def test_start_server(websocket_server):
    """Test starting the WebSocket server."""
    mock_sio = AsyncMock()
    websocket_server.sio = mock_sio
    
    await websocket_server.start()
    
    assert websocket_server.is_running is True
    mock_sio.start_background_task.assert_called_once()

@pytest.mark.asyncio
async def test_stop_server(websocket_server):
    """Test stopping the WebSocket server."""
    mock_sio = AsyncMock()
    websocket_server.sio = mock_sio
    websocket_server.is_running = True
    
    await websocket_server.stop()
    
    assert websocket_server.is_running is False
    mock_sio.stop.assert_called_once()

@pytest.mark.asyncio
async def test_handle_error(websocket_server):
    """Test handling server errors."""
    error = Exception('Test error')
    
    with pytest.raises(Exception):
        await websocket_server.handle_error(error)

@pytest.mark.asyncio
async def test_handle_ping(websocket_server):
    """Test handling ping messages."""
    sid = 'test_sid'
    data = {'timestamp': datetime.now().isoformat()}
    
    mock_sio = AsyncMock()
    websocket_server.sio = mock_sio
    
    await websocket_server.handle_ping(sid, data)
    
    mock_sio.emit.assert_called_once_with('pong', data, room=sid)

@pytest.mark.asyncio
async def test_handle_message(websocket_server):
    """Test handling custom messages."""
    sid = 'test_sid'
    data = {'type': 'custom', 'content': 'test message'}
    
    mock_sio = AsyncMock()
    websocket_server.sio = mock_sio
    
    await websocket_server.handle_message(sid, data)
    
    mock_sio.emit.assert_called_once_with('message', data, room=sid)

def test_validate_message(websocket_server):
    """Test message validation."""
    valid_message = {'type': 'custom', 'content': 'test message'}
    invalid_message = {'type': 'invalid'}
    
    assert websocket_server._validate_message(valid_message) is True
    assert websocket_server._validate_message(invalid_message) is False

@pytest.mark.asyncio
async def test_handle_reconnect(websocket_server):
    """Test handling client reconnection."""
    sid = 'test_sid'
    environ = {'REMOTE_ADDR': '127.0.0.1'}
    
    await websocket_server.handle_reconnect(sid, environ)
    
    assert sid in websocket_server.connected_clients
    assert sid in websocket_server.subscriptions

@pytest.mark.asyncio
async def test_handle_heartbeat(websocket_server):
    """Test handling heartbeat messages."""
    sid = 'test_sid'
    data = {'timestamp': datetime.now().isoformat()}
    
    mock_sio = AsyncMock()
    websocket_server.sio = mock_sio
    
    await websocket_server.handle_heartbeat(sid, data)
    
    mock_sio.emit.assert_called_once_with('heartbeat_ack', data, room=sid)

def test_websocket_connection(notification_system, mock_websocket):
    """Test websocket connection handling."""
    # Register websocket
    notification_system.register_websocket(mock_websocket)
    assert mock_websocket in notification_system.websockets
    
    # Unregister websocket
    notification_system.unregister_websocket(mock_websocket)
    assert mock_websocket not in notification_system.websockets

def test_websocket_broadcast(notification_system, mock_websocket, sample_notification):
    """Test broadcasting notifications to websockets."""
    # Register multiple websockets
    websockets = [Mock() for _ in range(3)]
    for ws in websockets:
        notification_system.register_websocket(ws)
    
    # Add notification
    notification_system.add_notification(**sample_notification)
    
    # Verify all websockets received the notification
    for ws in websockets:
        ws.send.assert_called_once()
        sent_data = json.loads(ws.send.call_args[0][0])
        assert sent_data['title'] == sample_notification['title']
        assert sent_data['message'] == sample_notification['message']

def test_websocket_error_handling(notification_system, mock_websocket, sample_notification):
    """Test websocket error handling."""
    # Register websocket that will raise an error
    mock_websocket.send.side_effect = Exception("Connection error")
    notification_system.register_websocket(mock_websocket)
    
    # Add notification
    notification_system.add_notification(**sample_notification)
    
    # Verify websocket was unregistered
    assert mock_websocket not in notification_system.websockets

def test_websocket_serialization(notification_system, mock_websocket, sample_notification):
    """Test websocket message serialization."""
    # Register websocket
    notification_system.register_websocket(mock_websocket)
    
    # Add notification
    notification_system.add_notification(**sample_notification)
    
    # Verify serialization
    sent_data = json.loads(mock_websocket.send.call_args[0][0])
    assert isinstance(sent_data, dict)
    assert 'type' in sent_data
    assert 'priority' in sent_data
    assert 'title' in sent_data
    assert 'message' in sent_data
    assert 'timestamp' in sent_data
    assert 'data' in sent_data

def test_websocket_multiple_notifications(notification_system, mock_websocket):
    """Test sending multiple notifications to websocket."""
    # Register websocket
    notification_system.register_websocket(mock_websocket)
    
    # Add multiple notifications
    for i in range(5):
        notification = {
            'type': NotificationType.TRADE,
            'priority': NotificationPriority.HIGH,
            'title': f'Trade {i}',
            'message': f'Trade {i} executed',
            'timestamp': datetime.now().isoformat(),
            'data': {'trade_id': i}
        }
        notification_system.add_notification(**notification)
    
    # Verify all notifications were sent
    assert mock_websocket.send.call_count == 5
    
    # Verify order of notifications
    calls = mock_websocket.send.call_args_list
    for i, call in enumerate(calls):
        sent_data = json.loads(call[0][0])
        assert sent_data['title'] == f'Trade {i}'
        assert sent_data['message'] == f'Trade {i} executed'

def test_websocket_connection_limits(notification_system):
    """Test websocket connection limits."""
    # Try to register many websockets
    websockets = [Mock() for _ in range(1000)]
    for ws in websockets:
        notification_system.register_websocket(ws)
    
    # Verify all websockets were registered
    assert len(notification_system.websockets) == 1000

def test_websocket_reconnection(notification_system, mock_websocket):
    """Test websocket reconnection handling."""
    # Register websocket
    notification_system.register_websocket(mock_websocket)
    
    # Unregister and register again
    notification_system.unregister_websocket(mock_websocket)
    notification_system.register_websocket(mock_websocket)
    
    # Verify websocket is registered
    assert mock_websocket in notification_system.websockets

def test_websocket_message_ordering(notification_system, mock_websocket):
    """Test websocket message ordering."""
    # Register websocket
    notification_system.register_websocket(mock_websocket)
    
    # Add notifications with different priorities
    priorities = [
        NotificationPriority.LOW,
        NotificationPriority.MEDIUM,
        NotificationPriority.HIGH,
        NotificationPriority.CRITICAL
    ]
    
    for priority in priorities:
        notification = {
            'type': NotificationType.SYSTEM,
            'priority': priority,
            'title': f'{priority.name} Priority',
            'message': f'This is a {priority.name.lower()} priority notification',
            'timestamp': datetime.now().isoformat(),
            'data': {}
        }
        notification_system.add_notification(**notification)
    
    # Verify message ordering
    calls = mock_websocket.send.call_args_list
    for i, call in enumerate(calls):
        sent_data = json.loads(call[0][0])
        assert sent_data['priority'] == priorities[-(i+1)].value  # Highest priority first

def test_websocket_message_filtering(notification_system, mock_websocket):
    """Test websocket message filtering."""
    # Register websocket
    notification_system.register_websocket(mock_websocket)
    
    # Add notifications of different types
    types = [
        NotificationType.TRADE,
        NotificationType.SYSTEM,
        NotificationType.ALERT,
        NotificationType.ERROR
    ]
    
    for notification_type in types:
        notification = {
            'type': notification_type,
            'priority': NotificationPriority.MEDIUM,
            'title': f'{notification_type.name} Notification',
            'message': f'This is a {notification_type.name.lower()} notification',
            'timestamp': datetime.now().isoformat(),
            'data': {}
        }
        notification_system.add_notification(**notification)
    
    # Verify all messages were sent
    assert mock_websocket.send.call_count == len(types)
    
    # Verify message types
    calls = mock_websocket.send.call_args_list
    for i, call in enumerate(calls):
        sent_data = json.loads(call[0][0])
        assert sent_data['type'] == types[i].value

def test_websocket_message_validation(notification_system, mock_websocket):
    """Test websocket message validation."""
    # Register websocket
    notification_system.register_websocket(mock_websocket)
    
    # Try to send invalid notification
    invalid_notification = {
        'type': 'INVALID_TYPE',
        'priority': NotificationPriority.HIGH,
        'title': 'Invalid Notification',
        'message': 'This is an invalid notification',
        'timestamp': datetime.now().isoformat(),
        'data': {}
    }
    
    # Add notification
    notification_system.add_notification(**invalid_notification)
    
    # Verify no message was sent
    mock_websocket.send.assert_not_called()

def test_websocket_message_retry(notification_system, mock_websocket):
    """Test websocket message retry mechanism."""
    # Register websocket that will fail once then succeed
    mock_websocket.send.side_effect = [Exception("Connection error"), None]
    notification_system.register_websocket(mock_websocket)
    
    # Add notification
    notification = {
        'type': NotificationType.TRADE,
        'priority': NotificationPriority.HIGH,
        'title': 'Trade Notification',
        'message': 'This is a trade notification',
        'timestamp': datetime.now().isoformat(),
        'data': {}
    }
    notification_system.add_notification(**notification)
    
    # Verify websocket was unregistered after first failure
    assert mock_websocket not in notification_system.websockets 