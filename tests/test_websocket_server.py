import pytest
import asyncio
from datetime import datetime, timezone
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from src.notification.notification_system import NotificationSystem, NotificationType, NotificationPriority, NotificationChannel
from src.websocket.server import WebSocketServer
from src.trading_engine import TradingEngine
from src.exceptions import WebSocketError

@pytest.fixture
def mock_exchange():
    """Create a mock exchange that doesn't make real network calls."""
    exchange = AsyncMock()
    exchange.load_markets = AsyncMock()
    exchange.fetch_positions = AsyncMock(return_value=[])
    exchange.fetch_ohlcv = AsyncMock(return_value=[])
    return exchange

@pytest.fixture
def mock_app():
    """Create a mock Flask application."""
    app = Mock()
    app.config = {
        'WEBSOCKET_HOST': '127.0.0.1',
        'WEBSOCKET_PORT': 5001,
        'WEBSOCKET_HEARTBEAT_INTERVAL': 30
    }
    return app

@pytest.fixture
def mock_trading_engine(mock_exchange):
    """Create a mock trading engine."""
    engine = Mock(spec=TradingEngine)
    engine.exchange = mock_exchange
    engine.get_market_data = AsyncMock(return_value={})
    engine.get_active_trades = AsyncMock(return_value=[])
    engine.get_performance_metrics = AsyncMock(return_value={})
    return engine

@pytest.fixture
def websocket_server(mock_app, mock_trading_engine):
    """Create a WebSocket server instance with mocked dependencies."""
    server = WebSocketServer(mock_app, mock_trading_engine)
    server.server = AsyncMock()  # Mock the actual server
    server.running = True
    return server

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
    websocket = AsyncMock()
    websocket.__iter__.return_value = []
    return websocket

@pytest.fixture
def sample_notification():
    """Create a sample notification."""
    return {
        'type': NotificationType.TRADE,
        'priority': NotificationPriority.HIGH,
        'title': 'Trade Executed',
        'message': 'Trade executed at 50000',
        'timestamp': datetime.now().isoformat(),
        'data': {
            'price': 50000,
            'amount': 0.1
        },
        'channels': [NotificationChannel.WEBSOCKET],
        'recipients': ['user1', 'user2']
    }

@pytest.fixture
def mock_websocket_with_message():
    """Create a mock websocket that yields a single message."""
    async def message_gen():
        yield json.dumps({'type': 'ping'})
    websocket = AsyncMock()
    websocket.__aiter__.side_effect = message_gen
    return websocket

@pytest.fixture
def mock_websocket_with_two_messages():
    """Mock websocket that yields two messages to keep connection open longer."""
    async def message_gen():
        yield json.dumps({'type': 'ping'})
        yield json.dumps({'type': 'ping'})
    websocket = AsyncMock()
    websocket.__aiter__.side_effect = message_gen
    return websocket

@pytest.fixture
def mock_websocket_infinite():
    """Mock websocket that never exhausts its iterator, keeping the connection open."""
    async def message_gen():
        while True:
            await asyncio.sleep(0.1)
            yield json.dumps({'type': 'ping'})
    websocket = AsyncMock()
    websocket.__aiter__.side_effect = message_gen
    return websocket

@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return {
        'host': 'localhost',
        'port': 8765,
        'max_connections': 100,
        'ping_interval': 30,
        'ping_timeout': 10,
        'max_message_size': 1024 * 1024,  # 1MB
        'allowed_origins': ['*']
    }

@pytest.fixture
def mock_message_handler():
    """Create a mock message handler."""
    handler = AsyncMock()
    handler.handle_message.return_value = {'status': 'success'}
    return handler

@pytest.fixture
def websocket_server(mock_config, mock_message_handler):
    """Create a websocket server instance."""
    return WebSocketServer(mock_config, mock_message_handler)

@pytest.mark.asyncio
async def test_server_initialization(websocket_server, mock_config):
    """Test server initialization."""
    assert websocket_server.host == mock_config['host']
    assert websocket_server.port == mock_config['port']
    assert websocket_server.max_connections == mock_config['max_connections']
    assert websocket_server.ping_interval == mock_config['ping_interval']
    assert websocket_server.ping_timeout == mock_config['ping_timeout']
    assert websocket_server.max_message_size == mock_config['max_message_size']
    assert websocket_server.allowed_origins == mock_config['allowed_origins']
    assert not websocket_server.is_running
    assert len(websocket_server.clients) == 0

@pytest.mark.asyncio
async def test_start_stop(websocket_server):
    """Test starting and stopping the server."""
    # Start server
    await websocket_server.start()
    assert websocket_server.is_running
    
    # Stop server
    await websocket_server.stop()
    assert not websocket_server.is_running

@pytest.mark.asyncio
async def test_handle_connection(websocket_server):
    """Test handling new connections."""
    # Mock websocket connection
    websocket = AsyncMock()
    websocket.remote_address = ('127.0.0.1', 12345)
    
    # Handle connection
    await websocket_server.handle_connection(websocket)
    
    # Verify connection handling
    assert len(websocket_server.clients) == 1
    assert websocket in websocket_server.clients
    websocket.send.assert_called_once()

@pytest.mark.asyncio
async def test_handle_disconnection(websocket_server):
    """Test handling disconnections."""
    # Add client
    websocket = AsyncMock()
    websocket.remote_address = ('127.0.0.1', 12345)
    websocket_server.clients.add(websocket)
    
    # Handle disconnection
    await websocket_server.handle_disconnection(websocket)
    
    # Verify disconnection handling
    assert len(websocket_server.clients) == 0
    assert websocket not in websocket_server.clients

@pytest.mark.asyncio
async def test_handle_message(websocket_server, mock_message_handler):
    """Test handling incoming messages."""
    # Mock websocket and message
    websocket = AsyncMock()
    websocket.remote_address = ('127.0.0.1', 12345)
    message = {
        'type': 'trade',
        'data': {
            'symbol': 'BTC/USDT',
            'price': 50000.0,
            'amount': 0.01
        }
    }
    
    # Handle message
    await websocket_server.handle_message(websocket, json.dumps(message))
    
    # Verify message handling
    mock_message_handler.handle_message.assert_called_once_with(message)
    websocket.send.assert_called_once()

@pytest.mark.asyncio
async def test_broadcast_message(websocket_server):
    """Test broadcasting messages to all clients."""
    # Add clients
    client1 = AsyncMock()
    client2 = AsyncMock()
    websocket_server.clients.add(client1)
    websocket_server.clients.add(client2)
    
    # Broadcast message
    message = {'type': 'broadcast', 'data': 'test'}
    await websocket_server.broadcast_message(message)
    
    # Verify broadcast
    client1.send.assert_called_once_with(json.dumps(message))
    client2.send.assert_called_once_with(json.dumps(message))

@pytest.mark.asyncio
async def test_send_message(websocket_server):
    """Test sending message to specific client."""
    # Add client
    websocket = AsyncMock()
    websocket.remote_address = ('127.0.0.1', 12345)
    websocket_server.clients.add(websocket)
    
    # Send message
    message = {'type': 'private', 'data': 'test'}
    await websocket_server.send_message(websocket, message)
    
    # Verify message sending
    websocket.send.assert_called_once_with(json.dumps(message))

@pytest.mark.asyncio
async def test_validate_message(websocket_server):
    """Test message validation."""
    # Valid message
    valid_message = {
        'type': 'trade',
        'data': {
            'symbol': 'BTC/USDT',
            'price': 50000.0
        }
    }
    assert websocket_server.validate_message(json.dumps(valid_message))
    
    # Invalid message (too large)
    invalid_message = 'x' * (websocket_server.max_message_size + 1)
    assert not websocket_server.validate_message(invalid_message)
    
    # Invalid message (invalid JSON)
    assert not websocket_server.validate_message('invalid json')

@pytest.mark.asyncio
async def test_handle_ping(websocket_server):
    """Test handling ping messages."""
    # Add client
    websocket = AsyncMock()
    websocket.remote_address = ('127.0.0.1', 12345)
    websocket_server.clients.add(websocket)
    
    # Send ping
    await websocket_server.handle_ping(websocket)
    
    # Verify ping handling
    websocket.send.assert_called_once_with(json.dumps({'type': 'pong'}))

@pytest.mark.asyncio
async def test_handle_pong(websocket_server):
    """Test handling pong messages."""
    # Add client
    websocket = AsyncMock()
    websocket.remote_address = ('127.0.0.1', 12345)
    websocket_server.clients.add(websocket)
    
    # Send pong
    await websocket_server.handle_pong(websocket)
    
    # Verify pong handling
    assert websocket in websocket_server.clients

@pytest.mark.asyncio
async def test_handle_error(websocket_server):
    """Test handling errors."""
    # Add client
    websocket = AsyncMock()
    websocket.remote_address = ('127.0.0.1', 12345)
    websocket_server.clients.add(websocket)
    
    # Handle error
    error = WebSocketError("Test error")
    await websocket_server.handle_error(websocket, error)
    
    # Verify error handling
    websocket.send.assert_called_once_with(json.dumps({
        'type': 'error',
        'data': str(error)
    }))

@pytest.mark.asyncio
async def test_cleanup(websocket_server):
    """Test cleanup of disconnected clients."""
    # Add clients
    client1 = AsyncMock()
    client2 = AsyncMock()
    websocket_server.clients.add(client1)
    websocket_server.clients.add(client2)
    
    # Simulate client1 as disconnected
    client1.closed = True
    
    # Cleanup
    await websocket_server.cleanup()
    
    # Verify cleanup
    assert len(websocket_server.clients) == 1
    assert client1 not in websocket_server.clients
    assert client2 in websocket_server.clients

@pytest.mark.asyncio
async def test_connection_limit(websocket_server):
    """Test connection limit handling."""
    # Fill up connections
    for i in range(websocket_server.max_connections):
        websocket = AsyncMock()
        websocket.remote_address = (f'127.0.0.{i}', 12345)
        websocket_server.clients.add(websocket)
    
    # Try to add one more connection
    websocket = AsyncMock()
    websocket.remote_address = ('127.0.0.100', 12345)
    
    # Verify connection is rejected
    with pytest.raises(WebSocketError, match="Maximum connections reached"):
        await websocket_server.handle_connection(websocket)

@pytest.mark.asyncio
async def test_origin_validation(websocket_server):
    """Test origin validation."""
    # Mock websocket with origin
    websocket = AsyncMock()
    websocket.remote_address = ('127.0.0.1', 12345)
    websocket.request_headers = {'Origin': 'http://localhost:3000'}
    
    # Test allowed origin
    assert websocket_server.validate_origin(websocket)
    
    # Test disallowed origin
    websocket.request_headers = {'Origin': 'http://malicious.com'}
    assert not websocket_server.validate_origin(websocket)

@pytest.mark.asyncio
async def test_initialize(websocket_server):
    """Test WebSocket server initialization."""
    await websocket_server.initialize(host='127.0.0.1', port=5001)
    assert websocket_server.server is not None
    assert websocket_server.connected_clients == {}
    assert websocket_server.subscriptions == {}
    assert websocket_server.message_queue is not None
    assert websocket_server.running is True
    await websocket_server.stop()

@pytest.mark.asyncio
async def test_initialize_with_custom_config(websocket_server):
    """Test initialization with custom configuration."""
    websocket_server.heartbeat_interval = 15
    await websocket_server.initialize(host='127.0.0.1', port=5002)
    assert websocket_server.server is not None
    assert websocket_server.running is True
    await websocket_server.stop()

@pytest.mark.asyncio
async def test_handle_connect(websocket_server, mock_websocket_infinite):
    task = asyncio.create_task(websocket_server.handle_connect(mock_websocket_infinite, '/'))
    await asyncio.sleep(0.1)
    session_id = str(id(mock_websocket_infinite))
    assert session_id in websocket_server.connected_clients
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

@pytest.mark.asyncio
async def test_handle_connect_with_existing_client(websocket_server, mock_websocket_infinite):
    session_id = str(id(mock_websocket_infinite))
    websocket_server.connected_clients[session_id] = mock_websocket_infinite
    task = asyncio.create_task(websocket_server.handle_connect(mock_websocket_infinite, '/'))
    await asyncio.sleep(0.1)
    assert session_id in websocket_server.connected_clients
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

@pytest.mark.asyncio
async def test_handle_disconnect(websocket_server, mock_websocket):
    """Test client disconnection handling."""
    session_id = str(id(mock_websocket))
    websocket_server.connected_clients[session_id] = mock_websocket
    await websocket_server.handle_disconnect(mock_websocket)
    assert session_id not in websocket_server.connected_clients

@pytest.mark.asyncio
async def test_handle_subscribe(websocket_server, mock_websocket):
    """Test channel subscription handling."""
    session_id = str(id(mock_websocket))
    websocket_server.connected_clients[session_id] = mock_websocket
    channels = ['trades_BTC/USD', 'prices_ETH/USD']
    await websocket_server.handle_subscribe(mock_websocket, channels)
    for channel in channels:
        assert channel in websocket_server.subscriptions
        assert session_id in websocket_server.subscriptions[channel]

@pytest.mark.asyncio
async def test_handle_subscribe_with_invalid_channels(websocket_server, mock_websocket):
    """Test handling subscription with invalid channels."""
    session_id = str(id(mock_websocket))
    websocket_server.connected_clients[session_id] = mock_websocket
    channels = ['invalid_channel', 'trades_BTC/USD']
    await websocket_server.handle_subscribe(mock_websocket, channels)
    assert 'invalid_channel' not in websocket_server.subscriptions
    assert 'trades_BTC/USD' in websocket_server.subscriptions

@pytest.mark.asyncio
async def test_handle_unsubscribe(websocket_server, mock_websocket):
    """Test channel unsubscription handling."""
    session_id = str(id(mock_websocket))
    websocket_server.connected_clients[session_id] = mock_websocket
    channel = 'trades_BTC/USD'
    websocket_server.subscriptions[channel] = {session_id}
    await websocket_server.handle_unsubscribe(mock_websocket, [channel])
    assert session_id not in websocket_server.subscriptions.get(channel, set())

@pytest.mark.asyncio
async def test_handle_unsubscribe_nonexistent_client(websocket_server):
    """Test handling unsubscription of nonexistent client."""
    channel = 'trades_BTC/USD'
    websocket_server.subscriptions[channel] = set()
    await websocket_server.handle_unsubscribe('nonexistent', [channel])
    assert 'nonexistent' not in websocket_server.subscriptions[channel]

@pytest.mark.asyncio
async def test_broadcast_trade(websocket_server, mock_websocket):
    """Test trade broadcasting."""
    session_id = str(id(mock_websocket))
    channel = 'trades_BTC/USD'
    trade_data = {'symbol': 'BTC/USD', 'price': 50000}
    
    websocket_server.subscriptions[channel] = {session_id}
    websocket_server.connected_clients[session_id] = mock_websocket
    
    await websocket_server.broadcast_trade(trade_data)
    mock_websocket.send.assert_called_once()
    sent_message = json.loads(mock_websocket.send.call_args[0][0])
    assert sent_message['type'] == 'trade'
    assert sent_message['symbol'] == trade_data['symbol']
    assert sent_message['price'] == trade_data['price']

@pytest.mark.asyncio
async def test_broadcast_trade_with_no_subscribers(websocket_server):
    """Test trade broadcasting with no subscribers."""
    trade_data = {'symbol': 'BTC/USD', 'price': 50000}
    await websocket_server.broadcast_trade(trade_data)
    # Should not raise any exceptions

@pytest.mark.asyncio
async def test_broadcast_price(websocket_server, mock_websocket):
    """Test price broadcasting."""
    session_id = 'test_client'
    channel = 'prices_BTC/USD'
    price_data = {'symbol': 'BTC/USD', 'price': 50000}
    
    websocket_server.subscriptions[channel] = {session_id}
    websocket_server.connected_clients[session_id] = mock_websocket
    
    await websocket_server.broadcast_price(price_data)
    mock_websocket.send.assert_called_once()
    sent_message = json.loads(mock_websocket.send.call_args[0][0])
    assert sent_message['type'] == 'price'
    assert sent_message['data'] == price_data

@pytest.mark.asyncio
async def test_broadcast_bot_status(websocket_server, mock_websocket):
    """Test bot status broadcasting."""
    session_id = 'test_client'
    channel = 'status_bot1'
    status_data = {'bot_id': 'bot1', 'status': 'running'}
    
    websocket_server.subscriptions[channel] = {session_id}
    websocket_server.connected_clients[session_id] = mock_websocket
    
    await websocket_server.broadcast_bot_status(status_data)
    mock_websocket.send.assert_called_once()
    sent_message = json.loads(mock_websocket.send.call_args[0][0])
    assert sent_message['type'] == 'status'
    assert sent_message['data'] == status_data

@pytest.mark.asyncio
async def test_broadcast_performance(websocket_server, mock_websocket):
    """Test performance broadcasting."""
    session_id = 'test_client'
    channel = 'performance_user1'
    performance_data = {'user_id': 'user1', 'profit': 1000}
    
    websocket_server.subscriptions[channel] = {session_id}
    websocket_server.connected_clients[session_id] = mock_websocket
    
    await websocket_server.broadcast_performance(performance_data)
    mock_websocket.send.assert_called_once()
    sent_message = json.loads(mock_websocket.send.call_args[0][0])
    assert sent_message['type'] == 'performance'
    assert sent_message['data'] == performance_data

@pytest.mark.asyncio
async def test_validate_subscription(websocket_server):
    """Test subscription validation."""
    assert websocket_server.validate_subscription('trades_BTC/USD')
    assert websocket_server.validate_subscription('prices_ETH/USD')
    assert websocket_server.validate_subscription('status_bot1')
    assert websocket_server.validate_subscription('performance_user1')
    assert not websocket_server.validate_subscription('invalid_channel')
    assert not websocket_server.validate_subscription(123)

@pytest.mark.asyncio
async def test_get_subscribed_clients(websocket_server):
    """Test getting subscribed clients."""
    channel = 'trades_BTC/USD'
    clients = {'client1', 'client2'}
    websocket_server.subscriptions[channel] = clients
    assert websocket_server.get_subscribed_clients(channel) == clients
    assert websocket_server.get_subscribed_clients('nonexistent') == set()

@pytest.mark.asyncio
async def test_start_server(websocket_server):
    """Test server start."""
    await websocket_server.start(host='127.0.0.1', port=5003)
    assert websocket_server.server is not None
    assert websocket_server.running is True
    await websocket_server.stop()

@pytest.mark.asyncio
async def test_stop_server(websocket_server):
    """Test server stop."""
    await websocket_server.start(host='127.0.0.1', port=5004)
    await websocket_server.stop()
    assert not websocket_server.running

@pytest.mark.asyncio
async def test_handle_ping(websocket_server, mock_websocket):
    """Test ping handling."""
    session_id = str(id(mock_websocket))
    websocket_server.connected_clients[session_id] = mock_websocket
    await websocket_server.handle_ping(mock_websocket)
    mock_websocket.send.assert_called_once()
    sent_message = json.loads(mock_websocket.send.call_args[0][0])
    assert sent_message['type'] == 'pong'

@pytest.mark.asyncio
async def test_handle_message(websocket_server, mock_websocket):
    """Test message handling."""
    session_id = str(id(mock_websocket))
    websocket_server.connected_clients[session_id] = mock_websocket
    message = json.dumps({'type': 'ping'})
    await websocket_server.handle_message(mock_websocket, message)
    mock_websocket.send.assert_called_once()
    sent_message = json.loads(mock_websocket.send.call_args[0][0])
    assert sent_message['type'] == 'pong'

@pytest.mark.asyncio
async def test_handle_heartbeat(websocket_server, mock_websocket):
    """Test heartbeat handling."""
    async def mock_send(msg):
        pass
    mock_websocket.send = mock_send
    
    # Start heartbeat in background
    heartbeat_task = asyncio.create_task(websocket_server.handle_heartbeat(mock_websocket))
    
    # Wait for a short time to let heartbeat run
    await asyncio.sleep(0.1)
    
    # Cancel the heartbeat task
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass

@pytest.mark.asyncio
async def test_websocket_message_ordering(websocket_server, mock_websocket_with_message):
    messages = []
    async def mock_send(msg):
        messages.append(json.loads(msg))
    mock_websocket_with_message.send = mock_send
    await websocket_server.handle_connect(mock_websocket_with_message, '/')
    for i in range(3):
        await websocket_server.handle_message(mock_websocket_with_message, json.dumps({'type': 'test', 'data': i}))
    # The first message is a pong, then 3 test messages
    assert len(messages) == 4
    assert messages[1]['type'] == 'test'
    assert messages[2]['type'] == 'test'
    assert messages[3]['type'] == 'test'

@pytest.mark.asyncio
async def test_websocket_message_filtering(websocket_server, mock_websocket_with_message):
    """Test that messages are properly filtered."""
    messages = []
    async def mock_send(msg):
        messages.append(json.loads(msg))
    mock_websocket_with_message.send = mock_send
    await websocket_server.handle_connect(mock_websocket_with_message, '/')
    await websocket_server.handle_subscribe(mock_websocket_with_message, ['trades_BTC'])
    await websocket_server.broadcast_trade({'id': 1, 'price': 100, 'symbol': 'BTC'})
    await websocket_server.broadcast_price({'symbol': 'BTC', 'price': 50000})
    assert len(messages) == 1

@pytest.mark.asyncio
async def test_websocket_message_retry(websocket_server):
    websocket = AsyncMock()
    async def message_gen():
        yield json.dumps({'type': 'ping'})
    websocket.__aiter__.side_effect = message_gen
    retry_count = 0
    async def mock_send(msg):
        nonlocal retry_count
        if retry_count < 2:
            retry_count += 1
            raise Exception("Temporary error")
        return True
    websocket.send = mock_send
    session_id = str(id(websocket))
    websocket_server.connected_clients[session_id] = websocket
    websocket_server.subscriptions['trades_BTC'] = {session_id}
    await websocket_server.broadcast_trade({'id': 1, 'price': 100, 'symbol': 'BTC'})
    assert retry_count == 2

@pytest.mark.asyncio
async def test_websocket_connection_limits(websocket_server, mock_websocket_infinite):
    """Deterministic test: Ensure server enforces max connection limit.
    The first 100 connections should succeed, the 101st should raise an exception.
    """
    websockets = []
    # Connect 100 clients sequentially
    for _ in range(100):
        ws = AsyncMock()
        async def message_gen():
            while True:
                await asyncio.sleep(0.1)
                yield json.dumps({'type': 'ping'})
        ws.__aiter__.side_effect = message_gen
        # Await the connection to ensure it's registered
        connect_task = asyncio.create_task(websocket_server.handle_connect(ws, '/'))
        await asyncio.sleep(0.05)  # Give time for registration
        websockets.append((ws, connect_task))
    # 101st connection should fail
    extra_ws = AsyncMock()
    async def message_gen():
        while True:
            await asyncio.sleep(0.1)
            yield json.dumps({'type': 'ping'})
    extra_ws.__aiter__.side_effect = message_gen
    with pytest.raises(Exception):
        await asyncio.wait_for(websocket_server.handle_connect(extra_ws, '/'), timeout=2)
    # Cancel all tasks
    for ws, task in websockets:
        task.cancel()
    for ws, task in websockets:
        with pytest.raises(asyncio.CancelledError):
            await task

@pytest.mark.asyncio
async def test_websocket_reconnection(websocket_server, mock_websocket_infinite):
    task = asyncio.create_task(websocket_server.handle_connect(mock_websocket_infinite, '/'))
    await asyncio.sleep(0.1)
    session_id = str(id(mock_websocket_infinite))
    assert session_id in websocket_server.connected_clients
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

@pytest.mark.asyncio
async def test_websocket_message_validation(websocket_server, mock_websocket_with_message):
    # Test invalid message format
    mock_websocket_with_message.send_json = AsyncMock()
    await websocket_server.handle_message(mock_websocket_with_message, "invalid json")
    mock_websocket_with_message.send_json.assert_called_once_with({
        "type": "error",
        "message": "Invalid message format"
    })

@pytest.mark.asyncio
async def test_handle_invalid_json(websocket_server, mock_websocket):
    """Test handling of invalid JSON messages.
    
    This test verifies that:
    1. Invalid JSON messages are properly caught
    2. An error message is sent back to the client
    3. The connection remains open for further messages
    """
    # Create a mock WebSocket connection
    mock_websocket.send_json = AsyncMock()
    
    # Test with invalid JSON
    await websocket_server.handle_message(mock_websocket, "invalid json")
    
    # Verify error response was sent
    mock_websocket.send_json.assert_called_once_with({
        "type": "error",
        "message": "Invalid message format"
    })

# Commented out legacy notification_system-based tests to avoid fixture errors
# def test_websocket_message_ordering(notification_system, mock_websocket):
#     pytest.skip('Skipping advanced WebSocket logic for deployment readiness')

# def test_websocket_message_filtering(notification_system, mock_websocket):
#     pytest.skip('Skipping advanced WebSocket logic for deployment readiness')

# def test_websocket_message_validation(notification_system, mock_websocket):
#     pytest.skip('Skipping advanced WebSocket logic for deployment readiness')

# def test_websocket_message_retry(notification_system, mock_websocket):
#     pytest.skip('Skipping advanced WebSocket logic for deployment readiness')

# def test_websocket_connection_limits(notification_system):
#     """Test websocket connection limits."""
#     # Try to register many websockets
#     websockets = [Mock() for _ in range(1000)]
#     for ws in websockets:
#         notification_system.register_websocket(ws)
#     
#     # Verify all websockets were registered
#     assert len(notification_system.websockets) == 1000

# def test_websocket_reconnection(notification_system, mock_websocket):
#     """Test websocket reconnection handling."""
#     # Register websocket
#     notification_system.register_websocket(mock_websocket)
#     
#     # Unregister and register again
#     notification_system.unregister_websocket(mock_websocket)
#     notification_system.register_websocket(mock_websocket)
#     
#     # Verify websocket is registered
#     assert mock_websocket in notification_system.websockets 