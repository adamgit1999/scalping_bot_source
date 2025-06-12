import pytest
import asyncio
import time
import json
import ssl
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import websockets.client
from network.websocket_manager import WebSocketManager, WebSocketConfig
from network.exceptions import WebSocketError, ConnectionError, MessageError

@pytest.fixture
def config():
    """Create WebSocket configuration."""
    return WebSocketConfig(
        url="wss://test.example.com/ws",
        ping_interval=1.0,
        ping_timeout=0.5,
        close_timeout=0.5,
        max_queue=1000,
        max_reconnect_attempts=5,
        ssl_context=ssl.create_default_context()
    )

@pytest.fixture
def manager(config):
    """Create WebSocket manager instance."""
    return WebSocketManager(config)

@pytest.fixture
def mock_websocket():
    """Create mock WebSocket connection."""
    ws = AsyncMock()
    ws.__aenter__.return_value = ws
    ws.__aexit__.return_value = None
    return ws

@pytest.mark.asyncio
async def test_initialization(manager, config):
    """Test WebSocket manager initialization."""
    assert manager.config == config
    assert manager.connection is None
    assert not manager.connected
    assert manager.reconnect_delay == 1.0
    assert manager.max_reconnect_delay == 30.0
    assert manager.message_queue.maxsize == config.max_queue
    assert manager.subscriptions == {}
    assert manager.last_ping_time == 0.0
    assert manager.last_pong_time == 0.0
    assert manager.latency_measurements == []
    assert not manager.running
    assert manager.connection_thread is None
    assert manager.processing_thread is None
    assert manager.reconnect_attempts == 0
    assert manager.max_reconnect_attempts == config.max_reconnect_attempts

@pytest.mark.asyncio
async def test_connect(manager, mock_websocket):
    """Test WebSocket connection."""
    with patch('websockets.connect', return_value=mock_websocket):
        await manager.connect()
        assert manager.connected
        assert manager.connection == mock_websocket
        assert manager.running
        assert manager.processing_thread is not None
        assert manager.processing_thread.is_alive()
        assert manager.reconnect_attempts == 0
        assert manager.reconnect_delay == 1.0

@pytest.mark.asyncio
async def test_connect_error(manager):
    """Test WebSocket connection error."""
    with patch('websockets.connect', side_effect=ConnectionError("Connection error")):
        with patch.object(manager, '_handle_reconnect') as mock_reconnect:
            await manager.connect()
            assert not manager.connected
            mock_reconnect.assert_called_once()
            assert manager.reconnect_attempts == 1

@pytest.mark.asyncio
async def test_connect_max_attempts(manager):
    """Test WebSocket connection with max attempts reached."""
    manager.reconnect_attempts = manager.max_reconnect_attempts
    with patch('websockets.connect', side_effect=ConnectionError("Connection error")):
        with pytest.raises(ConnectionError, match="Max reconnection attempts reached"):
            await manager.connect()

@pytest.mark.asyncio
async def test_disconnect(manager, mock_websocket):
    """Test WebSocket disconnection."""
    manager.connection = mock_websocket
    manager.connected = True
    manager.running = True
    
    await manager.disconnect()
    
    assert not manager.connected
    assert not manager.running
    mock_websocket.close.assert_called_once()
    assert manager.connection is None
    assert manager.reconnect_attempts == 0
    assert manager.reconnect_delay == 1.0

@pytest.mark.asyncio
async def test_disconnect_error(manager, mock_websocket):
    """Test WebSocket disconnection error."""
    manager.connection = mock_websocket
    manager.connected = True
    manager.running = True
    mock_websocket.close.side_effect = Exception("Close error")
    
    with pytest.raises(WebSocketError, match="Error during disconnection"):
        await manager.disconnect()

def test_handle_message(manager):
    """Test message handling."""
    # Test latency measurement
    message = {'timestamp': time.time() - 0.1}
    manager._handle_message(message)
    assert len(manager.latency_measurements) == 1
    assert manager.latency_measurements[0] == pytest.approx(0.1, rel=0.1)
    
    # Test pong message
    manager._handle_message({'type': 'pong'})
    assert manager.last_pong_time > 0
    
    # Test subscription callback
    mock_callback = Mock()
    manager.subscribe('test_channel', mock_callback)
    manager._handle_message({'channel': 'test_channel', 'data': 'test'})
    mock_callback.assert_called_once_with({'channel': 'test_channel', 'data': 'test'})
    
    # Test invalid message
    with pytest.raises(MessageError, match="Invalid message format"):
        manager._handle_message("invalid_message")
    
    # Test callback error
    def error_callback(message):
        raise Exception("Callback error")
    
    manager.subscribe('error_channel', error_callback)
    with pytest.raises(MessageError, match="Error in message callback"):
        manager._handle_message({'channel': 'error_channel', 'data': 'test'})

@pytest.mark.asyncio
async def test_send_message(manager, mock_websocket):
    """Test sending messages."""
    manager.connection = mock_websocket
    manager.connected = True
    
    message = {'type': 'test', 'data': 'test_data'}
    await manager.send_message(message)
    
    assert 'timestamp' in message
    mock_websocket.send.assert_called_once_with(json.dumps(message))
    
    # Test binary message
    binary_message = b'test_binary'
    await manager.send_message(binary_message, binary=True)
    mock_websocket.send.assert_called_with(binary_message)

@pytest.mark.asyncio
async def test_send_message_error(manager, mock_websocket):
    """Test sending message error."""
    mock_websocket.send.side_effect = Exception("Send error")
    manager.connection = mock_websocket
    manager.connected = True
    
    with patch.object(manager, '_handle_reconnect') as mock_reconnect:
        with pytest.raises(WebSocketError, match="Error sending message"):
            await manager.send_message({'type': 'test'})
        assert not manager.connected
        mock_reconnect.assert_called_once()

def test_subscribe_unsubscribe(manager):
    """Test subscription management."""
    mock_callback1 = Mock()
    mock_callback2 = Mock()
    
    # Subscribe
    manager.subscribe('test_channel', mock_callback1)
    assert 'test_channel' in manager.subscriptions
    assert mock_callback1 in manager.subscriptions['test_channel']
    
    # Subscribe multiple callbacks
    manager.subscribe('test_channel', mock_callback2)
    assert len(manager.subscriptions['test_channel']) == 2
    
    # Unsubscribe specific callback
    manager.unsubscribe('test_channel', mock_callback1)
    assert 'test_channel' in manager.subscriptions
    assert mock_callback1 not in manager.subscriptions['test_channel']
    assert mock_callback2 in manager.subscriptions['test_channel']
    
    # Unsubscribe all
    manager.unsubscribe_all('test_channel')
    assert 'test_channel' not in manager.subscriptions
    
    # Test invalid channel
    with pytest.raises(ValueError, match="Invalid channel name"):
        manager.subscribe('', mock_callback1)

def test_get_latency_stats(manager):
    """Test latency statistics."""
    # Test with no measurements
    stats = manager.get_latency_stats()
    assert stats['current'] == 0.0
    assert stats['average'] == 0.0
    assert stats['min'] == 0.0
    assert stats['max'] == 0.0
    assert stats['p95'] == 0.0
    assert stats['p99'] == 0.0
    
    # Test with measurements
    manager.latency_measurements = [0.1, 0.2, 0.3, 0.4, 0.5]
    stats = manager.get_latency_stats()
    assert stats['current'] == 0.5
    assert stats['average'] == 0.3
    assert stats['min'] == 0.1
    assert stats['max'] == 0.5
    assert stats['p95'] == 0.45
    assert stats['p99'] == 0.49

@pytest.mark.asyncio
async def test_ping(manager, mock_websocket):
    """Test ping functionality."""
    manager.connection = mock_websocket
    manager.connected = True
    
    await manager.ping()
    
    assert manager.last_ping_time > 0
    mock_websocket.send.assert_called_once_with(json.dumps({'type': 'ping'}))

@pytest.mark.asyncio
async def test_ping_error(manager, mock_websocket):
    """Test ping error handling."""
    manager.connection = mock_websocket
    manager.connected = True
    mock_websocket.send.side_effect = Exception("Ping error")
    
    with pytest.raises(WebSocketError, match="Error sending ping"):
        await manager.ping()

def test_is_healthy(manager):
    """Test health check."""
    # Test when not connected
    assert not manager.is_healthy()
    
    # Test when connected but no ping/pong
    manager.connected = True
    assert not manager.is_healthy()
    
    # Test when healthy
    manager.connected = True
    manager.last_ping_time = time.time()
    manager.last_pong_time = time.time()
    assert manager.is_healthy()
    
    # Test when ping timeout
    manager.last_ping_time = time.time() - manager.config.ping_interval - 1
    assert not manager.is_healthy()
    
    # Test when pong timeout
    manager.last_ping_time = time.time()
    manager.last_pong_time = time.time() - manager.config.ping_timeout - 1
    assert not manager.is_healthy()
    
    # Test when latency too high
    manager.last_ping_time = time.time()
    manager.last_pong_time = time.time()
    manager.latency_measurements = [1.0] * 10  # High latency
    assert not manager.is_healthy()

def test_message_queue_management(manager):
    """Test message queue management."""
    # Test queue size limit
    for i in range(manager.config.max_queue + 1):
        try:
            manager.message_queue.put({'data': f'test_{i}'}, block=False)
        except Exception:
            break
    
    assert manager.message_queue.qsize() == manager.config.max_queue
    
    # Test queue timeout
    with pytest.raises(MessageError, match="Message queue timeout"):
        manager.message_queue.put({'data': 'test'}, block=True, timeout=0.1)

def test_reconnect_handling(manager):
    """Test reconnection handling."""
    # Test exponential backoff
    initial_delay = manager.reconnect_delay
    manager._handle_reconnect()
    assert manager.reconnect_delay == initial_delay * 2
    assert manager.reconnect_attempts == 1
    
    # Test max delay limit
    manager.reconnect_delay = manager.max_reconnect_delay
    manager._handle_reconnect()
    assert manager.reconnect_delay == manager.max_reconnect_delay
    
    # Test reset on successful connection
    manager.reconnect_attempts = 5
    manager.reconnect_delay = 16.0
    manager._reset_reconnect_state()
    assert manager.reconnect_attempts == 0
    assert manager.reconnect_delay == 1.0

def test_ssl_context_initialization(manager):
    """Test SSL context initialization."""
    assert manager.config.ssl_context is not None
    assert isinstance(manager.config.ssl_context, ssl.SSLContext)
    
    # Test custom SSL context
    custom_context = ssl.create_default_context()
    custom_config = WebSocketConfig(
        url="wss://test.example.com/ws",
        ssl_context=custom_context
    )
    custom_manager = WebSocketManager(custom_config)
    assert custom_manager.config.ssl_context == custom_context

@pytest.mark.asyncio
async def test_message_processing(manager, mock_websocket):
    """Test message processing loop."""
    manager.connection = mock_websocket
    manager.connected = True
    manager.running = True
    
    # Mock message receiving
    messages = [
        {'type': 'test1', 'data': 'data1'},
        {'type': 'test2', 'data': 'data2'},
        {'type': 'pong'}
    ]
    mock_websocket.__aiter__.return_value = messages
    
    # Start processing
    with patch.object(manager, '_handle_message') as mock_handle:
        await manager._process_messages()
        assert mock_handle.call_count == 3

@pytest.mark.asyncio
async def test_message_processing_error(manager, mock_websocket):
    """Test message processing error handling."""
    manager.connection = mock_websocket
    manager.connected = True
    manager.running = True
    
    # Mock connection error
    mock_websocket.__aiter__.side_effect = Exception("Connection error")
    
    with patch.object(manager, '_handle_reconnect') as mock_reconnect:
        await manager._process_messages()
        assert not manager.connected
        mock_reconnect.assert_called_once() 