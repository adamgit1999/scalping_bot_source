import pytest
import json
import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Any, Dict, List, Optional

from src.notification_system import (
    NotificationSystem, NotificationType, NotificationPriority,
    NotificationChannel, NotificationError, ValidationError, DeliveryError
)
from src.websocket_server import WebSocketServer
from src.trading_engine import TradingEngine
from src.performance_monitoring import PerformanceMonitor

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
def mock_websocket_server():
    """Create a mock websocket server."""
    server = Mock(spec=WebSocketServer)
    server.broadcast_notification = AsyncMock()
    return server

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
def notification_system(mock_trading_engine, mock_websocket_server, mock_performance_monitor):
    """Create a notification system instance with mocked dependencies."""
    system = NotificationSystem()
    system.trading_engine = mock_trading_engine
    system.websocket_server = mock_websocket_server
    system.performance_monitor = mock_performance_monitor
    return system

@pytest.fixture
def notification_system():
    """Create a notification system instance."""
    return NotificationSystem(app)

@pytest.fixture
def mock_websocket():
    """Create a mock websocket connection."""
    ws = AsyncMock()
    ws.__aenter__.return_value = ws
    ws.__aexit__.return_value = None
    return ws

@pytest.fixture
def mock_email_client():
    """Create a mock email client."""
    client = AsyncMock()
    client.send_email = AsyncMock(return_value=True)
    return client

@pytest.fixture
def mock_sms_client():
    """Create a mock SMS client."""
    client = AsyncMock()
    client.send_sms = AsyncMock(return_value=True)
    return client

@pytest.fixture
def sample_trade():
    """Create a sample trade notification."""
    return {
        'id': '12345',
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'price': Decimal('50000.0'),
        'amount': Decimal('0.1'),
        'total': Decimal('5000.0'),
        'fee': Decimal('5.0'),
        'profit': Decimal('100.0'),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'leverage': Decimal('1.0'),
        'funding_payment': Decimal('0.0'),
        'trading_fee': Decimal('5.0')
    }

@pytest.fixture
def sample_performance():
    """Create a sample performance notification."""
    return {
        'total_profit': Decimal('1000.0'),
        'win_rate': Decimal('0.65'),
        'total_trades': 100,
        'active_trades': 5,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'sharpe_ratio': Decimal('1.5'),
        'max_drawdown': Decimal('0.1'),
        'profit_factor': Decimal('2.0'),
        'average_win': Decimal('50.0'),
        'average_loss': Decimal('-25.0'),
        'max_consecutive_wins': 10,
        'max_consecutive_losses': 5,
        'annualized_return': Decimal('0.15'),
        'annualized_volatility': Decimal('0.1')
    }

@pytest.fixture
def sample_notification():
    """Create a sample notification."""
    return {
        'type': NotificationType.TRADE,
        'priority': NotificationPriority.HIGH,
        'title': 'Trade Executed',
        'message': 'BTC/USDT buy order executed at 50000',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'data': {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'price': Decimal('50000'),
            'amount': Decimal('0.1')
        },
        'channels': [NotificationChannel.EMAIL, NotificationChannel.WEBSOCKET],
        'recipients': ['user@example.com'],
        'template': 'trade_notification.html',
        'metadata': {
            'strategy': 'scalping',
            'account_id': '12345',
            'session_id': '67890'
        }
    }

@pytest.mark.asyncio
async def test_notification_system_initialization(notification_system):
    """Test notification system initialization."""
    assert notification_system is not None
    assert notification_system.notifications == []
    assert notification_system.max_notifications == 100
    assert notification_system.retention_days == 7
    assert notification_system.default_channels == [NotificationChannel.EMAIL]
    assert notification_system.template_dir == 'templates/notifications'
    assert notification_system.max_retry_attempts == 3
    assert notification_system.retry_delay == 1.0
    assert notification_system.batch_size == 10
    assert notification_system.rate_limit == 100
    assert notification_system.rate_limit_window == 3600

@pytest.mark.asyncio
async def test_add_notification(notification_system, sample_notification):
    """Test adding a notification."""
    # Test normal notification
    notification_system.add_notification(**sample_notification)
    await asyncio.sleep(0.1)  # Give time for async processing
    
    assert len(notification_system.notifications) == 1
    assert notification_system.notifications[0]['title'] == sample_notification['title']
    assert notification_system.notifications[0]['message'] == sample_notification['message']
    assert notification_system.notifications[0]['type'] == sample_notification['type']
    assert notification_system.notifications[0]['priority'] == sample_notification['priority']
    assert notification_system.notifications[0]['channels'] == sample_notification['channels']
    assert notification_system.notifications[0]['recipients'] == sample_notification['recipients']
    assert notification_system.notifications[0]['template'] == sample_notification['template']
    assert notification_system.notifications[0]['metadata'] == sample_notification['metadata']
    
    # Test invalid notification
    invalid_notification = sample_notification.copy()
    invalid_notification['type'] = 'INVALID_TYPE'
    with pytest.raises(ValidationError, match="Invalid notification type"):
        notification_system.add_notification(**invalid_notification)
    
    # Test missing required fields
    invalid_notification = sample_notification.copy()
    del invalid_notification['title']
    with pytest.raises(ValidationError, match="Missing required fields"):
        notification_system.add_notification(**invalid_notification)

@pytest.mark.asyncio
async def test_get_notifications(notification_system, sample_notification):
    """Test getting notifications."""
    # Add multiple notifications
    for i in range(3):
        notification = sample_notification.copy()
        notification['title'] = f"Trade {i}"
        notification_system.add_notification(**notification)
    
    await asyncio.sleep(0.1)  # Give time for async processing
    
    # Test getting all notifications
    notifications = notification_system.get_notifications()
    assert len(notifications) == 3
    
    # Test getting notifications by type
    trade_notifications = notification_system.get_notifications(notification_type=NotificationType.TRADE)
    assert len(trade_notifications) == 3
    
    # Test getting notifications by priority
    high_priority = notification_system.get_notifications(priority=NotificationPriority.HIGH)
    assert len(high_priority) == 3
    
    # Test getting notifications by channel
    email_notifications = notification_system.get_notifications(channel=NotificationChannel.EMAIL)
    assert len(email_notifications) == 3
    
    # Test getting notifications by time range
    start_time = datetime.now(timezone.utc) - timedelta(hours=1)
    end_time = datetime.now(timezone.utc) + timedelta(hours=1)
    time_range_notifications = notification_system.get_notifications(start_time=start_time, end_time=end_time)
    assert len(time_range_notifications) == 3
    
    # Test getting notifications by metadata
    metadata_notifications = notification_system.get_notifications(metadata={'strategy': 'scalping'})
    assert len(metadata_notifications) == 3

@pytest.mark.asyncio
async def test_clear_notifications(notification_system, sample_notification):
    """Test clearing notifications."""
    notification_system.clear_notifications()  # Ensure test isolation
    # Add notifications
    for i in range(3):
        notification = sample_notification.copy()
        notification['title'] = f"Trade {i}"
        notification_system.add_notification(**notification)
    
    await asyncio.sleep(0.1)  # Give time for async processing
    
    # Clear all notifications
    notification_system.clear_notifications()
    assert len(notification_system.notifications) == 0
    
    # Add notifications again
    for i in range(3):
        notification = sample_notification.copy()
        notification['title'] = f"Trade {i}"
        notification_system.add_notification(**notification)
    
    await asyncio.sleep(0.1)  # Give time for async processing
    
    # Clear notifications by type
    notification_system.clear_notifications(notification_type=NotificationType.TRADE)
    assert len(notification_system.notifications) == 0
    
    # Add notifications again
    for i in range(3):
        notification = sample_notification.copy()
        notification['title'] = f"Trade {i}"
        notification_system.add_notification(**notification)
    
    await asyncio.sleep(0.1)  # Give time for async processing
    
    # Clear notifications by channel
    notification_system.clear_notifications(channel=NotificationChannel.EMAIL)
    assert len(notification_system.notifications) == 0
    
    # Add notifications again
    for i in range(3):
        notification = sample_notification.copy()
        notification['title'] = f"Trade {i}"
        notification_system.add_notification(**notification)
    
    await asyncio.sleep(0.1)  # Give time for async processing
    
    # Clear notifications by time range
    start_time = datetime.now(timezone.utc) - timedelta(hours=1)
    end_time = datetime.now(timezone.utc) + timedelta(hours=1)
    notification_system.clear_notifications(start_time=start_time, end_time=end_time)
    assert len(notification_system.notifications) == 0

@pytest.mark.asyncio
async def test_notification_retention(notification_system):
    """Test notification retention policy."""
    # Add old notification
    old_notification = {
        'type': NotificationType.SYSTEM,
        'priority': NotificationPriority.LOW,
        'title': 'Old Notification',
        'message': 'This is an old notification',
        'timestamp': (datetime.now(timezone.utc) - timedelta(days=notification_system.retention_days + 1)).isoformat(),
        'data': {},
        'channels': [NotificationChannel.EMAIL],
        'recipients': ['user@example.com']
    }
    notification_system.add_notification(**old_notification)
    
    # Add new notification
    new_notification = {
        'type': NotificationType.SYSTEM,
        'priority': NotificationPriority.LOW,
        'title': 'New Notification',
        'message': 'This is a new notification',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'data': {},
        'channels': [NotificationChannel.EMAIL],
        'recipients': ['user@example.com']
    }
    notification_system.add_notification(**new_notification)
    
    await asyncio.sleep(0.1)  # Give time for async processing
    
    # Clean old notifications
    notification_system.clean_old_notifications()
    
    assert len(notification_system.notifications) == 1
    assert notification_system.notifications[0]['title'] == 'New Notification'

@pytest.mark.asyncio
async def test_notification_max_limit(notification_system, sample_notification):
    """Test notification maximum limit."""
    # Add notifications up to max limit
    for i in range(notification_system.max_notifications + 5):
        notification = sample_notification.copy()
        notification['title'] = f"Trade {i}"
        notification_system.add_notification(**notification)
    
    await asyncio.sleep(0.1)  # Give time for async processing
    
    assert len(notification_system.notifications) == notification_system.max_notifications
    assert notification_system.notifications[0]['title'] == f"Trade {5}"  # Oldest ones should be removed

@pytest.mark.asyncio
async def test_websocket_notification(notification_system, mock_websocket, sample_notification):
    """Test sending notification via websocket."""
    # Register websocket
    notification_system.register_websocket(mock_websocket)
    
    # Add notification
    notification_system.add_notification(**sample_notification)
    
    # Verify websocket was called
    await asyncio.sleep(0.1)  # Give time for async processing
    mock_websocket.send.assert_called_once()
    sent_data = json.loads(mock_websocket.send.call_args[0][0])
    assert sent_data['title'] == sample_notification['title']
    assert sent_data['message'] == sample_notification['message']
    
    # Test websocket error
    mock_websocket.send.side_effect = Exception("WebSocket error")
    with pytest.raises(DeliveryError, match="Failed to send websocket notification"):
        await notification_system._send_websocket_notification(sample_notification)

@pytest.mark.asyncio
async def test_email_notification(notification_system, mock_email_client, sample_notification):
    """Test sending notification via email."""
    notification_system.email_client = mock_email_client
    
    # Send email notification
    await notification_system._send_email_notification(sample_notification)
    
    # Verify email was sent
    mock_email_client.send_email.assert_called_once()
    call_args = mock_email_client.send_email.call_args[1]  # Use kwargs instead of args
    assert call_args['recipients'] == sample_notification['recipients']
    assert call_args['subject'] == sample_notification['title']
    
    # Test email error
    mock_email_client.send_email.side_effect = Exception("Email error")
    with pytest.raises(DeliveryError, match="Failed to send email notification"):
        await notification_system._send_email_notification(sample_notification)

@pytest.mark.asyncio
async def test_sms_notification(notification_system, mock_sms_client, sample_notification):
    """Test sending notification via SMS."""
    notification_system.sms_client = mock_sms_client
    
    # Send SMS notification
    await notification_system._send_sms_notification(sample_notification)
    
    # Verify SMS was sent
    mock_sms_client.send_sms.assert_called_once()
    call_args = mock_sms_client.send_sms.call_args[1]  # Use kwargs instead of args
    assert call_args['recipients'] == sample_notification['recipients']
    assert sample_notification['message'] in call_args['message']
    
    # Test SMS error
    mock_sms_client.send_sms.side_effect = Exception("SMS error")
    with pytest.raises(DeliveryError, match="Failed to send SMS notification"):
        await notification_system._send_sms_notification(sample_notification)

@pytest.mark.asyncio
async def test_notification_serialization(notification_system, sample_notification):
    """Test notification serialization."""
    # Add notification
    notification_system.add_notification(**sample_notification)
    await asyncio.sleep(0.1)  # Give time for async processing
    # Get notification
    notification = notification_system.notifications[0]
    # Test serialization
    serialized = notification_system.serialize_notification(notification)
    assert isinstance(serialized, str)
    # Test deserialization
    deserialized = notification_system.deserialize_notification(serialized)
    assert deserialized['title'] == notification['title']
    assert deserialized['message'] == notification['message']
    assert deserialized['type'] == notification['type'].value
    assert deserialized['priority'] == notification['priority'].value
    assert deserialized['channels'] == [c.value for c in notification['channels']]
    assert deserialized['recipients'] == notification['recipients']
    assert deserialized['template'] == notification.get('template')
    assert deserialized['metadata'] == notification.get('metadata')
    # Test invalid serialization
    with pytest.raises(ValidationError, match="Invalid notification format"):
        notification_system.deserialize_notification("invalid_json")

@pytest.mark.asyncio
async def test_notification_validation(notification_system):
    """Test notification validation."""
    # Test valid notification
    valid_notification = {
        'type': NotificationType.TRADE,
        'priority': NotificationPriority.HIGH,
        'title': 'Valid Notification',
        'message': 'This is a valid notification',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'data': {},
        'channels': [NotificationChannel.EMAIL],
        'recipients': ['user@example.com'],
        'template': 'notification.html',
        'metadata': {}
    }
    assert notification_system.validate_notification(valid_notification) is True
    
    # Test invalid notification (missing required field)
    invalid_notification = valid_notification.copy()
    del invalid_notification['title']
    with pytest.raises(ValidationError, match="Missing required fields"):
        notification_system.validate_notification(invalid_notification)
    
    # Test invalid notification (invalid type)
    invalid_notification = valid_notification.copy()
    invalid_notification['type'] = 'INVALID_TYPE'
    with pytest.raises(ValidationError, match="Invalid notification type"):
        notification_system.validate_notification(invalid_notification)
    
    # Test invalid notification (invalid channel)
    invalid_notification = valid_notification.copy()
    invalid_notification['channels'] = ['INVALID_CHANNEL']
    with pytest.raises(ValidationError, match="Invalid notification channel"):
        notification_system.validate_notification(invalid_notification)
    
    # Test invalid notification (invalid recipient)
    invalid_notification = valid_notification.copy()
    invalid_notification['recipients'] = ['invalid_email']
    with pytest.raises(ValidationError, match="Invalid recipient format"):
        notification_system.validate_notification(invalid_notification)

@pytest.mark.asyncio
async def test_notification_priority_ordering(notification_system):
    """Test notification priority ordering."""
    notification_system.clear_notifications()  # Ensure test isolation
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
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data': {},
            'channels': [NotificationChannel.EMAIL],
            'recipients': ['user@example.com']
        }
        notification_system.add_notification(**notification)
    
    await asyncio.sleep(0.1)  # Give time for async processing
    
    # Get notifications
    notifications = notification_system.get_notifications()
    
    # Verify ordering (highest priority first)
    assert notifications[0]['priority'] == NotificationPriority.CRITICAL
    assert notifications[1]['priority'] == NotificationPriority.HIGH
    assert notifications[2]['priority'] == NotificationPriority.MEDIUM
    assert notifications[3]['priority'] == NotificationPriority.LOW

@pytest.mark.asyncio
async def test_notification_type_filtering(notification_system):
    """Test notification type filtering."""
    # Add notifications of different types
    types = [
        NotificationType.TRADE,
        NotificationType.SYSTEM,
        NotificationType.ERROR,
        NotificationType.WARNING
    ]
    
    for notification_type in types:
        notification = {
            'type': notification_type,
            'priority': NotificationPriority.MEDIUM,
            'title': f'{notification_type.name} Notification',
            'message': f'This is a {notification_type.name.lower()} notification',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data': {},
            'channels': [NotificationChannel.EMAIL],
            'recipients': ['user@example.com']
        }
        notification_system.add_notification(**notification)
    
    await asyncio.sleep(0.1)  # Give time for async processing
    
    # Test filtering by type
    trade_notifications = notification_system.get_notifications(notification_type=NotificationType.TRADE)
    assert len(trade_notifications) == 1
    assert trade_notifications[0]['type'] == NotificationType.TRADE
    
    system_notifications = notification_system.get_notifications(notification_type=NotificationType.SYSTEM)
    assert len(system_notifications) == 1
    assert system_notifications[0]['type'] == NotificationType.SYSTEM

@pytest.mark.asyncio
async def test_notification_data_handling(notification_system):
    """Test notification data handling."""
    # Test with complex data
    complex_data = {
        'trade': {
            'id': '12345',
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'price': Decimal('50000.0'),
            'amount': Decimal('0.1'),
            'total': Decimal('5000.0'),
            'fee': Decimal('5.0'),
            'profit': Decimal('100.0'),
            'timestamp': datetime.now(timezone.utc).isoformat()
        },
        'performance': {
            'total_profit': Decimal('1000.0'),
            'win_rate': Decimal('0.65'),
            'total_trades': 100,
            'active_trades': 5
        },
        'metadata': {
            'strategy': 'scalping',
            'account_id': '12345',
            'session_id': '67890'
        }
    }
    
    notification = {
        'type': NotificationType.TRADE,
        'priority': NotificationPriority.HIGH,
        'title': 'Complex Data Notification',
        'message': 'This is a notification with complex data',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'data': complex_data,
        'channels': [NotificationChannel.EMAIL],
        'recipients': ['user@example.com']
    }
    
    notification_system.clear_notifications()  # Ensure test isolation
    notification_system.add_notification(**notification)
    
    await asyncio.sleep(0.1)  # Give time for async processing
    
    # Verify data handling
    stored_notification = notification_system.notifications[0]
    assert stored_notification['data'] == complex_data
    
    # Test data serialization
    serialized = notification_system.serialize_notification(stored_notification)
    deserialized = notification_system.deserialize_notification(serialized)
    # Convert expected data to stringified Decimals for comparison
    def stringify_decimals(obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, dict):
            return {k: stringify_decimals(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [stringify_decimals(x) for x in obj]
        return obj
    expected_data = stringify_decimals(complex_data)
    assert deserialized['data'] == expected_data 

@pytest.mark.asyncio
async def test_trade_notification(notification_system, sample_trade):
    """Test trade notification handling."""
    # Send trade notification
    await notification_system.send_trade_notification(
        user_email='user@example.com',
        trade_data=sample_trade
    )
    
    # Verify notification was sent
    assert len(notification_system.notifications) == 1
    notification = notification_system.notifications[0]
    assert notification['type'] == NotificationType.TRADE
    assert notification['priority'] == NotificationPriority.HIGH
    assert 'BTC/USDT' in notification['title']
    assert notification['channels'] == [NotificationChannel.EMAIL, NotificationChannel.WEBSOCKET]

@pytest.mark.asyncio
async def test_error_notification(notification_system):
    """Test error notification handling."""
    error_data = {
        'type': 'execution_error',
        'message': 'Failed to execute trade',
        'details': {
            'symbol': 'BTC/USDT',
            'order_id': '12345',
            'error_code': 'INSUFFICIENT_BALANCE'
        }
    }
    
    # Send error notification
    await notification_system.send_error_notification(
        user_email='user@example.com',
        error_data=error_data
    )
    
    # Verify notification was sent
    assert len(notification_system.notifications) == 1
    notification = notification_system.notifications[0]
    assert notification['type'] == NotificationType.ERROR
    assert notification['priority'] == NotificationPriority.CRITICAL
    assert 'Error' in notification['title']
    assert notification['channels'] == [NotificationChannel.EMAIL, NotificationChannel.WEBSOCKET]

@pytest.mark.asyncio
async def test_performance_notification(notification_system, sample_performance):
    """Test performance notification handling."""
    # Send performance notification
    await notification_system.send_performance_notification(
        user_email='user@example.com',
        performance_data=sample_performance
    )
    
    # Verify notification was sent
    assert len(notification_system.notifications) == 1
    notification = notification_system.notifications[0]
    assert notification['type'] == NotificationType.PERFORMANCE
    assert notification['priority'] == NotificationPriority.MEDIUM
    assert 'Performance' in notification['title']
    assert notification['channels'] == [NotificationChannel.EMAIL, NotificationChannel.WEBSOCKET]

@pytest.mark.asyncio
async def test_system_notification(notification_system):
    """Test system notification handling."""
    system_data = {
        'type': 'system_update',
        'message': 'System maintenance scheduled',
        'details': {
            'start_time': datetime.now(timezone.utc).isoformat(),
            'duration': '2 hours',
            'affected_services': ['trading', 'market_data']
        }
    }
    
    # Send system notification
    await notification_system.send_system_notification(
        user_email='user@example.com',
        system_data=system_data
    )
    
    # Verify notification was sent
    assert len(notification_system.notifications) == 1
    notification = notification_system.notifications[0]
    assert notification['type'] == NotificationType.SYSTEM
    assert notification['priority'] == NotificationPriority.HIGH
    assert 'System' in notification['title']
    assert notification['channels'] == [NotificationChannel.EMAIL, NotificationChannel.WEBSOCKET]

@pytest.mark.asyncio
async def test_notification_rate_limiting(notification_system, sample_notification):
    """Test notification rate limiting."""
    # Send notifications up to rate limit
    for _ in range(notification_system.rate_limit):
        notification_system.add_notification(**sample_notification)
    
    # Try to send one more notification
    with pytest.raises(DeliveryError, match="Rate limit exceeded"):
        notification_system.add_notification(**sample_notification)

@pytest.mark.asyncio
async def test_notification_retry_mechanism(notification_system, mock_websocket_server, sample_notification):
    """Test notification retry mechanism."""
    # Configure websocket server to fail twice then succeed
    mock_websocket_server.broadcast_notification.side_effect = [
        DeliveryError("Failed to send"),
        DeliveryError("Failed to send"),
        None
    ]
    
    # Send notification
    notification_system.add_notification(**sample_notification)
    await asyncio.sleep(0.1)  # Give time for retries
    
    # Verify retry attempts
    assert mock_websocket_server.broadcast_notification.call_count == 3

@pytest.mark.asyncio
async def test_notification_batching(notification_system, sample_notification):
    """Test notification batching."""
    # Send multiple notifications
    for i in range(notification_system.batch_size + 1):
        notification = sample_notification.copy()
        notification['title'] = f"Notification {i}"
        notification_system.add_notification(**notification)
    
    await asyncio.sleep(0.1)  # Give time for batch processing
    
    # Verify notifications were batched
    assert len(notification_system.notifications) == notification_system.batch_size + 1 