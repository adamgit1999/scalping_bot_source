from enum import Enum
from datetime import datetime, timedelta, timezone
import json
from typing import List, Dict, Any, Optional, Union
import logging
import asyncio
from decimal import Decimal
import re
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

class NotificationType(Enum):
    """Types of notifications."""
    TRADE = "trade"
    SYSTEM = "system"
    ERROR = "error"
    WARNING = "warning"
    PERFORMANCE = "performance"
    ALERT = "alert"

class NotificationPriority(Enum):
    """Priority levels for notifications."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class NotificationChannel(Enum):
    """Available notification channels."""
    EMAIL = "email"
    SMS = "sms"
    WEBSOCKET = "websocket"
    PUSH = "push"

class NotificationError(Exception):
    """Base exception for notification system errors."""
    pass

class ValidationError(NotificationError):
    """Exception raised for validation errors."""
    pass

class DeliveryError(NotificationError):
    """Exception raised for delivery errors."""
    pass

@dataclass
class Notification:
    """Notification data class."""
    type: NotificationType
    priority: NotificationPriority
    title: str
    message: str
    timestamp: str
    data: Dict[str, Any]
    channels: List[NotificationChannel]
    recipients: List[str]
    template: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert notification to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Notification':
        """Create notification from dictionary."""
        return cls(**data)

class NotificationSystem:
    """Notification system for handling various types of notifications."""
    
    def __init__(self):
        """Initialize the notification system."""
        self.notifications: List[Dict[str, Any]] = []
        self.max_notifications: int = 100
        self.retention_days: int = 7
        self.default_channels: List[NotificationChannel] = [NotificationChannel.EMAIL]
        self.template_dir: str = 'templates/notifications'
        self.max_retry_attempts: int = 3
        self.retry_delay: float = 1.0
        self.batch_size: int = 10
        self.rate_limit: int = 100
        self.rate_limit_window: int = 3600
        self.websockets: List[Any] = []
        self.email_client: Optional[Any] = None
        self.sms_client: Optional[Any] = None
        self.push_client: Optional[Any] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def add_notification(self, **kwargs) -> None:
        """Add a new notification."""
        # Validate notification
        self.validate_notification(kwargs)
        
        # Add timestamp if not provided
        if 'timestamp' not in kwargs:
            kwargs['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        # Add default channels if not provided
        if 'channels' not in kwargs:
            kwargs['channels'] = self.default_channels
        
        # Add notification
        self.notifications.append(kwargs)
        
        # Enforce max notifications limit
        if len(self.notifications) > self.max_notifications:
            self.notifications = self.notifications[-self.max_notifications:]
        
        # Send notification through channels
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop is running, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        loop.create_task(self._send_notification(kwargs))

    def get_notifications(
        self,
        notification_type: Optional[NotificationType] = None,
        priority: Optional[NotificationPriority] = None,
        channel: Optional[NotificationChannel] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get notifications with optional filtering."""
        filtered_notifications = self.notifications
        
        if notification_type:
            filtered_notifications = [
                n for n in filtered_notifications
                if n['type'] == notification_type
            ]
        
        if priority:
            filtered_notifications = [
                n for n in filtered_notifications
                if n['priority'] == priority
            ]
        
        if channel:
            filtered_notifications = [
                n for n in filtered_notifications
                if channel in n['channels']
            ]
        
        if start_time:
            filtered_notifications = [
                n for n in filtered_notifications
                if datetime.fromisoformat(n['timestamp']) >= start_time
            ]
        
        if end_time:
            filtered_notifications = [
                n for n in filtered_notifications
                if datetime.fromisoformat(n['timestamp']) <= end_time
            ]
        
        if metadata:
            filtered_notifications = [
                n for n in filtered_notifications
                if all(n['metadata'].get(k) == v for k, v in metadata.items())
            ]
        
        # Sort by priority (highest first)
        filtered_notifications = sorted(filtered_notifications, key=lambda n: n['priority'].value, reverse=True)
        
        return filtered_notifications

    def clear_notifications(
        self,
        notification_type: Optional[NotificationType] = None,
        priority: Optional[NotificationPriority] = None,
        channel: Optional[NotificationChannel] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Clear notifications based on filters. If no filters, clear all."""
        if not any([notification_type, priority, channel, start_time, end_time, metadata]):
            self.notifications.clear()
            return
        self.notifications = [n for n in self.notifications if not (
            (notification_type and n['type'] == notification_type) or
            (priority and n['priority'] == priority) or
            (channel and channel in n['channels']) or
            (start_time and datetime.fromisoformat(n['timestamp']) >= start_time) or
            (end_time and datetime.fromisoformat(n['timestamp']) <= end_time) or
            (metadata and all(n['metadata'].get(k) == v for k, v in metadata.items()))
        )]

    def clean_old_notifications(self) -> None:
        """Remove notifications older than retention period."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        self.notifications = [
            n for n in self.notifications
            if datetime.fromisoformat(n['timestamp']) > cutoff_time
        ]

    def register_websocket(self, websocket: Any) -> None:
        """Register a new websocket connection."""
        self.websockets.append(websocket)

    def unregister_websocket(self, websocket: Any) -> None:
        """Unregister a websocket connection."""
        if websocket in self.websockets:
            self.websockets.remove(websocket)

    async def _send_notification(self, notification: Dict[str, Any]) -> None:
        """Send notification through all specified channels."""
        for channel in notification['channels']:
            try:
                if channel == NotificationChannel.WEBSOCKET:
                    await self._send_websocket_notification(notification)
                elif channel == NotificationChannel.EMAIL:
                    await self._send_email_notification(notification)
                elif channel == NotificationChannel.SMS:
                    await self._send_sms_notification(notification)
                elif channel == NotificationChannel.PUSH:
                    await self._send_push_notification(notification)
            except Exception as e:
                raise DeliveryError(f"Failed to send {channel.value} notification: {str(e)}")

    async def _send_websocket_notification(self, notification: Dict[str, Any]) -> None:
        """Send notification via websocket."""
        if not self.websockets:
            return
        
        message = self.serialize_notification(notification)
        for websocket in self.websockets:
            try:
                await websocket.send(message)
            except Exception as e:
                raise DeliveryError(f"Failed to send websocket notification: {str(e)}")

    async def _send_email_notification(self, notification: Dict[str, Any]) -> None:
        """Send notification via email."""
        if not self.email_client:
            return
        
        try:
            await self.email_client.send_email(
                recipients=notification['recipients'],
                subject=notification['title'],
                body=notification['message'],
                template=notification.get('template'),
                data=notification.get('data', {})
            )
        except Exception as e:
            raise DeliveryError(f"Failed to send email notification: {str(e)}")

    async def _send_sms_notification(self, notification: Dict[str, Any]) -> None:
        """Send notification via SMS."""
        if not self.sms_client:
            return
        
        try:
            await self.sms_client.send_sms(
                recipients=notification['recipients'],
                message=notification['message']
            )
        except Exception as e:
            raise DeliveryError(f"Failed to send SMS notification: {str(e)}")

    async def _send_push_notification(self, notification: Dict[str, Any]) -> None:
        """Send notification via push notification."""
        if not self.push_client:
            return
        
        try:
            await self.push_client.send_push(
                recipients=notification['recipients'],
                title=notification['title'],
                message=notification['message'],
                data=notification.get('data', {})
            )
        except Exception as e:
            raise DeliveryError(f"Failed to send push notification: {str(e)}")

    def validate_notification(self, notification: Dict[str, Any]) -> bool:
        """Validate notification data."""
        required_fields = ['type', 'priority', 'title', 'message', 'channels', 'recipients']
        for field in required_fields:
            if field not in notification:
                raise ValidationError(f"Missing required fields: {field}")
        
        if not isinstance(notification['type'], NotificationType):
            raise ValidationError("Invalid notification type")
        
        if not isinstance(notification['priority'], NotificationPriority):
            raise ValidationError("Invalid notification priority")
        
        if not isinstance(notification['channels'], list):
            raise ValidationError("Channels must be a list")
        
        for channel in notification['channels']:
            if not isinstance(channel, NotificationChannel):
                raise ValidationError("Invalid notification channel")
        
        if not isinstance(notification['recipients'], list):
            raise ValidationError("Recipients must be a list")
        
        for recipient in notification['recipients']:
            if not self._validate_recipient(recipient):
                raise ValidationError("Invalid recipient format")
        
        return True

    def _validate_recipient(self, recipient: str) -> bool:
        """Validate recipient format."""
        # Email validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, recipient):
            return True
        
        # Phone number validation (basic)
        phone_pattern = r'^\+?1?\d{9,15}$'
        if re.match(phone_pattern, recipient):
            return True
        
        # User ID validation (alphanumeric with optional underscore)
        user_id_pattern = r'^[a-zA-Z0-9_]+$'
        if re.match(user_id_pattern, recipient):
            return True
        
        return False

    def serialize_notification(self, notification: Dict[str, Any]) -> str:
        """Serialize notification to JSON. Decimals are converted to strings."""
        def default(obj):
            if isinstance(obj, Decimal):
                return str(obj)
            if isinstance(obj, (datetime,)):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return obj.value
            return str(obj)
        return json.dumps(notification, default=default)

    def deserialize_notification(self, notification_str: str) -> Dict[str, Any]:
        """Deserialize notification from JSON. Decimals will be strings."""
        try:
            return json.loads(notification_str)
        except Exception:
            raise ValidationError("Invalid notification format") 