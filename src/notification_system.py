from flask_mail import Mail, Message
from flask import current_app
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import jinja2
import os
from enum import Enum
import json

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
    """Base exception for notification errors."""
    pass

class ValidationError(NotificationError):
    """Exception for validation errors."""
    pass

class DeliveryError(NotificationError):
    """Exception for delivery errors."""
    pass

class NotificationSystem:
    """System for managing and sending notifications."""
    
    def __init__(self, app):
        """Initialize the notification system.
        
        Args:
            app: Flask application
        """
        self.mail = Mail(app)
        self.logger = logging.getLogger(__name__)
        self.template_loader = jinja2.FileSystemLoader('templates/email')
        self.template_env = jinja2.Environment(loader=self.template_loader)
        self.notifications = []
        self.websockets = set()
        self._load_config()
        
    def _load_config(self):
        """Load notification configuration."""
        self.config = {
            'email': {
                'enabled': True,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': self.mail.app.config.get('MAIL_USERNAME'),
                'password': self.mail.app.config.get('MAIL_PASSWORD')
            },
            'sms': {
                'enabled': False,
                'provider': 'twilio',
                'account_sid': self.mail.app.config.get('TWILIO_ACCOUNT_SID'),
                'auth_token': self.mail.app.config.get('TWILIO_AUTH_TOKEN')
            },
            'websocket': {
                'enabled': True
            },
            'push': {
                'enabled': False,
                'api_key': self.mail.app.config.get('PUSH_API_KEY')
            }
        }
        
    def send_trade_notification(self, user_email: str, trade_data: Dict):
        """Send trade notification.
        
        Args:
            user_email: User email
            trade_data: Trade data dictionary
        """
        notification = {
            'type': NotificationType.TRADE,
            'priority': NotificationPriority.MEDIUM,
            'title': 'Trade Executed',
            'message': f"Trade executed: {trade_data['symbol']} {trade_data['side']} at {trade_data['price']}",
            'data': trade_data,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'channels': [NotificationChannel.EMAIL, NotificationChannel.WEBSOCKET],
            'recipients': [user_email]
        }
        self.add_notification(**notification)
        
    def send_error_notification(self, user_email: str, error_data: Dict):
        """Send error notification.
        
        Args:
            user_email: User email
            error_data: Error data dictionary
        """
        notification = {
            'type': NotificationType.ERROR,
            'priority': NotificationPriority.HIGH,
            'title': 'Error Occurred',
            'message': f"Error: {error_data['message']}",
            'data': error_data,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'channels': [NotificationChannel.EMAIL, NotificationChannel.WEBSOCKET],
            'recipients': [user_email]
        }
        self.add_notification(**notification)
        
    def send_daily_report(self, user_email: str, performance_data: Dict):
        """Send daily performance report.
        
        Args:
            user_email: User email
            performance_data: Performance data dictionary
        """
        notification = {
            'type': NotificationType.PERFORMANCE,
            'priority': NotificationPriority.LOW,
            'title': 'Daily Performance Report',
            'message': f"Daily PnL: {performance_data['pnl']}, Trades: {performance_data['trades']}",
            'data': performance_data,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'channels': [NotificationChannel.EMAIL],
            'recipients': [user_email]
        }
        self.add_notification(**notification)
        
    def send_backtest_report(self, user_email: str, backtest_data: Dict):
        """Send backtest report.
        
        Args:
            user_email: User email
            backtest_data: Backtest data dictionary
        """
        notification = {
            'type': NotificationType.PERFORMANCE,
            'priority': NotificationPriority.MEDIUM,
            'title': 'Backtest Results',
            'message': f"Backtest completed: {backtest_data['strategy']}",
            'data': backtest_data,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'channels': [NotificationChannel.EMAIL],
            'recipients': [user_email]
        }
        self.add_notification(**notification)
        
    def send_welcome_email(self, user_email: str, username: str):
        """Send welcome email.
        
        Args:
            user_email: User email
            username: Username
        """
        notification = {
            'type': NotificationType.SYSTEM,
            'priority': NotificationPriority.LOW,
            'title': 'Welcome to Trading Bot',
            'message': f"Welcome {username}! Your account has been created.",
            'data': {'username': username},
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'channels': [NotificationChannel.EMAIL],
            'recipients': [user_email]
        }
        self.add_notification(**notification)
        
    def send_password_reset(self, user_email: str, reset_token: str):
        """Send password reset email.
        
        Args:
            user_email: User email
            reset_token: Reset token
        """
        notification = {
            'type': NotificationType.SYSTEM,
            'priority': NotificationPriority.HIGH,
            'title': 'Password Reset Request',
            'message': 'Click the link to reset your password',
            'data': {'reset_token': reset_token},
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'channels': [NotificationChannel.EMAIL],
            'recipients': [user_email]
        }
        self.add_notification(**notification)
        
    def add_notification(self, **kwargs):
        """Add a new notification.
        
        Args:
            **kwargs: Notification data
        """
        if not self.validate_notification(kwargs):
            raise ValidationError("Invalid notification data")
            
        self.notifications.append(kwargs)
        self._send_notification(kwargs)
        
    def get_notifications(self, notification_type: Optional[NotificationType] = None,
                         priority: Optional[NotificationPriority] = None,
                         channel: Optional[NotificationChannel] = None,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         metadata: Optional[Dict] = None) -> List[Dict]:
        """Get notifications matching criteria.
        
        Args:
            notification_type: Filter by notification type
            priority: Filter by priority
            channel: Filter by channel
            start_time: Filter by start time
            end_time: Filter by end time
            metadata: Filter by metadata
            
        Returns:
            List of matching notifications
        """
        filtered = self.notifications
        
        if notification_type:
            filtered = [n for n in filtered if n['type'] == notification_type]
            
        if priority:
            filtered = [n for n in filtered if n['priority'] == priority]
            
        if channel:
            filtered = [n for n in filtered if channel in n['channels']]
            
        if start_time:
            filtered = [n for n in filtered if datetime.fromisoformat(n['timestamp']) >= start_time]
            
        if end_time:
            filtered = [n for n in filtered if datetime.fromisoformat(n['timestamp']) <= end_time]
            
        if metadata:
            filtered = [n for n in filtered if all(n['data'].get(k) == v for k, v in metadata.items())]
            
        return filtered
        
    def clear_notifications(self, notification_type: Optional[NotificationType] = None,
                           priority: Optional[NotificationPriority] = None,
                           channel: Optional[NotificationChannel] = None,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           metadata: Optional[Dict] = None) -> None:
        """Clear notifications matching criteria.
        
        Args:
            notification_type: Filter by notification type
            priority: Filter by priority
            channel: Filter by channel
            start_time: Filter by start time
            end_time: Filter by end time
            metadata: Filter by metadata
        """
        to_keep = self.get_notifications(
            notification_type=notification_type,
            priority=priority,
            channel=channel,
            start_time=start_time,
            end_time=end_time,
            metadata=metadata
        )
        self.notifications = to_keep
        
    def clean_old_notifications(self) -> None:
        """Clean old notifications."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        self.notifications = [
            n for n in self.notifications
            if datetime.fromisoformat(n['timestamp']) > cutoff
        ]
        
    def register_websocket(self, websocket: Any) -> None:
        """Register a WebSocket connection.
        
        Args:
            websocket: WebSocket connection
        """
        self.websockets.add(websocket)
        
    def unregister_websocket(self, websocket: Any) -> None:
        """Unregister a WebSocket connection.
        
        Args:
            websocket: WebSocket connection
        """
        self.websockets.discard(websocket)
        
    async def _send_notification(self, notification: Dict) -> None:
        """Send notification through configured channels.
        
        Args:
            notification: Notification dictionary
        """
        for channel in notification['channels']:
            try:
                if channel == NotificationChannel.EMAIL:
                    await self._send_email_notification(notification)
                elif channel == NotificationChannel.SMS:
                    await self._send_sms_notification(notification)
                elif channel == NotificationChannel.WEBSOCKET:
                    await self._send_websocket_notification(notification)
                elif channel == NotificationChannel.PUSH:
                    await self._send_push_notification(notification)
            except Exception as e:
                logger.error(f"Error sending notification through {channel}: {e}")
                
    async def _send_websocket_notification(self, notification: Dict) -> None:
        """Send notification through WebSocket.
        
        Args:
            notification: Notification dictionary
        """
        message = self.serialize_notification(notification)
        for websocket in self.websockets:
            try:
                await websocket.send(message)
            except Exception as e:
                logger.error(f"Error sending WebSocket notification: {e}")
                
    async def _send_email_notification(self, notification: Dict) -> None:
        """Send notification through email.
        
        Args:
            notification: Notification dictionary
        """
        if not self.config['email']['enabled']:
            return
            
        template = self.template_env.get_template('trade_notification.html')
        html_content = template.render(
            trade=notification['data'],
            timestamp=datetime.fromisoformat(notification['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        )
        
        msg = Message(
            subject=f"New Trade: {notification['data']['symbol']} {notification['data']['type'].upper()}",
            sender=self.mail.app.config['MAIL_DEFAULT_SENDER'],
            recipients=[notification['recipients'][0]],
            html=html_content
        )
        
        self.mail.send(msg)
        self.logger.info(f"Trade notification sent to {notification['recipients'][0]}")
        
    async def _send_sms_notification(self, notification: Dict) -> None:
        """Send notification through SMS.
        
        Args:
            notification: Notification dictionary
        """
        if not self.config['sms']['enabled']:
            return
            
        # SMS sending logic here
        pass
        
    async def _send_push_notification(self, notification: Dict) -> None:
        """Send notification through push notification.
        
        Args:
            notification: Notification dictionary
        """
        if not self.config['push']['enabled']:
            return
            
        # Push notification logic here
        pass
        
    def validate_notification(self, notification: Dict) -> bool:
        """Validate notification data.
        
        Args:
            notification: Notification dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['type', 'priority', 'title', 'message', 'timestamp', 'channels', 'recipients']
        if not all(field in notification for field in required_fields):
            return False
            
        if not isinstance(notification['type'], NotificationType):
            return False
            
        if not isinstance(notification['priority'], NotificationPriority):
            return False
            
        if not all(isinstance(channel, NotificationChannel) for channel in notification['channels']):
            return False
            
        if not all(self._validate_recipient(recipient) for recipient in notification['recipients']):
            return False
            
        return True
        
    def _validate_recipient(self, recipient: str) -> bool:
        """Validate notification recipient.
        
        Args:
            recipient: Recipient string
            
        Returns:
            True if valid, False otherwise
        """
        # Basic email validation
        if '@' in recipient:
            return True
            
        # Basic phone number validation
        if recipient.isdigit() and len(recipient) >= 10:
            return True
            
        return False
        
    def serialize_notification(self, notification: Dict) -> str:
        """Serialize notification to JSON string.
        
        Args:
            notification: Notification dictionary
            
        Returns:
            JSON string
        """
        def default(obj):
            if isinstance(obj, (NotificationType, NotificationPriority, NotificationChannel)):
                return obj.value
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
        return json.dumps(notification, default=default)
        
    def deserialize_notification(self, notification_str: str) -> Dict:
        """Deserialize notification from JSON string.
        
        Args:
            notification_str: JSON string
            
        Returns:
            Notification dictionary
        """
        data = json.loads(notification_str)
        data['type'] = NotificationType(data['type'])
        data['priority'] = NotificationPriority(data['priority'])
        data['channels'] = [NotificationChannel(c) for c in data['channels']]
        return data 