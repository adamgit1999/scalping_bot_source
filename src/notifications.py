from enum import Enum
from typing import Dict, List, Optional, Union
from datetime import datetime
import asyncio
import logging
from dataclasses import dataclass
import json

class NotificationType(Enum):
    TRADE = "trade"
    ERROR = "error"
    SYSTEM = "system"
    ALERT = "alert"
    PERFORMANCE = "performance"

@dataclass
class Notification:
    type: NotificationType
    message: str
    timestamp: datetime
    data: Optional[Dict] = None
    priority: int = 0  # 0-5, where 5 is highest priority

class NotificationManager:
    def __init__(self):
        self.notifications: List[Notification] = []
        self.subscribers: Dict[str, List[callable]] = {}
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration for notifications"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    async def notify(self, notification: Notification) -> None:
        """Send a notification to all subscribers"""
        self.notifications.append(notification)
        self.logger.info(f"Notification: {notification.message}")
        
        # Notify subscribers
        for subscriber_list in self.subscribers.values():
            for subscriber in subscriber_list:
                try:
                    await subscriber(notification)
                except Exception as e:
                    self.logger.error(f"Error notifying subscriber: {str(e)}")

    def subscribe(self, notification_type: NotificationType, callback: callable) -> None:
        """Subscribe to notifications of a specific type"""
        if notification_type.value not in self.subscribers:
            self.subscribers[notification_type.value] = []
        self.subscribers[notification_type.value].append(callback)

    def unsubscribe(self, notification_type: NotificationType, callback: callable) -> None:
        """Unsubscribe from notifications of a specific type"""
        if notification_type.value in self.subscribers:
            self.subscribers[notification_type.value].remove(callback)

    def get_notifications(self, 
                         notification_type: Optional[NotificationType] = None,
                         limit: int = 100) -> List[Notification]:
        """Get recent notifications, optionally filtered by type"""
        notifications = self.notifications
        if notification_type:
            notifications = [n for n in notifications if n.type == notification_type]
        return notifications[-limit:]

    def clear_notifications(self) -> None:
        """Clear all stored notifications"""
        self.notifications.clear()

    async def notify_trade(self, 
                          symbol: str, 
                          side: str, 
                          quantity: float, 
                          price: float,
                          order_id: str) -> None:
        """Send a trade notification"""
        notification = Notification(
            type=NotificationType.TRADE,
            message=f"Trade executed: {side} {quantity} {symbol} @ {price}",
            timestamp=datetime.now(),
            data={
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "order_id": order_id
            },
            priority=3
        )
        await self.notify(notification)

    async def notify_error(self, 
                          error_message: str, 
                          error_type: str,
                          details: Optional[Dict] = None) -> None:
        """Send an error notification"""
        notification = Notification(
            type=NotificationType.ERROR,
            message=f"Error: {error_message}",
            timestamp=datetime.now(),
            data={
                "error_type": error_type,
                "details": details or {}
            },
            priority=5
        )
        await self.notify(notification)

    async def notify_system_status(self, 
                                 status: str,
                                 details: Optional[Dict] = None) -> None:
        """Send a system status notification"""
        notification = Notification(
            type=NotificationType.SYSTEM,
            message=f"System Status: {status}",
            timestamp=datetime.now(),
            data=details or {},
            priority=2
        )
        await self.notify(notification)

    async def notify_performance(self, 
                               metrics: Dict[str, float],
                               symbol: Optional[str] = None) -> None:
        """Send a performance metrics notification"""
        notification = Notification(
            type=NotificationType.PERFORMANCE,
            message=f"Performance Update: {json.dumps(metrics)}",
            timestamp=datetime.now(),
            data={
                "metrics": metrics,
                "symbol": symbol
            },
            priority=1
        )
        await self.notify(notification)

    async def notify_alert(self, 
                          alert_message: str,
                          alert_type: str,
                          details: Optional[Dict] = None) -> None:
        """Send an alert notification"""
        notification = Notification(
            type=NotificationType.ALERT,
            message=f"Alert: {alert_message}",
            timestamp=datetime.now(),
            data={
                "alert_type": alert_type,
                "details": details or {}
            },
            priority=4
        )
        await self.notify(notification) 