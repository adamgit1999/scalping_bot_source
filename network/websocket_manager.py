import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Callable
import logging
import websockets.client
import orjson
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty
import aiohttp
import ssl
import certifi

logger = logging.getLogger(__name__)

@dataclass
class WebSocketConfig:
    """WebSocket connection configuration."""
    url: str
    ping_interval: float = 20.0
    ping_timeout: float = 10.0
    close_timeout: float = 10.0
    max_size: int = 2**20  # 1MB
    max_queue: int = 2**10  # 1024 messages
    compression: Optional[str] = None
    ssl_context: Optional[ssl.SSLContext] = None

class WebSocketManager:
    """Manages WebSocket connections for low-latency trading."""
    
    def __init__(self, config: WebSocketConfig):
        """
        Initialize WebSocket manager.
        
        Args:
            config: WebSocket configuration
        """
        self.config = config
        self.connection: Optional[websockets.client.WebSocketClientProtocol] = None
        self.connected = False
        self.reconnect_delay = 1.0
        self.max_reconnect_delay = 30.0
        self.message_queue = Queue(maxsize=config.max_queue)
        self.subscriptions: Dict[str, List[Callable]] = {}
        self.last_ping_time = 0.0
        self.last_pong_time = 0.0
        self.latency_measurements: List[float] = []
        self.running = False
        self.connection_thread = None
        self.processing_thread = None
        
        # Initialize SSL context if not provided
        if not config.ssl_context:
            self.config.ssl_context = ssl.create_default_context(cafile=certifi.where())
            
    async def connect(self) -> None:
        """Establish WebSocket connection."""
        try:
            self.connection = await websockets.connect(
                self.config.url,
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                close_timeout=self.config.close_timeout,
                max_size=self.config.max_size,
                compression=self.config.compression,
                ssl=self.config.ssl_context
            )
            self.connected = True
            self.reconnect_delay = 1.0
            logger.info(f"Connected to WebSocket at {self.config.url}")
            
            # Start message processing
            self.running = True
            self.processing_thread = threading.Thread(target=self._process_messages)
            self.processing_thread.start()
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {str(e)}")
            self.connected = False
            await self._handle_reconnect()
            
    async def _handle_reconnect(self) -> None:
        """Handle reconnection with exponential backoff."""
        if not self.connected:
            try:
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
                await self.connect()
            except Exception as e:
                logger.error(f"Reconnection error: {str(e)}")
                
    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self.running = False
        if self.connection:
            await self.connection.close()
            self.connected = False
            logger.info("Disconnected from WebSocket")
            
    def _process_messages(self) -> None:
        """Process incoming WebSocket messages."""
        while self.running:
            try:
                message = self.message_queue.get(timeout=0.1)
                self._handle_message(message)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                
    def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle incoming WebSocket message."""
        try:
            # Update latency measurements
            if 'timestamp' in message:
                latency = time.time() - message['timestamp']
                self.latency_measurements.append(latency)
                if len(self.latency_measurements) > 100:
                    self.latency_measurements = self.latency_measurements[-100:]
                    
            # Handle pong messages
            if message.get('type') == 'pong':
                self.last_pong_time = time.time()
                return
                
            # Handle subscribed messages
            channel = message.get('channel')
            if channel in self.subscriptions:
                for callback in self.subscriptions[channel]:
                    try:
                        callback(message)
                    except Exception as e:
                        logger.error(f"Error in subscription callback: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")
            
    async def send_message(self, message: Dict[str, Any]) -> None:
        """
        Send message through WebSocket.
        
        Args:
            message: Message to send
        """
        if not self.connected:
            raise ConnectionError("WebSocket not connected")
            
        try:
            # Add timestamp for latency measurement
            message['timestamp'] = time.time()
            
            # Serialize message using orjson for better performance
            data = orjson.dumps(message)
            await self.connection.send(data)
            
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            self.connected = False
            await self._handle_reconnect()
            
    def subscribe(self, channel: str, callback: Callable) -> None:
        """
        Subscribe to a channel.
        
        Args:
            channel: Channel to subscribe to
            callback: Callback function for messages
        """
        if channel not in self.subscriptions:
            self.subscriptions[channel] = []
        self.subscriptions[channel].append(callback)
        
    def unsubscribe(self, channel: str, callback: Callable) -> None:
        """
        Unsubscribe from a channel.
        
        Args:
            channel: Channel to unsubscribe from
            callback: Callback function to remove
        """
        if channel in self.subscriptions:
            self.subscriptions[channel].remove(callback)
            if not self.subscriptions[channel]:
                del self.subscriptions[channel]
                
    def get_latency_stats(self) -> Dict[str, float]:
        """
        Get WebSocket latency statistics.
        
        Returns:
            Dictionary with latency statistics
        """
        if not self.latency_measurements:
            return {
                'current': 0.0,
                'average': 0.0,
                'min': 0.0,
                'max': 0.0
            }
            
        return {
            'current': self.latency_measurements[-1],
            'average': sum(self.latency_measurements) / len(self.latency_measurements),
            'min': min(self.latency_measurements),
            'max': max(self.latency_measurements)
        }
        
    async def ping(self) -> None:
        """Send ping message to check connection."""
        if self.connected:
            try:
                await self.send_message({'type': 'ping'})
                self.last_ping_time = time.time()
            except Exception as e:
                logger.error(f"Error sending ping: {str(e)}")
                self.connected = False
                await self._handle_reconnect()
                
    def is_healthy(self) -> bool:
        """
        Check if WebSocket connection is healthy.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        if not self.connected:
            return False
            
        # Check if we've received a pong recently
        if time.time() - self.last_pong_time > self.config.ping_timeout:
            return False
            
        # Check if we've sent a ping recently
        if time.time() - self.last_ping_time > self.config.ping_interval:
            return False
            
        return True 