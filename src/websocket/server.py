import asyncio
import json
import logging
from typing import Dict, Set, Any, Optional
from datetime import datetime, timezone
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

class WebSocketServer:
    """WebSocket server for real-time trading data."""
    
    def __init__(self, app, trading_engine):
        """Initialize the WebSocket server.
        
        Args:
            app: Flask application
            trading_engine: Trading engine instance
        """
        self.app = app
        self.trading_engine = trading_engine
        self.clients: Dict[str, WebSocketServerProtocol] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # channel -> set of client_ids
        self.running = False
        self.server = None
        
    async def initialize(self, host: str = '0.0.0.0', port: int = 5000):
        """Initialize the WebSocket server.
        
        Args:
            host: Server host
            port: Server port
        """
        from websockets.server import serve
        self.server = await serve(
            self.handle_connect,
            host,
            port,
            ping_interval=20,
            ping_timeout=10
        )
        self.running = True
        logger.info(f"WebSocket server started on {host}:{port}")
        
    async def handle_connect(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            path: Request path
        """
        client_id = id(websocket)
        self.clients[client_id] = websocket
        self.subscriptions[client_id] = set()
        
        try:
            async for message in websocket:
                await self.handle_message(client_id, message)
        except ConnectionClosed:
            pass
        finally:
            await self.handle_disconnect(client_id)
            
    def _get_session_id(self, websocket_or_id):
        """Get session ID from websocket or ID.
        
        Args:
            websocket_or_id: WebSocket connection or client ID
            
        Returns:
            Client ID
        """
        if isinstance(websocket_or_id, WebSocketServerProtocol):
            return id(websocket_or_id)
        return websocket_or_id
        
    def _get_websocket(self, websocket_or_id):
        """Get WebSocket connection from ID.
        
        Args:
            websocket_or_id: WebSocket connection or client ID
            
        Returns:
            WebSocket connection
        """
        if isinstance(websocket_or_id, WebSocketServerProtocol):
            return websocket_or_id
        return self.clients.get(websocket_or_id)
        
    async def handle_disconnect(self, websocket_or_id):
        """Handle client disconnection.
        
        Args:
            websocket_or_id: WebSocket connection or client ID
        """
        client_id = self._get_session_id(websocket_or_id)
        if client_id in self.clients:
            del self.clients[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
            
    async def handle_message(self, websocket_or_id, message: str):
        """Handle incoming WebSocket message.
        
        Args:
            websocket_or_id: WebSocket connection or client ID
            message: Message string
        """
        try:
            data = json.loads(message)
            if not isinstance(data, dict):
                raise ValueError("Invalid message format")
                
            action = data.get('action')
            if not action:
                raise ValueError("Missing action")
                
            if action == 'subscribe':
                await self.handle_subscribe(websocket_or_id, data.get('channels', []))
            elif action == 'unsubscribe':
                await self.handle_unsubscribe(websocket_or_id, data.get('channels', []))
            elif action == 'ping':
                await self.handle_heartbeat(websocket_or_id)
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except json.JSONDecodeError:
            await self.send_error(websocket_or_id, "Invalid JSON")
        except ValueError as e:
            await self.send_error(websocket_or_id, str(e))
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self.send_error(websocket_or_id, "Internal server error")
            
    async def handle_subscribe(self, websocket_or_id, channels: list):
        """Handle channel subscription.
        
        Args:
            websocket_or_id: WebSocket connection or client ID
            channels: List of channels to subscribe to
        """
        client_id = self._get_session_id(websocket_or_id)
        for channel in channels:
            if self.validate_subscription(channel):
                self.subscriptions[client_id].add(channel)
                
    async def handle_unsubscribe(self, websocket_or_id, channels: list):
        """Handle channel unsubscription.
        
        Args:
            websocket_or_id: WebSocket connection or client ID
            channels: List of channels to unsubscribe from
        """
        client_id = self._get_session_id(websocket_or_id)
        for channel in channels:
            self.subscriptions[client_id].discard(channel)
            
    async def handle_heartbeat(self, websocket=None):
        """Handle heartbeat message.
        
        Args:
            websocket: Optional WebSocket connection
        """
        if websocket:
            await self.send_message(websocket, {
                'type': 'pong',
                'timestamp': datetime.now(timezone.utc).timestamp()
            })
            
    async def send_message(self, websocket_or_id, message: dict):
        """Send message to client.
        
        Args:
            websocket_or_id: WebSocket connection or client ID
            message: Message dictionary
        """
        websocket = self._get_websocket(websocket_or_id)
        if websocket:
            try:
                await websocket.send(json.dumps(message))
            except ConnectionClosed:
                await self.handle_disconnect(websocket_or_id)
                
    async def send_error(self, websocket_or_id, error_message: str):
        """Send error message to client.
        
        Args:
            websocket_or_id: WebSocket connection or client ID
            error_message: Error message
        """
        await self.send_message(websocket_or_id, {
            'type': 'error',
            'message': error_message,
            'timestamp': datetime.now(timezone.utc).timestamp()
        })
        
    async def broadcast_trade(self, trade_data: Dict):
        """Broadcast trade data to subscribed clients.
        
        Args:
            trade_data: Trade data dictionary
        """
        message = {
            'type': 'trade',
            'data': trade_data,
            'timestamp': datetime.now(timezone.utc).timestamp()
        }
        await self._broadcast('trades', message)
        
    async def broadcast_price(self, price_data: Dict):
        """Broadcast price data to subscribed clients.
        
        Args:
            price_data: Price data dictionary
        """
        message = {
            'type': 'price',
            'data': price_data,
            'timestamp': datetime.now(timezone.utc).timestamp()
        }
        await self._broadcast('prices', message)
        
    async def broadcast_performance(self, performance_data: Dict):
        """Broadcast performance data to subscribed clients.
        
        Args:
            performance_data: Performance data dictionary
        """
        message = {
            'type': 'performance',
            'data': performance_data,
            'timestamp': datetime.now(timezone.utc).timestamp()
        }
        await self._broadcast('performance', message)
        
    async def broadcast_bot_status(self, status_data: Dict):
        """Broadcast bot status to subscribed clients.
        
        Args:
            status_data: Status data dictionary
        """
        message = {
            'type': 'status',
            'data': status_data,
            'timestamp': datetime.now(timezone.utc).timestamp()
        }
        await self._broadcast('status', message)
        
    def get_subscribed_clients(self, channel: str) -> set:
        """Get clients subscribed to a channel.
        
        Args:
            channel: Channel name
            
        Returns:
            Set of client IDs
        """
        return {
            client_id
            for client_id, channels in self.subscriptions.items()
            if channel in channels
        }
        
    def validate_message(self, message: Dict) -> bool:
        """Validate message format.
        
        Args:
            message: Message dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['type', 'data', 'timestamp']
        return all(field in message for field in required_fields)
        
    def validate_subscription(self, channel: str) -> bool:
        """Validate channel subscription.
        
        Args:
            channel: Channel name
            
        Returns:
            True if valid, False otherwise
        """
        valid_channels = {'trades', 'prices', 'performance', 'status'}
        return channel in valid_channels
        
    async def start(self, host: str = '0.0.0.0', port: int = 5000):
        """Start the WebSocket server.
        
        Args:
            host: Server host
            port: Server port
        """
        await self.initialize(host, port)
        
    async def stop(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.running = False
        
    async def handle_ping(self, websocket_or_id):
        """Handle ping message.
        
        Args:
            websocket_or_id: WebSocket connection or client ID
        """
        await self.send_message(websocket_or_id, {
            'type': 'pong',
            'timestamp': datetime.now(timezone.utc).timestamp()
        })
        
    async def _broadcast(self, channel: str, message: Dict):
        """Broadcast message to subscribed clients.
        
        Args:
            channel: Channel name
            message: Message dictionary
        """
        if not self.validate_message(message):
            logger.error(f"Invalid message format: {message}")
            return
            
        client_ids = self.get_subscribed_clients(channel)
        for client_id in client_ids:
            await self.send_message(client_id, message) 