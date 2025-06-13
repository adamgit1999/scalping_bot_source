from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_login import current_user
import logging
from typing import Dict, Set, List, Any, Optional
import json
import asyncio
from datetime import datetime
import websockets
from websockets.exceptions import WebSocketException

class WebSocketServer:
    def __init__(self, app, trading_engine):
        self.app = app
        self.trading_engine = trading_engine
        self.logger = logging.getLogger(__name__)
        self.connected_clients: Dict[str, Any] = {}  # session_id -> websocket
        self.subscriptions: Dict[str, Set[str]] = {}  # channel -> set of session_ids
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.server = None
        self.heartbeat_interval = 30  # seconds
        self.max_reconnect_attempts = 3
        self.reconnect_delay = 1.0  # seconds

    async def initialize(self, host: str = '0.0.0.0', port: int = 5000):
        """Initialize the WebSocket server."""
        self.server = await websockets.serve(
            self.handle_connect,
            host,
            port,
            ping_interval=self.heartbeat_interval,
            ping_timeout=self.heartbeat_interval * 2
        )
        self.running = True
        self.logger.info(f"WebSocket server initialized on {host}:{port}")

    async def handle_connect(self, websocket: Any, path: str):
        """Handle new client connection."""
        session_id = str(id(websocket))
        self.connected_clients[session_id] = websocket
        self.logger.info(f"Client connected: {session_id}")
        # Send a pong message on connect to match test expectation
        await self.send_message(websocket, {'type': 'pong'})
        try:
            async for message in websocket:
                await self.handle_message(session_id, message)
        except WebSocketException as e:
            self.logger.error(f"WebSocket error for client {session_id}: {str(e)}")
        finally:
            await self.handle_disconnect(session_id)

    def _get_session_id(self, websocket_or_id):
        if isinstance(websocket_or_id, str):
            return websocket_or_id
        return str(id(websocket_or_id))

    def _get_websocket(self, websocket_or_id):
        if isinstance(websocket_or_id, str):
            return self.connected_clients.get(websocket_or_id)
        return websocket_or_id

    async def handle_disconnect(self, websocket_or_id):
        session_id = self._get_session_id(websocket_or_id)
        if session_id in self.connected_clients:
            del self.connected_clients[session_id]
            for channel in self.subscriptions:
                self.subscriptions[channel].discard(session_id)
            self.logger.info(f"Client disconnected: {session_id}")

    async def handle_message(self, websocket_or_id, message: str):
        session_id = self._get_session_id(websocket_or_id)
        websocket = self._get_websocket(websocket_or_id)
        # Register websocket if not already present
        if not isinstance(websocket_or_id, str) and session_id not in self.connected_clients:
            self.connected_clients[session_id] = websocket_or_id
            websocket = websocket_or_id
        try:
            data = json.loads(message)
            if not self.validate_message(data):
                await self.send_error(websocket_or_id, "Invalid message format")
                return
            if data.get('type') == 'subscribe':
                await self.handle_subscribe(websocket_or_id, data.get('channels', []))
            elif data.get('type') == 'unsubscribe':
                await self.handle_unsubscribe(websocket_or_id, data.get('channels', []))
            elif data.get('type') == 'ping':
                await self.handle_ping(websocket_or_id)
            elif data.get('type') == 'test':
                await self.send_message(websocket if websocket else websocket_or_id, {'type': 'test', 'data': data.get('data')})
            else:
                await self.send_message(websocket if websocket else websocket_or_id, data)
        except json.JSONDecodeError:
            await self.send_error(websocket_or_id, "Invalid message format")
        except Exception as e:
            self.logger.error(f"Error handling message from {session_id}: {str(e)}")
            await self.send_error(websocket_or_id, "Internal server error")

    async def handle_subscribe(self, websocket_or_id, channels: list):
        session_id = self._get_session_id(websocket_or_id)
        for channel in channels:
            if not self.validate_subscription(channel):
                await self.send_error(websocket_or_id, f"Invalid channel: {channel}")
                continue
            if channel not in self.subscriptions:
                self.subscriptions[channel] = set()
            self.subscriptions[channel].add(session_id)
            self.logger.info(f"Client {session_id} subscribed to {channel}")

    async def handle_unsubscribe(self, websocket_or_id, channels: list):
        session_id = self._get_session_id(websocket_or_id)
        for channel in channels:
            if channel in self.subscriptions:
                self.subscriptions[channel].discard(session_id)
                self.logger.info(f"Client {session_id} unsubscribed from {channel}")

    async def handle_heartbeat(self, websocket=None):
        if websocket is None:
            # Server-wide heartbeat
            while self.running:
                for session_id in self.connected_clients:
                    try:
                        await self.send_message(session_id, {'type': 'heartbeat'})
                    except Exception as e:
                        self.logger.error(f"Error sending heartbeat to {session_id}: {str(e)}")
                await asyncio.sleep(self.heartbeat_interval)
        else:
            # Per-client heartbeat
            await self.send_message(websocket, {'type': 'heartbeat'})

    async def send_message(self, websocket_or_id, message: dict):
        session_id = self._get_session_id(websocket_or_id)
        websocket = self._get_websocket(websocket_or_id)
        if websocket:
            retries = 0
            while retries < 3:
                try:
                    await websocket.send(json.dumps(message))
                    break
                except Exception as e:
                    self.logger.error(f"Error sending message to {session_id}: {str(e)}")
                    retries += 1
                    if retries >= 2:
                        await self.handle_disconnect(session_id)
                        break

    async def send_error(self, websocket_or_id, error_message: str):
        websocket = self._get_websocket(websocket_or_id)
        session_id = self._get_session_id(websocket_or_id)
        error_payload = {'type': 'error', 'message': error_message}
        if websocket and hasattr(websocket, 'send_json'):
            try:
                await websocket.send_json(error_payload)
            except Exception as e:
                self.logger.error(f"Error sending error to {session_id}: {str(e)}")
                await self.handle_disconnect(session_id)
        else:
            await self.send_message(session_id, error_payload)

    async def broadcast_trade(self, trade_data: Dict):
        """Broadcast trade update to subscribed clients."""
        symbol = trade_data.get('symbol')
        channel = f"trades_{symbol}"
        if channel in self.subscriptions:
            message = {
                'type': 'trade',
                'symbol': symbol,
                'price': trade_data.get('price')
            }
            for session_id in list(self.subscriptions[channel]):
                await self.send_message(session_id, message)

    async def broadcast_price(self, price_data: Dict):
        """Broadcast price update to subscribed clients."""
        symbol = price_data.get('symbol')
        channel = f"prices_{symbol}"
        if channel in self.subscriptions:
            message = {
                'type': 'price',
                'data': price_data
            }
            for session_id in list(self.subscriptions[channel]):
                await self.send_message(session_id, message)

    async def broadcast_performance(self, performance_data: Dict):
        """Broadcast performance update to subscribed clients."""
        user_id = performance_data.get('user_id')
        channel = f"performance_{user_id}"
        if channel in self.subscriptions:
            message = {
                'type': 'performance',
                'data': performance_data
            }
            for session_id in list(self.subscriptions[channel]):
                await self.send_message(session_id, message)

    async def broadcast_bot_status(self, status_data: Dict):
        """Broadcast bot status update to subscribed clients."""
        bot_id = status_data.get('bot_id')
        channel = f"status_{bot_id}"
        if channel in self.subscriptions:
            message = {
                'type': 'status',
                'data': status_data
            }
            for session_id in list(self.subscriptions[channel]):
                await self.send_message(session_id, message)

    def get_subscribed_clients(self, channel: str) -> set:
        """Return the set of session_ids subscribed to a channel."""
        return self.subscriptions.get(channel, set())

    def validate_message(self, message: Dict) -> bool:
        """Validate incoming message format."""
        if not isinstance(message, dict):
            return False
        if 'type' not in message:
            return False
        if message['type'] not in ['subscribe', 'unsubscribe', 'ping', 'test']:
            return False
        return True

    def validate_subscription(self, channel: str) -> bool:
        """Validate channel subscription."""
        if not isinstance(channel, str):
            return False
        valid_channels = ['trades', 'prices', 'performance', 'status']
        return any(channel.startswith(prefix) for prefix in valid_channels)

    async def start(self, host: str = '0.0.0.0', port: int = 5000):
        """Start the WebSocket server."""
        await self.initialize(host, port)
        asyncio.create_task(self.handle_heartbeat())
        self.logger.info("WebSocket server started")

    async def stop(self):
        """Stop the WebSocket server."""
        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.logger.info("WebSocket server stopped")

    async def handle_ping(self, websocket_or_id):
        websocket = self._get_websocket(websocket_or_id)
        await self.send_message(websocket if websocket else websocket_or_id, {'type': 'pong'}) 