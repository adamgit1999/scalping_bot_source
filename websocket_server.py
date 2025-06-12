from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_login import current_user
import logging
from typing import Dict, Set
import json
import asyncio
from datetime import datetime

class WebSocketServer:
    def __init__(self, app, trading_engine):
        self.socketio = SocketIO(app, cors_allowed_origins="*")
        self.trading_engine = trading_engine
        self.logger = logging.getLogger(__name__)
        self.subscribed_symbols: Dict[str, Set[str]] = {}  # symbol -> set of session_ids
        self.user_rooms: Dict[str, Set[str]] = {}  # user_id -> set of room_ids
        
        # Register event handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register WebSocket event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            if not current_user.is_authenticated:
                return False
            self.logger.info(f"Client connected: {current_user.id}")
            return True

        @self.socketio.on('disconnect')
        def handle_disconnect():
            if current_user.is_authenticated:
                self._cleanup_user_rooms(current_user.id)
                self.logger.info(f"Client disconnected: {current_user.id}")

        @self.socketio.on('subscribe_trades')
        def handle_trade_subscription(data):
            if not current_user.is_authenticated:
                return
            
            symbol = data.get('symbol')
            if not symbol:
                return
            
            room = f"trades_{symbol}"
            join_room(room)
            self._add_user_room(current_user.id, room)
            
            if symbol not in self.subscribed_symbols:
                self.subscribed_symbols[symbol] = set()
            self.subscribed_symbols[symbol].add(current_user.id)
            
            self.logger.info(f"User {current_user.id} subscribed to trades for {symbol}")

        @self.socketio.on('unsubscribe_trades')
        def handle_trade_unsubscription(data):
            if not current_user.is_authenticated:
                return
            
            symbol = data.get('symbol')
            if not symbol:
                return
            
            room = f"trades_{symbol}"
            leave_room(room)
            self._remove_user_room(current_user.id, room)
            
            if symbol in self.subscribed_symbols:
                self.subscribed_symbols[symbol].discard(current_user.id)
                if not self.subscribed_symbols[symbol]:
                    del self.subscribed_symbols[symbol]
            
            self.logger.info(f"User {current_user.id} unsubscribed from trades for {symbol}")

        @self.socketio.on('subscribe_prices')
        def handle_price_subscription(data):
            if not current_user.is_authenticated:
                return
            
            symbol = data.get('symbol')
            if not symbol:
                return
            
            room = f"prices_{symbol}"
            join_room(room)
            self._add_user_room(current_user.id, room)
            
            self.logger.info(f"User {current_user.id} subscribed to prices for {symbol}")

        @self.socketio.on('unsubscribe_prices')
        def handle_price_unsubscription(data):
            if not current_user.is_authenticated:
                return
            
            symbol = data.get('symbol')
            if not symbol:
                return
            
            room = f"prices_{symbol}"
            leave_room(room)
            self._remove_user_room(current_user.id, room)
            
            self.logger.info(f"User {current_user.id} unsubscribed from prices for {symbol}")

        @self.socketio.on('subscribe_performance')
        def handle_performance_subscription():
            if not current_user.is_authenticated:
                return
            
            room = f"performance_{current_user.id}"
            join_room(room)
            self._add_user_room(current_user.id, room)
            
            self.logger.info(f"User {current_user.id} subscribed to performance updates")

        @self.socketio.on('unsubscribe_performance')
        def handle_performance_unsubscription():
            if not current_user.is_authenticated:
                return
            
            room = f"performance_{current_user.id}"
            leave_room(room)
            self._remove_user_room(current_user.id, room)
            
            self.logger.info(f"User {current_user.id} unsubscribed from performance updates")

    def _add_user_room(self, user_id: str, room: str):
        """Add a room to a user's subscribed rooms."""
        if user_id not in self.user_rooms:
            self.user_rooms[user_id] = set()
        self.user_rooms[user_id].add(room)

    def _remove_user_room(self, user_id: str, room: str):
        """Remove a room from a user's subscribed rooms."""
        if user_id in self.user_rooms:
            self.user_rooms[user_id].discard(room)

    def _cleanup_user_rooms(self, user_id: str):
        """Clean up all rooms for a user."""
        if user_id in self.user_rooms:
            for room in self.user_rooms[user_id]:
                leave_room(room)
            del self.user_rooms[user_id]

    async def broadcast_trade(self, symbol: str, trade_data: Dict):
        """Broadcast a trade update to all subscribed clients."""
        room = f"trades_{symbol}"
        self.socketio.emit('trade_update', trade_data, room=room)
        self.logger.debug(f"Broadcast trade update for {symbol}: {json.dumps(trade_data)}")

    async def broadcast_price(self, symbol: str, price_data: Dict):
        """Broadcast a price update to all subscribed clients."""
        room = f"prices_{symbol}"
        self.socketio.emit('price_update', price_data, room=room)
        self.logger.debug(f"Broadcast price update for {symbol}: {json.dumps(price_data)}")

    async def broadcast_performance(self, user_id: str, performance_data: Dict):
        """Broadcast performance update to a specific user."""
        room = f"performance_{user_id}"
        self.socketio.emit('performance_update', performance_data, room=room)
        self.logger.debug(f"Broadcast performance update for user {user_id}: {json.dumps(performance_data)}")

    async def broadcast_error(self, user_id: str, error_data: Dict):
        """Broadcast an error message to a specific user."""
        room = f"performance_{user_id}"
        self.socketio.emit('error_update', error_data, room=room)
        self.logger.error(f"Broadcast error for user {user_id}: {json.dumps(error_data)}")

    def start(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Start the WebSocket server."""
        self.socketio.run(self.socketio.app, host=host, port=port, debug=debug) 