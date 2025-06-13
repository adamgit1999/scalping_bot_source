from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime, timezone
import json
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    trades = db.relationship('Trade', backref='user', lazy=True)
    notifications = db.relationship('Notification', backref='user', lazy=True)
    webhooks = db.relationship('Webhook', backref='user', lazy=True)
    strategies = db.relationship('Strategy', backref='user', lazy=True)
    backtests = db.relationship('Backtest', backref='user', lazy=True)

    def generate_token(self, expires_in=3600):
        """Generate JWT token for API authentication."""
        from datetime import datetime, timedelta
        import jwt
        from src.config import Config
        
        payload = {
            'user_id': self.id,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in)
        }
        return jwt.encode(payload, Config.SECRET_KEY, algorithm='HS256')

    def verify_password(self, password):
        """Verify user password."""
        from werkzeug.security import check_password_hash
        return check_password_hash(self.password_hash, password)

    def set_password(self, password):
        """Set user password."""
        from werkzeug.security import generate_password_hash
        self.password_hash = generate_password_hash(password)

    def to_dict(self):
        """Convert user to dictionary."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email
        }

class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    side = db.Column(db.String(4), nullable=False)  # 'buy' or 'sell'
    price = db.Column(db.Float, nullable=False)
    quantity = db.Column(db.Float, nullable=False)
    total = db.Column(db.Float, nullable=False)
    fee = db.Column(db.Float, nullable=False)
    profit = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    strategy_id = db.Column(db.Integer, db.ForeignKey('strategies.id'))
    order_id = db.Column(db.String(50))
    
    def to_dict(self):
        """Convert trade to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'symbol': self.symbol,
            'side': self.side,
            'price': self.price,
            'quantity': self.quantity,
            'total': self.total,
            'fee': self.fee,
            'profit': self.profit,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'strategy_id': self.strategy_id,
            'order_id': self.order_id
        }

class Notification(db.Model):
    __tablename__ = 'notifications'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    type = db.Column(db.String(50), nullable=False)
    message = db.Column(db.String(500), nullable=False)
    notification_data = db.Column(db.JSON)
    read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __init__(self, user_id, type, message, notification_data=None):
        if type not in ['trade', 'balance', 'error', 'system']:
            raise ValueError("Invalid notification type")
        self.user_id = user_id
        self.type = type
        self.message = message
        self.notification_data = notification_data or {}
    
    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'message': self.message,
            'notification_data': self.notification_data,
            'read': self.read,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Webhook(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    url = db.Column(db.String(255), nullable=False)
    events = db.Column(db.JSON, nullable=False)  # List of event types
    secret = db.Column(db.String(128), nullable=False)
    active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    
    def to_dict(self):
        """Convert webhook to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'url': self.url,
            'events': self.events,
            'active': self.active,
            'created_at': self.created_at.isoformat()
        }

class Strategy(db.Model):
    __tablename__ = 'strategies'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(500))
    parameters = db.Column(db.JSON, nullable=False)
    active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
            'active': self.active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Backtest(db.Model):
    __tablename__ = 'backtests'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    strategy_id = db.Column(db.Integer, db.ForeignKey('strategies.id'))
    start_date = db.Column(db.DateTime, nullable=False)
    end_date = db.Column(db.DateTime, nullable=False)
    initial_balance = db.Column(db.Float, nullable=False)
    final_balance = db.Column(db.Float)
    total_trades = db.Column(db.Integer)
    win_rate = db.Column(db.Float)
    profit_factor = db.Column(db.Float)
    max_drawdown = db.Column(db.Float)
    sharpe_ratio = db.Column(db.Float)
    results = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            'id': self.id,
            'strategy_id': self.strategy_id,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'initial_balance': self.initial_balance,
            'final_balance': self.final_balance,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'results': self.results,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Order(db.Model):
    __tablename__ = 'orders'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    order_type = db.Column(db.String(20), nullable=False)  # 'market', 'limit', 'stop'
    side = db.Column(db.String(4), nullable=False)  # 'buy' or 'sell'
    price = db.Column(db.Float)
    quantity = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), nullable=False)  # 'new', 'filled', 'canceled', 'rejected'
    filled_quantity = db.Column(db.Float, default=0)
    average_price = db.Column(db.Float)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    strategy_id = db.Column(db.Integer, db.ForeignKey('strategies.id'))

    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'order_type': self.order_type,
            'side': self.side,
            'price': self.price,
            'quantity': self.quantity,
            'status': self.status,
            'filled_quantity': self.filled_quantity,
            'average_price': self.average_price,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'strategy_id': self.strategy_id
        }

class Position(db.Model):
    __tablename__ = 'positions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    quantity = db.Column(db.Float, nullable=False)
    entry_price = db.Column(db.Float, nullable=False)
    current_price = db.Column(db.Float)
    unrealized_pnl = db.Column(db.Float)
    realized_pnl = db.Column(db.Float, default=0)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    strategy_id = db.Column(db.Integer, db.ForeignKey('strategies.id'))

    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'strategy_id': self.strategy_id
        } 