import pytest
from datetime import datetime, timezone, timedelta
from src.models import User, Trade, Notification, Webhook, db
from werkzeug.security import generate_password_hash, check_password_hash
import json

@pytest.fixture
def test_user(app_context):
    """Create a test user."""
    user = User(
        username='testuser',
        email='test@example.com',
        password_hash=generate_password_hash('testpassword')
    )
    db.session.add(user)
    db.session.commit()
    return user

def test_user_model(app_context, test_user):
    """Test User model functionality."""
    # Test user creation
    assert test_user.username == 'testuser'
    assert test_user.email == 'test@example.com'
    assert test_user.password_hash != 'testpassword'
    
    # Test password verification
    assert test_user.verify_password('testpassword') is True
    assert test_user.verify_password('wrongpassword') is False
    
    # Test password update
    test_user.set_password('newpassword')
    assert test_user.verify_password('newpassword') is True
    
    # Test token generation
    token = test_user.generate_token()
    assert token is not None
    
    # Test serialization
    user_dict = test_user.to_dict()
    assert user_dict['username'] == 'testuser'
    assert user_dict['email'] == 'test@example.com'

def test_trade_model(app_context, test_user):
    """Test Trade model functionality."""
    # Create trade
    trade = Trade(
        user_id=test_user.id,
        symbol='BTC/USDT',
        side='buy',
        price=50000.0,
        quantity=0.1,
        total=5000.0,
        fee=5.0,
        profit=100.0
    )
    db.session.add(trade)
    db.session.commit()
    
    # Test trade creation
    assert trade.user_id == test_user.id
    assert trade.symbol == 'BTC/USDT'
    assert trade.side == 'buy'
    assert trade.price == 50000.0
    assert trade.quantity == 0.1
    assert trade.total == 5000.0
    assert trade.fee == 5.0
    assert trade.profit == 100.0
    
    # Test trade update
    trade.profit = 150.0
    db.session.commit()
    updated_trade = Trade.query.get(trade.id)
    assert updated_trade.profit == 150.0
    
    # Test serialization
    trade_dict = trade.to_dict()
    assert trade_dict['symbol'] == 'BTC/USDT'
    assert trade_dict['side'] == 'buy'
    assert trade_dict['price'] == 50000.0
    assert trade_dict['quantity'] == 0.1

def test_notification_model(app_context, test_user):
    """Test Notification model functionality."""
    # Create notification
    notification = Notification(
        user_id=test_user.id,
        type='trade',
        message='Trade executed successfully',
        notification_data={
            'trade_id': 1,
            'symbol': 'BTC/USDT',
            'price': 50000.0
        }
    )
    db.session.add(notification)
    db.session.commit()
    
    # Test notification creation
    assert notification.user_id == test_user.id
    assert notification.type == 'trade'
    assert notification.message == 'Trade executed successfully'
    assert notification.notification_data['trade_id'] == 1
    assert notification.notification_data['symbol'] == 'BTC/USDT'
    assert notification.notification_data['price'] == 50000.0
    assert notification.read is False
    
    # Test notification update
    notification.read = True
    db.session.commit()
    updated_notification = Notification.query.get(notification.id)
    assert updated_notification.read is True
    
    # Test serialization
    notification_dict = notification.to_dict()
    assert notification_dict['type'] == 'trade'
    assert notification_dict['message'] == 'Trade executed successfully'
    assert notification_dict['notification_data']['trade_id'] == 1

def test_webhook_model(app_context, test_user):
    """Test Webhook model functionality."""
    # Create webhook
    webhook = Webhook(
        user_id=test_user.id,
        url='https://example.com/webhook',
        events=['trade', 'balance'],
        secret='test_secret'
    )
    db.session.add(webhook)
    db.session.commit()
    
    # Test webhook creation
    assert webhook.user_id == test_user.id
    assert webhook.url == 'https://example.com/webhook'
    assert 'trade' in webhook.events
    assert 'balance' in webhook.events
    assert webhook.secret == 'test_secret'
    assert webhook.active is True
    
    # Test webhook update
    webhook.events = ['trade', 'balance', 'order']
    webhook.active = False
    db.session.commit()
    updated_webhook = Webhook.query.get(webhook.id)
    assert 'order' in updated_webhook.events
    assert updated_webhook.active is False
    
    # Test serialization
    webhook_dict = webhook.to_dict()
    assert webhook_dict['url'] == 'https://example.com/webhook'
    assert 'trade' in webhook_dict['events']
    assert 'balance' in webhook_dict['events']
    assert 'order' in webhook_dict['events']

def test_model_relationships(app_context, test_user):
    """Test model relationships."""
    # Create trades
    trades = []
    for i in range(3):
        trade = Trade(
            user_id=test_user.id,
            symbol='BTC/USDT',
            side='buy',
            price=50000.0,
            quantity=0.1,
            total=5000.0,
            fee=5.0,
            profit=100.0
        )
        trades.append(trade)
    db.session.add_all(trades)
    db.session.commit()
    
    # Test user-trade relationship
    assert len(test_user.trades) == 3
    for trade in test_user.trades:
        assert trade.user_id == test_user.id
    
    # Create notifications
    notifications = []
    for i in range(2):
        notification = Notification(
            user_id=test_user.id,
            type='trade',
            message=f'Trade {i} executed',
            notification_data={'trade_id': i}
        )
        notifications.append(notification)
    db.session.add_all(notifications)
    db.session.commit()
    
    # Test user-notification relationship
    assert len(test_user.notifications) == 2
    for notification in test_user.notifications:
        assert notification.user_id == test_user.id
    
    # Create webhooks
    webhooks = []
    for i in range(2):
        webhook = Webhook(
            user_id=test_user.id,
            url=f'https://example.com/webhook{i}',
            events=['trade'],
            secret=f'secret{i}'
        )
        webhooks.append(webhook)
    db.session.add_all(webhooks)
    db.session.commit()
    
    # Test user-webhook relationship
    assert len(test_user.webhooks) == 2
    for webhook in test_user.webhooks:
        assert webhook.user_id == test_user.id

def test_model_validation(app_context, test_user):
    """Test model validation."""
    # Test trade validation
    with pytest.raises(ValueError):
        Trade(
            user_id=test_user.id,
            symbol='BTC/USDT',
            side='invalid',  # Invalid side
            price=50000.0,
            quantity=0.1,
            total=5000.0,
            fee=5.0
        )
    
    # Test notification validation
    with pytest.raises(ValueError):
        Notification(
            user_id=test_user.id,
            type='invalid',  # Invalid type
            message='Test message'
        )
    
    # Test webhook validation
    with pytest.raises(ValueError):
        Webhook(
            user_id=test_user.id,
            url='invalid_url',  # Invalid URL
            events=['trade'],
            secret='test_secret'
        )

def test_model_indexes(app_context, test_user):
    """Test model indexes."""
    # Create multiple trades
    trades = []
    for i in range(5):
        trade = Trade(
            user_id=test_user.id,
            symbol='BTC/USDT',
            side='buy',
            price=50000.0,
            quantity=0.1,
            total=5000.0,
            fee=5.0,
            profit=100.0
        )
        trades.append(trade)
    db.session.add_all(trades)
    db.session.commit()
    
    # Test index on user_id
    user_trades = Trade.query.filter_by(user_id=test_user.id).all()
    assert len(user_trades) == 5
    
    # Test index on symbol
    btc_trades = Trade.query.filter_by(symbol='BTC/USDT').all()
    assert len(btc_trades) == 5
    
    # Test index on timestamp
    recent_trades = Trade.query.filter(
        Trade.timestamp >= datetime.now(timezone.utc) - timedelta(hours=1)
    ).all()
    assert len(recent_trades) == 5

def test_model_serialization(app_context, test_user):
    """Test model serialization."""
    # Create trade
    trade = Trade(
        user_id=test_user.id,
        symbol='BTC/USDT',
        side='buy',
        price=50000.0,
        quantity=0.1,
        total=5000.0,
        fee=5.0,
        profit=100.0
    )
    db.session.add(trade)
    db.session.commit()
    
    # Test JSON serialization
    trade_dict = trade.to_dict()
    assert isinstance(trade_dict, dict)
    assert trade_dict['symbol'] == 'BTC/USDT'
    assert trade_dict['side'] == 'buy'
    assert trade_dict['price'] == 50000.0
    assert trade_dict['quantity'] == 0.1
    assert 'timestamp' in trade_dict
    
    # Create notification
    notification = Notification(
        user_id=test_user.id,
        type='trade',
        message='Trade executed',
        notification_data={'trade_id': 1}
    )
    db.session.add(notification)
    db.session.commit()
    
    # Test JSON serialization
    notification_dict = notification.to_dict()
    assert isinstance(notification_dict, dict)
    assert notification_dict['type'] == 'trade'
    assert notification_dict['message'] == 'Trade executed'
    assert notification_dict['notification_data']['trade_id'] == 1
    assert 'created_at' in notification_dict 