import pytest
from app import app, db, User, Strategy, Trade, Backtest, Notification, Webhook
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
import json

@pytest.fixture
def app_context():
    """Create app context and initialize test database."""
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

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
    
    # Test API key generation
    api_key = test_user.generate_api_key()
    assert api_key is not None
    assert len(api_key) == 64
    
    # Test user update
    test_user.email = 'updated@example.com'
    db.session.commit()
    updated_user = User.query.get(test_user.id)
    assert updated_user.email == 'updated@example.com'
    
    # Test user deletion
    db.session.delete(test_user)
    db.session.commit()
    assert User.query.get(test_user.id) is None

def test_strategy_model(app_context, test_user):
    """Test Strategy model functionality."""
    # Create strategy
    strategy = Strategy(
        name='Test Strategy',
        description='Test Description',
        parameters={
            'sma_short': 10,
            'sma_long': 20
        },
        user_id=test_user.id
    )
    db.session.add(strategy)
    db.session.commit()
    
    # Test strategy creation
    assert strategy.name == 'Test Strategy'
    assert strategy.description == 'Test Description'
    assert strategy.parameters['sma_short'] == 10
    assert strategy.user_id == test_user.id
    
    # Test strategy update
    strategy.name = 'Updated Strategy'
    strategy.parameters['sma_short'] = 15
    db.session.commit()
    updated_strategy = Strategy.query.get(strategy.id)
    assert updated_strategy.name == 'Updated Strategy'
    assert updated_strategy.parameters['sma_short'] == 15
    
    # Test strategy deletion
    db.session.delete(strategy)
    db.session.commit()
    assert Strategy.query.get(strategy.id) is None

def test_trade_model(app_context, test_user):
    """Test Trade model functionality."""
    # Create strategy first
    strategy = Strategy(
        name='Test Strategy',
        description='Test Description',
        parameters={},
        user_id=test_user.id
    )
    db.session.add(strategy)
    db.session.commit()
    
    # Create trade
    trade = Trade(
        strategy_id=strategy.id,
        symbol='BTC/USDT',
        side='buy',
        price=50000.0,
        amount=0.1,
        total=5000.0,
        fee=5.0,
        profit=100.0,
        timestamp=datetime.now(datetime.UTC)
    )
    db.session.add(trade)
    db.session.commit()
    
    # Test trade creation
    assert trade.strategy_id == strategy.id
    assert trade.symbol == 'BTC/USDT'
    assert trade.side == 'buy'
    assert trade.price == 50000.0
    assert trade.amount == 0.1
    assert trade.total == 5000.0
    assert trade.fee == 5.0
    assert trade.profit == 100.0
    
    # Test trade update
    trade.profit = 200.0
    trade.status = 'closed'
    db.session.commit()
    updated_trade = Trade.query.get(trade.id)
    assert updated_trade.profit == 200.0
    assert updated_trade.status == 'closed'
    
    # Test trade deletion
    db.session.delete(trade)
    db.session.commit()
    assert Trade.query.get(trade.id) is None

def test_backtest_model(app_context, test_user):
    """Test Backtest model functionality."""
    # Create strategy first
    strategy = Strategy(
        name='Test Strategy',
        description='Test Description',
        parameters={},
        user_id=test_user.id
    )
    db.session.add(strategy)
    db.session.commit()
    
    # Create backtest
    backtest = Backtest(
        strategy_id=strategy.id,
        start_date=datetime.now(datetime.UTC) - timedelta(days=30),
        end_date=datetime.now(datetime.UTC),
        initial_balance=10000.0,
        final_balance=11000.0,
        total_trades=100,
        win_rate=0.6,
        profit_factor=1.5,
        max_drawdown=0.1
    )
    db.session.add(backtest)
    db.session.commit()
    
    # Test backtest creation
    assert backtest.strategy_id == strategy.id
    assert backtest.initial_balance == 10000.0
    assert backtest.final_balance == 11000.0
    assert backtest.total_trades == 100
    assert backtest.win_rate == 0.6
    assert backtest.profit_factor == 1.5
    assert backtest.max_drawdown == 0.1
    
    # Test backtest update
    backtest.final_balance = 12000.0
    backtest.win_rate = 0.7
    db.session.commit()
    updated_backtest = Backtest.query.get(backtest.id)
    assert updated_backtest.final_balance == 12000.0
    assert updated_backtest.win_rate == 0.7
    
    # Test backtest deletion
    db.session.delete(backtest)
    db.session.commit()
    assert Backtest.query.get(backtest.id) is None

def test_notification_model(app_context, test_user):
    """Test Notification model functionality."""
    # Create notification
    notification = Notification(
        user_id=test_user.id,
        type='trade',
        message='Trade executed successfully',
        data={
            'trade_id': 1,
            'symbol': 'BTC/USDT',
            'price': 50000.0
        },
        read=False
    )
    db.session.add(notification)
    db.session.commit()
    
    # Test notification creation
    assert notification.user_id == test_user.id
    assert notification.type == 'trade'
    assert notification.message == 'Trade executed successfully'
    assert notification.data['trade_id'] == 1
    assert not notification.read
    
    # Test notification update
    notification.read = True
    db.session.commit()
    updated_notification = Notification.query.get(notification.id)
    assert updated_notification.read is True
    
    # Test notification deletion
    db.session.delete(notification)
    db.session.commit()
    assert Notification.query.get(notification.id) is None

def test_webhook_model(app_context, test_user):
    """Test Webhook model functionality."""
    # Create webhook
    webhook = Webhook(
        user_id=test_user.id,
        url='https://example.com/webhook',
        events=['trade', 'balance'],
        secret='test_secret',
        active=True
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
    webhook.events.append('order')
    webhook.active = False
    db.session.commit()
    updated_webhook = Webhook.query.get(webhook.id)
    assert 'order' in updated_webhook.events
    assert updated_webhook.active is False
    
    # Test webhook deletion
    db.session.delete(webhook)
    db.session.commit()
    assert Webhook.query.get(webhook.id) is None

def test_model_relationships(app_context, test_user):
    """Test model relationships."""
    # Create strategy
    strategy = Strategy(
        name='Test Strategy',
        description='Test Description',
        parameters={},
        user_id=test_user.id
    )
    db.session.add(strategy)
    db.session.commit()
    
    # Create trades
    trades = []
    for i in range(3):
        trade = Trade(
            strategy_id=strategy.id,
            symbol='BTC/USDT',
            side='buy',
            price=50000.0,
            amount=0.1,
            total=5000.0,
            fee=5.0,
            profit=100.0,
            timestamp=datetime.now(datetime.UTC)
        )
        trades.append(trade)
    db.session.add_all(trades)
    db.session.commit()
    
    # Test user-strategy relationship
    assert len(test_user.strategies) == 1
    assert test_user.strategies[0].name == 'Test Strategy'
    
    # Test strategy-trade relationship
    assert len(strategy.trades) == 3
    assert all(trade.strategy_id == strategy.id for trade in strategy.trades)
    
    # Test cascade deletion
    db.session.delete(strategy)
    db.session.commit()
    assert Strategy.query.get(strategy.id) is None
    assert all(Trade.query.get(trade.id) is None for trade in trades)

def test_model_validation(app_context, test_user):
    """Test model validation."""
    # Test user validation
    with pytest.raises(ValueError):
        User(username='', email='test@example.com')
    
    with pytest.raises(ValueError):
        User(username='testuser', email='invalid-email')
    
    # Test strategy validation
    with pytest.raises(ValueError):
        Strategy(name='', description='Test Description', user_id=test_user.id)
    
    with pytest.raises(ValueError):
        Strategy(name='Test Strategy', description='Test Description', parameters='invalid')
    
    # Test trade validation
    with pytest.raises(ValueError):
        Trade(strategy_id=1, symbol='', side='buy', price=50000.0)
    
    with pytest.raises(ValueError):
        Trade(strategy_id=1, symbol='BTC/USDT', side='invalid', price=50000.0)
    
    with pytest.raises(ValueError):
        Trade(strategy_id=1, symbol='BTC/USDT', side='buy', price=-1)

def test_model_indexes(app_context, test_user):
    """Test model indexes."""
    # Create multiple strategies
    strategies = []
    for i in range(5):
        strategy = Strategy(
            name=f'Strategy {i}',
            description=f'Description {i}',
            parameters={},
            user_id=test_user.id
        )
        strategies.append(strategy)
    db.session.add_all(strategies)
    db.session.commit()
    
    # Test index on user_id
    user_strategies = Strategy.query.filter_by(user_id=test_user.id).all()
    assert len(user_strategies) == 5
    
    # Test index on strategy_id
    strategy = strategies[0]
    trades = []
    for i in range(3):
        trade = Trade(
            strategy_id=strategy.id,
            symbol='BTC/USDT',
            side='buy',
            price=50000.0,
            amount=0.1,
            total=5000.0,
            fee=5.0,
            profit=100.0,
            timestamp=datetime.now(datetime.UTC)
        )
        trades.append(trade)
    db.session.add_all(trades)
    db.session.commit()
    
    strategy_trades = Trade.query.filter_by(strategy_id=strategy.id).all()
    assert len(strategy_trades) == 3

def test_model_serialization(app_context, test_user):
    """Test model serialization."""
    # Create strategy
    strategy = Strategy(
        name='Test Strategy',
        description='Test Description',
        parameters={
            'sma_short': 10,
            'sma_long': 20
        },
        user_id=test_user.id
    )
    db.session.add(strategy)
    db.session.commit()
    
    # Test JSON serialization
    strategy_dict = strategy.to_dict()
    assert isinstance(strategy_dict, dict)
    assert strategy_dict['name'] == 'Test Strategy'
    assert strategy_dict['parameters']['sma_short'] == 10
    
    # Test JSON deserialization
    strategy_json = json.dumps(strategy_dict)
    loaded_dict = json.loads(strategy_json)
    assert loaded_dict['name'] == 'Test Strategy'
    assert loaded_dict['parameters']['sma_short'] == 10 