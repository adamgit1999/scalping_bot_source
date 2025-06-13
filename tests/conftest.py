import pytest
from src.bot_logic import TradingBot
from src.models import User, db
from src.config import Config

@pytest.fixture
def mock_broker():
    """Create a mock broker for testing."""
    class MockBroker:
        def __init__(self):
            self.candles = []
            self.orders = []
            self.balance = 1000.0
            
        def fetch_candles(self, symbol, interval):
            return self.candles
            
        def place_order(self, symbol, side, amount, price):
            order = {
                'id': f"mock_{len(self.orders)}",
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'status': 'FILLED'
            }
            self.orders.append(order)
            return order
            
        def get_balance(self):
            return self.balance
            
    return MockBroker()

@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        'symbol': 'BTC/USDT',
        'interval': '1m',
        'position_size': 0.1,
        'mock_mode': True,
        'auto_withdraw': False
    }

@pytest.fixture
def trading_bot(mock_broker, test_config):
    """Create a trading bot instance for testing."""
    return TradingBot(mock_broker, test_config)

@pytest.fixture
def app_context():
    """Create application context for database operations."""
    from src.app import create_app
    from src.models import db
    app = create_app()
    app.config['TESTING'] = True
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
        email='test@example.com'
    )
    user.set_password('password123')
    db.session.add(user)
    db.session.commit()
    return user 