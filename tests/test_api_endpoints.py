import pytest
from app import app, db, User, Strategy, Trade, Backtest
from datetime import datetime, timedelta
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
        password_hash='hashed_password'
    )
    db.session.add(user)
    db.session.commit()
    return user

@pytest.fixture
def auth_headers(test_user):
    """Create authentication headers."""
    return {
        'Authorization': f'Bearer {test_user.generate_token()}',
        'Content-Type': 'application/json'
    }

def test_user_endpoints(app_context, test_user, auth_headers):
    """Test user-related API endpoints."""
    client = app.test_client()
    
    # Test user registration
    response = client.post('/api/register', json={
        'username': 'newuser',
        'email': 'new@example.com',
        'password': 'password123'
    })
    assert response.status_code == 201
    assert 'user_id' in response.json
    
    # Test user login
    response = client.post('/api/login', json={
        'username': 'newuser',
        'password': 'password123'
    })
    assert response.status_code == 200
    assert 'token' in response.json
    
    # Test user profile
    response = client.get('/api/user/profile', headers=auth_headers)
    assert response.status_code == 200
    assert response.json['username'] == 'testuser'
    
    # Test user update
    response = client.put('/api/user/profile', json={
        'email': 'updated@example.com'
    }, headers=auth_headers)
    assert response.status_code == 200
    assert response.json['email'] == 'updated@example.com'
    
    # Test password change
    response = client.put('/api/user/password', json={
        'current_password': 'testpassword',
        'new_password': 'newpassword123'
    }, headers=auth_headers)
    assert response.status_code == 200

def test_strategy_endpoints(app_context, test_user, auth_headers):
    """Test strategy-related API endpoints."""
    client = app.test_client()
    
    # Test strategy creation
    response = client.post('/api/strategies', json={
        'name': 'Test Strategy',
        'description': 'Test Description',
        'parameters': {
            'sma_short': 10,
            'sma_long': 20
        }
    }, headers=auth_headers)
    assert response.status_code == 201
    strategy_id = response.json['id']
    
    # Test strategy retrieval
    response = client.get(f'/api/strategies/{strategy_id}', headers=auth_headers)
    assert response.status_code == 200
    assert response.json['name'] == 'Test Strategy'
    
    # Test strategy update
    response = client.put(f'/api/strategies/{strategy_id}', json={
        'name': 'Updated Strategy',
        'parameters': {
            'sma_short': 15,
            'sma_long': 30
        }
    }, headers=auth_headers)
    assert response.status_code == 200
    assert response.json['name'] == 'Updated Strategy'
    
    # Test strategy deletion
    response = client.delete(f'/api/strategies/{strategy_id}', headers=auth_headers)
    assert response.status_code == 204
    
    # Test strategy list
    response = client.get('/api/strategies', headers=auth_headers)
    assert response.status_code == 200
    assert len(response.json) == 0

def test_trade_endpoints(app_context, test_user, auth_headers):
    """Test trade-related API endpoints."""
    client = app.test_client()
    
    # Create a strategy first
    strategy_response = client.post('/api/strategies', json={
        'name': 'Test Strategy',
        'description': 'Test Description',
        'parameters': {}
    }, headers=auth_headers)
    strategy_id = strategy_response.json['id']
    
    # Test trade creation
    response = client.post('/api/trades', json={
        'strategy_id': strategy_id,
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'price': 50000.0,
        'amount': 0.1,
        'total': 5000.0,
        'fee': 5.0
    }, headers=auth_headers)
    assert response.status_code == 201
    trade_id = response.json['id']
    
    # Test trade retrieval
    response = client.get(f'/api/trades/{trade_id}', headers=auth_headers)
    assert response.status_code == 200
    assert response.json['symbol'] == 'BTC/USDT'
    
    # Test trade update
    response = client.put(f'/api/trades/{trade_id}', json={
        'profit': 100.0,
        'status': 'closed'
    }, headers=auth_headers)
    assert response.status_code == 200
    assert response.json['profit'] == 100.0
    
    # Test trade list
    response = client.get('/api/trades', headers=auth_headers)
    assert response.status_code == 200
    assert len(response.json) == 1
    
    # Test trade deletion
    response = client.delete(f'/api/trades/{trade_id}', headers=auth_headers)
    assert response.status_code == 204

def test_backtest_endpoints(app_context, test_user, auth_headers):
    """Test backtest-related API endpoints."""
    client = app.test_client()
    
    # Create a strategy first
    strategy_response = client.post('/api/strategies', json={
        'name': 'Test Strategy',
        'description': 'Test Description',
        'parameters': {}
    }, headers=auth_headers)
    strategy_id = strategy_response.json['id']
    
    # Test backtest creation
    response = client.post('/api/backtests', json={
        'strategy_id': strategy_id,
        'start_date': (datetime.now(datetime.UTC) - timedelta(days=30)).isoformat(),
        'end_date': datetime.now(datetime.UTC).isoformat(),
        'initial_balance': 10000.0
    }, headers=auth_headers)
    assert response.status_code == 201
    backtest_id = response.json['id']
    
    # Test backtest retrieval
    response = client.get(f'/api/backtests/{backtest_id}', headers=auth_headers)
    assert response.status_code == 200
    assert response.json['strategy_id'] == strategy_id
    
    # Test backtest update
    response = client.put(f'/api/backtests/{backtest_id}', json={
        'final_balance': 11000.0,
        'total_trades': 100,
        'win_rate': 0.6
    }, headers=auth_headers)
    assert response.status_code == 200
    assert response.json['win_rate'] == 0.6
    
    # Test backtest list
    response = client.get('/api/backtests', headers=auth_headers)
    assert response.status_code == 200
    assert len(response.json) == 1
    
    # Test backtest deletion
    response = client.delete(f'/api/backtests/{backtest_id}', headers=auth_headers)
    assert response.status_code == 204

def test_market_data_endpoints(app_context, test_user, auth_headers):
    """Test market data-related API endpoints."""
    client = app.test_client()
    
    # Test market data retrieval
    response = client.get('/api/market/price/BTC/USDT', headers=auth_headers)
    assert response.status_code == 200
    assert 'price' in response.json
    
    # Test historical data
    response = client.get('/api/market/history/BTC/USDT', headers=auth_headers)
    assert response.status_code == 200
    assert 'data' in response.json
    
    # Test order book
    response = client.get('/api/market/orderbook/BTC/USDT', headers=auth_headers)
    assert response.status_code == 200
    assert 'bids' in response.json
    assert 'asks' in response.json
    
    # Test trading pairs
    response = client.get('/api/market/pairs', headers=auth_headers)
    assert response.status_code == 200
    assert len(response.json) > 0

def test_balance_endpoints(app_context, test_user, auth_headers):
    """Test balance-related API endpoints."""
    client = app.test_client()
    
    # Test balance retrieval
    response = client.get('/api/balance', headers=auth_headers)
    assert response.status_code == 200
    assert 'total_balance' in response.json
    
    # Test balance history
    response = client.get('/api/balance/history', headers=auth_headers)
    assert response.status_code == 200
    assert 'data' in response.json
    
    # Test deposit address
    response = client.get('/api/balance/deposit/BTC', headers=auth_headers)
    assert response.status_code == 200
    assert 'address' in response.json
    
    # Test withdrawal
    response = client.post('/api/balance/withdraw', json={
        'currency': 'BTC',
        'amount': 0.1,
        'address': 'test_address'
    }, headers=auth_headers)
    assert response.status_code == 200
    assert 'transaction_id' in response.json

def test_webhook_endpoints(app_context, test_user, auth_headers):
    """Test webhook-related API endpoints."""
    client = app.test_client()
    
    # Test webhook creation
    response = client.post('/api/webhooks', json={
        'url': 'https://example.com/webhook',
        'events': ['trade', 'balance']
    }, headers=auth_headers)
    assert response.status_code == 201
    webhook_id = response.json['id']
    
    # Test webhook retrieval
    response = client.get(f'/api/webhooks/{webhook_id}', headers=auth_headers)
    assert response.status_code == 200
    assert response.json['url'] == 'https://example.com/webhook'
    
    # Test webhook update
    response = client.put(f'/api/webhooks/{webhook_id}', json={
        'events': ['trade', 'balance', 'order']
    }, headers=auth_headers)
    assert response.status_code == 200
    assert 'order' in response.json['events']
    
    # Test webhook list
    response = client.get('/api/webhooks', headers=auth_headers)
    assert response.status_code == 200
    assert len(response.json) == 1
    
    # Test webhook deletion
    response = client.delete(f'/api/webhooks/{webhook_id}', headers=auth_headers)
    assert response.status_code == 204

def test_error_handling(app_context, test_user, auth_headers):
    """Test API error handling."""
    client = app.test_client()
    
    # Test invalid endpoint
    response = client.get('/api/invalid', headers=auth_headers)
    assert response.status_code == 404
    
    # Test invalid method
    response = client.put('/api/login', headers=auth_headers)
    assert response.status_code == 405
    
    # Test invalid data
    response = client.post('/api/strategies', json={
        'invalid': 'data'
    }, headers=auth_headers)
    assert response.status_code == 400
    
    # Test unauthorized access
    response = client.get('/api/user/profile')
    assert response.status_code == 401
    
    # Test rate limiting
    for _ in range(100):
        client.get('/api/status', headers=auth_headers)
    response = client.get('/api/status', headers=auth_headers)
    assert response.status_code == 429

def test_pagination(app_context, test_user, auth_headers):
    """Test API pagination."""
    client = app.test_client()
    
    # Create multiple strategies
    for i in range(15):
        client.post('/api/strategies', json={
            'name': f'Strategy {i}',
            'description': f'Description {i}',
            'parameters': {}
        }, headers=auth_headers)
    
    # Test default pagination
    response = client.get('/api/strategies', headers=auth_headers)
    assert response.status_code == 200
    assert len(response.json) == 10  # Default page size
    
    # Test custom page size
    response = client.get('/api/strategies?page_size=5', headers=auth_headers)
    assert response.status_code == 200
    assert len(response.json) == 5
    
    # Test page navigation
    response = client.get('/api/strategies?page=2', headers=auth_headers)
    assert response.status_code == 200
    assert len(response.json) == 5
    
    # Test invalid pagination
    response = client.get('/api/strategies?page=invalid', headers=auth_headers)
    assert response.status_code == 400

def test_filtering_and_sorting(app_context, test_user, auth_headers):
    """Test API filtering and sorting."""
    client = app.test_client()
    
    # Create strategies with different parameters
    strategies = [
        {'name': 'Strategy A', 'parameters': {'sma_short': 10}},
        {'name': 'Strategy B', 'parameters': {'sma_short': 20}},
        {'name': 'Strategy C', 'parameters': {'sma_short': 15}}
    ]
    
    for strategy in strategies:
        client.post('/api/strategies', json=strategy, headers=auth_headers)
    
    # Test filtering
    response = client.get('/api/strategies?filter=sma_short:10', headers=auth_headers)
    assert response.status_code == 200
    assert len(response.json) == 1
    
    # Test sorting
    response = client.get('/api/strategies?sort=name:asc', headers=auth_headers)
    assert response.status_code == 200
    assert response.json[0]['name'] == 'Strategy A'
    
    # Test multiple filters
    response = client.get('/api/strategies?filter=sma_short:>10&filter=sma_short:<20', headers=auth_headers)
    assert response.status_code == 200
    assert len(response.json) == 1
    
    # Test invalid filter
    response = client.get('/api/strategies?filter=invalid', headers=auth_headers)
    assert response.status_code == 400 