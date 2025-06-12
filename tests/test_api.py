import pytest
from app import app, db, User, Strategy, Trade
from datetime import datetime, timedelta
import json

@pytest.fixture
def client():
    """Create a test client."""
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            yield client
            db.session.remove()
            db.drop_all()

@pytest.fixture
def auth_headers():
    """Create authentication headers."""
    return {
        'Authorization': 'Bearer test_token'
    }

@pytest.fixture
def test_user():
    """Create a test user."""
    user = User(
        username='testuser',
        email='test@example.com',
        password_hash='hashed_password'
    )
    db.session.add(user)
    db.session.commit()
    return user

def test_api_status(client, auth_headers):
    """Test API status endpoint."""
    response = client.get('/api/status', headers=auth_headers)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'status' in data
    assert 'broker' in data
    assert 'symbol' in data
    assert 'mock_mode' in data

def test_api_trades(client, auth_headers, test_user):
    """Test trades endpoint."""
    # Create test trades
    strategy = Strategy(
        name='Test Strategy',
        description='A test strategy',
        parameters={},
        user_id=test_user.id
    )
    db.session.add(strategy)
    db.session.commit()
    
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
    
    response = client.get('/api/trades', headers=auth_headers)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)
    assert len(data) > 0
    assert data[0]['symbol'] == 'BTC/USDT'
    assert data[0]['side'] == 'buy'
    assert data[0]['price'] == 50000.0

def test_api_balance(client, auth_headers):
    """Test balance endpoint."""
    response = client.get('/api/balance', headers=auth_headers)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, dict)
    assert 'error' not in data

def test_api_strategies(client, auth_headers, test_user):
    """Test strategies endpoint."""
    # Create test strategies
    strategy1 = Strategy(
        name='Strategy 1',
        description='First strategy',
        parameters={'param1': 'value1'},
        user_id=test_user.id
    )
    strategy2 = Strategy(
        name='Strategy 2',
        description='Second strategy',
        parameters={'param2': 'value2'},
        user_id=test_user.id
    )
    db.session.add_all([strategy1, strategy2])
    db.session.commit()
    
    response = client.get('/api/strategies', headers=auth_headers)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]['name'] == 'Strategy 1'
    assert data[1]['name'] == 'Strategy 2'

def test_api_create_strategy(client, auth_headers, test_user):
    """Test strategy creation endpoint."""
    strategy_data = {
        'name': 'New Strategy',
        'description': 'A new strategy',
        'parameters': {
            'sma_short': 10,
            'sma_long': 20
        }
    }
    
    response = client.post(
        '/api/strategies',
        headers=auth_headers,
        json=strategy_data
    )
    assert response.status_code == 201
    data = json.loads(response.data)
    assert data['name'] == 'New Strategy'
    assert data['description'] == 'A new strategy'
    assert data['parameters']['sma_short'] == 10
    assert data['parameters']['sma_long'] == 20

def test_api_update_strategy(client, auth_headers, test_user):
    """Test strategy update endpoint."""
    # Create test strategy
    strategy = Strategy(
        name='Test Strategy',
        description='A test strategy',
        parameters={},
        user_id=test_user.id
    )
    db.session.add(strategy)
    db.session.commit()
    
    update_data = {
        'name': 'Updated Strategy',
        'description': 'Updated description',
        'parameters': {
            'new_param': 'new_value'
        }
    }
    
    response = client.put(
        f'/api/strategies/{strategy.id}',
        headers=auth_headers,
        json=update_data
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['name'] == 'Updated Strategy'
    assert data['description'] == 'Updated description'
    assert data['parameters']['new_param'] == 'new_value'

def test_api_delete_strategy(client, auth_headers, test_user):
    """Test strategy deletion endpoint."""
    # Create test strategy
    strategy = Strategy(
        name='Test Strategy',
        description='A test strategy',
        parameters={},
        user_id=test_user.id
    )
    db.session.add(strategy)
    db.session.commit()
    
    response = client.delete(
        f'/api/strategies/{strategy.id}',
        headers=auth_headers
    )
    assert response.status_code == 204
    
    # Verify strategy is deleted
    assert Strategy.query.get(strategy.id) is None

def test_api_backtest(client, auth_headers, test_user):
    """Test backtest endpoint."""
    # Create test strategy
    strategy = Strategy(
        name='Test Strategy',
        description='A test strategy',
        parameters={},
        user_id=test_user.id
    )
    db.session.add(strategy)
    db.session.commit()
    
    backtest_data = {
        'strategy_id': strategy.id,
        'start_date': datetime.now(datetime.UTC).isoformat(),
        'end_date': (datetime.now(datetime.UTC) + timedelta(days=30)).isoformat(),
        'initial_balance': 10000.0
    }
    
    response = client.post(
        '/api/backtest',
        headers=auth_headers,
        json=backtest_data
    )
    assert response.status_code == 201
    data = json.loads(response.data)
    assert data['strategy_id'] == strategy.id
    assert data['initial_balance'] == 10000.0
    assert 'final_balance' in data
    assert 'total_trades' in data
    assert 'win_rate' in data

def test_api_webhook_events(client):
    """Test webhook events endpoint."""
    event_data = {
        'type': 'trade',
        'data': {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'price': 50000.0,
            'amount': 0.1
        }
    }
    
    response = client.post(
        '/api/webhooks/events',
        json=event_data
    )
    assert response.status_code == 204

def test_api_unauthorized_access(client):
    """Test unauthorized access to protected endpoints."""
    endpoints = [
        '/api/status',
        '/api/trades',
        '/api/balance',
        '/api/strategies'
    ]
    
    for endpoint in endpoints:
        response = client.get(endpoint)
        assert response.status_code == 401

def test_api_invalid_strategy_id(client, auth_headers):
    """Test invalid strategy ID handling."""
    response = client.get('/api/strategies/999', headers=auth_headers)
    assert response.status_code == 404

def test_api_invalid_request_data(client, auth_headers):
    """Test invalid request data handling."""
    invalid_data = {
        'name': 'Test Strategy'
        # Missing required fields
    }
    
    response = client.post(
        '/api/strategies',
        headers=auth_headers,
        json=invalid_data
    )
    assert response.status_code == 400

def test_api_rate_limiting(client, auth_headers):
    """Test API rate limiting."""
    # Make multiple requests in quick succession
    for _ in range(100):
        response = client.get('/api/status', headers=auth_headers)
    
    # The next request should be rate limited
    response = client.get('/api/status', headers=auth_headers)
    assert response.status_code == 429 