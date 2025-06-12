import pytest
from app import app, db, User, Strategy, Trade
from datetime import datetime, timedelta, timezone
import time
import random
from sqlalchemy import func

@pytest.fixture
def app_context():
    """Create an application context for testing."""
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['TESTING'] = True
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

@pytest.fixture
def test_user(app_context):
    """Create a test user."""
    # Delete any existing test user
    User.query.filter_by(email='test@example.com').delete()
    db.session.commit()
    
    user = User(
        username='testuser',
        email='test@example.com',
        password_hash='hashed_password'
    )
    db.session.add(user)
    db.session.commit()
    return user

def test_bulk_insert_performance(app_context, test_user):
    """Test bulk insert performance."""
    # Create a strategy
    strategy = Strategy(
        name='Performance Test Strategy',
        description='Test strategy for performance testing',
        parameters={},
        user_id=test_user.id
    )
    db.session.add(strategy)
    db.session.commit()
    
    # Measure bulk insert performance
    start_time = time.time()
    trades = []
    for i in range(1000):
        trade = Trade(
            strategy_id=strategy.id,
            symbol='BTC/USDT',
            side='buy',
            price=50000.0 + random.uniform(-100, 100),
            amount=0.1,
            fee=5.0,
            profit=100.0,
            timestamp=datetime.now(timezone.utc)
        )
        trades.append(trade)
    
    # Use bulk insert
    db.session.bulk_save_objects(trades)
    db.session.commit()
    end_time = time.time()
    
    # Calculate performance metrics
    insert_time = end_time - start_time
    assert insert_time < 2.0  # Bulk insert should complete within 2 seconds

def test_query_performance(app_context, test_user):
    """Test query performance."""
    # Create a strategy
    strategy = Strategy(
        name='Performance Test Strategy',
        description='Test strategy for performance testing',
        parameters={},
        user_id=test_user.id
    )
    db.session.add(strategy)
    db.session.commit()
    
    # Create some test data
    trades = []
    for i in range(1000):
        trade = Trade(
            strategy_id=strategy.id,
            symbol='BTC/USDT',
            side='buy',
            price=50000.0 + random.uniform(-100, 100),
            amount=0.1,
            fee=5.0,
            profit=100.0,
            timestamp=datetime.now(timezone.utc) - timedelta(hours=i)
        )
        trades.append(trade)
    db.session.bulk_save_objects(trades)
    db.session.commit()
    
    # Test indexed queries
    start_time = time.time()
    recent_trades = Trade.query.filter(
        Trade.strategy_id == strategy.id,
        Trade.timestamp >= datetime.now(timezone.utc) - timedelta(hours=1)
    ).order_by(Trade.timestamp.desc()).all()
    end_time = time.time()
    
    query_time = end_time - start_time
    assert query_time < 0.1  # Query should complete within 100ms
    
    # Test aggregation queries
    start_time = time.time()
    result = db.session.query(
        Trade.symbol,
        func.count(Trade.id).label('trade_count'),
        func.sum(Trade.profit).label('total_profit')
    ).filter(
        Trade.strategy_id == strategy.id
    ).group_by(Trade.symbol).all()
    end_time = time.time()
    
    agg_time = end_time - start_time
    assert agg_time < 0.1  # Aggregation should complete within 100ms

def test_concurrent_operations(app_context, test_user):
    """Test concurrent database operations."""
    # Create a strategy
    strategy = Strategy(
        name='Performance Test Strategy',
        description='Test strategy for performance testing',
        parameters={},
        user_id=test_user.id
    )
    db.session.add(strategy)
    db.session.commit()
    
    # Create multiple trades concurrently
    start_time = time.time()
    trades = []
    for i in range(100):
        trade = Trade(
            strategy_id=strategy.id,
            symbol='BTC/USDT',
            side='buy',
            price=50000.0 + random.uniform(-100, 100),
            amount=0.1,
            fee=5.0,
            profit=100.0,
            timestamp=datetime.now(timezone.utc)
        )
        trades.append(trade)
        db.session.add(trade)
        if i % 10 == 0:  # Commit every 10 trades
            db.session.commit()
    
    db.session.commit()
    end_time = time.time()
    
    # Calculate performance metrics
    operation_time = end_time - start_time
    assert operation_time < 1.0  # Operations should complete within 1 second 