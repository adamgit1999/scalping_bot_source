import pytest
from app import db, init_db
from config import Config
from datetime import datetime, timedelta
import os
import json
from werkzeug.security import generate_password_hash, check_password_hash

@pytest.fixture
def app():
    """Create a Flask app for testing."""
    from app import app
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    return app

@pytest.fixture
def test_config():
    """Create a test configuration."""
    return {
        'DATABASE': 'test.db',
        'SECRET_KEY': 'test_key',
        'BROKER': 'test_broker',
        'SYMBOL': 'BTC/USDT',
        'MOCK_MODE': True
    }

def test_database_initialization(app):
    """Test database initialization."""
    with app.app_context():
        init_db()
        # Verify tables are created
        assert 'users' in db.metadata.tables
        assert 'strategies' in db.metadata.tables
        assert 'trades' in db.metadata.tables
        assert 'backtests' in db.metadata.tables

def test_config_loading(test_config):
    """Test configuration loading."""
    # Save test config to file
    with open('test_config.json', 'w') as f:
        json.dump(test_config, f)
    
    # Load config
    config = Config()
    config.load_config('test_config.json')
    
    # Verify config values
    assert config.DATABASE == test_config['DATABASE']
    assert config.SECRET_KEY == test_config['SECRET_KEY']
    assert config.BROKER == test_config['BROKER']
    assert config.SYMBOL == test_config['SYMBOL']
    assert config.MOCK_MODE == test_config['MOCK_MODE']
    
    # Clean up
    os.remove('test_config.json')

def test_password_hashing():
    """Test password hashing and verification."""
    password = 'test_password'
    hashed = generate_password_hash(password)
    
    assert check_password_hash(hashed, password)
    assert not check_password_hash(hashed, 'wrong_password')

def test_api_key_generation():
    """Test API key generation."""
    from app import generate_api_key
    
    api_key = generate_api_key()
    assert isinstance(api_key, str)
    assert len(api_key) == 64
    assert api_key.isalnum()

def test_token_generation():
    """Test JWT token generation and validation."""
    from app import generate_token, validate_token
    
    user_id = 1
    token = generate_token(user_id)
    assert isinstance(token, str)
    
    # Validate token
    decoded = validate_token(token)
    assert decoded['user_id'] == user_id

def test_date_parsing():
    """Test date parsing utilities."""
    from app import parse_date, format_date
    
    # Test parsing
    date_str = '2024-01-01T00:00:00Z'
    parsed = parse_date(date_str)
    assert isinstance(parsed, datetime)
    
    # Test formatting
    formatted = format_date(parsed)
    assert isinstance(formatted, str)
    assert formatted == date_str

def test_number_formatting():
    """Test number formatting utilities."""
    from app import format_number, format_currency, format_percentage
    
    # Test number formatting
    assert format_number(1234.5678) == '1,234.57'
    assert format_number(1234.5678, precision=3) == '1,234.568'
    
    # Test currency formatting
    assert format_currency(1234.5678) == '$1,234.57'
    assert format_currency(1234.5678, currency='EUR') == '€1,234.57'
    
    # Test percentage formatting
    assert format_percentage(0.1234) == '12.34%'
    assert format_percentage(0.1234, precision=1) == '12.3%'

def test_error_handling():
    """Test error handling utilities."""
    from app import handle_error, APIError
    
    # Test API error
    with pytest.raises(APIError) as exc_info:
        handle_error('Test error', 400)
    assert exc_info.value.message == 'Test error'
    assert exc_info.value.status_code == 400

def test_validation():
    """Test input validation utilities."""
    from app import validate_strategy, validate_trade
    
    # Test strategy validation
    valid_strategy = {
        'name': 'Test Strategy',
        'description': 'A test strategy',
        'parameters': {
            'sma_short': 10,
            'sma_long': 20
        }
    }
    assert validate_strategy(valid_strategy) is True
    
    invalid_strategy = {
        'name': 'Test Strategy'
        # Missing required fields
    }
    with pytest.raises(ValueError):
        validate_strategy(invalid_strategy)
    
    # Test trade validation
    valid_trade = {
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'price': 50000.0,
        'amount': 0.1
    }
    assert validate_trade(valid_trade) is True
    
    invalid_trade = {
        'symbol': 'BTC/USDT'
        # Missing required fields
    }
    with pytest.raises(ValueError):
        validate_trade(invalid_trade)

def test_file_operations():
    """Test file operation utilities."""
    from app import save_file, load_file, delete_file
    
    # Test file saving
    test_data = {'key': 'value'}
    filename = 'test_file.json'
    save_file(filename, test_data)
    assert os.path.exists(filename)
    
    # Test file loading
    loaded_data = load_file(filename)
    assert loaded_data == test_data
    
    # Test file deletion
    delete_file(filename)
    assert not os.path.exists(filename)

def test_logging():
    """Test logging utilities."""
    from app import setup_logger, log_trade, log_error
    
    # Setup logger
    logger = setup_logger('test_logger')
    assert logger is not None
    
    # Test trade logging
    trade_data = {
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'price': 50000.0,
        'amount': 0.1
    }
    log_trade(trade_data)
    
    # Test error logging
    error_data = {
        'message': 'Test error',
        'code': 400
    }
    log_error(error_data)

def test_cache_operations():
    """Test cache operation utilities."""
    from app import cache_set, cache_get, cache_delete
    
    # Test cache setting
    cache_set('test_key', 'test_value')
    assert cache_get('test_key') == 'test_value'
    
    # Test cache deletion
    cache_delete('test_key')
    assert cache_get('test_key') is None

def test_rate_limiting():
    """Test rate limiting utilities."""
    from app import check_rate_limit, reset_rate_limit
    
    # Test rate limiting
    for _ in range(100):
        assert check_rate_limit('test_ip') is True
    
    # Should be rate limited
    assert check_rate_limit('test_ip') is False
    
    # Reset rate limit
    reset_rate_limit('test_ip')
    assert check_rate_limit('test_ip') is True 