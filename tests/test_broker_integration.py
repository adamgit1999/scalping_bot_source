import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import json
from broker.binance import BinanceBroker
from broker.coinbase import CoinbaseBroker
from broker.kraken import KrakenBroker

@pytest.fixture
def mock_binance():
    """Create a mock Binance broker."""
    with patch('broker.binance.Client') as mock_client:
        broker = BinanceBroker('test_key', 'test_secret')
        yield broker

@pytest.fixture
def mock_coinbase():
    """Create a mock Coinbase broker."""
    with patch('broker.coinbase.Client') as mock_client:
        broker = CoinbaseBroker('test_key', 'test_secret')
        yield broker

@pytest.fixture
def mock_kraken():
    """Create a mock Kraken broker."""
    with patch('broker.kraken.API') as mock_api:
        broker = KrakenBroker('test_key', 'test_secret')
        yield broker

def test_binance_fetch_candles(mock_binance):
    """Test Binance candle fetching."""
    # Mock response data
    mock_data = [
        [1625097600000, "35000.00", "35100.00", "34900.00", "35050.00", "100.5"],
        [1625097900000, "35050.00", "35200.00", "35000.00", "35150.00", "150.2"]
    ]
    
    mock_binance.client.get_klines.return_value = mock_data
    
    # Test candle fetching
    candles = mock_binance.fetch_candles('BTC/USDT', '1h', limit=2)
    
    assert len(candles) == 2
    assert candles[0]['timestamp'] == datetime.fromtimestamp(1625097600000/1000)
    assert candles[0]['open'] == 35000.00
    assert candles[0]['high'] == 35100.00
    assert candles[0]['low'] == 34900.00
    assert candles[0]['close'] == 35050.00
    assert candles[0]['volume'] == 100.5

def test_coinbase_fetch_candles(mock_coinbase):
    """Test Coinbase candle fetching."""
    # Mock response data
    mock_data = {
        'candles': [
            {
                'start': '2021-07-01T00:00:00Z',
                'open': '35000.00',
                'high': '35100.00',
                'low': '34900.00',
                'close': '35050.00',
                'volume': '100.5'
            },
            {
                'start': '2021-07-01T01:00:00Z',
                'open': '35050.00',
                'high': '35200.00',
                'low': '35000.00',
                'close': '35150.00',
                'volume': '150.2'
            }
        ]
    }
    
    mock_coinbase.client.get_product_candles.return_value = mock_data
    
    # Test candle fetching
    candles = mock_coinbase.fetch_candles('BTC-USD', '1h', limit=2)
    
    assert len(candles) == 2
    assert candles[0]['timestamp'] == datetime.fromisoformat('2021-07-01T00:00:00Z')
    assert candles[0]['open'] == 35000.00
    assert candles[0]['high'] == 35100.00
    assert candles[0]['low'] == 34900.00
    assert candles[0]['close'] == 35050.00
    assert candles[0]['volume'] == 100.5

def test_kraken_fetch_candles(mock_kraken):
    """Test Kraken candle fetching."""
    # Mock response data
    mock_data = {
        'result': {
            'XXBTZUSD': [
                [1625097600, '35000.00', '35100.00', '34900.00', '35050.00', '100.5', 0],
                [1625097900, '35050.00', '35200.00', '35000.00', '35150.00', '150.2', 0]
            ]
        }
    }
    
    mock_kraken.api.query.return_value = mock_data
    
    # Test candle fetching
    candles = mock_kraken.fetch_candles('BTC/USD', '1h', limit=2)
    
    assert len(candles) == 2
    assert candles[0]['timestamp'] == datetime.fromtimestamp(1625097600)
    assert candles[0]['open'] == 35000.00
    assert candles[0]['high'] == 35100.00
    assert candles[0]['low'] == 34900.00
    assert candles[0]['close'] == 35050.00
    assert candles[0]['volume'] == 100.5

def test_binance_get_balance(mock_binance):
    """Test Binance balance fetching."""
    # Mock response data
    mock_data = {
        'balances': [
            {'asset': 'BTC', 'free': '1.5', 'locked': '0.5'},
            {'asset': 'USDT', 'free': '50000.0', 'locked': '0.0'}
        ]
    }
    
    mock_binance.client.get_account.return_value = mock_data
    
    # Test balance fetching
    balance = mock_binance.get_balance()
    
    assert balance['BTC'] == 2.0  # free + locked
    assert balance['USDT'] == 50000.0

def test_coinbase_get_balance(mock_coinbase):
    """Test Coinbase balance fetching."""
    # Mock response data
    mock_data = {
        'accounts': [
            {'currency': 'BTC', 'balance': '1.5', 'hold': '0.5'},
            {'currency': 'USD', 'balance': '50000.0', 'hold': '0.0'}
        ]
    }
    
    mock_coinbase.client.get_accounts.return_value = mock_data
    
    # Test balance fetching
    balance = mock_coinbase.get_balance()
    
    assert balance['BTC'] == 2.0  # balance + hold
    assert balance['USD'] == 50000.0

def test_kraken_get_balance(mock_kraken):
    """Test Kraken balance fetching."""
    # Mock response data
    mock_data = {
        'result': {
            'XXBT': '1.5',
            'ZUSD': '50000.0'
        }
    }
    
    mock_kraken.api.query.return_value = mock_data
    
    # Test balance fetching
    balance = mock_kraken.get_balance()
    
    assert balance['BTC'] == 1.5
    assert balance['USD'] == 50000.0

def test_binance_place_order(mock_binance):
    """Test Binance order placement."""
    # Mock response data
    mock_data = {
        'orderId': '12345',
        'status': 'FILLED',
        'price': '35000.00',
        'executedQty': '0.1'
    }
    
    mock_binance.client.create_order.return_value = mock_data
    
    # Test order placement
    order = mock_binance.place_order('BTC/USDT', 'buy', 0.1, 35000.00)
    
    assert order['id'] == '12345'
    assert order['status'] == 'FILLED'
    assert order['price'] == 35000.00
    assert order['amount'] == 0.1

def test_coinbase_place_order(mock_coinbase):
    """Test Coinbase order placement."""
    # Mock response data
    mock_data = {
        'id': '12345',
        'status': 'done',
        'price': '35000.00',
        'filled_size': '0.1'
    }
    
    mock_coinbase.client.place_market_order.return_value = mock_data
    
    # Test order placement
    order = mock_coinbase.place_order('BTC-USD', 'buy', 0.1, 35000.00)
    
    assert order['id'] == '12345'
    assert order['status'] == 'done'
    assert order['price'] == 35000.00
    assert order['amount'] == 0.1

def test_kraken_place_order(mock_kraken):
    """Test Kraken order placement."""
    # Mock response data
    mock_data = {
        'result': {
            'txid': ['12345'],
            'status': 'closed',
            'price': '35000.00',
            'vol': '0.1'
        }
    }
    
    mock_kraken.api.query.return_value = mock_data
    
    # Test order placement
    order = mock_kraken.place_order('BTC/USD', 'buy', 0.1, 35000.00)
    
    assert order['id'] == '12345'
    assert order['status'] == 'closed'
    assert order['price'] == 35000.00
    assert order['amount'] == 0.1

def test_broker_error_handling(mock_binance, mock_coinbase, mock_kraken):
    """Test broker error handling."""
    # Test network error
    mock_binance.client.get_klines.side_effect = Exception('Network error')
    with pytest.raises(Exception) as exc_info:
        mock_binance.fetch_candles('BTC/USDT', '1h')
    assert str(exc_info.value) == 'Network error'
    
    # Test invalid API key
    mock_coinbase.client.get_accounts.side_effect = Exception('Invalid API key')
    with pytest.raises(Exception) as exc_info:
        mock_coinbase.get_balance()
    assert str(exc_info.value) == 'Invalid API key'
    
    # Test rate limit
    mock_kraken.api.query.side_effect = Exception('Rate limit exceeded')
    with pytest.raises(Exception) as exc_info:
        mock_kraken.place_order('BTC/USD', 'buy', 0.1, 35000.00)
    assert str(exc_info.value) == 'Rate limit exceeded'

def test_broker_connection_management(mock_binance, mock_coinbase, mock_kraken):
    """Test broker connection management."""
    # Test connection initialization
    assert mock_binance.is_connected() is True
    assert mock_coinbase.is_connected() is True
    assert mock_kraken.is_connected() is True
    
    # Test connection closure
    mock_binance.close()
    mock_coinbase.close()
    mock_kraken.close()
    
    assert mock_binance.is_connected() is False
    assert mock_coinbase.is_connected() is False
    assert mock_kraken.is_connected() is False
    
    # Test reconnection
    mock_binance.connect()
    mock_coinbase.connect()
    mock_kraken.connect()
    
    assert mock_binance.is_connected() is True
    assert mock_coinbase.is_connected() is True
    assert mock_kraken.is_connected() is True

def test_broker_data_validation(mock_binance, mock_coinbase, mock_kraken):
    """Test broker data validation."""
    # Test invalid symbol
    with pytest.raises(ValueError) as exc_info:
        mock_binance.fetch_candles('INVALID', '1h')
    assert 'Invalid symbol' in str(exc_info.value)
    
    # Test invalid timeframe
    with pytest.raises(ValueError) as exc_info:
        mock_coinbase.fetch_candles('BTC-USD', 'invalid')
    assert 'Invalid timeframe' in str(exc_info.value)
    
    # Test invalid order type
    with pytest.raises(ValueError) as exc_info:
        mock_kraken.place_order('BTC/USD', 'invalid', 0.1, 35000.00)
    assert 'Invalid order type' in str(exc_info.value)
    
    # Test invalid amount
    with pytest.raises(ValueError) as exc_info:
        mock_binance.place_order('BTC/USDT', 'buy', -0.1, 35000.00)
    assert 'Invalid amount' in str(exc_info.value)
    
    # Test invalid price
    with pytest.raises(ValueError) as exc_info:
        mock_coinbase.place_order('BTC-USD', 'buy', 0.1, -35000.00)
    assert 'Invalid price' in str(exc_info.value) 