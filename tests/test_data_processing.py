import pytest
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from src.data.data_processor import DataProcessor
from src.data.market_data_store import MarketDataStore
from src.exceptions import DataProcessingError, ValidationError

@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return {
        'symbol': 'BTC/USDT',
        'interval': '1m',
        'max_data_points': 1000,
        'indicators': {
            'sma': [20, 50, 200],
            'ema': [12, 26],
            'rsi': [14],
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger': {'period': 20, 'std_dev': 2}
        }
    }

@pytest.fixture
def mock_market_data_store():
    """Create a mock market data store."""
    store = Mock(spec=MarketDataStore)
    store.get_latest_data.return_value = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1min'),
        'open': np.random.uniform(45000, 55000, 100),
        'high': np.random.uniform(45000, 55000, 100),
        'low': np.random.uniform(45000, 55000, 100),
        'close': np.random.uniform(45000, 55000, 100),
        'volume': np.random.uniform(1, 10, 100)
    })
    return store

@pytest.fixture
def data_processor(mock_config, mock_market_data_store):
    """Create a data processor instance."""
    return DataProcessor(mock_config, mock_market_data_store)

def test_initialization(data_processor, mock_config):
    """Test data processor initialization."""
    assert data_processor.symbol == mock_config['symbol']
    assert data_processor.interval == mock_config['interval']
    assert data_processor.max_data_points == mock_config['max_data_points']
    assert data_processor.indicators == mock_config['indicators']
    assert data_processor.data_store == mock_market_data_store
    assert data_processor.processed_data is None

def test_load_data(data_processor, mock_market_data_store):
    """Test loading market data."""
    # Load data
    data = data_processor.load_data()
    
    # Verify data loading
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    mock_market_data_store.get_latest_data.assert_called_once()

def test_calculate_indicators(data_processor):
    """Test indicator calculation."""
    # Create sample data
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1min'),
        'open': np.random.uniform(45000, 55000, 100),
        'high': np.random.uniform(45000, 55000, 100),
        'low': np.random.uniform(45000, 55000, 100),
        'close': np.random.uniform(45000, 55000, 100),
        'volume': np.random.uniform(1, 10, 100)
    })
    
    # Calculate indicators
    result = data_processor.calculate_indicators(data)
    
    # Verify indicator calculation
    assert isinstance(result, pd.DataFrame)
    assert 'sma_20' in result.columns
    assert 'sma_50' in result.columns
    assert 'sma_200' in result.columns
    assert 'ema_12' in result.columns
    assert 'ema_26' in result.columns
    assert 'rsi_14' in result.columns
    assert 'macd' in result.columns
    assert 'macd_signal' in result.columns
    assert 'macd_hist' in result.columns
    assert 'bb_upper' in result.columns
    assert 'bb_middle' in result.columns
    assert 'bb_lower' in result.columns

def test_calculate_sma(data_processor):
    """Test SMA calculation."""
    # Create sample data
    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    })
    
    # Calculate SMA
    result = data_processor._calculate_sma(data, 5)
    
    # Verify SMA calculation
    assert isinstance(result, pd.Series)
    assert len(result) == len(data)
    assert not result.iloc[:4].notna().any()  # First 4 values should be NaN
    assert result.iloc[4] == 102  # (100 + 101 + 102 + 103 + 104) / 5

def test_calculate_ema(data_processor):
    """Test EMA calculation."""
    # Create sample data
    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    })
    
    # Calculate EMA
    result = data_processor._calculate_ema(data, 5)
    
    # Verify EMA calculation
    assert isinstance(result, pd.Series)
    assert len(result) == len(data)
    assert not result.iloc[:4].notna().any()  # First 4 values should be NaN
    assert result.iloc[4] == 102  # First EMA value equals SMA

def test_calculate_rsi(data_processor):
    """Test RSI calculation."""
    # Create sample data
    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    })
    
    # Calculate RSI
    result = data_processor._calculate_rsi(data, 14)
    
    # Verify RSI calculation
    assert isinstance(result, pd.Series)
    assert len(result) == len(data)
    assert not result.iloc[:14].notna().any()  # First 14 values should be NaN
    assert all(0 <= x <= 100 for x in result.dropna())  # RSI should be between 0 and 100

def test_calculate_macd(data_processor):
    """Test MACD calculation."""
    # Create sample data
    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    })
    
    # Calculate MACD
    macd, signal, hist = data_processor._calculate_macd(data)
    
    # Verify MACD calculation
    assert isinstance(macd, pd.Series)
    assert isinstance(signal, pd.Series)
    assert isinstance(hist, pd.Series)
    assert len(macd) == len(data)
    assert len(signal) == len(data)
    assert len(hist) == len(data)

def test_calculate_bollinger_bands(data_processor):
    """Test Bollinger Bands calculation."""
    # Create sample data
    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    })
    
    # Calculate Bollinger Bands
    upper, middle, lower = data_processor._calculate_bollinger_bands(data, 20, 2)
    
    # Verify Bollinger Bands calculation
    assert isinstance(upper, pd.Series)
    assert isinstance(middle, pd.Series)
    assert isinstance(lower, pd.Series)
    assert len(upper) == len(data)
    assert len(middle) == len(data)
    assert len(lower) == len(data)
    assert all(upper >= middle)
    assert all(middle >= lower)

def test_validate_data(data_processor):
    """Test data validation."""
    # Valid data
    valid_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1min'),
        'open': np.random.uniform(45000, 55000, 100),
        'high': np.random.uniform(45000, 55000, 100),
        'low': np.random.uniform(45000, 55000, 100),
        'close': np.random.uniform(45000, 55000, 100),
        'volume': np.random.uniform(1, 10, 100)
    })
    assert data_processor.validate_data(valid_data)
    
    # Invalid data (missing columns)
    invalid_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1min'),
        'open': np.random.uniform(45000, 55000, 100)
    })
    with pytest.raises(ValidationError):
        data_processor.validate_data(invalid_data)
    
    # Invalid data (empty DataFrame)
    with pytest.raises(ValidationError):
        data_processor.validate_data(pd.DataFrame())

def test_process_data(data_processor):
    """Test data processing pipeline."""
    # Process data
    result = data_processor.process_data()
    
    # Verify data processing
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert all(col in result.columns for col in [
        'open', 'high', 'low', 'close', 'volume',
        'sma_20', 'sma_50', 'sma_200',
        'ema_12', 'ema_26',
        'rsi_14',
        'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower'
    ])

def test_get_latest_data(data_processor):
    """Test getting latest processed data."""
    # Process data first
    data_processor.process_data()
    
    # Get latest data
    latest = data_processor.get_latest_data()
    
    # Verify latest data
    assert isinstance(latest, pd.DataFrame)
    assert len(latest) > 0
    assert all(col in latest.columns for col in [
        'open', 'high', 'low', 'close', 'volume',
        'sma_20', 'sma_50', 'sma_200',
        'ema_12', 'ema_26',
        'rsi_14',
        'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower'
    ])

def test_error_handling(data_processor):
    """Test error handling."""
    # Test invalid data
    with pytest.raises(ValidationError):
        data_processor.validate_data(None)
    
    # Test processing error
    data_processor.data_store.get_latest_data.side_effect = Exception("Data store error")
    with pytest.raises(DataProcessingError):
        data_processor.process_data()
    
    # Test indicator calculation error
    with patch.object(data_processor, '_calculate_sma', side_effect=Exception("Calculation error")):
        with pytest.raises(DataProcessingError):
            data_processor.calculate_indicators(pd.DataFrame())

def test_data_cleanup(data_processor):
    """Test data cleanup."""
    # Process data first
    data_processor.process_data()
    
    # Cleanup data
    data_processor.cleanup()
    
    # Verify cleanup
    assert data_processor.processed_data is None 