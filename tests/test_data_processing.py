import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app import app, db
from data_processing import DataProcessor
from unittest.mock import Mock, patch

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
def data_processor(app_context):
    """Create a data processor instance."""
    return DataProcessor()

@pytest.fixture
def sample_market_data():
    """Create sample market data."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1min')
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.normal(50000, 1000, len(dates)),
        'high': np.random.normal(51000, 1000, len(dates)),
        'low': np.random.normal(49000, 1000, len(dates)),
        'close': np.random.normal(50000, 1000, len(dates)),
        'volume': np.random.normal(100, 10, len(dates))
    })
    return data

def test_data_initialization(data_processor):
    """Test data processor initialization."""
    assert data_processor.app is not None
    assert data_processor.cache is not None
    assert data_processor.logger is not None

def test_data_cleaning(data_processor, sample_market_data):
    """Test market data cleaning."""
    # Add some invalid data
    sample_market_data.loc[0, 'close'] = np.nan
    sample_market_data.loc[1, 'volume'] = -1
    
    # Clean data
    cleaned_data = data_processor.clean_data(sample_market_data)
    
    # Verify cleaning
    assert cleaned_data.isnull().sum().sum() == 0
    assert (cleaned_data['volume'] >= 0).all()
    assert len(cleaned_data) == len(sample_market_data) - 2

def test_data_normalization(data_processor, sample_market_data):
    """Test data normalization."""
    # Normalize data
    normalized_data = data_processor.normalize_data(sample_market_data)
    
    # Verify normalization
    assert normalized_data['open'].mean() == pytest.approx(0, abs=1e-10)
    assert normalized_data['open'].std() == pytest.approx(1, abs=1e-10)
    assert normalized_data['volume'].mean() == pytest.approx(0, abs=1e-10)
    assert normalized_data['volume'].std() == pytest.approx(1, abs=1e-10)

def test_feature_engineering(data_processor, sample_market_data):
    """Test feature engineering."""
    # Add technical indicators
    features = data_processor.add_features(sample_market_data)
    
    # Verify features
    assert 'sma_20' in features.columns
    assert 'rsi_14' in features.columns
    assert 'macd' in features.columns
    assert 'bollinger_upper' in features.columns
    assert 'bollinger_lower' in features.columns
    assert 'atr' in features.columns

def test_data_splitting(data_processor, sample_market_data):
    """Test data splitting for training and testing."""
    # Split data
    train_data, test_data = data_processor.split_data(sample_market_data, test_size=0.2)
    
    # Verify split
    assert len(train_data) + len(test_data) == len(sample_market_data)
    assert len(test_data) / len(sample_market_data) == pytest.approx(0.2, abs=1e-2)

def test_data_resampling(data_processor, sample_market_data):
    """Test data resampling to different timeframes."""
    # Resample to different timeframes
    hourly_data = data_processor.resample_data(sample_market_data, '1H')
    daily_data = data_processor.resample_data(sample_market_data, '1D')
    
    # Verify resampling
    assert len(hourly_data) < len(sample_market_data)
    assert len(daily_data) < len(hourly_data)
    assert hourly_data.index.freq == 'H'
    assert daily_data.index.freq == 'D'

def test_data_validation(data_processor, sample_market_data):
    """Test data validation."""
    # Validate data
    is_valid = data_processor.validate_data(sample_market_data)
    assert is_valid is True
    
    # Test invalid data
    invalid_data = sample_market_data.copy()
    invalid_data.loc[0, 'close'] = -1
    is_valid = data_processor.validate_data(invalid_data)
    assert is_valid is False

def test_data_aggregation(data_processor, sample_market_data):
    """Test data aggregation."""
    # Aggregate data
    aggregated_data = data_processor.aggregate_data(sample_market_data)
    
    # Verify aggregation
    assert 'vwap' in aggregated_data.columns
    assert 'price_range' in aggregated_data.columns
    assert 'volume_profile' in aggregated_data.columns

def test_data_transformation(data_processor, sample_market_data):
    """Test data transformation."""
    # Transform data
    transformed_data = data_processor.transform_data(sample_market_data)
    
    # Verify transformation
    assert 'returns' in transformed_data.columns
    assert 'log_returns' in transformed_data.columns
    assert 'volatility' in transformed_data.columns

def test_data_imputation(data_processor, sample_market_data):
    """Test data imputation."""
    # Create data with missing values
    data_with_missing = sample_market_data.copy()
    data_with_missing.loc[0:5, 'close'] = np.nan
    
    # Impute missing values
    imputed_data = data_processor.impute_data(data_with_missing)
    
    # Verify imputation
    assert imputed_data.isnull().sum().sum() == 0
    assert len(imputed_data) == len(data_with_missing)

def test_data_smoothing(data_processor, sample_market_data):
    """Test data smoothing."""
    # Smooth data
    smoothed_data = data_processor.smooth_data(sample_market_data)
    
    # Verify smoothing
    assert 'smooth_close' in smoothed_data.columns
    assert 'smooth_volume' in smoothed_data.columns
    assert smoothed_data['smooth_close'].std() < sample_market_data['close'].std()

def test_data_outlier_detection(data_processor, sample_market_data):
    """Test outlier detection."""
    # Add outliers
    data_with_outliers = sample_market_data.copy()
    data_with_outliers.loc[0, 'close'] = 100000
    
    # Detect outliers
    outliers = data_processor.detect_outliers(data_with_outliers)
    
    # Verify outlier detection
    assert len(outliers) > 0
    assert 0 in outliers.index

def test_data_correlation(data_processor, sample_market_data):
    """Test correlation analysis."""
    # Calculate correlations
    correlations = data_processor.calculate_correlations(sample_market_data)
    
    # Verify correlations
    assert isinstance(correlations, pd.DataFrame)
    assert correlations.shape[0] == correlations.shape[1]
    assert correlations.index.equals(correlations.columns)

def test_data_statistics(data_processor, sample_market_data):
    """Test statistical analysis."""
    # Calculate statistics
    statistics = data_processor.calculate_statistics(sample_market_data)
    
    # Verify statistics
    assert 'mean' in statistics.columns
    assert 'std' in statistics.columns
    assert 'skew' in statistics.columns
    assert 'kurtosis' in statistics.columns

def test_data_visualization(data_processor, sample_market_data):
    """Test data visualization."""
    # Create visualizations
    with patch('matplotlib.pyplot.show') as mock_show:
        data_processor.visualize_data(sample_market_data)
        mock_show.assert_called()

def test_data_export(data_processor, sample_market_data):
    """Test data export."""
    # Export data
    with patch('pandas.DataFrame.to_csv') as mock_to_csv:
        data_processor.export_data(sample_market_data, 'test.csv')
        mock_to_csv.assert_called_once_with('test.csv', index=True)

def test_data_import(data_processor):
    """Test data import."""
    # Create test data
    test_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', end='2024-01-02', freq='1min'),
        'close': np.random.normal(50000, 1000, 1441)
    })
    test_data.to_csv('test_import.csv')
    
    # Import data
    imported_data = data_processor.import_data('test_import.csv')
    
    # Verify import
    assert isinstance(imported_data, pd.DataFrame)
    assert len(imported_data) == len(test_data)
    assert 'close' in imported_data.columns

def test_data_caching(data_processor, sample_market_data):
    """Test data caching."""
    # Cache data
    data_processor.cache_data('test_key', sample_market_data)
    
    # Retrieve cached data
    cached_data = data_processor.get_cached_data('test_key')
    
    # Verify caching
    assert cached_data.equals(sample_market_data)
    
    # Clear cache
    data_processor.clear_cache('test_key')
    assert data_processor.get_cached_data('test_key') is None

def test_data_compression(data_processor, sample_market_data):
    """Test data compression."""
    # Compress data
    compressed_data = data_processor.compress_data(sample_market_data)
    
    # Decompress data
    decompressed_data = data_processor.decompress_data(compressed_data)
    
    # Verify compression
    assert len(compressed_data) < len(sample_market_data)
    assert decompressed_data.equals(sample_market_data)

def test_data_validation_rules(data_processor, sample_market_data):
    """Test data validation rules."""
    # Define validation rules
    rules = {
        'close': lambda x: x > 0,
        'volume': lambda x: x >= 0,
        'high': lambda x: x >= sample_market_data['low']
    }
    
    # Validate data
    is_valid = data_processor.validate_data_rules(sample_market_data, rules)
    assert is_valid is True
    
    # Test invalid data
    invalid_data = sample_market_data.copy()
    invalid_data.loc[0, 'high'] = invalid_data.loc[0, 'low'] - 1
    is_valid = data_processor.validate_data_rules(invalid_data, rules)
    assert is_valid is False 