import pytest
from src.indicators import (
    calculateSMA,
    calculateEMA,
    calculateRSI,
    calculateMACD,
    calculateBollingerBands,
    calculateVWAP,
    calculateATR,
    calculateStochastic
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
    data = pd.DataFrame({
        'open': np.random.normal(100, 1, len(dates)),
        'high': np.random.normal(101, 1, len(dates)),
        'low': np.random.normal(99, 1, len(dates)),
        'close': np.random.normal(100, 1, len(dates)),
        'volume': np.random.normal(1000, 100, len(dates))
    }, index=dates)
    return data

def test_calculate_sma(sample_data):
    """Test Simple Moving Average calculation."""
    period = 20
    sma = calculateSMA(sample_data['close'], period)
    
    assert isinstance(sma, pd.Series)
    assert len(sma) == len(sample_data)
    assert sma.isna().sum() == period - 1  # First period-1 values should be NaN
    assert not sma.isna().any()  # Rest should not be NaN

def test_calculate_ema(sample_data):
    """Test Exponential Moving Average calculation."""
    period = 20
    ema = calculateEMA(sample_data['close'], period)
    
    assert isinstance(ema, pd.Series)
    assert len(ema) == len(sample_data)
    assert ema.isna().sum() == period - 1  # First period-1 values should be NaN
    assert not ema.isna().any()  # Rest should not be NaN

def test_calculate_rsi(sample_data):
    """Test Relative Strength Index calculation."""
    period = 14
    rsi = calculateRSI(sample_data['close'], period)
    
    assert isinstance(rsi, pd.Series)
    assert len(rsi) == len(sample_data)
    assert rsi.isna().sum() == period  # First period values should be NaN
    assert not rsi.isna().any()  # Rest should not be NaN
    assert all(0 <= x <= 100 for x in rsi.dropna())  # RSI should be between 0 and 100

def test_calculate_macd(sample_data):
    """Test MACD calculation."""
    fast_period = 12
    slow_period = 26
    signal_period = 9
    
    macd, signal, hist = calculateMACD(
        sample_data['close'],
        fast_period,
        slow_period,
        signal_period
    )
    
    assert isinstance(macd, pd.Series)
    assert isinstance(signal, pd.Series)
    assert isinstance(hist, pd.Series)
    assert len(macd) == len(sample_data)
    assert len(signal) == len(sample_data)
    assert len(hist) == len(sample_data)
    assert macd.isna().sum() == slow_period - 1  # First slow_period-1 values should be NaN
    assert signal.isna().sum() == slow_period + signal_period - 2  # First slow_period+signal_period-2 values should be NaN

def test_calculate_bollinger_bands(sample_data):
    """Test Bollinger Bands calculation."""
    period = 20
    multiplier = 2
    
    upper, middle, lower = calculateBollingerBands(
        sample_data['close'],
        period,
        multiplier
    )
    
    assert isinstance(upper, pd.Series)
    assert isinstance(middle, pd.Series)
    assert isinstance(lower, pd.Series)
    assert len(upper) == len(sample_data)
    assert len(middle) == len(sample_data)
    assert len(lower) == len(sample_data)
    assert upper.isna().sum() == period - 1  # First period-1 values should be NaN
    assert middle.isna().sum() == period - 1  # First period-1 values should be NaN
    assert lower.isna().sum() == period - 1  # First period-1 values should be NaN
    assert all(upper >= middle)  # Upper band should be above middle band
    assert all(middle >= lower)  # Middle band should be above lower band

def test_calculate_vwap(sample_data):
    """Test Volume Weighted Average Price calculation."""
    vwap = calculateVWAP(sample_data)
    
    assert isinstance(vwap, pd.Series)
    assert len(vwap) == len(sample_data)
    assert not vwap.isna().any()  # No NaN values should be present

def test_calculate_atr(sample_data):
    """Test Average True Range calculation."""
    period = 14
    atr = calculateATR(sample_data, period)
    
    assert isinstance(atr, pd.Series)
    assert len(atr) == len(sample_data)
    assert atr.isna().sum() == period  # First period values should be NaN
    assert not atr.isna().any()  # Rest should not be NaN
    assert all(x >= 0 for x in atr.dropna())  # ATR should be non-negative

def test_calculate_stochastic(sample_data):
    """Test Stochastic Oscillator calculation."""
    k_period = 14
    d_period = 3
    
    k, d = calculateStochastic(
        sample_data,
        k_period,
        d_period
    )
    
    assert isinstance(k, pd.Series)
    assert isinstance(d, pd.Series)
    assert len(k) == len(sample_data)
    assert len(d) == len(sample_data)
    assert k.isna().sum() == k_period - 1  # First k_period-1 values should be NaN
    assert d.isna().sum() == k_period + d_period - 2  # First k_period+d_period-2 values should be NaN
    assert all(0 <= x <= 100 for x in k.dropna())  # %K should be between 0 and 100
    assert all(0 <= x <= 100 for x in d.dropna())  # %D should be between 0 and 100

def test_edge_cases():
    """Test edge cases for indicator calculations."""
    # Empty DataFrame
    empty_data = pd.DataFrame()
    assert len(calculateSMA(empty_data, 20)) == 0
    assert len(calculateEMA(empty_data, 20)) == 0
    assert len(calculateRSI(empty_data, 14)) == 0
    
    # Single value
    single_data = pd.Series([100])
    assert len(calculateSMA(single_data, 20)) == 1
    assert len(calculateEMA(single_data, 20)) == 1
    assert len(calculateRSI(single_data, 14)) == 1
    
    # All NaN values
    nan_data = pd.Series([np.nan] * 100)
    assert calculateSMA(nan_data, 20).isna().all()
    assert calculateEMA(nan_data, 20).isna().all()
    assert calculateRSI(nan_data, 14).isna().all()
    
    # Constant values
    constant_data = pd.Series([100] * 100)
    sma = calculateSMA(constant_data, 20)
    assert all(x == 100 for x in sma.dropna())
    
    # Zero values
    zero_data = pd.Series([0] * 100)
    sma = calculateSMA(zero_data, 20)
    assert all(x == 0 for x in sma.dropna()) 