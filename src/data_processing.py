import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exceptions import DataProcessingError, ValidationError

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data data class."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int

class MarketDataStore:
    """Store and process market data."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize market data store.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.data: Dict[str, List[MarketData]] = {}
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes
        
    def add_data(self, data: Union[MarketData, List[MarketData]]) -> None:
        """Add market data to store.
        
        Args:
            data: MarketData object or list of MarketData objects
            
        Raises:
            ValidationError: If data validation fails
            DataProcessingError: If data processing fails
        """
        try:
            if isinstance(data, MarketData):
                data = [data]
                
            for item in data:
                if not self._validate_data(item):
                    raise ValidationError(f"Invalid market data: {item}")
                    
                if item.symbol not in self.data:
                    self.data[item.symbol] = []
                    
                self.data[item.symbol].append(item)
                self._update_cache(item.symbol)
                
        except Exception as e:
            raise DataProcessingError(f"Failed to add market data: {e}")
            
    def get_data(self, symbol: str, start_time: Optional[datetime] = None,
                 end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Get market data for symbol.
        
        Args:
            symbol: Trading pair symbol
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            DataFrame with market data
            
        Raises:
            DataProcessingError: If data retrieval fails
        """
        try:
            if symbol not in self.data:
                return pd.DataFrame()
                
            df = self._get_cached_data(symbol)
            
            if start_time:
                df = df[df['timestamp'] >= start_time]
            if end_time:
                df = df[df['timestamp'] <= end_time]
                
            return df
            
        except Exception as e:
            raise DataProcessingError(f"Failed to get market data: {e}")
            
    def calculate_indicators(self, symbol: str, indicators: List[str],
                           params: Optional[Dict] = None) -> pd.DataFrame:
        """Calculate technical indicators.
        
        Args:
            symbol: Trading pair symbol
            indicators: List of indicators to calculate
            params: Indicator parameters
            
        Returns:
            DataFrame with calculated indicators
            
        Raises:
            DataProcessingError: If indicator calculation fails
        """
        try:
            df = self.get_data(symbol)
            if df.empty:
                return df
                
            params = params or {}
            
            for indicator in indicators:
                if indicator == 'sma':
                    period = params.get('sma_period', 20)
                    df[f'sma_{period}'] = self._calculate_sma(df['close'], period)
                elif indicator == 'ema':
                    period = params.get('ema_period', 20)
                    df[f'ema_{period}'] = self._calculate_ema(df['close'], period)
                elif indicator == 'rsi':
                    period = params.get('rsi_period', 14)
                    df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
                elif indicator == 'macd':
                    fast = params.get('macd_fast', 12)
                    slow = params.get('macd_slow', 26)
                    signal = params.get('macd_signal', 9)
                    macd, signal_line, hist = self._calculate_macd(df['close'], fast, slow, signal)
                    df[f'macd_{fast}_{slow}'] = macd
                    df[f'macd_signal_{signal}'] = signal_line
                    df[f'macd_hist_{signal}'] = hist
                elif indicator == 'bollinger':
                    period = params.get('bb_period', 20)
                    std = params.get('bb_std', 2)
                    upper, middle, lower = self._calculate_bollinger_bands(df['close'], period, std)
                    df[f'bb_upper_{period}'] = upper
                    df[f'bb_middle_{period}'] = middle
                    df[f'bb_lower_{period}'] = lower
                    
            return df
            
        except Exception as e:
            raise DataProcessingError(f"Failed to calculate indicators: {e}")
            
    def _validate_data(self, data: MarketData) -> bool:
        """Validate market data.
        
        Args:
            data: MarketData object to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(data, MarketData):
            return False
            
        if not isinstance(data.timestamp, datetime):
            return False
            
        if not isinstance(data.symbol, str) or not data.symbol:
            return False
            
        if not all(isinstance(getattr(data, field), (int, float)) for field in ['open', 'high', 'low', 'close', 'volume']):
            return False
            
        if not isinstance(data.trades, int):
            return False
            
        if not (data.low <= data.open <= data.high and data.low <= data.close <= data.high):
            return False
            
        return True
        
    def _update_cache(self, symbol: str) -> None:
        """Update cache for symbol.
        
        Args:
            symbol: Trading pair symbol
        """
        if symbol in self.data:
            df = pd.DataFrame([vars(d) for d in self.data[symbol]])
            df.set_index('timestamp', inplace=True)
            self.cache[symbol] = df
            
    def _get_cached_data(self, symbol: str) -> pd.DataFrame:
        """Get cached data for symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            DataFrame with cached data
        """
        if symbol in self.cache:
            return self.cache[symbol].copy()
        return pd.DataFrame()
        
    def _calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average.
        
        Args:
            data: Price series
            period: SMA period
            
        Returns:
            Series with SMA values
        """
        return data.rolling(window=period).mean()
        
    def _calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average.
        
        Args:
            data: Price series
            period: EMA period
            
        Returns:
            Series with EMA values
        """
        return data.ewm(span=period, adjust=False).mean()
        
    def _calculate_rsi(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index.
        
        Args:
            data: Price series
            period: RSI period
            
        Returns:
            Series with RSI values
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_macd(self, data: pd.Series, fast: int, slow: int,
                       signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD.
        
        Args:
            data: Price series
            fast: Fast period
            slow: Slow period
            signal: Signal period
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - signal_line
        return macd, signal_line, hist
        
    def _calculate_bollinger_bands(self, data: pd.Series, period: int,
                                 std: float) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands.
        
        Args:
            data: Price series
            period: BB period
            std: Standard deviation multiplier
            
        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        middle = data.rolling(window=period).mean()
        std_dev = data.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower 