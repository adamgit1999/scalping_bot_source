import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from .exceptions import ValidationError

class DataProcessor:
    """Processes and analyzes market data."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the data processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.data_buffer = []
        self.processed_data = {}
        self.indicators = {}
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming market data.
        
        Args:
            data: Market data dictionary
            
        Returns:
            Processed data dictionary
        """
        if not self._validate_data(data):
            raise ValidationError("Invalid market data format")
            
        # Add timestamp if not present
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now(timezone.utc).timestamp()
            
        # Add to buffer
        self.data_buffer.append(data)
        
        # Process if buffer is full
        if len(self.data_buffer) >= self.config.get('buffer_size', 100):
            self._process_buffer()
            
        return self._get_latest_data()
        
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate market data format.
        
        Args:
            data: Market data dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['symbol', 'price', 'volume']
        return all(field in data for field in required_fields)
        
    def _process_buffer(self) -> None:
        """Process the data buffer."""
        if not self.data_buffer:
            return
            
        # Convert buffer to DataFrame
        df = pd.DataFrame(self.data_buffer)
        
        # Calculate indicators
        self._calculate_indicators(df)
        
        # Store processed data
        symbol = df['symbol'].iloc[0]
        self.processed_data[symbol] = df
        
        # Clear buffer
        self.data_buffer = []
        
    def _calculate_indicators(self, df: pd.DataFrame) -> None:
        """Calculate technical indicators.
        
        Args:
            df: Market data DataFrame
        """
        symbol = df['symbol'].iloc[0]
        
        # Calculate basic indicators
        self.indicators[symbol] = {
            'sma_20': self._calculate_sma(df['price'], 20),
            'sma_50': self._calculate_sma(df['price'], 50),
            'rsi_14': self._calculate_rsi(df['price'], 14),
            'macd': self._calculate_macd(df['price']),
            'bollinger_bands': self._calculate_bollinger_bands(df['price'])
        }
        
    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average.
        
        Args:
            prices: Price series
            period: SMA period
            
        Returns:
            SMA series
        """
        return prices.rolling(window=period).mean()
        
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index.
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate MACD.
        
        Args:
            prices: Price series
            
        Returns:
            Dictionary with MACD, signal, and histogram
        """
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return {
            'macd': macd,
            'signal': signal,
            'histogram': histogram
        }
        
    def _calculate_bollinger_bands(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands.
        
        Args:
            prices: Price series
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        period = 20
        std = prices.rolling(window=period).std()
        middle = prices.rolling(window=period).mean()
        upper = middle + (std * 2)
        lower = middle - (std * 2)
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
        
    def _get_latest_data(self) -> Dict[str, Any]:
        """Get latest processed data.
        
        Returns:
            Dictionary with latest data and indicators
        """
        if not self.processed_data:
            return {}
            
        latest = {}
        for symbol, df in self.processed_data.items():
            latest[symbol] = {
                'data': df.iloc[-1].to_dict(),
                'indicators': {
                    name: values.iloc[-1] if isinstance(values, pd.Series) else values
                    for name, values in self.indicators.get(symbol, {}).items()
                }
            }
        return latest
        
    def get_historical_data(self, symbol: str, start_time: Optional[float] = None,
                          end_time: Optional[float] = None) -> pd.DataFrame:
        """Get historical data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            DataFrame with historical data
        """
        if symbol not in self.processed_data:
            return pd.DataFrame()
            
        df = self.processed_data[symbol]
        if start_time:
            df = df[df['timestamp'] >= start_time]
        if end_time:
            df = df[df['timestamp'] <= end_time]
        return df
        
    def get_indicators(self, symbol: str) -> Dict[str, Any]:
        """Get calculated indicators for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with indicators
        """
        return self.indicators.get(symbol, {})
        
    def clear_data(self, symbol: Optional[str] = None) -> None:
        """Clear stored data.
        
        Args:
            symbol: Optional symbol to clear data for
        """
        if symbol:
            self.processed_data.pop(symbol, None)
            self.indicators.pop(symbol, None)
        else:
            self.processed_data.clear()
            self.indicators.clear()
            self.data_buffer.clear() 