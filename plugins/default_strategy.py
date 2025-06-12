"""
Default moving-average scalping strategy.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

class DefaultStrategy:
    def __init__(self):
        # Strategy parameters
        self.fast_period = 10
        self.slow_period = 20
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        
        # State
        self.last_signal = None
    
    def generate_signal(self, df: pd.DataFrame) -> str:
        """
        Generate trading signal based on technical indicators
        
        Returns:
            str: 'BUY', 'SELL', or None
        """
        # Calculate indicators
        df = self._calculate_indicators(df)
        
        # Get latest values
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Generate signal based on strategy rules
        signal = None
        
        # Moving average crossover
        ma_crossover = (
            previous['fast_ma'] <= previous['slow_ma'] and
            current['fast_ma'] > current['slow_ma']
        )
        
        ma_crossunder = (
            previous['fast_ma'] >= previous['slow_ma'] and
            current['fast_ma'] < current['slow_ma']
        )
        
        # RSI conditions
        rsi_oversold = current['rsi'] < self.rsi_oversold
        rsi_overbought = current['rsi'] > self.rsi_overbought
        
        # Volume confirmation
        volume_increase = current['volume'] > df['volume'].rolling(20).mean().iloc[-1]
        
        # Generate signals
        if ma_crossover and rsi_oversold and volume_increase:
            signal = 'BUY'
        elif ma_crossunder and rsi_overbought:
            signal = 'SELL'
        
        # Update state
        self.last_signal = signal
        
        return signal
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # Moving averages
        df['fast_ma'] = df['close'].rolling(window=self.fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_period).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['sma20'] + (df['std20'] * 2)
        df['lower_band'] = df['sma20'] - (df['std20'] * 2)
        
        # VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        return df
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'rsi_period': self.rsi_period,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold
        }
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """Set strategy parameters"""
        for key, value in parameters.items():
            if hasattr(self, key):
                setattr(self, key, value)

