"""
Moving Average Crossover strategy implementation.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any
from .base_strategy import BaseStrategy

class MovingAverageCrossover(BaseStrategy):
    """Moving Average Crossover strategy."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the strategy.
        
        Args:
            config: Dictionary containing strategy configuration
        """
        super().__init__(config)
        self.fast_period = config.get('fast_period', 10)
        self.slow_period = config.get('slow_period', 20)
        self.position = None
        self.prices = []
    
    async def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data and generate trading signals.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            Dictionary containing analysis results and trading signals
        """
        # Extract price data
        price = market_data.get('price')
        if not price:
            return {'signal': 'none', 'reason': 'No price data available'}
        
        # Update price history
        self.prices.append(price)
        if len(self.prices) < self.slow_period:
            return {'signal': 'none', 'reason': 'Insufficient price history'}
        
        # Calculate moving averages
        fast_ma = np.mean(self.prices[-self.fast_period:])
        slow_ma = np.mean(self.prices[-self.slow_period:])
        
        # Generate signal
        signal = 'none'
        reason = ''
        
        if fast_ma > slow_ma and self.position != 'long':
            signal = 'buy'
            reason = 'Fast MA crossed above Slow MA'
        elif fast_ma < slow_ma and self.position != 'short':
            signal = 'sell'
            reason = 'Fast MA crossed below Slow MA'
        
        return {
            'signal': signal,
            'reason': reason,
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'price': price
        }
    
    async def execute(self, signal: Dict[str, Any]) -> bool:
        """
        Execute a trading signal.
        
        Args:
            signal: Dictionary containing trading signal
            
        Returns:
            Boolean indicating whether the execution was successful
        """
        if signal['signal'] == 'buy':
            self.position = 'long'
            return True
        elif signal['signal'] == 'sell':
            self.position = 'short'
            return True
        return False
    
    async def update(self, market_data: Dict[str, Any]) -> None:
        """
        Update strategy state with new market data.
        
        Args:
            market_data: Dictionary containing market data
        """
        # Keep only necessary price history
        if len(self.prices) > self.slow_period * 2:
            self.prices = self.prices[-self.slow_period:] 