"""
Base strategy class for all trading strategies.
"""
from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the strategy.
        
        Args:
            config: Dictionary containing strategy configuration
        """
        self.config = config
        self.name = self.__class__.__name__
        logger.info(f"Initializing strategy: {self.name}")
    
    @abstractmethod
    async def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data and generate trading signals.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            Dictionary containing analysis results and trading signals
        """
        pass
    
    @abstractmethod
    async def execute(self, signal: Dict[str, Any]) -> bool:
        """
        Execute a trading signal.
        
        Args:
            signal: Dictionary containing trading signal
            
        Returns:
            Boolean indicating whether the execution was successful
        """
        pass
    
    @abstractmethod
    async def update(self, market_data: Dict[str, Any]) -> None:
        """
        Update strategy state with new market data.
        
        Args:
            market_data: Dictionary containing market data
        """
        pass
    
    def validate_config(self) -> bool:
        """
        Validate strategy configuration.
        
        Returns:
            Boolean indicating whether the configuration is valid
        """
        required_fields = ['symbol', 'timeframe', 'risk_per_trade']
        return all(field in self.config for field in required_fields)
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current strategy state.
        
        Returns:
            Dictionary containing current strategy state
        """
        return {
            'name': self.name,
            'config': self.config,
            'is_active': True
        } 