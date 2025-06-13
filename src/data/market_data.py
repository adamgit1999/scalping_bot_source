"""
Market data handler for fetching and processing market data.
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List
import ccxt.async_support as ccxt

logger = logging.getLogger(__name__)

class MarketDataHandler:
    """Handler for market data operations."""
    
    def __init__(self, exchange_id: str, config: Dict[str, Any]):
        """
        Initialize the market data handler.
        
        Args:
            exchange_id: ID of the exchange to use
            config: Dictionary containing exchange configuration
        """
        self.exchange_id = exchange_id
        self.config = config
        self.exchange = getattr(ccxt, exchange_id)(config)
        self.symbols: List[str] = []
        self.timeframes: List[str] = []
        logger.info(f"Initialized market data handler for {exchange_id}")
    
    async def connect(self) -> bool:
        """
        Connect to the exchange.
        
        Returns:
            Boolean indicating whether the connection was successful
        """
        try:
            await self.exchange.load_markets()
            logger.info(f"Connected to {self.exchange_id}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to {self.exchange_id}: {str(e)}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the exchange."""
        await self.exchange.close()
        logger.info(f"Disconnected from {self.exchange_id}")
    
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary containing ticker data
        """
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'timestamp': ticker['timestamp']
            }
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {str(e)}")
            return {}
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch OHLCV data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for the data
            limit: Number of candles to fetch
            
        Returns:
            List of dictionaries containing OHLCV data
        """
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return [{
                'timestamp': candle[0],
                'open': candle[1],
                'high': candle[2],
                'low': candle[3],
                'close': candle[4],
                'volume': candle[5]
            } for candle in ohlcv]
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {str(e)}")
            return []
    
    def add_symbol(self, symbol: str) -> None:
        """
        Add a symbol to track.
        
        Args:
            symbol: Trading pair symbol
        """
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            logger.info(f"Added symbol {symbol}")
    
    def remove_symbol(self, symbol: str) -> None:
        """
        Remove a symbol from tracking.
        
        Args:
            symbol: Trading pair symbol
        """
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            logger.info(f"Removed symbol {symbol}")
    
    def add_timeframe(self, timeframe: str) -> None:
        """
        Add a timeframe to track.
        
        Args:
            timeframe: Timeframe to track
        """
        if timeframe not in self.timeframes:
            self.timeframes.append(timeframe)
            logger.info(f"Added timeframe {timeframe}")
    
    def remove_timeframe(self, timeframe: str) -> None:
        """
        Remove a timeframe from tracking.
        
        Args:
            timeframe: Timeframe to remove
        """
        if timeframe in self.timeframes:
            self.timeframes.remove(timeframe)
            logger.info(f"Removed timeframe {timeframe}") 