import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, UTC
import logging
from typing import Dict, List, Optional, Tuple
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import aiohttp
from collections import deque
from decimal import Decimal

class TradingEngine:
    def __init__(self, exchange_id: str = None, api_key: str = None, api_secret: str = None, exchange=None, broker=None, risk_manager=None, strategy=None):
        if exchange is not None:
            self.exchange = exchange
        else:
            self.exchange = getattr(ccxt, exchange_id)({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'timeout': 30000,  # 30 second timeout
                'rateLimit': 100,  # 100ms between requests
            })
        
        # Initialize core components
        self.active_trades: Dict[str, Dict] = {}
        self.positions: Dict[str, float] = {}
        self.orders: Dict[str, Dict] = {}
        self.logger = logging.getLogger(__name__)
        self.active_strategies = set()
        
        # Performance optimizations
        self.market_data_cache = {}
        self.market_data_buffer = deque(maxlen=1000)  # Use deque for efficient buffer operations
        self.buffer_size = 100
        self.last_processing_time = time.time()
        self.processing_interval = 0.05  # Process buffer every 50ms
        self._processing_lock = asyncio.Lock()
        self._order_lock = asyncio.Lock()
        self._position_lock = asyncio.Lock()
        
        # Connection pooling
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance monitoring
        self.metrics = {
            'order_processing_time': [],
            'market_data_processing_time': [],
            'strategy_execution_time': []
        }

        self.broker = broker
        self.risk_manager = risk_manager
        self.strategy = strategy
        self.trades = []

    async def initialize(self):
        """Initialize the trading engine and load markets."""
        try:
            # Initialize HTTP session
            self.session = aiohttp.ClientSession()
            
            # Load markets
            await self.exchange.load_markets()
            
            # Initialize position tracking
            await self._initialize_positions()
            
            self.logger.info("Trading engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize trading engine: {str(e)}")
            raise

    async def _initialize_positions(self):
        """Initialize position tracking from exchange."""
        try:
            positions = await self.exchange.fetch_positions()
            for position in positions:
                if float(position['contracts']) != 0:
                    self.positions[position['symbol']] = float(position['contracts'])
        except Exception as e:
            self.logger.error(f"Failed to initialize positions: {str(e)}")
            raise

    @lru_cache(maxsize=1000)
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol with caching."""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            self.logger.error(f"Failed to fetch OHLCV data: {str(e)}")
            raise

    async def execute_strategy(self, strategy: str, symbol: str, timeframe: str, parameters: Dict):
        """Execute a trading strategy."""
        try:
            # Fetch market data
            df = await self.fetch_ohlcv(symbol, timeframe)
            
            # Calculate indicators based on strategy
            signals = self._calculate_signals(strategy, df, parameters)
            
            # Measure strategy execution time
            start_time = time.time()
            # Execute trades based on signals
            for signal in signals:
                if signal['action'] == 'buy':
                    await self.place_order(symbol, 'buy', signal['price'], signal['amount'])
                elif signal['action'] == 'sell':
                    await self.place_order(symbol, 'sell', signal['price'], signal['amount'])
            exec_time = time.time() - start_time
            self.metrics['strategy_execution_time'].append(exec_time)
        except Exception as e:
            self.logger.error(f"Strategy execution failed: {str(e)}")
            raise

    def _calculate_signals(self, strategy: str, df: pd.DataFrame, parameters: Dict) -> List[Dict]:
        """Calculate trading signals based on the selected strategy."""
        signals = []
        
        if strategy == 'scalping':
            signals = self._scalping_strategy(df, parameters)
        elif strategy == 'momentum':
            signals = self._momentum_strategy(df, parameters)
        elif strategy == 'mean_reversion':
            signals = self._mean_reversion_strategy(df, parameters)
            
        return signals

    def _scalping_strategy(self, df: pd.DataFrame, parameters: Dict) -> List[Dict]:
        """Implement scalping strategy logic."""
        signals = []
        
        # Calculate indicators
        df['sma_short'] = df['close'].rolling(window=parameters.get('sma_short', 10)).mean()
        df['sma_long'] = df['close'].rolling(window=parameters.get('sma_long', 20)).mean()
        df['rsi'] = self._calculate_rsi(df['close'], parameters.get('rsi_period', 14))
        
        # Generate signals
        for i in range(len(df)):
            if i < parameters.get('sma_long', 20):
                continue
                
            if (df['sma_short'].iloc[i] > df['sma_long'].iloc[i] and 
                df['rsi'].iloc[i] < parameters.get('rsi_oversold', 30)):
                signals.append({
                    'action': 'buy',
                    'price': df['close'].iloc[i],
                    'amount': parameters.get('position_size', 0.1)
                })
            elif (df['sma_short'].iloc[i] < df['sma_long'].iloc[i] and 
                  df['rsi'].iloc[i] > parameters.get('rsi_overbought', 70)):
                signals.append({
                    'action': 'sell',
                    'price': df['close'].iloc[i],
                    'amount': parameters.get('position_size', 0.1)
                })
                
        return signals

    def _momentum_strategy(self, df: pd.DataFrame, parameters: Dict) -> List[Dict]:
        """Implement momentum strategy logic."""
        signals = []
        
        # Calculate indicators
        df['roc'] = df['close'].pct_change(parameters.get('roc_period', 10))
        df['macd'], df['signal'] = self._calculate_macd(df['close'])
        
        # Generate signals
        for i in range(len(df)):
            if i < parameters.get('roc_period', 10):
                continue
                
            if (df['roc'].iloc[i] > parameters.get('roc_threshold', 0.02) and 
                df['macd'].iloc[i] > df['signal'].iloc[i]):
                signals.append({
                    'action': 'buy',
                    'price': df['close'].iloc[i],
                    'amount': parameters.get('position_size', 0.1)
                })
            elif (df['roc'].iloc[i] < -parameters.get('roc_threshold', 0.02) and 
                  df['macd'].iloc[i] < df['signal'].iloc[i]):
                signals.append({
                    'action': 'sell',
                    'price': df['close'].iloc[i],
                    'amount': parameters.get('position_size', 0.1)
                })
                
        return signals

    def _mean_reversion_strategy(self, df: pd.DataFrame, parameters: Dict) -> List[Dict]:
        """Implement mean reversion strategy logic."""
        signals = []
        
        # Calculate indicators
        df['sma'] = df['close'].rolling(window=parameters.get('sma_period', 20)).mean()
        df['std'] = df['close'].rolling(window=parameters.get('sma_period', 20)).std()
        df['upper_band'] = df['sma'] + parameters.get('std_multiplier', 2) * df['std']
        df['lower_band'] = df['sma'] - parameters.get('std_multiplier', 2) * df['std']
        
        # Generate signals
        for i in range(len(df)):
            if i < parameters.get('sma_period', 20):
                continue
                
            if df['close'].iloc[i] < df['lower_band'].iloc[i]:
                signals.append({
                    'action': 'buy',
                    'price': df['close'].iloc[i],
                    'amount': parameters.get('position_size', 0.1)
                })
            elif df['close'].iloc[i] > df['upper_band'].iloc[i]:
                signals.append({
                    'action': 'sell',
                    'price': df['close'].iloc[i],
                    'amount': parameters.get('position_size', 0.1)
                })
                
        return signals

    async def place_order(self, symbol: str, side: str, price: float, amount: float) -> Dict:
        """Place a new order with optimized error handling and validation."""
        start_time = time.time()
        try:
            async with self._order_lock:
                # Risk check first
                if self.risk_manager and not self.risk_manager.check_order(symbol, side, price, amount):
                    raise ValueError("Order rejected by risk manager")
                
                # Validate inputs
                if not symbol or not side or price <= 0 or amount <= 0:
                    raise ValueError("Invalid order parameters")
                
                # Check if symbol exists
                if symbol not in self.exchange.markets:
                    raise ValueError(f"Invalid symbol: {symbol}")
                
                # Check balance for buy orders
                if side.lower() == 'buy':
                    quote_currency = symbol.split('/')[1]
                    balance = await self.get_balance(quote_currency)
                    required_balance = price * amount
                    if balance < required_balance:
                        raise ValueError(f"Insufficient balance for {quote_currency}")
                
                # Place order
                order = await self.exchange.create_order(symbol, side, price, amount)
                
                # Update order tracking
                self.orders[order['id']] = order
                
                # Update trade tracking
                self.active_trades[order['id']] = {
                    'symbol': symbol,
                    'side': side,
                    'price': price,
                    'amount': amount,
                    'order_id': order['id'],
                    'status': order['status']
                }
                
                # Update position tracking
                if side.lower() == 'buy':
                    self.positions[symbol] += amount
                elif side.lower() == 'sell':
                    self.positions[symbol] -= amount
                
                # Update metrics
                self.metrics['order_processing_time'].append(time.time() - start_time)
                
                return order
        except Exception as e:
            self.logger.error(f"Failed to place order: {str(e)}")
            raise 