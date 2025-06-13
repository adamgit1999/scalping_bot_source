import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, UTC
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import aiohttp
from collections import deque
from decimal import Decimal
from src.models import Order, Position, Trade
from src.exceptions import InsufficientFundsError, InvalidOrderError, MarketDataError
import psutil
import uuid

class PerformanceMonitor:
    def start(self):
        pass
    def get_system_metrics(self):
        return {
            'memory_percent': psutil.virtual_memory().percent,
            'cpu_percent': psutil.cpu_percent(),
            'latency': 0  # Placeholder for latency
        }

class TradingEngine:
    def __init__(self, exchange_id: str = None, api_key: str = None, api_secret: str = None, exchange=None, broker=None, risk_manager=None, strategy=None):
        if exchange is not None:
            self.exchange = exchange
        else:
            if not exchange_id:
                raise ValueError("Either exchange object or exchange_id must be provided")
            self.exchange = getattr(ccxt, exchange_id)({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'timeout': 30000,  # 30 second timeout
                'rateLimit': 100,  # 100ms between requests
            })
        
        # Initialize core components
        self.active_trades: Dict[str, Dict] = {}
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Dict] = {}
        self.logger = logging.getLogger(__name__)
        self.active_strategies = set()
        
        # Enhanced logging setup
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Market data processing improvements
        self.market_data_cache = {}
        self.market_data_buffer = deque(maxlen=1000)
        self.buffer_size = 100
        self.last_processing_time = time.time()
        self.processing_interval = 0.05
        
        # Enhanced concurrency control
        self._processing_lock = asyncio.Lock()
        self._order_lock = asyncio.Lock()
        self._position_lock = asyncio.Lock()
        self._market_data_lock = asyncio.Lock()
        self._strategy_lock = asyncio.Lock()
        self._trade_lock = asyncio.Lock()
        self._balance_lock = asyncio.Lock()
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.metrics = {
            'processing_times': {
                'market_data': [],
                'order': [],
                'strategy': []
            },
            'error_counts': {
                'market_data_errors': 0,
                'order_errors': 0,
                'strategy_errors': 0
            },
            'system': {
                'memory_usage': 0,
                'cpu_usage': 0,
                'latency': 0
            },
            'operations': {
                'trades': 0,
                'position_updates': 0
            }
        }
        
        # Connection pooling
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Error handling
        self._market_data_errors = 0
        self._max_market_data_errors = 100
        self._error_reset_interval = 3600  # 1 hour
        self._last_error_reset = time.time()
        
        # Data validation rules
        self._data_validation_rules = {
            'price': lambda x: x > 0,
            'volume': lambda x: x >= 0,
            'bid': lambda x: x > 0,
            'ask': lambda x: x > 0,
            'timestamp': lambda x: isinstance(x, (int, float)) and x > 0
        }
        
        self.broker = broker
        self.risk_manager = risk_manager
        self.strategy = strategy
        self.trades = []
        
        # Initialize performance monitoring
        self._start_performance_monitoring()

    def _start_performance_monitoring(self) -> None:
        """Start performance monitoring."""
        try:
            # Initialize metrics structure
            self.metrics = {
                'processing_times': {
                    'market_data': [],
                    'order': [],
                    'strategy': []
                },
                'error_counts': {
                    'market_data_errors': 0,
                    'order_errors': 0,
                    'strategy_errors': 0
                },
                'system': {
                    'memory_usage': 0,
                    'cpu_usage': 0,
                    'latency': 0
                },
                'operations': {
                    'trades': 0,
                    'position_updates': 0
                }
            }
            
            # Start system metrics monitoring
            self.performance_monitor.start()
            
            # Schedule periodic metrics collection
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._collect_metrics())
            else:
                self.logger.warning("Performance monitoring not started: event loop is not running.")
                
        except Exception as e:
            self.logger.error(f"Failed to start performance monitoring: {str(e)}")

    async def _collect_metrics(self) -> None:
        """Collect performance metrics periodically."""
        while True:
            try:
                # Get system metrics
                system_metrics = self.performance_monitor.get_system_metrics()
                
                # Update metrics
                self.metrics['system'].update({
                    'memory_usage': system_metrics['memory_percent'],
                    'cpu_usage': system_metrics['cpu_percent'],
                    'latency': system_metrics['latency']
                })
                
                # Check for performance issues
                self._check_performance_issues()
                
                await asyncio.sleep(1)  # Collect metrics every second
                
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {str(e)}")
                await asyncio.sleep(5)  # Wait longer on error

    def _check_performance_issues(self) -> None:
        """Check for performance issues and take action."""
        try:
            # Check memory usage
            if self.metrics['system']['memory_usage'] > 80:
                self.logger.warning("High memory usage detected")
                self._optimize_memory_usage()
            
            # Check CPU usage
            if self.metrics['system']['cpu_usage'] > 80:
                self.logger.warning("High CPU usage detected")
                self._optimize_cpu_usage()
            
            # Check latency
            if self.metrics['system']['latency'] > 100:
                self.logger.warning("High latency detected")
                self._optimize_latency()
                
        except Exception as e:
            self.logger.error(f"Error checking performance issues: {str(e)}")

    def _optimize_memory_usage(self) -> None:
        """Optimize memory usage."""
        try:
            # Clear market data cache
            self.market_data_cache.clear()
            
            # Clear market data buffer
            self.market_data_buffer.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Error optimizing memory usage: {str(e)}")

    def _optimize_cpu_usage(self) -> None:
        """Optimize CPU usage."""
        try:
            # Adjust thread pool size
            cpu_count = psutil.cpu_count()
            current_workers = self.executor._max_workers
            cpu_percent = psutil.cpu_percent()
            
            if cpu_percent > 90 and current_workers > 1:
                self.executor._max_workers = max(1, current_workers - 1)
            elif cpu_percent < 50 and current_workers < cpu_count:
                self.executor._max_workers = min(cpu_count, current_workers + 1)
                
        except Exception as e:
            self.logger.error(f"Error optimizing CPU usage: {str(e)}")

    def _optimize_latency(self) -> None:
        """Optimize latency."""
        try:
            # Reduce processing interval
            self.processing_interval = max(0.01, self.processing_interval * 0.8)
            
            # Clear processing queue
            self.market_data_buffer.clear()
            
        except Exception as e:
            self.logger.error(f"Error optimizing latency: {str(e)}")

    async def process_market_data(self, data: Dict) -> None:
        """Process market data with enhanced error handling and performance monitoring."""
        start_time = time.time()
        
        try:
            # Validate data
            if not self._validate_market_data(data):
                self._market_data_errors += 1
                if self._market_data_errors >= self._max_market_data_errors:
                    raise MarketDataError("Too many market data errors")
                return
            
            # Process data with lock
            async with self._market_data_lock:
                # Add to buffer
                self.market_data_buffer.append(data)
                
                # Process if buffer is full or enough time has passed
                current_time = time.time()
                if (len(self.market_data_buffer) >= self.buffer_size or 
                    current_time - self.last_processing_time >= self.processing_interval):
                    await self._process_market_data_buffer()
                    self.last_processing_time = current_time
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics['processing_times']['market_data'].append(processing_time)
            
            # Check for performance issues
            if processing_time > 0.1:  # 100ms threshold
                self.logger.warning(f"Slow market data processing: {processing_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {str(e)}")
            self.metrics['error_counts']['market_data_errors'] += 1
            raise

    async def _process_market_data_buffer(self) -> None:
        """Process market data buffer with enhanced error handling."""
        try:
            # Process all data in buffer
            while self.market_data_buffer:
                data = self.market_data_buffer.popleft()
                
                # Update cache
                self.market_data_cache[data['symbol']] = data
                
                # Notify strategies
                async with self._strategy_lock:
                    for strategy in self.active_strategies:
                        try:
                            await strategy.on_market_data(data)
                        except Exception as e:
                            self.logger.error(f"Strategy error: {str(e)}")
                            self.metrics['error_counts']['strategy_errors'] += 1
                
        except Exception as e:
            self.logger.error(f"Error processing market data buffer: {str(e)}")
            raise

    def _validate_market_data(self, data: Dict) -> bool:
        """Validate market data with comprehensive checks."""
        try:
            # Check required fields
            required_fields = ['symbol', 'price', 'volume', 'timestamp']
            if not all(field in data for field in required_fields):
                self.logger.error(f"Missing required fields in market data: {data}")
                return False
            
            # Validate values
            if not isinstance(data['price'], (int, float)) or data['price'] <= 0:
                self.logger.error(f"Invalid price in market data: {data['price']}")
                return False
            
            if not isinstance(data['volume'], (int, float)) or data['volume'] < 0:
                self.logger.error(f"Invalid volume in market data: {data['volume']}")
                return False
            
            if not isinstance(data['timestamp'], (int, float)) or data['timestamp'] <= 0:
                self.logger.error(f"Invalid timestamp in market data: {data['timestamp']}")
                return False
            
            # Validate symbol format
            if not isinstance(data['symbol'], str) or '/' not in data['symbol']:
                self.logger.error(f"Invalid symbol format in market data: {data['symbol']}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Market data validation error: {str(e)}")
            return False

    async def execute_order(self, order: Order) -> None:
        """Execute order with enhanced error handling and performance monitoring."""
        start_time = time.time()

        try:
            # Validate order
            if not self._validate_order(order):
                raise InvalidOrderError("Invalid order")

            # Check balance
            if not await self._check_balance(order):
                raise InsufficientFundsError("Insufficient funds")

            # Execute order with lock
            async with self._order_lock:
                # Place order
                result = await self.broker.place_order(order)

                # Update order status
                if isinstance(result, dict):
                    order.status = result.get('status', 'UNKNOWN')
                    if 'id' in result:
                        order.id = result['id']
                else:
                    order.status = 'UNKNOWN'

                # Update position
                if order.status in ['FILLED', 'PARTIALLY_FILLED']:
                    position = Position(
                        symbol=order.symbol,
                        quantity=order.quantity,
                        entry_price=order.price,
                        current_price=order.price,
                        unrealized_pnl=Decimal('0.00'),
                        realized_pnl=Decimal('0.00')
                    )
                    await self.update_position(position)

                # Record trade
                self.record_trade(Trade(
                    order_id=order.id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    price=order.price,
                    timestamp=time.time()
                ))

                # Update metrics
                execution_time = time.time() - start_time
                self.metrics['processing_times']['order'].append(execution_time)

                # Record order in self.orders
                self.orders[order.id] = {
                    'symbol': order.symbol,
                    'side': order.side,
                    'price': float(order.price),
                    'amount': float(order.quantity),
                    'status': order.status,
                    'timestamp': time.time()
                }

                return result

        except Exception as e:
            self.logger.error(f"Error executing order: {str(e)}")
            raise

    def _validate_order(self, order: Order) -> bool:
        """Validate order with comprehensive checks."""
        try:
            # Check required fields
            if not all(hasattr(order, field) for field in ['symbol', 'side', 'price', 'quantity', 'type', 'order_type']):
                return False
            
            # Validate values
            if not (order.price > 0 and order.quantity > 0):
                return False
            
            # Validate side (normalize to uppercase)
            if order.side.upper() not in ['BUY', 'SELL']:
                return False
            
            # Validate order type
            if order.order_type.upper() not in ['MARKET', 'LIMIT']:
                return False
            
            # Validate symbol format
            if '/' not in order.symbol:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Order validation error: {str(e)}")
            return False

    async def _check_balance(self, order: Order) -> bool:
        """Check if sufficient balance is available."""
        try:
            async with self._balance_lock:
                balance = await self.broker.get_balance(order.symbol.split('/')[1])
                required = order.price * order.quantity
                return balance >= required
                
        except Exception as e:
            self.logger.error(f"Error checking balance: {str(e)}")
            return False

    async def update_position(self, position: Position) -> None:
        """Update position with enhanced error handling."""
        try:
            async with self._position_lock:
                # Update position
                self.positions[position.symbol] = position
                
                # Update metrics
                self.metrics['operations']['position_updates'] += 1
                
        except Exception as e:
            self.logger.error(f"Error updating position: {str(e)}")
            raise

    async def add_trade(self, trade: Trade) -> None:
        """Add trade with enhanced error handling."""
        try:
            async with self._trade_lock:
                # Add trade
                self.trades.append(trade)
                
                # Update metrics
                self.metrics['operations']['trades'] += 1
                
        except Exception as e:
            self.logger.error(f"Error adding trade: {str(e)}")
            raise

    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics."""
        try:
            # Ensure metrics structure exists
            if not hasattr(self, 'metrics'):
                self.metrics = {
                    'processing_times': {
                        'market_data': [],
                        'order': [],
                        'strategy': []
                    },
                    'error_counts': {
                        'market_data_errors': 0,
                        'order_errors': 0,
                        'strategy_errors': 0
                    },
                    'system': {
                        'memory_usage': 0,
                        'cpu_usage': 0,
                        'latency': 0
                    },
                    'operations': {
                        'trades': 0,
                        'position_updates': 0
                    }
                }
            
            return self.metrics
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {str(e)}")
            return {}

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
                    self.positions[position['symbol']] = Position(
                        symbol=position['symbol'],
                        quantity=Decimal(str(position['contracts'])),
                        entry_price=Decimal(str(position['entry_price'])),
                        current_price=Decimal(str(position['current_price'])),
                        unrealized_pnl=Decimal('0.00'),
                        realized_pnl=Decimal('0.00')
                    )
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
        """Execute a trading strategy with enhanced error handling and recovery."""
        try:
            start_time = time.time()
            
            # Validate strategy parameters
            if not self._validate_parameters(strategy, parameters):
                raise ValueError(f"Invalid parameters for strategy: {strategy}")
            
            # Fetch market data with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    df = await self.fetch_ohlcv(symbol, timeframe)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    self.logger.warning(f"Retry {attempt + 1}/{max_retries} fetching OHLCV data: {str(e)}")
                    await asyncio.sleep(1)
            
            # Calculate signals with performance monitoring
            signal_start_time = time.time()
            signals = self._calculate_signals(strategy, df, parameters)
            signal_time = time.time() - signal_start_time
            self.metrics['processing_times']['strategy'].append(signal_time)
            
            if signal_time > 0.1:  # Log if signal calculation takes more than 100ms
                self.logger.warning(f"Slow signal calculation: {signal_time:.3f}s")
            
            # Execute trades with error handling and position management
            for signal in signals:
                try:
                    # Check position limits before executing
                    current_position = self.get_position(symbol)
                    if current_position and abs(current_position.quantity) >= parameters.get('max_position_size', float('inf')):
                        self.logger.warning(f"Position limit reached for {symbol}")
                        continue
                    
                    # Validate signal
                    if not self._validate_signal(signal):
                        self.logger.warning(f"Invalid signal: {signal}")
                        continue
                    
                    # Execute trade with position tracking
                    if signal['action'] == 'BUY':
                        await self._execute_buy(symbol, signal['price'], signal['amount'])
                    elif signal['action'] == 'SELL':
                        await self._execute_sell(symbol, signal['price'], signal['amount'])
                    
                except Exception as e:
                    self.logger.error(f"Error executing signal: {str(e)}")
                    self.metrics['error_counts']['strategy_errors'] += 1
                    continue
            
            # Update performance metrics
            execution_time = time.time() - start_time
            self.metrics['processing_times']['strategy'].append(execution_time)
            
            if execution_time > 1.0:  # Log if total execution takes more than 1 second
                self.logger.warning(f"Slow strategy execution: {execution_time:.3f}s")
            
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

    async def place_order(self, symbol: str, side: str, price: float, amount: float) -> Dict[str, Any]:
        """Place an order with enhanced validation and error handling."""
        try:
            # Normalize and validate order side
            side = side.upper()
            if side not in ['BUY', 'SELL']:
                raise InvalidOrderError(f"Invalid order side: {side}. Must be 'BUY' or 'SELL'")

            # Validate price and amount
            if price <= 0:
                raise InvalidOrderError(f"Invalid price: {price}")
            if amount <= 0:
                raise InvalidOrderError(f"Invalid amount: {amount}")

            # Check balance for buy orders
            if side == 'BUY':
                required_funds = price * amount
                balance = await self.get_balance(self.exchange.quote_currency)
                if balance < required_funds:
                    raise InsufficientFundsError(f"Insufficient funds. Required: {required_funds}, Available: {balance}")

            # Check risk limits
            if self.risk_manager:
                is_allowed, reason = await self.risk_manager.check_order(symbol, side, price, amount)
                if not is_allowed:
                    raise ValueError("Order rejected by risk manager")

            # Create order
            order = Order(
                id=str(uuid.uuid4()),
                type='LIMIT',
                symbol=symbol,
                side=side,
                quantity=Decimal(str(amount)),
                price=Decimal(str(price)),
                order_type='LIMIT'
            )

            # Execute order
            result = await self.execute_order(order)

            # Return order details
            return {
                'id': order.id,
                'status': order.status,
                'filled_amount': float(order.quantity),
                'filled_price': float(order.price)
            }

        except Exception as e:
            self.logger.error(f"Order placement failed: {str(e)}")
            raise

    async def _update_position(self, position: Position):
        """Update position tracking."""
        async with self._position_lock:
            if position.quantity == 0:
                if position.symbol in self.positions:
                    del self.positions[position.symbol]
            else:
                self.positions[position.symbol] = position

    def record_trade(self, trade: Trade):
        """Record a trade in the trade history."""
        self.trades.append(trade)

    async def execute_strategy_signal(self, symbol: str):
        """Execute a strategy signal."""
        if not self.strategy:
            raise ValueError("No strategy provided")
        
        # Get signal
        signal = await self.strategy.generate_signal(symbol)
        if not signal:
            return None
        
        # Get entry price and levels
        entry_price = await self.strategy.calculate_entry_price(symbol)
        stop_loss = await self.strategy.calculate_stop_loss(symbol)
        take_profit = await self.strategy.calculate_take_profit(symbol)
        
        # Calculate position size
        if self.risk_manager:
            position_size = await self.risk_manager.calculate_position_size(symbol, signal, entry_price)
        else:
            position_size = Decimal('1.0')
        
        # Extract action from signal
        action = signal['action'].upper() if isinstance(signal, dict) else signal.upper()
        
        # Place order
        return await self.place_order(symbol, action, float(entry_price), float(position_size))

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)

    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        return [pos for pos in self.positions.values() if pos.quantity != 0]

    def get_trade_history(self, symbol: str) -> List[Trade]:
        """Get trade history for a symbol."""
        return [trade for trade in self.trades if trade.symbol == symbol]

    def calculate_pnl(self, symbol: str) -> Dict[str, Decimal]:
        """Calculate PnL for a position."""
        position = self.get_position(symbol)
        if not position:
            return {
                'unrealized_pnl': Decimal('0.00'),
                'realized_pnl': Decimal('0.00'),
                'total_pnl': Decimal('0.00')
            }
        
        return {
            'unrealized_pnl': position.unrealized_pnl,
            'realized_pnl': position.realized_pnl,
            'total_pnl': position.unrealized_pnl + position.realized_pnl
        }

    async def process_market_data(self, market_data: dict):
        """Process market data with enhanced error handling and validation."""
        try:
            # Validate market data
            if not all(k in market_data for k in ('symbol', 'price', 'volume', 'timestamp')):
                raise MarketDataError('Missing required market data fields')
            if market_data['price'] <= 0 or market_data['volume'] <= 0:
                raise MarketDataError('Invalid market data: price and volume must be positive')
            
            async with self._market_data_lock:
                # Reset error count if interval has passed
                current_time = time.time()
                if current_time - self._last_error_reset > self._error_reset_interval:
                    self._market_data_errors = 0
                    self._last_error_reset = current_time
                
                # Add timestamp if not present
                if 'timestamp' not in market_data:
                    market_data['timestamp'] = datetime.now(UTC).timestamp()
                
                # Add to buffer with size check
                if len(self.market_data_buffer) >= self.market_data_buffer.maxlen:
                    self.market_data_buffer.popleft()  # Remove oldest data point
                self.market_data_buffer.append(market_data)
                
                # Process buffer if it's time
                if (len(self.market_data_buffer) >= self.buffer_size or 
                    current_time - self.last_processing_time >= self.processing_interval):
                    await self._process_market_data_buffer()
                    self.last_processing_time = current_time
                    
        except Exception as e:
            self.metrics['error_counts']['market_data_errors'] += 1
            self.logger.error(f"Market data processing error: {str(e)}")
            raise

    def _validate_market_data(self, data: dict) -> bool:
        """Validate market data format and content."""
        try:
            # Check required fields
            required_fields = ['symbol', 'price', 'volume']
            if not all(field in data for field in required_fields):
                return False
            
            # Validate field values
            for field, validator in self._data_validation_rules.items():
                if field in data and not validator(data[field]):
                    return False
            
            # Validate bid/ask spread
            if 'bid' in data and 'ask' in data:
                if data['bid'] > data['ask']:
                    return False
            
            # Validate timestamp
            if 'timestamp' in data:
                if not isinstance(data['timestamp'], (int, float)) or data['timestamp'] <= 0:
                    return False
                # Check if timestamp is not too far in the future
                if data['timestamp'] > time.time() + 60:  # Allow 1 minute future tolerance
                    return False
            
            return True
            
        except Exception:
            return False

    async def _process_market_data_buffer(self):
        """Process the market data buffer efficiently."""
        try:
            start_time = time.time()
            
            # Process data in batches
            while self.market_data_buffer:
                batch = []
                for _ in range(min(self.buffer_size, len(self.market_data_buffer))):
                    if self.market_data_buffer:
                        batch.append(self.market_data_buffer.popleft())
                
                # Update cache and trigger strategy updates
                for data in batch:
                    symbol = data['symbol']
                    self.market_data_cache[symbol] = data
                    
                    # Update strategy if exists
                    if self.strategy:
                        try:
                            await self.strategy.update(data)
                        except Exception as e:
                            self.logger.error(f"Strategy update error: {str(e)}")
                            self.metrics['error_counts']['strategy_errors'] += 1
            
            processing_time = time.time() - start_time
            self.metrics['processing_times']['market_data'].append(processing_time)
            
            # Log if processing is slow
            if processing_time > 0.1:  # 100ms threshold
                self.logger.warning(f"Slow market data processing: {processing_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"Error processing market data buffer: {str(e)}")
            raise

    async def get_market_data(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> pd.DataFrame:
        """Get market data with caching and error handling."""
        try:
            cache_key = f"{symbol}_{timeframe}"
            
            # Check cache first
            if cache_key in self.market_data_cache:
                cached_data = self.market_data_cache[cache_key]
                if time.time() - cached_data['timestamp'] < 60:  # Cache valid for 1 minute
                    return cached_data['data']
            
            # Fetch new data
            ohlcv = await self._execute_with_retry(
                self.exchange.fetch_ohlcv,
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Update cache
            self.market_data_cache[cache_key] = {
                'data': df,
                'timestamp': time.time()
            }
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            raise MarketDataError(f"Failed to fetch market data: {str(e)}")

    def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics."""
        def is_number(x):
            return isinstance(x, (int, float))
        return {
            metric: {
                'count': len([t for t in times if is_number(t)]) if isinstance(times, list) else 0,
                'min': min([t for t in times if is_number(t)]) if times and any(is_number(t) for t in times) else 0,
                'max': max([t for t in times if is_number(t)]) if times and any(is_number(t) for t in times) else 0,
                'avg': sum([t for t in times if is_number(t)]) / len([t for t in times if is_number(t)]) if times and any(is_number(t) for t in times) else 0
            }
            for metric, times in self.metrics['processing_times'].items()
        }

    async def get_balance(self, currency: str) -> float:
        """Get account balance for a given currency."""
        if self.broker:
            return float(await self.broker.get_account_balance(currency))
        raise NotImplementedError("get_balance not implemented")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        if order_id in self.orders:
            del self.orders[order_id]
            return True
        raise ValueError("Order not found")

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi[:period] = np.nan
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
        """Calculate MACD indicator."""
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")
        exp1 = prices.ewm(span=fast_period, adjust=False).mean()
        exp2 = prices.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        macd[:slow_period-1] = np.nan
        signal[:slow_period-1] = np.nan
        return macd, signal

    def _calculate_commission(self, price: float, amount: float) -> float:
        """Calculate commission for a trade."""
        if amount <= 0:
            raise ValueError("Amount must be positive")
        return price * amount * 0.001  # 0.1% commission

    def _validate_parameters(self, strategy: str, parameters: dict) -> bool:
        """Validate strategy parameters with comprehensive checks."""
        try:
            # Validate strategy type
            valid_strategies = ['scalping', 'momentum', 'mean_reversion']
            if strategy not in valid_strategies:
                raise ValueError(f"Invalid strategy type: {strategy}")
            
            # Strategy-specific parameter validation
            if strategy == 'scalping':
                required_params = ['sma_short', 'sma_long', 'rsi_period', 'rsi_oversold', 'rsi_overbought']
                if not all(param in parameters for param in required_params):
                    raise ValueError("Missing required parameters for scalping strategy")
                
                # Validate parameter values
                if parameters['sma_short'] >= parameters['sma_long']:
                    raise ValueError("SMA short period must be less than SMA long period")
                if parameters['sma_short'] < 2 or parameters['sma_long'] > 200:
                    raise ValueError("Invalid SMA periods")
                if parameters['rsi_period'] < 2 or parameters['rsi_period'] > 100:
                    raise ValueError("Invalid RSI period")
                if parameters['rsi_oversold'] >= parameters['rsi_overbought']:
                    raise ValueError("RSI oversold level must be less than overbought level")
                if parameters['rsi_oversold'] < 1 or parameters['rsi_overbought'] > 99:
                    raise ValueError("Invalid RSI levels")
            
            elif strategy == 'momentum':
                required_params = ['fast_period', 'slow_period', 'signal_period']
                if not all(param in parameters for param in required_params):
                    raise ValueError("Missing required parameters for momentum strategy")
                
                # Validate parameter values
                if parameters['fast_period'] >= parameters['slow_period']:
                    raise ValueError("Fast period must be less than slow period")
                if parameters['fast_period'] < 2 or parameters['slow_period'] > 200:
                    raise ValueError("Invalid MACD periods")
                if parameters['signal_period'] < 2 or parameters['signal_period'] > 100:
                    raise ValueError("Invalid signal period")
            
            elif strategy == 'mean_reversion':
                required_params = ['lookback_period', 'entry_std', 'exit_std']
                if not all(param in parameters for param in required_params):
                    raise ValueError("Missing required parameters for mean reversion strategy")
                
                # Validate parameter values
                if parameters['lookback_period'] < 2 or parameters['lookback_period'] > 200:
                    raise ValueError("Invalid lookback period")
                if parameters['entry_std'] <= 0 or parameters['exit_std'] <= 0:
                    raise ValueError("Standard deviation parameters must be positive")
                if parameters['entry_std'] <= parameters['exit_std']:
                    raise ValueError("Entry standard deviation must be greater than exit standard deviation")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Strategy parameter validation error: {str(e)}")
            raise ValueError(str(e))

    async def close_all_positions(self):
        """Close all open positions."""
        for symbol, position in list(self.positions.items()):
            if position.quantity > 0:
                await self.place_order(symbol, 'sell', (await self.exchange.fetch_ticker(symbol))['last'], float(position.quantity))
            elif position.quantity < 0:
                await self.place_order(symbol, 'buy', (await self.exchange.fetch_ticker(symbol))['last'], float(abs(position.quantity)))
        self.positions.clear()

    async def update_position(self, position):
        """Update or add a position."""
        if hasattr(position, 'symbol') and hasattr(position, 'quantity'):
            if float(position.quantity) == 0.0:
                self.positions.pop(position.symbol, None)
            else:
                self.positions[position.symbol] = position
        else:
            raise ValueError("Invalid position object")

    async def _execute_buy(self, symbol: str, price: float, amount: float):
        """Execute buy order with position tracking."""
        try:
            # Check if we have sufficient funds
            required_funds = price * amount
            balance = await self.get_balance(self.exchange.quote_currency)
            if balance < required_funds:
                raise InsufficientFundsError(f"Insufficient funds. Required: {required_funds}, Available: {balance}")
            
            # Place order
            order = await self.place_order(symbol, 'BUY', price, amount)
            
            # Update position
            if order['status'] == 'closed':
                await self._update_position(Position(
                    symbol=symbol,
                    quantity=Decimal(str(amount)),
                    entry_price=Decimal(str(price)),
                    current_price=Decimal(str(price)),
                    unrealized_pnl=Decimal('0.00'),
                    realized_pnl=Decimal('0.00')
                ))
            
            return order
            
        except Exception as e:
            self.logger.error(f"Buy execution failed: {str(e)}")
            raise

    async def _execute_sell(self, symbol: str, price: float, amount: float):
        """Execute sell order with position tracking."""
        try:
            # Check if we have sufficient position
            current_position = self.get_position(symbol)
            if not current_position or current_position.quantity < amount:
                raise InvalidOrderError(f"Insufficient position. Required: {amount}, Available: {current_position.quantity if current_position else 0}")
            
            # Place order
            order = await self.place_order(symbol, 'SELL', price, amount)
            
            # Update position
            if order['status'] == 'closed':
                new_quantity = current_position.quantity - Decimal(str(amount))
                if new_quantity == 0:
                    await self._update_position(Position(
                        symbol=symbol,
                        quantity=Decimal('0'),
                        entry_price=Decimal('0'),
                        current_price=Decimal('0'),
                        unrealized_pnl=Decimal('0.00'),
                        realized_pnl=current_position.unrealized_pnl
                    ))
                else:
                    await self._update_position(Position(
                        symbol=symbol,
                        quantity=new_quantity,
                        entry_price=current_position.entry_price,
                        current_price=Decimal(str(price)),
                        unrealized_pnl=current_position.unrealized_pnl,
                        realized_pnl=current_position.realized_pnl
                    ))
            
            return order
            
        except Exception as e:
            self.logger.error(f"Sell execution failed: {str(e)}")
            raise

    def _validate_signal(self, signal: Dict) -> bool:
        """Validate trading signal."""
        try:
            required_fields = ['action', 'price', 'amount']
            if not all(field in signal for field in required_fields):
                return False
            
            if signal['action'] not in ['BUY', 'SELL']:
                return False
            
            if not isinstance(signal['price'], (int, float)) or signal['price'] <= 0:
                return False
            
            if not isinstance(signal['amount'], (int, float)) or signal['amount'] <= 0:
                return False
            
            return True
            
        except Exception:
            return False 