from typing import Dict, Optional, List, Tuple
from decimal import Decimal
import ccxt
import asyncio
import logging
from src.exceptions import InvalidOrderError, InsufficientFundsError, ConnectionError
import time

logger = logging.getLogger(__name__)

class Broker:
    def __init__(self, exchange_id: str, api_key: str, api_secret: str):
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange = getattr(ccxt, exchange_id)({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'timeout': 30000,  # 30 second timeout
            'rateLimit': 100,  # 100ms between requests
        })
        self.markets = {}
        self.balances = {}
        self.connected = False
        self.last_heartbeat = 0
        self.heartbeat_interval = 30  # seconds
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        self._connection_lock = asyncio.Lock()
        
    async def initialize(self) -> None:
        """Initialize the broker with retry logic and connection management."""
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                async with self._connection_lock:
                    # Load markets
                    self.markets = await self._execute_with_retry(self.exchange.load_markets)
                    
                    # Load balances
                    self.balances = await self._execute_with_retry(self.exchange.fetch_balance)
                    
                    self.connected = True
                    self.last_heartbeat = time.time()
                    logger.info(f"Successfully connected to {self.exchange_id}")
                    return
                    
            except Exception as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    logger.error(f"Failed to initialize broker after {self.max_retries} attempts: {str(e)}")
                    raise ConnectionError(f"Failed to initialize broker: {str(e)}")
                logger.warning(f"Retry {retry_count}/{self.max_retries} initializing broker: {str(e)}")
                await asyncio.sleep(self.retry_delay * (2 ** (retry_count - 1)))
                
    async def _execute_with_retry(self, func, *args, **kwargs) -> Any:
        """Execute a function with retry logic."""
        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retries:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count >= self.max_retries:
                    break
                logger.warning(f"Retry {retry_count}/{self.max_retries} executing {func.__name__}: {str(e)}")
                await asyncio.sleep(self.retry_delay * (2 ** (retry_count - 1)))
                
        raise last_error
        
    async def check_connection(self) -> bool:
        """Check broker connection and attempt reconnection if needed."""
        if not self.connected or time.time() - self.last_heartbeat > self.heartbeat_interval:
            try:
                async with self._connection_lock:
                    # Try to fetch time as a lightweight connection check
                    await self._execute_with_retry(self.exchange.fetch_time)
                    self.connected = True
                    self.last_heartbeat = time.time()
                    return True
            except Exception as e:
                logger.error(f"Connection check failed: {str(e)}")
                self.connected = False
                return False
        return True
        
    async def place_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None) -> Dict:
        """Place an order with enhanced error handling."""
        if not await self.check_connection():
            raise ConnectionError("Broker not connected")
            
        try:
            # Validate inputs
            if side.upper() not in ['BUY', 'SELL']:
                raise InvalidOrderError(f"Invalid order side: {side}")
            if amount <= 0:
                raise InvalidOrderError(f"Invalid amount: {amount}")
            if price is not None and price <= 0:
                raise InvalidOrderError(f"Invalid price: {price}")
                
            # Check balance
            if side.upper() == 'BUY':
                required_balance = amount * (price or await self.get_current_price(symbol))
                if not self._check_balance(required_balance):
                    raise InsufficientFundsError(f"Insufficient balance for order: {required_balance}")
                    
            # Place order
            order = await self._execute_with_retry(
                self.exchange.create_order,
                symbol=symbol,
                type='limit' if price else 'market',
                side=side.lower(),
                amount=amount,
                price=price
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to place order: {str(e)}")
            raise
            
    def _check_balance(self, required: float) -> bool:
        """Check if sufficient balance is available."""
        try:
            total_balance = float(self.balances.get('total', {}).get('USD', 0))
            return total_balance >= required
        except Exception as e:
            logger.error(f"Balance check failed: {str(e)}")
            return False
            
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        try:
            ticker = await self._execute_with_retry(self.exchange.fetch_ticker, symbol)
            return float(ticker['last'])
        except Exception as e:
            logger.error(f"Failed to get current price: {str(e)}")
            raise
            
    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None) -> Dict:
        """Create a new order."""
        try:
            # Validate inputs
            if side.upper() not in ['BUY', 'SELL']:
                raise InvalidOrderError(f"Invalid order side: {side}")
            if amount <= 0:
                raise InvalidOrderError(f"Invalid amount: {amount}")
            if price is not None and price <= 0:
                raise InvalidOrderError(f"Invalid price: {price}")
                
            # Check if we have sufficient funds
            if side.upper() == 'BUY':
                required_funds = amount * (price or self.get_current_price(symbol))
                if required_funds > self.get_available_balance(self.markets[symbol]['quote']):
                    raise InsufficientFundsError(f"Insufficient funds for buy order: {required_funds}")
                    
            # Create the order
            order = await self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side.lower(),
                amount=amount,
                price=price
            )
            
            return order
            
        except Exception as e:
            print(f"Order creation failed: {str(e)}")
            raise
            
    async def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """Cancel an existing order."""
        try:
            return await self.exchange.cancel_order(order_id, symbol)
        except Exception as e:
            print(f"Order cancellation failed: {str(e)}")
            raise
            
    async def fetch_order(self, order_id: str, symbol: str) -> Dict:
        """Fetch order details."""
        try:
            return await self.exchange.fetch_order(order_id, symbol)
        except Exception as e:
            print(f"Order fetch failed: {str(e)}")
            raise
            
    async def fetch_orders(self, symbol: str, since: Optional[int] = None, limit: Optional[int] = None) -> List[Dict]:
        """Fetch orders for a symbol."""
        try:
            return await self.exchange.fetch_orders(symbol, since, limit)
        except Exception as e:
            print(f"Orders fetch failed: {str(e)}")
            raise
            
    def get_available_balance(self, currency: str) -> float:
        """Get available balance for a currency."""
        try:
            return float(self.balances.get(currency, {}).get('free', 0))
        except Exception as e:
            print(f"Balance fetch failed: {str(e)}")
            raise 