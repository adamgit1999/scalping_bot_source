import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple, Callable, Set, Awaitable
import logging
from dataclasses import dataclass
import threading
from queue import PriorityQueue
import numpy as np
from decimal import Decimal, ROUND_DOWN
import uuid
from enum import Enum
import orjson
from concurrent.futures import ThreadPoolExecutor
from .exceptions import OrderError, ValidationError, ExecutionError
import weakref
import signal
import sys
from contextlib import asynccontextmanager
import concurrent.futures
from src.exceptions import (
    TradingError,
    ExchangeError,
    ExecutionError,
    ValidationError,
    InsufficientFundsError,
    InvalidOrderError
)
import gc
import collections

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order statuses."""
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class Order:
    """Order structure."""
    id: str
    symbol: str
    type: OrderType
    side: OrderSide
    quantity: Decimal
    price: Optional[Decimal]
    stop_price: Optional[Decimal]
    status: OrderStatus
    filled_quantity: Decimal
    average_price: Decimal
    created_at: float
    updated_at: float
    client_order_id: Optional[str]
    post_only: bool
    reduce_only: bool
    time_in_force: str
    expire_time: Optional[float]
    retry_count: int = 0  # Add retry count

    def __init__(self, id: str, symbol: str, type: OrderType, side: OrderSide,
                 quantity: Decimal, price: Optional[Decimal] = None,
                 stop_price: Optional[Decimal] = None, status: OrderStatus = OrderStatus.NEW,
                 filled_quantity: Decimal = Decimal("0"),
                 average_price: Decimal = Decimal("0"),
                 created_at: float = None, updated_at: float = None,
                 client_order_id: Optional[str] = None,
                 post_only: bool = False, reduce_only: bool = False,
                 time_in_force: str = "GTC", expire_time: Optional[float] = None,
                 retry_count: int = 0):
        """
        Initialize order.
        
        Args:
            id: Order ID
            symbol: Trading pair
            type: Order type
            side: Order side
            quantity: Order quantity
            price: Order price (for limit orders)
            stop_price: Stop price (for stop orders)
            status: Order status
            filled_quantity: Filled quantity
            average_price: Average fill price
            created_at: Creation timestamp
            updated_at: Last update timestamp
            client_order_id: Client order ID
            post_only: Post-only flag
            reduce_only: Reduce-only flag
            time_in_force: Time in force
            expire_time: Expiration timestamp
            retry_count: Retry count
        """
        self.id = id
        self.symbol = symbol
        self.type = type
        self.side = side
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.status = status
        self.filled_quantity = filled_quantity
        self.average_price = average_price
        self.created_at = created_at or time.time()
        self.updated_at = updated_at or time.time()
        self.client_order_id = client_order_id
        self.post_only = post_only
        self.reduce_only = reduce_only
        self.time_in_force = time_in_force
        self.expire_time = expire_time
        self.retry_count = retry_count

    def __lt__(self, other):
        """Compare orders for priority queue ordering."""
        if not isinstance(other, Order):
            return NotImplemented
        return self.created_at < other.created_at

    def __eq__(self, other):
        """Compare orders for equality."""
        if not isinstance(other, Order):
            return NotImplemented
        return self.id == other.id

    def __hash__(self):
        """Hash order for dictionary keys."""
        return hash(self.id)

class OrderEngine:
    """Order engine for managing order execution and lifecycle."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 0.1, max_errors: int = 10, error_reset_interval: float = 60.0, max_stored_orders: int = 2000):
        """
        Initialize order engine.
        
        Args:
            max_retries: Maximum number of retries for failed orders
            retry_delay: Delay between retries in seconds
            max_errors: Maximum number of errors before reset
            error_reset_interval: Interval in seconds to reset error count
            max_stored_orders: Maximum number of completed orders to keep in memory (increase for high-load tests)
        """
        self._running = False
        self._active_orders = {}
        self._order_lock = asyncio.Lock()
        self._error_lock = asyncio.Lock()
        self._tasks = set()
        self._cleanup_handlers = []
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._execution_times = collections.deque(maxlen=1000)  # Limit stored execution times
        self._max_stats_samples = 1000
        self._gc_counter = 0
        self._gc_interval = 100  # Run GC every 100 operations
        self._max_stored_orders = max_stored_orders
        self._completed_orders = collections.deque(maxlen=self._max_stored_orders)
        self.callbacks = {}
        self.error_count = 0
        self.last_error_reset = time.time()
        self.max_errors = max_errors
        self.error_reset_interval = error_reset_interval
        self._order_queue = asyncio.PriorityQueue()
        self._execution_queue = asyncio.Queue()
        self._last_cleanup = time.time()
        self._cleanup_interval = 60.0  # Run cleanup every 60 seconds
        self._current_prices = {}  # Store current prices for stop orders
        self._setup_signal_handlers()

    @property
    def running(self):
        """Get running state."""
        return self._running

    @running.setter
    def running(self, value: bool):
        """Set running state."""
        self._running = value

    async def start(self) -> None:
        """Start order engine."""
        if self._running:
            return
            
        logger.info("OrderEngine.start() called")
        self._running = True
        
        # Start task monitor
        self._tasks.add(asyncio.create_task(self._monitor_tasks()))
        
        # Start order processor
        self._tasks.add(asyncio.create_task(self._process_orders()))
        
        # Start execution processor
        self._tasks.add(asyncio.create_task(self._process_executions()))
        
        logger.info("OrderEngine started successfully")
        
    async def _monitor_tasks(self) -> None:
        """Monitor tasks and clean up completed/failed tasks."""
        while self._running:
            # Get completed tasks
            done = set()
            for task in self._tasks:
                if task.done():
                    try:
                        # Get result to propagate any exceptions
                        task.result()
                    except Exception as e:
                        logger.error(f"Task failed: {e}")
                    done.add(task)
            
            # Remove completed tasks
            self._tasks -= done
            
            # Wait a bit before next check
            await asyncio.sleep(0.1)
            
    async def stop(self) -> None:
        """
        Stop order engine.
        """
        logger.info("Stopping order engine")
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
        
        # Clear tasks
        self._tasks.clear()
        
        # Run cleanup handlers
        for handler in self._cleanup_handlers:
            try:
                await handler()
            except Exception as e:
                logger.error(f"Error in cleanup handler: {str(e)}")
        
        # Clear handlers
        self._cleanup_handlers.clear()
        
        # Clear orders
        self._active_orders.clear()
        self._completed_orders.clear()
        self._current_prices.clear()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Order engine stopped")

    async def _process_orders(self) -> None:
        """
        Process orders from queue.
        """
        logger.info("Starting _process_orders")
        while self._running:
            try:
                priority, timestamp, order = await self._order_queue.get()
                # Check if order is still valid
                if order.status not in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                    self._order_queue.task_done()
                    continue
                # Check expiration
                if order.expire_time and time.time() > order.expire_time:
                    self._update_order_status(order, OrderStatus.EXPIRED)
                    self._order_queue.task_done()
                    continue
                # For STOP/STOP_LIMIT, only execute if stop condition is met
                if order.type in (OrderType.STOP, OrderType.STOP_LIMIT):
                    if not self._check_stop_condition(order):
                        # Requeue with same priority
                        await self._order_queue.put((priority, timestamp, order))
                        self._order_queue.task_done()
                        await asyncio.sleep(0.01)
                        continue
                await self._execute_order(order)
                self._order_queue.task_done()
                # Run cleanup if needed
                current_time = time.time()
                if current_time - self._last_cleanup > self._cleanup_interval:
                    await self._cleanup()
                    self._last_cleanup = current_time
            except Exception as e:
                logger.error(f"Error processing order: {str(e)}")
                async with self._error_lock:
                    self.error_count += 1
                    if self.error_count >= self.max_errors:
                        logger.error("Maximum error count exceeded, stopping order engine")
                        await self.stop()
                        break

    async def _process_executions(self) -> None:
        """Process executions from the queue."""
        print("Starting _process_executions")
        while self._running:
            try:
                # Get execution from queue with timeout
                execution = await asyncio.wait_for(self._execution_queue.get(), timeout=0.1)
                print(f"Processing execution: {execution}")
                
                try:
                    # Process execution
                    await self._handle_execution(execution)
                except Exception as e:
                    print(f"Error processing execution: {str(e)}")
                    # Add to error queue or handle error
                finally:
                    self._execution_queue.task_done()
                    
            except asyncio.TimeoutError:
                # No executions in queue, continue
                continue
            except Exception as e:
                print(f"Error in _process_executions: {str(e)}")
                if not self._running:
                    break
                await asyncio.sleep(0.1)  # Prevent tight loop on error
                
        print("_process_executions completed")

    async def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle signal."""
        logger.info(f"Received signal {sig.name}")
        await self.stop()

    def register_cleanup_handler(self, handler: Callable[[], Awaitable[None]]) -> None:
        """
        Register cleanup handler.
        
        Args:
            handler: Cleanup handler
        """
        self._cleanup_handlers.append(handler)

    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register callback for event.
        
        Args:
            event: Event name
            callback: Callback function
        """
        if event not in self.callbacks:
            self.callbacks[event] = []
        if callback not in self.callbacks[event]:
            self.callbacks[event].append(callback)

    async def _handle_error(self, error: Exception, context: str) -> None:
        """Handle error."""
        async with self._error_lock:
            self.error_count += 1
            
            # Reset error count periodically
            current_time = time.time()
            if current_time - self.last_error_reset > self.error_reset_interval:
                self.error_count = 0
                self.last_error_reset = current_time
                
            # Notify error
            self._notify_error(error)
            
            # Log error
            logger.error(f"Error in {context}: {str(error)}")
            
            # Check if we've exceeded max errors
            if self.error_count >= self.max_errors:
                logger.error("Maximum error count exceeded, stopping order engine")
                await self.stop()
                raise RuntimeError("Order engine stopped due to excessive errors")

    def _notify_error(self, error: Exception) -> None:
        """Notify error callback."""
        if 'error' in self.callbacks:
            callbacks = self.callbacks['error']
            if callable(callbacks):
                callbacks = [callbacks]
            for callback in callbacks:
                try:
                    callback(error)
                except Exception as e:
                    logger.error(f"Error in error callback: {str(e)}")

    async def _execute_order(self, order: Order) -> None:
        retries = 0
        while retries <= self._max_retries:
            # Atomically check status under lock to prevent race with cancellation
            async with self._order_lock:
                if order.status == OrderStatus.CANCELED:
                    return
            try:
                # Check if order is still valid
                if order.status not in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                    return
                # Check expiration right before execution
                if order.expire_time and time.time() > order.expire_time:
                    self._update_order_status(order, OrderStatus.EXPIRED)
                    return
                # Reduce-only logic: allow SELL if position > 0, BUY if position < 0
                position = None
                if order.reduce_only:
                    get_pos = self._get_position
                    if hasattr(get_pos, "__call__"):
                        pos = get_pos(order.symbol)
                        if asyncio.iscoroutine(pos):
                            position = await pos
                        else:
                            position = pos
                    if (order.side == OrderSide.SELL and position <= 0) or (order.side == OrderSide.BUY and position >= 0):
                        self._update_order_status(order, OrderStatus.REJECTED)
                        return
                start_time = time.time()
                try:
                    result = await self._send_to_exchange(order)
                except ExecutionError as e:
                    if order.post_only and "Post-only order would have been filled immediately" in str(e):
                        self._update_order_status(order, OrderStatus.REJECTED)
                        return
                    raise
                execution_time = time.time() - start_time
                self._update_execution_stats(execution_time)
                if result:
                    status = result.get('status', 'filled')
                    # Check expiration again before filling
                    if order.expire_time and time.time() > order.expire_time:
                        self._update_order_status(order, OrderStatus.EXPIRED)
                        return
                    if status == 'filled':
                        order.filled_quantity = Decimal(str(result.get('filled_quantity', order.quantity)))
                        order.average_price = Decimal(str(result.get('average_price', order.price or 0)))
                        self._update_order_status(order, OrderStatus.FILLED)
                    elif status == 'partially_filled':
                        order.filled_quantity = Decimal(str(result.get('filled_quantity', 0)))
                        order.average_price = Decimal(str(result.get('average_price', order.price or 0)))
                        self._update_order_status(order, OrderStatus.PARTIALLY_FILLED)
                    self._notify_execution(order, result)
                    return
                else:
                    retries += 1
                    if retries > self._max_retries:
                        self._update_order_status(order, OrderStatus.REJECTED)
                        return
                    await asyncio.sleep(self._retry_delay)
            except Exception as e:
                # Always call error callbacks for all exceptions
                if 'error' in self.callbacks:
                    callbacks = self.callbacks['error']
                    if callable(callbacks):
                        callbacks = [callbacks]
                    for callback in callbacks:
                        try:
                            callback(e)
                        except Exception as err:
                            logger.error(f"Error in error callback: {str(err)}")
                retries += 1
                if retries > self._max_retries:
                    self._update_order_status(order, OrderStatus.REJECTED)
                    return
                await asyncio.sleep(self._retry_delay)

    def _can_execute(self, order: Order) -> bool:
        """Check if order can be executed."""
        # Check if order exists
        if order.id not in self._active_orders:
            return False
            
        # Check if order is already filled or canceled
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED):
            return False
            
        # Check if order is expired
        if order.expire_time and time.time() > order.expire_time:
            self._update_order_status(order, OrderStatus.EXPIRED)
            return False
            
        # Check reduce-only orders
        if order.reduce_only:
            position = self._get_position(order.symbol)
            if (order.side == OrderSide.BUY and position >= 0) or \
               (order.side == OrderSide.SELL and position <= 0):
                self._update_order_status(order, OrderStatus.REJECTED)
                return False
                
        # Check stop orders: only executable if stop condition is met
        if order.type in (OrderType.STOP, OrderType.STOP_LIMIT):
            if not self._check_stop_condition(order):
                return False
                
        return True
        
    def _check_stop_condition(self, order: Order) -> bool:
        """
        Check if stop condition is met.
        
        Args:
            order: Order to check
            
        Returns:
            bool: True if stop condition is met
        """
        if not order.stop_price:
            return True
        current_price = self._current_prices.get(order.symbol, Decimal("0"))
        if order.side == OrderSide.BUY:
            return current_price >= order.stop_price
        else:
            return current_price <= order.stop_price
            
    def _get_market_price(self, symbol: str) -> Optional[Decimal]:
        if hasattr(self, '_test_market_price') and self._test_market_price is not None:
            return self._test_market_price
        return Decimal("50000.0")
        
    async def _process_stop_orders(self, price: Decimal) -> None:
        """
        Process stop orders.
        
        Args:
            price: Current market price
        """
        for order in list(self._active_orders.values()):
            if order.type in (OrderType.STOP, OrderType.STOP_LIMIT) and \
               order.status == OrderStatus.NEW:
                # Check expiration before filling
                if order.expire_time and time.time() > order.expire_time:
                    self._update_order_status(order, OrderStatus.EXPIRED)
                    continue
                # Check stop condition
                if self._check_stop_condition(order):
                    # Execute order
                    await self._execute_order(order)
                    
    async def _cleanup(self) -> None:
        """
        Clean up old orders and run garbage collection.
        """
        current_time = time.time()
        async with self._order_lock:
            # Only remove completed orders older than 1 hour if we exceed max_stored_orders
            if len(self._completed_orders) > self._max_stored_orders:
                self._completed_orders = collections.deque(
                    (order for order in self._completed_orders if current_time - order.updated_at < 3600),
                    maxlen=self._max_stored_orders
                )
            # Remove old active orders
            orders_to_remove = []
            for order_id, order in self._active_orders.items():
                if current_time - order.updated_at > 3600:
                    orders_to_remove.append(order_id)
            for order_id in orders_to_remove:
                self._active_orders.pop(order_id, None)
        # Clear price cache
        self._current_prices.clear()
        # Force garbage collection
        gc.collect()

    def _validate_order(self, order: Order) -> bool:
        """
        Validate order.
        
        Args:
            order: Order to validate
            
        Returns:
            bool: True if order is valid
        """
        # Check required fields
        if not order.symbol or not order.quantity:
            logger.error(f"Invalid order: missing required fields - {order}")
            return False
            
        # Check quantity
        if order.quantity <= 0:
            logger.error(f"Invalid order: quantity must be positive - {order}")
            return False
            
        # Check price for limit orders
        if order.type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and not order.price:
            logger.error(f"Invalid order: limit orders must have price - {order}")
            return False
            
        # Check stop price for stop orders
        if order.type in (OrderType.STOP, OrderType.STOP_LIMIT) and not order.stop_price:
            logger.error(f"Invalid order: stop orders must have stop price - {order}")
            return False
            
        return True

    def _update_order_status(self, order: Order, status: OrderStatus) -> None:
        """
        Update order status and notify callbacks.
        """
        order.status = status
        order.updated_at = time.time()
        self._notify_order_update(order)
        # Move completed orders to completed_orders deque
        if status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            self._completed_orders.append(order)
            # Remove from active orders if present
            self._active_orders.pop(order.id, None)
        
        # Increment GC counter and run GC if needed
        self._gc_counter += 1
        if self._gc_counter >= self._gc_interval:
            self._gc_counter = 0
            gc.collect()

    async def _process_market_data(self, data: Dict) -> None:
        """
        Process market data.
        
        Args:
            data: Market data
        """
        try:
            # Notify callbacks
            if 'market_data' in self.callbacks:
                callbacks = self.callbacks['market_data']
                if callable(callbacks):
                    callbacks = [callbacks]
                for callback in callbacks:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"Error in market data callback: {str(e)}")
            # Check for order fills
            symbol = data.get('symbol')
            price = Decimal(str(data.get('price', 0)))
            for order in list(self._active_orders.values()):
                if (order.symbol == symbol and 
                    order.status == OrderStatus.NEW and 
                    order.type == OrderType.LIMIT):
                    # Check expiration before filling
                    if order.expire_time and time.time() > order.expire_time:
                        self._update_order_status(order, OrderStatus.EXPIRED)
                        continue
                    # Check if limit order can be filled
                    if ((order.side == OrderSide.BUY and price <= order.price) or
                        (order.side == OrderSide.SELL and price >= order.price)):
                        await self._execute_order(order)
        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")
            if 'error' in self.callbacks:
                callbacks = self.callbacks['error']
                if callable(callbacks):
                    callbacks = [callbacks]
                for callback in callbacks:
                    try:
                        callback(e)
                    except Exception as err:
                        logger.error(f"Error in error callback: {str(err)}")

    async def _send_to_exchange(self, order: Order, cancel: bool = False) -> Dict[str, Any]:
        """
        Send order to exchange.
        
        Args:
            order: Order to send
            cancel: Whether this is a cancellation request
            
        Returns:
            Dict containing execution details
        """
        # Simulate exchange response
        await asyncio.sleep(0.1)  # Simulate network delay
        
        if cancel:
            return {
                'order_id': order.id,
                'status': 'canceled',
                'timestamp': time.time()
            }
        
        return {
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side.value,
            'type': order.type.value,
            'quantity': float(order.quantity),
            'price': float(order.price) if order.price else None,
            'timestamp': time.time()
        }
        
    async def _handle_execution(self, execution: Dict[str, Any]) -> None:
        """
        Handle order execution.
        
        Args:
            execution: Execution details
        """
        try:
            order_id = execution.get('order_id')
            if not order_id:
                logger.error("Execution missing order_id")
                return
                
            order = self._active_orders.get(order_id)
            if not order:
                logger.error(f"Order {order_id} not found for execution")
                return
                
            self._update_order_status(order, execution_data=execution, status=OrderStatus.FILLED)
            self._notify_order_update(order)
            self._notify_execution(order, execution)
            
        except Exception as e:
            logger.error(f"Error handling execution: {str(e)}")
            
    def _notify_order_update(self, order: Order) -> None:
        """Notify callbacks that an order was updated."""
        if 'order_update' in self.callbacks:
            callbacks = self.callbacks['order_update']
            if callable(callbacks):
                callbacks = [callbacks]
            for callback in callbacks:
                try:
                    callback(order)
                except Exception as e:
                    logger.error(f"Error in order update callback: {e}")
            
    def _notify_execution(self, order: Order, execution: Dict[str, Any]) -> None:
        """
        Notify execution.
        
        Args:
            order: Order being executed
            execution: Execution details
        """
        try:
            callbacks = self.callbacks.get('execution', [])
            if callable(callbacks):
                callbacks = [callbacks]
            for cb in callbacks:
                cb(order, execution)
        except Exception as e:
            logger.error(f"Error notifying execution: {str(e)}")
            
    async def place_order(self, order: Order, priority: int = 0) -> None:
        """
        Place order.
        Args:
            order: Order to place
            priority: Order priority (higher number = higher priority)
        Note: For correct priority handling in tests, pass the priority argument explicitly.
        """
        logger.info(f"place_order called for order_id: {order.id}")
        if not self._running:
            raise OrderError("Order engine not running")
        if not self._validate_order(order):
            self._update_order_status(order, OrderStatus.REJECTED)
            return
        await self._order_queue.put((-priority, order.created_at, order))
        async with self._order_lock:
            self._active_orders[order.id] = order
        self._notify_order_placed(order)
        
    async def cancel_order(self, order_id: str) -> None:
        """
        Cancel order.
        
        Args:
            order_id: Order ID
        """
        logger.info(f"cancel_order called for order_id: {order_id}")
        if not self._running:
            raise OrderError("Order engine not running")
        async with self._order_lock:
            order = self._active_orders.get(order_id)
            if not order:
                for completed_order in self._completed_orders:
                    if completed_order.id == order_id:
                        if completed_order.status == OrderStatus.FILLED:
                            raise OrderError(f"Order {order_id} cannot be canceled (already filled)")
                        elif completed_order.status == OrderStatus.CANCELED:
                            raise OrderError(f"Order {order_id} cannot be canceled (already canceled)")
                        elif completed_order.status == OrderStatus.EXPIRED:
                            raise OrderError(f"Order {order_id} cannot be canceled (already expired)")
                        else:
                            raise OrderError(f"Order {order_id} not found")
                raise OrderError(f"Order {order_id} not found")
            if order.status == OrderStatus.FILLED:
                raise OrderError(f"Order {order_id} cannot be canceled (already filled)")
            elif order.status == OrderStatus.CANCELED:
                raise OrderError(f"Order {order_id} cannot be canceled (already canceled)")
            elif order.status == OrderStatus.EXPIRED:
                raise OrderError(f"Order {order_id} cannot be canceled (already expired)")
            self._update_order_status(order, OrderStatus.CANCELED)
            self._completed_orders.append(order)
            self._active_orders.pop(order_id, None)
            self._notify_order_canceled(order)

    def get_orders(self, symbol: Optional[str] = None, status: Optional[OrderStatus] = None,
                  side: Optional[OrderSide] = None) -> List[Order]:
        """
        Get orders matching criteria.
        
        Args:
            symbol: Filter by symbol
            status: Filter by status
            side: Filter by side
            
        Returns:
            List of matching orders
        """
        try:
            orders = []
            for order in self._active_orders.values():
                if symbol and order.symbol != symbol:
                    continue
                if status and order.status != status:
                    continue
                if side and order.side != side:
                    continue
                orders.append(order)
            return orders
            
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            if 'error' in self.callbacks:
                self.callbacks['error'](e)
            return []

    async def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            Optional[Order]: Order if found, None otherwise
        """
        # Check active orders first
        if order_id in self._active_orders:
            return self._active_orders[order_id]
        
        # Check completed orders
        for order in self._completed_orders:
            if order.id == order_id:
                return order
        
        return None
        
    def get_execution_stats(self) -> Dict[str, float]:
        """
        Get execution statistics.
        
        Returns:
            Dict[str, float]: Execution statistics
        """
        if not self._execution_times:
            return {
                'avg_execution_time': 0.0,
                'min_execution_time': 0.0,
                'max_execution_time': 0.0,
                'total_orders': 0
            }
        
        times = list(self._execution_times)
        return {
            'avg_execution_time': sum(times) / len(times),
            'min_execution_time': min(times),
            'max_execution_time': max(times),
            'total_orders': len(times)
        }
            
    def _get_position(self, symbol: str) -> int:
        """
        Get current position for symbol.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Current position size
        """
        # This is a placeholder - implement actual position tracking
        return 0

    def _notify_order_placed(self, order: Order) -> None:
        """Notify order placed."""
        if 'order_placed' in self.callbacks:
            callbacks = self.callbacks['order_placed']
            if callable(callbacks):
                callbacks = [callbacks]
            for callback in callbacks:
                try:
                    callback(order)
                except Exception as e:
                    logger.error(f"Error in order placed callback: {e}")
                    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers."""
        if sys.platform == 'win32':
            return
            
        loop = asyncio.get_event_loop()
        
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self._handle_signal(s))
            ) 

    def _update_execution_stats(self, execution_time: float) -> None:
        """
        Update execution statistics.
        
        Args:
            execution_time: Execution time in seconds
        """
        self._execution_times.append(execution_time)
        # No need to check length as deque has maxlen

    def _notify_order_canceled(self, order: Order) -> None:
        if 'order_canceled' in self.callbacks:
            callbacks = self.callbacks['order_canceled']
            if callable(callbacks):
                callbacks = [callbacks]
            for callback in callbacks:
                try:
                    callback(order)
                except Exception as e:
                    logger.error(f"Error in order canceled callback: {e}")

    def set_mock_price(self, symbol: str, price: Decimal):
        """Allow tests to set the current price for stop/stop-limit orders."""
        self._current_prices[symbol] = price

    def clear_all_orders(self):
        """Helper for tests: clear all orders and completed orders."""
        self._active_orders.clear()
        self._completed_orders.clear()