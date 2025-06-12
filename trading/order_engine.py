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

    def __init__(self, id: str, symbol: str, type: OrderType, side: OrderSide,
                 quantity: Decimal, price: Optional[Decimal] = None,
                 stop_price: Optional[Decimal] = None, status: OrderStatus = OrderStatus.NEW,
                 filled_quantity: Decimal = Decimal("0"),
                 average_price: Decimal = Decimal("0"),
                 created_at: float = None, updated_at: float = None,
                 client_order_id: Optional[str] = None,
                 post_only: bool = False, reduce_only: bool = False,
                 time_in_force: str = "GTC", expire_time: Optional[float] = None):
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
    """High-performance order execution engine."""
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize order engine.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.orders: Dict[str, Order] = {}
        self.order_queue = PriorityQueue()
        self._loop = asyncio.get_event_loop()
        self.execution_queue = asyncio.Queue()
        self.callbacks: Dict[str, List[Callable]] = {}
        self.running = False
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.Lock()
        self.last_execution_time = 0.0
        self.execution_times: List[float] = []
        self.max_retries = 3
        self.retry_delay = 0.1  # seconds
        self._tasks: Set[asyncio.Task] = set()
        self._cleanup_handlers: List[Callable] = []
        self._shutdown_event = asyncio.Event()
        self._test_market_price = None  # For test control of market price
        
        # Register signal handlers
        if sys.platform != 'win32':  # Windows doesn't support SIGTERM
            self._loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(self.stop()))
            self._loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(self.stop()))
        
    async def start(self) -> None:
        """Start order engine."""
        if self.running:
            return
            
        self.running = True
        self._shutdown_event.clear()
        
        # Start processing tasks
        self._tasks.add(asyncio.create_task(self._process_orders(), name='process_orders'))
        self._tasks.add(asyncio.create_task(self._process_executions(), name='process_executions'))
        
        # Register signal handlers
        if sys.platform != 'win32':
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    loop.add_signal_handler(
                        sig,
                        lambda s=sig: asyncio.create_task(self._handle_signal(s))
                    )
                except NotImplementedError:
                    # Some platforms don't support add_signal_handler
                    signal.signal(sig, lambda s, f: asyncio.create_task(self._handle_signal(s)))
        
        logger.info("Order engine started")
        
    async def _handle_signal(self, sig: signal.Signals) -> None:
        """
        Handle system signal.
        
        Args:
            sig: Signal received
        """
        logger.info(f"Received signal {sig.name}")
        if self.running:
            await self.stop()

    async def stop(self) -> None:
        """Stop order engine."""
        if not self.running:
            return
            
        logger.info("Stopping order engine...")
        self.running = False
        self._shutdown_event.set()
        
        # Cancel all tasks
        tasks_to_cancel = set(self._tasks)
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                except Exception as e:
                    logger.error(f"Error cancelling task: {str(e)}")
        
        self._tasks.clear()
        
        # Execute cleanup handlers
        for handler in self._cleanup_handlers:
            try:
                await asyncio.wait_for(handler(), timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning("Cleanup handler timed out")
            except Exception as e:
                logger.error(f"Error in cleanup handler: {str(e)}")
        
        # Shutdown thread pool
        try:
            self.thread_pool.shutdown(wait=True, cancel_futures=True)
        except Exception as e:
            logger.error(f"Error shutting down thread pool: {str(e)}")
        
        logger.info("Order engine stopped")
        
    def register_cleanup_handler(self, handler: Callable[[], Awaitable[None]]) -> None:
        """
        Register cleanup handler.
        
        Args:
            handler: Cleanup handler
        """
        self._cleanup_handlers.append(handler)

    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register callback.
        
        Args:
            event: Event type
            callback: Callback function
        """
        self.callbacks[event] = callback

    async def _process_orders(self) -> None:
        """Process orders from queue."""
        while self.running:
            try:
                try:
                    priority, order = await asyncio.get_event_loop().run_in_executor(
                        self.thread_pool,
                        lambda: self.order_queue.get(timeout=0.1)
                    )
                except Exception:
                    continue
                # Check cancellation
                if order.status == OrderStatus.CANCELED:
                    continue
                # Check expiration
                if order.expire_time and time.time() > order.expire_time:
                    if order.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED):
                        self._update_order_status(order, status=OrderStatus.EXPIRED)
                    continue
                # Stop/Stop-Limit: only execute if stop condition met
                if order.type in (OrderType.STOP, OrderType.STOP_LIMIT):
                    if not self._check_stop_condition(order):
                        self.order_queue.put((priority, order))
                        await asyncio.sleep(0.01)
                        continue
                # Post-only: reject if would be filled immediately
                if order.post_only:
                    self._update_order_status(order, status=OrderStatus.REJECTED)
                    continue
                # Reduce-only: reject if not reducing
                if order.reduce_only:
                    position = self._get_position(order.symbol)
                    if (order.side == OrderSide.BUY and position >= 0) or \
                       (order.side == OrderSide.SELL and position <= 0):
                        self._update_order_status(order, status=OrderStatus.REJECTED)
                        continue
                # Validate again before execution
                if not self._validate_order(order):
                    self._update_order_status(order, status=OrderStatus.REJECTED)
                    continue
                # Execute order
                await self._execute_order(order)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing orders: {str(e)}")
                self._notify_error(e)
                await asyncio.sleep(0.1)
                
    async def _process_executions(self) -> None:
        """Process executions from queue."""
        while self.running:
            try:
                # Get execution from queue
                execution = await self.execution_queue.get()
                
                # Handle execution
                await self._handle_execution(execution)
                
                # Mark task as done
                self.execution_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing executions: {str(e)}")
                await asyncio.sleep(0.1)
                
    async def _execute_order(self, order: Order) -> None:
        """
        Execute order.
        
        Args:
            order: Order to execute
        """
        start_time = time.time()
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                if order.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED):
                    return
                result = await self._send_to_exchange(order)
                fill_qty = Decimal(str(result.get('quantity', order.quantity)))
                fill_price = Decimal(str(result.get('price', order.price or 0)))
                prev_filled = order.filled_quantity
                new_filled = prev_filled + fill_qty
                if new_filled >= order.quantity:
                    fill_qty = order.quantity - prev_filled
                    order.filled_quantity = order.quantity
                    if prev_filled > 0:
                        order.average_price = ((order.average_price * prev_filled) + (fill_price * fill_qty)) / order.quantity
                    else:
                        order.average_price = fill_price
                    self._update_order_status(order, result, status=OrderStatus.FILLED)
                else:
                    order.filled_quantity = new_filled
                    if prev_filled > 0:
                        order.average_price = ((order.average_price * prev_filled) + (fill_price * fill_qty)) / new_filled
                    else:
                        order.average_price = fill_price
                    self._update_order_status(order, result, status=OrderStatus.PARTIALLY_FILLED)
                    self.order_queue.put((0, order))
                execution_time = time.time() - start_time
                self.execution_times.append(execution_time)
                self.last_execution_time = execution_time
                return
            except ExecutionError as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    self._update_order_status(order, status=OrderStatus.REJECTED)
                    self._notify_error(e)
                    return
                await asyncio.sleep(self.retry_delay * (2 ** (retry_count - 1)))
            except Exception as e:
                self._update_order_status(order, status=OrderStatus.REJECTED)
                self._notify_error(e)
                return
                
    def _can_execute(self, order: Order) -> bool:
        """
        Check if order can be executed.
        
        Args:
            order: Order to check
            
        Returns:
            bool: True if order can be executed
        """
        # Check if order exists
        if order.id not in self.orders:
            return False
            
        # Check if order is already filled or canceled
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            return False
            
        # Check if order is expired
        if order.expire_time and time.time() > order.expire_time:
            self._update_order_status(order, status=OrderStatus.EXPIRED)
            return False
            
        # Check reduce-only orders
        if order.reduce_only:
            position = self._get_position(order.symbol)
            if (order.side == OrderSide.BUY and position >= 0) or \
               (order.side == OrderSide.SELL and position <= 0):
                self._update_order_status(order, status=OrderStatus.REJECTED)
                return False
                
        # Check stop orders
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
            return False
            
        # Get current market price
        current_price = self._get_market_price(order.symbol)
        if not current_price:
            return False
            
        # Check stop condition
        if order.side == OrderSide.BUY:
            return current_price >= order.stop_price
        else:
            return current_price <= order.stop_price
            
    def _get_market_price(self, symbol: str) -> Optional[Decimal]:
        if self._test_market_price is not None:
            return self._test_market_price
        return Decimal("50000.0")
        
    async def _process_stop_orders(self, price: Decimal) -> None:
        """
        Process stop orders.
        
        Args:
            price: Current market price
        """
        for order in self.orders.values():
            if order.type in (OrderType.STOP, OrderType.STOP_LIMIT) and \
               order.status == OrderStatus.NEW:
                # Check stop condition
                if self._check_stop_condition(order):
                    # Execute order
                    await self._execute_order(order)
                    
    def _cleanup_old_orders(self) -> None:
        """Clean up old orders."""
        current_time = time.time()
        orders_to_remove = []
        
        for order_id, order in self.orders.items():
            # Remove orders older than 24 hours
            if current_time - order.created_at > 24 * 3600:
                orders_to_remove.append(order_id)
                
        for order_id in orders_to_remove:
            del self.orders[order_id]
            
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

    def _update_order_status(self, order: Order, result: Optional[Dict] = None, status: Optional[OrderStatus] = None) -> None:
        """
        Update order status.
        
        Args:
            order: Order to update
            result: Execution result
            status: New status
        """
        try:
            if status:
                order.status = status
            order.updated_at = time.time()
            if result and status == OrderStatus.FILLED:
                order.filled_quantity = order.quantity
                order.average_price = Decimal(str(result.get('price', order.price or 0)))
            self._notify_order_update(order)
            if status == OrderStatus.FILLED:
                self._notify_execution({
                    'order_id': order.id,
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'type': order.type.value,
                    'quantity': float(order.quantity),
                    'price': float(order.average_price),
                    'timestamp': order.updated_at
                })
            if status == OrderStatus.REJECTED:
                self._notify_error(OrderError(f"Order {order.id} rejected"))
            if status == OrderStatus.EXPIRED:
                self._notify_error(OrderError(f"Order {order.id} expired"))
            if status == OrderStatus.CANCELED:
                self._notify_error(OrderError(f"Order {order.id} canceled"))
        except Exception as e:
            logger.error(f"Error updating order status: {e}")

    async def _process_market_data(self, data: Dict) -> None:
        """
        Process market data.
        
        Args:
            data: Market data
        """
        try:
            # Notify callbacks
            if 'market_data' in self.callbacks:
                self.callbacks['market_data'](data)
                
            # Check for order fills
            symbol = data.get('symbol')
            price = Decimal(str(data.get('price', 0)))
            
            for order in self.orders.values():
                if (order.symbol == symbol and 
                    order.status == OrderStatus.NEW and 
                    order.type == OrderType.LIMIT):
                    
                    # Check if limit order can be filled
                    if ((order.side == OrderSide.BUY and price <= order.price) or
                        (order.side == OrderSide.SELL and price >= order.price)):
                        await self._execute_order(order)
                        
        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")
            if 'error' in self.callbacks:
                self.callbacks['error'](e)

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
                
            order = self.orders.get(order_id)
            if not order:
                logger.error(f"Order {order_id} not found for execution")
                return
                
            self._update_order_status(order, execution)
            self._notify_order_update(order)
            self._notify_execution(execution)
            
        except Exception as e:
            logger.error(f"Error handling execution: {str(e)}")
            
    def _notify_order_update(self, order: Order) -> None:
        """
        Notify order update.
        
        Args:
            order: Updated order
        """
        try:
            callbacks = self.callbacks.get('order_update', [])
            if callable(callbacks):
                callbacks = [callbacks]
            for cb in callbacks:
                cb(order)
        except Exception as e:
            logger.error(f"Error notifying order update: {str(e)}")
            
    def _notify_execution(self, execution: Dict[str, Any]) -> None:
        """
        Notify execution.
        
        Args:
            execution: Execution details
        """
        try:
            callbacks = self.callbacks.get('execution', [])
            if callable(callbacks):
                callbacks = [callbacks]
            for cb in callbacks:
                cb(execution)
        except Exception as e:
            logger.error(f"Error notifying execution: {str(e)}")
            
    def _notify_error(self, error: Exception) -> None:
        """
        Notify error.
        
        Args:
            error: Error to notify
        """
        try:
            callbacks = self.callbacks.get('error', [])
            if callable(callbacks):
                callbacks = [callbacks]
            for cb in callbacks:
                cb(error)
        except Exception as e:
            logger.error(f"Error notifying error: {str(e)}")
            
    async def place_order(self, order: Order) -> None:
        """
        Place order.
        
        Args:
            order: Order to place
        """
        if not self.running:
            raise OrderError("Order engine not running")
            
        # Validate order
        if not self._validate_order(order):
            with self.lock:
                self.orders[order.id] = order
            self._update_order_status(order, status=OrderStatus.REJECTED)
            return
            
        with self.lock:
            self.orders[order.id] = order
            
        # Add to queue with negative priority (higher number = higher priority)
        priority = 0
        if order.type == OrderType.MARKET:
            priority = -3
        elif order.type == OrderType.STOP:
            priority = -2
        elif order.type == OrderType.STOP_LIMIT:
            priority = -1
            
        self.order_queue.put((priority, order))
        
        # Notify order update
        self._notify_order_update(order)
        
    async def cancel_order(self, order_id: str) -> None:
        """
        Cancel order.
        
        Args:
            order_id: Order ID to cancel
        """
        try:
            if not self.running:
                raise OrderError("Order engine not running")
            
            # Check if order exists
            if order_id not in self.orders:
                raise OrderError(f"Order {order_id} not found")
            
            # Get order
            order = self.orders[order_id]
            
            # Check if order can be canceled
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                raise OrderError(f"Order {order_id} cannot be canceled (status: {order.status})")
            
            # Cancel order
            try:
                await asyncio.wait_for(
                    self._send_to_exchange(order, cancel=True),
                    timeout=5.0  # 5 second timeout
                )
            except asyncio.TimeoutError:
                raise OrderError(f"Order cancellation timed out for order {order_id}")
            
            # Update order status
            self._update_order_status(order, None, OrderStatus.CANCELED)
            
        except Exception as e:
            logger.error(f"Error canceling order: {str(e)}")
            if 'error' in self.callbacks:
                self.callbacks['error'](e)
            raise OrderError(f"Failed to cancel order: {str(e)}")
            
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
            for order in self.orders.values():
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

    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order if found, None otherwise
        """
        try:
            return self.orders.get(order_id)
            
        except Exception as e:
            logger.error(f"Error getting order: {str(e)}")
            if 'error' in self.callbacks:
                self.callbacks['error'](e)
            return None
        
    def get_execution_stats(self) -> Dict[str, float]:
        """
        Get execution statistics.
        
        Returns:
            Dictionary of statistics
        """
        try:
            if not self.execution_times:
                return {
                    'avg_execution_time': 0.0,
                    'min_execution_time': 0.0,
                    'max_execution_time': 0.0,
                    'p95_execution_time': 0.0,
                    'p99_execution_time': 0.0
                }
                
            times = np.array(self.execution_times)
            return {
                'avg_execution_time': float(np.mean(times)),
                'min_execution_time': float(np.min(times)),
                'max_execution_time': float(np.max(times)),
                'p95_execution_time': float(np.percentile(times, 95)),
                'p99_execution_time': float(np.percentile(times, 99))
            }
            
        except Exception as e:
            logger.error(f"Error getting execution stats: {str(e)}")
            return {}
            
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