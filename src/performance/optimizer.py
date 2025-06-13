import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
import logging
import psutil
import os
from concurrent.futures import ThreadPoolExecutor
import socket
import aiohttp
import uvloop
import orjson
from functools import lru_cache
import threading
from queue import Queue, Empty
import numpy as np
from dataclasses import dataclass, field
from src.exceptions import OptimizationError
import pandas as pd
from datetime import datetime, UTC

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """System for optimizing trading bot performance and minimizing latency."""
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.latency_measurements: List[float] = []
        self.execution_times: Dict[str, List[float]] = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.operation_queue = Queue()
        self.running = False
        self.optimization_thread = None
        
        # Set process priority to high
        self._set_process_priority()
        
        # Initialize async event loop with uvloop for better performance
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        
        self.metrics = {
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0
        }
        
    def _set_process_priority(self) -> None:
        """Set process priority to high for better performance."""
        try:
            os.nice(-10)  # Set high priority
            logger.info("Set process priority to high")
        except Exception as e:
            logger.warning(f"Could not set process priority: {str(e)}")
            
    def start(self) -> None:
        """Start the performance optimization system."""
        self.running = True
        self.optimization_thread = threading.Thread(target=self._optimization_loop)
        self.optimization_thread.start()
        logger.info("Started performance optimization system")
        
    def stop(self) -> None:
        """Stop the performance optimization system."""
        self.running = False
        if self.optimization_thread:
            self.optimization_thread.join()
        self.thread_pool.shutdown()
        logger.info("Stopped performance optimization system")
        
    def _optimization_loop(self) -> None:
        """Main optimization loop."""
        while self.running:
            try:
                # Process queued operations
                while True:
                    try:
                        operation = self.operation_queue.get_nowait()
                        self._execute_operation(operation)
                    except Empty:
                        break
                        
                # Monitor system resources
                self._monitor_resources()
                
                # Sleep briefly to prevent CPU overload
                time.sleep(0.001)
            except Exception as e:
                logger.error(f"Error in optimization loop: {str(e)}")
                
    def _monitor_resources(self) -> None:
        """Monitor system resources and optimize if needed."""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > 80:
                self._optimize_cpu_usage()
                
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 80:
                self._optimize_memory_usage()
                
            # Check network latency
            self._check_network_latency()
        except Exception as e:
            logger.error(f"Error monitoring resources: {str(e)}")
            
    def _optimize_cpu_usage(self) -> None:
        """Optimize CPU usage."""
        try:
            # Clear CPU cache
            self._clear_cpu_cache()
            
            # Adjust thread pool size based on CPU usage
            cpu_count = psutil.cpu_count()
            current_workers = self.thread_pool._max_workers
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > 90 and current_workers > 1:
                self.thread_pool._max_workers = max(1, current_workers - 1)
            elif cpu_percent < 50 and current_workers < cpu_count:
                self.thread_pool._max_workers = min(cpu_count, current_workers + 1)
        except Exception as e:
            logger.error(f"Error optimizing CPU usage: {str(e)}")
            
    def _optimize_memory_usage(self) -> None:
        """Optimize memory usage."""
        try:
            # Clear memory caches
            self._clear_memory_caches()
            
            # Force garbage collection
            import gc
            gc.collect()
        except Exception as e:
            logger.error(f"Error optimizing memory usage: {str(e)}")
            
    def _clear_cpu_cache(self) -> None:
        """Clear CPU cache."""
        try:
            # Clear LRU caches
            for func in self._get_lru_cached_functions():
                func.cache_clear()
        except Exception as e:
            logger.error(f"Error clearing CPU cache: {str(e)}")
            
    def _clear_memory_caches(self) -> None:
        """Clear memory caches."""
        try:
            # Clear operation caches
            self.execution_times.clear()
            
            # Clear latency measurements
            self.latency_measurements.clear()
        except Exception as e:
            logger.error(f"Error clearing memory caches: {str(e)}")
            
    def _get_lru_cached_functions(self) -> List[Callable]:
        """Get all LRU cached functions."""
        return [func for func in globals().values() 
                if callable(func) and hasattr(func, 'cache_clear')]
                
    def _check_network_latency(self) -> None:
        """Check and optimize network latency."""
        try:
            # Measure latency to exchange
            latency = self._measure_exchange_latency()
            self.latency_measurements.append(latency)
            
            # Keep only recent measurements
            if len(self.latency_measurements) > 100:
                self.latency_measurements = self.latency_measurements[-100:]
                
            # Calculate average latency
            avg_latency = np.mean(self.latency_measurements)
            
            # Log if latency is high
            if avg_latency > 100:  # 100ms threshold
                logger.warning(f"High network latency detected: {avg_latency:.2f}ms")
        except Exception as e:
            logger.error(f"Error checking network latency: {str(e)}")
            
    def _measure_exchange_latency(self) -> float:
        """Measure latency to exchange."""
        try:
            start_time = time.time()
            
            # Create socket connection to exchange
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(('exchange.example.com', 443))
                
            return (time.time() - start_time) * 1000  # Convert to milliseconds
        except Exception as e:
            logger.error(f"Error measuring exchange latency: {str(e)}")
            return float('inf')
            
    def _execute_operation(self, operation: Dict[str, Any]) -> None:
        """Execute an operation with performance monitoring."""
        try:
            start_time = time.time()
            
            # Execute operation
            result = operation['func'](*operation['args'], **operation['kwargs'])
            
            # Record execution time
            execution_time = time.time() - start_time
            if operation['name'] not in self.execution_times:
                self.execution_times[operation['name']] = []
            self.execution_times[operation['name']].append(execution_time)
            
            # Log if operation is slow
            if execution_time > 0.1:  # 100ms threshold
                logger.warning(f"Slow operation detected: {operation['name']} took {execution_time:.3f}s")
        except Exception as e:
            logger.error(f"Error executing operation {operation['name']}: {str(e)}")
            
    def queue_operation(self, name: str, func: Callable, *args, **kwargs) -> None:
        """
        Queue an operation for execution.
        
        Args:
            name: Operation name
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Raises:
            ValueError: If operation name is empty or function is None
            OptimizationError: If operation queue is full
        """
        if not name:
            raise ValueError("Invalid operation name")
        if func is None:
            raise ValueError("Invalid operation function")
            
        try:
            self.operation_queue.put({
                'name': name,
                'func': func,
                'args': args,
                'kwargs': kwargs
            }, block=False)
        except Exception:
            raise OptimizationError("Operation queue is full")
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            return {
                'latency': {
                    'current': self.latency_measurements[-1] if self.latency_measurements else None,
                    'average': np.mean(self.latency_measurements) if self.latency_measurements else None,
                    'min': np.min(self.latency_measurements) if self.latency_measurements else None,
                    'max': np.max(self.latency_measurements) if self.latency_measurements else None
                },
                'operations': {
                    name: {
                        'average_time': np.mean(times),
                        'min_time': np.min(times),
                        'max_time': np.max(times),
                        'count': len(times)
                    }
                    for name, times in self.execution_times.items()
                },
                'system': {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'thread_count': threading.active_count()
                }
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {}

    def calculate_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate performance metrics from trade history."""
        try:
            if not trades:
                raise OptimizationError("No trades provided for optimization")
                
            # Calculate returns
            returns = [trade['pnl'] for trade in trades]
            
            # Calculate metrics
            self.metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
            self.metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns)
            self.metrics['max_drawdown'] = self._calculate_max_drawdown(returns)
            self.metrics['win_rate'] = self._calculate_win_rate(trades)
            self.metrics['profit_factor'] = self._calculate_profit_factor(trades)
            
            # Calculate average win/loss
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]
            
            if winning_trades:
                self.metrics['average_win'] = sum(t['pnl'] for t in winning_trades) / len(winning_trades)
            if losing_trades:
                self.metrics['average_loss'] = sum(t['pnl'] for t in losing_trades) / len(losing_trades)
                
            return self.metrics
            
        except Exception as e:
            raise OptimizationError(f"Failed to calculate metrics: {str(e)}")
            
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio."""
        if not returns:
            return 0.0
        returns_array = np.array(returns)
        return np.mean(returns_array) / np.std(returns_array) if np.std(returns_array) != 0 else 0.0
        
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio."""
        if not returns:
            return 0.0
        returns_array = np.array(returns)
        downside_returns = returns_array[returns_array < 0]
        if len(downside_returns) == 0:
            return 0.0
        return np.mean(returns_array) / np.std(downside_returns) if np.std(downside_returns) != 0 else 0.0
        
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not returns:
            return 0.0
        cumulative_returns = np.cumsum(returns)
        max_dd = 0
        peak = cumulative_returns[0]
        for value in cumulative_returns:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak != 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd
        
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate."""
        if not trades:
            return 0.0
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        return winning_trades / len(trades)
        
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor."""
        if not trades:
            return 0.0
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        return gross_profit / gross_loss if gross_loss != 0 else 0.0

@dataclass
class PerformanceMetrics:
    latency: Dict[str, Any] = field(default_factory=dict)
    operations: Dict[str, Any] = field(default_factory=dict)
    system: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        return cls(
            latency=data.get('latency', {}),
            operations=data.get('operations', {}),
            system=data.get('system', {}),
            history=data.get('history', [])
        ) 