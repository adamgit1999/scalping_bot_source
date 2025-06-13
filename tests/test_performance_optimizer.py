import pytest
import time
import threading
import gc
import socket
import logging
from unittest.mock import Mock, patch, MagicMock
import psutil
import os
from datetime import datetime, timedelta
from src.performance.optimizer import PerformanceOptimizer, PerformanceMetrics
from src.exceptions import OptimizationError, ResourceError
from decimal import Decimal

@pytest.fixture
def optimizer():
    """Create a performance optimizer instance."""
    return PerformanceOptimizer()

@pytest.fixture
def mock_metrics():
    """Create mock performance metrics."""
    return {
        'latency': {'current': 50.0, 'average': 45.0, 'p95': 60.0, 'p99': 70.0},
        'operations': {
            'order_execution': {'average_time': 0.1, 'total_calls': 100},
            'market_data': {'average_time': 0.05, 'total_calls': 1000}
        },
        'system': {
            'cpu_usage': 45.0,
            'memory_usage': 60.0,
            'network_latency': 50.0
        }
    }

def test_initialization(optimizer):
    """Test optimizer initialization."""
    assert optimizer.latency_measurements == []
    assert optimizer.execution_times == {}
    assert optimizer.thread_pool._max_workers == 4
    assert not optimizer.running
    assert optimizer.optimization_thread is None
    assert optimizer.metrics_history == []
    assert optimizer.optimization_interval == 1.0
    assert optimizer.latency_threshold == 100.0
    assert optimizer.cpu_threshold == 80.0
    assert optimizer.memory_threshold == 80.0
    assert optimizer.max_metrics_history == 1000
    assert optimizer.max_latency_measurements == 100

def test_process_priority():
    """Test setting process priority."""
    with patch('os.nice') as mock_nice:
        optimizer = PerformanceOptimizer()
        mock_nice.assert_called_once_with(-10)
        
    # Test permission error
    with patch('os.nice', side_effect=PermissionError):
        with pytest.raises(ResourceError, match="Failed to set process priority"):
            PerformanceOptimizer()

def test_start_stop(optimizer):
    """Test starting and stopping the optimizer."""
    optimizer.start()
    assert optimizer.running
    assert optimizer.optimization_thread is not None
    
    optimizer.stop()
    assert not optimizer.running
    assert optimizer.optimization_thread is None

def test_queue_operation(optimizer):
    """Test queueing operations."""
    # Test normal operation
    mock_func = Mock(return_value="result")
    optimizer.queue_operation("test_op", mock_func, "arg1", kwarg1="value1")
    
    operation = optimizer.operation_queue.get()
    assert operation['name'] == "test_op"
    assert operation['func'] == mock_func
    assert operation['args'] == ("arg1",)
    assert operation['kwargs'] == {"kwarg1": "value1"}
    
    # Test queue full
    optimizer.operation_queue.maxsize = 1
    try:
        optimizer.operation_queue.put({}, block=False)  # Fill queue with non-blocking put
    except Exception:
        pass  # Ignore if queue is already full
        
    with pytest.raises(OptimizationError, match="Operation queue is full"):
        optimizer.queue_operation("test_op", mock_func)
    
    # Test invalid operation
    with pytest.raises(ValueError, match="Invalid operation name"):
        optimizer.queue_operation("", mock_func)
    
    with pytest.raises(ValueError, match="Invalid operation function"):
        optimizer.queue_operation("test_op", None)

def test_execute_operation(optimizer):
    """Test executing operations."""
    # Test successful execution
    mock_func = Mock(return_value="result")
    operation = {
        'name': 'test_op',
        'func': mock_func,
        'args': ("arg1",),
        'kwargs': {"kwarg1": "value1"}
    }
    
    optimizer._execute_operation(operation)
    mock_func.assert_called_once_with("arg1", kwarg1="value1")
    assert "test_op" in optimizer.execution_times
    assert len(optimizer.execution_times["test_op"]) == 1
    
    # Test execution error
    def failing_func():
        raise ValueError("Test error")
    
    operation['func'] = failing_func
    with pytest.raises(OptimizationError, match="Operation execution failed"):
        optimizer._execute_operation(operation)
    
    # Test timeout
    def slow_func():
        time.sleep(2)
    
    operation['func'] = slow_func
    with pytest.raises(OptimizationError, match="Operation execution timeout"):
        optimizer._execute_operation(operation, timeout=0.1)

def test_monitor_resources(optimizer):
    """Test resource monitoring."""
    # Test CPU monitoring
    with patch('psutil.cpu_percent', return_value=90.0):
        optimizer._monitor_resources()
        assert optimizer.cpu_usage == 90.0
    
    # Test memory monitoring
    with patch('psutil.virtual_memory', return_value=Mock(percent=85.0)):
        optimizer._monitor_resources()
        assert optimizer.memory_usage == 85.0

def test_optimize_cpu_usage(optimizer):
    """Test CPU optimization."""
    # Test high CPU usage
    optimizer.cpu_usage = 90.0
    optimizer._optimize_cpu_usage()
    assert optimizer.thread_pool._max_workers < 4  # Should reduce thread count
    
    # Test normal CPU usage
    optimizer.cpu_usage = 50.0
    optimizer._optimize_cpu_usage()
    assert optimizer.thread_pool._max_workers == 4  # Should maintain thread count

def test_optimize_memory_usage(optimizer):
    """Test memory optimization."""
    # Test high memory usage
    optimizer.memory_usage = 90.0
    optimizer._optimize_memory_usage()
    assert len(optimizer.metrics_history) < optimizer.max_metrics_history
    
    # Test normal memory usage
    optimizer.memory_usage = 50.0
    optimizer._optimize_memory_usage()
    assert len(optimizer.metrics_history) == optimizer.max_metrics_history

def test_clear_caches(optimizer):
    """Test cache clearing."""
    # Test CPU cache clearing
    optimizer._clear_cpu_cache()
    
    # Test memory cache clearing
    optimizer._clear_memory_caches()
    assert len(optimizer.metrics_history) == 0
    assert len(optimizer.latency_measurements) == 0

def test_check_network_latency(optimizer):
    """Test network latency checking."""
    # Test high latency
    optimizer.latency_measurements = [150.0, 160.0, 170.0]
    optimizer._check_network_latency()
    assert optimizer.operation_queue.maxsize < 1000  # Should reduce queue size
    
    # Test normal latency
    optimizer.latency_measurements = [50.0, 60.0, 70.0]
    optimizer._check_network_latency()
    assert optimizer.operation_queue.maxsize == 1000  # Should maintain queue size

def test_measure_exchange_latency(optimizer):
    """Test exchange latency measurement."""
    with patch('time.time', side_effect=[0.0, 0.1]):
        latency = optimizer._measure_exchange_latency()
        assert latency == pytest.approx(100.0, rel=0.1)

def test_calculate_metrics(optimizer):
    """Test metrics calculation."""
    # Test with sample trades
    trades = [
        {'profit': 100.0, 'loss': 0.0},
        {'profit': 200.0, 'loss': 0.0},
        {'profit': 0.0, 'loss': 50.0}
    ]
    
    metrics = optimizer.calculate_metrics(trades)
    assert isinstance(metrics, dict)
    assert "total_profit" in metrics
    assert "total_loss" in metrics
    assert "win_rate" in metrics
    assert "profit_factor" in metrics

def test_calculate_sharpe_ratio(optimizer):
    """Test Sharpe ratio calculation."""
    returns = [0.01, 0.02, -0.01, 0.03, -0.02]
    sharpe = optimizer._calculate_sharpe_ratio(returns)
    assert isinstance(sharpe, float)
    assert sharpe > 0

def test_calculate_sortino_ratio(optimizer):
    """Test Sortino ratio calculation."""
    returns = [0.01, 0.02, -0.01, 0.03, -0.02]
    sortino = optimizer._calculate_sortino_ratio(returns)
    assert isinstance(sortino, float)
    assert sortino > 0

def test_calculate_max_drawdown(optimizer):
    """Test maximum drawdown calculation."""
    equity = [100.0, 110.0, 105.0, 120.0, 115.0]
    drawdown = optimizer._calculate_max_drawdown(equity)
    assert isinstance(drawdown, float)
    assert drawdown >= 0.0
    assert drawdown <= 1.0

def test_calculate_win_rate(optimizer):
    """Test win rate calculation."""
    trades = [
        {'profit': 100.0, 'loss': 0.0},
        {'profit': 200.0, 'loss': 0.0},
        {'profit': 0.0, 'loss': 50.0}
    ]
    
    win_rate = optimizer._calculate_win_rate(trades)
    assert isinstance(win_rate, float)
    assert win_rate >= 0.0
    assert win_rate <= 1.0

def test_calculate_profit_factor(optimizer):
    """Test profit factor calculation."""
    trades = [
        {'profit': 100.0, 'loss': 0.0},
        {'profit': 200.0, 'loss': 0.0},
        {'profit': 0.0, 'loss': 50.0}
    ]
    
    profit_factor = optimizer._calculate_profit_factor(trades)
    assert isinstance(profit_factor, float)
    assert profit_factor > 0.0

def test_operation_queue_management(optimizer):
    """Test operation queue management."""
    # Test queue operations
    for i in range(5):
        optimizer.queue_operation(f"op_{i}", lambda x: x, i)
    
    assert optimizer.operation_queue.qsize() == 5
    
    # Test queue processing
    optimizer.start()
    time.sleep(0.1)
    optimizer.stop()
    assert optimizer.operation_queue.empty()
    
    # Test queue timeout
    optimizer.operation_queue.maxsize = 1
    optimizer.operation_queue.put({})
    with pytest.raises(OptimizationError, match="Operation queue is full"):
        optimizer.queue_operation("test_op", lambda x: x)

def test_resource_optimization_thresholds(optimizer):
    """Test resource optimization thresholds."""
    # Test CPU threshold
    with patch('psutil.cpu_percent', return_value=85):
        with patch.object(optimizer, '_optimize_cpu_usage') as mock_cpu:
            optimizer._monitor_resources()
            mock_cpu.assert_called_once()
    
    # Test memory threshold
    with patch('psutil.virtual_memory', return_value=MagicMock(percent=85)):
        with patch.object(optimizer, '_optimize_memory_usage') as mock_memory:
            optimizer._monitor_resources()
            mock_memory.assert_called_once()
    
    # Test threshold configuration
    optimizer.cpu_threshold = 90.0
    optimizer.memory_threshold = 90.0
    with patch('psutil.cpu_percent', return_value=85):
        with patch.object(optimizer, '_optimize_cpu_usage') as mock_cpu:
            optimizer._monitor_resources()
            mock_cpu.assert_not_called()

def test_latency_monitoring(optimizer):
    """Test latency monitoring and alerts."""
    # Test high latency detection
    with patch.object(optimizer, '_measure_exchange_latency', return_value=150.0):
        with patch('logging.Logger.warning') as mock_warning:
            optimizer._check_network_latency()
            mock_warning.assert_called_once()
    
    # Test latency history management
    optimizer.latency_measurements = [50.0] * 101
    optimizer._check_network_latency()
    assert len(optimizer.latency_measurements) == 100
    
    # Test latency threshold configuration
    optimizer.latency_threshold = 200.0
    with patch.object(optimizer, '_measure_exchange_latency', return_value=150.0):
        with patch('logging.Logger.warning') as mock_warning:
            optimizer._check_network_latency()
            mock_warning.assert_not_called()

def test_error_handling(optimizer):
    """Test error handling in various methods."""
    # Test process priority error
    with patch('os.nice', side_effect=PermissionError):
        with pytest.raises(ResourceError, match="Failed to set process priority"):
            PerformanceOptimizer()
    
    # Test operation execution error
    def failing_func():
        raise ValueError("Test error")
    
    optimizer.queue_operation("failing_op", failing_func)
    with pytest.raises(OptimizationError, match="Operation execution failed"):
        optimizer._execute_operation(optimizer.operation_queue.get())
    
    # Test resource monitoring error
    with patch('psutil.cpu_percent', side_effect=psutil.Error):
        with pytest.raises(ResourceError, match="Failed to monitor system resources"):
            optimizer._monitor_resources()

def test_thread_pool_management(optimizer):
    """Test thread pool management."""
    # Test worker reduction
    with patch('psutil.cpu_count', return_value=8):
        with patch('psutil.cpu_percent', return_value=95):
            optimizer._optimize_cpu_usage()
            assert optimizer.thread_pool._max_workers == 3
    
    # Test worker increase
    with patch('psutil.cpu_count', return_value=8):
        with patch('psutil.cpu_percent', return_value=30):
            optimizer._optimize_cpu_usage()
            assert optimizer.thread_pool._max_workers == 4
    
    # Test thread pool error
    with patch.object(optimizer, 'thread_pool', side_effect=Exception):
        with pytest.raises(ResourceError, match="Failed to manage thread pool"):
            optimizer._optimize_cpu_usage()

def test_clear_cpu_cache(optimizer):
    """Test clearing CPU cache."""
    # Test LRU cache clearing
    @optimizer._get_lru_cached_functions()
    def test_func():
        pass
    
    test_func()  # Add to cache
    optimizer._clear_cpu_cache()
    assert test_func.cache_info().hits == 0
    
    # Test cache clearing error
    with patch('functools.lru_cache', side_effect=Exception):
        with pytest.raises(OptimizationError, match="Failed to clear CPU cache"):
            optimizer._clear_cpu_cache()

def test_clear_memory_caches(optimizer):
    """Test clearing memory caches."""
    # Test normal cache clearing
    optimizer.execution_times = {"test": [1, 2, 3]}
    optimizer.latency_measurements = [1, 2, 3]
    optimizer.metrics_history = [{"test": "data"}]
    
    optimizer._clear_memory_caches()
    assert optimizer.execution_times == {}
    assert optimizer.latency_measurements == []
    assert optimizer.metrics_history == []
    
    # Test cache clearing error
    with patch.object(optimizer, 'execution_times', side_effect=Exception):
        with pytest.raises(OptimizationError, match="Failed to clear memory caches"):
            optimizer._clear_memory_caches() 