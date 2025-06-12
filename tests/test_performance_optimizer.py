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
from performance.optimizer import PerformanceOptimizer, PerformanceMetrics
from performance.exceptions import OptimizationError, ResourceError

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
    # Test normal start/stop
    optimizer.start()
    assert optimizer.running
    assert optimizer.optimization_thread is not None
    assert optimizer.optimization_thread.is_alive()
    
    optimizer.stop()
    assert not optimizer.running
    assert not optimizer.optimization_thread.is_alive()
    
    # Test double start
    optimizer.start()
    with pytest.raises(OptimizationError, match="Optimizer is already running"):
        optimizer.start()
    
    # Test double stop
    optimizer.stop()
    optimizer.stop()  # Should not raise error
    
    # Test start with invalid state
    optimizer.running = True
    optimizer.optimization_thread = None
    with pytest.raises(OptimizationError, match="Invalid optimizer state"):
        optimizer.start()

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
    # Test normal monitoring
    with patch.object(optimizer, '_optimize_cpu_usage') as mock_cpu:
        with patch.object(optimizer, '_optimize_memory_usage') as mock_memory:
            with patch.object(optimizer, '_check_network_latency') as mock_network:
                with patch('psutil.cpu_percent', return_value=90):
                    with patch('psutil.virtual_memory', return_value=MagicMock(percent=90)):
                        optimizer._monitor_resources()
                        mock_cpu.assert_called_once()
                        mock_memory.assert_called_once()
                        mock_network.assert_called_once()
    
    # Test resource monitoring error
    with patch('psutil.cpu_percent', side_effect=psutil.Error):
        with pytest.raises(ResourceError, match="Failed to monitor system resources"):
            optimizer._monitor_resources()

def test_optimize_cpu_usage(optimizer):
    """Test CPU usage optimization."""
    # Test high CPU usage
    with patch('psutil.cpu_count', return_value=4):
        with patch('psutil.cpu_percent', return_value=95):
            optimizer._optimize_cpu_usage()
            assert optimizer.thread_pool._max_workers == 3
    
    # Test low CPU usage
    with patch('psutil.cpu_count', return_value=4):
        with patch('psutil.cpu_percent', return_value=30):
            optimizer._optimize_cpu_usage()
            assert optimizer.thread_pool._max_workers == 4
    
    # Test CPU optimization error
    with patch('psutil.cpu_percent', side_effect=psutil.Error):
        with pytest.raises(ResourceError, match="Failed to optimize CPU usage"):
            optimizer._optimize_cpu_usage()

def test_optimize_memory_usage(optimizer):
    """Test memory usage optimization."""
    # Test normal optimization
    with patch('gc.collect') as mock_gc:
        optimizer._optimize_memory_usage()
        mock_gc.assert_called_once()
    
    # Test memory optimization error
    with patch('gc.collect', side_effect=Exception):
        with pytest.raises(ResourceError, match="Failed to optimize memory usage"):
            optimizer._optimize_memory_usage()

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

def test_check_network_latency(optimizer):
    """Test network latency checking."""
    # Test normal latency check
    with patch.object(optimizer, '_measure_exchange_latency', return_value=50.0):
        optimizer._check_network_latency()
        assert len(optimizer.latency_measurements) == 1
        assert optimizer.latency_measurements[0] == 50.0
    
    # Test high latency
    with patch.object(optimizer, '_measure_exchange_latency', return_value=150.0):
        with patch('logging.Logger.warning') as mock_warning:
            optimizer._check_network_latency()
            mock_warning.assert_called_once()
    
    # Test latency history management
    optimizer.latency_measurements = [50.0] * 101
    optimizer._check_network_latency()
    assert len(optimizer.latency_measurements) == 100
    
    # Test latency check error
    with patch.object(optimizer, '_measure_exchange_latency', side_effect=Exception):
        with pytest.raises(ResourceError, match="Failed to check network latency"):
            optimizer._check_network_latency()

def test_measure_exchange_latency(optimizer):
    """Test measuring exchange latency."""
    # Test successful measurement
    with patch('socket.socket') as mock_socket:
        mock_socket.return_value.__enter__.return_value.connect.return_value = None
        latency = optimizer._measure_exchange_latency()
        assert isinstance(latency, float)
        assert latency >= 0
    
    # Test connection error
    with patch('socket.socket') as mock_socket:
        mock_socket.return_value.__enter__.return_value.connect.side_effect = socket.error
        with pytest.raises(ResourceError, match="Failed to measure exchange latency"):
            optimizer._measure_exchange_latency()

def test_get_performance_metrics(optimizer, mock_metrics):
    """Test getting performance metrics."""
    # Test normal metrics retrieval
    optimizer.latency_measurements = [50.0, 60.0, 70.0]
    optimizer.execution_times = {
        "test_op": [0.1, 0.2, 0.3]
    }
    
    metrics = optimizer.get_performance_metrics()
    assert 'latency' in metrics
    assert 'operations' in metrics
    assert 'system' in metrics
    assert metrics['latency']['current'] == 70.0
    assert metrics['latency']['average'] == 60.0
    assert metrics['operations']['test_op']['average_time'] == 0.2
    
    # Test metrics history
    optimizer.metrics_history.append(mock_metrics)
    metrics = optimizer.get_performance_metrics()
    assert len(metrics['history']) == 1
    
    # Test metrics history limit
    optimizer.metrics_history = [mock_metrics] * 1001
    metrics = optimizer.get_performance_metrics()
    assert len(metrics['history']) == 1000

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