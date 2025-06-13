import pytest
from unittest.mock import Mock, patch, AsyncMock
import time
from datetime import datetime, timedelta
import psutil
import os
from decimal import Decimal
from typing import Dict, List, Any, Optional
import asyncio

from src.performance.optimizer import PerformanceOptimizer, PerformanceMetrics
from src.trading_engine import TradingEngine
from src.websocket_server import WebSocketServer
from src.notification_system import NotificationSystem

class PerformanceMonitor:
    """System for monitoring application performance."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {}
        self.start_time = time.time()
        
    def record_metric(self, name: str, value: float) -> None:
        """
        Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
        """
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append((time.time(), value))
        
    def get_metric(self, name: str, window: int = 3600) -> list:
        """
        Get metric values within time window.
        
        Args:
            name: Metric name
            window: Time window in seconds
            
        Returns:
            List of (timestamp, value) tuples
        """
        if name not in self.metrics:
            return []
            
        cutoff = time.time() - window
        return [(t, v) for t, v in self.metrics[name] if t >= cutoff]
        
    def get_metric_stats(self, name: str, window: int = 3600) -> dict:
        """
        Get metric statistics within time window.
        
        Args:
            name: Metric name
            window: Time window in seconds
            
        Returns:
            Dictionary of statistics
        """
        values = [v for _, v in self.get_metric(name, window)]
        if not values:
            return {
                'min': 0,
                'max': 0,
                'avg': 0,
                'count': 0
            }
            
        return {
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'count': len(values)
        }
        
    def get_system_metrics(self) -> dict:
        """
        Get current system metrics.
        
        Returns:
            Dictionary of system metrics
        """
        process = psutil.Process(os.getpid())
        
        return {
            'cpu_percent': process.cpu_percent(),
            'memory_percent': process.memory_percent(),
            'memory_info': process.memory_info()._asdict(),
            'num_threads': process.num_threads(),
            'num_connections': len(process.connections()),
            'disk_io': process.io_counters()._asdict() if hasattr(process, 'io_counters') else {},
            'net_io': psutil.net_io_counters()._asdict()
        }
        
    def get_uptime(self) -> float:
        """
        Get application uptime in seconds.
        
        Returns:
            Uptime in seconds
        """
        return time.time() - self.start_time

@pytest.fixture
def performance_monitor():
    """Create a performance monitor instance."""
    return PerformanceMonitor()

def test_performance_monitor_initialization(performance_monitor):
    """Test performance monitor initialization."""
    assert performance_monitor is not None
    assert performance_monitor.metrics == {}
    assert performance_monitor.start_time > 0

def test_record_metric(performance_monitor):
    """Test recording metrics."""
    # Record metrics
    performance_monitor.record_metric('response_time', 0.1)
    performance_monitor.record_metric('response_time', 0.2)
    performance_monitor.record_metric('error_rate', 0.01)
    
    # Verify metrics
    assert 'response_time' in performance_monitor.metrics
    assert 'error_rate' in performance_monitor.metrics
    assert len(performance_monitor.metrics['response_time']) == 2
    assert len(performance_monitor.metrics['error_rate']) == 1

def test_get_metric(performance_monitor):
    """Test getting metrics."""
    # Record metrics
    performance_monitor.record_metric('response_time', 0.1)
    time.sleep(0.1)
    performance_monitor.record_metric('response_time', 0.2)
    
    # Get all metrics
    metrics = performance_monitor.get_metric('response_time')
    assert len(metrics) == 2
    
    # Get metrics within window
    metrics = performance_monitor.get_metric('response_time', window=0.05)
    assert len(metrics) == 1

def test_get_metric_stats(performance_monitor):
    """Test getting metric statistics."""
    # Record metrics
    performance_monitor.record_metric('response_time', 0.1)
    performance_monitor.record_metric('response_time', 0.2)
    performance_monitor.record_metric('response_time', 0.3)
    
    # Get statistics
    stats = performance_monitor.get_metric_stats('response_time')
    assert stats['min'] == 0.1
    assert stats['max'] == 0.3
    assert stats['avg'] == 0.2
    assert stats['count'] == 3

def test_get_system_metrics(performance_monitor):
    """Test getting system metrics."""
    metrics = performance_monitor.get_system_metrics()
    
    assert 'cpu_percent' in metrics
    assert 'memory_percent' in metrics
    assert 'memory_info' in metrics
    assert 'num_threads' in metrics
    assert 'num_connections' in metrics
    assert 'disk_io' in metrics
    assert 'net_io' in metrics

def test_get_uptime(performance_monitor):
    """Test getting uptime."""
    uptime = performance_monitor.get_uptime()
    assert uptime > 0
    assert uptime < 1  # Should be less than 1 second since initialization

def test_metric_window(performance_monitor):
    """Test metric time window."""
    # Record metrics
    performance_monitor.record_metric('response_time', 0.1)
    time.sleep(0.1)
    performance_monitor.record_metric('response_time', 0.2)
    time.sleep(0.1)
    performance_monitor.record_metric('response_time', 0.3)
    
    # Get metrics with different windows
    assert len(performance_monitor.get_metric('response_time', window=0.05)) == 1
    assert len(performance_monitor.get_metric('response_time', window=0.15)) == 2
    assert len(performance_monitor.get_metric('response_time', window=0.25)) == 3

def test_metric_statistics(performance_monitor):
    """Test metric statistics calculations."""
    # Record metrics
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    for value in values:
        performance_monitor.record_metric('response_time', value)
    
    # Get statistics
    stats = performance_monitor.get_metric_stats('response_time')
    assert stats['min'] == min(values)
    assert stats['max'] == max(values)
    assert stats['avg'] == sum(values) / len(values)
    assert stats['count'] == len(values)

def test_system_metrics_accuracy(performance_monitor):
    """Test system metrics accuracy."""
    # Get initial metrics
    initial_metrics = performance_monitor.get_system_metrics()
    
    # Perform some CPU-intensive operation
    _ = [i * i for i in range(1000000)]
    
    # Get updated metrics
    updated_metrics = performance_monitor.get_system_metrics()
    
    # Verify metrics changed
    assert updated_metrics['cpu_percent'] != initial_metrics['cpu_percent']
    assert updated_metrics['memory_percent'] != initial_metrics['memory_percent']

def test_metric_persistence(performance_monitor):
    """Test metric persistence over time."""
    # Record metrics
    performance_monitor.record_metric('response_time', 0.1)
    time.sleep(0.1)
    performance_monitor.record_metric('response_time', 0.2)
    
    # Get metrics
    metrics = performance_monitor.get_metric('response_time')
    assert len(metrics) == 2
    
    # Wait and verify metrics are still there
    time.sleep(0.1)
    metrics = performance_monitor.get_metric('response_time')
    assert len(metrics) == 2

def test_multiple_metrics(performance_monitor):
    """Test handling multiple metrics."""
    # Record different metrics
    metrics = {
        'response_time': [0.1, 0.2, 0.3],
        'error_rate': [0.01, 0.02],
        'throughput': [100, 200, 300, 400]
    }
    
    for name, values in metrics.items():
        for value in values:
            performance_monitor.record_metric(name, value)
    
    # Verify all metrics
    for name, values in metrics.items():
        assert len(performance_monitor.get_metric(name)) == len(values)
        stats = performance_monitor.get_metric_stats(name)
        assert stats['count'] == len(values)
        assert stats['min'] == min(values)
        assert stats['max'] == max(values)
        assert stats['avg'] == sum(values) / len(values)

def test_metric_cleanup(performance_monitor):
    """Test metric cleanup over time."""
    # Record old metrics
    performance_monitor.record_metric('response_time', 0.1)
    time.sleep(0.1)
    
    # Record new metrics
    performance_monitor.record_metric('response_time', 0.2)
    
    # Get metrics with short window
    metrics = performance_monitor.get_metric('response_time', window=0.05)
    assert len(metrics) == 1
    assert metrics[0][1] == 0.2  # Should only get the newer metric 

@pytest.fixture
def mock_trading_engine():
    """Create a mock trading engine."""
    engine = Mock(spec=TradingEngine)
    engine.get_performance_metrics = Mock(return_value={
        'cpu_usage': 50.0,
        'memory_usage': 60.0,
        'latency': 100.0
    })
    return engine

@pytest.fixture
def mock_websocket_server():
    """Create a mock websocket server."""
    server = Mock(spec=WebSocketServer)
    server.broadcast_performance = AsyncMock()
    return server

@pytest.fixture
def mock_notification_system():
    """Create a mock notification system."""
    system = Mock(spec=NotificationSystem)
    system.send_notification = AsyncMock()
    return system

@pytest.fixture
def performance_optimizer(mock_trading_engine, mock_websocket_server, mock_notification_system):
    """Create a performance optimizer instance with mocked dependencies."""
    optimizer = PerformanceOptimizer()
    optimizer.trading_engine = mock_trading_engine
    optimizer.websocket_server = mock_websocket_server
    optimizer.notification_system = mock_notification_system