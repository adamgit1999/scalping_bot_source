import time
from typing import Dict, List, Optional
import logging

class PerformanceOptimizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.latency_measurements: List[float] = []
        self.performance_metrics: Dict[str, List[float]] = {
            'order_processing_time': [],
            'market_data_processing_time': [],
            'strategy_execution_time': []
        }

    async def _measure_exchange_latency(self) -> float:
        """Measure the latency to the exchange"""
        try:
            start_time = time.time()
            # Simulate exchange latency measurement
            await asyncio.sleep(0.05)  # 50ms latency
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.latency_measurements.append(latency)
            return latency
        except Exception as e:
            self.logger.error(f"Error measuring exchange latency: {str(e)}")
            return 0.0

    def record_metric(self, metric_name: str, value: float) -> None:
        """Record a performance metric"""
        if metric_name in self.performance_metrics:
            self.performance_metrics[metric_name].append(value)

    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a specific metric"""
        if metric_name not in self.performance_metrics:
            return {}
        
        values = self.performance_metrics[metric_name]
        if not values:
            return {}
        
        return {
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'count': len(values)
        }

    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics"""
        return {
            name: self.get_metric_stats(name)
            for name in self.performance_metrics
        }

    def clear_metrics(self) -> None:
        """Clear all performance metrics"""
        for metric_list in self.performance_metrics.values():
            metric_list.clear()
        self.latency_measurements.clear() 