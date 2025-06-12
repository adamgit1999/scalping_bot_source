import pytest
from unittest.mock import patch
from performance.optimizer import PerformanceOptimizer

@pytest.fixture(autouse=True)
def patch_measure_exchange_latency():
    with patch.object(PerformanceOptimizer, '_measure_exchange_latency', return_value=50.0):
        yield 