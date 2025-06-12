"""
Broker factory: returns the right module for configured broker.
"""
import importlib
from .binance import BinanceBroker
from .coinbase import CoinbaseBroker
from .kraken import KrakenBroker
from .broker import Broker, Order

def get_broker(broker_name):
    """Factory function to get the appropriate broker instance"""
    brokers = {
        'binance': BinanceBroker,
        'coinbase': CoinbaseBroker,
        'kraken': KrakenBroker
    }
    
    if broker_name not in brokers:
        raise ValueError(f"Unsupported broker: {broker_name}")
        
    return brokers[broker_name]()

__all__ = ['Broker', 'Order']

