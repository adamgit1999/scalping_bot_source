import ccxt
import pandas as pd
from datetime import datetime, UTC
from typing import Dict, List, Optional

class MockExchange(ccxt.Exchange):
    def __init__(self):
        super().__init__()
        self.id = 'mock'  # Ensure id is set
        self.markets = {
            'BTC/USDT': {
                'id': 'BTCUSDT',
                'symbol': 'BTC/USDT',
                'base': 'BTC',
                'quote': 'USDT',
                'active': True,
                'precision': {'price': 2, 'amount': 6},
                'limits': {'amount': {'min': 0.0001, 'max': 1000}}
            },
            'ETH/USDT': {
                'id': 'ETHUSDT',
                'symbol': 'ETH/USDT',
                'base': 'ETH',
                'quote': 'USDT',
                'active': True,
                'precision': {'price': 2, 'amount': 6},
                'limits': {'amount': {'min': 0.001, 'max': 1000}}
            },
            'BTC/USD': {},
            'GBP/USD': {},
            'EUR/USD': {}
        }
        self.balances = {
            'USDT': {'free': 10000.0, 'used': 0.0, 'total': 10000.0},
            'BTC': {'free': 1.0, 'used': 0.0, 'total': 1.0},
            'ETH': {'free': 10.0, 'used': 0.0, 'total': 10.0},
            'USD': {'free': 10000.0, 'used': 0.0, 'total': 10000.0},
            'GBP': {'free': 5000.0, 'used': 0.0, 'total': 5000.0},
            'EUR': {'free': 7000.0, 'used': 0.0, 'total': 7000.0}
        }
        self.orders = {}
        self.tickers = {
            'BTC/USDT': {'last': 50000, 'bid': 49900, 'ask': 50100},
            'ETH/USDT': {'last': 3000, 'bid': 2990, 'ask': 3010}
        }

    async def load_markets(self, reload=False):
        return self.markets

    async def fetch_balance(self):
        return self.balances

    async def fetch_balance_for_currency(self, currency):
        if currency not in self.balances:
            raise ValueError(f"Currency {currency} not found")
        return self.balances[currency]

    async def create_order(self, symbol: str, type: str, side: str, amount: float, price: Optional[float] = None):
        order_id = str(len(self.orders) + 1)
        order = {
            'id': order_id,
            'symbol': symbol,
            'type': type,
            'side': side,
            'amount': amount,
            'price': price,
            'status': 'open',
            'timestamp': datetime.now(UTC).timestamp()
        }
        self.orders[order_id] = order
        return order

    async def cancel_order(self, order_id: str):
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'canceled'
            return True
        return False

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100):
        # Only support 'BTC/USDT' and 'ETH/USDT' and '1h' timeframe
        if symbol not in self.markets:
            raise ValueError("Invalid symbol")
        if timeframe != '1h':
            raise ValueError("Invalid timeframe")
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1h')
        data = []
        base_price = 50000 if symbol == 'BTC/USDT' else 3000
        for date in dates:
            price = base_price * (1 + 0.001 * (hash(str(date)) % 100 - 50) / 100)
            data.append([
                int(date.timestamp() * 1000),
                price * 0.99,
                price * 1.01,
                price * 0.98,
                price,
                1000
            ])
        return data[:limit]

    async def fetch_ticker(self, symbol: str):
        return self.tickers.get(symbol, {'last': 0, 'bid': 0, 'ask': 0})

    async def fetch_positions(self, symbols=None, params={}):
        return [] 