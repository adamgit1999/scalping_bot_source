"""
Binance REST wrapper.
"""
from binance.client import Client
from binance.exceptions import BinanceAPIException
from config import Config
import pandas as pd
from datetime import datetime, timedelta

class BinanceBroker:
    def __init__(self):
        self.client = None
        self.config = Config.load_config()
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Binance client with API keys"""
        keys = Config.get_broker_keys('binance')
        if not keys:
            raise ValueError("Binance API keys not configured")
            
        self.client = Client(
            keys.get('api_key'),
            keys.get('api_secret')
        )
    
    def fetch_candles(self, symbol, interval, limit=100):
        """Fetch OHLCV candles from Binance"""
        try:
            # Convert symbol format if needed (e.g., BTC/USDT -> BTCUSDT)
            formatted_symbol = symbol.replace('/', '')
            
            # Get klines/candlestick data
            klines = self.client.get_klines(
                symbol=formatted_symbol,
                interval=interval,
                limit=limit
            )
            
            # Convert to standard format
            candles = []
            for k in klines:
                candle = {
                    'open_time': k[0],
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5]),
                    'close_time': k[6],
                    'quote_volume': float(k[7]),
                    'trades': int(k[8]),
                    'taker_buy_base': float(k[9]),
                    'taker_buy_quote': float(k[10])
                }
                candles.append(candle)
            
            return candles
            
        except BinanceAPIException as e:
            print(f"Binance API error: {e}")
            return []
    
    def get_balance(self):
        """Get account balance"""
        try:
            account = self.client.get_account()
            balances = {}
            
            for asset in account['balances']:
                free = float(asset['free'])
                locked = float(asset['locked'])
                if free > 0 or locked > 0:
                    balances[asset['asset']] = {
                        'free': free,
                        'locked': locked,
                        'total': free + locked
                    }
            
            return balances
            
        except BinanceAPIException as e:
            print(f"Binance API error: {e}")
            return {}
    
    def place_order(self, symbol, side, quantity, order_type='MARKET', price=None):
        """Place an order on Binance"""
        try:
            # Convert symbol format
            formatted_symbol = symbol.replace('/', '')
            
            # Prepare order parameters
            params = {
                'symbol': formatted_symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity
            }
            
            # Add price for limit orders
            if order_type == 'LIMIT' and price:
                params['price'] = price
                params['timeInForce'] = 'GTC'
            
            # Place the order
            order = self.client.create_order(**params)
            
            return {
                'id': order['orderId'],
                'symbol': order['symbol'],
                'side': order['side'],
                'type': order['type'],
                'price': float(order.get('price', 0)),
                'quantity': float(order['origQty']),
                'status': order['status'],
                'time': order['transactTime']
            }
            
        except BinanceAPIException as e:
            print(f"Binance API error: {e}")
            return None
    
    def get_order_status(self, symbol, order_id):
        """Get the status of an order"""
        try:
            formatted_symbol = symbol.replace('/', '')
            order = self.client.get_order(
                symbol=formatted_symbol,
                orderId=order_id
            )
            
            return {
                'id': order['orderId'],
                'symbol': order['symbol'],
                'side': order['side'],
                'type': order['type'],
                'price': float(order.get('price', 0)),
                'quantity': float(order['origQty']),
                'status': order['status'],
                'time': order['time']
            }
            
        except BinanceAPIException as e:
            print(f"Binance API error: {e}")
            return None
    
    def cancel_order(self, symbol, order_id):
        """Cancel an existing order"""
        try:
            formatted_symbol = symbol.replace('/', '')
            result = self.client.cancel_order(
                symbol=formatted_symbol,
                orderId=order_id
            )
            
            return {
                'id': result['orderId'],
                'symbol': result['symbol'],
                'status': 'CANCELED'
            }
            
        except BinanceAPIException as e:
            print(f"Binance API error: {e}")
            return None

