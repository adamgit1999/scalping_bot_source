"""
Coinbase Pro REST wrapper.
"""
from src.config import Config
import pandas as pd
from datetime import datetime, timedelta, timezone

# Stub for RESTClient to resolve import error
class RESTClient:
    def __init__(self, *args, **kwargs):
        pass
    def get_accounts(self):
        return []
    def get_orders(self):
        return []
    def place_order(self, *args, **kwargs):
        return {'id': 'stub-order-id'}

class CoinbaseBroker:
    def __init__(self):
        self.client = None
        self.config = Config.load_config()
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Coinbase client with API keys"""
        keys = Config.get_broker_keys('coinbase')
        if not keys:
            raise ValueError("Coinbase API keys not configured")
            
        self.client = RESTClient(
            api_key=keys.get('api_key'),
            api_secret=keys.get('api_secret')
        )
    
    def fetch_candles(self, symbol, interval, limit=100):
        """Fetch OHLCV candles from Coinbase"""
        try:
            # Convert interval to Coinbase format
            interval_map = {
                '1m': 'ONE_MINUTE',
                '5m': 'FIVE_MINUTE',
                '15m': 'FIFTEEN_MINUTE',
                '1h': 'ONE_HOUR',
                '6h': 'SIX_HOUR',
                '1d': 'ONE_DAY'
            }
            granularity = interval_map.get(interval, 'ONE_MINUTE')
            
            # Get historical data
            end = datetime.now(timezone.utc)
            start = end - timedelta(minutes=limit * int(interval[:-1]))
            
            candles = self.client.get_product_candles(
                product_id=symbol,
                start=start.isoformat(),
                end=end.isoformat(),
                granularity=granularity
            )
            
            # Convert to standard format
            formatted_candles = []
            for candle in candles:
                formatted_candle = {
                    'open_time': int(candle.start.timestamp() * 1000),
                    'open': float(candle.open),
                    'high': float(candle.high),
                    'low': float(candle.low),
                    'close': float(candle.close),
                    'volume': float(candle.volume),
                    'close_time': int(candle.end.timestamp() * 1000)
                }
                formatted_candles.append(formatted_candle)
            
            return formatted_candles[-limit:]
            
        except Exception as e:
            print(f"Coinbase API error: {e}")
            return []
    
    def get_balance(self):
        """Get account balance"""
        try:
            accounts = self.client.get_accounts()
            balances = {}
            
            for account in accounts:
                currency = account.currency
                balance = float(account.available)
                if balance > 0:
                    balances[currency] = {
                        'free': balance,
                        'locked': 0.0,
                        'total': balance
                    }
            
            return balances
            
        except Exception as e:
            print(f"Coinbase API error: {e}")
            return {}
    
    def place_order(self, symbol, side, quantity, order_type='MARKET', price=None):
        """Place an order on Coinbase"""
        try:
            # Prepare order parameters
            params = {
                'product_id': symbol,
                'side': side.lower(),  # Coinbase expects lowercase
                'size': str(quantity)
            }
            
            # Add price for limit orders
            if order_type == 'LIMIT' and price:
                params['type'] = 'limit'
                params['price'] = str(price)
            else:
                params['type'] = 'market'
            
            # Place the order
            order = self.client.create_order(**params)
            
            return {
                'id': order.id,
                'symbol': order.product_id,
                'side': order.side.upper(),  # Return uppercase side
                'type': order.type,
                'price': float(order.price) if order.price else 0,
                'quantity': float(order.size),
                'status': order.status,
                'time': int(order.created_at.timestamp() * 1000)
            }
            
        except Exception as e:
            print(f"Coinbase API error: {e}")
            return None
    
    def get_order_status(self, symbol, order_id):
        """Get the status of an order"""
        try:
            order = self.client.get_order(order_id)
            
            return {
                'id': order.id,
                'symbol': order.product_id,
                'side': order.side.upper(),  # Return uppercase side
                'type': order.type,
                'price': float(order.price) if order.price else 0,
                'quantity': float(order.size),
                'status': order.status,
                'time': int(order.created_at.timestamp() * 1000)
            }
            
        except Exception as e:
            print(f"Coinbase API error: {e}")
            return None
    
    def cancel_order(self, symbol, order_id):
        """Cancel an existing order"""
        try:
            result = self.client.cancel_order(order_id)
            
            return {
                'id': result.id,
                'symbol': result.product_id,
                'status': 'CANCELED'
            }
            
        except Exception as e:
            print(f"Coinbase API error: {e}")
            return None

