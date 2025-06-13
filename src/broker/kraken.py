"""
Kraken REST wrapper.
"""
# Stub for KrakenAPI to resolve import error
class KrakenAPI:
    def __init__(self, *args, **kwargs):
        pass
    def query_public(self, *args, **kwargs):
        return {}
    def query_private(self, *args, **kwargs):
        return {}

from src.config import Config

class KrakenBroker:
    def __init__(self):
        self.client = None
        self.api = None
        self.config = Config.load_config()
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Kraken client with API keys"""
        keys = Config.get_broker_keys('kraken')
        if not keys:
            raise ValueError("Kraken API keys not configured")
            
        self.client = krakenex.API(
            key=keys.get('api_key'),
            secret=keys.get('api_secret')
        )
        self.api = KrakenAPI(self.client)
    
    def fetch_candles(self, symbol, interval, limit=100):
        """Fetch OHLCV candles from Kraken"""
        try:
            # Convert interval to Kraken format
            interval_map = {
                '1m': 1,
                '5m': 5,
                '15m': 15,
                '1h': 60,
                '6h': 360,
                '1d': 1440
            }
            minutes = interval_map.get(interval, 1)
            
            # Get OHLC data
            ohlc, last = self.api.get_ohlc_data(
                symbol,
                interval=minutes,
                since=last
            )
            
            # Convert to standard format
            candles = []
            for index, row in ohlc.iterrows():
                candle = {
                    'open_time': int(index.timestamp() * 1000),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                    'close_time': int((index + timedelta(minutes=minutes)).timestamp() * 1000)
                }
                candles.append(candle)
            
            return candles[-limit:]
            
        except Exception as e:
            print(f"Kraken API error: {e}")
            return []
    
    def get_balance(self):
        """Get account balance"""
        try:
            balance = self.api.get_account_balance()
            balances = {}
            
            for currency, amount in balance.items():
                if amount > 0:
                    balances[currency] = {
                        'free': float(amount),
                        'locked': 0.0,
                        'total': float(amount)
                    }
            
            return balances
            
        except Exception as e:
            print(f"Kraken API error: {e}")
            return {}
    
    def place_order(self, symbol, side, quantity, order_type='MARKET', price=None):
        """Place an order on Kraken"""
        try:
            # Prepare order parameters
            params = {
                'pair': symbol,
                'type': side.lower(),  # Kraken expects lowercase
                'ordertype': order_type.lower(),
                'volume': str(quantity)
            }
            
            # Add price for limit orders
            if order_type == 'LIMIT' and price:
                params['price'] = str(price)
            
            # Place the order
            result = self.client.query_private('AddOrder', params)
            
            if result['error']:
                raise Exception(result['error'])
            
            order_id = result['result']['txid'][0]
            order = self.get_order_status(symbol, order_id)
            
            # Ensure side is uppercase in response
            if order:
                order['side'] = order['side'].upper()
            
            return order
            
        except Exception as e:
            print(f"Kraken API error: {e}")
            return None
    
    def get_order_status(self, symbol, order_id):
        """Get the status of an order"""
        try:
            result = self.client.query_private('QueryOrders', {
                'txid': order_id
            })
            
            if result['error']:
                raise Exception(result['error'])
            
            order = result['result'][order_id]
            
            return {
                'id': order_id,
                'symbol': order['descr']['pair'],
                'side': order['descr']['type'].upper(),  # Return uppercase side
                'type': order['descr']['ordertype'],
                'price': float(order['price']),
                'quantity': float(order['vol']),
                'status': order['status'],
                'time': int(order['opentm'] * 1000)
            }
            
        except Exception as e:
            print(f"Kraken API error: {e}")
            return None
    
    def cancel_order(self, symbol, order_id):
        """Cancel an existing order"""
        try:
            result = self.client.query_private('CancelOrder', {
                'txid': order_id
            })
            
            if result['error']:
                raise Exception(result['error'])
            
            return {
                'id': order_id,
                'symbol': symbol,
                'status': 'CANCELED'
            }
            
        except Exception as e:
            print(f"Kraken API error: {e}")
            return None

