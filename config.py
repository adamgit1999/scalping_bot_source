import os
import json
from pathlib import Path
from cryptography.fernet import Fernet
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Base configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///trading_bot.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'connect_args': {
            'timeout': 30,
            'check_same_thread': False
        }
    }
    
    # Performance settings
    SQLALCHEMY_POOL_SIZE = 10
    SQLALCHEMY_MAX_OVERFLOW = 20
    SQLALCHEMY_POOL_TIMEOUT = 30
    SQLALCHEMY_POOL_RECYCLE = 3600
    
    # SQLite specific settings
    SQLALCHEMY_SQLITE_OPTIONS = {
        'timeout': 30,
        'check_same_thread': False,
        'isolation_level': 'READ COMMITTED'
    }
    
    # Encryption key for sensitive data
    ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', Fernet.generate_key().decode())
    cipher_suite = Fernet(ENCRYPTION_KEY.encode())
    
    # Default settings
    DEFAULT_THEME = 'dark'
    DEFAULT_BROKER = 'binance'
    DEFAULT_SYMBOL = 'BTC/USDT'
    DEFAULT_INTERVAL = '1m'
    DEFAULT_POSITION_SIZE = 0.01
    DEFAULT_AUTO_WITHDRAW = 100
    
    # File paths
    CONFIG_DIR = Path.home() / '.scalping_bot'
    CONFIG_FILE = CONFIG_DIR / 'config.json'
    
    @classmethod
    def init_config(cls):
        """Initialize configuration directory and file"""
        cls.CONFIG_DIR.mkdir(exist_ok=True)
        if not cls.CONFIG_FILE.exists():
            cls.save_config({
                'theme': cls.DEFAULT_THEME,
                'broker': cls.DEFAULT_BROKER,
                'symbol': cls.DEFAULT_SYMBOL,
                'interval': cls.DEFAULT_INTERVAL,
                'position_size': cls.DEFAULT_POSITION_SIZE,
                'auto_withdraw': cls.DEFAULT_AUTO_WITHDRAW,
                'webhook_url': '',
                'show_qr': True,
                'mock_mode': True,
                'pair_filters': [],
                'broker_keys': {}
            })
    
    @classmethod
    def load_config(cls):
        """Load configuration from file"""
        if not cls.CONFIG_FILE.exists():
            cls.init_config()
        
        with open(cls.CONFIG_FILE, 'r') as f:
            config = json.load(f)
        
        # Decrypt sensitive data
        if 'broker_keys' in config:
            for broker, keys in config['broker_keys'].items():
                if isinstance(keys, dict):
                    for key, value in keys.items():
                        if value:
                            try:
                                config['broker_keys'][broker][key] = cls.cipher_suite.decrypt(value.encode()).decode()
                            except:
                                pass
        
        return config
    
    @classmethod
    def save_config(cls, config):
        """Save configuration to file"""
        # Encrypt sensitive data
        if 'broker_keys' in config:
            for broker, keys in config['broker_keys'].items():
                if isinstance(keys, dict):
                    for key, value in keys.items():
                        if value:
                            config['broker_keys'][broker][key] = cls.cipher_suite.encrypt(value.encode()).decode()
        
        with open(cls.CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
    
    @classmethod
    def update_config(cls, updates):
        """Update specific configuration values"""
        config = cls.load_config()
        config.update(updates)
        cls.save_config(config)
    
    @classmethod
    def get_broker_keys(cls, broker):
        """Get decrypted broker API keys"""
        config = cls.load_config()
        keys = config.get('broker_keys', {}).get(broker, {})
        return keys

# Initialize configuration on import
Config.init_config()

# Server
HOST = os.environ.get('HOST', '127.0.0.1')
PORT = int(os.environ.get('PORT', 5000))
DEBUG = True

# Stripe (encrypted config file)
STRIPE_CONFIG_PATH = os.path.join(os.getcwd(), 'stripe_config.enc')

# How often (seconds) to poll for new candles
CANDLE_POLL_INTERVAL = int(os.environ.get('CANDLE_POLL_INTERVAL' ,60))
