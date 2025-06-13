import os
import json
from pathlib import Path
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from typing import Dict

# Load environment variables
load_dotenv()

class Config:
    # Base configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///trading_bot.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Enhanced database configuration
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 20,  # Increased pool size
        'max_overflow': 30,  # Increased max overflow
        'pool_timeout': 30,
        'pool_recycle': 1800,  # Recycle connections every 30 minutes
        'pool_pre_ping': True,  # Enable connection health checks
        'echo': False,  # Disable SQL logging in production
        'connect_args': {
            'timeout': 30,
            'check_same_thread': False,
            'isolation_level': 'READ COMMITTED'  # Better concurrency
        }
    }
    
    # Performance settings
    SQLALCHEMY_POOL_SIZE = 20
    SQLALCHEMY_MAX_OVERFLOW = 30
    SQLALCHEMY_POOL_TIMEOUT = 30
    SQLALCHEMY_POOL_RECYCLE = 1800
    
    # SQLite specific settings
    SQLALCHEMY_SQLITE_OPTIONS = {
        'timeout': 30,
        'check_same_thread': False,
        'isolation_level': 'READ COMMITTED',
        'journal_mode': 'WAL',  # Write-Ahead Logging for better concurrency
        'synchronous': 'NORMAL',  # Balance between safety and performance
        'cache_size': -2000,  # Use 2MB of cache
        'temp_store': 'MEMORY',  # Store temp tables in memory
        'mmap_size': 30000000000  # 30GB memory map
    }
    
    # Connection pooling settings
    DB_POOL_RECYCLE = 1800
    DB_POOL_TIMEOUT = 30
    DB_MAX_OVERFLOW = 30
    DB_POOL_SIZE = 20
    
    # Performance monitoring
    ENABLE_PERFORMANCE_MONITORING = True
    PERFORMANCE_METRICS_INTERVAL = 60  # seconds
    PERFORMANCE_LOG_LEVEL = 'INFO'
    
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
    
    # Performance thresholds
    MAX_MEMORY_USAGE = 80  # percentage
    MAX_CPU_USAGE = 80  # percentage
    MAX_LATENCY = 100  # milliseconds
    MAX_CONCURRENT_CONNECTIONS = 50
    
    # Cache settings
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300
    CACHE_THRESHOLD = 1000
    
    # Logging configuration
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = CONFIG_DIR / 'trading_bot.log'
    
    # Error handling
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    ERROR_RESET_INTERVAL = 3600  # 1 hour
    
    # Market data settings
    MARKET_DATA_BUFFER_SIZE = 1000
    MARKET_DATA_PROCESSING_INTERVAL = 0.05  # seconds
    MARKET_DATA_CACHE_SIZE = 1000
    
    # Order processing settings
    ORDER_PROCESSING_TIMEOUT = 30  # seconds
    MAX_ORDER_RETRIES = 3
    ORDER_RETRY_DELAY = 1  # seconds
    
    # Strategy settings
    STRATEGY_UPDATE_INTERVAL = 1  # seconds
    MAX_STRATEGY_ERRORS = 100
    STRATEGY_ERROR_RESET_INTERVAL = 3600  # 1 hour
    
    # Risk management settings
    MAX_POSITION_SIZE = 1.0
    MIN_POSITION_SIZE = 0.001
    MAX_LEVERAGE = 10
    MAX_DAILY_TRADES = 100
    MAX_DAILY_LOSS = 0.1  # 10% of account
    MAX_DRAWDOWN = 0.2  # 20% of account
    
    # API settings
    API_TIMEOUT = 30  # seconds
    API_RATE_LIMIT = 100  # requests per minute
    API_MAX_RETRIES = 3
    API_RETRY_DELAY = 1  # seconds
    
    # WebSocket settings
    WS_PING_INTERVAL = 30  # seconds
    WS_PING_TIMEOUT = 10  # seconds
    WS_RECONNECT_DELAY = 1  # seconds
    WS_MAX_RECONNECT_ATTEMPTS = 5
    
    # Notification settings
    ENABLE_NOTIFICATIONS = True
    NOTIFICATION_CHANNELS = ['email', 'telegram']
    NOTIFICATION_THRESHOLD = 0.05  # 5% change
    
    # Backup settings
    ENABLE_AUTO_BACKUP = True
    BACKUP_INTERVAL = 3600  # 1 hour
    MAX_BACKUPS = 24  # Keep last 24 backups
    
    # Security settings
    ENABLE_2FA = True
    SESSION_TIMEOUT = 3600  # 1 hour
    MAX_LOGIN_ATTEMPTS = 5
    PASSWORD_MIN_LENGTH = 12
    
    # Monitoring settings
    ENABLE_HEALTH_CHECK = True
    HEALTH_CHECK_INTERVAL = 60  # seconds
    HEALTH_CHECK_TIMEOUT = 10  # seconds
    
    # Debug settings
    DEBUG = False
    TESTING = False
    DEVELOPMENT = False
    
    @classmethod
    def get_db_config(cls) -> Dict:
        """Get database configuration."""
        return {
            'pool_size': cls.SQLALCHEMY_POOL_SIZE,
            'max_overflow': cls.SQLALCHEMY_MAX_OVERFLOW,
            'pool_timeout': cls.SQLALCHEMY_POOL_TIMEOUT,
            'pool_recycle': cls.SQLALCHEMY_POOL_RECYCLE,
            'pool_pre_ping': True,
            'echo': cls.DEBUG,
            'connect_args': cls.SQLALCHEMY_SQLITE_OPTIONS
        }
    
    @classmethod
    def get_performance_config(cls) -> Dict:
        """Get performance configuration."""
        return {
            'max_memory_usage': cls.MAX_MEMORY_USAGE,
            'max_cpu_usage': cls.MAX_CPU_USAGE,
            'max_latency': cls.MAX_LATENCY,
            'max_concurrent_connections': cls.MAX_CONCURRENT_CONNECTIONS,
            'cache_type': cls.CACHE_TYPE,
            'cache_timeout': cls.CACHE_DEFAULT_TIMEOUT,
            'cache_threshold': cls.CACHE_THRESHOLD
        }
    
    @classmethod
    def get_error_handling_config(cls) -> Dict:
        """Get error handling configuration."""
        return {
            'max_retries': cls.MAX_RETRIES,
            'retry_delay': cls.RETRY_DELAY,
            'error_reset_interval': cls.ERROR_RESET_INTERVAL,
            'max_strategy_errors': cls.MAX_STRATEGY_ERRORS,
            'strategy_error_reset_interval': cls.STRATEGY_ERROR_RESET_INTERVAL
        }
    
    @classmethod
    def get_market_data_config(cls) -> Dict:
        """Get market data configuration."""
        return {
            'buffer_size': cls.MARKET_DATA_BUFFER_SIZE,
            'processing_interval': cls.MARKET_DATA_PROCESSING_INTERVAL,
            'cache_size': cls.MARKET_DATA_CACHE_SIZE
        }
    
    @classmethod
    def get_order_processing_config(cls) -> Dict:
        """Get order processing configuration."""
        return {
            'timeout': cls.ORDER_PROCESSING_TIMEOUT,
            'max_retries': cls.MAX_ORDER_RETRIES,
            'retry_delay': cls.ORDER_RETRY_DELAY
        }
    
    @classmethod
    def get_risk_management_config(cls) -> Dict:
        """Get risk management configuration."""
        return {
            'max_position_size': cls.MAX_POSITION_SIZE,
            'min_position_size': cls.MIN_POSITION_SIZE,
            'max_leverage': cls.MAX_LEVERAGE,
            'max_daily_trades': cls.MAX_DAILY_TRADES,
            'max_daily_loss': cls.MAX_DAILY_LOSS,
            'max_drawdown': cls.MAX_DRAWDOWN
        }
    
    @classmethod
    def get_api_config(cls) -> Dict:
        """Get API configuration."""
        return {
            'timeout': cls.API_TIMEOUT,
            'rate_limit': cls.API_RATE_LIMIT,
            'max_retries': cls.API_MAX_RETRIES,
            'retry_delay': cls.API_RETRY_DELAY
        }
    
    @classmethod
    def get_websocket_config(cls) -> Dict:
        """Get WebSocket configuration."""
        return {
            'ping_interval': cls.WS_PING_INTERVAL,
            'ping_timeout': cls.WS_PING_TIMEOUT,
            'reconnect_delay': cls.WS_RECONNECT_DELAY,
            'max_reconnect_attempts': cls.WS_MAX_RECONNECT_ATTEMPTS
        }
    
    @classmethod
    def get_notification_config(cls) -> Dict:
        """Get notification configuration."""
        return {
            'enabled': cls.ENABLE_NOTIFICATIONS,
            'channels': cls.NOTIFICATION_CHANNELS,
            'threshold': cls.NOTIFICATION_THRESHOLD
        }
    
    @classmethod
    def get_backup_config(cls) -> Dict:
        """Get backup configuration."""
        return {
            'enabled': cls.ENABLE_AUTO_BACKUP,
            'interval': cls.BACKUP_INTERVAL,
            'max_backups': cls.MAX_BACKUPS
        }
    
    @classmethod
    def get_security_config(cls) -> Dict:
        """Get security configuration."""
        return {
            'enable_2fa': cls.ENABLE_2FA,
            'session_timeout': cls.SESSION_TIMEOUT,
            'max_login_attempts': cls.MAX_LOGIN_ATTEMPTS,
            'password_min_length': cls.PASSWORD_MIN_LENGTH
        }
    
    @classmethod
    def get_monitoring_config(cls) -> Dict:
        """Get monitoring configuration."""
        return {
            'enabled': cls.ENABLE_HEALTH_CHECK,
            'interval': cls.HEALTH_CHECK_INTERVAL,
            'timeout': cls.HEALTH_CHECK_TIMEOUT
        }
    
    @classmethod
    def get_debug_config(cls) -> Dict:
        """Get debug configuration."""
        return {
            'debug': cls.DEBUG,
            'testing': cls.TESTING,
            'development': cls.DEVELOPMENT
        }
    
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
