class TradingError(Exception):
    """Base exception for trading errors."""
    pass

class DataProcessingError(TradingError):
    """Exception for data processing errors."""
    pass

class ValidationError(TradingError):
    """Exception for validation errors."""
    pass

class WebSocketError(TradingError):
    """Exception for WebSocket errors."""
    pass

class BrokerError(TradingError):
    """Exception for broker errors."""
    pass

class OrderError(TradingError):
    """Exception for order errors."""
    pass

class StrategyError(TradingError):
    """Exception for strategy errors."""
    pass

class BacktestError(TradingError):
    """Exception for backtest errors."""
    pass

class PerformanceError(TradingError):
    """Exception for performance monitoring errors."""
    pass

class NotificationError(TradingError):
    """Exception for notification errors."""
    pass

class DatabaseError(TradingError):
    """Exception for database errors."""
    pass

class ConfigurationError(TradingError):
    """Exception for configuration errors."""
    pass

class AuthenticationError(TradingError):
    """Exception for authentication errors."""
    pass

class AuthorizationError(TradingError):
    """Exception for authorization errors."""
    pass

class RateLimitError(TradingError):
    """Exception for rate limit errors."""
    pass

class NetworkError(TradingError):
    """Exception for network errors."""
    pass

class TimeoutError(TradingError):
    """Exception for timeout errors."""
    pass

class ResourceError(TradingError):
    """Exception for resource errors."""
    pass

class CacheError(TradingError):
    """Exception for cache errors."""
    pass

class MetricsError(TradingError):
    """Exception for metrics errors."""
    pass

class ThreadPoolError(TradingError):
    """Exception for thread pool errors."""
    pass

class InsufficientFundsError(TradingError):
    """Raised when there are insufficient funds for an order."""
    pass

class InvalidOrderError(TradingError):
    """Raised when an order is invalid."""
    pass

class ExchangeError(TradingError):
    """Raised when there is an error communicating with the exchange."""
    pass

class RiskManagerError(TradingError):
    """Raised when there is an error in risk management."""
    pass

class MarketDataError(TradingError):
    """Raised when there is an error with market data."""
    pass

class OptimizationError(Exception):
    """Raised when there is an error in performance optimization calculations."""
    pass

class ExecutionError(Exception):
    """Raised when there is an execution error in trading or backtesting."""
    pass

class StorageError(Exception):
    """Raised when there is a storage or file access error."""
    pass

class RiskLimitExceededError(TradingError):
    """Raised when a risk limit is exceeded."""
    pass

class ConfigurationError(TradingError):
    """Raised when there is an error with configuration."""
    pass 