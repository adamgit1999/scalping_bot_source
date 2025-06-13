import pytest
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
from datetime import datetime, timezone
from decimal import Decimal
from src.bot_logic import TradingBot
from src.exceptions import TradingError, ValidationError

@pytest.fixture
def mock_broker():
    """Create a mock broker instance."""
    broker = Mock()
    broker.get_balance = AsyncMock(return_value=Decimal('10000.0'))
    broker.place_order = AsyncMock(return_value={'id': 'test_order_id'})
    broker.get_order_status = AsyncMock(return_value='FILLED')
    broker.cancel_order = AsyncMock(return_value=True)
    return broker

@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return {
        'symbol': 'BTC/USDT',
        'interval': '1m',
        'position_size': 0.01,
        'auto_withdraw': 100,
        'mock_mode': True,
        'theme': 'dark',
        'pair_filters': []
    }

@pytest.fixture
def trading_bot(mock_broker, mock_config):
    """Create a trading bot instance."""
    return TradingBot(mock_broker, mock_config)

def test_initialization(trading_bot, mock_broker, mock_config):
    """Test trading bot initialization."""
    assert trading_bot.broker == mock_broker
    assert trading_bot.config == mock_config
    assert trading_bot.symbol == mock_config['symbol']
    assert trading_bot.interval == mock_config['interval']
    assert trading_bot.position_size == mock_config['position_size']
    assert trading_bot.auto_withdraw == mock_config['auto_withdraw']
    assert trading_bot.mock_mode == mock_config['mock_mode']
    assert not trading_bot.is_running
    assert trading_bot.trades == []
    assert trading_bot.equity_curve == []

@pytest.mark.asyncio
async def test_start_stop(trading_bot):
    """Test starting and stopping the trading bot."""
    # Test start
    await trading_bot.start()
    assert trading_bot.is_running
    
    # Test stop
    await trading_bot.stop()
    assert not trading_bot.is_running

@pytest.mark.asyncio
async def test_calculate_indicators(trading_bot):
    """Test indicator calculation."""
    # Create sample data
    data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [101, 102, 103, 104, 105],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    
    # Calculate indicators
    result = trading_bot._calculate_indicators(data)
    
    assert isinstance(result, pd.DataFrame)
    assert 'sma_20' in result.columns
    assert 'rsi_14' in result.columns
    assert 'macd' in result.columns
    assert 'macd_signal' in result.columns
    assert 'macd_hist' in result.columns

def test_generate_signal(trading_bot):
    """Test signal generation."""
    # Create sample data with indicators
    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104],
        'sma_20': [100, 100.5, 101, 101.5, 102],
        'rsi_14': [50, 55, 60, 65, 70],
        'macd': [0, 0.1, 0.2, 0.3, 0.4],
        'macd_signal': [0, 0.05, 0.1, 0.15, 0.2],
        'macd_hist': [0, 0.05, 0.1, 0.15, 0.2]
    })
    
    # Generate signal
    signal = trading_bot._generate_signal(data)
    
    assert isinstance(signal, str)
    assert signal in ['buy', 'sell', 'hold']

@pytest.mark.asyncio
async def test_open_position(trading_bot):
    """Test opening a position."""
    # Mock candle data
    candle = {
        'timestamp': datetime.now(timezone.utc).timestamp(),
        'open': 50000.0,
        'high': 51000.0,
        'low': 49000.0,
        'close': 50500.0,
        'volume': 100.0
    }
    
    # Open position
    await trading_bot._open_position(candle)
    
    # Verify broker calls
    trading_bot.broker.place_order.assert_called_once()
    assert len(trading_bot.trades) == 1
    assert trading_bot.trades[0]['side'] == 'buy'
    assert trading_bot.trades[0]['price'] == candle['close']

@pytest.mark.asyncio
async def test_close_position(trading_bot):
    """Test closing a position."""
    # Mock candle data
    candle = {
        'timestamp': datetime.now(timezone.utc).timestamp(),
        'open': 50000.0,
        'high': 51000.0,
        'low': 49000.0,
        'close': 50500.0,
        'volume': 100.0
    }
    
    # Add a trade to close
    trading_bot.trades.append({
        'side': 'buy',
        'price': 50000.0,
        'amount': 0.01,
        'timestamp': datetime.now(timezone.utc).timestamp()
    })
    
    # Close position
    await trading_bot._close_position(candle)
    
    # Verify broker calls
    trading_bot.broker.place_order.assert_called_once()
    assert len(trading_bot.trades) == 2
    assert trading_bot.trades[1]['side'] == 'sell'
    assert trading_bot.trades[1]['price'] == candle['close']

def test_update_equity(trading_bot):
    """Test equity curve updates."""
    # Mock candle data
    candle = {
        'timestamp': datetime.now(timezone.utc).timestamp(),
        'close': 50000.0
    }
    
    # Update equity
    trading_bot._update_equity(candle)
    
    assert len(trading_bot.equity_curve) == 1
    assert trading_bot.equity_curve[0]['timestamp'] == candle['timestamp']
    assert trading_bot.equity_curve[0]['equity'] == 10000.0  # Initial balance

@pytest.mark.asyncio
async def test_get_balance(trading_bot):
    """Test getting balance."""
    balance = await trading_bot.get_balance()
    assert isinstance(balance, Decimal)
    assert balance == Decimal('10000.0')
    trading_bot.broker.get_balance.assert_called_once()

def test_get_trades(trading_bot):
    """Test getting trades."""
    # Add some trades
    trading_bot.trades = [
        {'side': 'buy', 'price': 50000.0, 'amount': 0.01, 'timestamp': 1234567890},
        {'side': 'sell', 'price': 51000.0, 'amount': 0.01, 'timestamp': 1234567891}
    ]
    
    trades = trading_bot.get_trades()
    assert len(trades) == 2
    assert trades[0]['side'] == 'buy'
    assert trades[1]['side'] == 'sell'

def test_get_equity_curve(trading_bot):
    """Test getting equity curve."""
    # Add some equity points
    trading_bot.equity_curve = [
        {'timestamp': 1234567890, 'equity': 10000.0},
        {'timestamp': 1234567891, 'equity': 10100.0}
    ]
    
    equity_curve = trading_bot.get_equity_curve()
    assert len(equity_curve) == 2
    assert equity_curve[0]['equity'] == 10000.0
    assert equity_curve[1]['equity'] == 10100.0

@pytest.mark.asyncio
async def test_error_handling(trading_bot):
    """Test error handling."""
    # Test broker error
    trading_bot.broker.place_order.side_effect = TradingError("Broker error")
    
    with pytest.raises(TradingError, match="Broker error"):
        await trading_bot._open_position({
            'timestamp': datetime.now(timezone.utc).timestamp(),
            'close': 50000.0
        })
    
    # Test validation error
    with pytest.raises(ValidationError):
        trading_bot._validate_market_data({})
    
    # Test invalid signal
    with pytest.raises(ValueError):
        trading_bot._validate_signal({'type': 'invalid'})

@pytest.mark.asyncio
async def test_market_data_processing(trading_bot):
    """Test market data processing."""
    # Mock market data
    market_data = {
        'symbol': 'BTC/USDT',
        'price': 50000.0,
        'volume': 100.0,
        'timestamp': datetime.now(timezone.utc).timestamp()
    }
    
    # Process market data
    await trading_bot.process_market_data(market_data)
    
    # Verify data was processed
    assert trading_bot.last_price == market_data['price']
    assert trading_bot.last_volume == market_data['volume']
    assert trading_bot.last_update == market_data['timestamp']

@pytest.mark.asyncio
async def test_strategy_execution(trading_bot):
    """Test strategy execution."""
    # Mock strategy data
    strategy = 'scalping'
    parameters = {
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'take_profit': 0.02,
        'stop_loss': 0.01
    }
    
    # Execute strategy
    await trading_bot.execute_strategy(strategy, 'BTC/USDT', '1m', parameters)
    
    # Verify strategy was executed
    assert trading_bot.current_strategy == strategy
    assert trading_bot.strategy_parameters == parameters 