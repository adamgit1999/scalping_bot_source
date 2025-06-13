import pytest
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from src.models import User, Trade, Notification, Webhook, Backtest, db
from src.backtest_engine import BacktestEngine
from src.broker import BrokerInterface
from src.risk_manager import RiskManager
from src.data_processing import MarketDataStore
from src.performance_monitoring import PerformanceMonitor
from src.indicators import TechnicalIndicators
from src.config import Config

@pytest.fixture
def mock_broker():
    """Create a mock broker for testing."""
    broker = Mock(spec=BrokerInterface)
    broker.get_balance = AsyncMock(return_value=Decimal('10000.00'))
    broker.get_position = AsyncMock(return_value=None)
    broker.place_order = AsyncMock(return_value={
        'id': 'test_order_1',
        'symbol': 'BTC/USDT',
        'side': 'BUY',
        'type': 'limit',
        'price': Decimal('100.00'),
        'quantity': Decimal('1.0'),
        'status': 'FILLED'
    })
    return broker

@pytest.fixture
def mock_risk_manager():
    """Create a mock risk manager for testing."""
    risk_manager = Mock(spec=RiskManager)
    risk_manager.check_order = AsyncMock(return_value=(True, "OK"))
    risk_manager.calculate_position_size = AsyncMock(return_value=Decimal('1.0'))
    risk_manager.check_risk_limits = AsyncMock(return_value=True)
    return risk_manager

@pytest.fixture
def mock_market_data_store():
    """Create a mock market data store for testing."""
    store = Mock(spec=MarketDataStore)
    store.get_latest_data = AsyncMock(return_value=pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [102, 103, 104],
        'volume': [1000, 1100, 1200]
    }))
    return store

@pytest.fixture
def mock_performance_monitor():
    """Create a mock performance monitor for testing."""
    monitor = Mock(spec=PerformanceMonitor)
    monitor.update_metrics = AsyncMock(return_value={
        'win_rate': 0.6,
        'profit_factor': 1.5,
        'sharpe_ratio': 1.2
    })
    return monitor

@pytest.fixture
def mock_indicators():
    """Create a mock technical indicators instance."""
    indicators = Mock(spec=TechnicalIndicators)
    indicators.calculate_sma = AsyncMock(return_value=pd.Series([100, 101, 102]))
    indicators.calculate_rsi = AsyncMock(return_value=pd.Series([50, 51, 52]))
    indicators.calculate_macd = AsyncMock(return_value={
        'macd': pd.Series([1, 2, 3]),
        'signal': pd.Series([0.5, 1, 1.5]),
        'histogram': pd.Series([0.5, 1, 1.5])
    })
    return indicators

@pytest.fixture
def mock_config():
    """Create a mock configuration instance."""
    config = Mock(spec=Config)
    config.get = Mock(return_value={
        'risk_management': {
            'max_position_size': 1.0,
            'max_drawdown': 0.1,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.05
        },
        'backtest': {
            'initial_balance': 10000.0,
            'commission': 0.001,
            'slippage': 0.001,
            'risk_free_rate': 0.02,
            'trading_fee': 0.001,
            'funding_rate': 0.0001
        }
    })
    return config

@pytest.fixture
def backtest_engine(mock_broker, mock_risk_manager, mock_market_data_store,
                   mock_performance_monitor, mock_indicators, mock_config):
    """Create a backtest engine instance with updated dependencies."""
    return BacktestEngine(
        broker=mock_broker,
        risk_manager=mock_risk_manager,
        market_data_store=mock_market_data_store,
        performance_monitor=mock_performance_monitor,
        indicators=mock_indicators,
        config=mock_config
    )

@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='h', tz=timezone.utc)
    data = pd.DataFrame({
        'open': np.random.normal(100, 1, len(dates)),
        'high': np.random.normal(101, 1, len(dates)),
        'low': np.random.normal(99, 1, len(dates)),
        'close': np.random.normal(100, 1, len(dates)),
        'volume': np.random.normal(1000, 100, len(dates))
    }, index=dates)
    # Ensure price consistency
    data['high'] = data[['open', 'close']].max(axis=1) + np.random.uniform(0, 1, len(dates))
    data['low'] = data[['open', 'close']].min(axis=1) - np.random.uniform(0, 1, len(dates))
    return data

@pytest.fixture
def mock_strategy():
    """Create a mock trading strategy."""
    strategy = Mock(spec=Strategy)
    strategy.generate_signals = AsyncMock(return_value=pd.Series(
        [1, -1, 0, 1, -1],
        index=pd.date_range(start='2024-01-01', periods=5, freq='h', tz=timezone.utc)
    ))
    return strategy

@pytest.mark.asyncio
async def test_initialize(backtest_engine):
    """Test backtest engine initialization."""
    assert backtest_engine.initial_balance == Decimal('10000')
    assert backtest_engine.commission == Decimal('0.001')
    assert backtest_engine.positions == {}
    assert backtest_engine.trades == []
    assert backtest_engine.equity_curve == []
    assert backtest_engine.current_balance == Decimal('10000')
    assert backtest_engine.total_commission == Decimal('0')
    assert backtest_engine.total_trades == 0
    assert backtest_engine.winning_trades == 0
    assert backtest_engine.losing_trades == 0
    assert backtest_engine.max_position_size == Decimal('1.0')
    assert backtest_engine.min_position_size == Decimal('0.001')
    assert backtest_engine.slippage == Decimal('0.001')
    assert backtest_engine.risk_free_rate == Decimal('0.02')
    assert backtest_engine.trading_fee == Decimal('0.001')
    assert backtest_engine.funding_rate == Decimal('0.0001')

@pytest.mark.asyncio
async def test_run_backtest(backtest_engine, sample_data, mock_strategy):
    """Test running a backtest."""
    strategy = 'scalping'
    parameters = {
        'sma_short': 10,
        'sma_long': 20,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'stop_loss': 0.02,
        'take_profit': 0.03,
        'trailing_stop': 0.01
    }
    
    with patch('src.backtest.backtest_engine.get_strategy', return_value=mock_strategy):
        results = await backtest_engine.run_backtest(sample_data, strategy, parameters)
    
    assert isinstance(results, dict)
    assert 'total_return' in results
    assert 'sharpe_ratio' in results
    assert 'max_drawdown' in results
    assert 'win_rate' in results
    assert 'trades' in results
    assert 'equity_curve' in results
    assert 'total_commission' in results
    assert 'total_trades' in results
    assert 'winning_trades' in results
    assert 'losing_trades' in results
    assert 'average_win' in results
    assert 'average_loss' in results
    assert 'profit_factor' in results
    assert 'max_consecutive_wins' in results
    assert 'max_consecutive_losses' in results
    assert 'annualized_return' in results
    assert 'annualized_volatility' in results
    assert 'sortino_ratio' in results
    assert 'calmar_ratio' in results
    assert 'max_leverage' in results
    assert 'average_leverage' in results
    assert 'funding_payments' in results
    assert 'trading_fees' in results

@pytest.mark.asyncio
async def test_optimize_strategy(backtest_engine, sample_data, mock_strategy):
    """Test strategy optimization."""
    strategy = 'scalping'
    parameter_ranges = {
        'sma_short': range(5, 15),
        'sma_long': range(15, 25),
        'rsi_period': range(10, 20),
        'stop_loss': np.arange(0.01, 0.05, 0.01),
        'take_profit': np.arange(0.02, 0.06, 0.01)
    }
    
    with patch('src.backtest.backtest_engine.get_strategy', return_value=mock_strategy):
        optimized_params = await backtest_engine.optimize_strategy(
            sample_data,
            strategy,
            parameter_ranges
        )
    
    assert isinstance(optimized_params, dict)
    assert 'sma_short' in optimized_params
    assert 'sma_long' in optimized_params
    assert 'rsi_period' in optimized_params
    assert 'stop_loss' in optimized_params
    assert 'take_profit' in optimized_params
    assert optimized_params['sma_short'] >= 5
    assert optimized_params['sma_short'] <= 15
    assert optimized_params['sma_long'] >= 15
    assert optimized_params['sma_long'] <= 25
    assert optimized_params['rsi_period'] >= 10
    assert optimized_params['rsi_period'] <= 20

@pytest.mark.asyncio
async def test_risk_management_integration(backtest_engine, sample_data, mock_strategy):
    """Test risk management integration."""
    strategy = 'scalping'
    parameters = {
        'sma_short': 10,
        'sma_long': 20,
        'rsi_period': 14,
        'stop_loss': 0.02,
        'take_profit': 0.03
    }
    
    with patch('src.backtest.backtest_engine.get_strategy', return_value=mock_strategy):
        results = await backtest_engine.run_backtest(sample_data, strategy, parameters)
    
    # Verify risk management checks
    assert results['max_drawdown'] <= 0.1  # Max drawdown limit
    assert results['max_leverage'] <= 1.0  # Max leverage limit
    assert all(trade['position_size'] <= 1.0 for trade in results['trades'])  # Position size limit

@pytest.mark.asyncio
async def test_performance_monitoring_integration(backtest_engine, sample_data, mock_strategy):
    """Test performance monitoring integration."""
    strategy = 'scalping'
    parameters = {
        'sma_short': 10,
        'sma_long': 20,
        'rsi_period': 14,
        'stop_loss': 0.02,
        'take_profit': 0.03
    }
    
    with patch('src.backtest.backtest_engine.get_strategy', return_value=mock_strategy):
        results = await backtest_engine.run_backtest(sample_data, strategy, parameters)
    
    # Verify performance metrics
    assert results['sharpe_ratio'] is not None
    assert results['sortino_ratio'] is not None
    assert results['calmar_ratio'] is not None
    assert results['annualized_return'] is not None
    assert results['annualized_volatility'] is not None

@pytest.mark.asyncio
async def test_market_data_integration(backtest_engine, sample_data, mock_strategy):
    """Test market data integration."""
    strategy = 'scalping'
    parameters = {
        'sma_short': 10,
        'sma_long': 20,
        'rsi_period': 14,
        'stop_loss': 0.02,
        'take_profit': 0.03
    }
    
    with patch('src.backtest.backtest_engine.get_strategy', return_value=mock_strategy):
        results = await backtest_engine.run_backtest(sample_data, strategy, parameters)
    
    # Verify market data processing
    assert len(results['trades']) > 0
    assert all(trade['price'] > 0 for trade in results['trades'])
    assert all(trade['volume'] > 0 for trade in results['trades'])
    assert all(trade['timestamp'] is not None for trade in results['trades'])

@pytest.mark.asyncio
async def test_parameter_validation(backtest_engine, sample_data, mock_strategy):
    """Test parameter validation."""
    strategy = 'scalping'
    invalid_parameters = {
        'sma_short': -1,  # Invalid value
        'sma_long': 5,    # Less than sma_short
        'rsi_period': 0,  # Invalid value
        'stop_loss': 0,   # Invalid value
        'take_profit': -0.01  # Invalid value
    }
    
    with patch('src.backtest.backtest_engine.get_strategy', return_value=mock_strategy):
        with pytest.raises(ValueError):
            await backtest_engine.run_backtest(sample_data, strategy, invalid_parameters)

@pytest.mark.asyncio
async def test_memory_management(backtest_engine, sample_data, mock_strategy):
    """Test memory management during backtest."""
    strategy = 'scalping'
    parameters = {
        'sma_short': 10,
        'sma_long': 20,
        'rsi_period': 14,
        'stop_loss': 0.02,
        'take_profit': 0.03
    }
    
    # Create large dataset
    large_data = pd.concat([sample_data] * 10)
    
    with patch('src.backtest.backtest_engine.get_strategy', return_value=mock_strategy):
        results = await backtest_engine.run_backtest(large_data, strategy, parameters)
    
    # Verify memory usage
    assert len(results['trades']) > 0
    assert len(results['equity_curve']) > 0
    assert backtest_engine.memory_usage < 1024 * 1024 * 1024  # Less than 1GB

@pytest.mark.asyncio
async def test_export_results(backtest_engine, sample_data, mock_strategy):
    """Test exporting backtest results."""
    strategy = 'scalping'
    parameters = {
        'sma_short': 10,
        'sma_long': 20,
        'rsi_period': 14,
        'stop_loss': 0.02,
        'take_profit': 0.03
    }
    
    with patch('src.backtest.backtest_engine.get_strategy', return_value=mock_strategy):
        results = await backtest_engine.run_backtest(sample_data, strategy, parameters)
    
    # Export results
    export_data = await backtest_engine.export_results(results)
    
    assert isinstance(export_data, dict)
    assert 'summary' in export_data
    assert 'trades' in export_data
    assert 'equity_curve' in export_data
    assert 'parameters' in export_data
    assert 'metrics' in export_data

@pytest.mark.asyncio
async def test_technical_indicators_integration(backtest_engine, sample_data):
    """Test integration of technical indicators in backtesting."""
    # Test SMA calculation
    sma_values = await backtest_engine.indicators.calculate_sma(sample_data['close'], window=20)
    assert isinstance(sma_values, pd.Series)
    assert len(sma_values) == len(sample_data)
    assert not sma_values.isna().all()
    
    # Test RSI calculation
    rsi_values = await backtest_engine.indicators.calculate_rsi(sample_data['close'], period=14)
    assert isinstance(rsi_values, pd.Series)
    assert len(rsi_values) == len(sample_data)
    assert not rsi_values.isna().all()
    assert all(0 <= x <= 100 for x in rsi_values.dropna())
    
    # Test MACD calculation
    macd_result = await backtest_engine.indicators.calculate_macd(
        sample_data['close'],
        fast_period=12,
        slow_period=26,
        signal_period=9
    )
    assert isinstance(macd_result, dict)
    assert 'macd' in macd_result
    assert 'signal' in macd_result
    assert 'histogram' in macd_result
    assert all(isinstance(x, pd.Series) for x in macd_result.values())

@pytest.mark.asyncio
async def test_configuration_integration(backtest_engine):
    """Test integration of configuration settings in backtesting."""
    # Test risk management config
    risk_config = backtest_engine.config.get('risk_management')
    assert risk_config['max_position_size'] == 1.0
    assert risk_config['max_drawdown'] == 0.1
    assert risk_config['stop_loss_pct'] == 0.02
    assert risk_config['take_profit_pct'] == 0.05
    
    # Test backtest config
    backtest_config = backtest_engine.config.get('backtest')
    assert backtest_config['initial_balance'] == 10000.0
    assert backtest_config['commission'] == 0.001
    assert backtest_config['slippage'] == 0.001
    assert backtest_config['risk_free_rate'] == 0.02
    assert backtest_config['trading_fee'] == 0.001
    assert backtest_config['funding_rate'] == 0.0001

@pytest.mark.asyncio
async def test_multi_timeframe_analysis(backtest_engine, sample_data):
    """Test multi-timeframe analysis in backtesting."""
    # Create multi-timeframe data
    timeframes = ['1h', '4h', '1d']
    multi_tf_data = {}
    
    for tf in timeframes:
        if tf == '1h':
            multi_tf_data[tf] = sample_data
        else:
            # Resample data for higher timeframes
            resampled = sample_data.resample(tf).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            multi_tf_data[tf] = resampled
    
    # Test indicator calculation on multiple timeframes
    for tf, data in multi_tf_data.items():
        sma = await backtest_engine.indicators.calculate_sma(data['close'], window=20)
        rsi = await backtest_engine.indicators.calculate_rsi(data['close'], period=14)
        
        assert isinstance(sma, pd.Series)
        assert isinstance(rsi, pd.Series)
        assert len(sma) == len(data)
        assert len(rsi) == len(data) 