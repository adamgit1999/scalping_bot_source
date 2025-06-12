import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from backtest_engine import BacktestEngine, BacktestResult
from backtest.exceptions import BacktestError, ValidationError, ExecutionError
from unittest.mock import patch

@pytest.fixture
def backtest_engine():
    """Create a backtest engine instance."""
    return BacktestEngine()

@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H', tz=timezone.utc)
    data = pd.DataFrame({
        'open': np.random.normal(100, 1, len(dates)),
        'high': np.random.normal(101, 1, len(dates)),
        'low': np.random.normal(99, 1, len(dates)),
        'close': np.random.normal(100, 1, len(dates)),
        'volume': np.random.normal(1000, 100, len(dates)),
        'bid': np.random.normal(99.5, 0.5, len(dates)),
        'ask': np.random.normal(100.5, 0.5, len(dates))
    }, index=dates)
    return data

@pytest.fixture
def mock_strategy():
    """Create a mock trading strategy."""
    class MockStrategy:
        def __init__(self):
            self.signals = []
            self.parameters = {}
        
        def generate_signals(self, data, parameters):
            self.parameters = parameters
            return pd.Series([1, -1, 0, 1, -1], index=data.index[:5])
    
    return MockStrategy()

def test_initialize(backtest_engine):
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

def test_initialize_with_custom_parameters():
    """Test initialization with custom parameters."""
    params = {
        'initial_balance': Decimal('50000'),
        'commission': Decimal('0.002'),
        'max_position_size': Decimal('2.0'),
        'min_position_size': Decimal('0.01'),
        'slippage': Decimal('0.002'),
        'risk_free_rate': Decimal('0.03'),
        'trading_fee': Decimal('0.002'),
        'funding_rate': Decimal('0.0002')
    }
    
    engine = BacktestEngine(**params)
    
    for key, value in params.items():
        assert getattr(engine, key) == value

def test_run_backtest(backtest_engine, sample_data, mock_strategy):
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
    
    with patch('backtest_engine.get_strategy', return_value=mock_strategy):
        results = backtest_engine.run_backtest(sample_data, strategy, parameters)
    
    assert isinstance(results, BacktestResult)
    assert results.total_return is not None
    assert results.sharpe_ratio is not None
    assert results.max_drawdown is not None
    assert results.win_rate is not None
    assert isinstance(results.trades, list)
    assert isinstance(results.equity_curve, list)
    assert results.total_commission is not None
    assert results.total_trades > 0
    assert results.winning_trades >= 0
    assert results.losing_trades >= 0
    assert results.average_win is not None
    assert results.average_loss is not None
    assert results.profit_factor is not None
    assert results.max_consecutive_wins is not None
    assert results.max_consecutive_losses is not None
    assert results.annualized_return is not None
    assert results.annualized_volatility is not None
    assert results.sortino_ratio is not None
    assert results.calmar_ratio is not None
    assert results.max_leverage is not None
    assert results.average_leverage is not None
    assert results.funding_payments is not None
    assert results.trading_fees is not None

def test_run_backtest_with_invalid_data(backtest_engine):
    """Test running backtest with invalid data."""
    strategy = 'scalping'
    parameters = {
        'sma_short': 10,
        'sma_long': 20,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70
    }
    
    # Test empty dataframe
    with pytest.raises(ValidationError, match="Empty data provided"):
        backtest_engine.run_backtest(pd.DataFrame(), strategy, parameters)
    
    # Test missing required columns
    invalid_data = pd.DataFrame({'open': [1, 2, 3]})
    with pytest.raises(ValidationError, match="Missing required columns"):
        backtest_engine.run_backtest(invalid_data, strategy, parameters)
    
    # Test invalid data types
    invalid_data = pd.DataFrame({
        'open': ['1', '2', '3'],
        'high': ['2', '3', '4'],
        'low': ['0', '1', '2'],
        'close': ['1', '2', '3'],
        'volume': ['100', '200', '300']
    })
    with pytest.raises(ValidationError, match="Invalid data types"):
        backtest_engine.run_backtest(invalid_data, strategy, parameters)

def test_run_backtest_with_invalid_strategy(backtest_engine, sample_data):
    """Test running backtest with invalid strategy."""
    strategy = 'invalid_strategy'
    parameters = {
        'sma_short': 10,
        'sma_long': 20,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70
    }
    
    with pytest.raises(ValidationError, match="Invalid strategy"):
        backtest_engine.run_backtest(sample_data, strategy, parameters)

def test_calculate_metrics(backtest_engine):
    """Test calculating performance metrics."""
    # Create sample equity curve and trades
    backtest_engine.equity_curve = [
        Decimal('10000'), Decimal('10100'), Decimal('10200'),
        Decimal('10150'), Decimal('10300'), Decimal('10200'),
        Decimal('10100'), Decimal('10250')
    ]
    backtest_engine.trades = [
        {'profit': Decimal('100'), 'leverage': Decimal('1.0')},
        {'profit': Decimal('-50'), 'leverage': Decimal('1.0')},
        {'profit': Decimal('200'), 'leverage': Decimal('1.5')},
        {'profit': Decimal('-100'), 'leverage': Decimal('1.0')},
        {'profit': Decimal('150'), 'leverage': Decimal('2.0')}
    ]
    backtest_engine.total_commission = Decimal('50')
    backtest_engine.total_trades = 5
    backtest_engine.winning_trades = 3
    backtest_engine.losing_trades = 2
    backtest_engine.funding_payments = Decimal('10')
    backtest_engine.trading_fees = Decimal('40')
    
    metrics = backtest_engine._calculate_metrics()
    
    assert isinstance(metrics, dict)
    assert metrics['total_return'] > 0
    assert metrics['sharpe_ratio'] is not None
    assert 0 <= metrics['max_drawdown'] <= 1
    assert 0 <= metrics['win_rate'] <= 1
    assert metrics['profit_factor'] > 0
    assert metrics['average_win'] > 0
    assert metrics['average_loss'] < 0
    assert metrics['max_consecutive_wins'] > 0
    assert metrics['max_consecutive_losses'] > 0
    assert metrics['annualized_return'] is not None
    assert metrics['annualized_volatility'] is not None
    assert metrics['sortino_ratio'] is not None
    assert metrics['calmar_ratio'] is not None
    assert metrics['max_leverage'] == Decimal('2.0')
    assert metrics['average_leverage'] > 0
    assert metrics['funding_payments'] == Decimal('10')
    assert metrics['trading_fees'] == Decimal('40')

def test_calculate_metrics_with_no_trades(backtest_engine):
    """Test calculating metrics with no trades."""
    backtest_engine.equity_curve = [Decimal('10000')] * 10
    backtest_engine.trades = []
    backtest_engine.funding_payments = Decimal('0')
    backtest_engine.trading_fees = Decimal('0')
    
    metrics = backtest_engine._calculate_metrics()
    
    assert metrics['total_return'] == 0
    assert metrics['sharpe_ratio'] == 0
    assert metrics['max_drawdown'] == 0
    assert metrics['win_rate'] == 0
    assert metrics['profit_factor'] == 0
    assert metrics['average_win'] == 0
    assert metrics['average_loss'] == 0
    assert metrics['max_consecutive_wins'] == 0
    assert metrics['max_consecutive_losses'] == 0
    assert metrics['annualized_return'] == 0
    assert metrics['annualized_volatility'] == 0
    assert metrics['sortino_ratio'] == 0
    assert metrics['calmar_ratio'] == 0
    assert metrics['max_leverage'] == 0
    assert metrics['average_leverage'] == 0
    assert metrics['funding_payments'] == Decimal('0')
    assert metrics['trading_fees'] == Decimal('0')

def test_calculate_sharpe_ratio(backtest_engine):
    """Test Sharpe ratio calculation."""
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
    risk_free_rate = Decimal('0.02')
    
    sharpe = backtest_engine._calculate_sharpe_ratio(returns, risk_free_rate)
    
    assert isinstance(sharpe, float)
    assert sharpe is not None

def test_calculate_sharpe_ratio_with_constant_returns(backtest_engine):
    """Test Sharpe ratio calculation with constant returns."""
    returns = pd.Series([0.01] * 10)
    risk_free_rate = Decimal('0.02')
    
    sharpe = backtest_engine._calculate_sharpe_ratio(returns, risk_free_rate)
    
    assert sharpe == 0  # Sharpe ratio should be 0 for constant returns

def test_calculate_max_drawdown(backtest_engine):
    """Test maximum drawdown calculation."""
    equity_curve = pd.Series([
        Decimal('10000'), Decimal('10100'), Decimal('10200'),
        Decimal('10150'), Decimal('10300'), Decimal('10200'),
        Decimal('10100'), Decimal('10250')
    ])
    
    max_dd = backtest_engine._calculate_max_drawdown(equity_curve)
    
    assert isinstance(max_dd, float)
    assert 0 <= max_dd <= 1

def test_calculate_max_drawdown_with_constant_equity(backtest_engine):
    """Test maximum drawdown calculation with constant equity."""
    equity_curve = pd.Series([Decimal('10000')] * 10)
    
    max_dd = backtest_engine._calculate_max_drawdown(equity_curve)
    
    assert max_dd == 0  # No drawdown for constant equity

def test_calculate_win_rate(backtest_engine):
    """Test win rate calculation."""
    trades = [
        {'profit': Decimal('100')},
        {'profit': Decimal('-50')},
        {'profit': Decimal('200')},
        {'profit': Decimal('-100')},
        {'profit': Decimal('150')}
    ]
    
    win_rate = backtest_engine._calculate_win_rate(trades)
    
    assert isinstance(win_rate, float)
    assert 0 <= win_rate <= 1

def test_calculate_win_rate_with_no_trades(backtest_engine):
    """Test win rate calculation with no trades."""
    trades = []
    
    win_rate = backtest_engine._calculate_win_rate(trades)
    
    assert win_rate == 0

def test_execute_trade(backtest_engine):
    """Test trade execution."""
    timestamp = datetime.now(timezone.utc)
    symbol = 'BTC/USDT'
    side = 'buy'
    price = Decimal('50000')
    amount = Decimal('0.1')
    
    trade = backtest_engine._execute_trade(timestamp, symbol, side, price, amount)
    
    assert isinstance(trade, dict)
    assert trade['timestamp'] == timestamp
    assert trade['symbol'] == symbol
    assert trade['side'] == side
    assert trade['price'] == price
    assert trade['amount'] == amount
    assert 'commission' in trade
    assert 'total' in trade
    assert trade['commission'] == price * amount * backtest_engine.commission
    assert trade['total'] == price * amount + trade['commission']
    assert 'leverage' in trade
    assert 'funding_payment' in trade
    assert 'trading_fee' in trade

def test_execute_trade_with_zero_amount(backtest_engine):
    """Test trade execution with zero amount."""
    timestamp = datetime.now(timezone.utc)
    symbol = 'BTC/USDT'
    side = 'buy'
    price = Decimal('50000')
    amount = Decimal('0')
    
    with pytest.raises(ExecutionError, match="Invalid trade amount"):
        backtest_engine._execute_trade(timestamp, symbol, side, price, amount)

def test_execute_trade_with_invalid_side(backtest_engine):
    """Test trade execution with invalid side."""
    timestamp = datetime.now(timezone.utc)
    symbol = 'BTC/USDT'
    side = 'invalid'
    price = Decimal('50000')
    amount = Decimal('0.1')
    
    with pytest.raises(ExecutionError, match="Invalid trade side"):
        backtest_engine._execute_trade(timestamp, symbol, side, price, amount)

def test_execute_trade_with_insufficient_balance(backtest_engine):
    """Test trade execution with insufficient balance."""
    timestamp = datetime.now(timezone.utc)
    symbol = 'BTC/USDT'
    side = 'buy'
    price = Decimal('50000')
    amount = Decimal('1.0')  # Would require 50000 + commission
    
    with pytest.raises(ExecutionError, match="Insufficient balance"):
        backtest_engine._execute_trade(timestamp, symbol, side, price, amount)

def test_update_position(backtest_engine):
    """Test position update."""
    symbol = 'BTC/USDT'
    amount = Decimal('0.1')
    price = Decimal('50000')
    
    # Test new position
    backtest_engine._update_position(symbol, amount, price)
    assert symbol in backtest_engine.positions
    assert backtest_engine.positions[symbol]['amount'] == amount
    assert backtest_engine.positions[symbol]['price'] == price
    
    # Test position increase
    backtest_engine._update_position(symbol, Decimal('0.1'), Decimal('51000'))
    assert backtest_engine.positions[symbol]['amount'] == Decimal('0.2')
    assert backtest_engine.positions[symbol]['price'] == Decimal('50500')  # Average price
    
    # Test position decrease
    backtest_engine._update_position(symbol, Decimal('-0.1'), Decimal('52000'))
    assert backtest_engine.positions[symbol]['amount'] == Decimal('0.1')
    assert backtest_engine.positions[symbol]['price'] == Decimal('50500')  # Price unchanged

def test_update_position_with_zero_amount(backtest_engine):
    """Test position update with zero amount."""
    symbol = 'BTC/USDT'
    amount = Decimal('0')
    price = Decimal('50000')
    
    with pytest.raises(ExecutionError, match="Invalid position amount"):
        backtest_engine._update_position(symbol, amount, price)

def test_calculate_commission(backtest_engine):
    """Test commission calculation."""
    price = Decimal('50000')
    amount = Decimal('0.1')
    
    commission = backtest_engine._calculate_commission(price, amount)
    
    assert commission == price * amount * backtest_engine.commission
    assert isinstance(commission, Decimal)

def test_calculate_commission_with_zero_amount(backtest_engine):
    """Test commission calculation with zero amount."""
    price = Decimal('50000')
    amount = Decimal('0')
    
    with pytest.raises(ExecutionError, match="Invalid amount for commission calculation"):
        backtest_engine._calculate_commission(price, amount)

def test_validate_parameters(backtest_engine):
    """Test parameter validation."""
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
    
    backtest_engine._validate_parameters(strategy, parameters)
    
    # Test invalid parameter types
    invalid_parameters = parameters.copy()
    invalid_parameters['sma_short'] = '10'
    with pytest.raises(ValidationError, match="Invalid parameter type"):
        backtest_engine._validate_parameters(strategy, invalid_parameters)
    
    # Test parameter range validation
    invalid_parameters = parameters.copy()
    invalid_parameters['rsi_oversold'] = 100
    with pytest.raises(ValidationError, match="Invalid parameter range"):
        backtest_engine._validate_parameters(strategy, invalid_parameters)

def test_validate_parameters_with_missing_required(backtest_engine):
    """Test parameter validation with missing required parameters."""
    strategy = 'scalping'
    parameters = {
        'sma_short': 10,
        'sma_long': 20
    }
    
    with pytest.raises(ValidationError, match="Missing required parameters"):
        backtest_engine._validate_parameters(strategy, parameters)

def test_validate_parameters_with_invalid_strategy(backtest_engine):
    """Test parameter validation with invalid strategy."""
    strategy = 'invalid_strategy'
    parameters = {
        'sma_short': 10,
        'sma_long': 20,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70
    }
    
    with pytest.raises(ValidationError, match="Invalid strategy"):
        backtest_engine._validate_parameters(strategy, parameters)

def test_calculate_profit_factor(backtest_engine):
    """Test profit factor calculation."""
    trades = [
        {'profit': Decimal('100')},
        {'profit': Decimal('-50')},
        {'profit': Decimal('200')},
        {'profit': Decimal('-100')},
        {'profit': Decimal('150')}
    ]
    
    profit_factor = backtest_engine._calculate_profit_factor(trades)
    
    assert isinstance(profit_factor, float)
    assert profit_factor > 0

def test_calculate_profit_factor_with_no_trades(backtest_engine):
    """Test profit factor calculation with no trades."""
    trades = []
    
    profit_factor = backtest_engine._calculate_profit_factor(trades)
    
    assert profit_factor == 0

def test_calculate_average_win_loss(backtest_engine):
    """Test average win/loss calculation."""
    trades = [
        {'profit': Decimal('100')},
        {'profit': Decimal('-50')},
        {'profit': Decimal('200')},
        {'profit': Decimal('-100')},
        {'profit': Decimal('150')}
    ]
    
    avg_win, avg_loss = backtest_engine._calculate_average_win_loss(trades)
    
    assert isinstance(avg_win, float)
    assert isinstance(avg_loss, float)
    assert avg_win > 0
    assert avg_loss < 0

def test_calculate_consecutive_wins_losses(backtest_engine):
    """Test consecutive wins/losses calculation."""
    trades = [
        {'profit': Decimal('100')},
        {'profit': Decimal('200')},
        {'profit': Decimal('-50')},
        {'profit': Decimal('150')},
        {'profit': Decimal('-100')},
        {'profit': Decimal('-200')}
    ]
    
    max_wins, max_losses = backtest_engine._calculate_consecutive_wins_losses(trades)
    
    assert isinstance(max_wins, int)
    assert isinstance(max_losses, int)
    assert max_wins > 0
    assert max_losses > 0 