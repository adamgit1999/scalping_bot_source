import pytest
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from src.risk_manager import RiskManager
from src.trading_engine import TradingEngine
from src.notification_system import NotificationSystem
from src.exceptions import RiskLimitExceededError, InsufficientFundsError

@pytest.fixture
def mock_trading_engine():
    """Create a mock trading engine."""
    engine = Mock(spec=TradingEngine)
    engine.get_balance = AsyncMock(return_value=Decimal('10000.0'))
    engine.get_position = Mock(return_value=Decimal('0.0'))
    return engine

@pytest.fixture
def mock_notification_system():
    """Create a mock notification system."""
    system = Mock(spec=NotificationSystem)
    system.send_notification = AsyncMock()
    return system

@pytest.fixture
def risk_manager(mock_trading_engine, mock_notification_system):
    """Create a risk manager instance with mocked dependencies."""
    manager = RiskManager(
        max_position_size=Decimal('0.1'),  # 10% of balance
        max_drawdown=Decimal('0.1'),       # 10% max drawdown
        max_leverage=Decimal('10.0'),      # 10x max leverage
        max_daily_trades=100,              # 100 trades per day
        max_daily_loss=Decimal('0.05'),    # 5% max daily loss
        trading_engine=mock_trading_engine,
        notification_system=mock_notification_system
    )
    return manager

def test_risk_manager_initialization(risk_manager):
    """Test risk manager initialization."""
    assert risk_manager.max_position_size == Decimal('0.1')
    assert risk_manager.max_drawdown == Decimal('0.1')
    assert risk_manager.max_leverage == Decimal('10.0')
    assert risk_manager.max_daily_trades == 100
    assert risk_manager.max_daily_loss == Decimal('0.05')
    assert risk_manager.trading_engine is not None
    assert risk_manager.notification_system is not None

@pytest.mark.asyncio
async def test_check_order_risk(risk_manager):
    """Test order risk checking."""
    # Test valid order
    is_valid, message = await risk_manager.check_order(
        symbol='BTC/USDT',
        side='buy',
        price=Decimal('50000.0'),
        amount=Decimal('0.1')
    )
    assert is_valid
    assert message == ''

    # Test order exceeding position size
    is_valid, message = await risk_manager.check_order(
        symbol='BTC/USDT',
        side='buy',
        price=Decimal('50000.0'),
        amount=Decimal('1.0')
    )
    assert not is_valid
    assert 'position size' in message.lower()

    # Test order exceeding daily loss limit
    risk_manager.daily_loss = Decimal('400.0')  # Set high daily loss
    is_valid, message = await risk_manager.check_order(
        symbol='BTC/USDT',
        side='buy',
        price=Decimal('50000.0'),
        amount=Decimal('0.1')
    )
    assert not is_valid
    assert 'daily loss' in message.lower()

@pytest.mark.asyncio
async def test_update_daily_metrics(risk_manager):
    """Test daily metrics updating."""
    # Update daily loss
    risk_manager.update_daily_loss(Decimal('100.0'))
    assert risk_manager.daily_loss == Decimal('100.0')

    # Update daily trades
    risk_manager.update_daily_trades()
    assert risk_manager.daily_trades == 1

    # Test daily trade limit
    for _ in range(100):
        risk_manager.update_daily_trades()
    
    with pytest.raises(RiskLimitExceededError):
        risk_manager.update_daily_trades()

@pytest.mark.asyncio
async def test_calculate_position_size(risk_manager):
    """Test position size calculation."""
    # Calculate position size for valid order
    size = await risk_manager.calculate_position_size(
        symbol='BTC/USDT',
        side='buy',
        price=Decimal('50000.0')
    )
    assert size <= risk_manager.max_position_size

    # Test with high leverage
    risk_manager.max_leverage = Decimal('20.0')
    size = await risk_manager.calculate_position_size(
        symbol='BTC/USDT',
        side='buy',
        price=Decimal('50000.0')
    )
    assert size <= risk_manager.max_position_size * Decimal('20.0')

@pytest.mark.asyncio
async def test_check_balance(risk_manager):
    """Test balance checking."""
    # Test sufficient balance
    is_valid = await risk_manager.check_balance(
        symbol='BTC/USDT',
        amount=Decimal('0.1'),
        price=Decimal('50000.0')
    )
    assert is_valid

    # Test insufficient balance
    with patch.object(risk_manager.trading_engine, 'get_balance', return_value=Decimal('100.0')):
        is_valid = await risk_manager.check_balance(
            symbol='BTC/USDT',
            amount=Decimal('1.0'),
            price=Decimal('50000.0')
        )
        assert not is_valid

@pytest.mark.asyncio
async def test_risk_notifications(risk_manager):
    """Test risk notifications."""
    # Test position size warning
    await risk_manager.check_order(
        symbol='BTC/USDT',
        side='buy',
        price=Decimal('50000.0'),
        amount=Decimal('0.5')
    )
    assert risk_manager.notification_system.send_notification.called
    call_args = risk_manager.notification_system.send_notification.call_args
    assert 'risk' in call_args[1]['type'].lower()

    # Test daily loss warning
    risk_manager.update_daily_loss(Decimal('400.0'))
    await risk_manager.check_order(
        symbol='BTC/USDT',
        side='buy',
        price=Decimal('50000.0'),
        amount=Decimal('0.1')
    )
    assert risk_manager.notification_system.send_notification.called
    call_args = risk_manager.notification_system.send_notification.call_args
    assert 'loss' in call_args[1]['type'].lower()

@pytest.mark.asyncio
async def test_reset_daily_metrics(risk_manager):
    """Test daily metrics reset."""
    # Set some metrics
    risk_manager.update_daily_loss(Decimal('100.0'))
    risk_manager.update_daily_trades()
    
    # Reset metrics
    risk_manager.reset_daily_metrics()
    
    # Verify reset
    assert risk_manager.daily_loss == Decimal('0.0')
    assert risk_manager.daily_trades == 0

@pytest.mark.asyncio
async def test_position_limits(risk_manager):
    """Test position limits."""
    # Set position limit
    risk_manager.set_position_limit('BTC/USDT', Decimal('0.5'))
    
    # Test within limit
    is_valid, _ = await risk_manager.check_order(
        symbol='BTC/USDT',
        side='buy',
        price=Decimal('50000.0'),
        amount=Decimal('0.4')
    )
    assert is_valid
    
    # Test exceeding limit
    is_valid, message = await risk_manager.check_order(
        symbol='BTC/USDT',
        side='buy',
        price=Decimal('50000.0'),
        amount=Decimal('0.6')
    )
    assert not is_valid
    assert 'position limit' in message.lower()

@pytest.mark.asyncio
async def test_leverage_limits(risk_manager):
    """Test leverage limits."""
    # Test with high leverage
    risk_manager.max_leverage = Decimal('20.0')
    size = await risk_manager.calculate_position_size(
        symbol='BTC/USDT',
        side='buy',
        price=Decimal('50000.0')
    )
    assert size <= risk_manager.max_position_size * Decimal('20.0')
    
    # Test exceeding leverage
    with pytest.raises(RiskLimitExceededError):
        risk_manager.max_leverage = Decimal('30.0')  # Should fail if max allowed is 20x

@pytest.mark.asyncio
async def test_drawdown_monitoring(risk_manager):
    """Test drawdown monitoring."""
    # Set initial balance
    risk_manager.initial_balance = Decimal('10000.0')
    risk_manager.current_balance = Decimal('10000.0')
    
    # Simulate loss
    risk_manager.current_balance = Decimal('9000.0')
    drawdown = risk_manager.calculate_drawdown()
    assert drawdown == Decimal('0.1')  # 10% drawdown
    
    # Test exceeding max drawdown
    risk_manager.current_balance = Decimal('8000.0')
    with pytest.raises(RiskLimitExceededError):
        risk_manager.check_drawdown() 