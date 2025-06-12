import pytest
from app import app, db, User, Notification
from notification_system import NotificationSystem
from datetime import datetime, timedelta
import json
from unittest.mock import Mock, patch

@pytest.fixture
def app_context():
    """Create app context and initialize test database."""
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['MAIL_SERVER'] = 'smtp.test.com'
    app.config['MAIL_PORT'] = 587
    app.config['MAIL_USERNAME'] = 'test@example.com'
    app.config['MAIL_PASSWORD'] = 'test_password'
    app.config['MAIL_DEFAULT_SENDER'] = 'test@example.com'
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

@pytest.fixture
def test_user(app_context):
    """Create a test user."""
    user = User(
        username='testuser',
        email='test@example.com',
        password_hash='hashed_password'
    )
    db.session.add(user)
    db.session.commit()
    return user

@pytest.fixture
def notification_system(app_context):
    """Create a notification system instance."""
    return NotificationSystem()

def test_notification_initialization(notification_system):
    """Test notification system initialization."""
    assert notification_system.mail is not None
    assert notification_system.app is not None
    assert notification_system.template_dir is not None
    assert notification_system.default_sender is not None
    assert notification_system.default_recipient is not None

def test_trade_notification(notification_system, test_user):
    """Test trade notification sending."""
    # Create trade data
    trade = {
        'id': 1,
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'price': 50000.0,
        'amount': 0.1,
        'total': 5000.0,
        'fee': 5.0,
        'profit': 100.0,
        'timestamp': datetime.now(datetime.UTC).isoformat()
    }
    
    # Send trade notification
    with patch('notification_system.NotificationSystem.mail.send') as mock_send:
        result = notification_system.send_trade_notification(test_user.email, trade)
        assert result is True
        mock_send.assert_called_once()
        
        # Verify email content
        message = mock_send.call_args[0][0]
        assert message.subject == f"Trade Executed: {trade['symbol']}"
        assert message.recipients == [test_user.email]
        assert trade['symbol'] in message.body
        assert str(trade['price']) in message.body

def test_error_notification(notification_system, test_user):
    """Test error notification sending."""
    # Create error data
    error = {
        'type': 'execution_error',
        'message': 'Failed to execute trade',
        'timestamp': datetime.now(datetime.UTC).isoformat()
    }
    
    # Send error notification
    with patch('notification_system.NotificationSystem.mail.send') as mock_send:
        result = notification_system.send_error_notification(test_user.email, error)
        assert result is True
        mock_send.assert_called_once()
        
        # Verify email content
        message = mock_send.call_args[0][0]
        assert message.subject == f"Error Alert: {error['type']}"
        assert message.recipients == [test_user.email]
        assert error['message'] in message.body

def test_daily_report(notification_system, test_user):
    """Test daily report sending."""
    # Create report data
    report = {
        'date': datetime.now(datetime.UTC).date().isoformat(),
        'total_trades': 10,
        'win_rate': 0.6,
        'total_profit': 1000.0,
        'max_drawdown': 0.1,
        'recent_trades': [
            {
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'price': 50000.0,
                'profit': 100.0
            }
        ]
    }
    
    # Send daily report
    with patch('notification_system.NotificationSystem.mail.send') as mock_send:
        result = notification_system.send_daily_report(test_user.email, report)
        assert result is True
        mock_send.assert_called_once()
        
        # Verify email content
        message = mock_send.call_args[0][0]
        assert message.subject == f"Daily Trading Report: {report['date']}"
        assert message.recipients == [test_user.email]
        assert str(report['total_trades']) in message.body
        assert str(report['win_rate']) in message.body

def test_balance_notification(notification_system, test_user):
    """Test balance notification sending."""
    # Create balance data
    balance = {
        'total_balance': 10000.0,
        'available_balance': 9000.0,
        'locked_balance': 1000.0,
        'timestamp': datetime.now(datetime.UTC).isoformat()
    }
    
    # Send balance notification
    with patch('notification_system.NotificationSystem.mail.send') as mock_send:
        result = notification_system.send_balance_notification(test_user.email, balance)
        assert result is True
        mock_send.assert_called_once()
        
        # Verify email content
        message = mock_send.call_args[0][0]
        assert message.subject == "Balance Update"
        assert message.recipients == [test_user.email]
        assert str(balance['total_balance']) in message.body

def test_strategy_notification(notification_system, test_user):
    """Test strategy notification sending."""
    # Create strategy data
    strategy = {
        'name': 'Test Strategy',
        'status': 'active',
        'performance': {
            'total_trades': 100,
            'win_rate': 0.6,
            'profit_factor': 1.5
        },
        'timestamp': datetime.now(datetime.UTC).isoformat()
    }
    
    # Send strategy notification
    with patch('notification_system.NotificationSystem.mail.send') as mock_send:
        result = notification_system.send_strategy_notification(test_user.email, strategy)
        assert result is True
        mock_send.assert_called_once()
        
        # Verify email content
        message = mock_send.call_args[0][0]
        assert message.subject == f"Strategy Update: {strategy['name']}"
        assert message.recipients == [test_user.email]
        assert strategy['name'] in message.body
        assert str(strategy['performance']['win_rate']) in message.body

def test_system_notification(notification_system, test_user):
    """Test system notification sending."""
    # Create system data
    system = {
        'status': 'warning',
        'message': 'High memory usage detected',
        'metrics': {
            'memory_usage': 85,
            'cpu_usage': 70
        },
        'timestamp': datetime.now(datetime.UTC).isoformat()
    }
    
    # Send system notification
    with patch('notification_system.NotificationSystem.mail.send') as mock_send:
        result = notification_system.send_system_notification(test_user.email, system)
        assert result is True
        mock_send.assert_called_once()
        
        # Verify email content
        message = mock_send.call_args[0][0]
        assert message.subject == f"System Alert: {system['status']}"
        assert message.recipients == [test_user.email]
        assert system['message'] in message.body

def test_welcome_notification(notification_system, test_user):
    """Test welcome notification sending."""
    # Send welcome notification
    with patch('notification_system.NotificationSystem.mail.send') as mock_send:
        result = notification_system.send_welcome_notification(test_user.email)
        assert result is True
        mock_send.assert_called_once()
        
        # Verify email content
        message = mock_send.call_args[0][0]
        assert message.subject == "Welcome to Trading Bot"
        assert message.recipients == [test_user.email]
        assert test_user.username in message.body

def test_password_reset_notification(notification_system, test_user):
    """Test password reset notification sending."""
    # Create reset data
    reset_data = {
        'reset_token': 'test_token',
        'expires_at': (datetime.now(datetime.UTC) + timedelta(hours=1)).isoformat()
    }
    
    # Send password reset notification
    with patch('notification_system.NotificationSystem.mail.send') as mock_send:
        result = notification_system.send_password_reset_notification(test_user.email, reset_data)
        assert result is True
        mock_send.assert_called_once()
        
        # Verify email content
        message = mock_send.call_args[0][0]
        assert message.subject == "Password Reset Request"
        assert message.recipients == [test_user.email]
        assert reset_data['reset_token'] in message.body

def test_verification_notification(notification_system, test_user):
    """Test email verification notification sending."""
    # Create verification data
    verification_data = {
        'verification_token': 'test_token',
        'expires_at': (datetime.now(datetime.UTC) + timedelta(hours=24)).isoformat()
    }
    
    # Send verification notification
    with patch('notification_system.NotificationSystem.mail.send') as mock_send:
        result = notification_system.send_verification_notification(test_user.email, verification_data)
        assert result is True
        mock_send.assert_called_once()
        
        # Verify email content
        message = mock_send.call_args[0][0]
        assert message.subject == "Verify Your Email"
        assert message.recipients == [test_user.email]
        assert verification_data['verification_token'] in message.body

def test_backtest_notification(notification_system, test_user):
    """Test backtest notification sending."""
    # Create backtest data
    backtest = {
        'strategy_name': 'Test Strategy',
        'start_date': (datetime.now(datetime.UTC) - timedelta(days=30)).isoformat(),
        'end_date': datetime.now(datetime.UTC).isoformat(),
        'results': {
            'total_trades': 100,
            'win_rate': 0.6,
            'profit_factor': 1.5,
            'max_drawdown': 0.1
        }
    }
    
    # Send backtest notification
    with patch('notification_system.NotificationSystem.mail.send') as mock_send:
        result = notification_system.send_backtest_notification(test_user.email, backtest)
        assert result is True
        mock_send.assert_called_once()
        
        # Verify email content
        message = mock_send.call_args[0][0]
        assert message.subject == f"Backtest Results: {backtest['strategy_name']}"
        assert message.recipients == [test_user.email]
        assert str(backtest['results']['win_rate']) in message.body

def test_performance_notification(notification_system, test_user):
    """Test performance notification sending."""
    # Create performance data
    performance = {
        'period': 'monthly',
        'total_profit': 5000.0,
        'win_rate': 0.65,
        'profit_factor': 1.8,
        'max_drawdown': 0.15,
        'sharpe_ratio': 2.5
    }
    
    # Send performance notification
    with patch('notification_system.NotificationSystem.mail.send') as mock_send:
        result = notification_system.send_performance_notification(test_user.email, performance)
        assert result is True
        mock_send.assert_called_once()
        
        # Verify email content
        message = mock_send.call_args[0][0]
        assert message.subject == f"Performance Report: {performance['period']}"
        assert message.recipients == [test_user.email]
        assert str(performance['total_profit']) in message.body
        assert str(performance['sharpe_ratio']) in message.body

def test_market_notification(notification_system, test_user):
    """Test market notification sending."""
    # Create market data
    market = {
        'symbol': 'BTC/USDT',
        'price': 50000.0,
        'change_24h': 0.05,
        'volume_24h': 1000000.0,
        'timestamp': datetime.now(datetime.UTC).isoformat()
    }
    
    # Send market notification
    with patch('notification_system.NotificationSystem.mail.send') as mock_send:
        result = notification_system.send_market_notification(test_user.email, market)
        assert result is True
        mock_send.assert_called_once()
        
        # Verify email content
        message = mock_send.call_args[0][0]
        assert message.subject == f"Market Update: {market['symbol']}"
        assert message.recipients == [test_user.email]
        assert str(market['price']) in message.body
        assert str(market['change_24h']) in message.body

def test_risk_notification(notification_system, test_user):
    """Test risk notification sending."""
    # Create risk data
    risk = {
        'level': 'high',
        'message': 'High volatility detected',
        'metrics': {
            'volatility': 0.05,
            'var_95': 0.02,
            'max_drawdown': 0.1
        },
        'timestamp': datetime.now(datetime.UTC).isoformat()
    }
    
    # Send risk notification
    with patch('notification_system.NotificationSystem.mail.send') as mock_send:
        result = notification_system.send_risk_notification(test_user.email, risk)
        assert result is True
        mock_send.assert_called_once()
        
        # Verify email content
        message = mock_send.call_args[0][0]
        assert message.subject == f"Risk Alert: {risk['level']}"
        assert message.recipients == [test_user.email]
        assert risk['message'] in message.body
        assert str(risk['metrics']['volatility']) in message.body

def test_notification_persistence(app_context, test_user):
    """Test notification persistence in database."""
    # Create notification
    notification = Notification(
        user_id=test_user.id,
        type='trade',
        message='Trade executed successfully',
        data={
            'trade_id': 1,
            'symbol': 'BTC/USDT',
            'price': 50000.0
        },
        read=False
    )
    db.session.add(notification)
    db.session.commit()
    
    # Verify notification creation
    assert notification.id is not None
    assert notification.user_id == test_user.id
    assert notification.type == 'trade'
    assert not notification.read
    
    # Update notification
    notification.read = True
    db.session.commit()
    updated_notification = Notification.query.get(notification.id)
    assert updated_notification.read is True
    
    # Delete notification
    db.session.delete(notification)
    db.session.commit()
    assert Notification.query.get(notification.id) is None 