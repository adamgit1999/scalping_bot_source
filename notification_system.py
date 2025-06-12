from flask_mail import Mail, Message
from flask import current_app
import logging
from typing import Dict, List, Optional
from datetime import datetime
import jinja2
import os

class NotificationSystem:
    def __init__(self, app):
        self.mail = Mail(app)
        self.logger = logging.getLogger(__name__)
        self.template_loader = jinja2.FileSystemLoader('templates/email')
        self.template_env = jinja2.Environment(loader=self.template_loader)

    def send_trade_notification(self, user_email: str, trade_data: Dict):
        """Send a notification about a new trade."""
        try:
            template = self.template_env.get_template('trade_notification.html')
            html_content = template.render(
                trade=trade_data,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            msg = Message(
                subject=f"New Trade: {trade_data['symbol']} {trade_data['type'].upper()}",
                sender=current_app.config['MAIL_DEFAULT_SENDER'],
                recipients=[user_email],
                html=html_content
            )
            
            self.mail.send(msg)
            self.logger.info(f"Trade notification sent to {user_email}")
            
        except Exception as e:
            self.logger.error(f"Failed to send trade notification: {str(e)}")
            raise

    def send_error_notification(self, user_email: str, error_data: Dict):
        """Send a notification about an error."""
        try:
            template = self.template_env.get_template('error_notification.html')
            html_content = template.render(
                error=error_data,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            msg = Message(
                subject="Trading Bot Error Alert",
                sender=current_app.config['MAIL_DEFAULT_SENDER'],
                recipients=[user_email],
                html=html_content
            )
            
            self.mail.send(msg)
            self.logger.info(f"Error notification sent to {user_email}")
            
        except Exception as e:
            self.logger.error(f"Failed to send error notification: {str(e)}")
            raise

    def send_daily_report(self, user_email: str, performance_data: Dict):
        """Send a daily performance report."""
        try:
            template = self.template_env.get_template('daily_report.html')
            html_content = template.render(
                performance=performance_data,
                date=datetime.now().strftime('%Y-%m-%d')
            )
            
            msg = Message(
                subject="Daily Trading Performance Report",
                sender=current_app.config['MAIL_DEFAULT_SENDER'],
                recipients=[user_email],
                html=html_content
            )
            
            self.mail.send(msg)
            self.logger.info(f"Daily report sent to {user_email}")
            
        except Exception as e:
            self.logger.error(f"Failed to send daily report: {str(e)}")
            raise

    def send_backtest_report(self, user_email: str, backtest_data: Dict):
        """Send a backtest results report."""
        try:
            template = self.template_env.get_template('backtest_report.html')
            html_content = template.render(
                backtest=backtest_data,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            msg = Message(
                subject="Backtest Results Report",
                sender=current_app.config['MAIL_DEFAULT_SENDER'],
                recipients=[user_email],
                html=html_content
            )
            
            self.mail.send(msg)
            self.logger.info(f"Backtest report sent to {user_email}")
            
        except Exception as e:
            self.logger.error(f"Failed to send backtest report: {str(e)}")
            raise

    def send_welcome_email(self, user_email: str, username: str):
        """Send a welcome email to new users."""
        try:
            template = self.template_env.get_template('welcome.html')
            html_content = template.render(
                username=username,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            msg = Message(
                subject="Welcome to Scalping Bot",
                sender=current_app.config['MAIL_DEFAULT_SENDER'],
                recipients=[user_email],
                html=html_content
            )
            
            self.mail.send(msg)
            self.logger.info(f"Welcome email sent to {user_email}")
            
        except Exception as e:
            self.logger.error(f"Failed to send welcome email: {str(e)}")
            raise

    def send_password_reset(self, user_email: str, reset_token: str):
        """Send a password reset email."""
        try:
            template = self.template_env.get_template('password_reset.html')
            reset_url = f"{current_app.config['BASE_URL']}/reset-password/{reset_token}"
            html_content = template.render(
                reset_url=reset_url,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            msg = Message(
                subject="Password Reset Request",
                sender=current_app.config['MAIL_DEFAULT_SENDER'],
                recipients=[user_email],
                html=html_content
            )
            
            self.mail.send(msg)
            self.logger.info(f"Password reset email sent to {user_email}")
            
        except Exception as e:
            self.logger.error(f"Failed to send password reset email: {str(e)}")
            raise 