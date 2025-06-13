"""
Email & push notifications on thresholds or bot stops.
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from src.config import Config
import json
import os
import time
from datetime import datetime

class NotificationManager:
    def __init__(self):
        self.config = Config.load_config()
        self.email_config = self._load_email_config()
        self.webhook_url = self.config.get('webhook_url')
    
    def _load_email_config(self):
        """Load email configuration from environment variables"""
        return {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', 587)),
            'smtp_username': os.getenv('SMTP_USERNAME'),
            'smtp_password': os.getenv('SMTP_PASSWORD'),
            'from_email': os.getenv('FROM_EMAIL'),
            'to_email': os.getenv('TO_EMAIL')
        }
    
    def send_notification(self, subject, message, level='info'):
        """
        Send notification through all configured channels
        
        Args:
            subject (str): Notification subject
            message (str): Notification message
            level (str): Notification level ('info', 'warning', 'error')
        """
        # Send email notification
        if all(self.email_config.values()):
            self._send_email(subject, message)
        
        # Send webhook notification
        if self.webhook_url:
            self._send_webhook(subject, message, level)
    
    def _send_email(self, subject, message):
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = self.email_config['to_email']
            msg['Subject'] = f"[Trading Bot] {subject}"
            
            msg.attach(MIMEText(message, 'plain'))
            
            with smtplib.SMTP(
                self.email_config['smtp_server'],
                self.email_config['smtp_port']
            ) as server:
                server.starttls()
                server.login(
                    self.email_config['smtp_username'],
                    self.email_config['smtp_password']
                )
                server.send_message(msg)
                
        except Exception as e:
            print(f"Error sending email notification: {e}")
    
    def _send_webhook(self, subject, message, level):
        """Send webhook notification"""
        try:
            payload = {
                'subject': subject,
                'message': message,
                'level': level,
                'timestamp': int(time.time() * 1000)
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code != 200:
                print(f"Webhook notification failed: {response.text}")
                
        except Exception as e:
            print(f"Error sending webhook notification: {e}")
    
    def send_trade_notification(self, trade):
        """Send notification for trade execution"""
        subject = f"Trade {trade['side']} - {trade['symbol']}"
        message = (
            f"Symbol: {trade['symbol']}\n"
            f"Side: {trade['side']}\n"
            f"Price: {trade['price']}\n"
            f"Quantity: {trade['quantity']}\n"
            f"Status: {trade['status']}\n"
            f"Time: {datetime.fromtimestamp(trade['time']/1000)}"
        )
        
        if 'pnl' in trade:
            message += f"\nP&L: {trade['pnl']:.2f}"
        
        self.send_notification(subject, message)
    
    def send_error_notification(self, error):
        """Send notification for errors"""
        subject = "Trading Bot Error"
        message = f"An error occurred:\n{str(error)}"
        self.send_notification(subject, message, level='error')
    
    def send_balance_notification(self, balance):
        """Send notification for balance updates"""
        subject = "Balance Update"
        message = "Current balance:\n"
        for asset, amount in balance.items():
            message += f"{asset}: {amount['total']:.8f}\n"
        
        self.send_notification(subject, message)

def notify_email(subject, body, to_addr):
    # TODO: integrate SMTP or service
    pass

def notify_push(title, message):
    # TODO: integrate push service
    pass

