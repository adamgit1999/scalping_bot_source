"""
Main trading engine entrypoint.
Loads strategy plugins and executes them on incoming candles.
"""
from plugins import load_plugins
import threading
import time
from datetime import datetime
import pandas as pd
import numpy as np
from plugins.default_strategy import DefaultStrategy
from notifications import NotificationManager
import json

def start_trading():
    plugins = load_plugins()
    # TODO: wire up broker.fetch_candles, strategy decisions, order placement
    print("Trading engine started with plugins:", plugins)

class TradingBot:
    def __init__(self, broker, config):
        self.broker = broker
        self.config = config
        self.is_running = False
        self.thread = None
        self.trades = []
        self.strategy = DefaultStrategy()
        self.notification_manager = NotificationManager()
        
        # Load settings
        self.symbol = config['symbol']
        self.interval = config['interval']
        self.position_size = config['position_size']
        self.mock_mode = config['mock_mode']
        self.auto_withdraw = config['auto_withdraw']
        
        # Initialize state
        self.current_position = None
        self.last_candle = None
        self.equity_curve = []
    
    def start(self):
        """Start the trading bot"""
        if self.is_running:
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        
        self.notification_manager.send_notification(
            "Bot started",
            f"Trading {self.symbol} on {self.broker.__class__.__name__}"
        )
    
    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        if self.thread:
            self.thread.join()
            
        self.notification_manager.send_notification(
            "Bot stopped",
            f"Final balance: {self.get_balance()}"
        )
    
    def _run(self):
        """Main trading loop"""
        while self.is_running:
            try:
                # Fetch latest candles
                candles = self.broker.fetch_candles(
                    self.symbol,
                    self.interval,
                    limit=100
                )
                
                if not candles:
                    continue
                
                # Update last candle
                self.last_candle = candles[-1]
                
                # Convert to DataFrame for strategy
                df = pd.DataFrame(candles)
                df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Get trading signal
                signal = self.strategy.generate_signal(df)
                
                # Execute trades based on signal
                if signal == 'BUY' and not self.current_position:
                    self._open_position()
                elif signal == 'SELL' and self.current_position:
                    self._close_position()
                
                # Update equity curve
                self._update_equity()
                
                # Check auto-withdraw
                self._check_auto_withdraw()
                
                # Sleep until next candle
                time.sleep(60)  # Adjust based on interval
                
            except Exception as e:
                print(f"Error in trading loop: {e}")
                self.notification_manager.send_notification(
                    "Error",
                    f"Trading loop error: {str(e)}"
                )
                time.sleep(60)
    
    def _open_position(self):
        """Open a new trading position"""
        try:
            if self.mock_mode:
                # Simulate order in mock mode
                order = {
                    'id': f"mock_{len(self.trades)}",
                    'symbol': self.symbol,
                    'side': 'BUY',
                    'type': 'MARKET',
                    'price': self.last_candle['close'],
                    'quantity': self.position_size,
                    'status': 'FILLED',
                    'time': int(datetime.now().timestamp() * 1000)
                }
            else:
                # Place real order
                order = self.broker.place_order(
                    self.symbol,
                    'BUY',
                    self.position_size
                )
            
            if order:
                self.current_position = order
                self.trades.append(order)
                
                self.notification_manager.send_notification(
                    "Position opened",
                    f"Bought {self.position_size} {self.symbol} at {order['price']}"
                )
                
        except Exception as e:
            print(f"Error opening position: {e}")
            self.notification_manager.send_notification(
                "Error",
                f"Failed to open position: {str(e)}"
            )
    
    def _close_position(self):
        """Close the current trading position"""
        try:
            if self.mock_mode:
                # Simulate order in mock mode
                order = {
                    'id': f"mock_{len(self.trades)}",
                    'symbol': self.symbol,
                    'side': 'SELL',
                    'type': 'MARKET',
                    'price': self.last_candle['close'],
                    'quantity': self.position_size,
                    'status': 'FILLED',
                    'time': int(datetime.now().timestamp() * 1000)
                }
            else:
                # Place real order
                order = self.broker.place_order(
                    self.symbol,
                    'SELL',
                    self.position_size
                )
            
            if order:
                # Calculate P&L
                entry_price = self.current_position['price']
                exit_price = order['price']
                pnl = (exit_price - entry_price) * self.position_size
                
                # Update trade record
                order['pnl'] = pnl
                self.trades.append(order)
                
                # Clear current position
                self.current_position = None
                
                self.notification_manager.send_notification(
                    "Position closed",
                    f"Sold {self.position_size} {self.symbol} at {order['price']} (P&L: {pnl:.2f})"
                )
                
        except Exception as e:
            print(f"Error closing position: {e}")
            self.notification_manager.send_notification(
                "Error",
                f"Failed to close position: {str(e)}"
            )
    
    def _update_equity(self):
        """Update the equity curve"""
        balance = self.get_balance()
        total_equity = sum(asset['total'] for asset in balance.values())
        
        self.equity_curve.append({
            'timestamp': int(datetime.now().timestamp() * 1000),
            'equity': total_equity
        })
    
    def _check_auto_withdraw(self):
        """Check if auto-withdraw threshold is reached"""
        if not self.auto_withdraw:
            return
            
        balance = self.get_balance()
        total_equity = sum(asset['total'] for asset in balance.values())
        
        if total_equity >= self.auto_withdraw:
            self.notification_manager.send_notification(
                "Auto-withdraw threshold reached",
                f"Current balance: {total_equity}"
            )
            # Implement withdrawal logic here
    
    def get_trades(self):
        """Get list of executed trades"""
        return self.trades
    
    def get_balance(self):
        """Get current account balance"""
        return self.broker.get_balance()
    
    def get_equity_curve(self):
        """Get the equity curve data"""
        return self.equity_curve

