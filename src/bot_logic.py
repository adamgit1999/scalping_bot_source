"""
Main trading engine entrypoint.
Executes the default scalping strategy on incoming candles.
"""
import threading
import time
from datetime import datetime
import pandas as pd
import numpy as np
from src.notifications import NotificationManager
import json

class TradingBot:
    def __init__(self, broker, config):
        self.broker = broker
        self.config = config
        self.is_running = False
        self.thread = None
        self.trades = []
        self.notification_manager = NotificationManager()
        
        # Load settings
        self.symbol = config['symbol']
        self.interval = config['interval']
        self.position_size = config['position_size']
        self.mock_mode = config['mock_mode']
        self.auto_withdraw = config['auto_withdraw']
        
        # Strategy parameters
        self.fast_period = 10
        self.slow_period = 20
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        
        # Initialize state
        self.current_position = None
        self.last_candle = None
        self.equity_curve = []
        self.last_signal = None
    
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
                candles = self.broker.fetch_candles(self.symbol, self.interval)
                if not candles:
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(candles)
                df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Calculate indicators
                df = self._calculate_indicators(df)
                
                # Get latest values
                current = df.iloc[-1]
                previous = df.iloc[-2]
                
                # Generate signal
                signal = self._generate_signal(df)
                
                # Execute trades based on signal
                if signal == 'BUY' and not self.current_position:
                    self._open_position(current)
                elif signal == 'SELL' and self.current_position:
                    self._close_position(current)
                
                # Update equity curve
                self._update_equity(current)
                
                # Sleep to avoid rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"Error in trading loop: {str(e)}")
                time.sleep(5)
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # Moving averages
        df['fast_ma'] = df['close'].rolling(window=self.fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_period).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['sma20'] + (df['std20'] * 2)
        df['lower_band'] = df['sma20'] - (df['std20'] * 2)
        
        # VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        return df
    
    def _generate_signal(self, df: pd.DataFrame) -> str:
        """Generate trading signal based on technical indicators"""
        # Get latest values
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Moving average crossover
        ma_crossover = (
            previous['fast_ma'] <= previous['slow_ma'] and
            current['fast_ma'] > current['slow_ma']
        )
        
        ma_crossunder = (
            previous['fast_ma'] >= previous['slow_ma'] and
            current['fast_ma'] < current['slow_ma']
        )
        
        # RSI conditions
        rsi_oversold = current['rsi'] < self.rsi_oversold
        rsi_overbought = current['rsi'] > self.rsi_overbought
        
        # Volume confirmation
        volume_increase = current['volume'] > df['volume'].rolling(20).mean().iloc[-1]
        
        # Generate signals
        if ma_crossover and rsi_oversold and volume_increase:
            return 'BUY'
        elif ma_crossunder and rsi_overbought:
            return 'SELL'
        
        return None
    
    def _open_position(self, candle):
        """Open a new position"""
        try:
            # Calculate position size
            balance = self.get_balance()
            amount = (balance * self.position_size) / candle['close']
            
            # Place order
            order = self.broker.place_order(
                symbol=self.symbol,
                side='BUY',
                amount=amount,
                price=candle['close']
            )
            
            if order:
                self.current_position = {
                    'entry_price': candle['close'],
                    'amount': amount,
                    'entry_time': datetime.now()
                }
                
                self.notification_manager.send_notification(
                    "Position opened",
                    f"Bought {amount} {self.symbol} at {candle['close']}"
                )
                
        except Exception as e:
            print(f"Error opening position: {str(e)}")
    
    def _close_position(self, candle):
        """Close current position"""
        try:
            if not self.current_position:
                return
                
            # Place order
            order = self.broker.place_order(
                symbol=self.symbol,
                side='SELL',
                amount=self.current_position['amount'],
                price=candle['close']
            )
            
            if order:
                # Calculate profit/loss
                pnl = (candle['close'] - self.current_position['entry_price']) * self.current_position['amount']
                
                # Record trade
                self.trades.append({
                    'entry_price': self.current_position['entry_price'],
                    'exit_price': candle['close'],
                    'amount': self.current_position['amount'],
                    'pnl': pnl,
                    'entry_time': self.current_position['entry_time'],
                    'exit_time': datetime.now()
                })
                
                self.notification_manager.send_notification(
                    "Position closed",
                    f"Sold {self.current_position['amount']} {self.symbol} at {candle['close']} (PnL: {pnl})"
                )
                
                self.current_position = None
                
        except Exception as e:
            print(f"Error closing position: {str(e)}")
    
    def _update_equity(self, candle):
        """Update equity curve"""
        balance = self.get_balance()
        if self.current_position:
            unrealized_pnl = (candle['close'] - self.current_position['entry_price']) * self.current_position['amount']
            balance += unrealized_pnl
            
        self.equity_curve.append({
            'timestamp': candle.name,
            'equity': balance
        })
    
    def get_balance(self):
        """Get current account balance"""
        try:
            return self.broker.get_balance()
        except Exception as e:
            print(f"Error getting balance: {str(e)}")
            return 0
    
    def get_trades(self):
        """Get list of completed trades"""
        return self.trades
    
    def get_equity_curve(self):
        """Get equity curve data"""
        return self.equity_curve

