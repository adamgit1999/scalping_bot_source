from flask import Blueprint, render_template, request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.plugins.default_strategy import DefaultStrategy
from src.config import Config

backtest_bp = Blueprint('backtest', __name__, template_folder='templates')

@backtest_bp.route('/', methods=['GET', 'POST'])
def backtest():
    if request.method == 'POST':
        # TODO: run historical simulation
        pass
    return render_template('backtest.html')

class Backtest:
    def __init__(self, broker, strategy=None):
        self.broker = broker
        self.strategy = strategy or DefaultStrategy()
        self.trades = []
        self.equity_curve = []
        self.initial_balance = 1000  # Default initial balance
        self.current_balance = self.initial_balance
        self.position = None
    
    def run(self, symbol, interval, start_date, end_date, initial_balance=None):
        """
        Run backtest on historical data
        
        Args:
            symbol (str): Trading pair symbol
            interval (str): Candle interval
            start_date (datetime): Start date
            end_date (datetime): End date
            initial_balance (float): Initial balance for backtest
        """
        if initial_balance:
            self.initial_balance = initial_balance
            self.current_balance = initial_balance
        
        # Fetch historical data
        candles = self._fetch_historical_data(symbol, interval, start_date, end_date)
        if not candles:
            raise ValueError("No historical data available for the specified period")
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Run backtest
        for i in range(len(df)):
            # Get current candle
            current_candle = df.iloc[i]
            
            # Calculate indicators and get signal
            signal = self.strategy.generate_signal(df.iloc[:i+1])
            
            # Execute trades based on signal
            if signal == 'BUY' and not self.position:
                self._open_position(current_candle)
            elif signal == 'SELL' and self.position:
                self._close_position(current_candle)
            
            # Update equity curve
            self._update_equity(current_candle)
        
        # Close any open position at the end
        if self.position:
            self._close_position(df.iloc[-1])
        
        return self._generate_results()
    
    def _fetch_historical_data(self, symbol, interval, start_date, end_date):
        """Fetch historical data from broker"""
        try:
            # Calculate number of candles needed
            interval_minutes = int(interval[:-1])
            total_minutes = (end_date - start_date).total_seconds() / 60
            num_candles = int(total_minutes / interval_minutes)
            
            # Fetch candles
            candles = self.broker.fetch_candles(
                symbol,
                interval,
                limit=num_candles
            )
            
            # Filter by date range
            filtered_candles = [
                candle for candle in candles
                if start_date.timestamp() * 1000 <= candle['open_time'] <= end_date.timestamp() * 1000
            ]
            
            return filtered_candles
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return []
    
    def _open_position(self, candle):
        """Open a new position"""
        position_size = self.current_balance * 0.1  # Use 10% of balance
        
        self.position = {
            'entry_price': candle['close'],
            'entry_time': candle['open_time'],
            'size': position_size
        }
        
        self.trades.append({
            'type': 'BUY',
            'price': candle['close'],
            'size': position_size,
            'time': candle['open_time']
        })
    
    def _close_position(self, candle):
        """Close the current position"""
        if not self.position:
            return
        
        # Calculate P&L
        entry_price = self.position['entry_price']
        exit_price = candle['close']
        position_size = self.position['size']
        
        pnl = (exit_price - entry_price) * position_size
        self.current_balance += pnl
        
        # Record trade
        self.trades.append({
            'type': 'SELL',
            'price': candle['close'],
            'size': position_size,
            'time': candle['open_time'],
            'pnl': pnl
        })
        
        # Clear position
        self.position = None
    
    def _update_equity(self, candle):
        """Update equity curve"""
        equity = self.current_balance
        
        if self.position:
            # Add unrealized P&L
            entry_price = self.position['entry_price']
            current_price = candle['close']
            position_size = self.position['size']
            unrealized_pnl = (current_price - entry_price) * position_size
            equity += unrealized_pnl
        
        self.equity_curve.append({
            'timestamp': candle['open_time'],
            'equity': equity
        })
    
    def _generate_results(self):
        """Generate backtest results"""
        # Calculate statistics
        total_trades = len(self.trades) // 2  # Divide by 2 because each trade has buy and sell
        winning_trades = sum(1 for trade in self.trades if trade.get('pnl', 0) > 0)
        losing_trades = sum(1 for trade in self.trades if trade.get('pnl', 0) < 0)
        
        total_pnl = sum(trade.get('pnl', 0) for trade in self.trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate drawdown
        equity_values = [point['equity'] for point in self.equity_curve]
        peak = np.maximum.accumulate(equity_values)
        drawdown = (peak - equity_values) / peak * 100
        max_drawdown = np.max(drawdown)
        
        # Calculate returns
        returns = pd.Series([point['equity'] for point in self.equity_curve])
        returns = returns.pct_change().dropna()
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'final_balance': self.current_balance,
            'return_pct': (self.current_balance - self.initial_balance) / self.initial_balance * 100,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }
    
    def get_trades(self):
        """Get list of executed trades"""
        return self.trades
    
    def get_equity_curve(self):
        """Get the equity curve data"""
        return self.equity_curve

