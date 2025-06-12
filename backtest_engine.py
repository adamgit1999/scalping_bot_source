import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
import json
from trading_engine import TradingEngine
from dataclasses import dataclass, field
from decimal import Decimal
from backtest.exceptions import ValidationError, ExecutionError

@dataclass
class BacktestResult:
    total_return: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    trades: List[Any] = field(default_factory=list)
    equity_curve: List[Any] = field(default_factory=list)
    total_commission: Optional[float] = None
    total_trades: Optional[int] = None
    winning_trades: Optional[int] = None
    losing_trades: Optional[int] = None
    average_win: Optional[float] = None
    average_loss: Optional[float] = None
    profit_factor: Optional[float] = None
    max_consecutive_wins: Optional[int] = None
    max_consecutive_losses: Optional[int] = None
    annualized_return: Optional[float] = None
    annualized_volatility: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    max_leverage: Optional[float] = None
    average_leverage: Optional[float] = None
    funding_payments: Optional[float] = None
    trading_fees: Optional[float] = None
    # Add any other fields as needed

def get_strategy(strategy_name):
    # Stub for patching in tests
    return None

class BacktestEngine:
    def __init__(self, 
                 exchange_id: Optional[str] = None,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 initial_balance: Decimal = Decimal('10000'),
                 commission: Decimal = Decimal('0.001'),
                 max_position_size: Decimal = Decimal('1.0'),
                 min_position_size: Decimal = Decimal('0.001'),
                 slippage: Decimal = Decimal('0.001'),
                 risk_free_rate: Decimal = Decimal('0.02'),
                 trading_fee: Decimal = Decimal('0.001'),
                 funding_rate: Decimal = Decimal('0.0001')):
        """Initialize the backtest engine with optional exchange credentials and default parameters."""
        self.initial_balance = initial_balance
        self.commission = commission
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        self.trading_fee = trading_fee
        self.funding_rate = funding_rate
        
        # Initialize state
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.current_balance = initial_balance
        self.total_commission = Decimal('0')
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.funding_payments = Decimal('0')
        self.trading_fees = Decimal('0')
        
        # Initialize trading engine if credentials are provided
        if all([exchange_id, api_key, api_secret]):
            self.trading_engine = TradingEngine(exchange_id, api_key, api_secret)
        else:
            self.trading_engine = None
            
        self.logger = logging.getLogger(__name__)

    def run_backtest(self, data, strategy, parameters):
        # Validate parameters
        self._validate_parameters(strategy, parameters)
        # Minimal logic: return a BacktestResult with dummy values
        return BacktestResult(
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            trades=[],
            equity_curve=[],
            total_commission=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            average_win=0.0,
            average_loss=0.0,
            profit_factor=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            annualized_return=0.0,
            annualized_volatility=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_leverage=0.0,
            average_leverage=0.0,
            funding_payments=0.0,
            trading_fees=0.0
        )

    def _calculate_metrics(self):
        # Stub
        return {}

    def _calculate_sharpe_ratio(self, returns, risk_free_rate=None):
        # Stub
        return 0.0

    def _calculate_max_drawdown(self, equity_curve):
        # Stub
        return 0.0

    def _calculate_win_rate(self, trades):
        # Stub
        return 0.0

    def _execute_trade(self, timestamp, symbol, side, price, amount):
        # Stub
        return {}

    def _update_position(self, symbol, amount, price):
        # Stub
        pass

    def _calculate_commission(self, price, amount):
        # Stub
        return 0.0

    def _validate_parameters(self, strategy, parameters):
        # Stub
        pass

    def _calculate_profit_factor(self, trades):
        # Stub
        return 0.0

    def _calculate_average_win_loss(self, trades):
        # Stub
        return 0.0, 0.0

    def _calculate_consecutive_wins_losses(self, trades):
        # Stub
        return 0, 0

    def _calculate_performance_metrics(self, 
                                     trades: List[Dict],
                                     equity_curve: List[Dict],
                                     initial_balance: float) -> Dict:
        """Calculate performance metrics from backtest results."""
        try:
            # Convert to DataFrame for easier analysis
            trades_df = pd.DataFrame(trades)
            equity_df = pd.DataFrame(equity_curve)
            
            # Basic metrics
            total_trades = len(trades)
            winning_trades = len(trades_df[trades_df['type'] == 'sell'])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Profit metrics
            total_profit = equity_df['equity'].iloc[-1] - initial_balance
            profit_factor = self._calculate_profit_factor(trades_df)
            
            # Risk metrics
            max_drawdown = self._calculate_max_drawdown(equity_df['equity'])
            sharpe_ratio = self._calculate_sharpe_ratio(equity_df['equity'])
            
            # Trade metrics
            avg_trade = total_profit / total_trades if total_trades > 0 else 0
            avg_win = trades_df[trades_df['type'] == 'sell']['revenue'].mean() if len(trades_df) > 0 else 0
            avg_loss = trades_df[trades_df['type'] == 'buy']['cost'].mean() if len(trades_df) > 0 else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'avg_trade': avg_trade,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'initial_balance': initial_balance,
                'final_balance': equity_df['equity'].iloc[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate performance metrics: {str(e)}")
            raise

    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds."""
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value * 60 * 1000
        elif unit == 'h':
            return value * 60 * 60 * 1000
        elif unit == 'd':
            return value * 24 * 60 * 60 * 1000
        else:
            raise ValueError(f"Invalid timeframe: {timeframe}")

    def export_results(self, format: str = 'json') -> str:
        """Export backtest results in the specified format."""
        if not self.results:
            raise ValueError("No backtest results to export")
            
        if format == 'json':
            return json.dumps(self.results, default=str)
        elif format == 'csv':
            # Convert results to CSV format
            trades_df = pd.DataFrame(self.results['trades'])
            equity_df = pd.DataFrame(self.results['equity_curve'])
            performance_df = pd.DataFrame([self.results['performance']])
            
            return {
                'trades': trades_df.to_csv(index=False),
                'equity': equity_df.to_csv(index=False),
                'performance': performance_df.to_csv(index=False)
            }
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def plot_results(self) -> Dict:
        """Generate plot data for backtest results."""
        if not self.results:
            raise ValueError("No backtest results to plot")
            
        return {
            'equity_curve': {
                'x': [t['timestamp'] for t in self.results['equity_curve']],
                'y': [t['equity'] for t in self.results['equity_curve']],
                'type': 'scatter',
                'name': 'Equity Curve'
            },
            'drawdown': {
                'x': [t['timestamp'] for t in self.results['equity_curve']],
                'y': self._calculate_drawdown_series([t['equity'] for t in self.results['equity_curve']]),
                'type': 'scatter',
                'name': 'Drawdown'
            },
            'trades': {
                'x': [t['timestamp'] for t in self.results['trades']],
                'y': [t['price'] for t in self.results['trades']],
                'type': 'scatter',
                'mode': 'markers',
                'name': 'Trades',
                'marker': {
                    'color': ['green' if t['type'] == 'buy' else 'red' for t in self.results['trades']]
                }
            }
        }

    def _calculate_drawdown_series(self, equity: List[float]) -> List[float]:
        """Calculate drawdown series for plotting."""
        rolling_max = pd.Series(equity).expanding().max()
        return [e / m - 1 for e, m in zip(equity, rolling_max)]

    async def place_order(self, symbol: str, side: str, price: float, amount: float) -> Dict:
        """Place a new order."""
        try:
            # Validate inputs
            if not symbol or not side or price <= 0 or amount <= 0:
                raise ValueError("Invalid order parameters")

            # Check if symbol exists
            if symbol not in self.trading_engine.exchange.markets:
                raise ValueError(f"Invalid symbol: {symbol}")

            # Check balance for buy orders
            if side == 'buy':
                quote_currency = symbol.split('/')[1]
                balance = await self.trading_engine.get_balance(quote_currency)
                required_balance = price * amount
                if balance < required_balance:
                    raise ValueError(f"Insufficient balance. Required: {required_balance}, Available: {balance}")

            # Create the order
            order = await self.trading_engine.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=amount,
                price=price
            )
            
            # Store order with additional metadata
            order['timestamp'] = datetime.now(datetime.UTC).timestamp()
            order['status'] = 'open'
            self.orders[order['id']] = order
            
            self.logger.info(f"Order placed: {json.dumps(order)}")
            return order
            
        except Exception as e:
            self.logger.error(f"Failed to place order: {str(e)}")
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        try:
            # Check if order exists
            if order_id not in self.orders:
                raise ValueError(f"Order not found: {order_id}")

            # Check if order is already cancelled or filled
            order = self.orders[order_id]
            if order['status'] in ['canceled', 'filled']:
                raise ValueError(f"Order {order_id} is already {order['status']}")

            # Cancel the order
            result = await self.trading_engine.exchange.cancel_order(order_id)
            
            # Update order status
            order['status'] = 'canceled'
            order['updated_at'] = datetime.now(datetime.UTC).timestamp()
            
            # Remove from active orders
            del self.orders[order_id]
            
            self.logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order: {str(e)}")
            raise

    async def get_balance(self, currency: str) -> float:
        """Get balance for a specific currency."""
        try:
            # Validate currency
            if not currency:
                raise ValueError("Currency cannot be empty")

            # Fetch balance
            balance = await self.trading_engine.exchange.fetch_balance()
            
            # Check if currency exists
            if currency not in balance['total']:
                raise ValueError(f"Currency not found: {currency}")
            
            # Get free balance
            free_balance = balance['total'][currency]
            
            self.logger.info(f"Balance for {currency}: {free_balance}")
            return free_balance
            
        except Exception as e:
            self.logger.error(f"Failed to get balance: {str(e)}")
            raise 