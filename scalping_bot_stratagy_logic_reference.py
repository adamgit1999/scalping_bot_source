import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import time
import os
import requests
import sys

try:
    from exchange_api_config import API_KEY, API_SECRET
except ImportError:
    API_KEY = ""
    API_SECRET = ""

if not API_KEY or not API_SECRET:
    print("[INFO] No API key/secret set. Running in mock/live data mode only.")
else:
    print("[INFO] API key/secret loaded. (Real exchange integration placeholder)")
    # TODO: Add real exchange trading logic here using API_KEY and API_SECRET

# Parameters
START_BALANCE = 250.0  # GBP
N_STEPS = 400
N_TRADES = 20  # Perform 20 trades per launch
CRYPTO_LIST = ['BTC', 'ETH']
FIXED_POSITION_SIZE = 0.1
MA_FAST = 3
MA_SLOW = 7
MA_DIFF_THRESHOLD = 0.001
MAX_TRADE_PCT = 0.075  # 7.5% of balance per trade
TAKE_PROFIT_PCT = 0.10  # 10% take profit
STOP_LOSS_PCT = 0.05    # 5% stop loss
VOLATILITY_WINDOW = 10
VOLATILITY_THRESHOLD = 0.002
LIVE_BALANCE_FILE = "live_balance.txt"


def save_balance(balance, filename=LIVE_BALANCE_FILE):
    with open(filename, "w") as f:
        f.write(str(balance))

def load_balance(default_balance, filename=LIVE_BALANCE_FILE):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                return float(f.read())
            except Exception:
                return default_balance
    return default_balance

def fetch_live_ohlcv_binance(symbol, interval='1m', limit=1000):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    r = requests.get(url)
    data = r.json()
    if not isinstance(data, list) or len(data) == 0:
        print('Error fetching data:', data)
        return None
    df = pd.DataFrame(data, columns=[
        'OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
        'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades',
        'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'])
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    df['Datetime'] = pd.to_datetime(df['OpenTime'], unit='ms')
    df = df.set_index('Datetime')
    df = df.sort_index()
    return df

def compute_moving_averages(df, short_ma=MA_FAST, long_ma=MA_SLOW):
    df['SMA_fast'] = df['Close'].rolling(window=short_ma).mean()
    df['SMA_slow'] = df['Close'].rolling(window=long_ma).mean()
    df['Signal'] = 0
    buy_condition = (df['Close'] > df['SMA_fast']) & (df['SMA_fast'] > df['SMA_slow']) & (df['Close'].shift(1) <= df['SMA_fast'].shift(1))
    sell_condition = (df['Close'] < df['SMA_fast']) & (df['SMA_fast'] < df['SMA_slow']) & (df['Close'].shift(1) >= df['SMA_fast'].shift(1))
    df.loc[buy_condition, 'Signal'] = 1
    df.loc[sell_condition, 'Signal'] = -1
    df['Position'] = df['Signal'].diff()
    return df

def run_scalping_trades(crypto_dfs, n_trades, start_balance):
    print("Cycle Start")
    np.random.seed(int(time.time()) % 2**32)
    balance = start_balance
    trades = []
    trade_count = 0
    crossover_points = []
    signals_by_pair = {crypto: [] for crypto in crypto_dfs.keys()}
    
    # Collect all valid signals for each crypto
    for crypto in crypto_dfs.keys():
        df = crypto_dfs[crypto]
        for i in range(max(MA_SLOW, VOLATILITY_WINDOW), len(df)):
            ma_diff = abs(df['SMA_fast'].iloc[i] - df['SMA_slow'].iloc[i]) / df['SMA_slow'].iloc[i]
            recent_vol = np.std(df['Close'].iloc[i-VOLATILITY_WINDOW:i].pct_change().dropna())
            slow_ma_trend = df['SMA_slow'].iloc[i] - df['SMA_slow'].iloc[i-3]
            if df['Position'].iloc[i] == 1 and ma_diff > MA_DIFF_THRESHOLD and recent_vol > VOLATILITY_THRESHOLD and slow_ma_trend > 0:
                signals_by_pair[crypto].append((df.index[i], crypto, 'BUY'))
            elif df['Position'].iloc[i] == -1 and ma_diff > MA_DIFF_THRESHOLD and recent_vol > VOLATILITY_THRESHOLD and slow_ma_trend < 0:
                signals_by_pair[crypto].append((df.index[i], crypto, 'SELL'))
    
    # Step 1: Guarantee at least one trade per pair (if possible)
    guaranteed_signals = []
    for crypto, signals in signals_by_pair.items():
        if signals:
            # Pick the earliest signal for this pair
            guaranteed_signals.append(signals[0])
            # Remove it from the pool
            signals_by_pair[crypto] = signals[1:]
    
    # Step 2: Gather all remaining signals
    remaining_signals = []
    for signals in signals_by_pair.values():
        remaining_signals.extend(signals)
    
    # Step 3: Sort all signals chronologically
    guaranteed_signals.sort(key=lambda x: x[0])
    remaining_signals.sort(key=lambda x: x[0])
    
    # Step 4: Build the final crossover_points list
    crossover_points = guaranteed_signals + remaining_signals
    crossover_points = crossover_points[:n_trades]  # Only take up to n_trades
    
    print(f"Guaranteed at least one trade per pair (if possible). Total signals used: {len(crossover_points)}")
    if len(crossover_points) == 0:
        print("No valid signals found for any crypto pair. Exiting.")
        return [], start_balance, [start_balance]
    
    idx = 0
    max_attempts = n_trades * 50
    attempts = 0
    equity_curve = [balance]
    open_trades = []  # List of dicts for overlapping trades
    
    print(f"\nStarting trade execution (target: {n_trades} trades)...")
    while trade_count < n_trades and attempts < max_attempts:
        if idx >= len(crossover_points):
            print("No more signals available")
            break
        date, crypto, action = crossover_points[idx]
        df = crypto_dfs[crypto]
        i = df.index.get_loc(date)
        price = df['Close'].iloc[i]
        # Position size: up to 7.5% of balance per trade
        pos_size = FIXED_POSITION_SIZE
        max_affordable = balance / price
        max_allowed = (MAX_TRADE_PCT * balance) / price
        pos_size = min(pos_size, max_affordable, max_allowed)
        if pos_size < 0.0001:
            print(f"Skipping {crypto} {action} - position size too small or not enough balance")
            print(f"Live balance: £{balance:.2f}")
            idx += 1
            attempts += 1
            continue
        if action == 'BUY' and balance >= price * pos_size and pos_size > 0:
            entry_price = price
            stop_loss = entry_price * (1 - STOP_LOSS_PCT)
            take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
            balance -= price * pos_size
            open_trades.append({
                'crypto': crypto,
                'entry': entry_price,
                'size': pos_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'open_idx': i
            })
            trades.append({
                'Trade#': len(trades)+1,
                'Date': date,
                'Action': 'BUY',
                'Crypto': crypto,
                'Price': price,
                'Size': pos_size,
                'Balance': balance,
                'Position': pos_size
            })
            print(f"Opened BUY position for {crypto} at {price}")
            print(f"Live balance: £{balance:.2f}")
            trade_count += 1
        elif action == 'SELL':
            # Find all open trades for this crypto and close the earliest one
            open_for_crypto = [t for t in open_trades if t['crypto'] == crypto]
            if open_for_crypto:
                ot = open_for_crypto[0]
                entry = ot['entry']
                size = ot['size']
                stop_loss = ot['stop_loss']
                take_profit = ot['take_profit']
                open_idx = ot['open_idx']
                close_prices = df['Close'].iloc[open_idx:i+1]
                sl_hit = (close_prices <= stop_loss).any()
                tp_hit = (close_prices >= take_profit).any()
                exit_price = price
                exit_reason = 'SELL'
                if sl_hit:
                    exit_price = close_prices[close_prices <= stop_loss].iloc[0]
                    exit_reason = 'STOP_LOSS'
                elif tp_hit:
                    exit_price = close_prices[close_prices >= take_profit].iloc[0]
                    exit_reason = 'TAKE_PROFIT'
                balance += exit_price * size
                trades.append({
                    'Trade#': len(trades)+1,
                    'Date': date,
                    'Action': exit_reason,
                    'Crypto': crypto,
                    'Price': exit_price,
                    'Size': size,
                    'Balance': balance,
                    'Position': 0.0
                })
                # Calculate P&L
                pnl = (exit_price - entry) * size if entry else 0
                win = pnl > 0
                status = '[WIN]' if win else '[LOSS]'
                print(f"Closed {exit_reason} position for {crypto} at {exit_price} {status} P&L: £{pnl:.2f}")
                print(f"Live balance: £{balance:.2f}")
                trade_count += 1
                open_trades.remove(ot)
            else:
                print(f"No open position to close for {crypto}")
        equity_curve.append(balance)
        idx += 1
        attempts += 1
    print(f"\nTrade execution complete. Executed {trade_count} trades.")
    print(f"Open positions at end: {len(open_trades)}")
    # Liquidate all open trades at end
    for ot in open_trades:
        crypto = ot['crypto']
        df = crypto_dfs[crypto]
        price = df['Close'].iloc[-1]
        if price <= 0:
            price = 0.01
        size = ot['size']
        entry = ot['entry']
        balance += size * price
        trades.append({
            'Trade#': len(trades)+1,
            'Date': df.index[-1],
            'Action': 'SELL_END',
            'Crypto': crypto,
            'Price': price,
            'Size': size,
            'Balance': balance,
            'Position': 0.0
        })
        pnl = (price - entry) * size if entry else 0
        win = pnl > 0
        status = '[WIN]' if win else '[LOSS]'
        print(f"Liquidated {crypto} position at {price} {status} P&L: £{pnl:.2f}")
        print(f"Live balance: £{balance:.2f}")
        equity_curve.append(balance)
    print("Cycle End")
    return trades, balance, equity_curve

def print_performance_metrics(trades, start_balance):
    if not trades:
        print("No trades executed.")
        return
    profits = [t['Price']*t['Size'] if t['Action'] in ['SELL', 'SELL_END', 'TAKE_PROFIT', 'STOP_LOSS'] else -t['Price']*t['Size'] for t in trades]
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p < 0]
    win_rate = len(wins) / max(1, (len(wins) + len(losses)))
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = abs(np.sum(wins) / np.sum(losses)) if losses else float('inf')
    balances = [start_balance]
    for t in trades:
        if t['Action'] in ['SELL', 'SELL_END', 'TAKE_PROFIT', 'STOP_LOSS']:
            balances.append(balances[-1] + t['Price']*t['Size'])
        else:
            balances.append(balances[-1] - t['Price']*t['Size'])
    drawdowns = [0]
    peak = balances[0]
    for b in balances[1:]:
        if b > peak:
            peak = b
        drawdowns.append(peak - b)
    max_drawdown = max(drawdowns)
    print(f"Win rate: {win_rate*100:.2f}%")
    print(f"Average win: £{avg_win:.2f}")
    print(f"Average loss: £{avg_loss:.2f}")
    print(f"Profit factor: {profit_factor:.2f}")
    print(f"Max drawdown: £{max_drawdown:.2f}")

if __name__ == "__main__":
    print("Live Crypto mode (all 5 GBP pairs)")
    # Get number of trades from command line argument
    n_trades = N_TRADES
    continuous = False
    for arg in sys.argv[1:]:
        if arg == "--continuous":
            continuous = True
        else:
            try:
                n_trades = int(arg)
                if n_trades < 1:
                    n_trades = N_TRADES
            except ValueError:
                pass
    print(f"Target number of trades: {n_trades}")
    if continuous:
        print("Continuous trading mode enabled.")
    
    def run_once():
        # Fetch all 5 pairs
        symbol_list = [f'{base}GBP' for base in CRYPTO_LIST]
        crypto_dfs = {}
        for symbol in symbol_list:
            print(f"Fetching live data for {symbol} (Binance 1m)...")
            df = fetch_live_ohlcv_binance(symbol)
            if df is None or len(df) < max(MA_SLOW, VOLATILITY_WINDOW) + 1:
                print(f"Not enough data to run strategy for {symbol}. Skipping.")
                continue
            df = compute_moving_averages(df, MA_FAST, MA_SLOW)
            crypto_dfs[symbol] = df
        if not crypto_dfs:
            print("No valid data for any crypto pairs. Exiting.")
            return None, None, None
        # Load or initialize live balance
        live_balance = load_balance(250.0, LIVE_BALANCE_FILE)
        trades, final_balance, equity_curve = run_scalping_trades(crypto_dfs, n_trades, live_balance)
        print(f"Starting balance: £{live_balance:.2f}")
        for t in trades:
            print(f"Trade {t['Trade#']:2d}: {t['Date'].strftime('%Y-%m-%d %H:%M')} | {t['Action']:10s} | {t['Crypto']:7s} | £{t['Price']:7.5f} | Size: {t['Size']:.4f} | Balance: £{t['Balance']:7.2f} | Position: {t['Position']:.4f}")
        print(f"Final balance: £{final_balance:.2f}")
        print_performance_metrics(trades, live_balance)
        save_balance(final_balance, LIVE_BALANCE_FILE)
        return trades, final_balance, equity_curve
    
    if continuous:
        try:
            while True:
                print("\n--- Starting new trading cycle ---\n")
                run_once()
        except KeyboardInterrupt:
            print("Continuous trading stopped by user.")
    else:
        run_once()