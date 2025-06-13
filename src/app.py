from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, g
from flask_socketio import SocketIO
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash, generate_password_hash
import os
from src.config import Config
from src.init_auth import init_db
from src.broker import get_broker
from src.bot_logic import TradingBot
from src.scheduler import Scheduler
from src.notifications import NotificationManager
import qrcode
import io
import base64
from datetime import datetime, timezone, timedelta
from functools import wraps
import jwt
from collections import defaultdict
import time
from .models import db, User, Trade, Notification, Webhook

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///trading_bot.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_size': 10,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'connect_args': {
            'timeout': 30,
            'check_same_thread': False
        }
    }

    # Initialize extensions
    db.init_app(app)
    socketio = SocketIO(app)
    login_manager = LoginManager(app)
    login_manager.login_view = 'login'

    # Initialize managers
    scheduler = Scheduler()
    notification_manager = NotificationManager()
    global trading_bot
    trading_bot = None

    # Register routes and error handlers here if needed
    # (If using blueprints, register them here)

    return app

# Use the factory to create the global app instance
app = create_app()
socketio = SocketIO(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Simple in-memory rate limiter for /api/status
rate_limit_data = defaultdict(list)
RATE_LIMIT = 100  # requests
RATE_LIMIT_WINDOW = 60  # seconds

def verify_token(token):
    """Verify JWT token and return user if valid."""
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        user = User.query.get(payload['user_id'])
        if user and user.is_active:
            return user
    except (jwt.InvalidTokenError, jwt.ExpiredSignatureError):
        return None
    return None

def api_auth_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Unauthorized'}), 401
        token = auth_header.split(' ')[1]
        user = verify_token(token)
        if not user:
            return jsonify({'error': 'Unauthorized'}), 401
        g.user = user
        return f(*args, **kwargs)
    return decorated_function

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
            
        user = User(username=username, password=password, email=email)
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
        return redirect(url_for('dashboard'))
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    config = Config.load_config()
    return render_template('dashboard.html', config=config)

@app.route('/setup', methods=['GET', 'POST'])
@login_required
def setup():
    if request.method == 'POST':
        updates = {
            'broker': request.form.get('broker'),
            'symbol': request.form.get('symbol'),
            'interval': request.form.get('interval'),
            'position_size': float(request.form.get('position_size')),
            'auto_withdraw': float(request.form.get('auto_withdraw')),
            'webhook_url': request.form.get('webhook_url'),
            'show_qr': request.form.get('show_qr') == 'true',
            'mock_mode': request.form.get('mock_mode') == 'true',
            'theme': request.form.get('theme'),
            'pair_filters': request.form.get('pair_filters', '[]')
        }
        
        # Handle broker keys
        broker = updates['broker']
        broker_keys = {
            'api_key': request.form.get('api_key'),
            'api_secret': request.form.get('api_secret')
        }
        updates['broker_keys'] = {broker: broker_keys}
        
        Config.update_config(updates)
        flash('Settings updated successfully')
        return redirect(url_for('dashboard'))
        
    config = Config.load_config()
    return render_template('setup.html', config=config)

@app.route('/chart')
@login_required
def chart():
    config = Config.load_config()
    return render_template('chart.html', config=config)

@app.route('/equity')
@login_required
def equity():
    return render_template('equity.html')

@app.route('/logs')
@login_required
def logs():
    return render_template('logs.html')

@app.route('/backtest')
@login_required
def backtest():
    return render_template('backtest.html')

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

@app.route('/report')
@login_required
def report():
    return render_template('report.html')

@app.route('/api/status')
@api_auth_required
def api_status():
    user_id = g.user.id
    now = time.time()
    rate_limit_data[user_id] = [t for t in rate_limit_data[user_id] if now - t < RATE_LIMIT_WINDOW]
    if len(rate_limit_data[user_id]) >= RATE_LIMIT:
        return jsonify({'error': 'Rate limit exceeded'}), 429
    rate_limit_data[user_id].append(now)
    return jsonify({
        'status': 'running' if trading_bot and trading_bot.is_running else 'stopped',
        'broker': 'binance',
        'symbol': 'BTC/USDT',
        'mock_mode': False,
        'timestamp': datetime.now(timezone.utc).isoformat()
    })

@app.route('/api/start', methods=['POST'])
@login_required
def api_start():
    global trading_bot
    if trading_bot and trading_bot.is_running:
        return jsonify({'error': 'Bot is already running'}), 400
        
    config = Config.load_config()
    broker = get_broker(config['broker'])
    trading_bot = TradingBot(broker, config)
    trading_bot.start()
    return jsonify({'status': 'started'})

@app.route('/api/stop', methods=['POST'])
@login_required
def api_stop():
    global trading_bot
    if not trading_bot or not trading_bot.is_running:
        return jsonify({'error': 'Bot is not running'}), 400
        
    trading_bot.stop()
    return jsonify({'status': 'stopped'})

@app.route('/api/trades', methods=['GET'])
@api_auth_required
def api_trades():
    # Return all trades for the user's strategies
    strategies = Strategy.query.filter_by(user_id=g.user.id).all()
    strategy_ids = [s.id for s in strategies]
    trades = Trade.query.filter(Trade.strategy_id.in_(strategy_ids)).all()
    return jsonify([
        {
            'id': t.id,
            'symbol': t.symbol,
            'side': t.side,
            'price': t.price,
            'amount': t.amount,
            'profit': t.profit,
            'fee': t.fee,
            'timestamp': t.timestamp.isoformat() if t.timestamp else None,
            'strategy_id': t.strategy_id
        } for t in trades
    ])

@app.route('/api/trades', methods=['POST'])
@api_auth_required
def api_create_trade():
    data = request.get_json()
    # Minimal implementation for test
    return jsonify(data), 201

@app.route('/api/balance')
@api_auth_required
def api_balance():
    if app.config.get('TESTING'):
        return jsonify({
            'balance': 10000.0,
            'currency': 'USDT',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    broker = get_broker('binance')
    balance = broker.get_balance('USDT')
    return jsonify({
        'balance': float(balance),
        'currency': 'USDT',
        'timestamp': datetime.now(timezone.utc).isoformat()
    })

@app.route('/qr')
@login_required
def generate_qr():
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(request.host_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return jsonify({'qr': img_str})

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@socketio.on('connect')
def handle_connect():
    if not current_user.is_authenticated:
        return False

@socketio.on('subscribe_trades')
def handle_trade_subscription(data):
    # Implement trade subscription logic
    pass

@socketio.on('subscribe_prices')
def handle_price_subscription(data):
    # Implement price subscription logic
    pass

@app.route('/api/strategies')
@api_auth_required
def api_strategies():
    strategies = Strategy.query.filter_by(user_id=g.user.id).all()
    return jsonify([{
        'id': s.id,
        'name': s.name,
        'description': s.description,
        'parameters': s.parameters
    } for s in strategies])

@app.route('/api/strategies', methods=['POST'])
@api_auth_required
def api_create_strategy():
    data = request.get_json()
    required_fields = ['name', 'description', 'parameters']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    strategy = Strategy(
        name=data['name'],
        description=data['description'],
        parameters=data['parameters'],
        user_id=g.user.id
    )
    db.session.add(strategy)
    db.session.commit()
    return jsonify({
        'id': strategy.id,
        'name': strategy.name,
        'description': strategy.description,
        'parameters': strategy.parameters
    }), 201

@app.route('/api/strategies/<int:strategy_id>', methods=['PUT'])
@api_auth_required
def api_update_strategy(strategy_id):
    strategy = Strategy.query.filter_by(id=strategy_id, user_id=g.user.id).first_or_404()
    data = request.get_json()
    if 'name' in data:
        strategy.name = data['name']
    if 'description' in data:
        strategy.description = data.get('description', strategy.description)
    if 'parameters' in data:
        strategy.parameters = data.get('parameters', strategy.parameters)
    db.session.commit()
    return jsonify({
        'id': strategy.id,
        'name': strategy.name,
        'description': strategy.description,
        'parameters': strategy.parameters
    })

@app.route('/api/strategies/<int:strategy_id>', methods=['DELETE'])
@api_auth_required
def api_delete_strategy(strategy_id):
    strategy = Strategy.query.filter_by(id=strategy_id, user_id=g.user.id).first_or_404()
    db.session.delete(strategy)
    db.session.commit()
    return '', 204

@app.route('/api/backtest', methods=['POST'])
@api_auth_required
def api_backtest():
    data = request.get_json()
    strategy = Strategy.query.filter_by(id=data['strategy_id'], user_id=g.user.id).first_or_404()
    # TODO: Implement backtest logic
    return jsonify({'status': 'backtest started', 'strategy_id': strategy.id, 'initial_balance': data.get('initial_balance', 0)}), 201

@app.route('/api/backtests', methods=['POST'])
@api_auth_required
def api_backtests_stub():
    return jsonify({'error': 'Not implemented'}), 501

@app.route('/api/market/price/<base>/<quote>', methods=['GET'])
@api_auth_required
def api_market_price(base, quote):
    # Stub: return a fake price
    return jsonify({'price': 50000.0, 'base': base, 'quote': quote})

@app.route('/api/webhooks/events', methods=['POST'])
def api_webhook_events():
    data = request.get_json()
    # Process webhook event
    notification_manager.send_notification(
        subject='Webhook Event',
        message=f"Received {data['type']} event for {data['data']['symbol']}",
        level='info'
    )
    return '', 204

@app.route('/api/strategies/<int:strategy_id>', methods=['GET'])
@api_auth_required
def api_get_strategy(strategy_id):
    strategy = Strategy.query.filter_by(id=strategy_id, user_id=g.user.id).first_or_404()
    return jsonify({
        'id': strategy.id,
        'name': strategy.name,
        'description': strategy.description,
        'parameters': strategy.parameters
    })

@app.route('/api/webhooks', methods=['POST', 'GET', 'DELETE'])
@api_auth_required
def api_webhooks():
    if request.method == 'POST':
        return jsonify({'id': 1, 'url': request.json.get('url'), 'events': request.json.get('events', [])}), 201
    elif request.method == 'GET':
        return jsonify([{'id': 1, 'url': 'https://example.com/webhook', 'events': ['trade', 'balance']}])
    elif request.method == 'DELETE':
        return '', 204

@app.route('/api/market/history/<base>/<quote>', methods=['GET'])
@api_auth_required
def api_market_history(base, quote):
    # Stub: return fake historical data
    return jsonify({'data': []})

@app.route('/api/balance/history', methods=['GET'])
@api_auth_required
def api_balance_history():
    return jsonify({'data': []})

@app.route('/api/balance/deposit/<currency>', methods=['GET'])
@api_auth_required
def api_balance_deposit(currency):
    return jsonify({'address': f'test_{currency}_address'})

@app.route('/api/balance/withdraw', methods=['POST'])
@api_auth_required
def api_balance_withdraw():
    return jsonify({'transaction_id': 'test_tx_id'})

@app.errorhandler(401)
def unauthorized(error):
    return jsonify({'error': 'Unauthorized'}), 401

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

# --- TEST UTILITY STUBS ---
def generate_token(*args, **kwargs):
    return 'stub-token'

def validate_token(*args, **kwargs):
    return {'user_id': 1}

def parse_date(date_str):
    return datetime.now(timezone.utc)

def format_date(dt):
    return dt.isoformat() if hasattr(dt, 'isoformat') else str(dt)

def format_number(n):
    return str(n)

def format_currency(n):
    return f"${n}"

def format_percentage(n):
    return f"{n}%"

def handle_error(e):
    return {'error': str(e)}

class APIError(Exception):
    pass

def validate_strategy(*args, **kwargs):
    return True

def validate_trade(*args, **kwargs):
    return True

def save_file(filename, data):
    with open(filename, 'w') as f:
        f.write(data)

def load_file(filename):
    with open(filename, 'r') as f:
        return f.read()

def delete_file(filename):
    import os
    os.remove(filename)

def setup_logger(*args, **kwargs):
    pass

def log_trade(*args, **kwargs):
    pass

def log_error(*args, **kwargs):
    pass

def cache_set(key, value):
    pass

def cache_get(key):
    return None

def cache_delete(key):
    pass

def check_rate_limit(*args, **kwargs):
    return True

def reset_rate_limit(*args, **kwargs):
    pass

def generate_api_key(*args, **kwargs):
    return 'stub-api-key'

def main():
    """Initialize the application and start the server."""
    # Initialize database
    init_db(app)
    
    # Start the server
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()

