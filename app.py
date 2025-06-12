from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_socketio import SocketIO
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash
import os
from config import Config
from init_auth import db, User, init_db
from broker import get_broker
from bot_logic import TradingBot
from scheduler import Scheduler
from notifications import NotificationManager
import qrcode
import io
import base64
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime, timezone

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
db = SQLAlchemy(app)
socketio = SocketIO(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Initialize managers
scheduler = Scheduler()
notification_manager = NotificationManager()
trading_bot = None

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    api_key = db.Column(db.String(256))
    api_secret = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    strategies = db.relationship('Strategy', backref='user', lazy=True)

class Strategy(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    parameters = db.Column(db.JSON)
    is_active = db.Column(db.Boolean, default=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    trades = db.relationship('Trade', backref='strategy', lazy=True)

class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False, index=True)
    side = db.Column(db.String(10), nullable=False)  # 'buy' or 'sell'
    price = db.Column(db.Float, nullable=False)
    amount = db.Column(db.Float, nullable=False)
    profit = db.Column(db.Float)
    fee = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    strategy_id = db.Column(db.Integer, db.ForeignKey('strategy.id'), nullable=False, index=True)
    
    __table_args__ = (
        db.Index('idx_trade_strategy_timestamp', 'strategy_id', 'timestamp'),
        db.Index('idx_trade_symbol_timestamp', 'symbol', 'timestamp'),
    )

class Backtest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    strategy_id = db.Column(db.Integer, db.ForeignKey('strategy.id'), nullable=False)
    start_date = db.Column(db.DateTime, nullable=False)
    end_date = db.Column(db.DateTime, nullable=False)
    initial_balance = db.Column(db.Float, nullable=False)
    final_balance = db.Column(db.Float)
    total_trades = db.Column(db.Integer)
    win_rate = db.Column(db.Float)
    profit_factor = db.Column(db.Float)
    max_drawdown = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

class Notification:
    def __init__(self, *args, **kwargs):
        pass

class Webhook:
    def __init__(self, *args, **kwargs):
        pass

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
@login_required
def api_status():
    return jsonify({
        'status': 'running' if trading_bot and trading_bot.is_running else 'stopped',
        'broker': Config.load_config()['broker'],
        'symbol': Config.load_config()['symbol'],
        'mock_mode': Config.load_config()['mock_mode']
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

@app.route('/api/trades')
@login_required
def api_trades():
    if not trading_bot:
        return jsonify([])
    return jsonify(trading_bot.get_trades())

@app.route('/api/balance')
@login_required
def api_balance():
    if not trading_bot:
        return jsonify({'error': 'Bot not initialized'}), 400
    return jsonify(trading_bot.get_balance())

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

def main():
    with app.app_context():
        init_db()
        db.create_all()
    socketio.run(app, host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)

if __name__ == '__main__':
    main()

