from flask_login import LoginManager
from flask import g
import sqlite3
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from src.config import Config
import jwt
from datetime import datetime, timedelta
from .models import db, User

login_manager = LoginManager()
db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    email = db.Column(db.String(120), unique=True, nullable=False)
    api_key = db.Column(db.String(64), unique=True)
    is_active = db.Column(db.Boolean, default=True)
    
    def __init__(self, username, password, email):
        self.username = username
        self.set_password(password)
        self.email = email
        self.api_key = self.generate_api_key()

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def generate_token(self, expires_in=3600):
        payload = {
            'user_id': self.id,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in)
        }
        return jwt.encode(payload, Config.SECRET_KEY, algorithm='HS256')

    def generate_api_key(self):
        return generate_password_hash(self.username + self.email)[:64]

    def encrypt_sensitive_data(self, data):
        import json
        encrypted = Config.cipher_suite.encrypt(json.dumps(data).encode()).decode()
        return encrypted

    def decrypt_sensitive_data(self, encrypted_data):
        import json
        decrypted = Config.cipher_suite.decrypt(encrypted_data.encode()).decode()
        return json.loads(decrypted)

    def rotate_encryption_key(self, encrypted_data):
        # Decrypt with old key, re-encrypt with new key
        data = self.decrypt_sensitive_data(encrypted_data)
        # Optionally, you could rotate Config.ENCRYPTION_KEY here
        return self.encrypt_sensitive_data(data)

def init_auth(app):
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    # ensure our SQLite DB exists
    conn = sqlite3.connect('trading_bot.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY,
          username TEXT UNIQUE,
          password_hash TEXT
        )
    ''')
    conn.commit()
    conn.close()

@login_manager.user_loader
def load_user(user_id):
    # TODO: load user object from SQLite
    return None

def init_db(app):
    """Initialize the database and create tables."""
    with app.app_context():
        db.create_all()
        
        # Create admin user if it doesn't exist
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(
                username='admin',
                email='admin@example.com',
                password_hash=generate_password_hash('admin123')
            )
            db.session.add(admin)
            db.session.commit()

if __name__ == '__main__':
    from src.app import app
    init_db(app)

