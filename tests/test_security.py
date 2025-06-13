import pytest
from src.app import app, db, User
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import json
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import base64

@pytest.fixture
def app_context():
    """Create app context and initialize test database."""
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['SECRET_KEY'] = 'test-secret-key'
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

@pytest.fixture
def test_user(app_context):
    """Create a test user."""
    user = User(
        username='testuser',
        email='test@example.com',
        password_hash=generate_password_hash('testpassword')
    )
    db.session.add(user)
    db.session.commit()
    return user

def test_password_security(app_context, test_user):
    """Test password security mechanisms."""
    # Test password hashing
    assert test_user.password_hash != 'testpassword'
    assert check_password_hash(test_user.password_hash, 'testpassword')
    
    # Test password verification
    assert test_user.verify_password('testpassword') is True
    assert test_user.verify_password('wrongpassword') is False
    
    # Test password update
    test_user.set_password('newpassword')
    assert test_user.verify_password('newpassword') is True
    assert test_user.verify_password('testpassword') is False

def test_jwt_token_security(app_context, test_user):
    """Test JWT token security."""
    # Generate token
    token = test_user.generate_token()
    assert token is not None
    
    # Verify token
    payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
    assert payload['user_id'] == test_user.id
    
    # Test token expiration
    expired_token = jwt.encode(
        {
            'user_id': test_user.id,
            'exp': datetime.now(timezone.utc) - timedelta(hours=1)
        },
        app.config['SECRET_KEY'],
        algorithm='HS256'
    )
    with pytest.raises(jwt.ExpiredSignatureError):
        jwt.decode(expired_token, app.config['SECRET_KEY'], algorithms=['HS256'])
    
    # Test invalid token
    with pytest.raises(jwt.InvalidTokenError):
        jwt.decode('invalid_token', app.config['SECRET_KEY'], algorithms=['HS256'])

def test_api_key_security(app_context, test_user):
    """Test API key security."""
    # Generate API key
    api_key = test_user.generate_api_key()
    assert api_key is not None
    assert len(api_key) == 64
    
    # Verify API key
    assert test_user.verify_api_key(api_key) is True
    assert test_user.verify_api_key('invalid_key') is False
    
    # Test API key rotation
    new_api_key = test_user.rotate_api_key()
    assert new_api_key != api_key
    assert test_user.verify_api_key(new_api_key) is True
    assert test_user.verify_api_key(api_key) is False

def test_request_authentication(app_context, test_user):
    """Test request authentication mechanisms."""
    client = app.test_client()
    
    # Test login
    response = client.post('/api/login', json={
        'username': 'testuser',
        'password': 'testpassword'
    })
    assert response.status_code == 200
    assert 'token' in response.json
    
    # Test protected endpoint
    token = response.json['token']
    headers = {'Authorization': f'Bearer {token}'}
    response = client.get('/api/user/profile', headers=headers)
    assert response.status_code == 200
    
    # Test invalid token
    headers = {'Authorization': 'Bearer invalid_token'}
    response = client.get('/api/user/profile', headers=headers)
    assert response.status_code == 401

def test_rate_limiting(app_context, test_user):
    """Test rate limiting mechanisms."""
    client = app.test_client()
    
    # Test login attempts
    for _ in range(5):
        response = client.post('/api/login', json={
            'username': 'testuser',
            'password': 'wrongpassword'
        })
    
    # Should be rate limited
    response = client.post('/api/login', json={
        'username': 'testuser',
        'password': 'wrongpassword'
    })
    assert response.status_code == 429

def test_input_validation(app_context, test_user):
    """Test input validation and sanitization."""
    client = app.test_client()
    
    # Test SQL injection attempt
    response = client.post('/api/login', json={
        'username': "testuser' OR '1'='1",
        'password': "password' OR '1'='1"
    })
    assert response.status_code == 401
    
    # Test XSS attempt
    response = client.post('/api/strategy', json={
        'name': '<script>alert("xss")</script>',
        'description': '<img src="x" onerror="alert(\'xss\')">'
    })
    assert response.status_code == 400

def test_csrf_protection(app_context, test_user):
    """Test CSRF protection mechanisms."""
    client = app.test_client()
    
    # Test without CSRF token
    response = client.post('/api/strategy', json={
        'name': 'Test Strategy',
        'description': 'Test Description'
    })
    assert response.status_code == 403
    
    # Test with CSRF token
    response = client.get('/api/csrf-token')
    csrf_token = response.json['csrf_token']
    
    response = client.post('/api/strategy', json={
        'name': 'Test Strategy',
        'description': 'Test Description'
    }, headers={'X-CSRF-Token': csrf_token})
    assert response.status_code == 201

def test_webhook_security(app_context, test_user):
    """Test webhook security mechanisms."""
    client = app.test_client()
    
    # Generate webhook signature
    payload = json.dumps({'event': 'test'})
    timestamp = str(int(datetime.now(timezone.utc).timestamp()))
    signature = hmac.new(
        test_user.api_key.encode(),
        f"{timestamp}{payload}".encode(),
        hashlib.sha256
    ).hexdigest()
    
    # Test valid webhook
    headers = {
        'X-Webhook-Signature': signature,
        'X-Webhook-Timestamp': timestamp
    }
    response = client.post('/api/webhook', data=payload, headers=headers)
    assert response.status_code == 200
    
    # Test invalid signature
    headers['X-Webhook-Signature'] = 'invalid_signature'
    response = client.post('/api/webhook', data=payload, headers=headers)
    assert response.status_code == 401
    
    # Test expired timestamp
    headers['X-Webhook-Timestamp'] = str(int(datetime.now(timezone.utc).timestamp()) - 3600)
    response = client.post('/api/webhook', data=payload, headers=headers)
    assert response.status_code == 401

def test_data_encryption(app_context, test_user):
    """Test data encryption mechanisms."""
    # Test sensitive data encryption
    sensitive_data = {
        'api_key': 'test_api_key',
        'secret': 'test_secret'
    }
    
    encrypted_data = test_user.encrypt_sensitive_data(sensitive_data)
    assert encrypted_data != sensitive_data
    
    decrypted_data = test_user.decrypt_sensitive_data(encrypted_data)
    assert decrypted_data == sensitive_data
    
    # Test encryption key rotation
    new_encrypted_data = test_user.rotate_encryption_key(encrypted_data)
    assert new_encrypted_data != encrypted_data
    
    decrypted_data = test_user.decrypt_sensitive_data(new_encrypted_data)
    assert decrypted_data == sensitive_data

def test_audit_logging(app_context, test_user):
    """Test audit logging mechanisms."""
    # Test login attempt logging
    client = app.test_client()
    response = client.post('/api/login', json={
        'username': 'testuser',
        'password': 'testpassword'
    })
    assert response.status_code == 200
    
    # Verify audit log
    audit_log = test_user.get_audit_logs()
    assert len(audit_log) > 0
    assert audit_log[0]['action'] == 'login'
    assert audit_log[0]['ip_address'] is not None
    assert audit_log[0]['timestamp'] is not None
    
    # Test sensitive action logging
    test_user.rotate_api_key()
    audit_log = test_user.get_audit_logs()
    assert audit_log[0]['action'] == 'api_key_rotation'
    assert audit_log[0]['details'] is not None

def test_session_security(app_context, test_user):
    """Test session security mechanisms."""
    client = app.test_client()
    
    # Test login and session creation
    response = client.post('/api/login', json={
        'username': 'testuser',
        'password': 'testpassword'
    })
    assert response.status_code == 200
    
    # Get session cookie
    session_cookie = response.headers.get('Set-Cookie')
    assert session_cookie is not None
    
    # Test session validation
    headers = {'Cookie': session_cookie}
    response = client.get('/api/user/profile', headers=headers)
    assert response.status_code == 200
    
    # Test session expiration
    with app.test_request_context():
        app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(seconds=1)
        response = client.get('/api/user/profile', headers=headers)
        assert response.status_code == 401
    
    # Test session invalidation
    response = client.post('/api/logout')
    assert response.status_code == 200
    
    response = client.get('/api/user/profile', headers=headers)
    assert response.status_code == 401 