import pytest
from src.app import app
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time

@pytest.fixture
def driver():
    """Create a Selenium WebDriver instance."""
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    yield driver
    driver.quit()

@pytest.fixture
def app_context():
    """Create app context and initialize test database."""
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    with app.app_context():
        yield app

def test_login_page(driver, app_context):
    """Test login page functionality."""
    driver.get('http://localhost:5000/login')
    
    # Test page title
    assert 'Login' in driver.title
    
    # Test form elements
    username_input = driver.find_element(By.NAME, 'username')
    password_input = driver.find_element(By.NAME, 'password')
    submit_button = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
    
    assert username_input.is_displayed()
    assert password_input.is_displayed()
    assert submit_button.is_displayed()
    
    # Test form submission
    username_input.send_keys('testuser')
    password_input.send_keys('testpassword')
    submit_button.click()
    
    # Wait for redirect
    WebDriverWait(driver, 10).until(
        EC.url_contains('/dashboard')
    )

def test_dashboard_page(driver, app_context):
    """Test dashboard page functionality."""
    # Login first
    driver.get('http://localhost:5000/login')
    username_input = driver.find_element(By.NAME, 'username')
    password_input = driver.find_element(By.NAME, 'password')
    submit_button = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
    
    username_input.send_keys('testuser')
    password_input.send_keys('testpassword')
    submit_button.click()
    
    # Wait for dashboard to load
    WebDriverWait(driver, 10).until(
        EC.url_contains('/dashboard')
    )
    
    # Test dashboard elements
    assert 'Dashboard' in driver.title
    
    # Test navigation menu
    nav_items = driver.find_elements(By.CSS_SELECTOR, '.nav-item')
    assert len(nav_items) > 0
    
    # Test performance metrics
    metrics = driver.find_elements(By.CSS_SELECTOR, '.metric-card')
    assert len(metrics) > 0
    
    # Test recent trades table
    trades_table = driver.find_element(By.CSS_SELECTOR, '.trades-table')
    assert trades_table.is_displayed()

def test_strategy_creation(driver, app_context):
    """Test strategy creation functionality."""
    # Login first
    driver.get('http://localhost:5000/login')
    username_input = driver.find_element(By.NAME, 'username')
    password_input = driver.find_element(By.NAME, 'password')
    submit_button = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
    
    username_input.send_keys('testuser')
    password_input.send_keys('testpassword')
    submit_button.click()
    
    # Navigate to strategy creation page
    driver.get('http://localhost:5000/strategies/new')
    
    # Test form elements
    name_input = driver.find_element(By.NAME, 'name')
    description_input = driver.find_element(By.NAME, 'description')
    parameters_input = driver.find_element(By.NAME, 'parameters')
    submit_button = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
    
    # Fill form
    name_input.send_keys('Test Strategy')
    description_input.send_keys('Test Description')
    parameters_input.send_keys('{"sma_short": 10, "sma_long": 20}')
    submit_button.click()
    
    # Wait for redirect
    WebDriverWait(driver, 10).until(
        EC.url_contains('/strategies')
    )
    
    # Verify strategy creation
    strategy_list = driver.find_element(By.CSS_SELECTOR, '.strategy-list')
    assert 'Test Strategy' in strategy_list.text

def test_trade_execution(driver, app_context):
    """Test trade execution functionality."""
    # Login first
    driver.get('http://localhost:5000/login')
    username_input = driver.find_element(By.NAME, 'username')
    password_input = driver.find_element(By.NAME, 'password')
    submit_button = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
    
    username_input.send_keys('testuser')
    password_input.send_keys('testpassword')
    submit_button.click()
    
    # Navigate to trading page
    driver.get('http://localhost:5000/trading')
    
    # Test trade form
    symbol_input = driver.find_element(By.NAME, 'symbol')
    side_select = driver.find_element(By.NAME, 'side')
    amount_input = driver.find_element(By.NAME, 'amount')
    submit_button = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
    
    # Fill form
    symbol_input.send_keys('BTC/USDT')
    side_select.send_keys('buy')
    amount_input.send_keys('0.1')
    submit_button.click()
    
    # Wait for trade confirmation
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, '.trade-confirmation'))
    )
    
    # Verify trade execution
    confirmation = driver.find_element(By.CSS_SELECTOR, '.trade-confirmation')
    assert 'Trade executed successfully' in confirmation.text

def test_chart_interaction(driver, app_context):
    """Test chart interaction functionality."""
    # Login first
    driver.get('http://localhost:5000/login')
    username_input = driver.find_element(By.NAME, 'username')
    password_input = driver.find_element(By.NAME, 'password')
    submit_button = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
    
    username_input.send_keys('testuser')
    password_input.send_keys('testpassword')
    submit_button.click()
    
    # Navigate to chart page
    driver.get('http://localhost:5000/chart')
    
    # Test chart elements
    chart = driver.find_element(By.CSS_SELECTOR, '.trading-chart')
    assert chart.is_displayed()
    
    # Test chart controls
    timeframe_select = driver.find_element(By.NAME, 'timeframe')
    indicator_select = driver.find_element(By.NAME, 'indicator')
    
    # Change timeframe
    timeframe_select.send_keys('1h')
    time.sleep(1)  # Wait for chart update
    
    # Add indicator
    indicator_select.send_keys('SMA')
    time.sleep(1)  # Wait for indicator update
    
    # Verify chart updates
    assert chart.is_displayed()

def test_notifications(driver, app_context):
    """Test notification functionality."""
    # Login first
    driver.get('http://localhost:5000/login')
    username_input = driver.find_element(By.NAME, 'username')
    password_input = driver.find_element(By.NAME, 'password')
    submit_button = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
    
    username_input.send_keys('testuser')
    password_input.send_keys('testpassword')
    submit_button.click()
    
    # Navigate to notifications page
    driver.get('http://localhost:5000/notifications')
    
    # Test notification list
    notification_list = driver.find_element(By.CSS_SELECTOR, '.notification-list')
    assert notification_list.is_displayed()
    
    # Test notification settings
    settings_button = driver.find_element(By.CSS_SELECTOR, '.notification-settings')
    settings_button.click()
    
    # Test notification preferences
    email_toggle = driver.find_element(By.NAME, 'email_notifications')
    push_toggle = driver.find_element(By.NAME, 'push_notifications')
    
    email_toggle.click()
    push_toggle.click()
    
    # Save settings
    save_button = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
    save_button.click()
    
    # Verify settings update
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, '.settings-saved'))
    )

def test_mobile_responsiveness(driver, app_context):
    """Test mobile responsiveness."""
    # Set mobile viewport
    driver.set_window_size(375, 812)  # iPhone X dimensions
    
    # Login first
    driver.get('http://localhost:5000/login')
    username_input = driver.find_element(By.NAME, 'username')
    password_input = driver.find_element(By.NAME, 'password')
    submit_button = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
    
    username_input.send_keys('testuser')
    password_input.send_keys('testpassword')
    submit_button.click()
    
    # Test responsive navigation
    menu_button = driver.find_element(By.CSS_SELECTOR, '.mobile-menu-button')
    menu_button.click()
    
    # Verify mobile menu
    mobile_menu = driver.find_element(By.CSS_SELECTOR, '.mobile-menu')
    assert mobile_menu.is_displayed()
    
    # Test responsive layout
    dashboard = driver.find_element(By.CSS_SELECTOR, '.dashboard')
    assert dashboard.is_displayed()
    
    # Test responsive charts
    chart = driver.find_element(By.CSS_SELECTOR, '.trading-chart')
    assert chart.is_displayed()
    
    # Test responsive tables
    table = driver.find_element(By.CSS_SELECTOR, '.trades-table')
    assert table.is_displayed()

def test_error_handling(driver, app_context):
    """Test error handling in the frontend."""
    # Test invalid login
    driver.get('http://localhost:5000/login')
    username_input = driver.find_element(By.NAME, 'username')
    password_input = driver.find_element(By.NAME, 'password')
    submit_button = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
    
    username_input.send_keys('invalid_user')
    password_input.send_keys('invalid_password')
    submit_button.click()
    
    # Verify error message
    error_message = driver.find_element(By.CSS_SELECTOR, '.error-message')
    assert error_message.is_displayed()
    
    # Test invalid strategy creation
    driver.get('http://localhost:5000/strategies/new')
    submit_button = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
    submit_button.click()
    
    # Verify validation errors
    validation_errors = driver.find_elements(By.CSS_SELECTOR, '.validation-error')
    assert len(validation_errors) > 0
    
    # Test network error handling
    driver.get('http://localhost:5000/api/invalid-endpoint')
    error_message = driver.find_element(By.CSS_SELECTOR, '.error-message')
    assert error_message.is_displayed()

def test_data_refresh(driver, app_context):
    """Test data refresh functionality."""
    # Login first
    driver.get('http://localhost:5000/login')
    username_input = driver.find_element(By.NAME, 'username')
    password_input = driver.find_element(By.NAME, 'password')
    submit_button = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
    
    username_input.send_keys('testuser')
    password_input.send_keys('testpassword')
    submit_button.click()
    
    # Navigate to dashboard
    driver.get('http://localhost:5000/dashboard')
    
    # Test manual refresh
    refresh_button = driver.find_element(By.CSS_SELECTOR, '.refresh-button')
    refresh_button.click()
    
    # Wait for data refresh
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, '.data-updated'))
    )
    
    # Test auto-refresh
    time.sleep(30)  # Wait for auto-refresh interval
    assert driver.find_element(By.CSS_SELECTOR, '.data-updated').is_displayed()

def test_user_preferences(driver, app_context):
    """Test user preferences functionality."""
    # Login first
    driver.get('http://localhost:5000/login')
    username_input = driver.find_element(By.NAME, 'username')
    password_input = driver.find_element(By.NAME, 'password')
    submit_button = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
    
    username_input.send_keys('testuser')
    password_input.send_keys('testpassword')
    submit_button.click()
    
    # Navigate to preferences
    driver.get('http://localhost:5000/preferences')
    
    # Test theme selection
    theme_select = driver.find_element(By.NAME, 'theme')
    theme_select.send_keys('dark')
    
    # Test language selection
    language_select = driver.find_element(By.NAME, 'language')
    language_select.send_keys('es')
    
    # Test timezone selection
    timezone_select = driver.find_element(By.NAME, 'timezone')
    timezone_select.send_keys('UTC')
    
    # Save preferences
    save_button = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
    save_button.click()
    
    # Verify preferences update
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, '.preferences-saved'))
    )
    
    # Verify theme change
    assert 'dark-theme' in driver.find_element(By.TAG_NAME, 'body').get_attribute('class') 