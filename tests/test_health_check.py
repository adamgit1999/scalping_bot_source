import pytest
from unittest.mock import Mock, patch
import time
from datetime import datetime, timedelta
import psutil
import os
import requests
from typing import Dict, Any, List

class HealthCheck:
    """System for monitoring application health."""
    
    def __init__(self):
        """Initialize health check system."""
        self.checks = {}
        self.last_check = {}
        self.check_interval = 60  # seconds
        
    def register_check(self, name: str, check_func: callable) -> None:
        """
        Register a health check.
        
        Args:
            name: Check name
            check_func: Function that performs the check
        """
        self.checks[name] = check_func
        self.last_check[name] = None
        
    def run_check(self, name: str) -> Dict[str, Any]:
        """
        Run a specific health check.
        
        Args:
            name: Check name
            
        Returns:
            Dictionary with check results
        """
        if name not in self.checks:
            return {
                'status': 'error',
                'message': f'Check {name} not found',
                'timestamp': datetime.now().isoformat()
            }
            
        try:
            result = self.checks[name]()
            self.last_check[name] = datetime.now()
            
            return {
                'status': 'ok' if result else 'error',
                'message': 'Check passed' if result else 'Check failed',
                'timestamp': self.last_check[name].isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
    def run_all_checks(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary of check results
        """
        results = {}
        for name in self.checks:
            results[name] = self.run_check(name)
        return results
        
    def get_check_status(self, name: str) -> Dict[str, Any]:
        """
        Get status of a specific check.
        
        Args:
            name: Check name
            
        Returns:
            Dictionary with check status
        """
        if name not in self.checks:
            return {
                'status': 'error',
                'message': f'Check {name} not found',
                'last_check': None
            }
            
        if not self.last_check[name]:
            return {
                'status': 'unknown',
                'message': 'Check never run',
                'last_check': None
            }
            
        time_since_check = (datetime.now() - self.last_check[name]).total_seconds()
        
        return {
            'status': 'ok' if time_since_check < self.check_interval else 'stale',
            'message': 'Check is up to date' if time_since_check < self.check_interval else 'Check is stale',
            'last_check': self.last_check[name].isoformat()
        }
        
    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all checks.
        
        Returns:
            Dictionary of check statuses
        """
        statuses = {}
        for name in self.checks:
            statuses[name] = self.get_check_status(name)
        return statuses

@pytest.fixture
def health_check():
    """Create a health check instance."""
    return HealthCheck()

@pytest.fixture
def mock_check():
    """Create a mock health check function."""
    return Mock(return_value=True)

def test_health_check_initialization(health_check):
    """Test health check initialization."""
    assert health_check is not None
    assert health_check.checks == {}
    assert health_check.last_check == {}
    assert health_check.check_interval == 60

def test_register_check(health_check, mock_check):
    """Test registering health checks."""
    # Register check
    health_check.register_check('test_check', mock_check)
    
    assert 'test_check' in health_check.checks
    assert health_check.checks['test_check'] == mock_check
    assert health_check.last_check['test_check'] is None

def test_run_check(health_check, mock_check):
    """Test running health checks."""
    # Register and run check
    health_check.register_check('test_check', mock_check)
    result = health_check.run_check('test_check')
    
    assert result['status'] == 'ok'
    assert result['message'] == 'Check passed'
    assert 'timestamp' in result
    assert mock_check.called

def test_run_check_error(health_check):
    """Test running health check with error."""
    # Register check that raises exception
    def error_check():
        raise Exception("Test error")
    
    health_check.register_check('error_check', error_check)
    result = health_check.run_check('error_check')
    
    assert result['status'] == 'error'
    assert result['message'] == 'Test error'
    assert 'timestamp' in result

def test_run_all_checks(health_check, mock_check):
    """Test running all health checks."""
    # Register multiple checks
    health_check.register_check('check1', mock_check)
    health_check.register_check('check2', mock_check)
    
    # Run all checks
    results = health_check.run_all_checks()
    
    assert 'check1' in results
    assert 'check2' in results
    assert results['check1']['status'] == 'ok'
    assert results['check2']['status'] == 'ok'
    assert mock_check.call_count == 2

def test_get_check_status(health_check, mock_check):
    """Test getting check status."""
    # Register and run check
    health_check.register_check('test_check', mock_check)
    health_check.run_check('test_check')
    
    # Get status
    status = health_check.get_check_status('test_check')
    
    assert status['status'] == 'ok'
    assert status['message'] == 'Check is up to date'
    assert 'last_check' in status

def test_get_check_status_stale(health_check, mock_check):
    """Test getting stale check status."""
    # Register and run check
    health_check.register_check('test_check', mock_check)
    health_check.run_check('test_check')
    
    # Set last check to old time
    health_check.last_check['test_check'] = datetime.now() - timedelta(minutes=2)
    
    # Get status
    status = health_check.get_check_status('test_check')
    
    assert status['status'] == 'stale'
    assert status['message'] == 'Check is stale'
    assert 'last_check' in status

def test_get_all_statuses(health_check, mock_check):
    """Test getting all check statuses."""
    # Register multiple checks
    health_check.register_check('check1', mock_check)
    health_check.register_check('check2', mock_check)
    
    # Run checks
    health_check.run_check('check1')
    health_check.run_check('check2')
    
    # Get all statuses
    statuses = health_check.get_all_statuses()
    
    assert 'check1' in statuses
    assert 'check2' in statuses
    assert statuses['check1']['status'] == 'ok'
    assert statuses['check2']['status'] == 'ok'

def test_check_not_found(health_check):
    """Test handling non-existent check."""
    # Try to run non-existent check
    result = health_check.run_check('non_existent')
    
    assert result['status'] == 'error'
    assert result['message'] == 'Check non_existent not found'
    
    # Try to get status of non-existent check
    status = health_check.get_check_status('non_existent')
    
    assert status['status'] == 'error'
    assert status['message'] == 'Check non_existent not found'
    assert status['last_check'] is None

def test_check_interval(health_check, mock_check):
    """Test check interval handling."""
    # Register and run check
    health_check.register_check('test_check', mock_check)
    health_check.run_check('test_check')
    
    # Get initial status
    initial_status = health_check.get_check_status('test_check')
    assert initial_status['status'] == 'ok'
    
    # Set last check to just before interval
    health_check.last_check['test_check'] = datetime.now() - timedelta(seconds=59)
    status = health_check.get_check_status('test_check')
    assert status['status'] == 'ok'
    
    # Set last check to just after interval
    health_check.last_check['test_check'] = datetime.now() - timedelta(seconds=61)
    status = health_check.get_check_status('test_check')
    assert status['status'] == 'stale'

def test_check_result_persistence(health_check, mock_check):
    """Test check result persistence."""
    # Register and run check
    health_check.register_check('test_check', mock_check)
    result1 = health_check.run_check('test_check')
    
    # Run check again
    result2 = health_check.run_check('test_check')
    
    # Verify results are different (different timestamps)
    assert result1['timestamp'] != result2['timestamp']
    assert result1['status'] == result2['status']
    assert result1['message'] == result2['message']

def test_check_error_handling(health_check):
    """Test check error handling."""
    # Register check that raises different types of errors
    def type_error_check():
        raise TypeError("Type error")
    
    def value_error_check():
        raise ValueError("Value error")
    
    def runtime_error_check():
        raise RuntimeError("Runtime error")
    
    health_check.register_check('type_error', type_error_check)
    health_check.register_check('value_error', value_error_check)
    health_check.register_check('runtime_error', runtime_error_check)
    
    # Run checks
    type_result = health_check.run_check('type_error')
    value_result = health_check.run_check('value_error')
    runtime_result = health_check.run_check('runtime_error')
    
    # Verify error handling
    assert type_result['status'] == 'error'
    assert value_result['status'] == 'error'
    assert runtime_result['status'] == 'error'
    assert 'Type error' in type_result['message']
    assert 'Value error' in value_result['message']
    assert 'Runtime error' in runtime_result['message'] 