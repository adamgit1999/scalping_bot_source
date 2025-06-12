from enum import Enum
from datetime import datetime, timedelta
import json
from typing import List, Dict, Any, Optional, Union, Callable
import logging
import psutil
import os
import requests
from notification.notification_system import NotificationSystem, NotificationType, NotificationPriority

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health check status levels."""
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    UNKNOWN = "unknown"
    STALE = "stale"

class HealthCheck:
    """System for monitoring application health."""
    
    def __init__(self, notification_system: Optional[NotificationSystem] = None):
        """
        Initialize health check system.
        
        Args:
            notification_system: Optional notification system for alerts
        """
        self.checks: Dict[str, Callable] = {}
        self.last_check: Dict[str, datetime] = {}
        self.check_interval = 60  # seconds
        self.notification_system = notification_system
        
        # Register default checks
        self._register_default_checks()
        
    def _register_default_checks(self) -> None:
        """Register default health checks."""
        self.register_check('cpu_usage', self._check_cpu_usage)
        self.register_check('memory_usage', self._check_memory_usage)
        self.register_check('disk_usage', self._check_disk_usage)
        self.register_check('network_connectivity', self._check_network_connectivity)
        
    def register_check(self, name: str, check_func: Callable) -> None:
        """
        Register a health check.
        
        Args:
            name: Check name
            check_func: Function that performs the check
        """
        self.checks[name] = check_func
        self.last_check[name] = None
        logger.info(f"Registered health check: {name}")
        
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
                'status': HealthStatus.ERROR,
                'message': f'Check {name} not found',
                'timestamp': datetime.now().isoformat()
            }
            
        try:
            result = self.checks[name]()
            self.last_check[name] = datetime.now()
            
            # Send notification if status is not OK
            if result['status'] != HealthStatus.OK and self.notification_system:
                self.notification_system.add_notification(
                    type=NotificationType.ALERT,
                    priority=NotificationPriority.HIGH if result['status'] == HealthStatus.ERROR else NotificationPriority.MEDIUM,
                    title=f'Health Check Alert: {name}',
                    message=result['message'],
                    data=result
                )
            
            return {
                'status': result['status'],
                'message': result['message'],
                'timestamp': self.last_check[name].isoformat(),
                'data': result.get('data', {})
            }
        except Exception as e:
            logger.error(f"Error running health check {name}: {str(e)}")
            return {
                'status': HealthStatus.ERROR,
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
                'status': HealthStatus.ERROR,
                'message': f'Check {name} not found',
                'last_check': None
            }
            
        if not self.last_check[name]:
            return {
                'status': HealthStatus.UNKNOWN,
                'message': 'Check never run',
                'last_check': None
            }
            
        time_since_check = (datetime.now() - self.last_check[name]).total_seconds()
        
        return {
            'status': HealthStatus.OK if time_since_check < self.check_interval else HealthStatus.STALE,
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
        
    def _check_cpu_usage(self) -> Dict[str, Any]:
        """
        Check CPU usage.
        
        Returns:
            Dictionary with check results
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            status = HealthStatus.OK
            message = f'CPU usage: {cpu_percent}%'
            
            if cpu_percent > 90:
                status = HealthStatus.ERROR
                message = f'High CPU usage: {cpu_percent}%'
            elif cpu_percent > 70:
                status = HealthStatus.WARNING
                message = f'Elevated CPU usage: {cpu_percent}%'
                
            return {
                'status': status,
                'message': message,
                'data': {
                    'cpu_percent': cpu_percent,
                    'cpu_count': psutil.cpu_count(),
                    'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
                }
            }
        except Exception as e:
            logger.error(f"Error checking CPU usage: {str(e)}")
            return {
                'status': HealthStatus.ERROR,
                'message': f'Error checking CPU usage: {str(e)}'
            }
            
    def _check_memory_usage(self) -> Dict[str, Any]:
        """
        Check memory usage.
        
        Returns:
            Dictionary with check results
        """
        try:
            memory = psutil.virtual_memory()
            status = HealthStatus.OK
            message = f'Memory usage: {memory.percent}%'
            
            if memory.percent > 90:
                status = HealthStatus.ERROR
                message = f'High memory usage: {memory.percent}%'
            elif memory.percent > 70:
                status = HealthStatus.WARNING
                message = f'Elevated memory usage: {memory.percent}%'
                
            return {
                'status': status,
                'message': message,
                'data': {
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used,
                    'percent': memory.percent
                }
            }
        except Exception as e:
            logger.error(f"Error checking memory usage: {str(e)}")
            return {
                'status': HealthStatus.ERROR,
                'message': f'Error checking memory usage: {str(e)}'
            }
            
    def _check_disk_usage(self) -> Dict[str, Any]:
        """
        Check disk usage.
        
        Returns:
            Dictionary with check results
        """
        try:
            disk = psutil.disk_usage('/')
            status = HealthStatus.OK
            message = f'Disk usage: {disk.percent}%'
            
            if disk.percent > 90:
                status = HealthStatus.ERROR
                message = f'High disk usage: {disk.percent}%'
            elif disk.percent > 70:
                status = HealthStatus.WARNING
                message = f'Elevated disk usage: {disk.percent}%'
                
            return {
                'status': status,
                'message': message,
                'data': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': disk.percent
                }
            }
        except Exception as e:
            logger.error(f"Error checking disk usage: {str(e)}")
            return {
                'status': HealthStatus.ERROR,
                'message': f'Error checking disk usage: {str(e)}'
            }
            
    def _check_network_connectivity(self) -> Dict[str, Any]:
        """
        Check network connectivity.
        
        Returns:
            Dictionary with check results
        """
        try:
            # Try to connect to a reliable service
            response = requests.get('https://www.google.com', timeout=5)
            status = HealthStatus.OK
            message = 'Network connectivity OK'
            
            if response.status_code != 200:
                status = HealthStatus.WARNING
                message = f'Network connectivity warning: Status code {response.status_code}'
                
            return {
                'status': status,
                'message': message,
                'data': {
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds()
                }
            }
        except requests.RequestException as e:
            logger.error(f"Error checking network connectivity: {str(e)}")
            return {
                'status': HealthStatus.ERROR,
                'message': f'Network connectivity error: {str(e)}'
            } 