import mmap
import struct
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from dataclasses import dataclass
import threading
from queue import Queue
import time
import os
import json
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class MarketDataPoint:
    """Market data point structure."""
    timestamp: float
    price: float
    volume: float
    bid: float
    ask: float
    last_update_id: int

class MarketDataStore:
    """Memory-mapped market data store for fast access."""
    
    def __init__(self, symbol: str, max_points: int = 1000000):
        """
        Initialize market data store.
        
        Args:
            symbol: Trading symbol
            max_points: Maximum number of data points to store
        """
        if not symbol:
            raise ValidationError("Invalid symbol")
        if max_points <= 0:
            raise ValidationError("Invalid max_points")
            
        self.symbol = symbol
        self.max_points = max_points
        self.point_size = struct.calcsize('ddfddi')  # Size of MarketDataPoint
        self.total_size = self.point_size * max_points
        
        # Create memory-mapped file
        self.filename = f"market_data_{symbol}.mmap"
        self._create_mmap_file()
        
        # Initialize data structures
        self.current_index = 0
        self.lock = threading.Lock()
        self.update_queue = Queue(maxsize=10000)  # Limit queue size
        self.last_update_id = 0
        self._cleanup_interval = 3600  # 1 hour
        self._last_cleanup = time.time()
        
        # Start update thread
        self.running = True
        self.update_thread = threading.Thread(target=self._process_updates, daemon=True)
        self.update_thread.start()
        
    def _create_mmap_file(self) -> None:
        """Create memory-mapped file."""
        try:
            # Create file if it doesn't exist
            if not os.path.exists(self.filename):
                with open(self.filename, 'wb') as f:
                    f.write(b'\0' * self.total_size)
                    
            # Open file for reading and writing
            self.file = open(self.filename, 'r+b')
            self.mmap = mmap.mmap(self.file.fileno(), 0)
            
        except Exception as e:
            logger.error(f"Error creating memory-mapped file: {str(e)}")
            raise StorageError(f"Failed to create market data store: {str(e)}")
            
    def _process_updates(self) -> None:
        """Process market data updates with cleanup."""
        while self.running:
            try:
                # Check if cleanup is needed
                current_time = time.time()
                if current_time - self._last_cleanup > self._cleanup_interval:
                    self._cleanup_old_data()
                    self._last_cleanup = current_time
                
                # Get update from queue with timeout
                try:
                    update = self.update_queue.get(timeout=0.1)
                    self._apply_update(update)
                except Empty:
                    continue
                    
            except Exception as e:
                logger.error(f"Error processing update: {str(e)}")
                time.sleep(0.1)  # Prevent tight loop on error
                
    def _cleanup_old_data(self) -> None:
        """Clean up old data to prevent memory issues."""
        try:
            with self.lock:
                # Calculate cutoff time (24 hours ago)
                cutoff_time = time.time() - 86400
                
                # Find new start index
                new_start = 0
                for i in range(self.max_points):
                    index = (self.current_index - i - 1) % self.max_points
                    self.mmap.seek(index * self.point_size)
                    data = self.mmap.read(self.point_size)
                    if data:
                        point = MarketDataPoint(*struct.unpack('ddfddi', data))
                        if point.timestamp >= cutoff_time:
                            new_start = index
                            break
                
                # Clear old data
                if new_start > 0:
                    self.mmap.seek(0)
                    self.mmap.write(b'\0' * (new_start * self.point_size))
                    
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
            
    def _apply_update(self, update: Dict[str, Any]) -> None:
        """
        Apply market data update.
        
        Args:
            update: Market data update
        """
        with self.lock:
            # Check if update is newer
            if update['last_update_id'] <= self.last_update_id:
                return
                
            # Create data point
            point = MarketDataPoint(
                timestamp=time.time(),
                price=float(update['price']),
                volume=float(update['volume']),
                bid=float(update['bid']),
                ask=float(update['ask']),
                last_update_id=update['last_update_id']
            )
            
            # Pack data point
            data = struct.pack('ddfddi',
                point.timestamp,
                point.price,
                point.volume,
                point.bid,
                point.ask,
                point.last_update_id
            )
            
            # Write to memory-mapped file
            self.mmap.seek(self.current_index * self.point_size)
            self.mmap.write(data)
            
            # Update index
            self.current_index = (self.current_index + 1) % self.max_points
            self.last_update_id = update['last_update_id']
            
    def add_update(self, update: Dict[str, Any]) -> None:
        """
        Add market data update with validation.
        
        Args:
            update: Market data update
        """
        # Validate update
        if not self._validate_update(update):
            raise ValidationError("Invalid market data update")
            
        # Check queue size
        if self.update_queue.qsize() >= self.update_queue.maxsize:
            logger.warning("Market data update queue is full, dropping oldest update")
            try:
                self.update_queue.get_nowait()  # Remove oldest update
            except Empty:
                pass
                
        # Add update to queue
        self.update_queue.put(update)
        
    def _validate_update(self, update: Dict[str, Any]) -> bool:
        """Validate market data update."""
        try:
            # Check required fields
            required_fields = ['price', 'volume', 'bid', 'ask', 'last_update_id']
            if not all(field in update for field in required_fields):
                return False
                
            # Validate values
            if not (update['price'] > 0 and 
                   update['volume'] >= 0 and 
                   update['bid'] > 0 and 
                   update['ask'] > 0 and 
                   update['bid'] <= update['ask']):
                return False
                
            # Check update ID ordering
            if update['last_update_id'] <= self.last_update_id:
                return False
                
            return True
            
        except Exception:
            return False
            
    def get_latest(self, n: int = 1) -> List[MarketDataPoint]:
        """
        Get latest market data points.
        
        Args:
            n: Number of points to retrieve
            
        Returns:
            List of market data points
        """
        with self.lock:
            points = []
            for i in range(min(n, self.max_points)):
                index = (self.current_index - i - 1) % self.max_points
                self.mmap.seek(index * self.point_size)
                data = self.mmap.read(self.point_size)
                if data:
                    point = MarketDataPoint(*struct.unpack('ddfddi', data))
                    points.append(point)
            return points
            
    def get_range(self, start_time: float, end_time: float) -> List[MarketDataPoint]:
        """
        Get market data points within time range.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            List of market data points
        """
        with self.lock:
            points = []
            for i in range(self.max_points):
                index = (self.current_index - i - 1) % self.max_points
                self.mmap.seek(index * self.point_size)
                data = self.mmap.read(self.point_size)
                if data:
                    point = MarketDataPoint(*struct.unpack('ddfddi', data))
                    if start_time <= point.timestamp <= end_time:
                        points.append(point)
                    elif point.timestamp < start_time:
                        break
            return points
            
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get market data statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self.lock:
            points = self.get_latest(1000)  # Get last 1000 points
            if not points:
                return {}
                
            prices = [p.price for p in points]
            volumes = [p.volume for p in points]
            spreads = [p.ask - p.bid for p in points]
            
            return {
                'price': {
                    'current': prices[0],
                    'average': np.mean(prices),
                    'min': min(prices),
                    'max': max(prices),
                    'std': np.std(prices)
                },
                'volume': {
                    'current': volumes[0],
                    'average': np.mean(volumes),
                    'total': sum(volumes)
                },
                'spread': {
                    'current': spreads[0],
                    'average': np.mean(spreads),
                    'min': min(spreads),
                    'max': max(spreads)
                },
                'updates': {
                    'total': self.last_update_id,
                    'rate': len(points) / (points[0].timestamp - points[-1].timestamp)
                }
            }
            
    def close(self) -> None:
        """Close market data store and cleanup resources."""
        try:
            self.running = False
            if hasattr(self, 'update_thread') and self.update_thread:
                self.update_thread.join(timeout=5.0)
            if hasattr(self, 'mmap') and self.mmap:
                self.mmap.close()
            if hasattr(self, 'file') and self.file:
                self.file.close()
            # Remove mmap file
            if os.path.exists(self.filename):
                os.remove(self.filename)
        except Exception as e:
            logger.error(f"Error closing market data store: {str(e)}")
            
    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass 