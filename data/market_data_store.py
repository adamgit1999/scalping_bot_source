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
        self.update_queue = Queue()
        self.last_update_id = 0
        
        # Start update thread
        self.running = True
        self.update_thread = threading.Thread(target=self._process_updates)
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
            raise
            
    def _process_updates(self) -> None:
        """Process market data updates."""
        while self.running:
            try:
                # Get update from queue
                update = self.update_queue.get(timeout=0.1)
                self._apply_update(update)
            except Exception as e:
                logger.error(f"Error processing update: {str(e)}")
                
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
        Add market data update.
        
        Args:
            update: Market data update
        """
        self.update_queue.put(update)
        
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
        """Close market data store."""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
        if self.mmap:
            self.mmap.close()
        if self.file:
            self.file.close()
            
    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close() 