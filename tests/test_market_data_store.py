import pytest
import time
import os
import struct
import threading
import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from data.market_data_store import MarketDataStore, MarketDataPoint
from data.exceptions import MarketDataError, ValidationError, StorageError

@pytest.fixture
def store():
    """Create market data store instance."""
    store = MarketDataStore("BTC/USD", max_points=1000)
    yield store
    store.close()
    if os.path.exists(store.filename):
        os.remove(store.filename)

@pytest.fixture
def mock_market_data():
    """Create mock market data."""
    return {
        'price': Decimal('50000.0'),
        'volume': Decimal('1.5'),
        'bid': Decimal('49999.0'),
        'ask': Decimal('50001.0'),
        'last_update_id': 1,
        'timestamp': datetime.now(timezone.utc).timestamp()
    }

def test_initialization(store):
    """Test store initialization."""
    assert store.symbol == "BTC/USD"
    assert store.max_points == 1000
    assert store.current_index == 0
    assert store.last_update_id == 0
    assert store.running
    assert store.update_thread.is_alive()
    assert os.path.exists(store.filename)
    assert store.data_format == 'f'  # float format
    assert store.point_size == struct.calcsize(store.data_format * 5)  # price, volume, bid, ask, timestamp

def test_add_update(store, mock_market_data):
    """Test adding market data update."""
    store.add_update(mock_market_data)
    time.sleep(0.1)  # Give time for processing
    
    points = store.get_latest(1)
    assert len(points) == 1
    point = points[0]
    assert point.price == Decimal('50000.0')
    assert point.volume == Decimal('1.5')
    assert point.bid == Decimal('49999.0')
    assert point.ask == Decimal('50001.0')
    assert point.last_update_id == 1
    assert isinstance(point.timestamp, float)

def test_add_update_validation(store):
    """Test update validation."""
    # Test missing required fields
    with pytest.raises(ValidationError, match="Missing required field"):
        store.add_update({'price': 50000.0})
    
    # Test invalid price
    with pytest.raises(ValidationError, match="Invalid price"):
        store.add_update({
            'price': -50000.0,
            'volume': 1.5,
            'bid': 49999.0,
            'ask': 50001.0,
            'last_update_id': 1
        })
    
    # Test invalid volume
    with pytest.raises(ValidationError, match="Invalid volume"):
        store.add_update({
            'price': 50000.0,
            'volume': -1.5,
            'bid': 49999.0,
            'ask': 50001.0,
            'last_update_id': 1
        })
    
    # Test invalid bid/ask spread
    with pytest.raises(ValidationError, match="Invalid bid/ask spread"):
        store.add_update({
            'price': 50000.0,
            'volume': 1.5,
            'bid': 50002.0,  # bid > ask
            'ask': 50001.0,
            'last_update_id': 1
        })

def test_get_latest(store):
    """Test getting latest data points."""
    # Add multiple updates
    for i in range(5):
        update = {
            'price': Decimal('50000.0') + Decimal(str(i)),
            'volume': Decimal('1.0'),
            'bid': Decimal('49999.0') + Decimal(str(i)),
            'ask': Decimal('50001.0') + Decimal(str(i)),
            'last_update_id': i + 1,
            'timestamp': datetime.now(timezone.utc).timestamp()
        }
        store.add_update(update)
    
    time.sleep(0.1)  # Give time for processing
    
    # Get latest 3 points
    points = store.get_latest(3)
    assert len(points) == 3
    assert points[0].price == Decimal('50004.0')  # Most recent
    assert points[1].price == Decimal('50003.0')
    assert points[2].price == Decimal('50002.0')
    
    # Test getting more points than available
    points = store.get_latest(10)
    assert len(points) == 5  # Should return all available points

def test_get_range(store):
    """Test getting data points within time range."""
    # Add updates with different timestamps
    start_time = datetime.now(timezone.utc).timestamp()
    updates = []
    
    for i in range(5):
        update = {
            'price': Decimal('50000.0') + Decimal(str(i)),
            'volume': Decimal('1.0'),
            'bid': Decimal('49999.0') + Decimal(str(i)),
            'ask': Decimal('50001.0') + Decimal(str(i)),
            'last_update_id': i + 1,
            'timestamp': start_time + i
        }
        updates.append(update)
        store.add_update(update)
        time.sleep(0.1)
    
    end_time = start_time + 4
    
    time.sleep(0.1)  # Give time for processing
    
    # Get points within range
    points = store.get_range(start_time, end_time)
    assert len(points) == 5
    assert all(start_time <= p.timestamp <= end_time for p in points)
    
    # Test empty range
    future_time = datetime.now(timezone.utc).timestamp() + 3600
    points = store.get_range(future_time, future_time + 1)
    assert len(points) == 0
    
    # Test invalid range
    with pytest.raises(ValidationError, match="Invalid time range"):
        store.get_range(end_time, start_time)

def test_get_statistics(store):
    """Test getting market data statistics."""
    # Add updates with varying prices and volumes
    for i in range(10):
        update = {
            'price': Decimal('50000.0') + Decimal(str(i)),
            'volume': Decimal('1.0') + Decimal(str(i * 0.1)),
            'bid': Decimal('49999.0') + Decimal(str(i)),
            'ask': Decimal('50001.0') + Decimal(str(i)),
            'last_update_id': i + 1,
            'timestamp': datetime.now(timezone.utc).timestamp()
        }
        store.add_update(update)
    
    time.sleep(0.1)  # Give time for processing
    
    stats = store.get_statistics()
    assert 'price' in stats
    assert 'volume' in stats
    assert 'spread' in stats
    assert 'updates' in stats
    
    assert stats['price']['current'] == Decimal('50009.0')
    assert stats['price']['min'] == Decimal('50000.0')
    assert stats['price']['max'] == Decimal('50009.0')
    assert stats['price']['avg'] == Decimal('50004.5')
    
    assert stats['volume']['current'] == Decimal('1.9')
    assert stats['volume']['total'] == Decimal('14.5')
    assert stats['volume']['avg'] == Decimal('1.45')
    
    assert stats['spread']['current'] == Decimal('2.0')
    assert stats['spread']['min'] == Decimal('2.0')
    assert stats['spread']['max'] == Decimal('2.0')
    assert stats['spread']['avg'] == Decimal('2.0')
    
    assert stats['updates']['total'] == 10
    assert stats['updates']['rate'] > 0

def test_update_id_ordering(store):
    """Test update ID ordering."""
    # Add updates out of order
    updates = [
        {'price': Decimal('50000.0'), 'volume': Decimal('1.0'), 'bid': Decimal('49999.0'), 
         'ask': Decimal('50001.0'), 'last_update_id': 3, 'timestamp': time.time()},
        {'price': Decimal('50001.0'), 'volume': Decimal('1.0'), 'bid': Decimal('50000.0'), 
         'ask': Decimal('50002.0'), 'last_update_id': 1, 'timestamp': time.time()},
        {'price': Decimal('50002.0'), 'volume': Decimal('1.0'), 'bid': Decimal('50001.0'), 
         'ask': Decimal('50003.0'), 'last_update_id': 2, 'timestamp': time.time()}
    ]
    
    for update in updates:
        store.add_update(update)
    
    time.sleep(0.1)  # Give time for processing
    
    points = store.get_latest(3)
    assert len(points) == 3
    assert points[0].last_update_id == 3  # Most recent
    assert points[1].last_update_id == 2
    assert points[2].last_update_id == 1
    
    # Test duplicate update ID
    with pytest.raises(ValidationError, match="Duplicate update ID"):
        store.add_update(updates[0])

def test_circular_buffer(store):
    """Test circular buffer behavior."""
    # Add more updates than max_points
    for i in range(store.max_points + 5):
        update = {
            'price': Decimal('50000.0') + Decimal(str(i)),
            'volume': Decimal('1.0'),
            'bid': Decimal('49999.0') + Decimal(str(i)),
            'ask': Decimal('50001.0') + Decimal(str(i)),
            'last_update_id': i + 1,
            'timestamp': datetime.now(timezone.utc).timestamp()
        }
        store.add_update(update)
    
    time.sleep(0.1)  # Give time for processing
    
    # Should only have max_points number of points
    points = store.get_latest(store.max_points + 1)
    assert len(points) == store.max_points
    
    # Most recent points should be present
    assert points[0].price == Decimal('50004.0') + Decimal(str(store.max_points))
    assert points[-1].price == Decimal('50005.0')
    
    # Test buffer wrap-around
    assert store.current_index == 5  # Should have wrapped around

def test_concurrent_access(store):
    """Test concurrent access to store."""
    def add_updates():
        for i in range(100):
            update = {
                'price': Decimal('50000.0') + Decimal(str(i)),
                'volume': Decimal('1.0'),
                'bid': Decimal('49999.0') + Decimal(str(i)),
                'ask': Decimal('50001.0') + Decimal(str(i)),
                'last_update_id': i + 1,
                'timestamp': datetime.now(timezone.utc).timestamp()
            }
            store.add_update(update)
    
    # Start multiple threads adding updates
    threads = [threading.Thread(target=add_updates) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    time.sleep(0.1)  # Give time for processing
    
    # Check that all updates were processed
    points = store.get_latest(100)
    assert len(points) == 100
    assert all(p.last_update_id > 0 for p in points)
    
    # Verify no duplicate update IDs
    update_ids = [p.last_update_id for p in points]
    assert len(update_ids) == len(set(update_ids))

def test_error_handling(store):
    """Test error handling."""
    # Test invalid update
    with pytest.raises(ValidationError):
        store.add_update({'invalid': 'data'})
    
    # Test file access error
    store.close()
    with pytest.raises(StorageError):
        store.get_latest(1)
    
    # Test invalid max_points
    with pytest.raises(ValidationError, match="Invalid max_points"):
        MarketDataStore("BTC/USD", max_points=0)
    
    # Test invalid symbol
    with pytest.raises(ValidationError, match="Invalid symbol"):
        MarketDataStore("", max_points=1000)

def test_cleanup(store):
    """Test cleanup on deletion."""
    store.close()
    assert not store.running
    assert not store.update_thread.is_alive()
    
    # Test file is closed
    with pytest.raises(StorageError):
        store.mmap.read(1)
    
    # Test can't add updates after close
    with pytest.raises(StorageError):
        store.add_update({
            'price': Decimal('50000.0'),
            'volume': Decimal('1.0'),
            'bid': Decimal('49999.0'),
            'ask': Decimal('50001.0'),
            'last_update_id': 1,
            'timestamp': datetime.now(timezone.utc).timestamp()
        })

def test_market_data_point(store):
    """Test MarketDataPoint class."""
    point = MarketDataPoint(
        price=Decimal('50000.0'),
        volume=Decimal('1.0'),
        bid=Decimal('49999.0'),
        ask=Decimal('50001.0'),
        timestamp=datetime.now(timezone.utc).timestamp(),
        last_update_id=1
    )
    
    assert point.price == Decimal('50000.0')
    assert point.volume == Decimal('1.0')
    assert point.bid == Decimal('49999.0')
    assert point.ask == Decimal('50001.0')
    assert point.spread == Decimal('2.0')
    assert point.last_update_id == 1
    assert isinstance(point.timestamp, float)
    
    # Test string representation
    assert str(point) == f"MarketDataPoint(price=50000.0, volume=1.0, bid=49999.0, ask=50001.0, timestamp={point.timestamp}, last_update_id=1)"
    
    # Test equality
    point2 = MarketDataPoint(
        price=Decimal('50000.0'),
        volume=Decimal('1.0'),
        bid=Decimal('49999.0'),
        ask=Decimal('50001.0'),
        timestamp=point.timestamp,
        last_update_id=1
    )
    assert point == point2
    
    # Test inequality
    point3 = MarketDataPoint(
        price=Decimal('50001.0'),
        volume=Decimal('1.0'),
        bid=Decimal('50000.0'),
        ask=Decimal('50002.0'),
        timestamp=point.timestamp,
        last_update_id=2
    )
    assert point != point3 