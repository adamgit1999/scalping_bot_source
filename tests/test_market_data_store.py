import pytest
import time
import os
import struct
import threading
import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from src.data.market_data_store import MarketDataStore, MarketDataPoint
from src.exceptions import MarketDataError, ValidationError, StorageError
import tempfile

@pytest.fixture
def store():
    """Create a market data store instance."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = MarketDataStore("BTC/USDT", max_points=1000)
        yield store
        store.close()

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

@pytest.fixture
def sample_update():
    """Create a sample market data update."""
    return {
        "symbol": "BTC/USDT",
        "price": 50000.0,
        "volume": 100.0,
        "bid": 49999.0,
        "ask": 50001.0,
        "last_update_id": 123456,
        "timestamp": datetime.now(timezone.utc).timestamp()
    }

def test_store_initialization(store):
    """Test store initialization."""
    assert store.symbol == "BTC/USDT"
    assert store.max_points == 1000
    assert store.data_file is not None
    assert store.mmap_file is not None
    assert len(store.data) == 0

def test_add_update(store, sample_update):
    """Test adding market data updates."""
    # Test valid update
    store.add_update(sample_update)
    assert len(store.data) == 1
    
    # Test invalid update
    with pytest.raises(ValidationError):
        store.add_update({"invalid": "data"})
    
    # Test update with invalid symbol
    invalid_update = sample_update.copy()
    invalid_update["symbol"] = "ETH/USDT"
    with pytest.raises(ValidationError):
        store.add_update(invalid_update)

def test_get_latest(store, sample_update):
    """Test retrieving latest data points."""
    # Add test data
    store.add_update(sample_update)
    
    # Test get latest
    latest = store.get_latest(1)
    assert len(latest) == 1
    assert isinstance(latest[0], MarketDataPoint)
    assert latest[0].price == sample_update["price"]
    
    # Test get more than available
    latest = store.get_latest(10)
    assert len(latest) == 1
    
    # Test get with no data
    empty_store = MarketDataStore("ETH/USDT")
    latest = empty_store.get_latest(1)
    assert len(latest) == 0
    empty_store.close()

def test_get_range(store, sample_update):
    """Test retrieving data within a time range."""
    # Add test data
    store.add_update(sample_update)
    
    # Test get range
    start_time = sample_update["timestamp"] - 1
    end_time = sample_update["timestamp"] + 1
    data_range = store.get_range(start_time, end_time)
    assert len(data_range) == 1
    assert isinstance(data_range[0], MarketDataPoint)
    
    # Test get range with no data
    data_range = store.get_range(0, 1)
    assert len(data_range) == 0

def test_get_statistics(store, sample_update):
    """Test getting data statistics."""
    # Add test data
    store.add_update(sample_update)
    
    stats = store.get_statistics()
    assert isinstance(stats, dict)
    assert "total_points" in stats
    assert "first_timestamp" in stats
    assert "last_timestamp" in stats
    assert "min_price" in stats
    assert "max_price" in stats
    assert "avg_price" in stats
    assert "total_volume" in stats

def test_cleanup_old_data(store, sample_update):
    """Test cleaning up old data."""
    # Add test data
    store.add_update(sample_update)
    
    # Force cleanup
    store._cleanup_old_data()
    assert len(store.data) <= store.max_points

def test_validate_update(store, sample_update):
    """Test update validation."""
    # Test valid update
    assert store._validate_update(sample_update)
    
    # Test missing required fields
    invalid_update = sample_update.copy()
    del invalid_update["price"]
    assert not store._validate_update(invalid_update)
    
    # Test invalid price
    invalid_update = sample_update.copy()
    invalid_update["price"] = -1
    assert not store._validate_update(invalid_update)
    
    # Test invalid volume
    invalid_update = sample_update.copy()
    invalid_update["volume"] = -1
    assert not store._validate_update(invalid_update)

def test_error_handling(store):
    """Test error handling."""
    # Test invalid update
    with pytest.raises(ValidationError):
        store.add_update({"invalid": "data"})
    
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

def test_store_cleanup(store):
    """Test store cleanup."""
    store.close()
    assert store.data_file is None
    assert store.mmap_file is None

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

def test_get_latest_data_points(store):
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

def test_get_range_data_points(store):
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

def test_get_statistics_data(store):
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