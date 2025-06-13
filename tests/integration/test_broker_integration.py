import pytest
from decimal import Decimal
from src.broker.broker import Broker, Order

@pytest.fixture
def broker():
    return Broker(api_key="test_key", api_secret="test_secret", test_mode=True)

def test_place_order(broker):
    order_id = broker.place_order(
        symbol="BTC/USD",
        quantity=Decimal("0.1"),
        price=Decimal("50000"),
        order_type="LIMIT"
    )
    assert order_id is not None
    assert order_id in broker.orders
    assert broker.orders[order_id].symbol == "BTC/USD"
    assert broker.orders[order_id].quantity == Decimal("0.1")
    assert broker.orders[order_id].price == Decimal("50000")
    assert broker.orders[order_id].order_type == "LIMIT"
    assert broker.orders[order_id].status == "PENDING"

def test_cancel_order(broker):
    order_id = broker.place_order(
        symbol="ETH/USD",
        quantity=Decimal("1.0"),
        price=Decimal("3000"),
        order_type="LIMIT"
    )
    assert broker.cancel_order(order_id) is True
    assert broker.orders[order_id].status == "CANCELLED"

def test_cancel_nonexistent_order(broker):
    assert broker.cancel_order("nonexistent_order") is False

def test_get_order_status(broker):
    order_id = broker.place_order(
        symbol="BTC/USD",
        quantity=Decimal("0.1"),
        price=Decimal("50000"),
        order_type="LIMIT"
    )
    assert broker.get_order_status(order_id) == "PENDING"
    broker.cancel_order(order_id)
    assert broker.get_order_status(order_id) == "CANCELLED"
    assert broker.get_order_status("nonexistent_order") is None

def test_position_management(broker):
    symbol = "BTC/USD"
    assert broker.get_position(symbol) == Decimal("0")
    
    broker.update_position(symbol, Decimal("0.5"))
    assert broker.get_position(symbol) == Decimal("0.5")
    
    broker.update_position(symbol, Decimal("-0.2"))
    assert broker.get_position(symbol) == Decimal("0.3")

def test_balance_management(broker):
    assert broker.get_balance() == Decimal("0")
    
    broker.update_balance(Decimal("1000"))
    assert broker.get_balance() == Decimal("1000")
    
    broker.update_balance(Decimal("-500"))
    assert broker.get_balance() == Decimal("500")

def test_multiple_orders(broker):
    # Place multiple orders
    order_ids = []
    symbols = ["BTC/USD", "ETH/USD", "LTC/USD"]
    for i, symbol in enumerate(symbols):
        order_id = broker.place_order(
            symbol=symbol,
            quantity=Decimal(str(i + 1)),
            price=Decimal(str(100 * (i + 1))),
            order_type="LIMIT"
        )
        order_ids.append(order_id)
    
    # Verify all orders were placed
    assert len(broker.orders) == 3
    for i, order_id in enumerate(order_ids):
        assert broker.orders[order_id].symbol == symbols[i]
        assert broker.orders[order_id].quantity == Decimal(str(i + 1))
        assert broker.orders[order_id].price == Decimal(str(100 * (i + 1))) 