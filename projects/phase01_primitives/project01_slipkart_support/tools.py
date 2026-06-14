from langchain_core.tools import tool


ORDERS = {
    "ORD123": {"status": "Out for delivery. Expected by 6 PM today.", "tracking": "Left Mumbai warehouse at 9 AM. Currently in transit to Delhi.", "amount": 1299},
    "ORD456": {"status": "Delivered on 12 Jun 2026.", "tracking": "Delivered to front door at 2:34 PM.", "amount": 3499},
    "ORD789": {"status": "Delayed. New expected date: 16 Jun 2026.", "tracking": "Stuck at Pune sorting center due to weather.", "amount": 899},
}


@tool
def get_order_status(order_id: str) -> str:
    """Returns the current delivery status of a Slipkart order given its order ID."""
    order = ORDERS.get(order_id)
    return order["status"] if order else f"Order {order_id} not found."


@tool
def cancel_order(order_id: str) -> str:
    """Cancels a Slipkart order and initiates a refund."""
    if order_id in ORDERS:
        amount = ORDERS[order_id]["amount"]
        return f"Order {order_id} has been cancelled. Refund of ₹{amount} will be credited in 3-5 business days."
    return f"Order {order_id} not found. Cannot cancel."


@tool
def raise_refund(order_id: str, reason: str) -> str:
    """Raises a refund request for a Slipkart order given the order ID and reason."""
    if order_id in ORDERS:
        amount = ORDERS[order_id]["amount"]
        return f"Refund request of ₹{amount} raised for order {order_id}. Reason: {reason}. You will hear from us in 24-48 hours."
    return f"Order {order_id} not found. Cannot raise refund."


@tool
def track_delivery(order_id: str) -> str:
    """Returns live tracking information for a Slipkart order."""
    order = ORDERS.get(order_id)
    return order["tracking"] if order else f"Tracking info not available for order {order_id}."


ALL_TOOLS = [get_order_status, cancel_order, raise_refund, track_delivery]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}
