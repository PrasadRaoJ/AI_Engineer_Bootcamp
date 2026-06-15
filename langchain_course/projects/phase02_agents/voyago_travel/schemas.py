"""
Voyago Travel Booking Assistant
schemas.py: Pydantic models for structured output and context injection.
"""
from pydantic import BaseModel
from typing import Literal


class Context(BaseModel):
    user_id: str
    loyalty_tier: Literal["basic", "silver", "gold", "vip"]


class BookingConfirmation(BaseModel):
    booking_id: str
    type: Literal["flight", "hotel"]
    passenger_name: str
    details: str    # "HYD → BLR, 20 Jun, 08:30"
    amount: float
    status: Literal["confirmed", "pending", "cancelled"]
