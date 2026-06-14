from pydantic import BaseModel, Field
from typing import Literal


class SupportTicket(BaseModel):
    order_id: str = Field(description="Order ID from the message, e.g. ORD123")
    issue: str = Field(description="Short description of the customer's problem")
    priority: Literal["low", "medium", "high"]
    action_needed: Literal["status", "cancel", "refund", "track"]
