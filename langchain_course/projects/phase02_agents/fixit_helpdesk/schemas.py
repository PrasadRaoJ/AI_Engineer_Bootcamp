"""
FixIt IT Helpdesk Agent
schemas.py: Pydantic models for context injection and ticket classification.
"""
from pydantic import BaseModel
from typing import Literal


class Context(BaseModel):
    employee_id: str
    role: Literal["employee", "it_admin"]
    department: str


class TicketClassification(BaseModel):
    issue_type: Literal["password", "vpn", "software", "access", "hardware", "other"]
    urgency: Literal["low", "medium", "high", "critical"]
    self_serviceable: bool    # can be resolved without human approval
    requires_admin: bool      # needs grant_admin_access, reset_mfa, or wipe_device
