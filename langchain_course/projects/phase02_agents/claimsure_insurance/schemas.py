"""
ClaimSure Insurance Claims Agent
schemas.py: Context (per-call injection) and ClaimReport (structured output).
"""
from pydantic import BaseModel
from typing import Literal, List


class Context(BaseModel):
    user_id: str
    role: Literal["customer", "adjuster"]


class ClaimReport(BaseModel):
    claim_id: str
    policyholder_id: str
    incident_type: Literal["motor_accident", "theft", "health", "property", "travel"]
    incident_date: str          # YYYY-MM-DD
    amount: float
    description: str
    status: Literal["filed", "under_review", "approved", "rejected", "fraud_flagged"]
    documents_received: List[str]
