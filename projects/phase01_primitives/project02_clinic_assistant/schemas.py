from pydantic import BaseModel, Field
from typing import Literal, Optional


class AppointmentRequest(BaseModel):
    patient_name: Optional[str] = Field(None, description="Full name of the patient, if mentioned")
    reason: str = Field(description="Reason for the appointment or query")
    urgency: Literal["routine", "urgent", "emergency"]
    action_needed: Literal["check", "book", "cancel", "info"]
