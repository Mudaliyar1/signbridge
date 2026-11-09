from datetime import datetime, timezone
from pydantic import BaseModel, Field

class ParticipantCreate(BaseModel):
    participant_id: str = Field(..., min_length=2, max_length=50)
    display_name: str = Field(..., min_length=2, max_length=100)
    consent_status: str = Field("GRANTED", pattern="^(GRANTED|PENDING|REVOKED)$")
    notes: str = Field("", max_length=500)

class Participant(BaseModel):
    participant_id: str
    display_name: str
    consent_status: str = "GRANTED"
    notes: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
