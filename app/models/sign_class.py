from datetime import datetime, timezone
from pydantic import BaseModel, Field
import uuid

class SignClassCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    output_text: str = Field("", max_length=100)
    description: str = Field("", max_length=300)
    language: str = Field("ISL", max_length=20)
    category: str = Field("General", max_length=50)
    sign_type: str = Field("STATIC", pattern="^(STATIC|DYNAMIC)$")

class SignClass(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    name: str
    output_text: str = ""
    description: str = ""
    language: str = "ISL"
    category: str = "General"
    sign_type: str = "STATIC"  # STATIC or DYNAMIC
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
