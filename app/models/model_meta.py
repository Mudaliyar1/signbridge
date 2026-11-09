from datetime import datetime, timezone
from pydantic import BaseModel, Field
import uuid

class ModelCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    version: str = Field("1.0.0", min_length=1, max_length=20)
    description: str = Field("", max_length=500)
    architecture: str = Field("Custom CNN")
    input_type: str = Field("RGB image")
    input_resolution: str = Field("224x224")

class ModelUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    architecture: str | None = None
    input_resolution: str | None = None

class SignModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    version: str = "1.0.0"
    description: str = ""
    architecture: str = "Custom CNN"
    input_type: str = "RGB image"
    input_resolution: str = "224x224"
    status: str = "Draft"  # Draft, Ready, Training, Trained, Failed
    is_active: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
