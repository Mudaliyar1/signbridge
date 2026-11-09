from datetime import datetime, timezone
from pydantic import BaseModel, Field
import uuid

class TopPrediction(BaseModel):
    class_id: str
    class_name: str
    confidence: float

class TestingResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    model_version_id: str | None = None
    dataset_sample_id: str | None = None
    predicted_class_id: str
    predicted_class_name: str
    expected_class_id: str | None = None
    expected_class_name: str | None = None
    confidence: float
    is_correct: bool | None = None
    top_predictions: list[TopPrediction] = Field(default_factory=list)
    test_type: str = Field(..., pattern="^(IMAGE_UPLOAD|DATASET_SAMPLE|VIDEO_UPLOAD|LIVE_FRAME)$")
    inference_time_ms: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
