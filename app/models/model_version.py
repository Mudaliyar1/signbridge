from datetime import datetime, timezone
from pydantic import BaseModel, Field
import uuid

class ModelVersion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    version: str = "v1.0.0"
    architecture: str = "CustomSignCNN"
    weights_path: str
    num_classes: int
    class_mapping: dict[str, str] = Field(default_factory=dict)  # index_str -> class_name
    class_id_mapping: dict[str, str] = Field(default_factory=dict)  # index_str -> class_id
    training_run_id: str | None = None
    input_resolution: str = "224x224"
    train_samples_count: int = 0
    val_accuracy: float | None = None
    status: str = "AVAILABLE"  # AVAILABLE, ARCHIVED
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
