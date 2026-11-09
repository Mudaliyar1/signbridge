from datetime import datetime, timezone
from pydantic import BaseModel, Field
import uuid

class MLExperiment(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    algorithm: str  # KNN, SVM, DECISION_TREE, CUSTOM_CNN, KMEANS
    dataset_snapshot: dict = Field(default_factory=dict)
    split_info: dict = Field(default_factory=dict)
    hyperparameters: dict = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict)
    status: str = "COMPLETED"  # COMPLETED, FAILED
    error_message: str | None = None
    artifact_path: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
