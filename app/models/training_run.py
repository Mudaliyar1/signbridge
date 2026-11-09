from datetime import datetime, timezone
from pydantic import BaseModel, Field
import uuid

class EpochMetric(BaseModel):
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float | None = None
    val_acc: float | None = None

class TrainingRun(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    version: str = "v1.0.0"
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 0.001
    optimizer: str = "Adam"
    loss_function: str = "CrossEntropyLoss"
    total_samples: int = 0
    train_samples: int = 0
    val_samples: int = 0
    num_classes: int = 0
    metrics: list[EpochMetric] = Field(default_factory=list)
    final_train_acc: float | None = None
    final_val_acc: float | None = None
    error_message: str | None = None
    weights_path: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
