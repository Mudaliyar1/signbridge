from datetime import datetime, timezone
from pydantic import BaseModel, Field
import uuid

class CloudinaryMetadata(BaseModel):
    public_id: str
    resource_type: str = "image"  # "image" or "video"
    secure_url: str
    format: str = ""
    width: int = 0
    height: int = 0
    bytes: int = 0
    duration: float = 0.0

class DatasetSample(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    class_id: str
    participant_id: str
    media_type: str = Field(..., pattern="^(IMAGE|VIDEO)$")
    storage_provider: str = "cloudinary"  # "cloudinary" or "local"
    storage_path: str = ""                 # legacy or local fallback path
    cloudinary: CloudinaryMetadata | None = None
    original_filename: str = ""
    mime_type: str = ""
    file_size: int = 0
    width: int = 0
    height: int = 0
    duration: float = 0.0  # seconds
    label: str
    language: str = "ISL"
    notes: str = ""
    status: str = "VALID"  # VALID, REVIEW_REQUIRED, INVALID
    split: str = Field("TRAIN", pattern="^(TRAIN|VALIDATION|TEST)$")
    handedness: str | None = None  # e.g. "Right", "Left", "Auto"
    hand_detection_confidence: float | None = None
    hand_bbox: list[float] | None = None  # [ymin, xmin, ymax, xmax] normalized
    landmarks: list[dict[str, float]] | None = None  # 21 MediaPipe 3D coordinates (x, y, z)
    preprocessing_version: str = "hand_landmark_v1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def media_url(self) -> str:
        """Returns direct Cloudinary HTTPS URL or legacy local proxy URL."""
        if self.cloudinary and self.cloudinary.secure_url:
            return self.cloudinary.secure_url
        if self.storage_path:
            return f"/api/media/{self.storage_path}"
        return ""
