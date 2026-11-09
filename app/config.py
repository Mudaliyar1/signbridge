import os
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    APP_NAME: str = "SignBridge"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    BASE_DIR: Path = BASE_DIR
    
    # MongoDB Atlas Configuration
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "signbridge"
    
    # Cloudinary Configuration
    CLOUDINARY_CLOUD_NAME: str = ""
    CLOUDINARY_API_KEY: str = ""
    CLOUDINARY_API_SECRET: str = ""
    CLOUDINARY_FOLDER_PREFIX: str = "signbridge"
    
    # Storage Configuration (Local cache/temp and model weights)
    STORAGE_DIR: Path = BASE_DIR / "storage"
    DATASETS_DIR: Path = BASE_DIR / "storage" / "datasets"
    MODELS_DIR: Path = BASE_DIR / "storage" / "models"
    
    # Media & Upload Validation
    MAX_IMAGE_SIZE_MB: int = 20
    MAX_VIDEO_SIZE_MB: int = 100
    ALLOWED_IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".webp"}
    ALLOWED_VIDEO_EXTS: set[str] = {".mp4", ".webm", ".mov", ".mkv"}
    
    # Image preprocessing resolution
    TARGET_IMAGE_SIZE: tuple[int, int] = (224, 224)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()

# Ensure directories exist
settings.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
