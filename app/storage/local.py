import os
import uuid
from pathlib import Path
from app.config import settings

class StorageService:
    @staticmethod
    def get_sample_dir(model_id: str) -> Path:
        """Returns the directory for a specific model's dataset samples."""
        directory = settings.DATASETS_DIR / "models" / model_id / "samples"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def save_sample_file(model_id: str, content: bytes, original_filename: str, media_type: str) -> tuple[str, Path]:
        """
        Saves content to disk using a safe server-side UUID filename.
        Returns:
            (relative_storage_path, absolute_file_path)
        """
        ext = Path(original_filename).suffix.lower()
        if not ext:
            ext = ".jpg" if media_type == "IMAGE" else ".webm"
            
        # Security validation on extension
        allowed = settings.ALLOWED_IMAGE_EXTS if media_type == "IMAGE" else settings.ALLOWED_VIDEO_EXTS
        if ext not in allowed:
            # Fallback to standard
            ext = ".jpg" if media_type == "IMAGE" else ".webm"

        file_id = str(uuid.uuid4())
        safe_filename = f"{file_id}{ext}"
        
        target_dir = StorageService.get_sample_dir(model_id)
        target_path = target_dir / safe_filename
        
        # Write bytes
        with open(target_path, "wb") as f:
            f.write(content)
            
        relative_path = f"datasets/models/{model_id}/samples/{safe_filename}"
        return relative_path, target_path

    @staticmethod
    def resolve_path(relative_storage_path: str) -> Path:
        """Resolves relative storage path to absolute path safely."""
        clean_path = Path(relative_storage_path)
        if clean_path.is_absolute():
            clean_path = Path(*clean_path.parts[1:])  # remove leading slash if any
        full_path = (settings.STORAGE_DIR / clean_path).resolve()
        
        # Verify path traversal prevention
        if not str(full_path).startswith(str(settings.STORAGE_DIR.resolve())):
            raise ValueError("Path traversal attempt detected")
            
        return full_path

    @staticmethod
    def delete_sample_file(relative_storage_path: str) -> bool:
        """Safely removes physical media file from storage."""
        try:
            full_path = StorageService.resolve_path(relative_storage_path)
            if full_path.exists() and full_path.is_file():
                full_path.unlink()
                return True
        except Exception as e:
            pass
        return False
