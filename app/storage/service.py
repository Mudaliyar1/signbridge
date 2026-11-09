import logging
import tempfile
from pathlib import Path
from typing import Any
from app.storage.cloudinary import CloudinaryStorage
from app.storage.local import StorageService
from app.ml.preprocessing import validate_and_inspect_image, validate_and_inspect_video

logger = logging.getLogger("signbridge.storage_service")

class StorageServiceManager:
    @classmethod
    def process_and_upload_image(
        cls,
        image_bytes: bytes,
        model_id: str,
        class_id: str,
        original_filename: str
    ) -> dict[str, Any]:
        """
        1. Validates image bytes with PIL & OpenCV.
        2. Uploads to Cloudinary (resource_type='image').
        3. Returns Cloudinary metadata dictionary.
        """
        # Server-side validation
        width, height, fmt = validate_and_inspect_image(image_bytes)
        
        # Upload to Cloudinary
        cloud_meta = CloudinaryStorage.upload_image(
            file_input=image_bytes,
            model_id=model_id,
            class_id=class_id
        )
        
        # Merge validated dimensions
        cloud_meta["width"] = cloud_meta.get("width") or width
        cloud_meta["height"] = cloud_meta.get("height") or height
        return cloud_meta

    @classmethod
    def process_and_upload_video(
        cls,
        video_bytes: bytes,
        model_id: str,
        class_id: str,
        original_filename: str,
        client_duration: float = 0.0
    ) -> dict[str, Any]:
        """
        1. Writes to temp file to allow OpenCV VideoCapture inspection.
        2. Validates video stream and extracts dimensions & duration.
        3. Uploads to Cloudinary (resource_type='video').
        4. Cleans up temp file.
        5. Returns Cloudinary metadata dictionary.
        """
        ext = Path(original_filename).suffix.lower() or ".webm"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = Path(tmp.name)

        try:
            width, height, computed_dur, frames = validate_and_inspect_video(tmp_path)
            duration = computed_dur if computed_dur > 0 else client_duration
            
            # Upload to Cloudinary
            cloud_meta = CloudinaryStorage.upload_video(
                file_input=str(tmp_path),
                model_id=model_id,
                class_id=class_id
            )
            cloud_meta["width"] = cloud_meta.get("width") or width
            cloud_meta["height"] = cloud_meta.get("height") or height
            cloud_meta["duration"] = cloud_meta.get("duration") or duration
            return cloud_meta
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    @classmethod
    def rollback_cloudinary(cls, public_id: str, resource_type: str = "image"):
        """Compensating transaction: destroys uploaded Cloudinary asset if MongoDB insertion fails."""
        try:
            logger.warning(f"Initiating rollback: deleting Cloudinary asset '{public_id}' ({resource_type})...")
            success = CloudinaryStorage.delete_asset(public_id, resource_type)
            if success:
                logger.info(f"Rollback successful: '{public_id}' destroyed.")
            else:
                logger.error(f"CRITICAL: Rollback failed to destroy Cloudinary asset '{public_id}'. Manual cleanup required.")
        except Exception as e:
            logger.error(f"CRITICAL: Exception during Cloudinary rollback: {e}")

    @classmethod
    def delete_sample_media(cls, sample_doc: dict[str, Any]) -> bool:
        """
        Deletes physical media corresponding to a dataset sample document.
        Handles both Cloudinary assets and legacy local storage paths.
        """
        success = True
        
        # 1. Cloudinary asset deletion
        cloudinary_data = sample_doc.get("cloudinary")
        if cloudinary_data and "public_id" in cloudinary_data:
            public_id = cloudinary_data["public_id"]
            res_type = cloudinary_data.get("resource_type", "image")
            c_success = CloudinaryStorage.delete_asset(public_id, res_type)
            if not c_success:
                logger.warning(f"Could not destroy Cloudinary asset '{public_id}'")
                success = False

        # 2. Legacy local file deletion if present
        if "storage_path" in sample_doc and sample_doc["storage_path"]:
            StorageService.delete_sample_file(sample_doc["storage_path"])

        return success
