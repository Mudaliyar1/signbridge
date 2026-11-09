import logging
import uuid
from typing import Any
import cloudinary
import cloudinary.uploader
import cloudinary.api
from app.config import settings

logger = logging.getLogger("signbridge.cloudinary")

def init_cloudinary():
    """Initializes Cloudinary configuration from settings."""
    if settings.CLOUDINARY_CLOUD_NAME and settings.CLOUDINARY_API_KEY and settings.CLOUDINARY_API_SECRET:
        cloudinary.config(
            cloud_name=settings.CLOUDINARY_CLOUD_NAME,
            api_key=settings.CLOUDINARY_API_KEY,
            api_secret=settings.CLOUDINARY_API_SECRET,
            secure=True
        )
        logger.info(f"Cloudinary configured for cloud: '{settings.CLOUDINARY_CLOUD_NAME}'")
    else:
        logger.warning("Cloudinary credentials are not fully configured in settings.")

# Initialize on module load
init_cloudinary()

class CloudinaryStorage:
    @staticmethod
    def get_folder_path(model_id: str, media_type: str, class_id: str | None = None) -> str:
        """
        Builds predictable folder structure per specification:
        signbridge/models/{model_id}/samples/{images|videos}/[{class_id}/]
        """
        prefix = settings.CLOUDINARY_FOLDER_PREFIX.strip("/")
        subfolder = "images" if media_type.upper() == "IMAGE" else "videos"
        if class_id:
            return f"{prefix}/models/{model_id}/samples/{subfolder}/{class_id}"
        return f"{prefix}/models/{model_id}/samples/{subfolder}"

    @classmethod
    def upload_image(
        cls,
        file_input: Any,
        model_id: str,
        class_id: str | None = None,
        public_id_suffix: str | None = None
    ) -> dict[str, Any]:
        """
        Uploads image to Cloudinary with resource_type='image'.
        Returns normalized metadata dictionary.
        """
        folder = cls.get_folder_path(model_id, "IMAGE", class_id)
        asset_id = public_id_suffix or str(uuid.uuid4())
        
        try:
            res = cloudinary.uploader.upload(
                file_input,
                folder=folder,
                public_id=asset_id,
                resource_type="image",
                overwrite=True
            )
            return {
                "public_id": res.get("public_id"),
                "resource_type": "image",
                "secure_url": res.get("secure_url"),
                "format": res.get("format", "jpg"),
                "width": res.get("width", 0),
                "height": res.get("height", 0),
                "bytes": res.get("bytes", 0)
            }
        except Exception as e:
            logger.error(f"Cloudinary image upload failed: {e}")
            raise RuntimeError(f"Cloudinary upload error: {str(e)}")

    @classmethod
    def upload_video(
        cls,
        file_input: Any,
        model_id: str,
        class_id: str | None = None,
        public_id_suffix: str | None = None
    ) -> dict[str, Any]:
        """
        Uploads video to Cloudinary with resource_type='video'.
        Returns normalized metadata dictionary.
        """
        folder = cls.get_folder_path(model_id, "VIDEO", class_id)
        asset_id = public_id_suffix or str(uuid.uuid4())
        
        try:
            res = cloudinary.uploader.upload(
                file_input,
                folder=folder,
                public_id=asset_id,
                resource_type="video",
                overwrite=True
            )
            return {
                "public_id": res.get("public_id"),
                "resource_type": "video",
                "secure_url": res.get("secure_url"),
                "format": res.get("format", "mp4"),
                "width": res.get("width", 0),
                "height": res.get("height", 0),
                "bytes": res.get("bytes", 0),
                "duration": float(res.get("duration") or 0.0)
            }
        except Exception as e:
            logger.error(f"Cloudinary video upload failed: {e}")
            raise RuntimeError(f"Cloudinary video upload error: {str(e)}")

    @classmethod
    def delete_asset(cls, public_id: str, resource_type: str = "image") -> bool:
        """
        Deletes asset from Cloudinary using public_id and matching resource_type.
        """
        try:
            res = cloudinary.uploader.destroy(public_id, resource_type=resource_type)
            result_status = res.get("result")
            if result_status in ["ok", "not found"]:
                logger.info(f"Cloudinary destroy for '{public_id}' ({resource_type}): {result_status}")
                return True
            else:
                logger.warning(f"Cloudinary destroy returned unexpected status: {res}")
                return False
        except Exception as e:
            logger.error(f"Error deleting Cloudinary asset '{public_id}': {e}")
            return False

    @classmethod
    def get_account_usage(cls) -> dict[str, Any]:
        """Queries actual usage details from Cloudinary Admin API."""
        try:
            usage = cloudinary.api.usage()
            return {
                "success": True,
                "plan": usage.get("plan", "Free"),
                "credits": usage.get("credits", {}),
                "storage": usage.get("storage", {}),
                "bandwidth": usage.get("bandwidth", {}),
                "transformations": usage.get("transformations", {}),
                "objects": usage.get("objects", {})
            }
        except Exception as e:
            logger.warning(f"Could not fetch Cloudinary usage: {e}")
            return {
                "success": False,
                "error": str(e)
            }
