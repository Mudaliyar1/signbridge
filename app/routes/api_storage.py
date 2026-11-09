from fastapi import APIRouter
from app.database import samples_col
from app.storage.cloudinary import CloudinaryStorage
from app.config import settings

router = APIRouter(prefix="/api/storage", tags=["Storage"])

@router.get("/overview")
async def get_storage_overview():
    """Returns real storage breakdown from MongoDB Atlas and Cloudinary Admin API."""
    total_cloudinary = await samples_col().count_documents({"storage_provider": "cloudinary"})
    total_local = await samples_col().count_documents({"storage_provider": "local"})
    
    images_count = await samples_col().count_documents({"storage_provider": "cloudinary", "media_type": "IMAGE"})
    videos_count = await samples_col().count_documents({"storage_provider": "cloudinary", "media_type": "VIDEO"})
    
    # Query Cloudinary live account usage
    cloud_usage = CloudinaryStorage.get_account_usage()
    
    return {
        "provider": "Cloudinary",
        "cloud_name": settings.CLOUDINARY_CLOUD_NAME,
        "folder_prefix": settings.CLOUDINARY_FOLDER_PREFIX,
        "database_stats": {
            "total_cloudinary_assets": total_cloudinary,
            "cloudinary_images": images_count,
            "cloudinary_videos": videos_count,
            "legacy_local_samples": total_local
        },
        "cloudinary_account_usage": cloud_usage
    }
