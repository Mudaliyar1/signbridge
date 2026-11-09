"""
Migration Script: Local Disk Media -> Cloudinary
Scans MongoDB Atlas dataset_samples for samples stored locally,
uploads them to Cloudinary in predictable folder paths,
updates MongoDB Atlas with Cloudinary metadata, and marks them as migrated.
Preserves the local file without deleting it.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.database import Database, samples_col
from app.storage.cloudinary import CloudinaryStorage
from app.storage.local import StorageService
from datetime import datetime, timezone

async def migrate_samples():
    print("=" * 60)
    print("SIGNBRIDGE: LOCAL MEDIA TO CLOUDINARY MIGRATION UTILITY")
    print("=" * 60)

    # 1. Connect to MongoDB Atlas
    await Database.connect()
    if not Database.is_connected():
        print(f"ERROR: Cannot connect to MongoDB Atlas at {settings.MONGODB_URL}")
        return

    # 2. Query for unmigrated samples
    query = {
        "$or": [
            {"storage_provider": "local"},
            {"cloudinary": None},
            {"storage_provider": {"$exists": False}}
        ]
    }
    unmigrated_cursor = samples_col().find(query)
    unmigrated_samples = await unmigrated_cursor.to_list(length=5000)

    total = len(unmigrated_samples)
    print(f"\nFound {total} sample(s) requiring migration to Cloudinary.\n")

    if total == 0:
        print("All dataset samples are already migrated to Cloudinary!")
        await Database.close()
        return

    migrated_count = 0
    failed_count = 0

    for idx, sample in enumerate(unmigrated_samples, 1):
        sample_id = sample.get("id")
        label = sample.get("label", "UNKNOWN")
        media_type = sample.get("media_type", "IMAGE").upper()
        storage_path = sample.get("storage_path", "")
        model_id = sample.get("model_id", "default")
        class_id = sample.get("class_id", "general")

        print(f"[{idx}/{total}] Migrating sample '{sample_id}' (Sign: {label}, Type: {media_type})...")

        if not storage_path:
            print(f"  -> SKIPPED: Sample {sample_id} has no storage_path.")
            failed_count += 1
            continue

        try:
            local_file = StorageService.resolve_path(storage_path)
            if not local_file.exists() or not local_file.is_file():
                print(f"  -> WARNING: Local file not found at {local_file}. Skipping.")
                failed_count += 1
                continue

            # Upload to Cloudinary
            if media_type == "IMAGE":
                cloud_meta = CloudinaryStorage.upload_image(
                    file_input=str(local_file),
                    model_id=model_id,
                    class_id=class_id,
                    public_id_suffix=f"migrated_{sample_id}"
                )
            else:
                cloud_meta = CloudinaryStorage.upload_video(
                    file_input=str(local_file),
                    model_id=model_id,
                    class_id=class_id,
                    public_id_suffix=f"migrated_{sample_id}"
                )

            # Update MongoDB Atlas record
            update_data = {
                "storage_provider": "cloudinary",
                "cloudinary": cloud_meta,
                "updated_at": datetime.now(timezone.utc)
            }
            await samples_col().update_one({"id": sample_id}, {"$set": update_data})

            print(f"  -> SUCCESS: Uploaded to Cloudinary! URL: {cloud_meta['secure_url']}")
            print(f"  -> Local copy preserved at: {local_file}")
            migrated_count += 1

        except Exception as e:
            print(f"  -> FAILED: {e}")
            failed_count += 1

    print("\n" + "=" * 60)
    print(f"MIGRATION COMPLETE: {migrated_count} migrated, {failed_count} failed, {total} total.")
    print("=" * 60)
    await Database.close()

if __name__ == "__main__":
    asyncio.run(migrate_samples())
