from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, status
from pydantic import BaseModel
from app.database import models_col, classes_col, participants_col, samples_col
from app.models.dataset_sample import DatasetSample, CloudinaryMetadata
from app.storage.service import StorageServiceManager
from app.config import settings
from datetime import datetime, timezone
import logging

logger = logging.getLogger("signbridge.samples")
router = APIRouter(prefix="/api/models/{model_id}", tags=["Dataset Samples"])

class BulkDeletePayload(BaseModel):
    sample_ids: list[str]

class StatusUpdatePayload(BaseModel):
    status: str

@router.post("/samples/capture-image", status_code=status.HTTP_201_CREATED)
async def capture_image_sample(
    model_id: str,
    file: UploadFile = File(...),
    class_id: str = Form(...),
    participant_id: str = Form(...),
    notes: str = Form(""),
    handedness: str | None = Form(None),
    hand_detection_confidence: float | None = Form(None),
    hand_bbox: str | None = Form(None)
):
    """
    Saves an image captured directly from the camera or uploaded.
    Validation: OpenCV/PIL -> Cloudinary Upload -> MongoDB Atlas persistence.
    Compensating transaction rollback if MongoDB insert fails.
    """
    # 1. Validate parent model
    model = await models_col().find_one({"id": model_id})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # 2. Validate sign class
    sign_class = await classes_col().find_one({"model_id": model_id, "id": class_id})
    if not sign_class:
        raise HTTPException(status_code=404, detail="Sign class not found in this model")

    # 3. Validate participant
    participant = await participants_col().find_one({"participant_id": participant_id})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")
        
    if participant.get("consent_status") == "REVOKED":
        raise HTTPException(status_code=400, detail="Cannot collect data for participant with REVOKED consent status.")

    # 4. Read bytes
    content = await file.read()
    if not content or len(content) == 0:
        raise HTTPException(status_code=400, detail="Received empty image file")

    parsed_bbox = None
    if hand_bbox:
        try:
            import json
            parsed_bbox = json.loads(hand_bbox) if isinstance(hand_bbox, str) else hand_bbox
        except Exception:
            parsed_bbox = None

    # 5. Process & upload to Cloudinary
    try:
        cloud_meta = StorageServiceManager.process_and_upload_image(
            image_bytes=content,
            model_id=model_id,
            class_id=class_id,
            original_filename=file.filename or "camera_capture.jpg"
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Image validation failed: {str(ve)}")
    except Exception as ce:
        logger.error(f"Cloudinary image upload error: {ce}")
        raise HTTPException(status_code=502, detail=f"Cloud media upload failed: {str(ce)}")

    # 6. Create MongoDB Atlas record
    sample = DatasetSample(
        model_id=model_id,
        class_id=class_id,
        participant_id=participant_id,
        media_type="IMAGE",
        storage_provider="cloudinary",
        storage_path=cloud_meta["public_id"],
        cloudinary=CloudinaryMetadata(**cloud_meta),
        original_filename=file.filename or "camera_capture.jpg",
        mime_type=file.content_type or f"image/{cloud_meta.get('format', 'jpg')}",
        file_size=cloud_meta.get("bytes") or len(content),
        width=cloud_meta.get("width") or 0,
        height=cloud_meta.get("height") or 0,
        duration=0.0,
        label=sign_class["name"],
        language=sign_class.get("language", "ISL"),
        notes=notes.strip(),
        handedness=handedness,
        hand_detection_confidence=hand_detection_confidence,
        hand_bbox=parsed_bbox,
        preprocessing_version="hand_landmark_v1",
        status="VALID"
    )

    doc = sample.model_dump()
    try:
        await samples_col().insert_one(doc)
    except Exception as e:
        # Atomic compensating transaction: destroy uploaded Cloudinary asset
        StorageServiceManager.rollback_cloudinary(cloud_meta["public_id"], "image")
        logger.error(f"Failed to record dataset sample in MongoDB Atlas: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database error while saving dataset sample: {str(e)}"
        )

    doc.pop("_id", None)
    doc["media_url"] = cloud_meta["secure_url"]
    return doc

@router.post("/samples/capture-video", status_code=status.HTTP_201_CREATED)
async def capture_video_sample(
    model_id: str,
    file: UploadFile = File(...),
    class_id: str = Form(...),
    participant_id: str = Form(...),
    duration: float = Form(0.0),
    notes: str = Form(""),
    handedness: str | None = Form(None),
    hand_detection_confidence: float | None = Form(None),
    hand_bbox: str | None = Form(None)
):
    """
    Saves a video captured directly from the camera or uploaded.
    Validation: OpenCV frames -> Cloudinary Upload -> MongoDB Atlas persistence.
    Compensating transaction rollback if MongoDB insert fails.
    """
    model = await models_col().find_one({"id": model_id})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    sign_class = await classes_col().find_one({"model_id": model_id, "id": class_id})
    if not sign_class:
        raise HTTPException(status_code=404, detail="Sign class not found in this model")

    participant = await participants_col().find_one({"participant_id": participant_id})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")
        
    if participant.get("consent_status") == "REVOKED":
        raise HTTPException(status_code=400, detail="Cannot collect data for participant with REVOKED consent status.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty video data received")

    parsed_bbox = None
    if hand_bbox:
        try:
            import json
            parsed_bbox = json.loads(hand_bbox) if isinstance(hand_bbox, str) else hand_bbox
        except Exception:
            parsed_bbox = None

    # Process & upload video to Cloudinary
    try:
        cloud_meta = StorageServiceManager.process_and_upload_video(
            video_bytes=content,
            model_id=model_id,
            class_id=class_id,
            original_filename=file.filename or "camera_recording.webm",
            client_duration=duration
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Video validation failed: {str(ve)}")
    except Exception as ce:
        logger.error(f"Cloudinary video upload error: {ce}")
        raise HTTPException(status_code=502, detail=f"Cloud video upload failed: {str(ce)}")

    sample = DatasetSample(
        model_id=model_id,
        class_id=class_id,
        participant_id=participant_id,
        media_type="VIDEO",
        storage_provider="cloudinary",
        storage_path=cloud_meta["public_id"],
        cloudinary=CloudinaryMetadata(**cloud_meta),
        original_filename=file.filename or "camera_recording.webm",
        mime_type=file.content_type or "video/webm",
        file_size=cloud_meta.get("bytes") or len(content),
        width=cloud_meta.get("width") or 0,
        height=cloud_meta.get("height") or 0,
        duration=cloud_meta.get("duration") or duration,
        label=sign_class["name"],
        language=sign_class.get("language", "ISL"),
        notes=notes.strip(),
        handedness=handedness,
        hand_detection_confidence=hand_detection_confidence,
        hand_bbox=parsed_bbox,
        preprocessing_version="hand_landmark_v1",
        status="VALID"
    )

    doc = sample.model_dump()
    try:
        await samples_col().insert_one(doc)
    except Exception as e:
        StorageServiceManager.rollback_cloudinary(cloud_meta["public_id"], "video")
        logger.error(f"Failed to record video dataset sample in MongoDB Atlas: {e}")
        raise HTTPException(status_code=500, detail=f"Database error while saving video sample: {str(e)}")

    doc.pop("_id", None)
    doc["media_url"] = cloud_meta["secure_url"]
    return doc

@router.post("/samples/upload", status_code=status.HTTP_201_CREATED)
async def upload_media_sample(
    model_id: str,
    file: UploadFile = File(...),
    class_id: str = Form(...),
    participant_id: str = Form(...),
    media_type: str = Form(...),
    notes: str = Form("")
):
    """Handles uploaded existing media (JPG, PNG, MP4, WEBM) to Cloudinary."""
    media_type = media_type.upper()
    if media_type not in ["IMAGE", "VIDEO"]:
        raise HTTPException(status_code=400, detail="Invalid media type. Must be IMAGE or VIDEO.")

    if media_type == "IMAGE":
        return await capture_image_sample(
            model_id=model_id,
            file=file,
            class_id=class_id,
            participant_id=participant_id,
            notes=notes
        )
    else:
        return await capture_video_sample(
            model_id=model_id,
            file=file,
            class_id=class_id,
            participant_id=participant_id,
            duration=0.0,
            notes=notes
        )

@router.get("/dataset")
async def list_dataset_samples(
    model_id: str,
    class_id: str | None = None,
    participant_id: str | None = None,
    media_type: str | None = None,
    status_filter: str | None = Query(None, alias="status"),
    skip: int = 0,
    limit: int = 100
):
    """
    List dataset samples with filters.
    Includes direct Cloudinary secure URLs.
    """
    query = {"model_id": model_id}
    if class_id:
        query["class_id"] = class_id
    if participant_id:
        query["participant_id"] = participant_id
    if media_type:
        query["media_type"] = media_type.upper()
    if status_filter:
        query["status"] = status_filter.upper()

    total = await samples_col().count_documents(query)
    cursor = samples_col().find(query).sort("created_at", -1).skip(skip).limit(limit)
    samples_list = await cursor.to_list(length=limit)

    for s in samples_list:
        s.pop("_id", None)
        # Compute direct media URL
        c_meta = s.get("cloudinary")
        if c_meta and c_meta.get("secure_url"):
            s["media_url"] = c_meta["secure_url"]
        elif s.get("storage_path"):
            s["media_url"] = f"/api/media/{s['storage_path']}"
        elif s.get("storage_provider") == "cloudinary":
            s["media_url"] = f"https://res.cloudinary.com/{settings.CLOUDINARY_CLOUD_NAME}/image/upload/{s.get('id', 'sample')}.jpg"
        else:
            s["media_url"] = ""

    return {
        "samples": samples_list,
        "total": total,
        "skip": skip,
        "limit": limit
    }

@router.get("/samples/{sample_id}/model-input-preview")
async def get_sample_model_input_preview(model_id: str, sample_id: str):
    """
    Returns the exact 224x224 hand-cropped JPEG image tensor as fed to CustomSignCNN.
    Allows administrators to visually verify that the face is excluded and the hand/fingers
    are captured within the region of interest.
    """
    sample = await samples_col().find_one({"model_id": model_id, "id": sample_id})
    if not sample:
        raise HTTPException(status_code=404, detail="Dataset sample not found")

    import cv2
    import tempfile
    from pathlib import Path
    from fastapi.responses import Response
    from app.ml.training import fetch_sample_bytes
    from app.ml.preprocessing import get_model_input_preview_jpeg

    media_bytes = await fetch_sample_bytes(sample)
    if not media_bytes:
        raise HTTPException(status_code=404, detail="Media file content could not be retrieved")

    media_type = sample.get("media_type", "IMAGE")
    try:
        if media_type == "VIDEO":
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_f:
                temp_f.write(media_bytes)
                temp_f_path = temp_f.name

            cap = cv2.VideoCapture(temp_f_path)
            frame_ret, frame = cap.read()
            cap.release()
            Path(temp_f_path).unlink(missing_ok=True)

            if not frame_ret or frame is None:
                raise ValueError("Could not extract frame from video for preview")

            success, encoded_frame = cv2.imencode(".jpg", frame)
            if not success:
                raise ValueError("Could not encode video frame")
            preview_bytes = get_model_input_preview_jpeg(encoded_frame.tobytes())
        else:
            preview_bytes = get_model_input_preview_jpeg(media_bytes)

        return Response(content=preview_bytes, media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Failed to generate model input preview for sample {sample_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate model input preview: {str(e)}")

@router.delete("/dataset/{sample_id}")
async def delete_dataset_sample(model_id: str, sample_id: str):
    """
    Deletes a sample:
    1. Removes media from Cloudinary (using public_id & resource_type).
    2. Deletes record from MongoDB Atlas.
    """
    sample = await samples_col().find_one({"model_id": model_id, "id": sample_id})
    if not sample:
        raise HTTPException(status_code=404, detail="Dataset sample not found")

    # Delete physical media from Cloudinary / local
    success = StorageServiceManager.delete_sample_media(sample)
    if not success and sample.get("storage_provider") == "cloudinary":
        logger.warning(f"Cloudinary asset deletion returned warning for sample '{sample_id}'")

    await samples_col().delete_one({"model_id": model_id, "id": sample_id})
    return {"success": True, "message": "Dataset sample deleted successfully from Cloudinary and database"}

@router.post("/dataset/bulk-delete")
async def bulk_delete_samples(model_id: str, payload: BulkDeletePayload):
    """Deletes multiple dataset samples simultaneously from Cloudinary and MongoDB Atlas."""
    if not payload.sample_ids:
        return {"success": True, "deleted_count": 0, "failed_count": 0}

    cursor = samples_col().find({"model_id": model_id, "id": {"$in": payload.sample_ids}})
    samples = await cursor.to_list(length=len(payload.sample_ids))

    deleted_count = 0
    failed_count = 0
    ids_to_remove = []

    for s in samples:
        try:
            StorageServiceManager.delete_sample_media(s)
            ids_to_remove.append(s["id"])
            deleted_count += 1
        except Exception as e:
            logger.error(f"Failed to delete sample '{s['id']}': {e}")
            failed_count += 1

    if ids_to_remove:
        await samples_col().delete_many({"model_id": model_id, "id": {"$in": ids_to_remove}})

    return {
        "success": True,
        "deleted_count": deleted_count,
        "failed_count": failed_count
    }

@router.patch("/dataset/{sample_id}/status")
async def update_sample_status(model_id: str, sample_id: str, payload: StatusUpdatePayload):
    """Updates sample quality status (VALID, REVIEW_REQUIRED, INVALID)."""
    valid_statuses = {"VALID", "REVIEW_REQUIRED", "INVALID"}
    new_status = payload.status.upper()
    if new_status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Status must be one of {valid_statuses}")

    res = await samples_col().update_one(
        {"model_id": model_id, "id": sample_id},
        {"$set": {"status": new_status, "updated_at": datetime.now(timezone.utc)}}
    )
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Dataset sample not found")

    return {"success": True, "status": new_status}

@router.post("/detect-hand")
async def detect_hand_live(
    model_id: str,
    file: UploadFile | None = File(None),
    frame_base64: str | None = Form(None)
):
    """
    High-performance endpoint for real-time hand detection and 21 landmark extraction.
    Ensures 100% reliability across all browser environments.
    """
    import base64
    import numpy as np
    import cv2
    from PIL import Image
    import io
    from app.ml.preprocessing import HandLocalizationService

    image_bytes = None
    if file:
        image_bytes = await file.read()
    elif frame_base64:
        if "," in frame_base64:
            frame_base64 = frame_base64.split(",")[1]
        image_bytes = base64.b64decode(frame_base64)
    else:
        raise HTTPException(status_code=400, detail="No frame provided")

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(pil_img)
        res = HandLocalizationService.detect_hands(img_np)
        return {"success": True, **res}
    except Exception as e:
        logger.error(f"Hand detection endpoint error: {e}")
        return {"success": False, "hand_present": False, "status": "NO_HAND", "error": str(e)}
