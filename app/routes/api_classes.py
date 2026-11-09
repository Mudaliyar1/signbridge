from fastapi import APIRouter, HTTPException, status
from app.database import models_col, classes_col, samples_col
from app.models.sign_class import SignClass, SignClassCreate
from app.storage.local import StorageService
from datetime import datetime, timezone

router = APIRouter(prefix="/api/models/{model_id}/classes", tags=["Sign Classes"])

@router.get("")
async def list_sign_classes(model_id: str):
    """List all sign classes for a specific model with real sample statistics."""
    model = await models_col().find_one({"id": model_id})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
        
    cursor = classes_col().find({"model_id": model_id}).sort("name", 1)
    classes_list = await cursor.to_list(length=500)
    
    result = []
    for c in classes_list:
        c_id = c["id"]
        img_count = await samples_col().count_documents({"model_id": model_id, "class_id": c_id, "media_type": "IMAGE"})
        vid_count = await samples_col().count_documents({"model_id": model_id, "class_id": c_id, "media_type": "VIDEO"})
        total_samples = img_count + vid_count
        
        c_data = dict(c)
        c_data.pop("_id", None)
        c_data["images_count"] = img_count
        c_data["videos_count"] = vid_count
        c_data["total_samples"] = total_samples
        result.append(c_data)
        
    return {"classes": result, "total": len(result)}

@router.post("", status_code=status.HTTP_201_CREATED)
async def create_sign_class(model_id: str, payload: SignClassCreate):
    """Create a new sign class under a model."""
    model = await models_col().find_one({"id": model_id})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
        
    normalized_name = payload.name.strip().upper()
    existing = await classes_col().find_one({"model_id": model_id, "name": normalized_name})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Sign class '{normalized_name}' already exists in this model."
        )
        
    output_txt = payload.output_text.strip() if payload.output_text else normalized_name.replace('_', ' ').title()
    new_class = SignClass(
        model_id=model_id,
        name=normalized_name,
        output_text=output_txt,
        description=payload.description.strip(),
        language=payload.language.strip().upper(),
        category=payload.category.strip(),
        sign_type=payload.sign_type.strip().upper()
    )
    
    doc = new_class.model_dump()
    await classes_col().insert_one(doc)
    doc.pop("_id", None)
    return doc

@router.get("/{class_id}")
async def get_sign_class(model_id: str, class_id: str):
    """Get single sign class details and its samples."""
    doc = await classes_col().find_one({"model_id": model_id, "id": class_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Sign class not found")
        
    doc.pop("_id", None)
    doc["images_count"] = await samples_col().count_documents({"model_id": model_id, "class_id": class_id, "media_type": "IMAGE"})
    doc["videos_count"] = await samples_col().count_documents({"model_id": model_id, "class_id": class_id, "media_type": "VIDEO"})
    
    # Fetch distinct participants who contributed to this sign class
    distinct_participants = await samples_col().distinct("participant_id", {"model_id": model_id, "class_id": class_id})
    doc["participants_count"] = len(distinct_participants)
    return doc

@router.delete("/{class_id}")
async def delete_sign_class(model_id: str, class_id: str):
    """Delete a sign class and cascade delete its samples and files."""
    doc = await classes_col().find_one({"model_id": model_id, "id": class_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Sign class not found")
        
    # Delete associated physical media files
    samples_cursor = samples_col().find({"model_id": model_id, "class_id": class_id})
    samples = await samples_cursor.to_list(length=5000)
    for sample in samples:
        if "storage_path" in sample:
            StorageService.delete_sample_file(sample["storage_path"])
            
    await samples_col().delete_many({"model_id": model_id, "class_id": class_id})
    await classes_col().delete_one({"model_id": model_id, "id": class_id})
    
    return {"success": True, "message": f"Sign class '{doc['name']}' and its samples deleted."}
