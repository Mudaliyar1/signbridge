from fastapi import APIRouter, HTTPException, status
from app.database import models_col, classes_col, samples_col, runs_col
from app.models.model_meta import SignModel, ModelCreate, ModelUpdate
from app.storage.local import StorageService
from datetime import datetime, timezone
import shutil
from pathlib import Path
from app.config import settings

router = APIRouter(prefix="/api/models", tags=["Models"])

@router.get("")
async def list_models():
    """List all models with real aggregated counts from MongoDB."""
    cursor = models_col().find().sort("created_at", -1)
    models_list = await cursor.to_list(length=100)
    
    result = []
    for m in models_list:
        m_id = m["id"]
        classes_count = await classes_col().count_documents({"model_id": m_id})
        samples_count = await samples_col().count_documents({"model_id": m_id})
        runs_count = await runs_col().count_documents({"model_id": m_id})
        
        m_data = dict(m)
        m_data.pop("_id", None)
        m_data["classes_count"] = classes_count
        m_data["samples_count"] = samples_count
        m_data["training_runs_count"] = runs_count
        result.append(m_data)
        
    return {"models": result, "total": len(result)}

@router.post("", status_code=status.HTTP_201_CREATED)
async def create_model(payload: ModelCreate):
    """Create a new model record in MongoDB."""
    # Check if model name already exists
    existing = await models_col().find_one({"name": payload.name.strip()})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model with name '{payload.name}' already exists."
        )
        
    new_model = SignModel(
        name=payload.name.strip(),
        version=payload.version.strip(),
        description=payload.description.strip(),
        architecture=payload.architecture.strip(),
        input_type=payload.input_type.strip(),
        input_resolution=payload.input_resolution.strip()
    )
    
    doc = new_model.model_dump()
    await models_col().insert_one(doc)
    doc.pop("_id", None)
    return doc

@router.get("/{model_id}")
async def get_model(model_id: str):
    """Retrieve details for a single model."""
    doc = await models_col().find_one({"id": model_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Model not found")
        
    doc.pop("_id", None)
    doc["classes_count"] = await classes_col().count_documents({"model_id": model_id})
    doc["samples_count"] = await samples_col().count_documents({"model_id": model_id})
    doc["training_runs_count"] = await runs_col().count_documents({"model_id": model_id})
    return doc

@router.patch("/{model_id}")
async def update_model(model_id: str, payload: ModelUpdate):
    """Update model metadata."""
    doc = await models_col().find_one({"id": model_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Model not found")
        
    updates = {}
    if payload.name:
        updates["name"] = payload.name.strip()
    if payload.description is not None:
        updates["description"] = payload.description.strip()
    if payload.architecture:
        updates["architecture"] = payload.architecture.strip()
    if payload.input_resolution:
        updates["input_resolution"] = payload.input_resolution.strip()
        
    if updates:
        updates["updated_at"] = datetime.now(timezone.utc)
        await models_col().update_one({"id": model_id}, {"$set": updates})
        
    updated = await models_col().find_one({"id": model_id})
    updated.pop("_id", None)
    return updated

@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """Deletes model, cascade removes sign classes, samples, physical media files, and runs."""
    doc = await models_col().find_one({"id": model_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Model not found")
        
    # Delete all physical media files for this model
    samples_cursor = samples_col().find({"model_id": model_id})
    samples = await samples_cursor.to_list(length=10000)
    for sample in samples:
        if "storage_path" in sample:
            StorageService.delete_sample_file(sample["storage_path"])
            
    # Remove directory if exists
    model_storage_dir = settings.DATASETS_DIR / "models" / model_id
    if model_storage_dir.exists():
        shutil.rmtree(model_storage_dir, ignore_errors=True)
        
    # Cascade MongoDB collections
    await samples_col().delete_many({"model_id": model_id})
    await classes_col().delete_many({"model_id": model_id})
    await runs_col().delete_many({"model_id": model_id})
    await models_col().delete_one({"id": model_id})
    
    return {"success": True, "message": f"Model '{doc['name']}' and all associated data deleted."}

@router.post("/{model_id}/activate")
async def activate_model(model_id: str):
    """
    Activates a model.
    Per specification: Activation must only be allowed when a valid trained model exists.
    """
    doc = await models_col().find_one({"id": model_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Model not found")
        
    # Verify trained weights exist
    weights_path = settings.MODELS_DIR / "versions" / model_id / "final" / "model.pt"
    if not weights_path.exists() and doc.get("status") != "Trained":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot activate an untrained model. A trained model checkpoint is required before activation."
        )
        
    # Deactivate any currently active model
    await models_col().update_many({}, {"$set": {"is_active": False}})
    # Mark this model active
    await models_col().update_one({"id": model_id}, {"$set": {"is_active": True, "updated_at": datetime.now(timezone.utc)}})
    
    return {"success": True, "message": f"Model '{doc['name']}' is now the active production model."}
