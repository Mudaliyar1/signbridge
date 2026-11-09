from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.database import Database, models_col, classes_col, participants_col, samples_col, runs_col
from app.storage.cloudinary import CloudinaryStorage
from app.config import settings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

router = APIRouter(include_in_schema=False)

def check_db_health(request: Request):
    """If MongoDB Atlas is not connected, render the explicit MongoDB error page."""
    if not Database.is_connected():
        return templates.TemplateResponse(
            request=request,
            name="error_db.html",
            context={
                "error_message": Database.get_last_error() or "MongoDB Atlas is offline or unreachable.",
                "database_url": settings.MONGODB_URL.split("@")[-1] if "@" in settings.MONGODB_URL else settings.MONGODB_URL
            },
            status_code=503
        )
    return None

def attach_media_url(sample_dict: dict):
    """Utility to attach direct media_url for template rendering."""
    c_meta = sample_dict.get("cloudinary")
    if c_meta and c_meta.get("secure_url"):
        sample_dict["media_url"] = c_meta["secure_url"]
    elif sample_dict.get("storage_path"):
        sample_dict["media_url"] = f"/api/media/{sample_dict['storage_path']}"
    else:
        sample_dict["media_url"] = ""
    return sample_dict

@router.get("/", response_class=HTMLResponse)
@router.get("/admin", response_class=HTMLResponse)
@router.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    err = check_db_health(request)
    if err:
        return err

    # Strictly real data from MongoDB Atlas
    models_count = await models_col().count_documents({})
    samples_count = await samples_col().count_documents({})
    classes_count = await classes_col().count_documents({})
    participants_count = await participants_col().count_documents({})
    runs_count = await runs_col().count_documents({})
    
    active_model = await models_col().find_one({"is_active": True})
    if active_model:
        active_model.pop("_id", None)
        
    recent_models = await models_col().find().sort("created_at", -1).limit(5).to_list(5)
    for m in recent_models:
        m.pop("_id", None)
        m["samples_count"] = await samples_col().count_documents({"model_id": m["id"]})
        m["classes_count"] = await classes_col().count_documents({"model_id": m["id"]})

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "active_page": "dashboard",
            "models_count": models_count,
            "samples_count": samples_count,
            "classes_count": classes_count,
            "participants_count": participants_count,
            "runs_count": runs_count,
            "active_model": active_model,
            "recent_models": recent_models
        }
    )

@router.get("/admin/models", response_class=HTMLResponse)
async def admin_models_list(request: Request):
    err = check_db_health(request)
    if err:
        return err

    cursor = models_col().find().sort("created_at", -1)
    models_list = await cursor.to_list(100)
    for m in models_list:
        m.pop("_id", None)
        m["classes_count"] = await classes_col().count_documents({"model_id": m["id"]})
        m["samples_count"] = await samples_col().count_documents({"model_id": m["id"]})
        m["training_runs_count"] = await runs_col().count_documents({"model_id": m["id"]})

    return templates.TemplateResponse(
        request=request,
        name="models/list.html",
        context={
            "active_page": "models",
            "models": models_list
        }
    )

@router.get("/admin/models/{model_id}", response_class=HTMLResponse)
async def admin_model_workspace(request: Request, model_id: str, tab: str = "overview"):
    err = check_db_health(request)
    if err:
        return err

    model = await models_col().find_one({"id": model_id})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    model.pop("_id", None)

    classes = await classes_col().find({"model_id": model_id}).sort("name", 1).to_list(500)
    for c in classes:
        c.pop("_id", None)
        c["samples_count"] = await samples_col().count_documents({"model_id": model_id, "class_id": c["id"]})

    participants = await participants_col().find().sort("participant_id", 1).to_list(500)
    for p in participants:
        p.pop("_id", None)

    total_samples = await samples_col().count_documents({"model_id": model_id})
    distinct_participants = await samples_col().distinct("participant_id", {"model_id": model_id})

    return templates.TemplateResponse(
        request=request,
        name="models/workspace.html",
        context={
            "active_page": "models",
            "active_tab": tab,
            "model": model,
            "classes": classes,
            "classes_count": len(classes),
            "participants": participants,
            "participants_count": len(distinct_participants),
            "samples_count": total_samples
        }
    )

@router.get("/admin/models/{model_id}/classes", response_class=HTMLResponse)
async def admin_model_classes(request: Request, model_id: str):
    err = check_db_health(request)
    if err:
        return err

    model = await models_col().find_one({"id": model_id})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    model.pop("_id", None)

    classes = await classes_col().find({"model_id": model_id}).sort("name", 1).to_list(500)
    for c in classes:
        c.pop("_id", None)
        c["images_count"] = await samples_col().count_documents({"model_id": model_id, "class_id": c["id"], "media_type": "IMAGE"})
        c["videos_count"] = await samples_col().count_documents({"model_id": model_id, "class_id": c["id"], "media_type": "VIDEO"})
        c["total_samples"] = c["images_count"] + c["videos_count"]

    return templates.TemplateResponse(
        request=request,
        name="models/classes.html",
        context={
            "active_page": "models",
            "active_tab": "classes",
            "model": model,
            "classes": classes
        }
    )

@router.get("/admin/models/{model_id}/classes/{class_id}", response_class=HTMLResponse)
async def admin_sign_class_detail(request: Request, model_id: str, class_id: str):
    err = check_db_health(request)
    if err:
        return err

    model = await models_col().find_one({"id": model_id})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    model.pop("_id", None)

    sign_class = await classes_col().find({"model_id": model_id, "id": class_id}).to_list(1)
    if not sign_class:
        raise HTTPException(status_code=404, detail="Sign class not found")
    sign_class = sign_class[0]
    sign_class.pop("_id", None)

    images_count = await samples_col().count_documents({"model_id": model_id, "class_id": class_id, "media_type": "IMAGE"})
    videos_count = await samples_col().count_documents({"model_id": model_id, "class_id": class_id, "media_type": "VIDEO"})
    distinct_participants = await samples_col().distinct("participant_id", {"model_id": model_id, "class_id": class_id})

    # Get recent samples for this sign
    samples = await samples_col().find({"model_id": model_id, "class_id": class_id}).sort("created_at", -1).limit(50).to_list(50)
    for s in samples:
        s.pop("_id", None)
        attach_media_url(s)

    return templates.TemplateResponse(
        request=request,
        name="models/class_detail.html",
        context={
            "active_page": "models",
            "active_tab": "classes",
            "model": model,
            "sign_class": sign_class,
            "images_count": images_count,
            "videos_count": videos_count,
            "participants_count": len(distinct_participants),
            "samples": samples
        }
    )

@router.get("/admin/models/{model_id}/capture", response_class=HTMLResponse)
async def admin_model_capture(request: Request, model_id: str, sign: str | None = None):
    err = check_db_health(request)
    if err:
        return err

    model = await models_col().find_one({"id": model_id})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    model.pop("_id", None)

    classes = await classes_col().find({"model_id": model_id}).sort("name", 1).to_list(500)
    for c in classes:
        c.pop("_id", None)

    participants = await participants_col().find().sort("participant_id", 1).to_list(500)
    for p in participants:
        p.pop("_id", None)

    return templates.TemplateResponse(
        request=request,
        name="models/capture.html",
        context={
            "active_page": "models",
            "active_tab": "capture",
            "model": model,
            "classes": classes,
            "participants": participants,
            "preselected_sign": sign
        }
    )

@router.get("/admin/models/{model_id}/dataset", response_class=HTMLResponse)
async def admin_model_dataset(
    request: Request,
    model_id: str,
    class_id: str | None = None,
    participant_id: str | None = None,
    media_type: str | None = None,
    status: str | None = None
):
    err = check_db_health(request)
    if err:
        return err

    model = await models_col().find_one({"id": model_id})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    model.pop("_id", None)

    classes = await classes_col().find({"model_id": model_id}).sort("name", 1).to_list(500)
    for c in classes:
        c.pop("_id", None)

    participants = await participants_col().find().sort("participant_id", 1).to_list(500)
    for p in participants:
        p.pop("_id", None)

    # Build query
    query = {"model_id": model_id}
    if class_id:
        query["class_id"] = class_id
    if participant_id:
        query["participant_id"] = participant_id
    if media_type:
        query["media_type"] = media_type.upper()
    if status:
        query["status"] = status.upper()

    total_samples = await samples_col().count_documents(query)
    samples = await samples_col().find(query).sort("created_at", -1).limit(100).to_list(100)
    for s in samples:
        s.pop("_id", None)
        attach_media_url(s)

    return templates.TemplateResponse(
        request=request,
        name="models/dataset.html",
        context={
            "active_page": "models",
            "active_tab": "dataset",
            "model": model,
            "classes": classes,
            "participants": participants,
            "samples": samples,
            "total_samples": total_samples,
            "selected_class": class_id,
            "selected_participant": participant_id,
            "selected_media_type": media_type,
            "selected_status": status
        }
    )

@router.get("/admin/participants", response_class=HTMLResponse)
async def admin_participants(request: Request):
    err = check_db_health(request)
    if err:
        return err

    participants = await participants_col().find().sort("participant_id", 1).to_list(500)
    for p in participants:
        p.pop("_id", None)
        p["samples_count"] = await samples_col().count_documents({"participant_id": p["participant_id"]})

    return templates.TemplateResponse(
        request=request,
        name="participants/list.html",
        context={
            "active_page": "participants",
            "participants": participants
        }
    )

@router.get("/admin/storage", response_class=HTMLResponse)
async def admin_storage_overview(request: Request):
    err = check_db_health(request)
    if err:
        return err

    total_cloudinary = await samples_col().count_documents({"storage_provider": "cloudinary"})
    total_local = await samples_col().count_documents({"storage_provider": "local"})
    
    images_count = await samples_col().count_documents({"storage_provider": "cloudinary", "media_type": "IMAGE"})
    videos_count = await samples_col().count_documents({"storage_provider": "cloudinary", "media_type": "VIDEO"})
    total_all = await samples_col().count_documents({})

    cloud_usage = CloudinaryStorage.get_account_usage()

    return templates.TemplateResponse(
        request=request,
        name="storage/overview.html",
        context={
            "active_page": "storage",
            "cloud_name": settings.CLOUDINARY_CLOUD_NAME,
            "folder_prefix": settings.CLOUDINARY_FOLDER_PREFIX,
            "total_cloudinary": total_cloudinary,
            "total_local": total_local,
            "images_count": images_count,
            "videos_count": videos_count,
            "total_all": total_all,
            "cloud_usage": cloud_usage
        }
    )

@router.get("/admin/models/{model_id}/testing", response_class=HTMLResponse)
async def admin_model_testing(request: Request, model_id: str):
    err = check_db_health(request)
    if err:
        return err

    model = await models_col().find_one({"id": model_id})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    model.pop("_id", None)

    classes = await classes_col().find({"model_id": model_id}).sort("name", 1).to_list(500)
    for c in classes:
        c.pop("_id", None)

    participants = await participants_col().find().sort("participant_id", 1).to_list(500)
    for p in participants:
        p.pop("_id", None)

    samples = await samples_col().find({"model_id": model_id, "status": "VALID"}).sort("created_at", -1).limit(200).to_list(200)
    for s in samples:
        s.pop("_id", None)
        attach_media_url(s)

    from app.ml.inference import CustomSignInferenceService
    availability = CustomSignInferenceService.check_model_availability(model_id)

    return templates.TemplateResponse(
        request=request,
        name="models/testing.html",
        context={
            "active_page": "models",
            "active_tab": "testing",
            "model": model,
            "classes": classes,
            "classes_count": len(classes),
            "participants": participants,
            "samples": samples,
            "samples_count": len(samples),
            "availability": availability
        }
    )

@router.get("/admin/models/{model_id}/ml-analysis", response_class=HTMLResponse)
async def admin_model_ml_analysis(request: Request, model_id: str):
    err = check_db_health(request)
    if err:
        return err

    model = await models_col().find_one({"id": model_id})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    model.pop("_id", None)

    classes = await classes_col().find({"model_id": model_id}).sort("name", 1).to_list(500)
    for c in classes:
        c.pop("_id", None)

    participants = await participants_col().find().sort("participant_id", 1).to_list(500)
    for p in participants:
        p.pop("_id", None)

    valid_samples_count = await samples_col().count_documents({"model_id": model_id, "status": "VALID"})
    total_samples_count = await samples_col().count_documents({"model_id": model_id})

    # Fetch existing classical comparison if any
    from app.ml.classical_models import get_classical_storage_dir
    meta_file = get_classical_storage_dir(model_id) / "classical_meta.json"
    comparison_data = None
    if meta_file.exists():
        import json
        with open(meta_file, "r", encoding="utf-8") as f:
            comparison_data = json.load(f)

    return templates.TemplateResponse(
        request=request,
        name="models/ml_analysis.html",
        context={
            "active_page": "models",
            "active_tab": "ml_analysis",
            "model": model,
            "classes": classes,
            "classes_count": len(classes),
            "participants": participants,
            "participants_count": len(participants),
            "valid_samples_count": valid_samples_count,
            "total_samples_count": total_samples_count,
            "comparison_data": comparison_data
        }
    )

