import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from app.config import settings
from app.database import Database
from app.storage.local import StorageService

# Routers
from app.routes import admin_views, api_models, api_classes, api_participants, api_samples, api_storage, api_testing, api_training, api_ml_analysis

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("signbridge")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to MongoDB strictly
    logger.info("Initializing SignBridge application...")
    await Database.connect()
    yield
    # Shutdown
    logger.info("Shutting down SignBridge application...")
    await Database.close()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(settings.BASE_DIR / "app" / "static")), name="static")

# Secure Media delivery endpoint
@app.get("/api/media/{file_path:path}")
async def serve_media(file_path: str):
    """Securely stream/serve physical dataset images and videos."""
    try:
        resolved_path = StorageService.resolve_path(file_path)
        if not resolved_path.exists() or not resolved_path.is_file():
            raise HTTPException(status_code=404, detail="Media file not found")
        
        # Determine media content-type
        suffix = resolved_path.suffix.lower()
        content_type = "image/jpeg"
        if suffix in [".png"]:
            content_type = "image/png"
        elif suffix in [".webp"]:
            content_type = "image/webp"
        elif suffix in [".mp4"]:
            content_type = "video/mp4"
        elif suffix in [".webm"]:
            content_type = "video/webm"

        return FileResponse(resolved_path, media_type=content_type)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file path")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/api/health")
async def health_check():
    is_connected = await Database.check_health()
    return {
        "status": "healthy" if is_connected else "degraded",
        "database": {
            "type": "mongodb",
            "connected": is_connected,
            "url": settings.MONGODB_URL,
            "error": Database.get_last_error()
        }
    }

# Global real-time hand detection endpoint for browser camera fallback
@app.post("/api/detect-hand")
async def detect_hand_global(
    file: UploadFile | None = File(None),
    frame_base64: str | None = Form(None)
):
    from app.routes.api_samples import detect_hand_live
    return await detect_hand_live(model_id="global", file=file, frame_base64=frame_base64)

# Register Routers
app.include_router(admin_views.router)
app.include_router(api_models.router)
app.include_router(api_classes.router)
app.include_router(api_participants.router)
app.include_router(api_samples.router)
app.include_router(api_storage.router)
app.include_router(api_testing.router)
app.include_router(api_training.router)
app.include_router(api_ml_analysis.router)
