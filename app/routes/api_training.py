import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from pydantic import BaseModel

from app.database import models_col, runs_col, classes_col, samples_col
from app.ml.training import train_model_from_dataset

logger = logging.getLogger("signbridge.api_training")
router = APIRouter(prefix="/api/models/{model_id}/training", tags=["Model Training"])

class TrainingConfigPayload(BaseModel):
    epochs: int = 12
    batch_size: int = 8
    learning_rate: float = 0.001

@router.post("/start")
async def start_training(model_id: str, payload: TrainingConfigPayload | None = None):
    """
    Triggers scratch-built PyTorch training of CustomSignCNN on real dataset samples.
    """
    model = await models_col().find_one({"id": model_id})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    classes_count = await classes_col().count_documents({"model_id": model_id})
    if classes_count < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model has {classes_count} sign class(es). At least 2 sign classes are required to train a classifier. Please add another sign class."
        )

    samples_count = await samples_col().count_documents({"model_id": model_id, "status": "VALID"})
    if samples_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid dataset samples found. Please upload or capture images/videos for each sign class before training."
        )

    epochs = payload.epochs if payload else 12
    batch_size = payload.batch_size if payload else 8
    lr = payload.learning_rate if payload else 0.001

    try:
        # Run training loop directly
        result = await train_model_from_dataset(
            model_id=model_id,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Training failed: {str(e)}")

@router.get("/status")
async def get_training_status(model_id: str):
    """Fetches the latest training run metrics and status for this model."""
    latest_run = await runs_col().find_one({"model_id": model_id}, sort=[("created_at", -1)])
    if not latest_run:
        return {
            "has_run": False,
            "status": "NOT_TRAINED",
            "message": "No training runs recorded for this model yet."
        }
    latest_run.pop("_id", None)
    return {
        "has_run": True,
        "run": latest_run
    }
