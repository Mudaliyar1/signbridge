"""
SignBridge ML Analysis REST API.

Provides endpoints for:
- Live Dataset Overview & EDA Analytics
- Classical ML Model Training & Hyperparameter Tuning (KNN, SVM, Decision Tree)
- Multi-Model Performance Comparison Table (KNN vs SVM vs Decision Tree vs Custom CNN)
- Unsupervised K-Means Clustering & PCA Visualization
- Experiment History Querying
- Single-Sample Classical Inference
"""

import logging
from typing import Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from pydantic import BaseModel, Field

from app.database import models_col, classes_col, samples_col, experiments_col
from app.ml.classical_models import ClassicalMLEngine, get_classical_storage_dir
from app.ml.eda import EDAService
from app.ml.clustering import KMeansClusteringService
from app.ml.training import fetch_sample_bytes

logger = logging.getLogger("signbridge.api_ml_analysis")
router = APIRouter(prefix="/api/models/{model_id}/ml", tags=["ML Analysis"])


class TrainingPayload(BaseModel):
    knn: dict[str, Any] | None = None
    svm: dict[str, Any] | None = None
    dt: dict[str, Any] | None = None
    include_cnn: bool = True


class ClusterPayload(BaseModel):
    n_clusters: int = Field(default=2, ge=2, le=10)


@router.get("/overview")
async def get_ml_dataset_overview(model_id: str):
    """Fetches high-level dataset metrics and sign vocabulary counts."""
    model = await models_col().find_one({"id": model_id})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    classes_count = await classes_col().count_documents({"model_id": model_id})
    samples_count = await samples_col().count_documents({"model_id": model_id, "status": "VALID"})
    total_samples = await samples_col().count_documents({"model_id": model_id})
    images_count = await samples_col().count_documents({"model_id": model_id, "media_type": "IMAGE"})
    videos_count = await samples_col().count_documents({"model_id": model_id, "media_type": "VIDEO"})

    classes = await classes_col().find({"model_id": model_id}).to_list(100)
    class_names = [c["name"] for c in classes]

    return {
        "model_id": model_id,
        "model_name": model.get("name"),
        "total_samples": total_samples,
        "valid_samples": samples_count,
        "classes_count": classes_count,
        "class_names": class_names,
        "images_count": images_count,
        "videos_count": videos_count
    }


@router.get("/eda")
async def get_eda_report(model_id: str):
    """
    Computes Exploratory Data Analysis statistics and renders Matplotlib & Seaborn
    visualizations from real MongoDB Atlas dataset records.
    """
    try:
        report = await EDAService.generate_eda_report(model_id)
        return report
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"EDA generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"EDA calculation failed: {str(e)}")


@router.post("/train")
async def train_classical_models(model_id: str, payload: TrainingPayload | None = None):
    """
    Trains and tunes KNN, SVM, and Decision Tree on real dataset samples
    with participant-independent splitting and comparative evaluation against Custom CNN.
    """
    try:
        knn_p = payload.knn if payload else None
        svm_p = payload.svm if payload else None
        dt_p = payload.dt if payload else None
        inc_cnn = payload.include_cnn if payload else True

        result = await ClassicalMLEngine.train_and_evaluate_all(
            model_id=model_id,
            knn_params=knn_p,
            svm_params=svm_p,
            dt_params=dt_p,
            include_cnn=inc_cnn
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Classical ML training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training pipeline error: {str(e)}")


@router.get("/comparison")
async def get_model_comparison(model_id: str):
    """
    Retrieves the latest comparative evaluation results, metrics table,
    confusion matrices, and best-performing model designation.
    """
    # Check local artifact file first
    storage_dir = get_classical_storage_dir(model_id)
    meta_file = storage_dir / "classical_meta.json"
    if meta_file.exists():
        import json
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return {
            "has_results": True,
            "comparison_table": meta.get("comparison_table", []),
            "best_model": meta.get("best_model"),
            "split_info": meta.get("split_info"),
            "models_detail": meta.get("models", {}),
            "class_names": meta.get("class_names", []),
            "trained_at": meta.get("trained_at")
        }

    # Fallback to Atlas ml_experiments
    exp = await experiments_col().find_one(
        {"model_id": model_id, "algorithm": "COMPARATIVE_ML_STUDY"},
        sort=[("created_at", -1)]
    )
    if exp:
        exp.pop("_id", None)
        return {
            "has_results": True,
            "comparison_table": exp.get("comparison_table", []),
            "best_model": exp.get("best_model"),
            "split_info": exp.get("split_info"),
            "models_detail": exp.get("models_detail", {}),
            "class_names": exp.get("dataset_snapshot", {}).get("classes", []),
            "trained_at": exp.get("created_at")
        }

    return {
        "has_results": False,
        "message": "No classical model training runs completed yet. Run training to generate comparisons."
    }


@router.post("/cluster")
async def run_kmeans_clustering(model_id: str, payload: ClusterPayload):
    """
    Executes unsupervised K-Means clustering on normalized landmark vectors,
    projects clusters to 2D via PCA, and evaluates alignment with ground-truth labels.
    """
    try:
        result = await KMeansClusteringService.run_clustering(
            model_id=model_id,
            n_clusters=payload.n_clusters
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"K-Means clustering failed: {e}")
        raise HTTPException(status_code=500, detail=f"Clustering error: {str(e)}")


@router.get("/experiments")
async def get_experiment_history(model_id: str):
    """Lists past logged experiments for this model from MongoDB Atlas."""
    cursor = experiments_col().find({"model_id": model_id}).sort("created_at", -1).limit(50)
    experiments = await cursor.to_list(50)
    for exp in experiments:
        exp.pop("_id", None)
    return {
        "count": len(experiments),
        "experiments": experiments
    }


@router.post("/predict")
async def predict_with_classical_model(
    model_id: str,
    algorithm: str = Form("SVM"),
    file: UploadFile | None = File(None),
    sample_id: str | None = Form(None)
):
    """
    Runs classical ML inference (KNN, SVM, Decision Tree) on an uploaded image
    or dataset sample. Enforces hand presence gating (returns NO_HAND if absent).
    """
    image_bytes = None
    if sample_id:
        sample = await samples_col().find_one({"model_id": model_id, "id": sample_id})
        if not sample:
            raise HTTPException(status_code=404, detail="Dataset sample not found")
        image_bytes = await fetch_sample_bytes(sample)
    elif file:
        image_bytes = await file.read()
    else:
        raise HTTPException(status_code=400, detail="Provide either file upload or sample_id")

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image data")

    try:
        res = ClassicalMLEngine.predict_sample(
            model_id=model_id,
            image_bytes=image_bytes,
            algorithm=algorithm
        )
        return res
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=400, detail=str(fnf))
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
