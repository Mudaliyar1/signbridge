import base64
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, status
from pydantic import BaseModel

from app.database import models_col, classes_col, samples_col, testing_results_col
from app.models.testing_result import TestingResult, TopPrediction
from app.ml.inference import CustomSignInferenceService
from app.ml.training import fetch_sample_bytes
from app.ml.preprocessing import validate_and_inspect_image

logger = logging.getLogger("signbridge.api_testing")
router = APIRouter(prefix="/api/models/{model_id}/testing", tags=["Testing Studio"])

@router.get("/status")
async def get_testing_status(model_id: str):
    """Verifies whether a trained model checkpoint exists for inference."""
    model = await models_col().find_one({"id": model_id})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    avail = CustomSignInferenceService.check_model_availability(model_id)
    return {
        "model_id": model_id,
        "model_name": model.get("name"),
        "model_status": model.get("status"),
        "availability": avail
    }

@router.post("/image")
async def test_image_inference(
    model_id: str,
    file: UploadFile | None = File(None),
    sample_id: str | None = Form(None),
    top_k: int = Form(3)
):
    """
    Runs scratch CustomSignCNN inference on an uploaded image or an existing dataset sample.
    Compares Expected vs Predicted if dataset sample is selected.
    Persists test result in MongoDB testing_results collection.
    """
    # 1. Model Availability Check
    avail = CustomSignInferenceService.check_model_availability(model_id)
    if not avail["available"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=avail.get("message", "No trained model available. Collect dataset samples and train the model before running inference.")
        )

    image_bytes = None
    expected_class_name = None
    expected_class_id = None
    origin_sample_id = None

    if sample_id:
        sample = await samples_col().find_one({"model_id": model_id, "id": sample_id})
        if not sample:
            raise HTTPException(status_code=404, detail=f"Dataset sample '{sample_id}' not found.")
        image_bytes = await fetch_sample_bytes(sample)
        if not image_bytes:
            raise HTTPException(status_code=502, detail="Failed to retrieve media file for this dataset sample.")
        origin_sample_id = sample["id"]
        expected_class_id = sample.get("class_id")
        expected_class_name = sample.get("label")
    elif file:
        image_bytes = await file.read()
        if not image_bytes or len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file received.")
        try:
            validate_and_inspect_image(image_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image content: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Must provide either an uploaded image file or a dataset sample_id.")

    # 2. Run Inference
    try:
        res = CustomSignInferenceService.predict_image(model_id, image_bytes, top_k=top_k)
    except Exception as e:
        logger.error(f"Inference error for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    pred = res.get("prediction")
    is_correct = None
    if pred and expected_class_name:
        is_correct = (expected_class_name.strip().upper() == pred["class_name"].strip().upper())
    elif not pred and expected_class_name:
        is_correct = False

    # 3. Persist to testing_results
    test_record = TestingResult(
        model_id=model_id,
        model_version_id=res.get("model_version"),
        dataset_sample_id=origin_sample_id,
        predicted_class_id=pred["class_id"] if pred else None,
        predicted_class_name=pred["class_name"] if pred else "NO_HAND",
        expected_class_id=expected_class_id,
        expected_class_name=expected_class_name,
        confidence=pred["confidence"] if pred else 0.0,
        is_correct=is_correct,
        top_predictions=[
            TopPrediction(class_id=p["class_id"], class_name=p["class_name"], confidence=p["confidence"])
            for p in res.get("top_predictions", [])
        ] if pred else [],
        test_type="DATASET_SAMPLE" if origin_sample_id else "IMAGE_UPLOAD",
        inference_time_ms=res.get("inference_time_ms", 0.0)
    )
    await testing_results_col().insert_one(test_record.model_dump())

    return {
        "success": True,
        "status": res.get("status", "HAND_CONFIRMED" if pred else "NO_HAND"),
        "hand_present": res.get("hand_present", pred is not None),
        "test_type": test_record.test_type,
        "prediction": pred,
        "actual_model_input_base64": res.get("actual_model_input_base64"),
        "metrics": res.get("metrics"),
        "message": res.get("message"),
        "expected": {
            "class_id": expected_class_id,
            "class_name": expected_class_name
        } if expected_class_name else None,
        "is_correct": is_correct,
        "result_label": "CORRECT" if is_correct is True else ("INCORRECT" if is_correct is False else "UNLABELED"),
        "top_predictions": res.get("top_predictions", []),
        "model_version": res.get("model_version"),
        "inference_time_ms": res.get("inference_time_ms")
    }

@router.post("/video")
async def test_video_inference(
    model_id: str,
    file: UploadFile | None = File(None),
    sample_id: str | None = Form(None),
    sample_fps: float = Form(2.0),
    top_k: int = Form(3)
):
    """
    Frame-Based Video Test using static CustomSignCNN.
    Samples frames at regular interval, evaluates each frame, and aggregates via confidence-weighted voting.
    """
    # 1. Model Availability Check
    avail = CustomSignInferenceService.check_model_availability(model_id)
    if not avail["available"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=avail.get("message", "No trained model available. Collect dataset samples and train the model before running inference.")
        )

    video_bytes = None
    expected_class_name = None
    expected_class_id = None
    origin_sample_id = None

    if sample_id:
        sample = await samples_col().find_one({"model_id": model_id, "id": sample_id})
        if not sample:
            raise HTTPException(status_code=404, detail=f"Dataset sample '{sample_id}' not found.")
        video_bytes = await fetch_sample_bytes(sample)
        if not video_bytes:
            raise HTTPException(status_code=502, detail="Failed to retrieve media file for this dataset sample.")
        origin_sample_id = sample["id"]
        expected_class_id = sample.get("class_id")
        expected_class_name = sample.get("label")
    elif file:
        video_bytes = await file.read()
        if not video_bytes or len(video_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty video file received.")
    else:
        raise HTTPException(status_code=400, detail="Must provide either an uploaded video file or a dataset sample_id.")

    # 2. Run Frame-Based Inference
    try:
        res = CustomSignInferenceService.predict_video_frames(
            model_id=model_id,
            video_bytes_or_path=video_bytes,
            sample_fps=sample_fps,
            top_k=top_k
        )
    except Exception as e:
        logger.error(f"Video inference error for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Video inference failed: {str(e)}")

    final_pred = res.get("final_prediction", {})
    is_correct = None
    if expected_class_name and final_pred.get("class_name"):
        is_correct = (expected_class_name.strip().upper() == final_pred["class_name"].strip().upper())
    elif not final_pred.get("class_id") and expected_class_name:
        is_correct = False

    # 3. Persist to testing_results
    test_record = TestingResult(
        model_id=model_id,
        model_version_id="v1.0.0",
        dataset_sample_id=origin_sample_id,
        predicted_class_id=final_pred.get("class_id"),
        predicted_class_name=final_pred.get("class_name", "NO_HAND"),
        expected_class_id=expected_class_id,
        expected_class_name=expected_class_name,
        confidence=final_pred.get("confidence", 0.0),
        is_correct=is_correct,
        top_predictions=[
            TopPrediction(class_id=p["class_id"], class_name=p["class_name"], confidence=p["confidence"])
            for p in res.get("top_predictions", [])
        ],
        test_type="VIDEO_UPLOAD",
        inference_time_ms=res.get("inference_time_ms", 0.0)
    )
    await testing_results_col().insert_one(test_record.model_dump())

    res["expected"] = {
        "class_id": expected_class_id,
        "class_name": expected_class_name
    } if expected_class_name else None
    res["is_correct"] = is_correct
    res["result_label"] = "CORRECT" if is_correct is True else ("INCORRECT" if is_correct is False else "UNLABELED")
    return res

@router.post("/evaluate")
async def evaluate_dataset_samples(
    model_id: str,
    split: str | None = Query(None),
    class_id: str | None = Query(None),
    participant_id: str | None = Query(None)
):
    """
    Evaluates labeled dataset samples against the trained model.
    Calculates genuine Accuracy, Macro Precision, Recall, F1, and Confusion Matrix.
    """
    avail = CustomSignInferenceService.check_model_availability(model_id)
    if not avail["available"]:
        raise HTTPException(status_code=400, detail=avail.get("message"))

    query = {"model_id": model_id, "status": "VALID"}
    if split and split.upper() != "ALL":
        query["split"] = split.upper()
    if class_id:
        query["class_id"] = class_id
    if participant_id:
        query["participant_id"] = participant_id

    samples = await samples_col().find(query).limit(100).to_list(100)
    if not samples:
        return {
            "success": True,
            "total_samples": 0,
            "sufficient_samples": False,
            "message": "No matching labeled dataset samples found with the selected filters."
        }

    classes = await classes_col().find({"model_id": model_id}).to_list(100)
    classes_names = [c["name"] for c in classes]

    evaluations = []
    for s in samples:
        expected_label = s.get("label", "").strip().upper()
        if not expected_label:
            continue

        media_bytes = await fetch_sample_bytes(s)
        if not media_bytes:
            continue

        media_type = s.get("media_type", "IMAGE").upper()
        try:
            if media_type == "IMAGE":
                pred_res = CustomSignInferenceService.predict_image(model_id, media_bytes, top_k=1)
                predicted_label = pred_res["prediction"]["class_name"].strip().upper()
                conf = pred_res["prediction"]["confidence"]
            else:
                pred_res = CustomSignInferenceService.predict_video_frames(model_id, media_bytes, sample_fps=1.0, top_k=1)
                predicted_label = pred_res["final_prediction"]["class_name"].strip().upper()
                conf = pred_res["final_prediction"]["confidence"]

            is_corr = (expected_label == predicted_label)
            evaluations.append({
                "sample_id": s["id"],
                "expected": expected_label,
                "predicted": predicted_label,
                "confidence": conf,
                "is_correct": is_corr,
                "media_type": media_type,
                "participant_id": s.get("participant_id")
            })
        except Exception as e:
            logger.warning(f"Failed evaluation on sample {s['id']}: {e}")

    metrics = CustomSignInferenceService.calculate_evaluation_metrics(evaluations, classes_names)
    metrics["evaluations"] = evaluations
    return metrics

@router.post("/live-frame")
async def live_frame_inference(
    model_id: str,
    file: UploadFile | None = File(None),
    frame_base64: str | None = Form(None)
):
    """
    Lightweight endpoint for real-time live camera feed.
    Accepts snapshot frame, returns instantaneous prediction, confidence, and top-3 classes.
    """
    avail = CustomSignInferenceService.check_model_availability(model_id)
    if not avail["available"]:
        return {
            "success": False,
            "reason": avail.get("reason", "NO_TRAINED_MODEL"),
            "retrain_required": avail.get("retrain_required", False),
            "message": avail.get("message", "Model not trained yet.")
        }

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
        res = CustomSignInferenceService.predict_image(model_id, image_bytes, top_k=3)
        return res
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@router.get("/history")
async def get_test_history(model_id: str, limit: int = 50):
    """Returns stored test results from MongoDB testing_results."""
    cursor = testing_results_col().find({"model_id": model_id}).sort("created_at", -1).limit(limit)
    results = await cursor.to_list(limit)
    for r in results:
        r.pop("_id", None)
    return {"results": results, "total": len(results)}
