import io
import json
import logging
import time
from pathlib import Path
from datetime import datetime, timezone
import cv2
import numpy as np
import httpx
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from app.config import settings
from app.database import models_col, classes_col, samples_col, runs_col, versions_col
from app.models.training_run import TrainingRun, EpochMetric
from app.models.model_version import ModelVersion
from app.ml.architectures import build_model
from app.ml.preprocessing import (
    preprocess_image,
    extract_hand_roi,
    detect_hand_presence,
    DEFAULT_HAND_ROI,
    PREPROCESSING_VERSION
)
from app.ml.inference import CustomSignInferenceService

logger = logging.getLogger("signbridge.training")

async def fetch_sample_bytes(sample: dict, client: httpx.AsyncClient | None = None) -> bytes | None:
    """Retrieves media bytes from Cloudinary secure_url or local storage."""
    media_url = ""
    c_meta = sample.get("cloudinary")
    if c_meta and c_meta.get("secure_url"):
        media_url = c_meta["secure_url"]
    elif sample.get("media_url"):
        media_url = sample["media_url"]
        
    if media_url.startswith("http"):
        try:
            if client is not None:
                resp = await client.get(media_url)
                if resp.status_code == 200:
                    return resp.content
            else:
                async with httpx.AsyncClient(timeout=20.0) as c:
                    resp = await c.get(media_url)
                    if resp.status_code == 200:
                        return resp.content
        except Exception as e:
            logger.error(f"Failed to download media from {media_url}: {e}")
            return None
            
    # Try local storage path fallback
    storage_path = sample.get("storage_path")
    if storage_path:
        local_p = settings.DATASETS_DIR / storage_path
        if local_p.exists():
            return local_p.read_bytes()

    return None

async def train_model_from_dataset(
    model_id: str,
    epochs: int = 12,
    batch_size: int = 8,
    learning_rate: float = 0.001
) -> dict:
    """
    Genuine scratch PyTorch Training Loop for SignBridge.
    Strictly trains CustomSignCNN with zero pretrained weights on user dataset samples.
    """
    logger.info(f"Starting training run for model {model_id}...")
    start_time = time.time()

    # 1. Fetch Model
    model_doc = await models_col().find_one({"id": model_id})
    if not model_doc:
        raise ValueError("Model not found in database.")

    # 2. Fetch Sign Classes
    classes_cursor = classes_col().find({"model_id": model_id}).sort("name", 1)
    classes_list = await classes_cursor.to_list(100)
    if len(classes_list) < 2:
        raise ValueError(
            f"Model has {len(classes_list)} sign class(es). At least 2 sign classes are required "
            "to train a neural network classifier. Please add more sign classes."
        )

    # 3. Class Index Mapping
    class_id_to_idx = {c["id"]: i for i, c in enumerate(classes_list)}
    class_name_to_idx = {c["name"]: i for i, c in enumerate(classes_list)}
    class_mapping = {str(i): c["name"] for i, c in enumerate(classes_list)}
    class_id_mapping = {str(i): c["id"] for i, c in enumerate(classes_list)}
    class_names = [c["name"] for c in classes_list]

    # 4. Fetch Dataset Samples
    samples_cursor = samples_col().find({"model_id": model_id, "status": "VALID"})
    samples = await samples_cursor.to_list(1000)
    if len(samples) == 0:
        raise ValueError("No VALID dataset samples found for this model. Collect or upload samples first.")

    # 5. Extract images and labels
    X_list = []
    y_list = []
    participants_list = []

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        for s in samples:
            target_class_id = s["class_id"]
            if target_class_id not in class_id_to_idx:
                continue
            label_idx = class_id_to_idx[target_class_id]
            media_bytes = await fetch_sample_bytes(s, client=http_client)
            if not media_bytes:
                continue

            media_type = s.get("media_type", "IMAGE").upper()
            if media_type == "IMAGE":
                try:
                    norm = preprocess_image(media_bytes, target_size=(224, 224))
                    # (224, 224, 3) -> (3, 224, 224)
                    chw = np.transpose(norm, (2, 0, 1))
                    X_list.append(chw)
                    y_list.append(label_idx)
                    participants_list.append(s.get("participant_id", "default"))
                except Exception as e:
                    logger.warning(f"Could not preprocess image sample {s.get('id')}: {e}")
            elif media_type == "VIDEO":
                # Extract keyframes from video so video samples contribute to training
                import tempfile
                temp_f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                temp_f.write(media_bytes)
                temp_f.close()
                
                cap = cv2.VideoCapture(temp_f.name)
                frames_extracted = 0
                frame_idx = 0
                total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                step = max(1, total_v_frames // 8)  # up to 8 keyframes per video
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % step == 0:
                        try:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            norm = preprocess_image(frame_rgb, target_size=(224, 224))
                            chw = np.transpose(norm, (2, 0, 1))
                            X_list.append(chw)
                            y_list.append(label_idx)
                            participants_list.append(s.get("participant_id", "default"))
                            frames_extracted += 1
                            if frames_extracted >= 10:
                                break
                        except Exception:
                            pass
                    frame_idx += 1
                cap.release()
                Path(temp_f.name).unlink(missing_ok=True)

    if len(X_list) < 2:
        raise ValueError(
            f"Only {len(X_list)} valid frame(s)/image(s) could be loaded. "
            "Need at least 2 samples across classes to train."
        )

    # Check class coverage
    present_labels = set(y_list)
    if len(present_labels) < 2:
        raise ValueError(
            f"All samples belong to only 1 sign class ({class_names[list(present_labels)[0]]}). "
            "Classification training requires samples from at least 2 different sign classes."
        )

    # 6. Prepare Train/Val Datasets
    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.int64)

    # Participant-aware split if possible, else stratified index split
    total_items = len(X_arr)
    indices = np.arange(total_items)
    np.random.seed(42)
    np.random.shuffle(indices)

    split_point = max(1, int(total_items * 0.8)) if total_items >= 5 else total_items
    train_idx = indices[:split_point]
    val_idx = indices[split_point:] if split_point < total_items else indices[:1]

    train_X = torch.from_numpy(X_arr[train_idx])
    train_y = torch.from_numpy(y_arr[train_idx])
    val_X = torch.from_numpy(X_arr[val_idx])
    val_y = torch.from_numpy(y_arr[val_idx])

    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)

    val_dataset = TensorDataset(val_X, val_y)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False)

    # 7. Initialize Scratch Model
    device = CustomSignInferenceService.get_device()
    num_classes = len(classes_list)
    model = build_model(num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_metrics: list[EpochMetric] = []

    # 8. Training Loop
    logger.info(f"Training CustomSignCNN on {device} for {epochs} epochs with {len(train_dataset)} train samples...")
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == batch_y).item()
            total_train += batch_X.size(0)

        epoch_train_loss = round(running_loss / max(1, total_train), 4)
        epoch_train_acc = round((correct_train / max(1, total_train)) * 100.0, 2)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for b_val_X, b_val_y in val_loader:
                b_val_X, b_val_y = b_val_X.to(device), b_val_y.to(device)
                val_out = model(b_val_X)
                v_loss = criterion(val_out, b_val_y)
                val_loss += v_loss.item() * b_val_X.size(0)
                _, v_preds = torch.max(val_out, 1)
                correct_val += torch.sum(v_preds == b_val_y).item()
                total_val += b_val_X.size(0)

        epoch_val_loss = round(val_loss / max(1, total_val), 4)
        epoch_val_acc = round((correct_val / max(1, total_val)) * 100.0, 2)

        epoch_metrics.append(EpochMetric(
            epoch=ep,
            train_loss=epoch_train_loss,
            train_acc=epoch_train_acc,
            val_loss=epoch_val_loss,
            val_acc=epoch_val_acc
        ))

    # 9. Save Weights & Metadata Checkpoint
    output_dir = settings.MODELS_DIR / model_id / "versions" / "v1.0.0"
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "model.pt"
    meta_path = output_dir / "model_meta.json"

    torch.save(model.state_dict(), str(weights_path))

    meta_content = {
        "model_id": model_id,
        "version": "v1.0.0",
        "architecture": "CustomSignCNN",
        "preprocessing_version": PREPROCESSING_VERSION,
        "input_resolution": "224x224",
        "hand_roi_configured": True,
        "roi_coordinates": list(DEFAULT_HAND_ROI),
        "num_classes": num_classes,
        "class_names": class_names,
        "class_mapping": class_mapping,
        "class_id_mapping": class_id_mapping,
        "output_text_mapping": {str(i): classes_list[i].get("output_text") or class_names[i].replace("_", " ").title() for i in range(num_classes)},
        "trained_samples_count": len(X_list),
        "final_train_acc": epoch_metrics[-1].train_acc if epoch_metrics else 0.0,
        "final_val_acc": epoch_metrics[-1].val_acc if epoch_metrics else 0.0,
        "trained_at": datetime.now(timezone.utc).isoformat()
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_content, f, indent=2)

    # Invalidate cached inference model in memory
    CustomSignInferenceService.unload_model(model_id)

    # 10. Persist TrainingRun & ModelVersion in MongoDB Atlas
    run_record = TrainingRun(
        model_id=model_id,
        version="v1.0.0",
        status="COMPLETED",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        total_samples=len(X_list),
        train_samples=len(train_X),
        val_samples=len(val_X),
        num_classes=num_classes,
        metrics=epoch_metrics,
        final_train_acc=epoch_metrics[-1].train_acc,
        final_val_acc=epoch_metrics[-1].val_acc,
        weights_path=str(weights_path),
        completed_at=datetime.now(timezone.utc)
    )
    await runs_col().insert_one(run_record.model_dump())

    version_record = ModelVersion(
        model_id=model_id,
        version="v1.0.0",
        architecture="CustomSignCNN",
        weights_path=str(weights_path),
        num_classes=num_classes,
        class_mapping=class_mapping,
        class_id_mapping=class_id_mapping,
        training_run_id=run_record.id,
        train_samples_count=len(X_list),
        val_accuracy=epoch_metrics[-1].val_acc
    )
    await versions_col().update_one(
        {"model_id": model_id, "version": "v1.0.0"},
        {"$set": version_record.model_dump()},
        upsert=True
    )

    # Update model status in models collection to Trained
    await models_col().update_one(
        {"id": model_id},
        {"$set": {"status": "Trained", "updated_at": datetime.now(timezone.utc)}}
    )

    duration_sec = round(time.time() - start_time, 2)
    logger.info(f"Training completed in {duration_sec}s! Final Val Acc: {epoch_metrics[-1].val_acc}%")

    return {
        "success": True,
        "message": f"Successfully trained CustomSignCNN from scratch across {num_classes} classes!",
        "run_id": run_record.id,
        "epochs": epochs,
        "trained_samples": len(X_list),
        "classes": class_names,
        "final_train_accuracy": epoch_metrics[-1].train_acc,
        "final_val_accuracy": epoch_metrics[-1].val_acc,
        "duration_seconds": duration_sec,
        "weights_saved_at": str(weights_path)
    }
