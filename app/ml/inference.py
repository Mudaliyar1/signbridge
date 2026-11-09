import io
import time
import json
import base64
import logging
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.config import settings
from app.ml.architectures import CustomSignCNN, build_model
from app.ml.preprocessing import (
    preprocess_image,
    extract_hand_roi,
    detect_hand_presence,
    HandLocalizationService,
    DEFAULT_HAND_ROI,
    PREPROCESSING_VERSION
)

logger = logging.getLogger("signbridge.inference")

class CustomSignInferenceService:
    """
    Production-grade Inference Service for SignBridge.
    Strict zero-pretrained models policy.
    Uses CustomSignCNN trained from scratch on user dataset.
    """
    _loaded_models: dict[str, dict] = {}

    @classmethod
    def get_device(cls) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def get_model_weights_paths(cls, model_id: str) -> list[Path]:
        """Returns candidate paths where trained model weights and metadata might reside."""
        return [
            settings.MODELS_DIR / model_id / "versions" / "v1.0.0",
            settings.MODELS_DIR / "versions" / model_id / "final",
            settings.MODELS_DIR / model_id
        ]

    @classmethod
    def check_model_availability(cls, model_id: str) -> dict:
        """
        Strictly verifies whether a genuine trained model checkpoint exists.
        Verifies that the checkpoint matches the required PREPROCESSING_VERSION ("hand_roi_v2").
        Outdated or contaminated checkpoints are flagged as RETRAIN REQUIRED.
        """
        for base_path in cls.get_model_weights_paths(model_id):
            weights_file = base_path / "model.pt"
            meta_file = base_path / "model_meta.json"
            if weights_file.exists() and weights_file.is_file():
                try:
                    meta = {}
                    if meta_file.exists():
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)

                    # Strict Preprocessing Version Check
                    prep_version = meta.get("preprocessing_version")
                    if prep_version != PREPROCESSING_VERSION:
                        return {
                            "available": False,
                            "status": "RETRAIN REQUIRED",
                            "reason": "RETRAIN_REQUIRED",
                            "retrain_required": True,
                            "message": (
                                "RETRAIN REQUIRED: The current model was trained using the previous input "
                                "preprocessing and cannot be trusted for hand-focused inference. Please retrain."
                            ),
                            "weights_path": str(weights_file),
                            "meta_path": str(meta_file) if meta_file.exists() else None,
                            "version": meta.get("version", "v1.0.0"),
                            "num_classes": meta.get("num_classes", 0),
                            "classes": meta.get("class_names", []),
                            "architecture": meta.get("architecture", "CustomSignCNN"),
                        }

                    return {
                        "available": True,
                        "status": "READY FOR TESTING",
                        "retrain_required": False,
                        "weights_path": str(weights_file),
                        "meta_path": str(meta_file) if meta_file.exists() else None,
                        "version": meta.get("version", "v1.0.0"),
                        "num_classes": meta.get("num_classes", 0),
                        "classes": meta.get("class_names", []),
                        "architecture": meta.get("architecture", "CustomSignCNN"),
                        "input_resolution": meta.get("input_resolution", "224x224"),
                        "preprocessing_version": prep_version
                    }
                except Exception as e:
                    logger.error(f"Error inspecting model at {base_path}: {e}")
                    return {
                        "available": False,
                        "status": "CORRUPT_WEIGHTS",
                        "message": f"Trained model checkpoint found at {base_path}, but could not be parsed: {str(e)}"
                    }

        return {
            "available": False,
            "status": "NO TRAINED MODEL AVAILABLE",
            "retrain_required": False,
            "message": "No trained model available. Collect dataset samples and train the model before running inference."
        }

    @classmethod
    def load_model(cls, model_id: str) -> dict:
        """
        Loads the trained CustomSignCNN and class mapping into memory cache.
        Raises FileNotFoundError if weights do not exist or retrain is required.
        """
        if model_id in cls._loaded_models:
            return cls._loaded_models[model_id]

        avail = cls.check_model_availability(model_id)
        if not avail["available"]:
            raise FileNotFoundError(avail.get("message", "Trained model weights not available."))

        weights_path = Path(avail["weights_path"])
        meta_path = Path(avail["meta_path"]) if avail.get("meta_path") else None
        
        meta = {
            "num_classes": 2,
            "class_mapping": {},      # "0": "NAME"
            "class_id_mapping": {},   # "0": "UUID"
            "class_names": [],
            "version": "v1.0.0"
        }
        if meta_path and meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta.update(json.load(f))

        device = cls.get_device()
        num_classes = max(2, meta.get("num_classes", 2))
        model = build_model(num_classes=num_classes)

        logger.info(f"Loading PyTorch checkpoint from {weights_path} onto {device}...")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        entry = {
            "model": model,
            "meta": meta,
            "device": device,
            "loaded_at": time.time()
        }
        cls._loaded_models[model_id] = entry
        return entry

    @classmethod
    def unload_model(cls, model_id: str):
        """Clears cached model from memory."""
        if model_id in cls._loaded_models:
            del cls._loaded_models[model_id]

    @classmethod
    def predict_image(cls, model_id: str, image_bytes: bytes, top_k: int = 3) -> dict:
        """
        Hand-Focused Inference with Classical Hand Presence Gate.
        
        Execution Flow:
        1. Decode raw image bytes to RGB
        2. Crop controlled Hand ROI (top 45% strictly excluded)
        3. Hand Presence Gate (Classical CV skin occupancy & contour analysis)
           -> If NO HAND: Return immediately (no CNN evaluation, no class, no TTS).
        4. If HAND CONFIRMED:
           -> Resize to 224x224, normalize to [0, 1]
           -> Feed to scratch-trained CustomSignCNN
           -> Return predicted class, confidence, and output text
        """
        start_time = time.perf_counter()

        avail = cls.check_model_availability(model_id)
        if not avail["available"]:
            return {
                "success": False,
                "status": avail.get("status", "RETRAIN_REQUIRED"),
                "retrain_required": avail.get("retrain_required", False),
                "error": avail.get("message", "Model not ready for inference. Retraining required.")
            }

        entry = cls.load_model(model_id)
        model: CustomSignCNN = entry["model"]
        meta = entry["meta"]
        device = entry["device"]

        # Decode image bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Unable to decode image bytes for inference")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1. Run Dynamic Hand Landmark & Presence Detection
        det = HandLocalizationService.detect_hands(img_rgb)
        is_hand_present = det["hand_present"]
        primary_hand = det.get("primary_hand")

        if is_hand_present and primary_hand and primary_hand.get("pixel_bbox"):
            py1, px1, py2, px2 = primary_hand["pixel_bbox"]
            h, w = img_rgb.shape[:2]
            py1 = max(0, min(h - 1, py1))
            py2 = max(py1 + 10, min(h, py2))
            px1 = max(0, min(w - 1, px1))
            px2 = max(px1 + 10, min(w, px2))
            roi_rgb = img_rgb[py1:py2, px1:px2]
        else:
            roi_rgb = extract_hand_roi(img_rgb, roi=DEFAULT_HAND_ROI)

        # Generate exact 224x224 preview JPEG for admin debugging
        preview_resized = cv2.resize(roi_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        preview_bgr = cv2.cvtColor(preview_resized, cv2.COLOR_RGB2BGR)
        _, preview_enc = cv2.imencode(".jpg", preview_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        preview_base64 = base64.b64encode(preview_enc).decode("utf-8")

        # 2. IF NO HAND DETECTED: Return immediately without running CustomSignCNN!
        if not is_hand_present:
            return {
                "success": True,
                "status": "NO_HAND",
                "hand_present": False,
                "prediction": None,
                "confidence": 0.0,
                "speak": False,
                "message": "NO HAND DETECTED",
                "hands": [],
                "primary_hand": None,
                "bbox": None,
                "detection_method": det.get("detection_method", "none"),
                "actual_model_input_base64": f"data:image/jpeg;base64,{preview_base64}",
                "inference_time_ms": round((time.perf_counter() - start_time) * 1000.0, 2)
            }

        # 3. Hand IS present: Run CustomSignCNN inference
        norm = preview_resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logits = model(tensor)
            probabilities = F.softmax(logits, dim=1)[0].cpu().numpy()

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Map probabilities to classes
        class_mapping: dict[str, str] = meta.get("class_mapping", {})
        class_id_mapping: dict[str, str] = meta.get("class_id_mapping", {})
        output_text_mapping: dict[str, str] = meta.get("output_text_mapping", {})
        num_classes = len(probabilities)

        sorted_indices = np.argsort(probabilities)[::-1]
        top_preds = []
        for idx in sorted_indices[:min(top_k, num_classes)]:
            c_name = class_mapping.get(str(idx), f"Class_{idx}")
            c_id = class_id_mapping.get(str(idx), f"id_{idx}")
            c_out = output_text_mapping.get(str(idx), c_name.replace("_", " ").title())
            top_preds.append({
                "class_id": c_id,
                "class_name": c_name,
                "output_text": c_out,
                "confidence": float(round(probabilities[idx], 4))
            })

        best_idx = int(sorted_indices[0])
        best_name = class_mapping.get(str(best_idx), f"Class_{best_idx}")
        best_id = class_id_mapping.get(str(best_idx), f"id_{best_idx}")
        best_output = output_text_mapping.get(str(best_idx), best_name.replace("_", " ").title())
        best_conf = float(round(probabilities[best_idx], 4))

        return {
            "success": True,
            "status": "HAND_DETECTED",
            "hand_present": True,
            "hands": det.get("hands", []),
            "primary_hand": primary_hand,
            "bbox": det.get("bbox"),
            "detection_method": det.get("detection_method", "mediapipe_landmarks"),
            "actual_model_input_base64": f"data:image/jpeg;base64,{preview_base64}",
            "prediction": {
                "class_id": best_id,
                "class_name": best_name,
                "output_text": best_output,
                "confidence": best_conf,
                "hand_roi_applied": True,
                "handedness": primary_hand.get("handedness") if primary_hand else "Right",
                "hand_bbox": det.get("bbox")
            },
            "top_predictions": top_preds,
            "all_probabilities": [float(p) for p in probabilities],
            "model_version": meta.get("version", "v1.0.0"),
            "inference_time_ms": round(latency_ms, 2)
        }

    @classmethod
    def predict_video_frames(
        cls,
        model_id: str,
        video_bytes_or_path: bytes | Path | str,
        sample_fps: float = 2.0,
        top_k: int = 3
    ) -> dict:
        """
        Frame-Based Video Test using static CustomSignCNN.
        Samples frames at regular interval, runs inference per frame, and aggregates via confidence-weighted voting.
        """
        start_time = time.perf_counter()
        entry = cls.load_model(model_id)
        meta = entry["meta"]

        temp_file = None
        if isinstance(video_bytes_or_path, bytes):
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_file.write(video_bytes_or_path)
            temp_file.close()
            path_to_open = temp_file.name
        else:
            path_to_open = str(video_bytes_or_path)

        cap = cv2.VideoCapture(path_to_open)
        if not cap.isOpened():
            if temp_file:
                Path(temp_file.name).unlink(missing_ok=True)
            raise ValueError("Could not open video file with OpenCV.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = round(total_frames / fps, 2) if fps > 0 else 0.0

        # Calculate frame step interval based on sample_fps
        frame_interval = max(1, int(fps / sample_fps))
        frame_idx = 0
        analyzed_frames = []
        accumulated_probs = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Convert BGR frame directly to bytes for inference
                success, encoded = cv2.imencode(".jpg", frame)
                if success:
                    res = cls.predict_image(model_id, encoded.tobytes(), top_k=top_k)
                    timestamp = round(frame_idx / fps, 2)

                    if res.get("hand_present") and res.get("prediction"):
                        probs = np.array(res["all_probabilities"])
                        if accumulated_probs is None:
                            accumulated_probs = probs.copy()
                        else:
                            accumulated_probs += probs

                        analyzed_frames.append({
                            "frame_index": frame_idx,
                            "timestamp_sec": timestamp,
                            "class_name": res["prediction"]["class_name"],
                            "class_id": res["prediction"]["class_id"],
                            "output_text": res["prediction"].get("output_text"),
                            "confidence": res["prediction"]["confidence"],
                            "hand_present": True
                        })
                    else:
                        analyzed_frames.append({
                            "frame_index": frame_idx,
                            "timestamp_sec": timestamp,
                            "class_name": "NO HAND DETECTED",
                            "class_id": "",
                            "output_text": "",
                            "confidence": 0.0,
                            "hand_present": False
                        })

            frame_idx += 1
            if len(analyzed_frames) >= 60:  # Safety cap at 60 sampled frames
                break

        cap.release()
        if temp_file:
            Path(temp_file.name).unlink(missing_ok=True)

        # Check if any frames had a hand detected
        hand_frames = [f for f in analyzed_frames if f.get("hand_present")]
        if not hand_frames or accumulated_probs is None:
            return {
                "success": True,
                "status": "NO_HAND",
                "hand_present": False,
                "message": "NO HAND DETECTED across sampled video frames",
                "total_frames_analyzed": len(analyzed_frames),
                "hand_frames_count": 0,
                "frame_predictions": analyzed_frames,
                "final_prediction": {
                    "class_name": "NO HAND DETECTED",
                    "output_text": "",
                    "confidence": 0.0
                },
                "top_predictions": [],
                "inference_time_ms": round((time.perf_counter() - start_time) * 1000.0, 2)
            }

        # Aggregate: Confidence-weighted average
        avg_probs = accumulated_probs / len(analyzed_frames)
        class_mapping: dict[str, str] = meta.get("class_mapping", {})
        class_id_mapping: dict[str, str] = meta.get("class_id_mapping", {})
        output_text_mapping: dict[str, str] = meta.get("output_text_mapping", {})

        sorted_indices = np.argsort(avg_probs)[::-1]
        top_preds = []
        for idx in sorted_indices[:min(top_k, len(avg_probs))]:
            c_name = class_mapping.get(str(idx), f"Class_{idx}")
            c_id = class_id_mapping.get(str(idx), f"id_{idx}")
            c_out = output_text_mapping.get(str(idx), c_name.replace("_", " ").title())
            top_preds.append({
                "class_id": c_id,
                "class_name": c_name,
                "output_text": c_out,
                "confidence": float(round(avg_probs[idx], 4))
            })

        best_idx = int(sorted_indices[0])
        best_name = class_mapping.get(str(best_idx), f"Class_{best_idx}")
        final_prediction = {
            "class_id": class_id_mapping.get(str(best_idx), f"id_{best_idx}"),
            "class_name": best_name,
            "output_text": output_text_mapping.get(str(best_idx), best_name.replace("_", " ").title()),
            "confidence": float(round(avg_probs[best_idx], 4))
        }

        total_latency = (time.perf_counter() - start_time) * 1000.0

        return {
            "success": True,
            "test_type": "FRAME_BASED_VIDEO",
            "hand_roi_applied": True,
            "roi_coordinates": [0.25, 0.20, 0.90, 0.80],
            "method_label": "Frame-Based Video Inference (Hand ROI Focused)",
            "architecture_note": "This is frame-based evaluation with confidence-weighted voting. It does not claim temporal sign recognition.",
            "final_prediction": final_prediction,
            "top_predictions": top_preds,
            "total_frames_analyzed": len(analyzed_frames),
            "video_metadata": {
                "duration_seconds": duration,
                "fps": round(fps, 1),
                "resolution": f"{width}x{height}",
                "total_frames": total_frames
            },
            "frame_predictions": analyzed_frames,
            "inference_time_ms": round(total_latency, 2)
        }

    @classmethod
    def calculate_evaluation_metrics(cls, evaluations: list[dict], classes_list: list[str]) -> dict:
        """
        Calculates genuine mathematical evaluation metrics:
        Accuracy, Macro Precision, Macro Recall, Macro F1, Per-class accuracy, and Confusion Matrix.
        No placeholder or fake numbers.
        """
        total = len(evaluations)
        if total == 0:
            return {
                "total_samples": 0,
                "sufficient_samples": False,
                "message": "Insufficient test samples for reliable evaluation."
            }

        correct = sum(1 for e in evaluations if e.get("is_correct") is True)
        incorrect = total - correct
        accuracy = round((correct / total) * 100.0, 2)

        # Unique labels present in dataset or classes
        unique_labels = sorted(list(set(classes_list + [e["expected"] for e in evaluations])))
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        n_labels = len(unique_labels)

        # Confusion matrix [actual, predicted]
        cm = [[0 for _ in range(n_labels)] for _ in range(n_labels)]
        for e in evaluations:
            act_idx = label_to_idx.get(e["expected"])
            pred_idx = label_to_idx.get(e["predicted"])
            if act_idx is not None and pred_idx is not None:
                cm[act_idx][pred_idx] += 1

        # Per-class metrics
        per_class = {}
        precisions = []
        recalls = []
        f1s = []

        for label in unique_labels:
            i = label_to_idx[label]
            tp = cm[i][i]
            fp = sum(cm[r][i] for r in range(n_labels) if r != i)
            fn = sum(cm[i][c] for c in range(n_labels) if c != i)
            total_actual = tp + fn

            p = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            r = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)

            per_class[label] = {
                "total_samples": total_actual,
                "correct": tp,
                "accuracy": round((tp / total_actual) * 100.0, 2) if total_actual > 0 else 0.0,
                "precision": round(p * 100.0, 2),
                "recall": round(r * 100.0, 2),
                "f1_score": round(f1 * 100.0, 2)
            }

        macro_precision = round((sum(precisions) / len(precisions)) * 100.0, 2) if precisions else 0.0
        macro_recall = round((sum(recalls) / len(recalls)) * 100.0, 2) if recalls else 0.0
        macro_f1 = round((sum(f1s) / len(f1s)) * 100.0, 2) if f1s else 0.0

        sufficient = total >= 3 and len(unique_labels) >= 2

        return {
            "total_samples": total,
            "correct_predictions": correct,
            "incorrect_predictions": incorrect,
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "per_class": per_class,
            "confusion_matrix": {
                "labels": unique_labels,
                "matrix": cm
            },
            "sufficient_samples": sufficient,
            "message": "Evaluation completed with real test metrics." if sufficient else "Insufficient test samples for reliable evaluation (minimum 3 samples across 2+ classes recommended)."
        }
