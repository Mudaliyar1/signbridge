"""
SignBridge Canonical Hand Landmark Feature Extraction Layer.

Reproducible 63-dimensional feature representation for classical ML:
KNN, SVM, and Decision Tree.
Extracts 21 3D hand landmarks via MediaPipe HandLandmarker.
Applies wrist-centered translation normalization and Euclidean scale normalization.
Strict presence gating: returns None / NO_HAND if no hand is detected. Zero fabricated values.
"""

import logging
from typing import Any
import cv2
import numpy as np

from app.ml.preprocessing import HandLocalizationService

logger = logging.getLogger("signbridge.ml.features")

FEATURE_VERSION = "landmark_63_v1"
FEATURE_DIMENSION = 63  # 21 landmarks * 3 coordinates (x, y, z)
LANDMARK_COUNT = 21

# Canonical feature column names for documentation, explainability, and tabular reporting
FEATURE_NAMES = [
    f"landmark_{i}_{axis}"
    for i in range(LANDMARK_COUNT)
    for axis in ("x", "y", "z")
]

# MediaPipe landmark indices definition
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]


def normalize_landmark_coordinates(landmarks: list[dict[str, float]]) -> np.ndarray:
    """
    Transforms raw 21-landmark (x, y, z) coordinates into an invariant 63-dimensional vector:
    1. Wrist-centered translation normalization (wrist = landmark 0 becomes (0, 0, 0))
    2. Scale normalization by maximum Euclidean distance from the wrist
    
    Returns:
        np.ndarray of shape (63,), float32.
    """
    if len(landmarks) != LANDMARK_COUNT:
        raise ValueError(f"Expected exactly {LANDMARK_COUNT} landmarks, got {len(landmarks)}")

    # Extract coordinates into (21, 3) matrix
    coords = np.zeros((LANDMARK_COUNT, 3), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        coords[i, 0] = float(lm.get("x", 0.0))
        coords[i, 1] = float(lm.get("y", 0.0))
        coords[i, 2] = float(lm.get("z", 0.0))

    # 1. Translation Invariance: Shift all landmarks so wrist (landmark 0) is at origin
    wrist = coords[0].copy()
    shifted = coords - wrist

    # 2. Scale Invariance: Scale by maximum Euclidean distance from wrist
    distances = np.linalg.norm(shifted, axis=1)
    max_dist = float(np.max(distances))
    scale = max_dist if max_dist > 1e-6 else 1.0
    normalized = shifted / scale

    # Flatten to (63,)
    feature_vector = normalized.flatten().astype(np.float32)
    return feature_vector


def extract_features_from_image(
    image_input: bytes | np.ndarray,
    hand_selection: str = "Auto"
) -> dict[str, Any]:
    """
    Primary end-to-end landmark feature extraction function.
    
    Accepts:
        - raw image bytes (JPEG, PNG, WebP)
        - numpy array (BGR or RGB)
        
    Returns dict:
        - hand_present: bool
        - status: "HAND_DETECTED" | "NO_HAND"
        - features: np.ndarray of shape (63,) or None
        - raw_landmarks: list of 21 dicts or None
        - bbox: normalized bounding box or None
        - handedness: "Right" | "Left" | "Auto"
        - confidence: float
    """
    # 1. Decode image to RGB if given as bytes
    if isinstance(image_input, bytes):
        if not image_input or len(image_input) == 0:
            return {
                "hand_present": False,
                "status": "NO_HAND",
                "features": None,
                "raw_landmarks": None,
                "bbox": None,
                "handedness": None,
                "confidence": 0.0,
                "reason": "EMPTY_IMAGE_BYTES"
            }
        nparr = np.frombuffer(image_input, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return {
                "hand_present": False,
                "status": "NO_HAND",
                "features": None,
                "raw_landmarks": None,
                "bbox": None,
                "handedness": None,
                "confidence": 0.0,
                "reason": "IMAGE_DECODE_FAILED"
            }
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif len(image_input.shape) == 3 and image_input.shape[2] == 4:
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = image_input
    else:
        raise TypeError("image_input must be bytes or np.ndarray")

    # 2. Run MediaPipe Hand Landmarker detection
    det = HandLocalizationService.detect_hands(img_rgb, hand_selection=hand_selection)

    if not det["hand_present"] or not det.get("primary_hand"):
        return {
            "hand_present": False,
            "status": "NO_HAND",
            "features": None,
            "raw_landmarks": None,
            "bbox": None,
            "handedness": None,
            "confidence": 0.0,
            "reason": "NO_HAND_IN_FRAME"
        }

    primary = det["primary_hand"]
    landmarks = primary.get("landmarks", [])

    if len(landmarks) != LANDMARK_COUNT:
        return {
            "hand_present": False,
            "status": "NO_HAND",
            "features": None,
            "raw_landmarks": None,
            "bbox": None,
            "handedness": primary.get("handedness"),
            "confidence": primary.get("confidence", 0.0),
            "reason": f"INCOMPLETE_LANDMARKS_{len(landmarks)}"
        }

    # 3. Compute invariant 63-dimensional normalized feature vector
    feature_vec = normalize_landmark_coordinates(landmarks)

    return {
        "hand_present": True,
        "status": "HAND_DETECTED",
        "features": feature_vec,
        "raw_landmarks": landmarks,
        "bbox": primary.get("bbox"),
        "handedness": primary.get("handedness"),
        "confidence": primary.get("confidence", 1.0),
        "reason": "HAND_CONFIRMED"
    }
