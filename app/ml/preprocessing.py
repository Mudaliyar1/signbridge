import io
import logging
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

logger = logging.getLogger("signbridge.preprocessing")

# Preprocessing Pipeline Version
# Marks models trained with Hand Landmark & Dynamic Bounding Box detection
PREPROCESSING_VERSION = "hand_landmark_v1"

# Legacy fallback ROI for cases where no landmarks or skin contours can be isolated
DEFAULT_HAND_ROI = (0.45, 0.32, 0.95, 0.68)


def validate_and_inspect_image(file_bytes: bytes) -> tuple[int, int, str]:
    """
    Validates image bytes using PIL and OpenCV.
    Returns (width, height, format_name). Raises ValueError if corrupted or invalid.
    """
    if not file_bytes or len(file_bytes) == 0:
        raise ValueError("Image file is empty")
        
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img.verify()
    except Exception as e:
        raise ValueError(f"Corrupted or invalid image file: {e}")
        
    pil_img = Image.open(io.BytesIO(file_bytes))
    width, height = pil_img.size
    fmt = pil_img.format or "JPEG"
    
    if width < 10 or height < 10:
        raise ValueError("Image resolution is too small to be a usable sign sample")
        
    return width, height, fmt


class HandLocalizationService:
    """
    Dedicated Hand & Finger Landmark Localization Layer.
    Uses MediaPipe Tasks HandLandmarker for 21-point hand tracking and dynamic bounding box computation.
    
    IMPORTANT ARCHITECTURE NOTE:
    This service is strictly for hand localization and presence gating (finding WHERE the hand is).
    The actual sign classification remains CustomSignCNN (trained from scratch on user dataset samples).
    Zero pretrained sign language classifiers are used.
    """
    _detector = None
    _initialized = False

    @classmethod
    def get_detector(cls):
        if not cls._initialized and MEDIAPIPE_AVAILABLE:
            cls._initialized = True
            model_path = Path(__file__).parent / "assets" / "hand_landmarker.task"
            if model_path.exists():
                try:
                    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
                    options = vision.HandLandmarkerOptions(
                        base_options=base_options,
                        num_hands=2,
                        min_hand_detection_confidence=0.40,
                        min_hand_presence_confidence=0.40,
                        min_tracking_confidence=0.40
                    )
                    cls._detector = vision.HandLandmarker.create_from_options(options)
                    logger.info("Initialized MediaPipe HandLandmarker successfully.")
                except Exception as e:
                    logger.warning(f"Failed to initialize MediaPipe HandLandmarker: {e}")
                    cls._detector = None
        return cls._detector

    @classmethod
    def detect_hands(cls, img_rgb: np.ndarray, hand_selection: str = "Auto") -> dict:
        """
        Detects hands, 21 finger landmarks, handedness, and dynamic bounding boxes in the RGB image.
        Returns:
            dict containing:
                - hand_present: bool
                - status: "HAND_DETECTED" | "NO_HAND"
                - hands: list of detected hand dictionaries
                - primary_hand: primary hand dict (or None)
                - bbox: [ymin, xmin, ymax, xmax] normalized
                - pixel_bbox: [y1, x1, y2, x2]
                - detection_method: "mediapipe_landmarks" | "classical_contour" | "none"
        """
        if img_rgb is None or img_rgb.size == 0:
            return {
                "hand_present": False,
                "status": "NO_HAND",
                "hands": [],
                "primary_hand": None,
                "bbox": None,
                "pixel_bbox": None,
                "detection_method": "none",
                "landmarks_count": 0
            }

        h, w = img_rgb.shape[:2]

        # 1. Primary: MediaPipe Hand Landmarker
        detector = cls.get_detector()
        if detector is not None:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                result = detector.detect(mp_image)
                if result.hand_landmarks and len(result.hand_landmarks) > 0:
                    hands_list = []
                    for i, landmarks in enumerate(result.hand_landmarks):
                        lm_coords = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in landmarks]
                        xs = [lm.x for lm in landmarks]
                        ys = [lm.y for lm in landmarks]
                        
                        min_x, max_x = max(0.0, min(xs)), min(1.0, max(xs))
                        min_y, max_y = max(0.0, min(ys)), min(1.0, max(ys))
                        
                        handedness_label = "Right"
                        conf = 0.90
                        if result.handedness and i < len(result.handedness) and len(result.handedness[i]) > 0:
                            h_cat = result.handedness[i][0]
                            handedness_label = h_cat.category_name or "Right"
                            conf = float(h_cat.score)
                            
                        # Padding around hand (22%) + Make square aspect ratio
                        box_w = max_x - min_x
                        box_h = max_y - min_y
                        pad_w = box_w * 0.22
                        pad_h = box_h * 0.22
                        
                        cx = (min_x + max_x) / 2.0
                        cy = (min_y + max_y) / 2.0
                        side = max(box_w + 2 * pad_w, box_h + 2 * pad_h)
                        
                        sq_ymin = max(0.0, cy - side / 2.0)
                        sq_ymax = min(1.0, cy + side / 2.0)
                        sq_xmin = max(0.0, cx - side / 2.0)
                        sq_xmax = min(1.0, cx + side / 2.0)
                        
                        py1 = int(sq_ymin * h)
                        py2 = int(sq_ymax * h)
                        px1 = int(sq_xmin * w)
                        px2 = int(sq_xmax * w)
                        
                        area = (sq_ymax - sq_ymin) * (sq_xmax - sq_xmin)
                        
                        hands_list.append({
                            "index": i,
                            "handedness": handedness_label,
                            "confidence": round(conf, 4),
                            "landmarks": lm_coords,
                            "bbox": [round(sq_ymin, 4), round(sq_xmin, 4), round(sq_ymax, 4), round(sq_xmax, 4)],
                            "pixel_bbox": [py1, px1, py2, px2],
                            "area": area
                        })
                        
                    # Determine primary hand based on selection setting
                    primary = hands_list[0]
                    if hand_selection == "Right":
                        r_hands = [h_d for h_d in hands_list if h_d["handedness"] == "Right"]
                        primary = r_hands[0] if r_hands else max(hands_list, key=lambda x: x["area"])
                    elif hand_selection == "Left":
                        l_hands = [h_d for h_d in hands_list if h_d["handedness"] == "Left"]
                        primary = l_hands[0] if l_hands else max(hands_list, key=lambda x: x["area"])
                    else:
                        primary = max(hands_list, key=lambda x: x["area"])
                        
                    return {
                        "hand_present": True,
                        "status": "HAND_DETECTED",
                        "hands": hands_list,
                        "primary_hand": primary,
                        "bbox": primary["bbox"],
                        "pixel_bbox": primary["pixel_bbox"],
                        "detection_method": "mediapipe_landmarks",
                        "landmarks_count": 21
                    }
                else:
                    # MediaPipe successfully ran and found NO hand in this frame -> STRICT NO_HAND
                    return {
                        "hand_present": False,
                        "status": "NO_HAND",
                        "hands": [],
                        "primary_hand": None,
                        "bbox": None,
                        "pixel_bbox": None,
                        "detection_method": "mediapipe_landmarks",
                        "landmarks_count": 0
                    }
            except Exception as e:
                logger.debug(f"MediaPipe detection exception: {e}")

        # 2. Check if image is already a pre-cropped square sample (e.g. 224x224 sample)
        if abs(h - w) <= 4 and h <= 300:
            return {
                "hand_present": True,
                "status": "HAND_DETECTED",
                "hands": [{
                    "handedness": "Auto",
                    "confidence": 1.0,
                    "landmarks": [],
                    "bbox": [0.0, 0.0, 1.0, 1.0],
                    "pixel_bbox": [0, 0, h, w],
                    "area": 1.0
                }],
                "primary_hand": {
                    "handedness": "Auto",
                    "confidence": 1.0,
                    "landmarks": [],
                    "bbox": [0.0, 0.0, 1.0, 1.0],
                    "pixel_bbox": [0, 0, h, w],
                    "area": 1.0
                },
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "pixel_bbox": [0, 0, h, w],
                "detection_method": "pre_cropped_sample",
                "landmarks_count": 0
            }

        # 3. Classical Skin-Contour Fallback ONLY if MediaPipe detector is unavailable
        if detector is None:
            classical_present, occ, metrics = detect_hand_presence(img_rgb)
            if classical_present:
                return {
                    "hand_present": True,
                    "status": "HAND_DETECTED",
                "hands": [{
                    "handedness": "Auto",
                    "confidence": round(occ, 3),
                    "landmarks": [],
                    "bbox": list(DEFAULT_HAND_ROI),
                    "pixel_bbox": [
                        int(DEFAULT_HAND_ROI[0] * h),
                        int(DEFAULT_HAND_ROI[1] * w),
                        int(DEFAULT_HAND_ROI[2] * h),
                        int(DEFAULT_HAND_ROI[3] * w)
                    ],
                    "area": (DEFAULT_HAND_ROI[2] - DEFAULT_HAND_ROI[0]) * (DEFAULT_HAND_ROI[3] - DEFAULT_HAND_ROI[1])
                }],
                "primary_hand": {
                    "handedness": "Auto",
                    "confidence": round(occ, 3),
                    "landmarks": [],
                    "bbox": list(DEFAULT_HAND_ROI),
                    "pixel_bbox": [
                        int(DEFAULT_HAND_ROI[0] * h),
                        int(DEFAULT_HAND_ROI[1] * w),
                        int(DEFAULT_HAND_ROI[2] * h),
                        int(DEFAULT_HAND_ROI[3] * w)
                    ],
                    "area": (DEFAULT_HAND_ROI[2] - DEFAULT_HAND_ROI[0]) * (DEFAULT_HAND_ROI[3] - DEFAULT_HAND_ROI[1])
                },
                "bbox": list(DEFAULT_HAND_ROI),
                "pixel_bbox": [
                    int(DEFAULT_HAND_ROI[0] * h),
                    int(DEFAULT_HAND_ROI[1] * w),
                    int(DEFAULT_HAND_ROI[2] * h),
                    int(DEFAULT_HAND_ROI[3] * w)
                ],
                "detection_method": "classical_contour",
                "landmarks_count": 0
            }

        return {
            "hand_present": False,
            "status": "NO_HAND",
            "hands": [],
            "primary_hand": None,
            "bbox": None,
            "pixel_bbox": None,
            "detection_method": "none",
            "landmarks_count": 0
        }


def extract_hand_roi(
    img_rgb: np.ndarray,
    roi: tuple[float, float, float, float] | list[float] | None = None
) -> np.ndarray:
    """
    Crops the hand-focused Region of Interest from an RGB image.
    Uses dynamic HandLocalizationService if roi is not specified.
    """
    h, w = img_rgb.shape[:2]
    if abs(h - w) <= 4 and h <= 300:
        return img_rgb

    if roi is not None:
        ymin, xmin, ymax, xmax = roi
        y1 = max(0, min(h - 1, int(ymin * h)))
        y2 = max(y1 + 10, min(h, int(ymax * h)))
        x1 = max(0, min(w - 1, int(xmin * w)))
        x2 = max(x1 + 10, min(w, int(xmax * w)))
        cropped = img_rgb[y1:y2, x1:x2]
        return cropped if (cropped.size > 0 and cropped.shape[0] >= 5 and cropped.shape[1] >= 5) else img_rgb

    # Dynamic landmark detection crop
    det = HandLocalizationService.detect_hands(img_rgb)
    if det["hand_present"] and det["pixel_bbox"]:
        py1, px1, py2, px2 = det["pixel_bbox"]
        py1 = max(0, min(h - 1, py1))
        py2 = max(py1 + 10, min(h, py2))
        px1 = max(0, min(w - 1, px1))
        px2 = max(px1 + 10, min(w, px2))
        cropped = img_rgb[py1:py2, px1:px2]
        if cropped.size > 0 and cropped.shape[0] >= 5 and cropped.shape[1] >= 5:
            return cropped

    # Default fallback
    ymin, xmin, ymax, xmax = DEFAULT_HAND_ROI
    y1 = max(0, min(h - 1, int(ymin * h)))
    y2 = max(y1 + 10, min(h, int(ymax * h)))
    x1 = max(0, min(w - 1, int(xmin * w)))
    x2 = max(x1 + 10, min(w, int(xmax * w)))
    return img_rgb[y1:y2, x1:x2]


def detect_hand_presence(
    roi_rgb: np.ndarray,
    min_occupancy: float = 0.07,
    max_occupancy: float = 0.92,
    min_contour_area: float = 600.0
) -> tuple[bool, float, dict]:
    """
    Classical skin color segmentation presence check.
    Dual HSV & YCrCb skin segmentation with morphological filtering.
    """
    if roi_rgb is None or roi_rgb.size == 0 or roi_rgb.shape[0] < 10 or roi_rgb.shape[1] < 10:
        return False, 0.0, {"reason": "EMPTY_ROI", "occupancy": 0.0}

    total_pixels = roi_rgb.shape[0] * roi_rgb.shape[1]

    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2YCrCb)

    lower_hsv1 = np.array([0, 30, 50], dtype=np.uint8)
    upper_hsv1 = np.array([25, 255, 255], dtype=np.uint8)
    lower_hsv2 = np.array([165, 30, 50], dtype=np.uint8)
    upper_hsv2 = np.array([180, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.bitwise_or(
        cv2.inRange(hsv, lower_hsv1, upper_hsv1),
        cv2.inRange(hsv, lower_hsv2, upper_hsv2)
    )

    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    skin_pixels = int(np.count_nonzero(cleaned))
    occupancy_ratio = skin_pixels / total_pixels

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, occupancy_ratio, {
            "reason": "NO_SKIN_CONTOURS",
            "occupancy": round(occupancy_ratio, 4),
            "max_contour_area": 0.0
        }

    largest_contour = max(contours, key=cv2.contourArea)
    largest_area = float(cv2.contourArea(largest_contour))
    largest_ratio = largest_area / total_pixels

    has_sufficient_points = len(largest_contour) >= 5
    is_valid_occupancy = min_occupancy <= largest_ratio <= max_occupancy
    is_valid_area = largest_area >= min_contour_area

    is_hand_present = bool(has_sufficient_points and is_valid_occupancy and is_valid_area)

    reason = "HAND_CONFIRMED" if is_hand_present else (
        "BELOW_MIN_OCCUPANCY" if largest_ratio < min_occupancy else (
            "ABOVE_MAX_OCCUPANCY" if largest_ratio > max_occupancy else "INSUFFICIENT_AREA"
        )
    )

    metrics = {
        "reason": reason,
        "occupancy": round(occupancy_ratio, 4),
        "largest_contour_ratio": round(largest_ratio, 4),
        "largest_contour_area": round(largest_area, 1),
        "contour_points": len(largest_contour)
    }

    return is_hand_present, occupancy_ratio, metrics


def preprocess_image(
    image_input: bytes | np.ndarray | Path,
    target_size: tuple[int, int] = (224, 224),
    roi: tuple[float, float, float, float] | list[float] | None = None,
    crop_roi: bool = True
) -> np.ndarray:
    """
    Canonical hand-focused preprocessing pipeline for CustomSignCNN:
    1. Load image and convert to RGB
    2. Crop hand-focused region (removes face and background)
    3. Resize cropped region to target_size (default 224x224)
    4. Normalize pixel values to [0.0, 1.0]
    Returns float32 numpy array of shape (target_size[0], target_size[1], 3).
    """
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise ValueError(f"Unable to read image file at {image_input}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Unable to decode image bytes with OpenCV")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif len(image_input.shape) == 3 and image_input.shape[2] == 4:
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = image_input
    else:
        raise TypeError("Unsupported image input type")

    # Crop hand-focused region (Face excluded)
    if crop_roi:
        img_rgb = extract_hand_roi(img_rgb, roi=roi)

    # Resize with bilinear interpolation to 224x224
    resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0.0, 1.0]
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def get_model_input_preview_jpeg(
    image_input: bytes | np.ndarray | Path,
    roi: tuple[float, float, float, float] | list[float] | None = None
) -> bytes:
    """
    Returns JPEG bytes of the exact 224x224 hand-cropped image tensor for admin debugging preview.
    Visualizes precisely what visual features CustomSignCNN receives.
    """
    norm = preprocess_image(image_input, target_size=(224, 224), roi=roi, crop_roi=True)
    uint8_bgr = cv2.cvtColor((norm * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode(".jpg", uint8_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return encoded.tobytes()


def validate_and_inspect_video(file_path: Path | str) -> tuple[int, int, float, int]:
    """
    Validates a video file using OpenCV.
    Returns (width, height, duration_seconds, frame_count).
    """
    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        raise ValueError("Cannot open video file. Video stream may be corrupted or codec unsupported.")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count <= 0 or width <= 0 or height <= 0:
        count = 0
        w, h = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count == 0:
                h, w = frame.shape[:2]
            count += 1
            if count > 5000:
                break
        cap.release()
        
        if count == 0:
            raise ValueError("Video contains 0 readable frames.")
        width, height = w, h
        frame_count = count
        fps = 25.0
    else:
        cap.release()
        
    duration = round(frame_count / fps, 2)
    return width, height, duration, frame_count
