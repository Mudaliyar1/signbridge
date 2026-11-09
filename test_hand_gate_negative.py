import io
import cv2
import numpy as np
from PIL import Image
from app.ml.preprocessing import (
    DEFAULT_HAND_ROI,
    PREPROCESSING_VERSION,
    detect_hand_presence,
    extract_hand_roi,
    preprocess_image
)
from app.ml.inference import CustomSignInferenceService

def create_face_only_frame():
    """
    Creates a synthetic 640x480 webcam frame where:
    - User head/face is located in the upper region: y in [40, 180], x in [240, 400]
    - The lower chest / signing region (DEFAULT_HAND_ROI: y in [216, 456], x in [204, 435]) has NO hand.
    """
    # Neutral gray background in BGR
    img = np.full((480, 640, 3), (180, 180, 180), dtype=np.uint8)

    # Torso/Shirt in dark blue (BGR: 70, 30, 30) spanning chest region
    cv2.rectangle(img, (160, 210), (480, 480), (70, 30, 30), -1)

    # Face/head in upper region (y: 60..180) with skin tone (BGR: 120, 150, 210 -> RGB: 210, 150, 120)
    cv2.ellipse(img, (320, 120), (65, 80), 0, 0, 360, (120, 150, 210), -1)
    # Eyes
    cv2.circle(img, (295, 105), 6, (30, 30, 30), -1)
    cv2.circle(img, (345, 105), 6, (30, 30, 30), -1)
    # Mouth
    cv2.ellipse(img, (320, 150), (18, 6), 0, 0, 180, (60, 60, 120), -1)
    # Hair
    cv2.ellipse(img, (320, 65), (70, 40), 0, 180, 360, (20, 20, 20), -1)

    success, encoded = cv2.imencode('.jpg', img)
    return encoded.tobytes()

def create_empty_background_frame():
    """Creates a synthetic empty room background."""
    img = np.full((480, 640, 3), (180, 180, 180), dtype=np.uint8)
    cv2.line(img, (0, 240), (640, 240), (140, 140, 140), 2)
    success, encoded = cv2.imencode('.jpg', img)
    return encoded.tobytes()

def create_hand_present_frame():
    """Creates a frame with a hand with fingers inside the designated hand ROI."""
    # Neutral gray background in BGR
    img = np.full((480, 640, 3), (180, 180, 180), dtype=np.uint8)
    # Torso/Shirt
    cv2.rectangle(img, (160, 210), (480, 480), (70, 30, 30), -1)

    # Hand palm in the ROI box (y: 280..420, x: 270..370) with skin tone (BGR: 120, 150, 210)
    cv2.ellipse(img, (320, 350), (40, 50), 0, 0, 360, (120, 150, 210), -1)
    # Fingers
    for fx in [295, 312, 328, 345]:
        cv2.ellipse(img, (fx, 285), (8, 24), 0, 0, 360, (120, 150, 210), -1)
    # Thumb
    cv2.ellipse(img, (270, 340), (12, 16), -30, 0, 360, (120, 150, 210), -1)

    success, encoded = cv2.imencode('.jpg', img)
    return encoded.tobytes()

def run_tests():
    model_id = "d05861de-1757-454f-a2c4-45acd92512bb"
    print("==================================================================")
    print(f"RUNNING HAND PRESENCE GATE & NO-FACE-FALSE-POSITIVE TESTS")
    print(f"Model ID: {model_id}")
    print(f"Preprocessing Version: {PREPROCESSING_VERSION}")
    print(f"Hand ROI Box: {DEFAULT_HAND_ROI}")
    print("==================================================================\n")

    # TEST 1: FACE ONLY
    print("[TEST 1] Testing Frame: FACE ONLY (User sitting looking at camera, NO HANDS)")
    face_bytes = create_face_only_frame()
    res1 = CustomSignInferenceService.predict_image(model_id, face_bytes)
    print(f"  - Status: {res1.get('status')}")
    print(f"  - Hand Present: {res1.get('hand_present')}")
    print(f"  - Prediction: {res1.get('prediction')}")
    print(f"  - Confidence: {res1.get('confidence')}")
    print(f"  - Speak (TTS): {res1.get('speak')}")
    print(f"  - Message: {res1.get('message')}")
    
    assert res1.get('status') == 'NO_HAND', f"FAIL: Expected status NO_HAND, got {res1.get('status')}"
    assert res1.get('hand_present') is False, "FAIL: Expected hand_present False"
    assert res1.get('prediction') is None, "FAIL: Expected prediction None"
    assert res1.get('confidence') == 0.0, "FAIL: Expected confidence 0.0"
    assert res1.get('speak') is False, "FAIL: Expected speak False"
    print("  >>> PASS: Face-only frame was correctly intercepted by Hand Gate! No false prediction, no TTS.\n")

    # TEST 2: EMPTY BACKGROUND
    print("[TEST 2] Testing Frame: EMPTY BACKGROUND (No person/hand in camera)")
    bg_bytes = create_empty_background_frame()
    res2 = CustomSignInferenceService.predict_image(model_id, bg_bytes)
    print(f"  - Status: {res2.get('status')}")
    print(f"  - Hand Present: {res2.get('hand_present')}")
    print(f"  - Prediction: {res2.get('prediction')}")
    print(f"  - Confidence: {res2.get('confidence')}")
    assert res2.get('status') == 'NO_HAND', f"FAIL: Expected status NO_HAND, got {res2.get('status')}"
    assert res2.get('hand_present') is False, "FAIL: Expected hand_present False"
    print("  >>> PASS: Empty background correctly returned NO_HAND.\n")

    # TEST 3: HAND PRESENT
    print("[TEST 3] Testing Frame: HAND PRESENT IN ROI")
    hand_bytes = create_hand_present_frame()
    res3 = CustomSignInferenceService.predict_image(model_id, hand_bytes)
    print(f"  - Status: {res3.get('status')}")
    print(f"  - Hand Present: {res3.get('hand_present')}")
    print(f"  - Prediction: {res3.get('prediction')}")
    print(f"  - Top Predictions: {res3.get('top_predictions')}")
    assert res3.get('status') in ['HAND_DETECTED', 'HAND_CONFIRMED'], f"FAIL: Expected HAND_DETECTED, got {res3.get('status')}"
    assert res3.get('hand_present') is True, "FAIL: Expected hand_present True"
    assert res3.get('prediction') is not None, "FAIL: Expected valid prediction"
    print(f"  >>> PASS: Hand presence confirmed, CNN evaluated: {res3.get('prediction').get('class_name')} ({res3.get('prediction').get('confidence') * 100:.1f}%)\n")

    print("==================================================================")
    print("ALL TESTS PASSED! Hand Gate successfully isolates hands from faces.")
    print("==================================================================")

if __name__ == '__main__':
    run_tests()
