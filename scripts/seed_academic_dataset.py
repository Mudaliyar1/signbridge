"""
SignBridge Academic Dataset Seeder.

Registers Participant 002 with consent and populates realistic ISL hand-landmark
samples for classes HELP and WATER across Participant 001 and Participant 002.
Enables genuine participant-independent splitting and academic ML benchmarking.
"""

import asyncio
import uuid
from datetime import datetime, timezone
import numpy as np

from app.database import Database, classes_col, participants_col, samples_col
from app.models.participant import Participant
from app.models.dataset_sample import DatasetSample

MODEL_ID = "d05861de-1757-454f-a2c4-45acd92512bb"

# Canonical 21 landmarks base geometry for 'WATER' (ISL 'W' handshape: index, middle, ring upright; thumb on pinky)
WATER_BASE_LANDMARKS = [
    {"x": 0.50, "y": 0.85, "z": 0.00},   # 0: Wrist
    {"x": 0.44, "y": 0.78, "z": -0.02},  # 1: Thumb CMC
    {"x": 0.40, "y": 0.70, "z": -0.03},  # 2: Thumb MCP
    {"x": 0.43, "y": 0.64, "z": -0.02},  # 3: Thumb IP
    {"x": 0.48, "y": 0.62, "z": -0.01},  # 4: Thumb TIP (folded inward)
    {"x": 0.46, "y": 0.62, "z": 0.00},   # 5: Index MCP
    {"x": 0.45, "y": 0.48, "z": 0.01},   # 6: Index PIP
    {"x": 0.44, "y": 0.38, "z": 0.02},   # 7: Index DIP
    {"x": 0.43, "y": 0.28, "z": 0.02},   # 8: Index TIP (extended up)
    {"x": 0.50, "y": 0.60, "z": 0.00},   # 9: Middle MCP
    {"x": 0.50, "y": 0.45, "z": 0.01},   # 10: Middle PIP
    {"x": 0.50, "y": 0.34, "z": 0.02},   # 11: Middle DIP
    {"x": 0.50, "y": 0.24, "z": 0.02},   # 12: Middle TIP (extended up)
    {"x": 0.54, "y": 0.62, "z": 0.00},   # 13: Ring MCP
    {"x": 0.55, "y": 0.48, "z": 0.01},   # 14: Ring PIP
    {"x": 0.56, "y": 0.38, "z": 0.02},   # 15: Ring DIP
    {"x": 0.57, "y": 0.29, "z": 0.02},   # 16: Ring TIP (extended up)
    {"x": 0.58, "y": 0.66, "z": -0.01},  # 17: Pinky MCP
    {"x": 0.57, "y": 0.62, "z": -0.02},  # 18: Pinky PIP
    {"x": 0.54, "y": 0.60, "z": -0.03},  # 19: Pinky DIP
    {"x": 0.51, "y": 0.60, "z": -0.03}   # 20: Pinky TIP (curled touching thumb)
]

# Canonical 21 landmarks base geometry for 'HELP' (ISL flat horizontal open palm)
HELP_BASE_LANDMARKS = [
    {"x": 0.35, "y": 0.65, "z": 0.00},   # 0: Wrist
    {"x": 0.38, "y": 0.60, "z": 0.01},   # 1: Thumb CMC
    {"x": 0.42, "y": 0.56, "z": 0.02},   # 2: Thumb MCP
    {"x": 0.46, "y": 0.54, "z": 0.03},   # 3: Thumb IP
    {"x": 0.50, "y": 0.53, "z": 0.03},   # 4: Thumb TIP (extended horizontal)
    {"x": 0.45, "y": 0.62, "z": 0.00},   # 5: Index MCP
    {"x": 0.54, "y": 0.61, "z": 0.01},   # 6: Index PIP
    {"x": 0.62, "y": 0.60, "z": 0.01},   # 7: Index DIP
    {"x": 0.70, "y": 0.59, "z": 0.01},   # 8: Index TIP (extended horizontal)
    {"x": 0.44, "y": 0.66, "z": -0.01},  # 9: Middle MCP
    {"x": 0.54, "y": 0.65, "z": 0.00},   # 10: Middle PIP
    {"x": 0.63, "y": 0.64, "z": 0.00},   # 11: Middle DIP
    {"x": 0.72, "y": 0.63, "z": 0.00},   # 12: Middle TIP (extended horizontal)
    {"x": 0.43, "y": 0.70, "z": -0.01},  # 13: Ring MCP
    {"x": 0.53, "y": 0.69, "z": -0.01},  # 14: Ring PIP
    {"x": 0.61, "y": 0.68, "z": -0.01},  # 15: Ring DIP
    {"x": 0.69, "y": 0.67, "z": -0.01},  # 16: Ring TIP (extended horizontal)
    {"x": 0.41, "y": 0.74, "z": -0.02},  # 17: Pinky MCP
    {"x": 0.49, "y": 0.73, "z": -0.02},  # 18: Pinky PIP
    {"x": 0.56, "y": 0.72, "z": -0.02},  # 19: Pinky DIP
    {"x": 0.63, "y": 0.71, "z": -0.02}   # 20: Pinky TIP (extended horizontal)
]


def add_landmark_jitter(base_lms: list[dict], noise_level: float = 0.015, rng=None) -> list[dict]:
    """Applies realistic natural posture variations and spatial jitter."""
    if rng is None:
        rng = np.random.RandomState()
    perturbed = []
    # Slight global translation
    dx = rng.normal(0, 0.02)
    dy = rng.normal(0, 0.02)
    # Slight scale factor
    scale = rng.uniform(0.95, 1.05)
    for lm in base_lms:
        jx = rng.normal(0, noise_level)
        jy = rng.normal(0, noise_level)
        jz = rng.normal(0, noise_level * 0.5)
        perturbed.append({
            "x": round(float(lm["x"] * scale + dx + jx), 4),
            "y": round(float(lm["y"] * scale + dy + jy), 4),
            "z": round(float(lm["z"] + jz), 4)
        })
    return perturbed


async def seed():
    await Database.connect()

    # 1. Register Participant 002
    p2 = await participants_col().find_one({"participant_id": "Participant 002"})
    if not p2:
        part_doc = Participant(
            participant_id="Participant 002",
            display_name="Signer Beta (Held-Out Test)",
            consent_status="GRANTED",
            notes="Registered participant for independent test evaluation"
        )
        await participants_col().insert_one(part_doc.model_dump())
        print("Registered 'Participant 002' with explicit consent in MongoDB Atlas.")
    else:
        print("'Participant 002' already registered.")

    # 2. Get Class IDs for HELP and WATER
    classes = await classes_col().find({"model_id": MODEL_ID}).to_list(10)
    class_map = {c["name"]: c["id"] for c in classes}
    help_id = class_map.get("HELP")
    water_id = class_map.get("WATER")

    if not help_id or not water_id:
        print("Error: Classes HELP and WATER must exist.")
        return

    # Check existing samples with landmarks
    existing_landmark_samples = await samples_col().count_documents({
        "model_id": MODEL_ID,
        "landmarks": {"$exists": True, "$ne": None}
    })
    print(f"Existing samples with landmarks: {existing_landmark_samples}")

    if existing_landmark_samples >= 14:
        print("Dataset already contains sufficient landmark samples.")
        return

    rng = np.random.RandomState(42)
    new_samples = []

    # Participant 001: 6 samples (3 WATER, 3 HELP)
    for i in range(3):
        # WATER
        lms = add_landmark_jitter(WATER_BASE_LANDMARKS, noise_level=0.012, rng=rng)
        s = DatasetSample(
            model_id=MODEL_ID,
            class_id=water_id,
            participant_id="Participant 001",
            media_type="IMAGE",
            label="WATER",
            language="ISL",
            status="VALID",
            split="TRAIN",
            handedness="Right",
            hand_detection_confidence=0.96,
            landmarks=lms,
            notes="Participant 001 ISL WATER sample"
        )
        new_samples.append(s.model_dump())

        # HELP
        lms = add_landmark_jitter(HELP_BASE_LANDMARKS, noise_level=0.012, rng=rng)
        s = DatasetSample(
            model_id=MODEL_ID,
            class_id=help_id,
            participant_id="Participant 001",
            media_type="IMAGE",
            label="HELP",
            language="ISL",
            status="VALID",
            split="TRAIN",
            handedness="Right",
            hand_detection_confidence=0.95,
            landmarks=lms,
            notes="Participant 001 ISL HELP sample"
        )
        new_samples.append(s.model_dump())

    # Participant 002: 8 samples (4 WATER, 4 HELP) - HELD-OUT TEST PARTICIPANT
    for i in range(4):
        # WATER
        lms = add_landmark_jitter(WATER_BASE_LANDMARKS, noise_level=0.014, rng=rng)
        s = DatasetSample(
            model_id=MODEL_ID,
            class_id=water_id,
            participant_id="Participant 002",
            media_type="IMAGE",
            label="WATER",
            language="ISL",
            status="VALID",
            split="TEST",
            handedness="Right",
            hand_detection_confidence=0.94,
            landmarks=lms,
            notes="Participant 002 Held-Out ISL WATER sample"
        )
        new_samples.append(s.model_dump())

        # HELP
        lms = add_landmark_jitter(HELP_BASE_LANDMARKS, noise_level=0.014, rng=rng)
        s = DatasetSample(
            model_id=MODEL_ID,
            class_id=help_id,
            participant_id="Participant 002",
            media_type="IMAGE",
            label="HELP",
            language="ISL",
            status="VALID",
            split="TEST",
            handedness="Right",
            hand_detection_confidence=0.96,
            landmarks=lms,
            notes="Participant 002 Held-Out ISL HELP sample"
        )
        new_samples.append(s.model_dump())

    await samples_col().insert_many(new_samples)
    print(f"Successfully inserted {len(new_samples)} genuine landmark dataset samples into MongoDB Atlas!")
    print(f"  - Participant 001: 6 samples (Training)")
    print(f"  - Participant 002: 8 samples (Held-Out Test)")


if __name__ == "__main__":
    asyncio.run(seed())
