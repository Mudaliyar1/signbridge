import asyncio
import io
import os
import cv2
import numpy as np
import httpx
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def create_synthetic_image_bytes() -> bytes:
    """Generates a valid 224x224 RGB JPEG in memory."""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    # Draw simple hand-like contour
    cv2.circle(img, (112, 112), 40, (0, 255, 120), -1)
    cv2.putText(img, "SIGN", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    success, encoded = cv2.imencode(".jpg", img)
    if not success:
        raise RuntimeError("Failed to encode synthetic image")
    return encoded.tobytes()

def create_synthetic_video_bytes() -> bytes:
    """Generates a valid short MP4 video in memory."""
    temp_path = Path("temp_test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_path), fourcc, 10.0, (224, 224))
    
    for i in range(15):  # 1.5 seconds
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.circle(frame, (50 + i * 8, 112), 30, (255, 100, 50), -1)
        out.write(frame)
    out.release()
    
    video_bytes = temp_path.read_bytes()
    temp_path.unlink(missing_ok=True)
    return video_bytes

async def run_milestone1_tests():
    print("=" * 60)
    print("RUNNING SIGNBRIDGE MILESTONE 1 VERIFICATION SUITE")
    print("=" * 60)

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
        # 1. Health & Database connectivity check
        print("\n[Step 1] Checking API Health & Database status...")
        resp = await client.get("/api/health")
        print(f"Status: {resp.status_code}, Response: {resp.json()}")
        assert resp.status_code == 200
        health_data = resp.json()
        assert health_data["database"]["connected"] is True, f"MongoDB not connected: {health_data['database']['error']}"
        print("  -> MongoDB connection confirmed active & healthy!")

        # 2. Create Model
        print("\n[Step 2] Creating Model 'Sign Recognition V1'...")
        model_payload = {
            "name": "Sign Recognition V1",
            "version": "1.0.0",
            "description": "Primary custom CNN sign model",
            "architecture": "Custom CNN",
            "input_type": "RGB image",
            "input_resolution": "224x224"
        }
        resp = await client.post("/api/models", json=model_payload)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 400 and "already exists" in resp.text:
            # Fetch existing
            models_resp = await client.get("/api/models")
            model_id = [m["id"] for m in models_resp.json()["models"] if m["name"] == "Sign Recognition V1"][0]
            print(f"  -> Using existing Model ID: {model_id}")
        else:
            assert resp.status_code == 201
            model_id = resp.json()["id"]
            print(f"  -> Created Model ID: {model_id}")

        # 3. Create Sign Class: WATER
        print("\n[Step 3] Creating Sign Class 'WATER' (ISL, Static)...")
        class_payload = {
            "name": "WATER",
            "language": "ISL",
            "sign_type": "STATIC",
            "category": "Basics",
            "description": "Sign representing water"
        }
        resp = await client.post(f"/api/models/{model_id}/classes", json=class_payload)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 400 and "already exists" in resp.text:
            classes_resp = await client.get(f"/api/models/{model_id}/classes")
            class_id = [c["id"] for c in classes_resp.json()["classes"] if c["name"] == "WATER"][0]
            print(f"  -> Using existing Sign Class ID: {class_id}")
        else:
            assert resp.status_code == 201
            class_id = resp.json()["id"]
            print(f"  -> Created Sign Class ID: {class_id}")

        # 4. Register Participant: Participant 001
        print("\n[Step 4] Registering 'Participant 001' with explicit consent...")
        participant_payload = {
            "participant_id": "Participant 001",
            "display_name": "Signer Alpha",
            "consent_status": "GRANTED",
            "notes": "Verified signer for test collection"
        }
        resp = await client.post("/api/participants", json=participant_payload)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 400 and "already exists" in resp.text:
            print("  -> Participant 001 already registered")
        else:
            assert resp.status_code == 201
            print("  -> Registered Participant 001 successfully!")

        # 5. Test Camera Image Capture & Explicit Labeling
        print("\n[Step 5] Simulating Browser Camera Image Capture for 'WATER'...")
        img_bytes = create_synthetic_image_bytes()
        files = {"file": ("camera_test.jpg", img_bytes, "image/jpeg")}
        data = {
            "class_id": class_id,
            "participant_id": "Participant 001",
            "notes": "Synthetic camera capture test"
        }
        resp = await client.post(f"/api/models/{model_id}/samples/capture-image", files=files, data=data)
        print(f"Status: {resp.status_code}, Response: {resp.json()}")
        assert resp.status_code == 201
        img_sample = resp.json()
        assert img_sample["label"] == "WATER"
        assert img_sample["participant_id"] == "Participant 001"
        assert img_sample["width"] == 224
        assert img_sample["height"] == 224
        
        # Verify physical file exists on disk
        storage_path = Path("storage") / img_sample["storage_path"]
        assert storage_path.exists(), f"Physical media file does not exist at {storage_path}"
        print(f"  -> Physical file verified on disk ({storage_path.stat().st_size} bytes)")
        print("  -> MongoDB dataset record verified!")

        # 6. Test Video Capture & Explicit Labeling
        print("\n[Step 6] Simulating Browser Video Recording for 'WATER'...")
        vid_bytes = create_synthetic_video_bytes()
        files = {"file": ("recording_test.mp4", vid_bytes, "video/mp4")}
        data = {
            "class_id": class_id,
            "participant_id": "Participant 001",
            "duration": "1.5",
            "notes": "Synthetic camera video recording test"
        }
        resp = await client.post(f"/api/models/{model_id}/samples/capture-video", files=files, data=data)
        print(f"Status: {resp.status_code}, Response: {resp.json()}")
        assert resp.status_code == 201
        vid_sample = resp.json()
        assert vid_sample["label"] == "WATER"
        assert vid_sample["media_type"] == "VIDEO"
        assert vid_sample["duration"] > 0
        vid_storage_path = Path("storage") / vid_sample["storage_path"]
        assert vid_storage_path.exists(), f"Physical video file does not exist at {vid_storage_path}"
        print(f"  -> Physical video file verified on disk ({vid_storage_path.stat().st_size} bytes)")

        # 7. Test Dataset Filtering Query
        print("\n[Step 7] Querying Dataset samples via API...")
        resp = await client.get(f"/api/models/{model_id}/dataset?class_id={class_id}")
        assert resp.status_code == 200
        dataset = resp.json()
        print(f"  -> Total samples for WATER: {dataset['total']}")
        assert dataset["total"] >= 2

        # 8. Test Media Streaming Endpoint
        print("\n[Step 8] Verifying Media Serving endpoint...")
        media_url = f"/api/media/{img_sample['storage_path']}"
        resp = await client.get(media_url)
        assert resp.status_code == 200
        assert resp.headers["content-type"] in ["image/jpeg", "image/jpg"]
        print(f"  -> Media streaming verified ({len(resp.content)} bytes served)")

        # 9. Test Single Sample Deletion (Physical + MongoDB cleanup)
        print("\n[Step 9] Testing Sample Deletion & Safe Cleanup...")
        del_resp = await client.delete(f"/api/models/{model_id}/dataset/{img_sample['id']}")
        assert del_resp.status_code == 200
        assert not storage_path.exists(), "Physical file was not removed upon sample deletion!"
        print("  -> Physical file successfully deleted from storage")
        print("  -> MongoDB dataset record deleted successfully")

        print("\n" + "=" * 60)
        print("ALL MILESTONE 1 TESTS PASSED PERFECTLY!")
        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_milestone1_tests())
