import asyncio
import io
import cv2
import numpy as np
import httpx
from pathlib import Path
from app.config import settings

BASE_URL = "http://127.0.0.1:8000"

def create_synthetic_image_bytes() -> bytes:
    """Generates a valid 224x224 RGB JPEG in memory."""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.circle(img, (112, 112), 40, (0, 255, 120), -1)
    cv2.putText(img, "CLOUD", (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    success, encoded = cv2.imencode(".jpg", img)
    if not success:
        raise RuntimeError("Failed to encode synthetic image")
    return encoded.tobytes()

def create_synthetic_video_bytes() -> bytes:
    """Generates a valid short MP4 video in memory."""
    temp_path = Path("temp_cloud_test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_path), fourcc, 10.0, (224, 224))
    
    for i in range(15):  # 1.5 seconds
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.circle(frame, (50 + i * 8, 112), 30, (255, 100, 50), -1)
        cv2.putText(frame, "SIGN", (60, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    
    video_bytes = temp_path.read_bytes()
    temp_path.unlink(missing_ok=True)
    return video_bytes

async def run_cloud_migration_tests():
    print("=" * 65)
    print("RUNNING SIGNBRIDGE CLOUD STORAGE VERIFICATION SUITE")
    print("MongoDB Atlas + Cloudinary Live End-to-End Pipeline")
    print("=" * 65)

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        # 1. Health check & Atlas verification
        print("\n[Step 1] Verifying MongoDB Atlas connection...")
        resp = await client.get("/api/health")
        print(f"Status: {resp.status_code}, Response: {resp.json()}")
        assert resp.status_code == 200
        health_data = resp.json()
        assert health_data["database"]["connected"] is True, f"Atlas not connected: {health_data['database']['error']}"
        assert "mongodb.net" in health_data["database"]["url"], "Not connected to MongoDB Atlas cluster!"
        print("  -> Live MongoDB Atlas connection verified!")

        # 2. Cloudinary Admin & Storage API check
        print("\n[Step 2] Verifying Cloudinary Storage integration...")
        resp = await client.get("/api/storage/overview")
        assert resp.status_code == 200
        storage_overview = resp.json()
        print(f"Provider: {storage_overview['provider']}")
        print(f"Cloud Name: {storage_overview['cloud_name']}")
        print(f"Cloudinary Account Plan: {storage_overview['cloudinary_account_usage'].get('plan')}")
        assert storage_overview["cloud_name"] == settings.CLOUDINARY_CLOUD_NAME
        assert storage_overview["cloudinary_account_usage"]["success"] is True
        print("  -> Live Cloudinary API connection verified!")

        # 3. Create Model on Atlas
        print("\n[Step 3] Creating Model on MongoDB Atlas...")
        model_payload = {
            "name": "Sign Recognition Cloud V1",
            "version": "1.0.0",
            "description": "Production Cloud Model with Atlas + Cloudinary",
            "architecture": "Custom CNN",
            "input_type": "RGB image",
            "input_resolution": "224x224"
        }
        resp = await client.post("/api/models", json=model_payload)
        if resp.status_code == 400 and "already exists" in resp.text:
            models_resp = await client.get("/api/models")
            model_id = [m["id"] for m in models_resp.json()["models"] if m["name"] == "Sign Recognition Cloud V1"][0]
            print(f"  -> Using existing Model ID: {model_id}")
        else:
            assert resp.status_code == 201
            model_id = resp.json()["id"]
            print(f"  -> Created Model ID in Atlas: {model_id}")

        # 4. Create Sign Class: WATER
        print("\n[Step 4] Creating Sign Class 'WATER' in Atlas...")
        class_payload = {
            "name": "WATER",
            "language": "ISL",
            "sign_type": "STATIC",
            "category": "Basics",
            "description": "Sign representing water"
        }
        resp = await client.post(f"/api/models/{model_id}/classes", json=class_payload)
        if resp.status_code == 400 and "already exists" in resp.text:
            classes_resp = await client.get(f"/api/models/{model_id}/classes")
            class_id = [c["id"] for c in classes_resp.json()["classes"] if c["name"] == "WATER"][0]
            print(f"  -> Using existing Sign Class ID: {class_id}")
        else:
            assert resp.status_code == 201
            class_id = resp.json()["id"]
            print(f"  -> Created Sign Class ID in Atlas: {class_id}")

        # 5. Register Participant: Participant 001
        print("\n[Step 5] Registering 'Participant 001' with explicit consent in Atlas...")
        participant_payload = {
            "participant_id": "Participant 001",
            "display_name": "Signer Alpha (Atlas)",
            "consent_status": "GRANTED",
            "notes": "Verified signer for cloud dataset"
        }
        resp = await client.post("/api/participants", json=participant_payload)
        if resp.status_code == 400 and "already exists" in resp.text:
            print("  -> Participant 001 already registered in Atlas")
        else:
            assert resp.status_code == 201
            print("  -> Registered Participant 001 in Atlas!")

        # 6. Real Camera Image Capture -> Cloudinary Upload -> Atlas metadata
        print("\n[Step 6] Testing Image Upload to Cloudinary & Atlas...")
        img_bytes = create_synthetic_image_bytes()
        files = {"file": ("camera_capture_cloud.jpg", img_bytes, "image/jpeg")}
        data = {
            "class_id": class_id,
            "participant_id": "Participant 001",
            "notes": "Cloudinary capture test"
        }
        resp = await client.post(f"/api/models/{model_id}/samples/capture-image", files=files, data=data)
        print(f"Status: {resp.status_code}")
        assert resp.status_code == 201
        img_sample = resp.json()
        
        print(f"  -> Cloudinary public_id: {img_sample['storage_path']}")
        print(f"  -> Cloudinary secure_url: {img_sample['media_url']}")
        assert img_sample["storage_provider"] == "cloudinary"
        assert "cloudinary.com" in img_sample["media_url"]
        assert img_sample["label"] == "WATER"
        assert img_sample["participant_id"] == "Participant 001"
        assert img_sample["width"] == 224
        assert img_sample["height"] == 224

        # 7. Real Video Recording -> Cloudinary Video Upload -> Atlas metadata
        print("\n[Step 7] Testing Video Upload to Cloudinary (resource_type='video')...")
        vid_bytes = create_synthetic_video_bytes()
        files = {"file": ("recording_cloud.mp4", vid_bytes, "video/mp4")}
        data = {
            "class_id": class_id,
            "participant_id": "Participant 001",
            "duration": "1.5",
            "notes": "Cloudinary video recording test"
        }
        resp = await client.post(f"/api/models/{model_id}/samples/capture-video", files=files, data=data)
        print(f"Status: {resp.status_code}")
        assert resp.status_code == 201
        vid_sample = resp.json()

        print(f"  -> Cloudinary video public_id: {vid_sample['storage_path']}")
        print(f"  -> Cloudinary video secure_url: {vid_sample['media_url']}")
        assert vid_sample["storage_provider"] == "cloudinary"
        assert "cloudinary.com" in vid_sample["media_url"]
        assert vid_sample["media_type"] == "VIDEO"
        assert vid_sample["duration"] > 0

        # 8. Test Direct Cloudinary CDN Media Delivery
        print("\n[Step 8] Verifying direct delivery from Cloudinary CDN...")
        async with httpx.AsyncClient(timeout=15.0) as cloud_client:
            img_cdn_resp = await cloud_client.get(img_sample["media_url"])
            assert img_cdn_resp.status_code == 200
            assert "image" in img_cdn_resp.headers.get("content-type", "")
            print(f"  -> Image fetched directly from Cloudinary CDN ({len(img_cdn_resp.content)} bytes)")

            vid_cdn_resp = await cloud_client.get(vid_sample["media_url"])
            assert vid_cdn_resp.status_code == 200
            print(f"  -> Video fetched directly from Cloudinary CDN ({len(vid_cdn_resp.content)} bytes)")

        # 9. Test Dataset Browser API with Cloudinary URLs
        print("\n[Step 9] Querying Dataset via API...")
        ds_resp = await client.get(f"/api/models/{model_id}/dataset?class_id={class_id}")
        assert ds_resp.status_code == 200
        samples = ds_resp.json()["samples"]
        for s in samples:
            assert "media_url" in s and "cloudinary.com" in s["media_url"]
        print(f"  -> {len(samples)} samples retrieved with verified Cloudinary CDN URLs!")

        # 10. Test Cloud Deletion: Removes from Cloudinary & Atlas
        print("\n[Step 10] Testing Cloud Media Deletion...")
        del_resp = await client.delete(f"/api/models/{model_id}/dataset/{img_sample['id']}")
        assert del_resp.status_code == 200
        print(f"  -> Deleted image '{img_sample['id']}' from Cloudinary and Atlas")

        # 11. Test Admin Storage Page Render
        print("\n[Step 11] Testing /admin/storage HTML page...")
        storage_page_resp = await client.get("/admin/storage")
        assert storage_page_resp.status_code == 200
        assert "Cloud Media Storage" in storage_page_resp.text
        assert settings.CLOUDINARY_CLOUD_NAME in storage_page_resp.text
        print("  -> /admin/storage page rendered with status 200 OK!")

        print("\n" + "=" * 65)
        print("ALL CLOUD STORAGE MIGRATION TESTS PASSED PERFECTLY!")
        print("MongoDB Atlas & Cloudinary Live Pipeline 100% Operational")
        print("=" * 65)

if __name__ == "__main__":
    asyncio.run(run_cloud_migration_tests())
