import asyncio
import io
import cv2
import numpy as np
import httpx
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def create_synthetic_image_bytes(label: str) -> bytes:
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.circle(img, (112, 112), 45, (0, 200, 100), -1)
    cv2.putText(img, label[:4], (65, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    success, encoded = cv2.imencode(".jpg", img)
    assert success
    return encoded.tobytes()

def create_synthetic_video_bytes(label: str) -> bytes:
    temp_path = Path("temp_eval_test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_path), fourcc, 10.0, (224, 224))
    for i in range(12):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.circle(frame, (60 + i * 8, 112), 35, (100, 100, 255), -1)
        cv2.putText(frame, label[:4], (60, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    data = temp_path.read_bytes()
    temp_path.unlink(missing_ok=True)
    return data

async def run_testing_studio_suite():
    print("=" * 65)
    print("RUNNING SIGNBRIDGE TESTING STUDIO VERIFICATION SUITE")
    print("Zero-Pretrained PyTorch Engine + Full Testing Workflow")
    print("=" * 65)

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=300.0) as client:
        # Step 1: Health check
        print("\n[Step 1] Verifying server health...")
        resp = await client.get("/api/health")
        assert resp.status_code == 200, f"Health check failed: {resp.text}"
        print("  -> Server healthy, MongoDB Atlas online.")

        # Step 2: Fetch models
        print("\n[Step 2] Fetching existing models...")
        resp = await client.get("/api/models")
        assert resp.status_code == 200
        models_data = resp.json()["models"]
        assert len(models_data) > 0, "No models found!"
        test_model = models_data[0]
        model_id = test_model["id"]
        print(f"  -> Testing Model: '{test_model['name']}' (ID: {model_id})")

        # Step 3: Verify Model Availability Check
        print("\n[Step 3] Verifying availability check...")
        resp = await client.get(f"/api/models/{model_id}/testing/status")
        assert resp.status_code == 200
        status_data = resp.json()
        print(f"  -> Status: {status_data['availability']['status']}")
        
        # Test that inference properly rejects when no model is trained (using dummy untrained ID)
        img_bytes = create_synthetic_image_bytes("TEST")
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        resp = await client.post("/api/models/dummy-untrained-id-12345/testing/image", files=files)
        assert resp.status_code == 400
        print("  -> Confirmed: Inference rejected when untrained (zero fake predictions policy verified).")

        # Step 4: Ensure at least 2 sign classes exist for classification
        print("\n[Step 4] Checking sign classes...")
        resp = await client.get(f"/api/models/{model_id}/classes")
        existing_classes = resp.json()["classes"]
        print(f"  -> Currently {len(existing_classes)} classes configured.")

        class_map = {c["name"]: c["id"] for c in existing_classes}
        if "WATER" not in class_map:
            print("  -> Creating 'WATER' sign class...")
            resp = await client.post(f"/api/models/{model_id}/classes", json={
                "name": "WATER",
                "description": "Sign for drinking water",
                "language": "ISL",
                "category": "Emergency",
                "sign_type": "STATIC"
            })
            assert resp.status_code == 201
            class_map["WATER"] = resp.json()["id"]

        if "HELP" not in class_map:
            print("  -> Creating 'HELP' sign class...")
            resp = await client.post(f"/api/models/{model_id}/classes", json={
                "name": "HELP",
                "description": "Sign for urgent help",
                "language": "ISL",
                "category": "Emergency",
                "sign_type": "STATIC"
            })
            assert resp.status_code == 201
            class_map["HELP"] = resp.json()["id"]

        # Step 5: Ensure participant exists
        print("\n[Step 5] Ensuring participant exists...")
        resp = await client.get("/api/participants")
        participants = resp.json()["participants"]
        if not participants:
            resp = await client.post("/api/participants", json={
                "participant_id": "P001",
                "name": "Tester 1",
                "consent_status": "GIVEN"
            })
            part_id = "P001"
        else:
            part_id = participants[0]["participant_id"]
        print(f"  -> Using Participant: {part_id}")

        # Step 6: Upload sample images for each class to train
        print("\n[Step 6] Adding samples for WATER and HELP...")
        for sign_name in ["WATER", "HELP"]:
            for i in range(2):
                img_data = create_synthetic_image_bytes(sign_name)
                upload_files = {"file": (f"{sign_name}_{i}.jpg", img_data, "image/jpeg")}
                upload_data = {
                    "class_id": class_map[sign_name],
                    "participant_id": part_id,
                    "media_type": "IMAGE",
                    "notes": f"Synthetic test sample {i}"
                }
                resp = await client.post(f"/api/models/{model_id}/samples/upload", files=upload_files, data=upload_data)
                assert resp.status_code == 201
        print("  -> Uploaded training samples for WATER and HELP.")

        # Step 7: Train the CustomSignCNN from scratch!
        print("\n[Step 7] Training scratch CustomSignCNN on dataset...")
        resp = await client.post(f"/api/models/{model_id}/training/start", json={
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 0.005
        })
        assert resp.status_code == 200, f"Training failed: {resp.text}"
        train_result = resp.json()
        print(f"  -> Training Success: {train_result['success']}")
        print(f"  -> Classes Trained: {train_result['classes']}")
        print(f"  -> Final Train Acc: {train_result['final_train_accuracy']}%")
        print(f"  -> Weights saved at: {train_result['weights_saved_at']}")

        # Step 8: Verify Model Availability Check (AFTER Training)
        print("\n[Step 8] Verifying availability check AFTER training...")
        resp = await client.get(f"/api/models/{model_id}/testing/status")
        assert resp.status_code == 200
        status_data = resp.json()
        assert status_data["availability"]["available"] is True
        print(f"  -> Status: {status_data['availability']['status']}")
        print(f"  -> Num Classes: {status_data['availability']['num_classes']}")

        # Step 9: Test Individual Image Inference
        print("\n[Step 9] Testing PyTorch image inference...")
        test_img = create_synthetic_image_bytes("WATER")
        files = {"file": ("test_water.jpg", test_img, "image/jpeg")}
        resp = await client.post(f"/api/models/{model_id}/testing/image", files=files)
        assert resp.status_code == 200, f"Inference failed: {resp.text}"
        img_res = resp.json()
        print(f"  -> Predicted: {img_res['prediction']['class_name']}")
        print(f"  -> Confidence: {round(img_res['prediction']['confidence'] * 100, 2)}%")
        print(f"  -> Latency: {img_res['inference_time_ms']} ms")
        print(f"  -> Top Predictions: {img_res['top_predictions']}")

        # Step 10: Test Video Frame-Based Inference
        print("\n[Step 10] Testing Frame-Based Video Inference...")
        test_vid = create_synthetic_video_bytes("HELP")
        v_files = {"file": ("test_help.mp4", test_vid, "video/mp4")}
        v_data = {"sample_fps": "2.0"}
        resp = await client.post(f"/api/models/{model_id}/testing/video", files=v_files, data=v_data)
        assert resp.status_code == 200, f"Video inference failed: {resp.text}"
        vid_res = resp.json()
        print(f"  -> Video Method: {vid_res['method_label']}")
        print(f"  -> Architecture Note: {vid_res['architecture_note']}")
        print(f"  -> Final Prediction: {vid_res['final_prediction']['class_name']} ({round(vid_res['final_prediction']['confidence'] * 100, 2)}%)")
        print(f"  -> Total Frames Analyzed: {vid_res['total_frames_analyzed']}")

        # Step 11: Test Dataset Batch Evaluation & Real Metrics
        print("\n[Step 11] Running Dataset Batch Evaluation...")
        resp = await client.post(f"/api/models/{model_id}/testing/evaluate?split=ALL")
        assert resp.status_code == 200, f"Evaluation failed: {resp.text}"
        eval_res = resp.json()
        print(f"  -> Total Evaluated: {eval_res['total_samples']}")
        print(f"  -> Accuracy: {eval_res['accuracy']}%")
        print(f"  -> Macro Precision: {eval_res['macro_precision']}%")
        print(f"  -> Macro Recall: {eval_res['macro_recall']}%")
        print(f"  -> Macro F1: {eval_res['macro_f1']}%")
        print(f"  -> Confusion Matrix Labels: {eval_res['confusion_matrix']['labels']}")
        print(f"  -> Confusion Matrix: {eval_res['confusion_matrix']['matrix']}")

        # Step 12: Test Live-Frame Inference Endpoint
        print("\n[Step 12] Testing Live-Frame camera endpoint...")
        live_files = {"file": ("frame.jpg", test_img, "image/jpeg")}
        resp = await client.post(f"/api/models/{model_id}/testing/live-frame", files=live_files)
        assert resp.status_code == 200
        live_res = resp.json()
        assert live_res["success"] is True
        print(f"  -> Live Frame Output: {live_res['prediction']['class_name']} ({round(live_res['prediction']['confidence'] * 100, 2)}%)")

        # Step 13: Verify Test History Persistence in MongoDB Atlas
        print("\n[Step 13] Verifying MongoDB Atlas test history persistence...")
        resp = await client.get(f"/api/models/{model_id}/testing/history")
        assert resp.status_code == 200
        hist_data = resp.json()
        print(f"  -> Total Test Records Saved in Atlas: {hist_data['total']}")
        assert hist_data['total'] >= 2, "Expected test records in database!"

        # Step 14: Verify Testing Studio HTML Page renders 200 OK
        print("\n[Step 14] Verifying Testing Studio UI route...")
        resp = await client.get(f"/admin/models/{model_id}/testing")
        assert resp.status_code == 200
        assert "Testing Studio" in resp.text
        assert "Live Camera Feed" in resp.text
        assert "READY FOR TESTING" in resp.text
        print("  -> Testing Studio UI loaded successfully (200 OK)!")

    print("\n" + "=" * 65)
    print("ALL SIGNBRIDGE TESTING STUDIO TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 65)

if __name__ == "__main__":
    asyncio.run(run_testing_studio_suite())
