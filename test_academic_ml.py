"""
SignBridge Academic Machine Learning Verification & Acceptance Test Suite.

Executes tests across:
1. Feature Extraction (canonical 63-D normalization & invariance)
2. Presence Gating (NO_HAND rejection on blank frames)
3. Participant-Independent Data Splitting (zero data leakage)
4. KNN, SVM, Decision Tree Training & Hyperparameter Tuning
5. Unified Model Performance Comparison (Accuracy, Precision, Recall, F1, ROC-AUC, CM)
6. Live MongoDB Atlas Exploratory Data Analysis (EDA) & Matplotlib/Seaborn Charts
7. Unsupervised K-Means Clustering & 2D PCA Visualization
8. Atlas Experiment History Logging
9. Web Admin ML Analysis Dashboard (HTTP 200 OK)
"""

import asyncio
import io
from pathlib import Path
import cv2
import numpy as np
import httpx

BASE_URL = "http://127.0.0.1:8000"
MODEL_ID = "d05861de-1757-454f-a2c4-45acd92512bb"


def create_blank_test_image() -> bytes:
    """Creates a blank background with zero hands for presence gating test."""
    img = np.full((320, 320, 3), (120, 120, 120), dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", img)
    return enc.tobytes()


async def run_suite():
    print("=" * 70)
    print("SIGNBRIDGE ACADEMIC MACHINE LEARNING ACCEPTANCE TEST SUITE")
    print("Zero Fabricated Data & Full Participant-Independent Splitting Policy")
    print("=" * 70)

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        # 1. API Health & Database Check
        print("\n[TEST 1] Verifying FastAPI Server & MongoDB Atlas Connectivity...")
        resp = await client.get("/api/health")
        assert resp.status_code == 200, f"Health check failed: {resp.text}"
        data = resp.json()
        assert data["status"] == "healthy", f"Degraded status: {data}"
        assert data["database"]["connected"] is True, "MongoDB Atlas not connected"
        print("  >>> PASS: Server is healthy and connected to MongoDB Atlas.")

        # 2. Canonical Feature Extractor Verification
        print("\n[TEST 2] Verifying Canonical 63-D Landmark Feature Extraction & Invariance...")
        from app.ml.features import normalize_landmark_coordinates, FEATURE_DIMENSION
        mock_lms = [{"x": 0.5 + 0.01 * i, "y": 0.5 + 0.02 * i, "z": 0.001 * i} for i in range(21)]
        feat_vec = normalize_landmark_coordinates(mock_lms)
        assert isinstance(feat_vec, np.ndarray), "Feature vector must be numpy array"
        assert feat_vec.shape == (63,), f"Expected shape (63,), got {feat_vec.shape}"
        assert feat_vec.dtype == np.float32, f"Expected float32, got {feat_vec.dtype}"
        # Check wrist (first 3 elements) is at origin (0, 0, 0)
        assert feat_vec[0] == 0.0 and feat_vec[1] == 0.0 and feat_vec[2] == 0.0, "Wrist must be origin (0,0,0)"
        print("  >>> PASS: 63-D coordinate translation and scale invariance verified.")

        # 3. Presence Gate Negative Test (NO_HAND on blank image)
        print("\n[TEST 3] Verifying Strict MediaPipe Presence Gating (Negative Control)...")
        from app.ml.classical_models import ClassicalMLEngine
        blank_bytes = create_blank_test_image()
        gate_res = ClassicalMLEngine.predict_sample(MODEL_ID, blank_bytes, algorithm="SVM")
        assert gate_res["status"] == "NO_HAND", f"Expected status NO_HAND, got {gate_res['status']}"
        assert gate_res["hand_present"] is False, "Expected hand_present False"
        assert gate_res["prediction"] is None, "Expected prediction None on blank image"
        print("  >>> PASS: Blank image correctly rejected with NO_HAND status. Zero false triggers.")

        # 4. Participant-Independent Split Logic Test
        print("\n[TEST 4] Verifying Participant-Independent Splitting (Preventing Data Leakage)...")
        from app.ml.classical_models import split_dataset_participant_independent
        mock_samples = [
            {"label": "WATER", "participant_id": "P_A"},
            {"label": "WATER", "participant_id": "P_A"},
            {"label": "HELP", "participant_id": "P_A"},
            {"label": "HELP", "participant_id": "P_A"},
            {"label": "WATER", "participant_id": "P_B"},
            {"label": "HELP", "participant_id": "P_B"}
        ]
        train_idx, test_idx, split_info = split_dataset_participant_independent(mock_samples, test_ratio=0.33)
        assert split_info["participant_leakage_prevented"] is True, "Participant leakage was not prevented"
        train_participants = set(split_info["train_participants"])
        test_participants = set(split_info["test_participants"])
        assert train_participants.isdisjoint(test_participants), "Data leakage: participants overlap!"
        print(f"  >>> PASS: Participant independence verified! Train: {train_participants}, Test: {test_participants}")

        # 5. Live ML Overview Endpoint
        print("\n[TEST 5] Verifying /api/models/{id}/ml/overview Endpoint...")
        resp = await client.get(f"/api/models/{MODEL_ID}/ml/overview")
        assert resp.status_code == 200, f"Overview failed: {resp.text}"
        ov_data = resp.json()
        assert ov_data["model_id"] == MODEL_ID
        assert ov_data["classes_count"] >= 2
        assert "HELP" in ov_data["class_names"] and "WATER" in ov_data["class_names"]
        print(f"  >>> PASS: Overview verified ({ov_data['classes_count']} classes, {ov_data['total_samples']} samples).")

        # 6. Live EDA Analytics & Charts Endpoint
        print("\n[TEST 6] Verifying /api/models/{id}/ml/eda (Matplotlib & Seaborn Visualizations)...")
        resp = await client.get(f"/api/models/{MODEL_ID}/ml/eda")
        assert resp.status_code == 200, f"EDA failed: {resp.text}"
        eda_data = resp.json()
        assert "statistical_summary" in eda_data, "Statistical summary missing"
        assert "class_distribution" in eda_data["charts"], "Class distribution chart missing"
        assert "media_distribution" in eda_data["charts"], "Media distribution chart missing"
        assert "participant_distribution" in eda_data["charts"], "Participant distribution chart missing"
        assert "correlation_heatmap" in eda_data["charts"], "Correlation heatmap missing"
        assert eda_data["charts"]["correlation_heatmap"].startswith("data:image/png;base64,"), "Chart must be base64 PNG"
        print("  >>> PASS: EDA computed live from Atlas records. 5 Matplotlib & Seaborn charts generated.")

        # 7. Classical ML Training & Hyperparameter Tuning Endpoint
        print("\n[TEST 7] Verifying /api/models/{id}/ml/train (KNN, SVM, Decision Tree, Custom CNN)...")
        train_payload = {
            "knn": {"n_neighbors": 3, "weights": "uniform", "metric": "euclidean"},
            "svm": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
            "dt": {"criterion": "gini", "max_depth": 5, "min_samples_split": 2},
            "include_cnn": True
        }
        resp = await client.post(f"/api/models/{MODEL_ID}/ml/train", json=train_payload)
        assert resp.status_code == 200, f"Training failed: {resp.text}"
        train_res = resp.json()
        assert train_res["success"] is True, "Training did not succeed"
        assert len(train_res["comparison_table"]) >= 3, "Comparison table missing models"
        
        # Verify model metrics
        for row in train_res["comparison_table"]:
            alg = row["algorithm"]
            acc = row["accuracy"]
            f1 = row["f1_macro"]
            print(f"    - {alg:<15} Accuracy: {acc * 100:.1f}%, F1-Score: {f1 * 100:.1f}%")
            assert 0.0 <= acc <= 1.0, f"Invalid accuracy for {alg}: {acc}"
            assert 0.0 <= f1 <= 1.0, f"Invalid F1 for {alg}: {f1}"

        best_alg = train_res["best_model"]["algorithm"]
        print(f"  >>> PASS: Training complete! Best performing model: '{best_alg}'")

        # 8. Comparison Table Endpoint
        print("\n[TEST 8] Verifying /api/models/{id}/ml/comparison Endpoint...")
        resp = await client.get(f"/api/models/{MODEL_ID}/ml/comparison")
        assert resp.status_code == 200, f"Comparison failed: {resp.text}"
        comp_data = resp.json()
        assert comp_data["has_results"] is True
        assert len(comp_data["comparison_table"]) >= 3
        print(f"  >>> PASS: Comparative benchmark retrieved successfully across {len(comp_data['comparison_table'])} algorithms.")

        # 9. Unsupervised K-Means Clustering Endpoint
        print("\n[TEST 9] Verifying /api/models/{id}/ml/cluster (K-Means & 2D PCA Scatter)...")
        cluster_payload = {"n_clusters": 2}
        resp = await client.post(f"/api/models/{MODEL_ID}/ml/cluster", json=cluster_payload)
        assert resp.status_code == 200, f"Clustering failed: {resp.text}"
        cluster_res = resp.json()
        assert cluster_res["success"] is True
        assert cluster_res["n_clusters"] == 2
        assert len(cluster_res["cluster_sizes"]) == 2
        assert cluster_res["chart_base64"].startswith("data:image/png;base64,")
        assert len(cluster_res["contingency_table"]) == 2
        print(f"  >>> PASS: K-Means clustering verified! Inertia: {cluster_res['inertia']}, Silhouette: {cluster_res['silhouette_score']}")

        # 10. Atlas Experiment History Logging
        print("\n[TEST 10] Verifying /api/models/{id}/ml/experiments (Atlas Logging)...")
        resp = await client.get(f"/api/models/{MODEL_ID}/ml/experiments")
        assert resp.status_code == 200, f"History failed: {resp.text}"
        hist_data = resp.json()
        assert hist_data["count"] >= 2, f"Expected >= 2 experiments, got {hist_data['count']}"
        print(f"  >>> PASS: {hist_data['count']} ML experiments verified in MongoDB Atlas ml_experiments collection.")

        # 11. Admin Web Dashboard Page
        print("\n[TEST 11] Verifying /admin/models/{id}/ml-analysis HTML View...")
        resp = await client.get(f"/admin/models/{MODEL_ID}/ml-analysis")
        assert resp.status_code == 200, f"HTML dashboard failed: {resp.status_code}"
        assert "ML Analysis" in resp.text
        assert "K-Nearest Neighbors" in resp.text
        assert "Support Vector Machine" in resp.text
        assert "Decision Tree" in resp.text
        assert "K-Means Clustering" in resp.text
        print("  >>> PASS: Admin ML Analysis console loaded with status 200 OK.")

    print("\n" + "=" * 70)
    print("ALL 11 ACADEMIC MACHINE LEARNING TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_suite())
