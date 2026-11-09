"""
SignBridge Classical Machine Learning Engine:
K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Decision Tree.

Features:
- Canonical 63-D MediaPipe hand-landmark representation
- Participant-independent data splitting (preventing data leakage)
- Data scaling fitted strictly on training partition
- Basic hyperparameter tuning (GridSearchCV with Cross-Validation)
- Comprehensive academic evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
- Joint comparison against CustomSignCNN
- Model artifact persistence and unified inference
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import httpx
import torch

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from app.config import settings
from app.database import models_col, classes_col, samples_col, experiments_col
from app.ml.features import (
    extract_features_from_image,
    normalize_landmark_coordinates,
    FEATURE_DIMENSION,
    FEATURE_VERSION
)
from app.ml.training import fetch_sample_bytes
from app.ml.inference import CustomSignInferenceService

logger = logging.getLogger("signbridge.ml.classical")

ARTIFACT_VERSION = "classical_v1.0.0"


def get_classical_storage_dir(model_id: str) -> Path:
    """Returns local storage path for classical model artifacts."""
    p = settings.MODELS_DIR / model_id / "classical"
    p.mkdir(parents=True, exist_ok=True)
    return p


# In-memory cache for loaded classical models and scalers
_CLASSICAL_CACHE: dict[str, dict[str, Any]] = {}


def split_dataset_participant_independent(
    samples_data: list[dict],
    test_ratio: float = 0.3,
    random_state: int = 42
) -> tuple[list[int], list[int], dict[str, Any]]:
    """
    Splits samples ensuring participants in the test set NEVER appear in the training set.
    If only 1 participant is available, falls back to stratified splitting and flags the limitation.
    
    Returns:
        (train_indices, test_indices, split_info_dict)
    """
    total = len(samples_data)
    if total < 2:
        return [0], [0], {
            "strategy": "insufficient_data",
            "warning": "Dataset too small for meaningful train/test partitioning."
        }

    # Group sample indices by participant_id
    participant_to_indices: dict[str, list[int]] = {}
    for idx, s in enumerate(samples_data):
        pid = s.get("participant_id") or "UNKNOWN_PARTICIPANT"
        participant_to_indices.setdefault(pid, []).append(idx)

    unique_participants = list(participant_to_indices.keys())
    rng = np.random.RandomState(random_state)

    if len(unique_participants) >= 2:
        # Genuine Participant-Independent Split
        shuffled_participants = unique_participants.copy()
        rng.shuffle(shuffled_participants)

        num_test_participants = max(1, int(round(len(unique_participants) * test_ratio)))
        test_participants = set(shuffled_participants[:num_test_participants])
        train_participants = set(shuffled_participants[num_test_participants:])

        # Guard: Ensure train set is not empty
        if not train_participants:
            train_participants = {shuffled_participants[0]}
            test_participants = set(shuffled_participants[1:])

        train_indices = []
        for p in train_participants:
            train_indices.extend(participant_to_indices[p])

        test_indices = []
        for p in test_participants:
            test_indices.extend(participant_to_indices[p])

        # Verify class coverage in train indices
        train_labels = {samples_data[i]["label"] for i in train_indices}
        all_labels = {s["label"] for s in samples_data}
        
        # If participant split isolated a class entirely in test, fallback to stratified
        if len(train_labels) < len(all_labels) and len(all_labels) >= 2:
            logger.warning(
                "Participant-independent split omitted a class from training. "
                "Adjusting to stratified splitting to guarantee class representation."
            )
            return _stratified_fallback_split(samples_data, test_ratio, rng, unique_participants)

        split_info = {
            "strategy": "participant_independent",
            "participant_leakage_prevented": True,
            "total_participants": len(unique_participants),
            "train_participants": list(train_participants),
            "test_participants": list(test_participants),
            "train_samples_count": len(train_indices),
            "test_samples_count": len(test_indices),
            "warning": None
        }
        return train_indices, test_indices, split_info
    else:
        # Fallback when only 1 participant exists in dataset
        return _stratified_fallback_split(samples_data, test_ratio, rng, unique_participants)


def _stratified_fallback_split(
    samples_data: list[dict],
    test_ratio: float,
    rng: np.random.RandomState,
    unique_participants: list[str]
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Stratified splitting fallback when participants < 2 or single participant spans all data."""
    label_to_indices: dict[str, list[int]] = {}
    for idx, s in enumerate(samples_data):
        label_to_indices.setdefault(s["label"], []).append(idx)

    train_indices = []
    test_indices = []

    for lbl, idxs in label_to_indices.items():
        shuffled = idxs.copy()
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            train_indices.append(shuffled[0])
            test_indices.append(shuffled[0])  # Duplicate for test if only 1 sample
        else:
            n_test = max(1, int(round(len(shuffled) * test_ratio)))
            test_indices.extend(shuffled[:n_test])
            train_indices.extend(shuffled[n_test:])

    # Final guard
    if not train_indices:
        train_indices = test_indices.copy()

    split_info = {
        "strategy": "stratified_single_participant_fallback",
        "participant_leakage_prevented": False,
        "total_participants": len(unique_participants),
        "train_participants": unique_participants,
        "test_participants": unique_participants,
        "train_samples_count": len(train_indices),
        "test_samples_count": len(test_indices),
        "warning": (
            "Dataset currently contains only 1 participant or insufficient participants. "
            "A stratified split was utilized; participant-independent generalization cannot be verified "
            "until samples from >= 2 participants are collected."
        )
    }
    return train_indices, test_indices, split_info


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    class_names: list[str]
) -> dict[str, Any]:
    """
    Computes genuine academic evaluation metrics:
    - Accuracy
    - Precision (macro, weighted)
    - Recall (macro, weighted)
    - F1-Score (macro, weighted)
    - Multi-Class ROC-AUC (OvR macro, when valid)
    - Confusion Matrix
    - Classification Report
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_list = cm.tolist()

    # Classification Report
    try:
        report_dict = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
    except Exception:
        report_dict = {}

    # Multi-class ROC-AUC (One-vs-Rest)
    roc_auc = None
    roc_auc_explanation = None
    if y_proba is not None and len(class_names) >= 2:
        try:
            # Check if all classes are present in y_true
            unique_true = set(y_true)
            if len(unique_true) >= 2:
                if len(class_names) == 2:
                    # Binary ROC-AUC using positive class probability
                    # Column 1 probability
                    roc_auc = float(roc_auc_score(y_true, y_proba[:, 1]))
                    roc_auc_explanation = "Binary ROC-AUC computed on class 1 probabilities."
                else:
                    # Multi-class OvR ROC-AUC
                    roc_auc = float(roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class="ovr",
                        average="macro",
                        labels=list(range(len(class_names)))
                    ))
                    roc_auc_explanation = "Multi-class One-vs-Rest (OvR) macro-averaged ROC-AUC."
            else:
                roc_auc = None
                roc_auc_explanation = "ROC-AUC undefined: Held-out test set contains samples from only one class."
        except Exception as e:
            roc_auc = None
            roc_auc_explanation = f"ROC-AUC calculation skipped: {str(e)}"
    else:
        roc_auc_explanation = "Probabilities unavailable or class count < 2."

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_macro": round(rec_macro, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_macro": round(f1_m, 4),
        "f1_weighted": round(f1_w, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "roc_auc_explanation": roc_auc_explanation,
        "confusion_matrix": cm_list,
        "classification_report": report_dict
    }


class ClassicalMLEngine:
    """
    Manages extraction, training, hyperparameter tuning, evaluation,
    and inference for KNN, SVM, and Decision Tree.
    """

    @classmethod
    async def extract_dataset_features(cls, model_id: str) -> dict[str, Any]:
        """
        Loads all VALID dataset samples for model_id and extracts 63-D landmark features.
        Returns extracted feature matrices, labels, participant references, and raw sample records.
        """
        # Fetch classes
        classes_cursor = classes_col().find({"model_id": model_id}).sort("name", 1)
        classes_list = await classes_cursor.to_list(100)
        if len(classes_list) < 2:
            raise ValueError(f"Model has {len(classes_list)} class(es). At least 2 classes are required.")

        class_id_to_idx = {c["id"]: i for i, c in enumerate(classes_list)}
        class_name_to_idx = {c["name"]: i for i, c in enumerate(classes_list)}
        class_names = [c["name"] for c in classes_list]

        # Fetch VALID samples
        samples_cursor = samples_col().find({"model_id": model_id, "status": "VALID"})
        samples = await samples_cursor.to_list(1000)
        if len(samples) < 2:
            raise ValueError(f"Found {len(samples)} valid sample(s). Need at least 2 samples to train ML models.")

        X_list = []
        y_list = []
        sample_meta_list = []

        async with httpx.AsyncClient(timeout=30.0) as http_client:
            for s in samples:
                cid = s.get("class_id")
                if cid not in class_id_to_idx:
                    continue
                label_idx = class_id_to_idx[cid]

                # Check if sample document already contains pre-extracted 21 landmarks
                feat_vec = None
                raw_lms = None
                if s.get("landmarks") and len(s["landmarks"]) == 21:
                    try:
                        feat_vec = normalize_landmark_coordinates(s["landmarks"])
                        raw_lms = s["landmarks"]
                    except Exception:
                        feat_vec = None
                elif s.get("features") and len(s["features"]) == 63:
                    feat_vec = np.array(s["features"], dtype=np.float32)

                media_bytes = None
                if feat_vec is None:
                    media_bytes = await fetch_sample_bytes(s, client=http_client)
                    if media_bytes:
                        feat_res = extract_features_from_image(media_bytes)
                        if feat_res["hand_present"] and feat_res["features"] is not None:
                            feat_vec = feat_res["features"]
                            raw_lms = feat_res.get("raw_landmarks")
                        else:
                            logger.debug(f"Sample {s['id']} skipped by feature extractor: {feat_res.get('reason')}")
                else:
                    # Retrieve media bytes if available for optional CNN evaluation
                    try:
                        media_bytes = await fetch_sample_bytes(s, client=http_client)
                    except Exception:
                        media_bytes = None

                if feat_vec is not None:
                    X_list.append(feat_vec)
                    y_list.append(label_idx)
                    sample_meta_list.append({
                        "id": s["id"],
                        "class_id": cid,
                        "label": s["label"],
                        "participant_id": s.get("participant_id") or "Participant 001",
                        "media_type": s.get("media_type", "IMAGE"),
                        "features": feat_vec,
                        "media_bytes": media_bytes
                    })

        if len(X_list) < 2:
            raise ValueError(
                f"Extracted valid hand landmarks from only {len(X_list)} sample(s). "
                "Ensure dataset contains clear hand-sign images."
            )

        unique_labels = set(y_list)
        if len(unique_labels) < 2:
            raise ValueError(
                f"Valid landmark samples belong to only 1 sign class ({class_names[list(unique_labels)[0]]}). "
                "At least 2 distinct sign classes are required."
            )

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)

        return {
            "X": X,
            "y": y,
            "classes_list": classes_list,
            "class_names": class_names,
            "class_id_to_idx": class_id_to_idx,
            "samples_meta": sample_meta_list
        }

    @classmethod
    async def train_and_evaluate_all(
        cls,
        model_id: str,
        knn_params: dict | None = None,
        svm_params: dict | None = None,
        dt_params: dict | None = None,
        include_cnn: bool = True
    ) -> dict[str, Any]:
        """
        Executes unified training, hyperparameter tuning, and comparative evaluation
        for KNN, SVM, Decision Tree, and Custom CNN.
        """
        start_overall = time.time()
        logger.info(f"Initiating classical ML training pipeline for model {model_id}...")

        # 1. Extract consistent landmark features
        dataset = await cls.extract_dataset_features(model_id)
        X: np.ndarray = dataset["X"]
        y: np.ndarray = dataset["y"]
        class_names: list[str] = dataset["class_names"]
        samples_meta: list[dict] = dataset["samples_meta"]

        # 2. Participant-Independent Split
        train_idx, test_idx, split_info = split_dataset_participant_independent(samples_meta)
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        logger.info(f"Dataset split: {len(X_train)} train, {len(X_test)} test. Strategy: {split_info['strategy']}")

        # Ensure directory
        storage_dir = get_classical_storage_dir(model_id)

        results: dict[str, Any] = {}

        # -------------------------------------------------------------
        # 3. K-Nearest Neighbors (KNN)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        knn_k_candidates = [1, 3, 5, 7]
        max_k = min(len(X_train), 7)
        valid_ks = [k for k in knn_k_candidates if k <= max_k] or [1]

        knn_grid = {
            "n_neighbors": valid_ks,
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
        if knn_params:
            if "n_neighbors" in knn_params:
                knn_grid["n_neighbors"] = [min(int(knn_params["n_neighbors"]), len(X_train))]
            if "weights" in knn_params:
                knn_grid["weights"] = [knn_params["weights"]]
            if "metric" in knn_params:
                knn_grid["metric"] = [knn_params["metric"]]

        # Cross-validation folds based on train size and class counts
        min_class_count = min(np.bincount(y_train, minlength=len(class_names)))
        cv_folds = max(2, min(min_class_count, 3)) if min_class_count >= 2 else None

        base_knn = KNeighborsClassifier()
        if cv_folds and cv_folds >= 2:
            knn_search = GridSearchCV(base_knn, knn_grid, cv=cv_folds, scoring="accuracy")
            knn_search.fit(X_train, y_train)
            best_knn = knn_search.best_estimator_
            selected_knn_params = knn_search.best_params_
        else:
            best_knn = KNeighborsClassifier(
                n_neighbors=valid_ks[0],
                weights=knn_grid["weights"][0],
                metric=knn_grid["metric"][0]
            )
            best_knn.fit(X_train, y_train)
            selected_knn_params = {
                "n_neighbors": valid_ks[0],
                "weights": knn_grid["weights"][0],
                "metric": knn_grid["metric"][0]
            }

        knn_train_duration = round(time.perf_counter() - t0, 3)

        # Inference on test set
        t_inf_start = time.perf_counter()
        knn_preds = best_knn.predict(X_test)
        knn_proba = best_knn.predict_proba(X_test) if hasattr(best_knn, "predict_proba") else None
        knn_inf_ms = round(((time.perf_counter() - t_inf_start) / max(1, len(X_test))) * 1000.0, 3)

        knn_metrics = compute_comprehensive_metrics(y_test, knn_preds, knn_proba, class_names)
        knn_metrics["inference_time_ms"] = knn_inf_ms
        knn_metrics["training_duration_s"] = knn_train_duration

        # Save KNN artifact
        knn_path = storage_dir / "knn_model.joblib"
        joblib.dump(best_knn, knn_path)

        results["KNN"] = {
            "algorithm": "KNN",
            "model_path": str(knn_path),
            "hyperparameters": selected_knn_params,
            "metrics": knn_metrics
        }

        # -------------------------------------------------------------
        # 4. Support Vector Machine (SVM)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        # Scale features strictly on training set!
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        svm_grid = {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"]
        }
        if svm_params:
            if "C" in svm_params:
                svm_grid["C"] = [float(svm_params["C"])]
            if "kernel" in svm_params:
                svm_grid["kernel"] = [svm_params["kernel"]]
            if "gamma" in svm_params:
                svm_grid["gamma"] = [svm_params["gamma"]]

        base_svm = SVC(probability=True, random_state=42)
        if cv_folds and cv_folds >= 2:
            svm_search = GridSearchCV(base_svm, svm_grid, cv=cv_folds, scoring="accuracy")
            svm_search.fit(X_train_scaled, y_train)
            best_svm = svm_search.best_estimator_
            selected_svm_params = svm_search.best_params_
        else:
            best_svm = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=42)
            best_svm.fit(X_train_scaled, y_train)
            selected_svm_params = {"C": 1.0, "kernel": "rbf", "gamma": "scale"}

        svm_train_duration = round(time.perf_counter() - t0, 3)

        # Inference on test set
        t_inf_start = time.perf_counter()
        svm_preds = best_svm.predict(X_test_scaled)
        svm_proba = best_svm.predict_proba(X_test_scaled)
        svm_inf_ms = round(((time.perf_counter() - t_inf_start) / max(1, len(X_test))) * 1000.0, 3)

        svm_metrics = compute_comprehensive_metrics(y_test, svm_preds, svm_proba, class_names)
        svm_metrics["inference_time_ms"] = svm_inf_ms
        svm_metrics["training_duration_s"] = svm_train_duration

        # Save SVM artifact and Scaler
        svm_path = storage_dir / "svm_model.joblib"
        scaler_path = storage_dir / "svm_scaler.joblib"
        joblib.dump(best_svm, svm_path)
        joblib.dump(scaler, scaler_path)

        results["SVM"] = {
            "algorithm": "SVM",
            "model_path": str(svm_path),
            "scaler_path": str(scaler_path),
            "hyperparameters": selected_svm_params,
            "metrics": svm_metrics
        }

        # -------------------------------------------------------------
        # 5. Decision Tree (DT)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        dt_grid = {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 4],
            "min_samples_leaf": [1, 2]
        }
        if dt_params:
            if "criterion" in dt_params:
                dt_grid["criterion"] = [dt_params["criterion"]]
            if "max_depth" in dt_params:
                dt_grid["max_depth"] = [None if dt_params["max_depth"] in (None, "None") else int(dt_params["max_depth"])]
            if "min_samples_split" in dt_params:
                dt_grid["min_samples_split"] = [int(dt_params["min_samples_split"])]
            if "min_samples_leaf" in dt_params:
                dt_grid["min_samples_leaf"] = [int(dt_params["min_samples_leaf"])]

        base_dt = DecisionTreeClassifier(random_state=42)
        if cv_folds and cv_folds >= 2:
            dt_search = GridSearchCV(base_dt, dt_grid, cv=cv_folds, scoring="accuracy")
            dt_search.fit(X_train, y_train)
            best_dt = dt_search.best_estimator_
            selected_dt_params = dt_search.best_params_
        else:
            best_dt = DecisionTreeClassifier(criterion="gini", max_depth=None, random_state=42)
            best_dt.fit(X_train, y_train)
            selected_dt_params = {"criterion": "gini", "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1}

        dt_train_duration = round(time.perf_counter() - t0, 3)

        # Inference on test set
        t_inf_start = time.perf_counter()
        dt_preds = best_dt.predict(X_test)
        dt_proba = best_dt.predict_proba(X_test)
        dt_inf_ms = round(((time.perf_counter() - t_inf_start) / max(1, len(X_test))) * 1000.0, 3)

        dt_metrics = compute_comprehensive_metrics(y_test, dt_preds, dt_proba, class_names)
        dt_metrics["inference_time_ms"] = dt_inf_ms
        dt_metrics["training_duration_s"] = dt_train_duration

        # Save Decision Tree artifact
        dt_path = storage_dir / "dt_model.joblib"
        joblib.dump(best_dt, dt_path)

        results["Decision Tree"] = {
            "algorithm": "Decision Tree",
            "model_path": str(dt_path),
            "hyperparameters": selected_dt_params,
            "metrics": dt_metrics
        }

        # -------------------------------------------------------------
        # 6. Custom CNN Evaluation (on the identical held-out test partition)
        # -------------------------------------------------------------
        cnn_result_entry = None
        if include_cnn:
            try:
                avail = CustomSignInferenceService.check_model_availability(model_id)
                if avail["available"]:
                    t0 = time.perf_counter()
                    cnn_preds_list = []
                    cnn_proba_list = []
                    valid_cnn_test_targets = []

                    test_samples_meta = [samples_meta[i] for i in test_idx]
                    for sm in test_samples_meta:
                        media_b = sm.get("media_bytes")
                        if not media_b:
                            continue
                        t_single = time.perf_counter()
                        inf_res = CustomSignInferenceService.predict_image(model_id, media_b)
                        if inf_res.get("success") and inf_res.get("prediction"):
                            pred_name = inf_res["prediction"]["class_name"]
                            pred_idx = class_names.index(pred_name) if pred_name in class_names else 0
                            cnn_preds_list.append(pred_idx)
                            # Top probabilities
                            top_preds = inf_res.get("top_predictions", [])
                            prob_vec = np.zeros(len(class_names), dtype=np.float32)
                            for tp in top_preds:
                                cn = tp.get("class_name")
                                if cn in class_names:
                                    prob_vec[class_names.index(cn)] = tp.get("confidence", 0.0)
                            cnn_proba_list.append(prob_vec)
                            valid_cnn_test_targets.append(dataset["class_id_to_idx"][sm["class_id"]])

                    if len(cnn_preds_list) > 0 and len(valid_cnn_test_targets) > 0:
                        cnn_y_true = np.array(valid_cnn_test_targets, dtype=np.int64)
                        cnn_y_pred = np.array(cnn_preds_list, dtype=np.int64)
                        cnn_y_proba = np.array(cnn_proba_list, dtype=np.float32) if cnn_proba_list else None

                        cnn_metrics = compute_comprehensive_metrics(cnn_y_true, cnn_y_pred, cnn_y_proba, class_names)
                        cnn_inf_ms = round(((time.perf_counter() - t0) / max(1, len(cnn_preds_list))) * 1000.0, 3)
                        cnn_metrics["inference_time_ms"] = cnn_inf_ms
                        cnn_metrics["training_duration_s"] = None  # Trained separately via PyTorch loop

                        cnn_result_entry = {
                            "algorithm": "Custom CNN",
                            "architecture": "CustomSignCNN (4 Conv Blocks)",
                            "model_path": avail.get("weights_path"),
                            "hyperparameters": {
                                "input_resolution": "224x224",
                                "channels": 3,
                                "framework": "PyTorch"
                            },
                            "metrics": cnn_metrics
                        }
                        results["Custom CNN"] = cnn_result_entry
            except Exception as e:
                logger.warning(f"Custom CNN evaluation encountered issue: {e}")

        # -------------------------------------------------------------
        # 7. Model Comparison Table and Winner Determination
        # -------------------------------------------------------------
        comparison_table = []
        for alg_name, item in results.items():
            m = item["metrics"]
            comparison_table.append({
                "algorithm": alg_name,
                "accuracy": m["accuracy"],
                "precision_macro": m["precision_macro"],
                "recall_macro": m["recall_macro"],
                "f1_macro": m["f1_macro"],
                "f1_weighted": m["f1_weighted"],
                "roc_auc": m["roc_auc"],
                "inference_time_ms": m["inference_time_ms"],
                "training_duration_s": m["training_duration_s"]
            })

        # Best-performing model based on actual F1-macro (tie-break on Accuracy)
        best_model = max(comparison_table, key=lambda x: (x["f1_macro"], x["accuracy"]))

        # -------------------------------------------------------------
        # 8. Persist Classical Metadata Checkpoint
        # -------------------------------------------------------------
        meta_content = {
            "model_id": model_id,
            "version": ARTIFACT_VERSION,
            "feature_version": FEATURE_VERSION,
            "feature_dimension": FEATURE_DIMENSION,
            "class_names": class_names,
            "split_info": split_info,
            "comparison_table": comparison_table,
            "best_model": best_model,
            "trained_at": time.time(),
            "models": {
                "KNN": {
                    "path": str(storage_dir / "knn_model.joblib"),
                    "hyperparameters": results["KNN"]["hyperparameters"],
                    "metrics": results["KNN"]["metrics"]
                },
                "SVM": {
                    "path": str(storage_dir / "svm_model.joblib"),
                    "scaler_path": str(storage_dir / "svm_scaler.joblib"),
                    "hyperparameters": results["SVM"]["hyperparameters"],
                    "metrics": results["SVM"]["metrics"]
                },
                "Decision Tree": {
                    "path": str(storage_dir / "dt_model.joblib"),
                    "hyperparameters": results["Decision Tree"]["hyperparameters"],
                    "metrics": results["Decision Tree"]["metrics"]
                }
            }
        }
        meta_file = storage_dir / "classical_meta.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_content, f, indent=2)

        # Clear in-memory cache for this model
        _CLASSICAL_CACHE.pop(model_id, None)

        # -------------------------------------------------------------
        # 9. Log in MongoDB Atlas ml_experiments Collection
        # -------------------------------------------------------------
        experiment_doc = {
            "model_id": model_id,
            "algorithm": "COMPARATIVE_ML_STUDY",
            "dataset_snapshot": {
                "total_samples": len(X),
                "classes": class_names,
                "classes_count": len(class_names),
                "participants_count": split_info.get("total_participants", 1)
            },
            "split_info": split_info,
            "comparison_table": comparison_table,
            "best_model": best_model,
            "models_detail": results,
            "artifact_path": str(storage_dir),
            "status": "COMPLETED",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        try:
            await experiments_col().insert_one(experiment_doc)
            logger.info("Successfully recorded ML experiment in MongoDB Atlas ml_experiments collection.")
        except Exception as e:
            logger.error(f"Failed to log experiment in Atlas: {e}")

        total_duration = round(time.time() - start_overall, 2)
        logger.info(f"Comparative ML pipeline completed in {total_duration}s. Best model: {best_model['algorithm']}")

        return {
            "success": True,
            "model_id": model_id,
            "total_samples": len(X),
            "classes": class_names,
            "split_info": split_info,
            "comparison_table": comparison_table,
            "best_model": best_model,
            "models": results,
            "duration_seconds": total_duration,
            "artifacts_dir": str(storage_dir)
        }

    @classmethod
    def load_classical_bundle(cls, model_id: str) -> dict[str, Any]:
        """Loads trained KNN, SVM, Decision Tree, and Scaler into memory."""
        if model_id in _CLASSICAL_CACHE:
            return _CLASSICAL_CACHE[model_id]

        storage_dir = get_classical_storage_dir(model_id)
        meta_file = storage_dir / "classical_meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"No trained classical models found for model {model_id}. Train first.")

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        knn_p = storage_dir / "knn_model.joblib"
        svm_p = storage_dir / "svm_model.joblib"
        scaler_p = storage_dir / "svm_scaler.joblib"
        dt_p = storage_dir / "dt_model.joblib"

        bundle = {
            "meta": meta,
            "class_names": meta.get("class_names", []),
            "knn": joblib.load(knn_p) if knn_p.exists() else None,
            "svm": joblib.load(svm_p) if svm_p.exists() else None,
            "scaler": joblib.load(scaler_p) if scaler_p.exists() else None,
            "dt": joblib.load(dt_p) if dt_p.exists() else None,
            "loaded_at": time.time()
        }
        _CLASSICAL_CACHE[model_id] = bundle
        return bundle

    @classmethod
    def predict_sample(
        cls,
        model_id: str,
        image_bytes: bytes,
        algorithm: str = "SVM",
        hand_selection: str = "Auto"
    ) -> dict[str, Any]:
        """
        Runs classical model inference on an input image:
        1. MediaPipe 21 landmark detection & presence gate
        2. Invariant 63-D feature extraction
        3. Feature scaling (if SVM)
        4. Prediction via selected algorithm (KNN, SVM, Decision Tree)
        5. Returns predicted sign, confidence, and landmark metadata
        """
        start_time = time.perf_counter()
        norm_alg = algorithm.strip().upper().replace(" ", "_")

        # Extract features
        feat_res = extract_features_from_image(image_bytes, hand_selection=hand_selection)

        # STRICT PRESENCE GATE: Return NO_HAND if hand is absent
        if not feat_res["hand_present"] or feat_res["features"] is None:
            return {
                "success": True,
                "status": "NO_HAND",
                "hand_present": False,
                "prediction": None,
                "confidence": 0.0,
                "algorithm": algorithm,
                "speak": False,
                "message": "NO HAND DETECTED",
                "inference_time_ms": round((time.perf_counter() - start_time) * 1000.0, 2)
            }

        bundle = cls.load_classical_bundle(model_id)
        class_names = bundle["class_names"]
        features = feat_res["features"].reshape(1, -1)

        pred_class_idx = 0
        confidence = 1.0
        probabilities = {}

        if norm_alg in ("KNN", "K_NEAREST_NEIGHBORS"):
            model = bundle["knn"]
            if not model:
                raise ValueError("KNN model is not available.")
            pred_class_idx = int(model.predict(features)[0])
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(features)[0]
                confidence = float(probas[pred_class_idx])
                probabilities = {class_names[i]: float(probas[i]) for i in range(len(class_names))}

        elif norm_alg == "SVM":
            model = bundle["svm"]
            scaler = bundle["scaler"]
            if not model or not scaler:
                raise ValueError("SVM model or scaler is not available.")
            features_scaled = scaler.transform(features)
            pred_class_idx = int(model.predict(features_scaled)[0])
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(features_scaled)[0]
                confidence = float(probas[pred_class_idx])
                probabilities = {class_names[i]: float(probas[i]) for i in range(len(class_names))}

        elif norm_alg in ("DT", "DECISION_TREE"):
            model = bundle["dt"]
            if not model:
                raise ValueError("Decision Tree model is not available.")
            pred_class_idx = int(model.predict(features)[0])
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(features)[0]
                confidence = float(probas[pred_class_idx])
                probabilities = {class_names[i]: float(probas[i]) for i in range(len(class_names))}
        else:
            raise ValueError(f"Unknown classical algorithm '{algorithm}'. Choose KNN, SVM, or Decision Tree.")

        predicted_class_name = class_names[pred_class_idx]
        output_text = predicted_class_name.replace("_", " ").title()

        return {
            "success": True,
            "status": "HAND_DETECTED",
            "hand_present": True,
            "algorithm": algorithm,
            "prediction": {
                "class_index": pred_class_idx,
                "class_name": predicted_class_name,
                "output_text": output_text,
                "confidence": round(confidence, 4)
            },
            "probabilities": probabilities,
            "speak": confidence >= 0.50,
            "handedness": feat_res.get("handedness"),
            "hand_detection_confidence": feat_res.get("confidence"),
            "bbox": feat_res.get("bbox"),
            "inference_time_ms": round((time.perf_counter() - start_time) * 1000.0, 2)
        }
