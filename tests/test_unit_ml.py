"""
SignBridge Offline Unit Test Suite
Designed for local verification and automated GitHub Actions CI/CD workflows.
Runs without requiring external camera hardware or live cloud database connectivity.
"""

import unittest
import numpy as np
import torch

from app.ml.features import InvariantLandmarkExtractor
from app.ml.architectures import CustomSignCNN
from app.ml.classical_models import ClassicalMLEngine
from app.ml.clustering import LandmarkClusterEngine


class TestInvariantLandmarkExtractor(unittest.TestCase):
    """Verifies invariant 63-D hand-landmark feature representation."""

    def setUp(self):
        self.extractor = InvariantLandmarkExtractor()

    def test_presence_gating_none_or_empty(self):
        """Missing landmarks must return None (triggering NO_HAND)."""
        self.assertIsNone(self.extractor.extract_from_landmarks(None))
        self.assertIsNone(self.extractor.extract_from_landmarks([]))

    def test_invariant_normalization(self):
        """Wrist must be at origin (0,0,0) and feature vector dimension must be 63."""
        # Create synthetic 21 landmarks with wrist at (10, 20, 30)
        raw_landmarks = []
        for i in range(21):
            raw_landmarks.append({
                "x": 10.0 + i * 2.0,
                "y": 20.0 + i * 3.0,
                "z": 30.0 + i * 1.0,
            })

        features = self.extractor.extract_from_landmarks(raw_landmarks)
        self.assertIsNotNone(features)
        self.assertEqual(features.shape, (63,))
        self.assertEqual(features.dtype, np.float32)

        # First 3 coordinates (wrist Landmark 0) must be exactly 0.0, 0.0, 0.0
        self.assertAlmostEqual(features[0], 0.0, places=5)
        self.assertAlmostEqual(features[1], 0.0, places=5)
        self.assertAlmostEqual(features[2], 0.0, places=5)


class TestCustomSignCNN(unittest.TestCase):
    """Verifies PyTorch CustomSignCNN architecture and tensor shapes."""

    def test_forward_pass_shape(self):
        """Input batch of (2, 3, 224, 224) must output (2, num_classes)."""
        num_classes = 4
        model = CustomSignCNN(num_classes=num_classes)
        model.eval()

        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)

        self.assertEqual(output.shape, (2, num_classes))

    def test_single_class_or_binary(self):
        """Model initializes and processes binary classification head."""
        model = CustomSignCNN(num_classes=2)
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        self.assertEqual(output.shape, (1, 2))


class TestClassicalMLEngine(unittest.TestCase):
    """Verifies KNN, SVM, and Decision Tree training and inference pipeline."""

    def setUp(self):
        self.engine = ClassicalMLEngine()
        # Create 20 synthetic samples across 2 classes (10 samples each, 63-D features)
        np.random.seed(42)
        self.X_cls0 = np.random.normal(loc=0.0, scale=0.1, size=(10, 63)).astype(np.float32)
        self.X_cls1 = np.random.normal(loc=1.0, scale=0.1, size=(10, 63)).astype(np.float32)
        self.X = np.vstack([self.X_cls0, self.X_cls1])
        self.y = np.array(["CLASS_A"] * 10 + ["CLASS_B"] * 10)
        self.participants = ["P1"] * 6 + ["P2"] * 4 + ["P1"] * 6 + ["P2"] * 4

    def test_training_and_evaluation(self):
        """Train KNN, SVM, and Decision Tree on synthetic landmark vectors."""
        results = self.engine.train_all(self.X, self.y, self.participants)

        self.assertIn("knn", results)
        self.assertIn("svm", results)
        self.assertIn("decision_tree", results)
        self.assertIn("best_model", results)

        # High synthetic separability should yield high accuracy
        self.assertGreaterEqual(results["knn"]["metrics"]["accuracy"], 0.8)
        self.assertGreaterEqual(results["svm"]["metrics"]["accuracy"], 0.8)

    def test_inference_after_training(self):
        """Model inference must output valid predicted class and confidence scores."""
        self.engine.train_all(self.X, self.y, self.participants)

        # Test single query vector from Class A distribution
        query = np.random.normal(loc=0.0, scale=0.05, size=(63,)).astype(np.float32)
        pred = self.engine.predict(query, model_name="svm")

        self.assertIn(pred["predicted_class"], ["CLASS_A", "CLASS_B"])
        self.assertGreater(pred["confidence"], 0.5)


class TestLandmarkClusterEngine(unittest.TestCase):
    """Verifies unsupervised K-Means clustering and 2D PCA projection."""

    def test_clustering_and_pca(self):
        np.random.seed(42)
        X = np.vstack([
            np.random.normal(loc=0.0, scale=0.2, size=(8, 63)),
            np.random.normal(loc=2.0, scale=0.2, size=(8, 63)),
        ]).astype(np.float32)
        labels = ["SIGN_1"] * 8 + ["SIGN_2"] * 8

        cluster_engine = LandmarkClusterEngine()
        result = cluster_engine.fit_and_project(X, labels, n_clusters=2)

        self.assertIn("inertia", result)
        self.assertIn("silhouette_score", result)
        self.assertIn("pca_2d_points", result)
        self.assertIn("contingency_table", result)
        self.assertEqual(len(result["pca_2d_points"]), 16)


if __name__ == "__main__":
    unittest.main()
