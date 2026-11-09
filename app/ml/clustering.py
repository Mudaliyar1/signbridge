"""
SignBridge Advanced Machine Learning: Unsupervised K-Means Clustering Layer.

Features:
- Extracts invariant 63-D MediaPipe hand landmarks from valid dataset samples.
- Fits KMeans(n_clusters=k, random_state=42, n_init=10) on scaled feature vectors.
- Projects clusters to 2D via Principal Component Analysis (PCA).
- Generates high-resolution Matplotlib/Seaborn scatter visualization.
- Computes cluster sizes, inertia, and silhouette score.
- Cross-tabulation (contingency table) against ground-truth sign labels.
- Strictly unsupervised: does NOT use sign labels during clustering fitting.
"""

import base64
import io
import logging
import time
from typing import Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from app.database import experiments_col
from app.ml.classical_models import ClassicalMLEngine

logger = logging.getLogger("signbridge.ml.clustering")


def fig_to_base64(fig: plt.Figure) -> str:
    """Converts figure into base64 data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


class KMeansClusteringService:
    """
    Executes unsupervised K-Means clustering on normalized hand-landmark vectors.
    """

    @classmethod
    async def run_clustering(
        cls,
        model_id: str,
        n_clusters: int = 2,
        random_state: int = 42
    ) -> dict[str, Any]:
        """
        Runs full K-Means pipeline for model_id with specified cluster count.
        """
        start_time = time.time()
        logger.info(f"Running K-Means clustering (K={n_clusters}) for model {model_id}...")

        # 1. Extract 63-D landmark features
        dataset = await ClassicalMLEngine.extract_dataset_features(model_id)
        X: np.ndarray = dataset["X"]
        y: np.ndarray = dataset["y"]
        class_names: list[str] = dataset["class_names"]
        samples_meta: list[dict] = dataset["samples_meta"]

        n_samples = len(X)
        if n_samples < n_clusters:
            raise ValueError(
                f"Cannot cluster {n_samples} sample(s) into {n_clusters} clusters. "
                f"Sample count must be greater than or equal to K."
            )

        # 2. Feature Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 3. K-Means Unsupervised Fitting
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        centroids = kmeans.cluster_centers_
        inertia = float(kmeans.inertia_)

        # Silhouette score if samples > n_clusters
        sil_score = None
        if n_samples > n_clusters:
            try:
                sil_score = round(float(silhouette_score(X_scaled, cluster_labels)), 4)
            except Exception:
                sil_score = None

        # 4. Cluster Size Analysis
        cluster_sizes: dict[str, int] = {}
        for c_id in range(n_clusters):
            count = int(np.sum(cluster_labels == c_id))
            cluster_sizes[f"Cluster {c_id}"] = count

        # 5. Cross-Tabulation against Ground-Truth Sign Labels
        # (Evaluates natural visual grouping without using labels for training)
        contingency_matrix = {}
        for c_id in range(n_clusters):
            contingency_matrix[f"Cluster {c_id}"] = {cn: 0 for cn in class_names}

        for idx, c_id in enumerate(cluster_labels):
            true_label = class_names[y[idx]]
            contingency_matrix[f"Cluster {c_id}"][true_label] += 1

        # Format contingency table for display
        contingency_table = []
        for c_name, counts_dict in contingency_matrix.items():
            row = {"cluster": c_name}
            row.update(counts_dict)
            row["total"] = sum(counts_dict.values())
            contingency_table.append(row)

        # 6. PCA 2D Dimensionality Reduction & Visualization
        pca = PCA(n_components=2, random_state=random_state)
        X_pca = pca.fit_transform(X_scaled)
        centroids_pca = pca.transform(centroids)
        var_explained = [round(float(v) * 100.0, 1) for v in pca.explained_variance_ratio_]

        # Render Matplotlib / Seaborn Scatter Plot
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        palette = sns.color_palette("tab10", n_clusters)

        for c_id in range(n_clusters):
            mask = (cluster_labels == c_id)
            ax.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                label=f"Cluster {c_id} ({np.sum(mask)} samples)",
                color=palette[c_id],
                s=80,
                alpha=0.85,
                edgecolors="#ffffff",
                linewidth=0.8
            )

        # Plot Centroids
        ax.scatter(
            centroids_pca[:, 0],
            centroids_pca[:, 1],
            color="#ef4444",
            marker="X",
            s=220,
            linewidths=2,
            edgecolors="#ffffff",
            label="Cluster Centroids",
            zorder=10
        )

        ax.set_title(
            f"K-Means Clustering (K={n_clusters}) & 2D PCA Projection",
            fontsize=13,
            fontweight="bold",
            pad=14,
            color="#38bdf8"
        )
        ax.set_xlabel(f"Principal Component 1 ({var_explained[0]}% variance)", fontsize=10)
        ax.set_ylabel(f"Principal Component 2 ({var_explained[1]}% variance)", fontsize=10)
        ax.legend(loc="best", framealpha=0.3, fontsize=9)
        plt.tight_layout()

        chart_base64 = fig_to_base64(fig)

        # 7. Academic Interpretation Summary
        dominant_classes = []
        for c_id in range(n_clusters):
            row_dict = contingency_matrix[f"Cluster {c_id}"]
            if row_dict:
                top_cls = max(row_dict.items(), key=lambda x: x[1])
                dominant_classes.append(f"Cluster {c_id}: dominated by '{top_cls[0]}' ({top_cls[1]} samples)")

        interpretation = (
            f"K-Means successfully partitioned {n_samples} hand-sign landmark samples into {n_clusters} clusters "
            f"using 63-dimensional normalized keypoint coordinates. PCA explains {sum(var_explained):.1f}% of the total variance. "
            + "; ".join(dominant_classes) + ". "
            "Note: This is an unsupervised grouping demonstrating geometric clustering without label guidance."
        )

        # 8. Persist in MongoDB Atlas ml_experiments
        experiment_record = {
            "model_id": model_id,
            "algorithm": "KMEANS",
            "hyperparameters": {
                "n_clusters": n_clusters,
                "random_state": random_state,
                "scaling": "StandardScaler"
            },
            "metrics": {
                "inertia": round(inertia, 2),
                "silhouette_score": sil_score,
                "total_samples": n_samples,
                "cluster_sizes": cluster_sizes,
                "pca_variance_ratio": var_explained
            },
            "contingency_table": contingency_table,
            "interpretation": interpretation,
            "status": "COMPLETED",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        try:
            await experiments_col().insert_one(experiment_record)
        except Exception as e:
            logger.error(f"Failed to log K-Means experiment in Atlas: {e}")

        duration = round(time.time() - start_time, 2)

        return {
            "success": True,
            "model_id": model_id,
            "n_clusters": n_clusters,
            "total_samples": n_samples,
            "inertia": round(inertia, 2),
            "silhouette_score": sil_score,
            "cluster_sizes": cluster_sizes,
            "pca_variance_explained": var_explained,
            "contingency_table": contingency_table,
            "class_names": class_names,
            "chart_base64": chart_base64,
            "interpretation": interpretation,
            "duration_seconds": duration
        }
