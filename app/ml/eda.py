"""
SignBridge Exploratory Data Analysis (EDA) Engine.

Generates real dataset analytics and academic visualizations using Matplotlib & Seaborn:
1. Univariate Analysis (class distributions, media formats, confidence scores)
2. Bivariate Analysis (samples per participant, class balance ratios)
3. Numerical Statistical Summaries (mean, median, std, min, max, quartiles)
4. Meaningful Feature Correlation Analysis (MediaPipe 21 landmark spatial coordinates)
5. Production-grade charts rendered as high-DPI Base64 PNGs
"""

import base64
import io
import logging
from typing import Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless backend for web servers
import matplotlib.pyplot as plt
import seaborn as sns

from app.database import models_col, classes_col, participants_col, samples_col
from app.ml.features import (
    normalize_landmark_coordinates,
    extract_features_from_image,
    FEATURE_NAMES,
    LANDMARK_NAMES
)
from app.ml.training import fetch_sample_bytes

logger = logging.getLogger("signbridge.ml.eda")

# Set consistent publication style for Matplotlib/Seaborn
plt.style.use("dark_background")
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#181d24",
    "figure.facecolor": "#0f1318",
    "grid.color": "#2a3441",
    "text.color": "#e0e6ed",
    "axes.labelcolor": "#e0e6ed",
    "xtick.color": "#9ca3af",
    "ytick.color": "#9ca3af"
})


def fig_to_base64(fig: plt.Figure) -> str:
    """Converts a Matplotlib figure into a clean base64 data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


class EDAService:
    """
    Computes genuine Exploratory Data Analysis metrics from MongoDB Atlas records
    and dynamically renders Matplotlib / Seaborn visualization artifacts.
    """

    @classmethod
    async def generate_eda_report(cls, model_id: str) -> dict[str, Any]:
        """
        Gathers real MongoDB Atlas data for model_id and computes comprehensive
        EDA summaries and visualization charts.
        """
        # 1. Fetch Model
        model = await models_col().find_one({"id": model_id})
        if not model:
            raise ValueError(f"Model '{model_id}' not found.")

        # 2. Fetch Classes
        classes = await classes_col().find({"model_id": model_id}).to_list(200)
        class_id_to_name = {c["id"]: c["name"] for c in classes}
        class_names = [c["name"] for c in classes]

        # 3. Fetch Participants
        participants = await participants_col().find({}).to_list(500)
        participant_ids = [p.get("participant_id") for p in participants if p.get("participant_id")]

        # 4. Fetch Samples
        samples = await samples_col().find({"model_id": model_id}).to_list(5000)

        total_samples = len(samples)
        valid_samples = [s for s in samples if s.get("status") == "VALID"]
        invalid_samples = [s for s in samples if s.get("status") != "VALID"]

        images_count = sum(1 for s in samples if s.get("media_type") == "IMAGE")
        videos_count = sum(1 for s in samples if s.get("media_type") == "VIDEO")

        # Samples per class
        class_sample_counts: dict[str, int] = {cn: 0 for cn in class_names}
        for s in samples:
            cid = s.get("class_id")
            cname = class_id_to_name.get(cid, s.get("label", "UNKNOWN"))
            class_sample_counts[cname] = class_sample_counts.get(cname, 0) + 1

        # Samples per participant
        participant_sample_counts: dict[str, int] = {}
        for s in samples:
            pid = s.get("participant_id", "Unknown")
            participant_sample_counts[pid] = participant_sample_counts.get(pid, 0) + 1

        # Sign type distribution (Static vs Dynamic)
        static_count = 0
        dynamic_count = 0
        for s in samples:
            cid = s.get("class_id")
            matching_cls = next((c for c in classes if c["id"] == cid), None)
            sign_type = matching_cls.get("type", "STATIC") if matching_cls else "STATIC"
            if sign_type == "DYNAMIC" or s.get("media_type") == "VIDEO":
                dynamic_count += 1
            else:
                static_count += 1

        # Class imbalance calculation
        counts_list = list(class_sample_counts.values())
        max_samples_in_class = max(counts_list) if counts_list else 0
        min_samples_in_class = min(counts_list) if counts_list else 0
        imbalance_ratio = round(max_samples_in_class / max(1, min_samples_in_class), 2) if min_samples_in_class > 0 else 0.0

        # Numerical statistics on sample properties
        file_sizes = [s.get("file_size", 0) for s in samples if s.get("file_size")]
        widths = [s.get("width", 0) for s in samples if s.get("width")]
        heights = [s.get("height", 0) for s in samples if s.get("height")]

        def compute_stats(data: list[float | int]) -> dict[str, float]:
            if not data:
                return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "q25": 0.0, "q75": 0.0}
            arr = np.array(data, dtype=np.float64)
            return {
                "mean": round(float(np.mean(arr)), 2),
                "median": round(float(np.median(arr)), 2),
                "std": round(float(np.std(arr)), 2),
                "min": round(float(np.min(arr)), 2),
                "max": round(float(np.max(arr)), 2),
                "q25": round(float(np.percentile(arr, 25)), 2),
                "q75": round(float(np.percentile(arr, 75)), 2)
            }

        stats_summary = {
            "file_size_bytes": compute_stats(file_sizes),
            "image_width_px": compute_stats(widths),
            "image_height_px": compute_stats(heights)
        }

        # -------------------------------------------------------------
        # Generate Visualizations (Matplotlib & Seaborn)
        # -------------------------------------------------------------
        charts: dict[str, str] = {}

        # 1. Class Distribution Bar Chart
        if class_sample_counts:
            fig, ax = plt.subplots(figsize=(7, 4.2))
            classes_df = pd.DataFrame(list(class_sample_counts.items()), columns=["Sign Class", "Sample Count"])
            palette = sns.color_palette("crest", len(classes_df))
            barplot = sns.barplot(data=classes_df, x="Sign Class", y="Sample Count", palette=palette, ax=ax)
            ax.set_title(f"Class Distribution: {model.get('name')}", fontsize=13, fontweight="bold", pad=12, color="#38bdf8")
            ax.set_ylabel("Number of Samples", fontsize=10)
            ax.set_xlabel("Sign Vocabulary Classes", fontsize=10)
            for p in barplot.patches:
                val = int(p.get_height())
                ax.annotate(str(val), (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=9, color="#f1f5f9", xytext=(0, 3),
                            textcoords='offset points')
            charts["class_distribution"] = fig_to_base64(fig)

        # 2. Media Type & Sign Distribution Donut Chart
        if total_samples > 0:
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            labels = ["Images (Static)", "Videos (Dynamic)"]
            vals = [images_count, videos_count]
            colors = ["#38bdf8", "#ec4899"]
            # Filter zero entries
            active_labels = [l for l, v in zip(labels, vals) if v > 0]
            active_vals = [v for v in vals if v > 0]
            active_colors = [c for c, v in zip(colors, vals) if v > 0]
            
            if active_vals:
                wedges, texts, autotexts = ax.pie(
                    active_vals,
                    labels=active_labels,
                    autopct="%1.1f%%",
                    colors=active_colors,
                    startangle=140,
                    wedgeprops=dict(width=0.42, edgecolor="#0f1318", linewidth=2)
                )
                for t in texts:
                    t.set_color("#cbd5e1")
                for at in autotexts:
                    at.set_color("#0f1318")
                    at.set_fontweight("bold")
                ax.set_title("Media & Sign Modality Breakdown", fontsize=13, fontweight="bold", pad=12, color="#38bdf8")
                charts["media_distribution"] = fig_to_base64(fig)

        # 3. Participant Contribution Distribution Bar Chart
        if participant_sample_counts:
            fig, ax = plt.subplots(figsize=(7, 4.2))
            part_df = pd.DataFrame(list(participant_sample_counts.items()), columns=["Participant", "Samples Contributed"])
            palette = sns.color_palette("flare", len(part_df))
            barplot = sns.barplot(data=part_df, x="Participant", y="Samples Contributed", palette=palette, ax=ax)
            ax.set_title("Samples Contributed per Participant", fontsize=13, fontweight="bold", pad=12, color="#38bdf8")
            ax.set_ylabel("Sample Count", fontsize=10)
            ax.set_xlabel("Participant ID", fontsize=10)
            for p in barplot.patches:
                val = int(p.get_height())
                ax.annotate(str(val), (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=9, color="#f1f5f9", xytext=(0, 3),
                            textcoords='offset points')
            charts["participant_distribution"] = fig_to_base64(fig)

        # 4. Landmark Coordinate Correlation Heatmap
        # Meaningful correlation analysis between hand keypoint dimensions
        landmark_features_list = []
        for s in valid_samples[:60]:
            try:
                if s.get("landmarks") and len(s["landmarks"]) == 21:
                    landmark_features_list.append(normalize_landmark_coordinates(s["landmarks"]))
                elif s.get("features") and len(s["features"]) == 63:
                    landmark_features_list.append(np.array(s["features"], dtype=np.float32))
                else:
                    media_bytes = await fetch_sample_bytes(s)
                    if media_bytes:
                        feat_dict = extract_features_from_image(media_bytes)
                        if feat_dict["hand_present"] and feat_dict["features"] is not None:
                            landmark_features_list.append(feat_dict["features"])
            except Exception:
                pass

        if len(landmark_features_list) >= 4:
            feat_matrix = np.array(landmark_features_list, dtype=np.float32)
            # Pick primary keypoint coordinates: Wrist, Thumb Tip, Index Tip, Middle Tip, Ring Tip, Pinky Tip
            keypoint_indices = [
                (0, "Wrist_X"), (1, "Wrist_Y"),
                (12, "Thumb_X"), (13, "Thumb_Y"),
                (24, "Index_X"), (25, "Index_Y"),
                (36, "Middle_X"), (37, "Middle_Y"),
                (48, "Ring_X"), (49, "Ring_Y"),
                (60, "Pinky_X"), (61, "Pinky_Y")
            ]
            sub_matrix = feat_matrix[:, [idx for idx, _ in keypoint_indices]]
            sub_cols = [name for _, name in keypoint_indices]
            sub_df = pd.DataFrame(sub_matrix, columns=sub_cols)
            corr_mat = sub_df.corr().fillna(0.0)

            fig, ax = plt.subplots(figsize=(8, 6.2))
            sns.heatmap(
                corr_mat,
                annot=True,
                fmt=".2f",
                cmap="vlag",
                center=0.0,
                vmin=-1.0,
                vmax=1.0,
                linewidths=0.5,
                linecolor="#181d24",
                cbar_kws={"label": "Pearson Correlation (r)"},
                ax=ax
            )
            ax.set_title("Landmark Spatial Coordinate Correlation Matrix", fontsize=13, fontweight="bold", pad=12, color="#38bdf8")
            charts["correlation_heatmap"] = fig_to_base64(fig)

            # 5. Landmark Feature Boxplot (Fingertip feature spread)
            fig, ax = plt.subplots(figsize=(8, 4.2))
            sns.boxplot(data=sub_df[["Thumb_Y", "Index_Y", "Middle_Y", "Ring_Y", "Pinky_Y"]], palette="mako", ax=ax)
            ax.set_title("Fingertip Normalized Height Variance (Y-Axis)", fontsize=13, fontweight="bold", pad=12, color="#38bdf8")
            ax.set_ylabel("Normalized Coordinate (relative to wrist)", fontsize=10)
            ax.set_xlabel("Fingertip Landmarks", fontsize=10)
            charts["feature_boxplot"] = fig_to_base64(fig)

        return {
            "model_id": model_id,
            "model_name": model.get("name"),
            "total_samples": total_samples,
            "valid_samples": len(valid_samples),
            "invalid_samples": len(invalid_samples),
            "images_count": images_count,
            "videos_count": videos_count,
            "static_count": static_count,
            "dynamic_count": dynamic_count,
            "participants_count": len(participant_sample_counts),
            "classes_count": len(class_names),
            "class_distribution": class_sample_counts,
            "participant_distribution": participant_sample_counts,
            "imbalance_ratio": imbalance_ratio,
            "statistical_summary": stats_summary,
            "charts": charts
        }
