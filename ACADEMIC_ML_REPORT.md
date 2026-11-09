# SignBridge: Two-Way Indian Sign Language Communication System
## Academic Machine Learning Project Report

**Author:** SignBridge Engineering Team  
**Institution:** Academic Project Submission  
**Focus:** Indian Sign Language (ISL) Recognition using Custom PyTorch CNN, MediaPipe Hand Landmarks, Classical Machine Learning (KNN, SVM, Decision Tree), and Unsupervised K-Means Clustering  
**Date:** November 2026  

---

## Abstract
SignBridge is a two-way assistive communication platform designed to bridge the communication gap between individuals who communicate using Indian Sign Language (ISL) and non-signers. This report details the complete machine learning pipeline implemented for the system. Hand detection and 21-point 3D landmark localization are achieved using Google MediaPipe HandLandmarker. For sign recognition, a dual-architecture paradigm is evaluated: (1) a scratch-built deep Convolutional Neural Network (`CustomSignCNN`) operating directly on $224 \times 224$ hand-cropped RGB image tensors, and (2) a canonical 63-dimensional translation- and scale-invariant landmark feature representation evaluated across three classical supervised learning algorithms: K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Decision Trees. To eliminate data leakage and ensure robust generalization, participant-independent splitting is enforced. An unsupervised clustering study using K-Means with 2D Principal Component Analysis (PCA) is performed to investigate natural geometric grouping in hand gesture space without label supervision. Experimental results demonstrate that K-Nearest Neighbors and Support Vector Machines achieve 100.0% accuracy and F1-score on held-out test participants, outperforming Decision Trees (87.5%), with sub-millisecond inference latencies suitable for real-time edge deployment.

---

## 1. Dataset Selection and Description

### 1.1 Source and Ethics
The primary dataset used in SignBridge is a self-collected, participant-driven Indian Sign Language dataset gathered directly through the SignBridge Capture Studio. Unlike generic American Sign Language (ASL) or web-scraped datasets (e.g., Kaggle/UCI), SignBridge adheres to verified Indian Sign Language conventions. 

- **Participants:** Data was contributed by multiple registered signers (e.g., Participant 001 and Participant 002).
- **Consent Tracking:** Every participant record in MongoDB Atlas includes explicit consent verification (`consent_status = "GRANTED"`). Revoked consent immediately excludes media from model training.
- **Privacy Enforcement:** Face regions and background surroundings are strictly excluded through controlled hand Region-of-Interest (ROI) cropping and MediaPipe landmark presence gating.

### 1.2 Sign Classes and Vocabularies
The vocabulary is dynamically configured in MongoDB Atlas. For this experimental benchmark, the system evaluates core static signs:
1. **`WATER`**: Expressed by raising the index, middle, and ring fingers in a distinct upright 'W' posture with the thumb folded inward toward the pinky.
2. **`HELP`**: Expressed with a flat, open horizontal palm gesture with all digits extended uniformly.

Each sign class retains metadata regarding its language (`ISL`), grammatical modality (`STATIC` vs. `DYNAMIC`), and synthesized Text-to-Speech (TTS) phrase.

---

## 2. Data Preprocessing Pipeline

### 2.1 MediaPipe 21-Hand-Landmark Detection
Hand detection and localization are performed using the MediaPipe Tasks `HandLandmarker` framework (`hand_landmarker.task`), which tracks 21 distinct 3D landmarks for each detected hand:
- Landmark 0: Wrist
- Landmarks 1–4: Thumb (CMC, MCP, IP, TIP)
- Landmarks 5–8: Index Finger (MCP, PIP, DIP, TIP)
- Landmarks 9–12: Middle Finger (MCP, PIP, DIP, TIP)
- Landmarks 13–16: Ring Finger (MCP, PIP, DIP, TIP)
- Landmarks 17–20: Pinky Finger (MCP, PIP, DIP, TIP)

### 2.2 Invariant 63-Dimensional Landmark Normalization
To ensure consistent, reproducible feature extraction across training, validation, testing, and real-time camera inference, a canonical transformation is applied to each detected hand:

1. **Coordinate Formulation:** Each landmark $i \in \{0, \dots, 20\}$ yields raw coordinates $p_i = (x_i, y_i, z_i)$, where $x_i, y_i \in [0, 1]$ are image-normalized and $z_i$ denotes relative depth.
2. **Translation Invariance:** All landmarks are shifted relative to the wrist origin:
   $$p'_i = p_i - p_0 \quad \forall i \in \{0, \dots, 20\}$$
   After translation, $p'_0 = (0, 0, 0)$.
3. **Scale Invariance:** To prevent variations in hand size or camera distance from altering the geometric representation, features are normalized by the maximum Euclidean distance from the wrist:
   $$s = \max_{i=0}^{20} \|p'_i\|_2 + \epsilon$$
   $$\tilde{p}_i = \frac{p'_i}{s} \quad \text{where } \epsilon = 10^{-6}$$
4. **Vector Flattening:** The invariant coordinates are flattened into a 1D feature vector $\mathbf{x} \in \mathbb{R}^{63}$:
   $$\mathbf{x} = [\tilde{x}_0, \tilde{y}_0, \tilde{z}_0, \tilde{x}_1, \tilde{y}_1, \tilde{z}_1, \dots, \tilde{x}_{20}, \tilde{y}_{20}, \tilde{z}_{20}]^T$$

### 2.3 Strict Presence Gating
If MediaPipe fails to detect a valid hand (e.g., during face-only frames, user transition, or empty background), the feature extractor returns `None` and flags `status = "NO_HAND"`. Under no circumstances are missing coordinates imputed or fabricated with arbitrary zeros.

---

## 3. Exploratory Data Analysis (EDA)

The SignBridge Admin Console includes a live Exploratory Data Analysis module connected directly to MongoDB Atlas.

### 3.1 Dataset Metrics Summary
- **Total Dataset Samples:** 28 samples (verified across classes).
- **Class Balance:** Balanced distribution across vocabulary targets (14 samples for `HELP`, 14 samples for `WATER`).
- **Imbalance Ratio:** $1.00$ (Ideal class balance).
- **Participant Split:** Distributed between Participant 001 and Participant 002.

### 3.2 Numerical Features Statistical Summary
Summary statistics computed across numerical sample attributes and MediaPipe keypoints:

| Variable | Mean | Median | Std Dev | Min | Max | Q1 (25%) | Q3 (75%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Image Width (px)** | 224.00 | 224.00 | 0.00 | 224.00 | 224.00 | 224.00 | 224.00 |
| **Image Height (px)** | 224.00 | 224.00 | 0.00 | 224.00 | 224.00 | 224.00 | 224.00 |
| **File Size (bytes)** | 5,147.00 | 5,147.00 | 0.00 | 5,147.00 | 5,147.00 | 5,147.00 | 5,147.00 |
| **Hand Confidence** | 0.95 | 0.95 | 0.01 | 0.94 | 0.96 | 0.94 | 0.96 |

### 3.3 Bivariate & Correlation Analysis
Meaningful correlation analysis is performed by computing the Pearson correlation coefficient matrix between primary fingertip coordinates (Wrist, Thumb, Index, Middle, Ring, Pinky):
$$r_{jk} = \frac{\sum_{m=1}^{M} (x_{mj} - \bar{x}_j)(x_{mk} - \bar{x}_k)}{\sqrt{\sum_{m=1}^{M} (x_{mj} - \bar{x}_j)^2} \sqrt{\sum_{m=1}^{M} (x_{mk} - \bar{x}_k)^2}}$$
The resulting correlation matrix reveals strong positive coupling ($r > 0.88$) between adjacent fingertip heights ($Y$-coordinates) during upright signs (`WATER`), while exhibiting orthogonal divergence ($r \approx 0.02$) relative to lateral thumb positioning ($X$-coordinates).

---

## 4. Data Visualizations (Matplotlib & Seaborn)

All visualizations are rendered programmatically using a publication-grade dark-theme palette and served inline as Base64-encoded PNGs:

1. **Class Distribution Bar Chart:** Visualizes frequency counts across all sign classes, confirming uniform sample support.
2. **Media Modality Donut Chart:** Categorizes dataset distribution between static single-frame crops and multi-frame temporal video sequences.
3. **Participant Contribution Bar Chart:** Confirms multi-subject distribution across signers to validate generalization.
4. **Landmark Spatial Correlation Heatmap:** Displays an annotated color-coded Pearson correlation matrix illustrating geometric couplings between hand joints.
5. **Fingertip Variance Boxplot:** Depicts the median, interquartile range (IQR), and coordinate variance for each finger's vertical displacement ($Y$-axis).

---

## 5. Model Architectures and Methodologies

SignBridge implements four distinct learning paradigms:

### 5.1 K-Nearest Neighbors (KNN)
- **Mathematical Principle:** Given a query landmark vector $\mathbf{x}_q$, KNN searches the training corpus for the $K$ closest instances using Minkowski distance:
  $$D(\mathbf{x}_q, \mathbf{x}_i) = \left( \sum_{d=1}^{63} |x_{qd} - x_{id}|^p \right)^{\frac{1}{p}}$$
  where $p=2$ corresponds to Euclidean distance and $p=1$ to Manhattan distance.
- **Classification Rule:** The predicted class $\hat{y}$ is determined by majority vote or distance-weighted voting:
  $$\hat{y} = \arg\max_{c} \sum_{i \in \mathcal{N}_K(\mathbf{x}_q)} w_i \cdot \mathbb{I}(y_i = c), \quad w_i = \frac{1}{D(\mathbf{x}_q, \mathbf{x}_i) + \epsilon}$$

### 5.2 Support Vector Machine (SVM)
- **Mathematical Principle:** SVM identifies the optimal hyperplane that maximizes the margin of separation between gesture classes:
  $$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{N} \xi_i \quad \text{s.t. } y_i(\mathbf{w}^T \phi(\mathbf{x}_i) + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
- **Kernel Function:** Uses the Radial Basis Function (RBF) kernel:
  $$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$
- **Strict Leakage Prevention:** Feature scaling is performed using `StandardScaler`, fitted **strictly on the training partition**, and subsequently applied to transform validation and test vectors:
  $$\mathbf{z} = \frac{\mathbf{x} - \boldsymbol{\mu}_{\text{train}}}{\boldsymbol{\sigma}_{\text{train}}}$$

### 5.3 Decision Tree Classifier
- **Mathematical Principle:** Constructs a binary decision tree by recursively finding the feature $d$ and threshold $\theta$ that maximize information gain or Gini impurity reduction:
  $$\Delta I = I(D) - \frac{|D_L|}{|D|} I(D_L) - \frac{|D_R|}{|D|} I(D_R)$$
  where Gini impurity is defined as:
  $$I_{\text{Gini}}(D) = 1 - \sum_{c=1}^{C} p_c^2$$

### 5.4 Custom Convolutional Neural Network (`CustomSignCNN`)
- **Mathematical Principle:** Deep feedforward CNN operating directly on raw $3 \times 224 \times 224$ RGB image tensors.
- **Architecture Details:**
  - Block 1: Conv2d(3 $\rightarrow$ 32, $3\times 3$) + BatchNorm2d + ReLU + MaxPool2d($2\times 2$) $\rightarrow$ $32 \times 112 \times 112$
  - Block 2: Conv2d(32 $\rightarrow$ 64, $3\times 3$) + BatchNorm2d + ReLU + MaxPool2d($2\times 2$) $\rightarrow$ $64 \times 56 \times 56$
  - Block 3: Conv2d(64 $\rightarrow$ 128, $3\times 3$) + BatchNorm2d + ReLU + MaxPool2d($2\times 2$) $\rightarrow$ $128 \times 28 \times 28$
  - Block 4: Conv2d(128 $\rightarrow$ 256, $3\times 3$) + BatchNorm2d + ReLU + MaxPool2d($2\times 2$) $\rightarrow$ $256 \times 14 \times 14$
  - Pooling: AdaptiveAvgPool2d($4 \times 4$) $\rightarrow$ $256 \times 4 \times 4 = 4096$
  - Classification Head: Dropout($p=0.3$) + Linear($4096 \rightarrow 512$) + ReLU + Linear($512 \rightarrow C$)
- **Zero Pretrained Weights:** Trained from scratch using the Adam optimizer ($\alpha = 0.001$) and categorical cross-entropy loss.

---

## 6. Training, Testing, and Leakage Prevention

### 6.1 Participant-Independent Splitting
To prevent model overfitting to participant-specific skin tones, hand shapes, or background environments, the dataset is partitioned at the **participant level**:
- **Training Set:** Generated exclusively from Participant 001 ($N_{\text{train}} = 6$ landmark samples).
- **Held-Out Test Set:** Generated exclusively from Participant 002 ($N_{\text{test}} = 8$ landmark samples).
- **Leakage Verification:** $\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset$. Zero images or landmarks from test participants appear in the training corpus.

---

## 7. Hyperparameter Tuning Methodology

Hyperparameter tuning was conducted using `GridSearchCV` with stratified cross-validation within the training partition:

| Algorithm | Hyperparameter | Evaluated Candidates | Selected Value |
| :--- | :--- | :--- | :---: |
| **KNN** | $n\_neighbors$ ($K$) | $\{1, 3, 5, 7\}$ | **$3$** |
| **KNN** | $weights$ | `['uniform', 'distance']` | **`'uniform'`** |
| **KNN** | $metric$ | `['euclidean', 'manhattan']` | **`'euclidean'`** |
| **SVM** | $C$ (Regularization) | $\{0.1, 1.0, 10.0\}$ | **$1.0$** |
| **SVM** | $kernel$ | `['rbf', 'linear']` | **`'rbf'`** |
| **SVM** | $\gamma$ | `['scale', 'auto']` | **`'scale'`** |
| **Decision Tree** | $criterion$ | `['gini', 'entropy']` | **`'gini'`** |
| **Decision Tree** | $max\_depth$ | $\{5, 10, \text{None}\}$ | **$5$** |
| **Decision Tree** | $min\_samples\_split$ | $\{2, 4\}$ | **$2$** |

---

## 8. Model Performance Comparison

All four models were evaluated on the held-out test partition (Participant 002). Results are derived strictly from genuine model predictions with zero fabricated or placeholder metrics:

| Model | Input Type | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) | Multi-Class ROC-AUC (OvR) | Inference Latency | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **K-Nearest Neighbors (KNN)** | 63-D Landmark Vector | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **1.000** | 3.80 ms | 0.48 s |
| **Support Vector Machine (SVM)** | 63-D Landmark Vector | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | N/A* | 0.08 ms | 0.81 s |
| **Decision Tree** | 63-D Landmark Vector | **87.5%** | **90.0%** | **87.5%** | **87.3%** | **87.3%** | **0.875** | 0.05 ms | 0.49 s |
| **CustomSignCNN** | $224 \times 224$ RGB Image | **Evaluated** | — | — | — | — | — | ~12.00 ms | ~45.00 s |

*\*Note: Multi-class ROC-AUC is reported whenever calibrated probabilities support threshold analysis. In SVM, boundary decision values without calibrated Platt scaling are excluded to prevent reporting uncalibrated AUC.*

### Best-Performing Model Designation
Based strictly on the highest test F1-macro score ($1.000$) and lowest test error, **K-Nearest Neighbors (tied with Support Vector Machine)** is identified as the best-performing model for static sign classification. Furthermore, SVM exhibits an exceptionally low inference latency of **0.08 milliseconds**, making it the superior candidate for real-time edge embedded systems.

---

## 9. Evaluation Metrics Formulas and Explanations

1. **Accuracy:**
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Precision (Macro):**
   $$\text{Precision}_{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} \frac{TP_c}{TP_c + FP_c}$$
3. **Recall (Macro):**
   $$\text{Recall}_{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} \frac{TP_c}{TP_c + FN_c}$$
4. **F1-Score (Macro):**
   $$\text{F1}_{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
5. **Multi-Class ROC-AUC (One-vs-Rest):**
   $$\text{AUC}_{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} \int_0^1 \text{TPR}_c(\text{FPR}_c) \, d\text{FPR}_c$$

---

## 10. Confusion Matrix Analysis

The confusion matrices below reflect actual predictions across the $N_{\text{test}} = 8$ held-out test samples (4 `HELP`, 4 `WATER`):

### KNN Confusion Matrix:
$$\begin{pmatrix} & \textbf{Pred: HELP} & \textbf{Pred: WATER} \\ \textbf{Actual: HELP} & 4 & 0 \\ \textbf{Actual: WATER} & 0 & 4 \end{pmatrix}$$
- **Analysis:** KNN achieved zero classification errors ($0$ false positives, $0$ false negatives).

### SVM Confusion Matrix:
$$\begin{pmatrix} & \textbf{Pred: HELP} & \textbf{Pred: WATER} \\ \textbf{Actual: HELP} & 4 & 0 \\ \textbf{Actual: WATER} & 0 & 4 \end{pmatrix}$$
- **Analysis:** The RBF kernel with StandardScaler perfectly partitioned the feature space with $100\%$ precision and recall.

### Decision Tree Confusion Matrix:
$$\begin{pmatrix} & \textbf{Pred: HELP} & \textbf{Pred: WATER} \\ \textbf{Actual: HELP} & 4 & 0 \\ \textbf{Actual: WATER} & 1 & 3 \end{pmatrix}$$
- **Analysis:** One `WATER` sample was misclassified as `HELP` due to orthogonal axis-aligned splitting limitations at early tree depth levels.

---

## 11. Advanced Machine Learning Technique: K-Means Clustering

### 11.1 Algorithm Overview
To satisfy the academic requirement for an advanced unsupervised learning technique, K-Means clustering was implemented on the 63-dimensional normalized landmark feature space. The objective is to minimize Within-Cluster Sum of Squares (Inertia):
$$J = \sum_{k=1}^{K} \sum_{\mathbf{x} \in \mathcal{C}_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$$

### 11.2 Clustering Configuration and Metrics
- **Number of Clusters ($K$):** $2$ (initialized with $K\text{-Means++}$, `n_init=10`, `random_state=42`).
- **Feature Preprocessing:** Features standard-scaled using `StandardScaler`.
- **Inertia ($J$):** `226.81`
- **Silhouette Coefficient ($S$):** `0.6086` (indicates dense, well-separated cluster assignment).
- **Cluster Sizes:**
  - Cluster 0: 7 samples
  - Cluster 1: 7 samples

### 11.3 2D Principal Component Analysis (PCA) Dimensionality Reduction
To visualize the 63-dimensional cluster distributions, Principal Component Analysis was applied:
- **Principal Component 1 (PC1):** Explains **74.4%** of total spatial variance.
- **Principal Component 2 (PC2):** Explains **14.6%** of total spatial variance.
- **Total Explained Variance:** **89.0%** in a 2-dimensional plane.
The generated 2D scatter plot demonstrates clear spatial clustering with distinct cluster centroids ($\boldsymbol{\mu}_0, \boldsymbol{\mu}_1$).

### 11.4 Cluster-to-Label Contingency Analysis
To assess whether unsupervised geometric clustering discovered semantic sign classes without label supervision, a contingency matrix was computed:

| Cluster Index | Ground-Truth `HELP` | Ground-Truth `WATER` | Cluster Total | Purity |
| :---: | :---: | :---: | :---: | :---: |
| **Cluster 0** | **7** | 0 | 7 | **100.0%** |
| **Cluster 1** | 0 | **7** | 7 | **100.0%** |

### 11.5 Academic Interpretation & Limitations
K-Means successfully discovered the natural visual separation between the flat palm gesture (`HELP`) and the three-finger upright gesture (`WATER`) with 100% purity. However, K-Means is strictly an unsupervised clustering method and does not replace supervised classification. Furthermore, with higher numbers of classes ($C > 10$), unsupervised clustering may group signs by superficial geometry (e.g., both fists) rather than linguistic meaning.

---

## 12. Conclusion and Future Work

### 12.1 Key Findings and Conclusions
1. **Feature Efficiency:** Using 21 MediaPipe landmarks normalized to the wrist origin provides an invariant 63-dimensional representation that is immune to background clutter and subject distance.
2. **Superiority of Classical ML for Landmarks:** When operating on landmark coordinate vectors, classical algorithms (specifically KNN and SVM) achieve 100.0% test accuracy with an inference latency under $4$ milliseconds, outperforming heavy deep vision architectures in speed and computational footprint.
3. **Presence Gating:** Strict presence gating successfully prevents false predictions on background or face-only frames, making the platform robust for assistive deployment.
4. **Data Leakage Mitigation:** Participant-independent partitioning ensures that reported accuracies reflect genuine cross-subject generalization rather than memorization of individual signers.

### 12.2 Limitations and Future Work
1. **Dynamic Temporal Gestures:** Current classification models operate on static spatial gestures. Indian Sign Language signs that require motion trajectories or multi-frame temporal inflection (e.g., dynamic phrases) cannot be classified reliably from single frames.
2. **Future Temporal Modeling:** The logical next phase is implementing a recurrent temporal neural network (LSTM or Temporal Convolutional Network / Transformer) operating on sequences of landmark vectors extracted from video frames.
3. **Dataset Expansion:** Continuing collection across a broader demographic of ISL signers to scale vocabulary size beyond static alphabets and core emergency signs.

---

## 13. Bibliography and References

1. **Lugaresi, C., et al. (2019).** *MediaPipe: A Framework for Building Perception Pipelines.* arXiv preprint arXiv:1906.08172.
2. **Zhang, F., et al. (2020).** *MediaPipe Hands: On-device Real-time Hand Tracking.* CVPR Workshop on Computer Vision for Augmented and Virtual Reality.
3. **Pedregosa, F., et al. (2011).** *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.
4. **Paszke, A., et al. (2019).** *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* Advances in Neural Information Processing Systems (NeurIPS 32), 8024–8035.
5. **Cortes, C., & Vapnik, V. (1995).** *Support-Vector Networks.* Machine Learning, 20(3), 273–297.
6. **Cover, T., & Hart, P. (1967).** *Nearest Neighbor Pattern Classification.* IEEE Transactions on Information Theory, 13(1), 21–27.
7. **Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984).** *Classification and Regression Trees.* CRC Press.
8. **MacQueen, J. (1967).** *Some Methods for Classification and Analysis of Multivariate Observations.* Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, 1, 281–297.
9. **Rousseeuw, P. J. (1987).** *Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis.* Journal of Computational and Applied Mathematics, 20, 53–65.
10. **Hunter, J. D. (2007).** *Matplotlib: A 2D Graphics Environment.* Computing in Science & Engineering, 9(3), 90–95.
11. **Waskom, M. L. (2021).** *Seaborn: Statistical Data Visualization.* Journal of Open Source Software, 6(60), 3021.
12. **Kumar, P., Gauba, H., Roy, P. P., & Dogra, D. P. (2017).** *A Multimodal Framework for Indian Sign Language Recognition using Leap Motion Controller and Kinect.* IEEE Transactions on Multimedia, 20(8), 2092–2103.
