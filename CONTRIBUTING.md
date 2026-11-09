# Contributing to SignBridge

Thank you for your interest in contributing to **SignBridge**! We welcome contributions to enhance Indian Sign Language (ISL) recognition accuracy, add new sign classes, improve frontend interfaces, or extend the machine learning pipeline.

---

## Code of Conduct

Please be respectful, collaborative, and constructive when interacting with community members, participants, and maintainers.

---

## Getting Started

1. **Fork the Repository:** Create a personal fork on GitHub.
2. **Clone your Fork:**
   ```bash
   git clone https://github.com/<your-username>/signbridge.git
   cd signbridge
   ```
3. **Set Up Python Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```
4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
5. **Configure Environment:**
   Copy `.env.example` to `.env` and fill in your MongoDB Atlas and Cloudinary credentials:
   ```bash
   cp .env.example .env
   ```

---

## Development Workflow

1. Create a descriptive feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Follow established architectural principles:
   - **No external heavy frontend frameworks** (keep Vanilla JS, Bootstrap 5, Jinja2).
   - **Strict Presence Gating**: Every sign recognition must reject non-hands and return `NO_HAND` if MediaPipe detection confidence is insufficient.
   - **Data Privacy**: Never commit participant personal information, video binaries, or credentials to git.
   - **Feature Representation**: Classical models (KNN, SVM, Decision Tree) must consume normalized 63-D landmark vectors (`app/ml/features.py`). CustomSignCNN operates on 224×224 RGB crops.
3. Test your changes:
   ```bash
   python test_academic_ml.py
   python test_cloud_migration.py
   ```
4. Commit with clear, conventional messages:
   ```bash
   git commit -m "feat(ml): improve SVM hyperparameter grid search"
   ```
5. Push to your fork and submit a Pull Request.

---

## Submitting Pull Requests

- Ensure all existing verification tests pass.
- Update documentation or `ACADEMIC_ML_REPORT.md` if your change modifies ML benchmarks or schemas.
- Describe the motivation, testing methodology, and screenshots (if UI changes were made) in the PR description.
