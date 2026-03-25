# 🏋️ PoseAI Trainer

A web application that analyzes squat form using a custom-trained deep learning model and biomechanical pose metrics.

---

## How It Works

1. **Train** – Upload a Google Drive folder containing labeled squat videos (`good/` and `bad/` subfolders). The app trains a 3D CNN video classifier on them.
2. **Analyze** – Upload any squat video and get an instant classification (Good / Needs Improvement) with two personalized coaching tips.

The model only trains once. After training, the app switches automatically to analysis mode.

---

## Project Structure

```
PoseAITraining_web/
├── app.py          # Streamlit UI
├── main.py         # Model architecture, training pipeline
├── backend.py      # Video analysis, pose metrics (MediaPipe), feedback
├── outputs/        # Saved model files (auto-created after training)
├── .streamlit/     # Streamlit config
├── .env            # Environment variables (not committed)
└── requirements.txt
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

### 3. Train the model

Paste a Google Drive folder link that follows this structure:

```
root/
  good/
    video1.mp4
    ...
  bad/
    video2.mp4
    ...
```

Click **Train model** and wait. Training runs once — the model is saved to `outputs/pose_model.pt`.

### 4. Analyze a video

Upload a squat video (MP4, MOV, AVI, MKV, WEBM) and click **Analyze video**.

---

## Feedback System

Coaching tips are generated from two sources:

| Source | When used |
|--------|-----------|
| Biomechanical rules (MediaPipe pose metrics) | Always |
| OpenAI GPT-4o-mini | Only if `OPENAI_API_KEY` is set in `.env` |

**Metrics measured:** squat depth, torso lean angle, knee tracking, left/right symmetry, movement stability.

---

## Environment Variables (`.env`)

```env
OPENAI_API_KEY=sk-...   # Optional – enables AI-generated feedback
EPOCHS=4                # Training epochs (default: 4)
BATCH_SIZE=2            # Training batch size (default: 2)
LEARNING_RATE=0.0001    # Learning rate (default: 1e-4)
```

---

## Requirements

- Python 3.9+
- PyTorch
- OpenCV
- MediaPipe
- Streamlit
- gdown

---

## Notes

- MediaPipe may fail when the `.venv` path contains non-ASCII characters (e.g. Hebrew folder names). In that case, pose metrics are skipped and classification still works normally.
- The model file (`outputs/`) is excluded from version control.
