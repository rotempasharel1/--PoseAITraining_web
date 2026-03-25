from __future__ import annotations

import csv
import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

# ============================================================
# Paths / env
# ============================================================
ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "pose_model.pt"
MODEL_META_PATH = OUTPUT_DIR / "model_meta.json"
TRAINING_HISTORY_PATH = OUTPUT_DIR / "training_history.json"
TRAINING_SUMMARY_PATH = OUTPUT_DIR / "training_summary.json"
TRAINING_CSV_PATH = OUTPUT_DIR / "training_accuracy.csv"
TRAINING_PLOT_PATH = OUTPUT_DIR / "training_accuracy.png"
SUMMARY_PLOT_PATH = OUTPUT_DIR / "model_summary.png"
POSE_GRU_PATH = OUTPUT_DIR / "pose_gru_model.pt"
POSE_GRU_META_PATH = OUTPUT_DIR / "pose_gru_meta.json"

# Landmark indices for side-view squat analysis (MediaPipe Pose)
# Left side: shoulder, hip, knee, ankle, heel, foot_index
_SIDE_LANDMARKS = [11, 23, 25, 27, 29, 31]
_POSE_FEATURES = len(_SIDE_LANDMARKS) * 2  # x, y per landmark (normalized)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================
# Config
# ============================================================
def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    return float(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


@dataclass
class CFG:
    seed: int = _env_int("SEED", 42)
    val_ratio: float = _env_float("VAL_RATIO", 0.25)
    clip_len: int = _env_int("CLIP_LEN", 16)
    clip_stride: int = _env_int("CLIP_STRIDE", 2)
    image_size: int = _env_int("IMAGE_SIZE", 112)
    batch_size: int = _env_int("BATCH_SIZE", 2)
    num_workers: int = _env_int("NUM_WORKERS", 0)
    epochs: int = _env_int("EPOCHS", 4)
    learning_rate: float = _env_float("LEARNING_RATE", 1e-4)
    weight_decay: float = _env_float("WEIGHT_DECAY", 1e-4)
    early_stop_patience: int = _env_int("EARLY_STOP_PATIENCE", 3)
    train_clips_per_video: int = _env_int("TRAIN_CLIPS_PER_VIDEO", 8)
    val_clips_per_video: int = _env_int("VAL_CLIPS_PER_VIDEO", 2)
    max_eval_clips: int = _env_int("MAX_EVAL_CLIPS", 6)
    use_pretrained_backbone: bool = _env_bool("USE_PRETRAINED_BACKBONE", False)


cfg = CFG()
CLASS_NAMES = ["bad", "good"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(cfg.seed)


# ============================================================
# OpenAI feedback (optional)
# ============================================================
_OPENAI_CLIENT: Optional[Any] = None


def _get_openai_client() -> Optional[Any]:
    global _OPENAI_CLIENT
    if OpenAI is None:
        return None
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


# ============================================================
# Video discovery / loading
# ============================================================
def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def infer_label_from_path(path: Path) -> Optional[str]:
    lowered_parts = [part.lower() for part in path.parts]
    if "good" in lowered_parts:
        return "good"
    if "bad" in lowered_parts:
        return "bad"
    return None


def collect_labeled_videos(data_root: Path | str) -> List[Dict[str, Any]]:
    root = Path(data_root)
    if not root.exists():
        return []

    entries: List[Dict[str, Any]] = []
    for path in root.rglob("*"):
        if not is_video_file(path):
            continue
        label = infer_label_from_path(path)
        if label is None:
            continue
        entries.append(
            {
                "path": str(path.resolve()),
                "label": label,
                "video_name": path.name,
                "relative_path": str(path.relative_to(root)),
            }
        )

    entries.sort(key=lambda item: (item["label"], item["relative_path"]))
    return entries


def summarize_video_entries(entries: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"good": 0, "bad": 0}
    for item in entries:
        label = str(item["label"]).lower()
        if label in counts:
            counts[label] += 1
    counts["total"] = counts["good"] + counts["bad"]
    return counts


def get_video_frame_count(video_path: str | Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    return max(total, 0)


def _sample_indices(total_frames: int, clip_len: int, stride: int, *, center: bool, start_idx: Optional[int]) -> List[int]:
    if total_frames <= 0:
        raise RuntimeError("Could not read the video or the video has zero frames.")

    if total_frames == 1:
        return [0] * clip_len

    span = 1 + (clip_len - 1) * stride
    if total_frames >= span:
        if start_idx is not None:
            start = max(0, min(start_idx, total_frames - span))
        elif center:
            start = (total_frames - span) // 2
        else:
            start = random.randint(0, total_frames - span)
        return [start + i * stride for i in range(clip_len)]

    positions = np.linspace(0, total_frames - 1, num=clip_len)
    return [int(round(float(pos))) for pos in positions]


def _resize_rgb(frame_rgb: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)


def load_video_clip(
    video_path: str | Path,
    clip_len: int,
    stride: int,
    image_size: int,
    *,
    center: bool = False,
    start_idx: Optional[int] = None,
) -> torch.Tensor:
    """Load a whole-video clip tensor in C x T x H x W format."""
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if total_frames <= 0:
        cap.release()
        # Fallback: sequential read to count frames
        cap = cv2.VideoCapture(video_path)
        fallback_frames: List[np.ndarray] = []
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            fallback_frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        if not fallback_frames:
            raise RuntimeError(f"Could not decode video: {video_path}")
        indices = _sample_indices(len(fallback_frames), clip_len, stride, center=center, start_idx=start_idx)
        frames = [_resize_rgb(fallback_frames[idx], image_size) for idx in indices]
    else:
        indices = _sample_indices(total_frames, clip_len, stride, center=center, start_idx=start_idx)
        wanted = set(indices)
        frames_by_idx: Dict[int, np.ndarray] = {}
        current = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if current in wanted:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames_by_idx[current] = _resize_rgb(frame_rgb, image_size)
                if len(frames_by_idx) == len(wanted):
                    break
            current += 1
        cap.release()

        if not frames_by_idx:
            raise RuntimeError(f"Could not sample frames from video: {video_path}")

        last_frame = frames_by_idx[min(frames_by_idx.keys())]
        frames = []
        for idx in indices:
            if idx in frames_by_idx:
                last_frame = frames_by_idx[idx]
            frames.append(last_frame)

    clip = np.stack(frames, axis=0).astype(np.float32) / 255.0  # T,H,W,C
    clip = (clip - _IMAGENET_MEAN) / _IMAGENET_STD
    clip = np.transpose(clip, (3, 0, 1, 2))  # C,T,H,W
    return torch.from_numpy(clip)


# ============================================================
# Dataset
# ============================================================
class VideoClipDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[Dict[str, Any]],
        *,
        clip_len: int,
        stride: int,
        image_size: int,
        train: bool,
        clips_per_video: int,
    ) -> None:
        self.entries = list(entries)
        self.clip_len = clip_len
        self.stride = stride
        self.image_size = image_size
        self.train = train
        self.clips_per_video = max(1, clips_per_video)

    def __len__(self) -> int:
        return len(self.entries) * self.clips_per_video

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        video_index = index % len(self.entries)
        item = self.entries[video_index]
        clip = load_video_clip(
            item["path"],
            clip_len=self.clip_len,
            stride=self.stride,
            image_size=self.image_size,
            center=not self.train,
            start_idx=None,
        )
        if self.train and random.random() < 0.5:
            clip = torch.flip(clip, dims=[3])
        label = CLASS_TO_IDX[str(item["label"]).lower()]
        return clip, label


# ============================================================
# Model
# ============================================================
class VideoClassifier3D(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model(pretrained: bool = False, device: str = DEVICE) -> nn.Module:
    # pretrained is kept for API compatibility with the old project structure.
    model = VideoClassifier3D(num_classes=len(CLASS_NAMES))
    model.to(device)
    return model


# ============================================================
# Split / train helpers
# ============================================================
def _split_entries(entries: Sequence[Dict[str, Any]], val_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, Any]]] = {"good": [], "bad": []}
    for item in entries:
        by_label[str(item["label"]).lower()].append(dict(item))

    train_entries: List[Dict[str, Any]] = []
    val_entries: List[Dict[str, Any]] = []

    for label, items in by_label.items():
        if not items:
            continue
        rng.shuffle(items)
        if len(items) == 1:
            train_entries.extend(items)
            continue
        n_val = max(1, int(round(len(items) * val_ratio)))
        n_val = min(n_val, len(items) - 1)
        val_entries.extend(items[:n_val])
        train_entries.extend(items[n_val:])

    rng.shuffle(train_entries)
    rng.shuffle(val_entries)
    return train_entries, val_entries


def _make_loader(dataset: Dataset, labels: Sequence[int], batch_size: int, num_workers: int, train: bool) -> DataLoader:
    if train and labels:
        class_counts = np.bincount(np.array(labels), minlength=len(CLASS_NAMES))
        class_weights = np.array([1.0 / max(1, count) for count in class_counts], dtype=np.float32)
        sample_weights = np.array([class_weights[label] for label in labels], dtype=np.float32)
        sample_weights = np.repeat(sample_weights, getattr(dataset, "clips_per_video", 1))
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    if len(loader) == 0:
        return {"loss": 0.0, "accuracy": 0.0}

    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for clips, labels in loader:
        clips = clips.to(device)
        labels = labels.to(device)
        logits = model(clips)
        loss = criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((preds == labels).sum().item())
        total_samples += int(labels.size(0))

    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": total_correct / max(1, total_samples),
    }


@torch.no_grad()
def _collect_predictions(model: nn.Module, loader: DataLoader, device: str) -> Tuple[List[int], List[int]]:
    """Returns (all_true_labels, all_pred_labels) for every sample in loader."""
    all_true: List[int] = []
    all_pred: List[int] = []
    model.eval()
    for clips, labels in loader:
        clips = clips.to(device)
        logits = model(clips)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        all_pred.extend(preds)
        all_true.extend(labels.tolist())
    return all_true, all_pred


def _save_summary_plot(
    all_true: List[int],
    all_pred: List[int],
    class_names: List[str],
    best_metric: float,
    history: List[Dict[str, float]],
    save_path: Path,
) -> None:
    """Saves a 3-panel summary: confusion matrix, correct vs wrong donut, accuracy curve."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        n = len(class_names)
        cm = [[0] * n for _ in range(n)]
        for t, p in zip(all_true, all_pred):
            cm[t][p] += 1

        total = sum(sum(row) for row in cm)
        correct = sum(cm[i][i] for i in range(n))
        wrong = total - correct

        fig = plt.figure(figsize=(14, 5), facecolor="#1a1a2e")
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

        # ── Panel 1: Confusion matrix ──────────────────────────────
        ax1 = fig.add_subplot(gs[0])
        cm_arr = np.array(cm, dtype=float)
        im = ax1.imshow(cm_arr, cmap="Blues", vmin=0)
        ax1.set_xticks(range(n))
        ax1.set_yticks(range(n))
        ax1.set_xticklabels([c.capitalize() for c in class_names], color="white", fontsize=11)
        ax1.set_yticklabels([c.capitalize() for c in class_names], color="white", fontsize=11)
        ax1.set_xlabel("Predicted", color="#aaaacc", fontsize=10)
        ax1.set_ylabel("Actual", color="#aaaacc", fontsize=10)
        ax1.set_title("Confusion Matrix\n(validation set)", color="white", fontsize=12, fontweight="bold")
        for i in range(n):
            for j in range(n):
                val = int(cm[i][j])
                color = "white" if cm_arr[i, j] > cm_arr.max() / 2 else "black"
                ax1.text(j, i, str(val), ha="center", va="center", color=color, fontsize=14, fontweight="bold")
        ax1.tick_params(colors="white")
        for spine in ax1.spines.values():
            spine.set_edgecolor("#444466")

        # ── Panel 2: Donut correct vs wrong ────────────────────────
        ax2 = fig.add_subplot(gs[1])
        sizes = [correct, wrong] if total > 0 else [1, 0]
        colors = ["#4CAF50", "#F44336"]
        wedges, _ = ax2.pie(
            sizes, colors=colors, startangle=90,
            wedgeprops=dict(width=0.5, edgecolor="#1a1a2e", linewidth=2),
        )
        pct = correct / max(total, 1) * 100
        ax2.text(0, 0, f"{pct:.0f}%", ha="center", va="center",
                 color="white", fontsize=22, fontweight="bold")
        ax2.set_title("Val Accuracy", color="white", fontsize=12, fontweight="bold")
        ax2.legend(
            [f"Correct ({correct})", f"Wrong ({wrong})"],
            loc="lower center", bbox_to_anchor=(0.5, -0.18),
            fontsize=9, frameon=False, labelcolor="white",
        )

        # ── Panel 3: Accuracy curve ────────────────────────────────
        ax3 = fig.add_subplot(gs[2])
        ax3.set_facecolor("#12122a")
        for spine in ax3.spines.values():
            spine.set_edgecolor("#444466")
        epochs_x = [int(r["epoch"]) for r in history]
        train_acc = [float(r["train_accuracy"]) * 100 for r in history]
        val_acc   = [float(r["val_accuracy"])   * 100 for r in history]
        ax3.plot(epochs_x, train_acc, marker="o", color="#4CAF50", label="Train", linewidth=2)
        ax3.plot(epochs_x, val_acc,   marker="s", color="#2196F3", label="Val",   linewidth=2, linestyle="--")
        ax3.set_ylim(0, 105)
        ax3.set_xlabel("Epoch", color="#aaaacc", fontsize=10)
        ax3.set_ylabel("Accuracy (%)", color="#aaaacc", fontsize=10)
        ax3.set_title("Accuracy per Epoch", color="white", fontsize=12, fontweight="bold")
        ax3.tick_params(colors="white")
        ax3.legend(fontsize=9, frameon=False, labelcolor="white")
        ax3.grid(True, alpha=0.2, color="#555577")

        fig.suptitle(
            f"Model Summary  |  Best val accuracy: {best_metric * 100:.1f}%",
            color="white", fontsize=13, fontweight="bold", y=1.02,
        )
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    except Exception:
        pass  # matplotlib not available — skip


# ============================================================
# Model readiness
# ============================================================
def is_model_ready(model_path: Path | str = MODEL_PATH) -> bool:
    model_path = Path(model_path)
    if not model_path.exists() or not MODEL_META_PATH.exists():
        return False
    try:
        meta = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    return meta.get("checkpoint_format") == "poseai_video_v1"


# ============================================================
# Feedback
# ============================================================
def _fallback_feedback(pred_label: str, confidence: float, metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    metrics = metrics or {}

    keep_points: List[str] = []
    improve_points: List[str] = []

    max_depth_ratio = metrics.get("max_depth_ratio")
    mean_torso_lean = metrics.get("mean_torso_lean_deg")
    knee_alignment = metrics.get("mean_knee_tracking_error")
    symmetry = metrics.get("mean_left_right_depth_diff")
    stability = metrics.get("torso_lean_std_deg")

    if max_depth_ratio is not None and max_depth_ratio >= 0.05:
        keep_points.append("Nice squat depth through the bottom position.")
    if mean_torso_lean is not None and mean_torso_lean <= 30:
        keep_points.append("Good chest position with controlled torso angle.")
    if knee_alignment is not None and knee_alignment <= 0.18:
        keep_points.append("Good knee tracking over the feet.")
    if symmetry is not None and symmetry <= 0.14:
        keep_points.append("Both sides move with good symmetry.")
    if stability is not None and stability <= 10:
        keep_points.append("Your movement looks steady across the full repetition.")

    if max_depth_ratio is not None and max_depth_ratio < 0.0:
        improve_points.append("Go slightly deeper so the hips reach at least knee level.")
    if mean_torso_lean is not None and mean_torso_lean > 35:
        improve_points.append("Keep your chest a bit prouder and reduce forward torso lean.")
    if knee_alignment is not None and knee_alignment > 0.22:
        improve_points.append("Keep your knees tracking more in line with your toes.")
    if symmetry is not None and symmetry > 0.18:
        improve_points.append("Try to keep both sides descending and rising more evenly.")
    if stability is not None and stability > 12:
        improve_points.append("Keep the descent and ascent steadier through the full video.")

    if not keep_points:
        keep_points.append(
            "Good overall control through parts of the squat cycle." if pred_label == "good" else "You still show moments of useful control to build on."
        )

    if not improve_points:
        improve_points.append(
            "Keep building consistency across the full squat from top to bottom."
            if pred_label == "good"
            else "Focus on a steadier, more aligned squat pattern across the full repetition."
        )

    primary_keep_tip = keep_points[0]
    primary_improve_tip = improve_points[0]

    return {
        "llm_keep": " ; ".join(keep_points[:2]),
        "llm_improve": " ; ".join(improve_points[:2]),
        "keep_points": keep_points,
        "improve_points": improve_points,
        "primary_keep_tip": primary_keep_tip,
        "primary_improve_tip": primary_improve_tip,
        "confidence_note": f"Model confidence: {confidence:.0%}",
    }


def llm_feedback_for_row(
    true_label: str,
    pred_label: str,
    confidence: float,
    correct: bool,
    metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    fallback = _fallback_feedback(pred_label=pred_label, confidence=confidence, metrics=metrics)
    client = _get_openai_client()
    if client is None:
        return fallback

    try:
        system_msg = (
            "You are a helpful squat coach. The classification was produced from a whole-video spatiotemporal model, not per-frame voting. "
            "Return concise coaching feedback in English only."
        )
        metrics_text = json.dumps(metrics or {}, ensure_ascii=False)
        user_msg = f"""
Context:
- true_label: {true_label}
- pred_label: {pred_label}
- confidence: {confidence:.4f}
- correct: {correct}
- video_metrics: {metrics_text}

Task:
Return strict JSON with these keys only:
- llm_keep: 2 short points separated by ' ; '
- llm_improve: 2 short points separated by ' ; '
"""

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "llm_keep": {"type": "string"},
                "llm_improve": {"type": "string"},
            },
            "required": ["llm_keep", "llm_improve"],
        }

        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "squat_feedback",
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        data = json.loads(response.output_text)
        keep_points = [part.strip() for part in data["llm_keep"].split(";") if part.strip()]
        improve_points = [part.strip() for part in data["llm_improve"].split(";") if part.strip()]
        if not keep_points:
            keep_points = fallback["keep_points"]
        if not improve_points:
            improve_points = fallback["improve_points"]
        return {
            "llm_keep": data["llm_keep"],
            "llm_improve": data["llm_improve"],
            "keep_points": keep_points,
            "improve_points": improve_points,
            "primary_keep_tip": keep_points[0],
            "primary_improve_tip": improve_points[0],
            "confidence_note": fallback["confidence_note"],
        }
    except Exception:
        return fallback


# ============================================================
# Pose-sequence model (side-view squat GRU classifier)
# ============================================================

def _extract_pose_seq(
    video_path: str | Path,
    n_frames: int = 32,
    landmark_indices: List[int] = _SIDE_LANDMARKS,
) -> Optional[np.ndarray]:
    """Extract normalized landmark sequence from a side-view squat video.

    Returns ndarray of shape (n_frames, len(landmark_indices)*2) or None if
    MediaPipe is unavailable or no pose is detected in the video.
    """
    try:
        import mediapipe as mp  # lazy import — may fail on unicode paths
        pose_sol = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # fastest
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
    except Exception:
        return None

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Sample n_frames evenly
    if total > 0:
        positions = np.linspace(0, total - 1, num=n_frames)
        wanted = {int(round(float(p))) for p in positions}
    else:
        wanted = None

    frames_lm: Dict[int, List[float]] = {}
    current = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if wanted is None or current in wanted:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = pose_sol.process(frame_rgb)
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                # Normalize: origin = left hip (idx 23), scale = hip-to-shoulder dist
                hip_x = float(lm[23].x)
                hip_y = float(lm[23].y)
                shoulder_y = float(lm[11].y)
                scale = abs(hip_y - shoulder_y) + 1e-6
                coords: List[float] = []
                for idx in landmark_indices:
                    coords.append((float(lm[idx].x) - hip_x) / scale)
                    coords.append((float(lm[idx].y) - hip_y) / scale)
                frames_lm[current] = coords
        current += 1

    cap.release()
    try:
        pose_sol.close()
    except Exception:
        pass

    if not frames_lm:
        return None

    # Build ordered sequence, carry last valid frame forward
    ordered_keys = sorted(frames_lm.keys())
    seq: List[List[float]] = []
    last = frames_lm[ordered_keys[0]]
    wanted_sorted = sorted(wanted) if wanted else list(range(n_frames))
    for idx in wanted_sorted:
        # find closest detected frame
        best = min(frames_lm.keys(), key=lambda k: abs(k - idx))
        if abs(best - idx) < 10:
            last = frames_lm[best]
        seq.append(last)

    # Pad/truncate to exactly n_frames
    while len(seq) < n_frames:
        seq.append(seq[-1])
    seq = seq[:n_frames]

    return np.array(seq, dtype=np.float32)


class PoseSequenceDataset(Dataset):
    """Dataset that feeds normalized pose-landmark sequences to the GRU model."""

    def __init__(
        self,
        entries: Sequence[Dict[str, Any]],
        n_frames: int = 32,
        augment: bool = False,
    ) -> None:
        self.entries = list(entries)
        self.n_frames = n_frames
        self.augment = augment
        self._cache: Dict[int, Optional[np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx not in self._cache:
            self._cache[idx] = _extract_pose_seq(
                self.entries[idx]["path"], self.n_frames
            )
        seq = self._cache[idx]

        if seq is None:
            seq = np.zeros((self.n_frames, _POSE_FEATURES), dtype=np.float32)
        else:
            seq = seq.copy()

        if self.augment:
            # Horizontal flip: negate x-coords (even columns)
            if random.random() < 0.5:
                seq[:, 0::2] = -seq[:, 0::2]
            # Small landmark jitter (simulates measurement noise)
            seq += np.random.normal(0, 0.015, seq.shape).astype(np.float32)
            # Temporal shift ±2 frames
            shift = random.randint(-2, 2)
            if shift > 0:
                seq = np.concatenate([seq[shift:], np.tile(seq[-1:], (shift, 1))])
            elif shift < 0:
                seq = np.concatenate([np.tile(seq[:1], (-shift, 1)), seq[:shift]])

        label = CLASS_TO_IDX[str(self.entries[idx]["label"]).lower()]
        return torch.from_numpy(seq), label


class PoseGRUClassifier(nn.Module):
    """Lightweight GRU that classifies squat quality from landmark sequences."""

    def __init__(
        self,
        input_size: int = _POSE_FEATURES,
        hidden: int = 64,
        layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden, layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, F)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])  # use last timestep


def is_pose_model_ready(model_path: Path | str = POSE_GRU_PATH) -> bool:
    model_path = Path(model_path)
    if not model_path.exists() or not POSE_GRU_META_PATH.exists():
        return False
    try:
        meta = json.loads(POSE_GRU_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    return meta.get("checkpoint_format") == "poseai_gru_v1"


def run_pose_training_pipeline(data_root: Optional[str | Path] = None) -> Tuple[Path, float]:
    """Train the pose-landmark GRU classifier. Returns (model_path, best_val_accuracy)."""
    data_root = Path(data_root) if data_root is not None else ROOT / "temp_gdrive"
    entries = collect_labeled_videos(data_root)
    counts = summarize_video_entries(entries)

    if counts["good"] == 0:
        raise RuntimeError("No GOOD videos found.")
    if counts["bad"] == 0:
        raise RuntimeError("No BAD videos found.")
    if counts["total"] < 2:
        raise RuntimeError("Need at least 2 videos.")

    train_entries, val_entries = _split_entries(entries, cfg.val_ratio, cfg.seed)
    if not train_entries:
        raise RuntimeError("Training split is empty.")

    train_ds = PoseSequenceDataset(train_entries, augment=True)
    val_ds = PoseSequenceDataset(val_entries, augment=False)

    # Pre-extract all sequences (cache) — fail early if MediaPipe broken
    sample_seq, _ = train_ds[0]
    if sample_seq.abs().sum().item() == 0.0:
        raise RuntimeError(
            "MediaPipe could not extract pose landmarks from any training video. "
            "The pose model requires MediaPipe to work correctly."
        )

    train_labels = [CLASS_TO_IDX[e["label"]] for e in train_entries]
    batch = min(cfg.batch_size, 8)
    train_loader = _make_loader(train_ds, train_labels, batch, cfg.num_workers, train=True)
    val_loader = _make_loader(val_ds, [], batch, cfg.num_workers, train=False)

    model = PoseGRUClassifier(num_classes=len(CLASS_NAMES))
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-5)

    best_metric = -float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        run_loss, run_correct, run_samples = 0.0, 0, 0
        for seqs, labels in train_loader:
            seqs = seqs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(seqs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            preds = torch.argmax(logits.detach(), dim=1)
            run_loss += float(loss.item()) * labels.size(0)
            run_correct += int((preds == labels).sum().item())
            run_samples += int(labels.size(0))
        scheduler.step()

        train_m = {"loss": run_loss / max(1, run_samples), "accuracy": run_correct / max(1, run_samples)}
        val_m = _evaluate(model, val_loader, DEVICE) if val_entries else {"loss": 0.0, "accuracy": 0.0}
        monitor = val_m["accuracy"] if val_entries else train_m["accuracy"]

        if monitor > best_metric:
            best_metric = monitor
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        history.append({
            "epoch": float(epoch),
            "train_loss": float(train_m["loss"]),
            "train_accuracy": float(train_m["accuracy"]),
            "val_loss": float(val_m["loss"]),
            "val_accuracy": float(val_m["accuracy"]),
        })

        if patience >= cfg.early_stop_patience:
            break

    if best_state is None:
        raise RuntimeError("Pose GRU training did not produce a valid model.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "checkpoint_format": "poseai_gru_v1",
        "model_name": "pose_gru",
        "class_names": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "n_frames": 32,
        "pose_features": _POSE_FEATURES,
        "side_landmarks": _SIDE_LANDMARKS,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "train_videos": len(train_entries),
        "val_videos": len(val_entries),
        "video_counts": counts,
        "history": history,
        "model_state": best_state,
        "best_accuracy": best_metric,
    }
    torch.save(checkpoint, POSE_GRU_PATH)

    meta = {k: v for k, v in checkpoint.items() if k != "model_state"}
    POSE_GRU_META_PATH.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    return POSE_GRU_PATH, best_metric


# ============================================================
# Training entry point
# ============================================================
def run_training_pipeline(data_root: Optional[str | Path] = None) -> Path:
    data_root = Path(data_root) if data_root is not None else ROOT / "temp_gdrive"
    entries = collect_labeled_videos(data_root)
    counts = summarize_video_entries(entries)

    if counts["good"] == 0:
        raise RuntimeError("No GOOD videos were found. Place videos under a folder named 'good'.")
    if counts["bad"] == 0:
        raise RuntimeError("No BAD videos were found. Place videos under a folder named 'bad'.")
    if counts["total"] < 2:
        raise RuntimeError("At least 2 videos are required to train a video model.")

    train_entries, val_entries = _split_entries(entries, cfg.val_ratio, cfg.seed)
    if not train_entries:
        raise RuntimeError("Training split is empty. Add more labeled videos and try again.")

    train_labels = [CLASS_TO_IDX[item["label"]] for item in train_entries]
    val_labels = [CLASS_TO_IDX[item["label"]] for item in val_entries]

    train_dataset = VideoClipDataset(
        train_entries,
        clip_len=cfg.clip_len,
        stride=cfg.clip_stride,
        image_size=cfg.image_size,
        train=True,
        clips_per_video=cfg.train_clips_per_video,
    )
    val_dataset = VideoClipDataset(
        val_entries,
        clip_len=cfg.clip_len,
        stride=cfg.clip_stride,
        image_size=cfg.image_size,
        train=False,
        clips_per_video=cfg.val_clips_per_video,
    )

    train_loader = _make_loader(train_dataset, train_labels, cfg.batch_size, cfg.num_workers, train=True)
    val_loader = _make_loader(val_dataset, val_labels, cfg.batch_size, cfg.num_workers, train=False)

    model = build_model(pretrained=cfg.use_pretrained_backbone, device=DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    history: List[Dict[str, float]] = []
    best_metric = -float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience = 0

    use_amp = DEVICE == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for clips, labels in train_loader:
            clips = clips.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            amp_context = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            with amp_context:
                logits = model(clips)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(logits.detach(), dim=1)
            batch_size = labels.size(0)
            running_loss += float(loss.item()) * batch_size
            running_correct += int((preds == labels).sum().item())
            running_samples += int(batch_size)

        train_metrics = {
            "loss": running_loss / max(1, running_samples),
            "accuracy": running_correct / max(1, running_samples),
        }
        val_metrics = _evaluate(model, val_loader, DEVICE) if len(val_entries) > 0 else {"loss": 0.0, "accuracy": 0.0}

        monitor_metric = val_metrics["accuracy"] if len(val_entries) > 0 else train_metrics["accuracy"]
        if monitor_metric > best_metric:
            best_metric = monitor_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
            }
        )

        if patience >= cfg.early_stop_patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    checkpoint = {
        "checkpoint_format": "poseai_video_v1",
        "model_name": "simple_3d_cnn",
        "class_names": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "clip_len": cfg.clip_len,
        "clip_stride": cfg.clip_stride,
        "image_size": cfg.image_size,
        "device_trained": DEVICE,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "train_videos": len(train_entries),
        "val_videos": len(val_entries),
        "video_counts": counts,
        "history": history,
        "model_state": best_state,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, MODEL_PATH)

    meta = {
        "checkpoint_format": "poseai_video_v1",
        "model_name": "simple_3d_cnn",
        "class_names": CLASS_NAMES,
        "clip_len": cfg.clip_len,
        "clip_stride": cfg.clip_stride,
        "image_size": cfg.image_size,
        "device_trained": DEVICE,
        "trained_at": checkpoint["trained_at"],
        "train_videos": len(train_entries),
        "val_videos": len(val_entries),
        "video_counts": counts,
        "best_accuracy": best_metric,
    }
    MODEL_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    TRAINING_HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
    TRAINING_SUMMARY_PATH.write_text(
        json.dumps(
            {
                "data_root": str(data_root),
                "train_entries": train_entries,
                "val_entries": val_entries,
                "config": asdict(cfg),
                "meta": meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not MODEL_PATH.exists():
        raise RuntimeError("Training ended but pose_model.pt was not created inside outputs/.")

    # --- Export accuracy CSV ---
    csv_fields = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    with open(TRAINING_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k, "") for k in csv_fields})

    # --- Export accuracy plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        epochs_x = [int(r["epoch"]) for r in history]
        train_acc = [float(r["train_accuracy"]) * 100 for r in history]
        val_acc = [float(r["val_accuracy"]) * 100 for r in history]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs_x, train_acc, marker="o", label="Train Accuracy", color="#4CAF50")
        ax.plot(epochs_x, val_acc, marker="s", label="Val Accuracy", color="#2196F3", linestyle="--")
        ax.set_title("Training Accuracy per Epoch", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(TRAINING_PLOT_PATH, dpi=150)
        plt.close(fig)
    except Exception:
        pass  # matplotlib not installed — skip plot

    # --- Export summary plot (confusion matrix + donut + accuracy curve) ---
    if val_entries:
        val_true, val_pred = _collect_predictions(model, val_loader, DEVICE)
        model.load_state_dict(best_state)
        _save_summary_plot(
            all_true=val_true,
            all_pred=val_pred,
            class_names=CLASS_NAMES,
            best_metric=best_metric,
            history=history,
            save_path=SUMMARY_PLOT_PATH,
        )

    return MODEL_PATH
