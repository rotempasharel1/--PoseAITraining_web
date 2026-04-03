
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

try:
    from torchvision.models.video import r3d_18, R3D_18_Weights
except Exception:  # pragma: no cover
    r3d_18 = None
    R3D_18_Weights = None


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
POSE_TABULAR_PATH = OUTPUT_DIR / "pose_tabular_model.pt"
POSE_TABULAR_META_PATH = OUTPUT_DIR / "pose_tabular_meta.json"

# Landmark indices for side-view squat analysis (MediaPipe Pose)
_SIDE_LANDMARKS = [11, 23, 25, 27, 29, 31]
_POSE_COORD_FEATURES = len(_SIDE_LANDMARKS) * 2  # x, y per landmark (normalized)
_POSE_DERIVED_FEATURES = 6
_POSE_FEATURES = _POSE_COORD_FEATURES + _POSE_COORD_FEATURES + _POSE_DERIVED_FEATURES

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Kinetics-400 normalization, commonly used with pretrained video backbones.
_VIDEO_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
_VIDEO_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


# ============================================================
# Config helpers
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

    clip_len: int = _env_int("CLIP_LEN", 24)
    clip_stride: int = _env_int("CLIP_STRIDE", 2)
    image_size: int = _env_int("IMAGE_SIZE", 112)

    batch_size: int = _env_int("BATCH_SIZE", 4)
    num_workers: int = _env_int("NUM_WORKERS", 0)

    epochs: int = _env_int("EPOCHS", 25)
    learning_rate: float = _env_float("LEARNING_RATE", 3e-4)
    weight_decay: float = _env_float("WEIGHT_DECAY", 1e-4)
    label_smoothing: float = _env_float("LABEL_SMOOTHING", 0.05)
    early_stop_patience: int = _env_int("EARLY_STOP_PATIENCE", 6)

    train_clips_per_video: int = _env_int("TRAIN_CLIPS_PER_VIDEO", 16)
    val_clips_per_video: int = _env_int("VAL_CLIPS_PER_VIDEO", 4)
    max_eval_clips: int = _env_int("MAX_EVAL_CLIPS", 6)
    quality_scan_frames: int = _env_int("QUALITY_SCAN_FRAMES", 24)
    min_video_frames: int = _env_int("MIN_VIDEO_FRAMES", 24)
    min_motion_score: float = _env_float("MIN_MOTION_SCORE", 0.012)
    min_pose_motion_score: float = _env_float("MIN_POSE_MOTION_SCORE", 0.015)

    use_pretrained_backbone: bool = _env_bool("USE_PRETRAINED_BACKBONE", True)

    pose_frames: int = _env_int("POSE_FRAMES", 32)
    pose_hidden_size: int = _env_int("POSE_HIDDEN_SIZE", 64)
    pose_layers: int = _env_int("POSE_LAYERS", 2)
    pose_dropout: float = _env_float("POSE_DROPOUT", 0.35)
    pose_learning_rate: float = _env_float("POSE_LEARNING_RATE", 1e-3)
    pose_ensemble_members: int = _env_int("POSE_ENSEMBLE_MEMBERS", 5)


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


def estimate_video_motion_score(video_path: str | Path, max_samples: int = 24) -> float:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    wanted = set(_uniform_indices_for_scan(total_frames, max_samples)) if total_frames > 0 else None

    prev_gray: Optional[np.ndarray] = None
    diffs: List[float] = []
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if wanted is not None and frame_idx not in wanted:
            frame_idx += 1
            continue
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        if prev_gray is not None:
            diffs.append(float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))) / 255.0))
        prev_gray = gray
        frame_idx += 1
    cap.release()
    if not diffs:
        return 0.0
    return float(np.mean(diffs))


def _uniform_indices_for_scan(total_frames: int, max_samples: int) -> List[int]:
    if total_frames <= 0:
        return []
    if total_frames <= max_samples:
        return list(range(total_frames))
    positions = np.linspace(0, total_frames - 1, num=max_samples)
    return [int(round(float(pos))) for pos in positions]


def select_motion_focused_clip_starts(
    video_path: str | Path,
    total_frames: int,
    clip_len: int,
    stride: int,
    max_clips: int,
    scan_frames: int,
) -> List[int]:
    uniform_starts = select_clip_starts_for_video(total_frames, clip_len, stride, max_clips)
    if total_frames <= 0 or len(uniform_starts) <= 1:
        return uniform_starts

    clip_span = 1 + (clip_len - 1) * stride
    max_start = max(0, total_frames - clip_span)
    scan_positions = _uniform_indices_for_scan(total_frames, max(scan_frames, max_clips + 2))

    cap = cv2.VideoCapture(str(video_path))
    frames_by_idx: Dict[int, np.ndarray] = {}
    wanted = set(scan_positions)
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx in wanted:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            frames_by_idx[frame_idx] = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
            if len(frames_by_idx) == len(wanted):
                break
        frame_idx += 1
    cap.release()

    available_positions = [idx for idx in scan_positions if idx in frames_by_idx]
    if len(available_positions) < 3:
        return uniform_starts

    motion_scores: List[Tuple[float, int]] = []
    for start in uniform_starts:
        end = min(total_frames - 1, start + clip_span - 1)
        inside = [idx for idx in available_positions if start <= idx <= end]
        if len(inside) < 2:
            motion_scores.append((0.0, start))
            continue
        diffs = []
        prev = frames_by_idx[inside[0]]
        for idx in inside[1:]:
            cur = frames_by_idx[idx]
            diffs.append(float(np.mean(np.abs(cur.astype(np.float32) - prev.astype(np.float32))) / 255.0))
            prev = cur
        motion_scores.append((float(np.mean(diffs)) if diffs else 0.0, start))

    motion_scores.sort(key=lambda item: item[0], reverse=True)
    ranked = [start for _, start in motion_scores[:max_clips]]
    return sorted(set(ranked)) or uniform_starts


def select_clip_starts_for_video(
    total_frames: int,
    clip_len: int,
    stride: int,
    max_clips: int,
) -> List[int]:
    if total_frames <= 0:
        return [0]

    clip_span = 1 + (clip_len - 1) * stride
    if total_frames <= clip_span or max_clips <= 1:
        return [0]

    max_start = max(0, total_frames - clip_span)
    positions = np.linspace(0, max_start, num=max(1, max_clips))
    starts = sorted({int(round(float(pos))) for pos in positions})
    return starts or [0]


def _sample_indices(
    total_frames: int,
    clip_len: int,
    stride: int,
    *,
    center: bool,
    start_idx: Optional[int],
) -> List[int]:
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
        indices = _sample_indices(
            len(fallback_frames),
            clip_len,
            stride,
            center=center,
            start_idx=start_idx,
        )
        frames = [_resize_rgb(fallback_frames[idx], image_size) for idx in indices]
    else:
        indices = _sample_indices(
            total_frames,
            clip_len,
            stride,
            center=center,
            start_idx=start_idx,
        )
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
    clip = (clip - _VIDEO_MEAN) / _VIDEO_STD
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
            clip = torch.flip(clip, dims=[3])  # horizontal flip
        if self.train and random.random() < 0.25:
            clip = clip + (0.01 * torch.randn_like(clip))
        label = CLASS_TO_IDX[str(item["label"]).lower()]
        return clip, label


# ============================================================
# Model
# ============================================================
class VideoClassifier3D(nn.Module):
    """Fallback lightweight 3D CNN when torchvision video models are unavailable."""

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


def build_model(
    pretrained: bool = False,
    device: str = DEVICE,
    freeze_backbone: bool = False,
) -> nn.Module:
    if r3d_18 is not None:
        weights = None
        if pretrained and R3D_18_Weights is not None:
            weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        if freeze_backbone:
            for name, param in model.named_parameters():
                param.requires_grad = name.startswith("fc.")
    else:
        model = VideoClassifier3D(num_classes=len(CLASS_NAMES))
    model.to(device)
    return model


# ============================================================
# Split / train helpers
# ============================================================
def _split_entries(
    entries: Sequence[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, Any]]] = {"good": [], "bad": []}
    for item in entries:
        by_label[str(item["label"]).lower()].append(dict(item))

    train_entries: List[Dict[str, Any]] = []
    val_entries: List[Dict[str, Any]] = []

    for _label, items in by_label.items():
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


def _angle_feature(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ab = a - b
    cb = c - b
    denom = (np.linalg.norm(ab, axis=1) * np.linalg.norm(cb, axis=1)) + 1e-6
    cosine = np.sum(ab * cb, axis=1) / denom
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine)).astype(np.float32)


def build_pose_feature_sequence(coords_seq: np.ndarray) -> np.ndarray:
    if coords_seq.size == 0:
        return np.zeros((0, _POSE_FEATURES), dtype=np.float32)

    coords = coords_seq.astype(np.float32, copy=True)
    delta = np.vstack([np.zeros((1, coords.shape[1]), dtype=np.float32), np.diff(coords, axis=0)])

    shoulder = coords[:, 0:2]
    hip = coords[:, 2:4]
    knee = coords[:, 4:6]
    ankle = coords[:, 6:8]
    heel = coords[:, 8:10]
    foot = coords[:, 10:12]

    torso_lean = np.degrees(np.arctan2(np.abs(shoulder[:, 0] - hip[:, 0]), np.abs(shoulder[:, 1] - hip[:, 1]) + 1e-6))
    hip_angle = _angle_feature(shoulder, hip, knee)
    knee_angle = _angle_feature(hip, knee, ankle)
    ankle_angle = _angle_feature(knee, ankle, foot)
    knee_over_toe = knee[:, 0] - foot[:, 0]
    hip_depth = hip[:, 1] - knee[:, 1]

    derived = np.stack(
        [
            torso_lean / 45.0,
            hip_angle / 180.0,
            knee_angle / 180.0,
            ankle_angle / 180.0,
            knee_over_toe,
            hip_depth,
        ],
        axis=1,
    ).astype(np.float32)

    return np.concatenate([coords, delta, derived], axis=1).astype(np.float32)


def summarize_pose_feature_sequence(feature_seq: np.ndarray) -> np.ndarray:
    if feature_seq.size == 0:
        return np.zeros(_POSE_FEATURES * 4, dtype=np.float32)
    mean = feature_seq.mean(axis=0)
    std = feature_seq.std(axis=0)
    min_v = feature_seq.min(axis=0)
    max_v = feature_seq.max(axis=0)
    return np.concatenate([mean, std, min_v, max_v], axis=0).astype(np.float32)


def pose_motion_score_from_sequence(coords_seq: Optional[np.ndarray]) -> float:
    if coords_seq is None or len(coords_seq) < 2:
        return 0.0
    deltas = np.diff(coords_seq.astype(np.float32), axis=0)
    return float(np.mean(np.linalg.norm(deltas, axis=1)))


def filter_entries_for_pose_training(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for item in entries:
        seq = _extract_pose_seq(item["path"], n_frames=cfg.pose_frames)
        if seq is None:
            continue
        if pose_motion_score_from_sequence(seq) < cfg.min_pose_motion_score:
            continue
        filtered.append(dict(item))
    return filtered


def filter_entries_for_video_training(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for item in entries:
        frame_count = get_video_frame_count(item["path"])
        if frame_count < cfg.min_video_frames:
            continue
        motion_score = estimate_video_motion_score(item["path"], max_samples=cfg.quality_scan_frames)
        if motion_score < cfg.min_motion_score:
            continue
        enriched = dict(item)
        enriched["frame_count"] = frame_count
        enriched["motion_score"] = motion_score
        filtered.append(enriched)
    return filtered


def _make_loader(
    dataset: Dataset,
    labels: Sequence[int],
    batch_size: int,
    num_workers: int,
    train: bool,
) -> DataLoader:
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


def _set_video_backbone_mode(model: nn.Module, backbone_trainable: bool) -> None:
    if backbone_trainable or not hasattr(model, "fc"):
        return
    for name, module in model.named_children():
        if name != "fc":
            module.eval()


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    if len(loader) == 0:
        return {"loss": 0.0, "accuracy": 0.0}

    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
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
    all_true: List[int] = []
    all_pred: List[int] = []
    model.eval()
    for inputs, labels in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        all_pred.extend(preds)
        all_true.extend(labels.tolist())
    return all_true, all_pred


@torch.no_grad()
def _collect_logits_labels(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    model.eval()
    for inputs, labels in loader:
        inputs = inputs.to(device)
        logits = model(inputs).detach().cpu()
        logits_list.append(logits)
        labels_list.append(labels.detach().cpu())
    if not logits_list:
        return None, None
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def _accuracy_to_reliability(accuracy: float) -> float:
    bounded = min(1.0, max(0.0, float(accuracy)))
    return min(1.0, max(0.0, (bounded - 0.5) / 0.35))


@torch.no_grad()
def _predict_video_logits(
    model: nn.Module,
    video_path: str | Path,
    *,
    clip_len: int,
    stride: int,
    image_size: int,
    device: str,
    max_eval_clips: int,
) -> Tuple[Optional[torch.Tensor], List[Dict[str, Any]], float, int]:
    total_frames = get_video_frame_count(video_path)
    clip_starts = select_motion_focused_clip_starts(
        video_path,
        total_frames,
        clip_len,
        stride,
        max_eval_clips,
        cfg.quality_scan_frames,
    )
    if not clip_starts:
        return None, [], 0.0, total_frames

    logits_list: List[torch.Tensor] = []
    clip_results: List[Dict[str, Any]] = []
    clip_votes: List[int] = []

    for start in clip_starts:
        clip = load_video_clip(
            video_path,
            clip_len=clip_len,
            stride=stride,
            image_size=image_size,
            center=False,
            start_idx=start,
        ).unsqueeze(0).to(device)
        logits = model(clip).detach().cpu()
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = int(torch.argmax(probs).item())
        clip_votes.append(pred_idx)
        logits_list.append(logits)
        clip_results.append(
            {
                "start_frame": int(start),
                "prediction": CLASS_NAMES[pred_idx],
                "bad_probability": float(probs[0].item()),
                "good_probability": float(probs[1].item()),
            }
        )

    if not logits_list:
        return None, [], 0.0, total_frames

    mean_logits = torch.mean(torch.cat(logits_list, dim=0), dim=0, keepdim=True)
    pred_idx = int(torch.argmax(mean_logits, dim=1).item())
    agreement = float(sum(1 for vote in clip_votes if vote == pred_idx) / max(1, len(clip_votes)))
    return mean_logits, clip_results, agreement, total_frames


@torch.no_grad()
def _evaluate_video_entries(
    model: nn.Module,
    entries: Sequence[Dict[str, Any]],
    *,
    clip_len: int,
    stride: int,
    image_size: int,
    device: str,
    max_eval_clips: int,
) -> Dict[str, Any]:
    if not entries:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "logits": None,
            "labels": None,
            "predictions": [],
        }

    criterion = nn.CrossEntropyLoss()
    logits_list: List[torch.Tensor] = []
    labels_list: List[int] = []
    predictions: List[int] = []
    total_loss = 0.0

    model.eval()
    for item in entries:
        logits, _, _, _ = _predict_video_logits(
            model,
            item["path"],
            clip_len=clip_len,
            stride=stride,
            image_size=image_size,
            device=device,
            max_eval_clips=max_eval_clips,
        )
        if logits is None:
            continue

        label = CLASS_TO_IDX[str(item["label"]).lower()]
        label_tensor = torch.tensor([label], dtype=torch.long)
        total_loss += float(criterion(logits, label_tensor).item())
        logits_list.append(logits)
        labels_list.append(label)
        predictions.append(int(torch.argmax(logits, dim=1).item()))

    if not logits_list:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "logits": None,
            "labels": None,
            "predictions": [],
        }

    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    pred_tensor = torch.tensor(predictions, dtype=torch.long)
    return {
        "loss": total_loss / len(logits_list),
        "accuracy": float((pred_tensor == labels_tensor).float().mean().item()),
        "logits": torch.cat(logits_list, dim=0),
        "labels": labels_tensor,
        "predictions": predictions,
    }


def _fit_temperature(logits: Optional[torch.Tensor], labels: Optional[torch.Tensor]) -> float:
    if logits is None or labels is None or logits.numel() == 0:
        return 1.0

    logits = logits.float()
    labels = labels.long()
    temperature = nn.Parameter(torch.ones(1))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=50)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temp = torch.clamp(temperature, min=0.5, max=5.0)
        loss = criterion(logits / temp, labels)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        temp_value = float(torch.clamp(temperature.detach(), min=0.5, max=5.0).item())
        return temp_value if math.isfinite(temp_value) else 1.0
    except Exception:
        return 1.0


def _save_summary_plot(
    all_true: List[int],
    all_pred: List[int],
    class_names: List[str],
    best_metric: float,
    history: List[Dict[str, float]],
    save_path: Path,
) -> None:
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

        ax1 = fig.add_subplot(gs[0])
        cm_arr = np.array(cm, dtype=float)
        ax1.imshow(cm_arr, cmap="Blues", vmin=0)
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
                color = "white" if cm_arr[i, j] > max(1.0, cm_arr.max()) / 2 else "black"
                ax1.text(j, i, str(val), ha="center", va="center", color=color, fontsize=14, fontweight="bold")
        ax1.tick_params(colors="white")

        ax2 = fig.add_subplot(gs[1])
        sizes = [correct, wrong] if total > 0 else [1, 0]
        colors = ["#4CAF50", "#F44336"]
        ax2.pie(
            sizes,
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.5, edgecolor="#1a1a2e", linewidth=2),
        )
        pct = correct / max(total, 1) * 100
        ax2.text(0, 0, f"{pct:.0f}%", ha="center", va="center", color="white", fontsize=22, fontweight="bold")
        ax2.set_title("Val Accuracy", color="white", fontsize=12, fontweight="bold")
        ax2.legend(
            [f"Correct ({correct})", f"Wrong ({wrong})"],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.18),
            fontsize=9,
            frameon=False,
            labelcolor="white",
        )

        ax3 = fig.add_subplot(gs[2])
        ax3.set_facecolor("#12122a")
        epochs_x = [int(r["epoch"]) for r in history]
        train_acc = [float(r["train_accuracy"]) * 100 for r in history]
        val_acc = [float(r["val_accuracy"]) * 100 for r in history]
        ax3.plot(epochs_x, train_acc, marker="o", color="#4CAF50", label="Train", linewidth=2)
        ax3.plot(epochs_x, val_acc, marker="s", color="#2196F3", label="Val", linewidth=2, linestyle="--")
        ax3.set_ylim(0, 105)
        ax3.set_xlabel("Epoch", color="#aaaacc", fontsize=10)
        ax3.set_ylabel("Accuracy (%)", color="#aaaacc", fontsize=10)
        ax3.set_title("Accuracy per Epoch", color="white", fontsize=12, fontweight="bold")
        ax3.tick_params(colors="white")
        ax3.legend(fontsize=9, frameon=False, labelcolor="white")
        ax3.grid(True, alpha=0.2, color="#555577")

        fig.suptitle(
            f"Model Summary  |  Best val accuracy: {best_metric * 100:.1f}%",
            color="white",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    except Exception:
        pass


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
    return meta.get("checkpoint_format") in {"poseai_video_v1", "poseai_video_v2"}


def is_pose_model_ready(model_path: Path | str = POSE_GRU_PATH) -> bool:
    model_path = Path(model_path)
    if not model_path.exists() or not POSE_GRU_META_PATH.exists():
        return False
    try:
        meta = json.loads(POSE_GRU_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    return meta.get("checkpoint_format") in {"poseai_gru_v1", "poseai_gru_v2", "poseai_gru_v3", "poseai_gru_v4"}


def is_any_model_ready() -> bool:
    return is_model_ready() or is_pose_model_ready()


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
            "Good overall control through parts of the squat cycle."
            if pred_label == "good"
            else "You still show moments of useful control to build on."
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
            "You are a helpful squat coach. The classification was produced from a whole-video model and/or a pose-sequence model. "
            "Return concise coaching feedback in English only. "
            "You must explicitly mention KNEES and BACK/TORSO in the feedback."
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
    try:
        import mediapipe as mp
        pose_sol = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
    except Exception:
        return None

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

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

    seq: List[List[float]] = []
    last = frames_lm[min(frames_lm.keys())]
    wanted_sorted = sorted(wanted) if wanted else list(range(n_frames))
    for idx in wanted_sorted:
        best = min(frames_lm.keys(), key=lambda k: abs(k - idx))
        if abs(best - idx) < 10:
            last = frames_lm[best]
        seq.append(last)

    while len(seq) < n_frames:
        seq.append(seq[-1])
    seq = seq[:n_frames]

    return np.array(seq, dtype=np.float32)


class PoseSequenceDataset(Dataset):
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
            self._cache[idx] = _extract_pose_seq(self.entries[idx]["path"], self.n_frames)
        seq = self._cache[idx]

        if seq is None:
            seq = np.zeros((self.n_frames, _POSE_COORD_FEATURES), dtype=np.float32)
        else:
            seq = seq.copy()

        if self.augment:
            if random.random() < 0.5:
                seq[:, 0::2] = -seq[:, 0::2]
            seq += np.random.normal(0, 0.015, seq.shape).astype(np.float32)
            shift = random.randint(-2, 2)
            if shift > 0:
                seq = np.concatenate([seq[shift:], np.tile(seq[-1:], (shift, 1))])
            elif shift < 0:
                seq = np.concatenate([np.tile(seq[:1], (-shift, 1)), seq[:shift]])

        seq = build_pose_feature_sequence(seq)
        label = CLASS_TO_IDX[str(self.entries[idx]["label"]).lower()]
        return torch.from_numpy(seq), label


class PoseGRUClassifier(nn.Module):
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
            input_size,
            hidden,
            layers,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class PoseHybridClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = _POSE_FEATURES,
        hidden: int = 64,
        layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(
            input_size,
            hidden,
            layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 8),
            nn.Dropout(dropout),
            nn.Linear(hidden * 8, hidden * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.8),
            nn.Linear(hidden * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        attn_scores = self.attention(out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        attn_pool = torch.sum(out * attn_weights, dim=1)
        mean_pool = out.mean(dim=1)
        max_pool = out.max(dim=1).values
        std_pool = torch.sqrt(torch.var(out, dim=1, unbiased=False) + 1e-6)
        fused = torch.cat([attn_pool, mean_pool, max_pool, std_pool], dim=1)
        return self.head(fused)


class PoseTabularClassifier(nn.Module):
    def __init__(self, input_size: int, hidden: int = 128, num_classes: int = 2, dropout: float = 0.25) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.8),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Training pipelines
# ============================================================
def _train_single_pose_model(
    train_entries: Sequence[Dict[str, Any]],
    val_entries: Sequence[Dict[str, Any]],
    *,
    split_seed: int,
) -> Dict[str, Any]:
    set_seed(split_seed)

    train_ds = PoseSequenceDataset(train_entries, n_frames=cfg.pose_frames, augment=True)
    val_ds = PoseSequenceDataset(val_entries, n_frames=cfg.pose_frames, augment=False)

    sample_seq, _ = train_ds[0]
    if sample_seq.abs().sum().item() == 0.0:
        raise RuntimeError(
            "MediaPipe could not extract pose landmarks from any training video. "
            "The pose model requires MediaPipe to work correctly."
        )

    train_labels = [CLASS_TO_IDX[e["label"]] for e in train_entries]
    batch = min(max(2, cfg.batch_size), 8)
    train_loader = _make_loader(train_ds, train_labels, batch, cfg.num_workers, train=True)
    val_loader = _make_loader(val_ds, [], batch, cfg.num_workers, train=False)

    model = PoseHybridClassifier(
        input_size=_POSE_FEATURES,
        hidden=cfg.pose_hidden_size,
        layers=cfg.pose_layers,
        num_classes=len(CLASS_NAMES),
        dropout=cfg.pose_dropout,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.pose_learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

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

        train_m = {
            "loss": run_loss / max(1, run_samples),
            "accuracy": run_correct / max(1, run_samples),
        }
        val_m = _evaluate(model, val_loader, DEVICE) if val_entries else {"loss": 0.0, "accuracy": 0.0}
        scheduler.step(val_m["accuracy"])
        monitor = val_m["accuracy"] if val_entries else train_m["accuracy"]

        if monitor > best_metric:
            best_metric = monitor
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_m["loss"]),
                "train_accuracy": float(train_m["accuracy"]),
                "val_loss": float(val_m["loss"]),
                "val_accuracy": float(val_m["accuracy"]),
            }
        )

        if patience >= cfg.early_stop_patience:
            break

    if best_state is None:
        raise RuntimeError("Pose GRU training did not produce a valid model.")

    model.load_state_dict(best_state)
    val_logits, val_labels = _collect_logits_labels(model, val_loader, DEVICE)
    temperature = _fit_temperature(val_logits, val_labels)

    return {
        "split_seed": split_seed,
        "train_videos": len(train_entries),
        "val_videos": len(val_entries),
        "best_accuracy": best_metric,
        "reliability": _accuracy_to_reliability(best_metric),
        "temperature": temperature,
        "history": history,
        "model_family": "pose_hybrid_v1",
        "model_state": best_state,
    }


def _extract_pose_tabular_vector(video_path: str | Path, n_frames: int) -> Optional[np.ndarray]:
    seq = _extract_pose_seq(video_path, n_frames=n_frames)
    if seq is None:
        return None
    feature_seq = build_pose_feature_sequence(seq)
    return summarize_pose_feature_sequence(feature_seq)


def _train_pose_tabular_model(
    entries: Sequence[Dict[str, Any]],
    *,
    n_frames: int,
) -> Optional[Dict[str, Any]]:
    vectors: List[np.ndarray] = []
    labels: List[int] = []
    for item in entries:
        vec = _extract_pose_tabular_vector(item["path"], n_frames=n_frames)
        if vec is None:
            continue
        vectors.append(vec)
        labels.append(CLASS_TO_IDX[str(item["label"]).lower()])

    if len(vectors) < 4 or len(set(labels)) < 2:
        return None

    x = np.stack(vectors).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    x = (x - mean) / std

    x_tensor = torch.from_numpy(x).to(DEVICE)
    y_tensor = torch.from_numpy(y).to(DEVICE)

    model = PoseTabularClassifier(input_size=x.shape[1], hidden=128, num_classes=len(CLASS_NAMES), dropout=0.2).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=max(0.0, cfg.label_smoothing * 0.5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_acc = -float("inf")
    history: List[Dict[str, float]] = []

    for epoch in range(1, min(cfg.epochs, 20) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            eval_logits = model(x_tensor)
            preds = torch.argmax(eval_logits, dim=1)
            acc = float((preds == y_tensor).float().mean().item())
            history.append({"epoch": float(epoch), "train_loss": float(loss.item()), "train_accuracy": acc})
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        return None

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_logits = model(x_tensor).detach().cpu()
    temperature = _fit_temperature(final_logits, torch.from_numpy(y))

    return {
        "checkpoint_format": "poseai_tabular_v1",
        "model_name": "pose_tabular_mlp",
        "input_size": int(x.shape[1]),
        "hidden": 128,
        "temperature": temperature,
        "best_accuracy": best_acc,
        "reliability": _accuracy_to_reliability(best_acc),
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "history": history,
        "trained_samples": int(len(vectors)),
        "model_state": best_state,
    }


def run_pose_training_pipeline(data_root: Optional[str | Path] = None) -> Tuple[Path, float]:
    data_root = Path(data_root) if data_root is not None else ROOT / "temp_gdrive"
    entries = filter_entries_for_pose_training(collect_labeled_videos(data_root))
    counts = summarize_video_entries(entries)

    if counts["good"] == 0:
        raise RuntimeError("No GOOD pose videos with usable landmarks were found.")
    if counts["bad"] == 0:
        raise RuntimeError("No BAD pose videos with usable landmarks were found.")
    if counts["total"] < 2:
        raise RuntimeError("Need at least 2 usable videos for pose training.")

    members: List[Dict[str, Any]] = []
    all_histories: List[Dict[str, Any]] = []
    ensemble_members = max(1, min(cfg.pose_ensemble_members, 5))

    for member_idx in range(ensemble_members):
        split_seed = cfg.seed + (member_idx * 17)
        train_entries, val_entries = _split_entries(entries, cfg.val_ratio, split_seed)
        if not train_entries:
            continue
        member = _train_single_pose_model(train_entries, val_entries, split_seed=split_seed)
        members.append(member)
        all_histories.append(
            {
                "member_index": member_idx,
                "split_seed": split_seed,
                "history": member["history"],
                "best_accuracy": member["best_accuracy"],
            }
        )

    if not members:
        raise RuntimeError("Pose ensemble training did not produce any valid model.")

    best_metric = float(np.mean([float(member["best_accuracy"]) for member in members]))
    reliability = float(np.mean([float(member["reliability"]) for member in members]))
    tabular_checkpoint = _train_pose_tabular_model(entries, n_frames=cfg.pose_frames)

    checkpoint = {
        "checkpoint_format": "poseai_gru_v4",
        "model_name": "pose_hybrid_ensemble",
        "model_family": "pose_hybrid_v1",
        "class_names": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "n_frames": cfg.pose_frames,
        "pose_features": _POSE_FEATURES,
        "side_landmarks": _SIDE_LANDMARKS,
        "hidden": cfg.pose_hidden_size,
        "layers": cfg.pose_layers,
        "dropout": cfg.pose_dropout,
        "temperature": float(np.mean([float(member["temperature"]) for member in members])),
        "best_accuracy": best_metric,
        "reliability": reliability,
        "ensemble_members": len(members),
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "train_videos": max(int(member["train_videos"]) for member in members),
        "val_videos": max(int(member["val_videos"]) for member in members),
        "video_counts": counts,
        "history": all_histories,
        "members": members,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, POSE_GRU_PATH)

    meta = {k: v for k, v in checkpoint.items() if k != "members"}
    POSE_GRU_META_PATH.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    if tabular_checkpoint is not None:
        torch.save(tabular_checkpoint, POSE_TABULAR_PATH)
        tabular_meta = {k: v for k, v in tabular_checkpoint.items() if k not in {"model_state", "feature_mean", "feature_std"}}
        POSE_TABULAR_META_PATH.write_text(json.dumps(tabular_meta, indent=2, default=str), encoding="utf-8")
    return POSE_GRU_PATH, best_metric


def run_training_pipeline(data_root: Optional[str | Path] = None) -> Path:
    data_root = Path(data_root) if data_root is not None else ROOT / "temp_gdrive"
    entries = filter_entries_for_video_training(collect_labeled_videos(data_root))
    counts = summarize_video_entries(entries)

    if counts["good"] == 0:
        raise RuntimeError("No GOOD videos passed the video quality checks. Add clearer motion clips and try again.")
    if counts["bad"] == 0:
        raise RuntimeError("No BAD videos passed the video quality checks. Add clearer motion clips and try again.")
    if counts["total"] < 2:
        raise RuntimeError("At least 2 usable videos are required to train a video model.")

    train_entries, val_entries = _split_entries(entries, cfg.val_ratio, cfg.seed)
    if not train_entries:
        raise RuntimeError("Training split is empty. Add more labeled videos and try again.")

    train_labels = [CLASS_TO_IDX[item["label"]] for item in train_entries]
    train_dataset = VideoClipDataset(
        train_entries,
        clip_len=cfg.clip_len,
        stride=cfg.clip_stride,
        image_size=cfg.image_size,
        train=True,
        clips_per_video=cfg.train_clips_per_video,
    )
    train_loader = _make_loader(train_dataset, train_labels, cfg.batch_size, cfg.num_workers, train=True)
    freeze_backbone = bool(cfg.use_pretrained_backbone and r3d_18 is not None and counts["total"] < 60)
    model = build_model(
        pretrained=cfg.use_pretrained_backbone,
        device=DEVICE,
        freeze_backbone=freeze_backbone,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    history: List[Dict[str, float]] = []
    best_metric = -float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience = 0

    use_amp = DEVICE == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        _set_video_backbone_mode(model, backbone_trainable=not freeze_backbone)
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
        val_metrics = (
            _evaluate_video_entries(
                model,
                val_entries,
                clip_len=cfg.clip_len,
                stride=cfg.clip_stride,
                image_size=cfg.image_size,
                device=DEVICE,
                max_eval_clips=cfg.max_eval_clips,
            )
            if len(val_entries) > 0
            else {"loss": 0.0, "accuracy": 0.0, "logits": None, "labels": None, "predictions": []}
        )
        scheduler.step(val_metrics["accuracy"])

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

    model.load_state_dict(best_state)
    final_val_metrics = (
        _evaluate_video_entries(
            model,
            val_entries,
            clip_len=cfg.clip_len,
            stride=cfg.clip_stride,
            image_size=cfg.image_size,
            device=DEVICE,
            max_eval_clips=cfg.max_eval_clips,
        )
        if val_entries
        else {"logits": None, "labels": None, "predictions": []}
    )
    val_logits = final_val_metrics.get("logits")
    val_labels = final_val_metrics.get("labels")
    temperature = _fit_temperature(val_logits, val_labels)
    reliability = _accuracy_to_reliability(best_metric)

    checkpoint = {
        "checkpoint_format": "poseai_video_v2",
        "model_name": "r3d_18" if r3d_18 is not None else "simple_3d_cnn",
        "class_names": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "clip_len": cfg.clip_len,
        "clip_stride": cfg.clip_stride,
        "image_size": cfg.image_size,
        "temperature": temperature,
        "reliability": reliability,
        "use_pretrained_backbone": bool(cfg.use_pretrained_backbone and r3d_18 is not None),
        "freeze_backbone": freeze_backbone,
        "device_trained": DEVICE,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "train_videos": len(train_entries),
        "val_videos": len(val_entries),
        "video_counts": counts,
        "history": history,
        "best_accuracy": best_metric,
        "model_state": best_state,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, MODEL_PATH)

    meta = {k: v for k, v in checkpoint.items() if k != "model_state"}
    MODEL_META_PATH.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
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
            default=str,
        ),
        encoding="utf-8",
    )

    if not MODEL_PATH.exists():
        raise RuntimeError("Training ended but pose_model.pt was not created inside outputs/.")

    csv_fields = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    with open(TRAINING_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k, "") for k in csv_fields})

    try:
        import matplotlib
        matplotlib.use("Agg")
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
        pass

    if val_entries:
        model.load_state_dict(best_state)
        val_true = final_val_metrics["labels"].tolist() if final_val_metrics.get("labels") is not None else []
        val_pred = list(final_val_metrics.get("predictions") or [])
        _save_summary_plot(
            all_true=val_true,
            all_pred=val_pred,
            class_names=CLASS_NAMES,
            best_metric=best_metric,
            history=history,
            save_path=SUMMARY_PLOT_PATH,
        )

    return MODEL_PATH
