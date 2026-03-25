from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
try:
    import gdown
except Exception:  # pragma: no cover
    gdown = None
import numpy as np
import torch

from main import (
    CLASS_NAMES,
    DEVICE,
    MODEL_META_PATH,
    MODEL_PATH,
    OUTPUT_DIR,
    POSE_GRU_PATH,
    POSE_GRU_META_PATH,
    PoseGRUClassifier,
    _POSE_FEATURES,
    _SIDE_LANDMARKS,
    _extract_pose_seq,
    build_model,
    collect_labeled_videos,
    get_video_frame_count,
    is_model_ready,
    is_pose_model_ready,
    llm_feedback_for_row,
    load_video_clip,
    summarize_video_entries,
)

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None


def download_gdrive_folder(url: str, output_dir: str) -> None:
    if gdown is None:
        raise RuntimeError("gdown is not installed. Install it with: python -m pip install gdown")
    os.makedirs(output_dir, exist_ok=True)
    gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)


def count_labeled_videos(data_root: str | Path) -> Dict[str, int]:
    entries = collect_labeled_videos(data_root)
    return summarize_video_entries(entries)


def _load_checkpoint(model_path: str | Path) -> Dict[str, Any]:
    checkpoint = torch.load(str(model_path), map_location="cpu")
    if isinstance(checkpoint, dict) and checkpoint.get("checkpoint_format") == "poseai_video_v1":
        return checkpoint
    raise RuntimeError(
        "The saved model is not a valid whole-video checkpoint. Please retrain with the updated video-based pipeline."
    )


def _uniform_frame_indices(total_frames: int, max_samples: int) -> List[int]:
    if total_frames <= 0:
        return []
    if total_frames <= max_samples:
        return list(range(total_frames))
    positions = np.linspace(0, total_frames - 1, num=max_samples)
    return [int(round(float(pos))) for pos in positions]


def _safe_div(a: float, b: float) -> float:
    return a / b if abs(b) > 1e-6 else 0.0


class PoseMetricsExtractor:
    def __init__(self) -> None:
        self.available = mp is not None
        self.pose = None
        if self.available:
            try:
                self.pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception:
                # MediaPipe can fail when the .venv path contains unicode/special
                # characters (e.g. Hebrew folder names). Pose metrics will be
                # skipped; the video-based classification still works normally.
                self.pose = None

    def close(self) -> None:
        if self.pose is not None:
            self.pose.close()

    def extract(self, video_path: str | Path, max_frames: int = 32) -> Dict[str, float]:
        if self.pose is None:
            return {}

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        wanted = set(_uniform_frame_indices(total_frames, max_frames)) if total_frames > 0 else None

        frames_seen = 0
        sampled = 0
        torso_angles: List[float] = []
        knee_errors: List[float] = []
        depth_ratios: List[float] = []
        symmetry_scores: List[float] = []

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if wanted is not None and frames_seen not in wanted:
                frames_seen += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self.pose.process(frame_rgb)
            frames_seen += 1
            if not result.pose_landmarks:
                continue

            lm = result.pose_landmarks.landmark
            try:
                ls = lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                rs = lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
                lh = lm[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                rh = lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
                lk = lm[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
                rk = lm[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]
                la = lm[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
                ra = lm[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]
            except Exception:
                continue

            shoulder_mid_x = (ls.x + rs.x) / 2.0
            shoulder_mid_y = (ls.y + rs.y) / 2.0
            hip_mid_x = (lh.x + rh.x) / 2.0
            hip_mid_y = (lh.y + rh.y) / 2.0
            knee_mid_y = (lk.y + rk.y) / 2.0

            torso_dx = shoulder_mid_x - hip_mid_x
            torso_dy = shoulder_mid_y - hip_mid_y
            torso_angle = math.degrees(math.atan2(abs(torso_dx), abs(torso_dy) + 1e-6))
            torso_angles.append(float(torso_angle))

            shoulder_width = math.sqrt((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2) + 1e-6
            left_knee_error = abs(lk.x - la.x) / shoulder_width
            right_knee_error = abs(rk.x - ra.x) / shoulder_width
            knee_errors.append(float((left_knee_error + right_knee_error) / 2.0))

            torso_length = math.sqrt((shoulder_mid_x - hip_mid_x) ** 2 + (shoulder_mid_y - hip_mid_y) ** 2) + 1e-6
            depth_ratio = (hip_mid_y - knee_mid_y) / torso_length
            depth_ratios.append(float(depth_ratio))

            left_depth = (lh.y - lk.y) / torso_length
            right_depth = (rh.y - rk.y) / torso_length
            symmetry_scores.append(float(abs(left_depth - right_depth)))
            sampled += 1

        cap.release()

        if sampled == 0:
            return {}

        return {
            "pose_frames_used": float(sampled),
            "mean_torso_lean_deg": float(np.mean(torso_angles)),
            "torso_lean_std_deg": float(np.std(torso_angles)),
            "mean_knee_tracking_error": float(np.mean(knee_errors)),
            "max_depth_ratio": float(np.max(depth_ratios)),
            "mean_left_right_depth_diff": float(np.mean(symmetry_scores)),
        }


class SquatAnalyzer:
    def __init__(self, model_path: str | Path | None = None):
        self.device = DEVICE
        self.model_path = Path(model_path) if model_path is not None else MODEL_PATH
        # --- Video CNN ---
        self.model = None
        self.checkpoint: Optional[Dict[str, Any]] = None
        self.clip_len = 16
        self.clip_stride = 2
        self.image_size = 112
        # --- Pose GRU ---
        self.pose_gru: Optional[Any] = None
        self.pose_gru_meta: Optional[Dict[str, Any]] = None
        # --- Shared ---
        self.metrics_extractor = PoseMetricsExtractor()
        self.is_loaded = False
        self._load_model_if_available()

    def _load_model_if_available(self) -> None:
        # Try pose GRU first (preferred for side-view squats)
        if is_pose_model_ready():
            try:
                ckpt = torch.load(str(POSE_GRU_PATH), map_location="cpu")
                gru = PoseGRUClassifier(
                    input_size=int(ckpt.get("pose_features", _POSE_FEATURES)),
                    num_classes=len(CLASS_NAMES),
                )
                gru.load_state_dict(ckpt["model_state"])
                gru.eval()
                self.pose_gru = gru.to(self.device)
                self.pose_gru_meta = {k: v for k, v in ckpt.items() if k != "model_state"}
                self.is_loaded = True
            except Exception:
                self.pose_gru = None

        # Always try to load video CNN as backup
        if self.model_path.exists():
            try:
                checkpoint = torch.load(str(self.model_path), map_location="cpu")
                if isinstance(checkpoint, dict) and checkpoint.get("checkpoint_format") == "poseai_video_v1":
                    self.checkpoint = checkpoint
                    self.clip_len = int(checkpoint.get("clip_len", 16))
                    self.clip_stride = int(checkpoint.get("clip_stride", 2))
                    self.image_size = int(checkpoint.get("image_size", 112))
                    m = build_model(pretrained=False, device=self.device)
                    m.load_state_dict(checkpoint["model_state"])
                    m.eval()
                    self.model = m
                    self.is_loaded = True
            except Exception:
                pass

    def _select_clip_starts(self, total_frames: int) -> List[int]:
        if total_frames <= 0:
            return [0]
        clip_span = 1 + (self.clip_len - 1) * self.clip_stride
        if total_frames <= clip_span:
            return [0]

        max_start = max(0, total_frames - clip_span)
        num_clips = min(6, max(1, total_frames // max(clip_span // 2, 1)))
        positions = np.linspace(0, max_start, num=num_clips)
        starts = sorted({int(round(float(pos))) for pos in positions})
        return starts or [0]

    def analyze_video(self, video_path: str | Path) -> Dict[str, Any]:
        if not self.is_loaded:
            self._load_model_if_available()
        if not self.is_loaded or self.model is None:
            return {"error": "Model not trained or not found. Please train the whole-video model first."}

        total_frames = get_video_frame_count(video_path)
        clip_starts = self._select_clip_starts(total_frames)
        if not clip_starts:
            return {"error": "Could not sample clips from the uploaded video."}

        good_probs: List[float] = []
        clip_results: List[Dict[str, Any]] = []

        with torch.no_grad():
            for start in clip_starts:
                clip = load_video_clip(
                    video_path,
                    clip_len=self.clip_len,
                    stride=self.clip_stride,
                    image_size=self.image_size,
                    center=False,
                    start_idx=start,
                ).unsqueeze(0)
                clip = clip.to(self.device)
                logits = self.model(clip)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                good_prob = float(probs[1])
                good_probs.append(good_prob)
                clip_results.append(
                    {
                        "start_frame": int(start),
                        "good_probability": good_prob,
                        "bad_probability": float(probs[0]),
                    }
                )

        avg_good = float(np.mean(good_probs)) if good_probs else 0.0
        pred_label = "good" if avg_good >= 0.5 else "bad"
        confidence = avg_good if pred_label == "good" else (1.0 - avg_good)

        metrics = self.metrics_extractor.extract(video_path)
        feedback = llm_feedback_for_row(
            true_label="unknown",
            pred_label=pred_label,
            confidence=confidence,
            correct=True,
            metrics=metrics,
        )

        result = {
            "prediction": pred_label,
            "confidence": confidence,
            "video_model": "simple_3d_cnn",
            "clips_analyzed": len(clip_results),
            "clip_results": clip_results,
            "total_frames": total_frames,
            "metrics": metrics,
            "llm_keep": feedback["llm_keep"],
            "llm_improve": feedback["llm_improve"],
            "keep_points": feedback.get("keep_points", []),
            "improve_points": feedback.get("improve_points", []),
            "primary_keep_tip": feedback.get("primary_keep_tip", ""),
            "primary_improve_tip": feedback.get("primary_improve_tip", ""),
        }
        return result
