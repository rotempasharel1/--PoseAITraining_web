
from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
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
    POSE_TABULAR_PATH,
    PoseGRUClassifier,
    PoseHybridClassifier,
    PoseTabularClassifier,
    _POSE_FEATURES,
    _SIDE_LANDMARKS,
    _extract_pose_seq,
    build_pose_feature_sequence,
    summarize_pose_feature_sequence,
    build_model,
    collect_labeled_videos,
    get_video_frame_count,
    is_any_model_ready,
    is_model_ready,
    is_pose_model_ready,
    llm_feedback_for_row,
    load_video_clip,
    select_motion_focused_clip_starts,
    summarize_video_entries,
    _tta_logits,
)

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None


_PROXY_ENV_KEYS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]


@contextmanager
def _without_proxy_env() -> Any:
    previous = {key: os.environ.get(key) for key in _PROXY_ENV_KEYS + ["NO_PROXY", "no_proxy"]}
    try:
        for key in _PROXY_ENV_KEYS:
            os.environ.pop(key, None)
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def download_gdrive_folder(url: str, output_dir: str) -> None:
    if gdown is None:
        raise RuntimeError("gdown is not installed. Install it with: python -m pip install gdown")
    os.makedirs(output_dir, exist_ok=True)
    with _without_proxy_env():
        gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)


def count_labeled_videos(data_root: str | Path) -> Dict[str, int]:
    entries = collect_labeled_videos(data_root)
    return summarize_video_entries(entries)


def _load_checkpoint(model_path: str | Path) -> Dict[str, Any]:
    checkpoint = torch.load(str(model_path), map_location="cpu")
    if isinstance(checkpoint, dict) and checkpoint.get("checkpoint_format") in {"poseai_video_v1", "poseai_video_v2", "poseai_video_v3"}:
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


def _softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    temperature = float(temperature) if temperature else 1.0
    temperature = max(0.5, min(5.0, temperature))
    return torch.softmax(logits / temperature, dim=-1)


def _confidence_level(confidence: float, margin: float, agreement: float) -> str:
    if confidence >= 0.85 and margin >= 0.45 and agreement >= 0.75:
        return "high"
    if confidence >= 0.65 and margin >= 0.20 and agreement >= 0.50:
        return "medium"
    return "low"


def _source_reliability(meta: Optional[Dict[str, Any]], default_accuracy: float) -> float:
    accuracy = float((meta or {}).get("reliability", -1.0))
    if accuracy >= 0.0:
        return min(1.0, max(0.0, accuracy))

    best_accuracy = float((meta or {}).get("best_accuracy", default_accuracy) or default_accuracy)
    reliability = (best_accuracy - 0.5) / 0.35
    return min(1.0, max(0.0, reliability))


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
        edge_density_scores: List[float] = []
        edge_change_scores: List[float] = []
        prev_edges: Optional[np.ndarray] = None

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if wanted is not None and frames_seen not in wanted:
                frames_seen += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            edge_density_scores.append(float(np.mean(edges > 0)))
            if prev_edges is not None:
                edge_change_scores.append(float(np.mean(edges != prev_edges)))
            prev_edges = edges
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
            "mean_edge_density": float(np.mean(edge_density_scores)) if edge_density_scores else 0.0,
            "mean_edge_change": float(np.mean(edge_change_scores)) if edge_change_scores else 0.0,
        }


class SquatAnalyzer:
    def __init__(self, model_path: str | Path | None = None):
        self.device = DEVICE
        self.model_path = Path(model_path) if model_path is not None else MODEL_PATH

        self.model = None
        self.video_members: List[Dict[str, Any]] = []
        self.checkpoint: Optional[Dict[str, Any]] = None
        self.model_temperature = 1.0
        self.clip_len = 24
        self.clip_stride = 2
        self.image_size = 112
        self._current_video_path = ""

        self.pose_gru: Optional[Any] = None
        self.pose_gru_meta: Optional[Dict[str, Any]] = None
        self.pose_gru_members: List[Dict[str, Any]] = []
        self.pose_tabular: Optional[Any] = None
        self.pose_tabular_meta: Optional[Dict[str, Any]] = None
        self.pose_temperature = 1.0
        self.pose_frames = 32

        self.metrics_extractor = PoseMetricsExtractor()
        self.is_loaded = False
        self._load_model_if_available()

    def _load_model_if_available(self) -> None:
        self.is_loaded = False

        if is_pose_model_ready():
            try:
                ckpt = torch.load(str(POSE_GRU_PATH), map_location="cpu")
                pose_model_family = str(ckpt.get("model_family", "") or "")
                pose_model_cls = PoseHybridClassifier if pose_model_family == "pose_hybrid_v1" else PoseGRUClassifier
                self.pose_gru_members = []
                member_payloads = ckpt.get("members")
                if isinstance(member_payloads, list) and member_payloads:
                    for member_payload in member_payloads:
                        gru = pose_model_cls(
                            input_size=int(ckpt.get("pose_features", _POSE_FEATURES)),
                            hidden=int(ckpt.get("hidden", 64) or 64),
                            layers=int(ckpt.get("layers", 2) or 2),
                            num_classes=len(CLASS_NAMES),
                            dropout=float(ckpt.get("dropout", 0.35) or 0.35),
                        )
                        gru.load_state_dict(member_payload["model_state"])
                        gru.eval()
                        self.pose_gru_members.append(
                            {
                                "model": gru.to(self.device),
                                "temperature": float(member_payload.get("temperature", ckpt.get("temperature", 1.0)) or 1.0),
                                "reliability": float(member_payload.get("reliability", member_payload.get("best_accuracy", 0.6)) or 0.6),
                            }
                        )
                    self.pose_gru = self.pose_gru_members[0]["model"]
                else:
                    gru = pose_model_cls(
                        input_size=int(ckpt.get("pose_features", _POSE_FEATURES)),
                        hidden=int(ckpt.get("hidden", 64) or 64),
                        layers=int(ckpt.get("layers", 2) or 2),
                        num_classes=len(CLASS_NAMES),
                        dropout=float(ckpt.get("dropout", 0.35) or 0.35),
                    )
                    gru.load_state_dict(ckpt["model_state"])
                    gru.eval()
                    self.pose_gru = gru.to(self.device)
                self.pose_gru_meta = {k: v for k, v in ckpt.items() if k not in {"model_state", "members"}}
                self.pose_temperature = float(ckpt.get("temperature", 1.0) or 1.0)
                self.pose_frames = int(ckpt.get("n_frames", 32) or 32)
                self.is_loaded = True
            except Exception:
                self.pose_gru = None
                self.pose_gru_meta = None
                self.pose_gru_members = []

        if POSE_TABULAR_PATH.exists():
            try:
                ckpt = torch.load(str(POSE_TABULAR_PATH), map_location="cpu")
                tab = PoseTabularClassifier(
                    input_size=int(ckpt.get("input_size", _POSE_FEATURES * 4)),
                    hidden=int(ckpt.get("hidden", 128) or 128),
                    num_classes=len(CLASS_NAMES),
                    dropout=0.2,
                )
                tab.load_state_dict(ckpt["model_state"])
                tab.eval()
                self.pose_tabular = tab.to(self.device)
                self.pose_tabular_meta = ckpt
                self.is_loaded = True
            except Exception:
                self.pose_tabular = None
                self.pose_tabular_meta = None

        if self.model_path.exists():
            try:
                checkpoint = _load_checkpoint(self.model_path)
                self.checkpoint = checkpoint
                self.clip_len = int(checkpoint.get("clip_len", 24))
                self.clip_stride = int(checkpoint.get("clip_stride", 2))
                self.image_size = int(checkpoint.get("image_size", 112))
                self.model_temperature = float(checkpoint.get("temperature", 1.0) or 1.0)

                self.video_members = []
                member_payloads = checkpoint.get("members")
                if isinstance(member_payloads, list) and member_payloads:
                    for member_payload in member_payloads:
                        m = build_model(
                            pretrained=bool(checkpoint.get("use_pretrained_backbone", False)),
                            device=self.device,
                            freeze_backbone=bool(checkpoint.get("freeze_backbone", False)),
                            finetune_mode=str(checkpoint.get("finetune_mode", "last_block")),
                        )
                        m.load_state_dict(member_payload["model_state"])
                        m.eval()
                        self.video_members.append(
                            {
                                "model": m,
                                "temperature": float(member_payload.get("temperature", checkpoint.get("temperature", 1.0)) or 1.0),
                                "reliability": float(member_payload.get("reliability", member_payload.get("best_accuracy", 0.7)) or 0.7),
                            }
                        )
                    self.model = self.video_members[0]["model"]
                else:
                    m = build_model(
                        pretrained=bool(checkpoint.get("use_pretrained_backbone", False)),
                        device=self.device,
                        freeze_backbone=bool(checkpoint.get("freeze_backbone", False)),
                        finetune_mode=str(checkpoint.get("finetune_mode", "last_block")),
                    )
                    m.load_state_dict(checkpoint["model_state"])
                    m.eval()
                    self.model = m
                self.is_loaded = True
            except Exception:
                self.model = None
                self.checkpoint = None
                self.video_members = []

    def _select_clip_starts(self, total_frames: int) -> List[int]:
        return select_motion_focused_clip_starts(
            video_path=self._current_video_path,
            total_frames=total_frames,
            clip_len=self.clip_len,
            stride=self.clip_stride,
            max_clips=6,
            scan_frames=24,
        )

    def _pose_probs(self, video_path: str | Path) -> Optional[Dict[str, Any]]:
        if self.pose_gru is None:
            return None

        seq_coords = _extract_pose_seq(video_path, n_frames=self.pose_frames)
        if seq_coords is None:
            return None
        seq = build_pose_feature_sequence(seq_coords)

        with torch.no_grad():
            x = torch.from_numpy(seq).unsqueeze(0).to(self.device)
            if self.pose_gru_members:
                member_probs: List[np.ndarray] = []
                member_weights: List[float] = []
                for member in self.pose_gru_members:
                    logits = member["model"](x)
                    probs = _softmax_with_temperature(logits, float(member.get("temperature", 1.0))).cpu().numpy()[0]
                    member_probs.append(probs)
                    member_weights.append(max(0.01, _source_reliability({"reliability": member.get("reliability", 0.6)}, 0.6)))
                probs = np.average(np.stack(member_probs, axis=0), axis=0, weights=np.array(member_weights, dtype=np.float32))
            else:
                logits = self.pose_gru(x)
                probs = _softmax_with_temperature(logits, self.pose_temperature).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        return {
            "name": "pose_gru_ensemble" if self.pose_gru_members else "pose_gru",
            "probabilities": probs.tolist(),
            "prediction": CLASS_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "weight_hint": _source_reliability(self.pose_gru_meta, 0.75),
        }

    def _pose_tabular_probs(self, video_path: str | Path) -> Optional[Dict[str, Any]]:
        if self.pose_tabular is None or self.pose_tabular_meta is None:
            return None

        seq_coords = _extract_pose_seq(video_path, n_frames=self.pose_frames)
        if seq_coords is None:
            return None

        feature_seq = build_pose_feature_sequence(seq_coords)
        vector = summarize_pose_feature_sequence(feature_seq)
        mean = np.asarray(self.pose_tabular_meta.get("feature_mean"), dtype=np.float32).reshape(-1)
        std = np.asarray(self.pose_tabular_meta.get("feature_std"), dtype=np.float32).reshape(-1)
        vector = (vector - mean) / (std + 1e-6)

        with torch.no_grad():
            x = torch.from_numpy(vector).unsqueeze(0).to(self.device)
            logits = self.pose_tabular(x)
            probs = _softmax_with_temperature(logits, float(self.pose_tabular_meta.get("temperature", 1.0))).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        return {
            "name": "pose_tabular",
            "probabilities": probs.tolist(),
            "prediction": CLASS_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "weight_hint": _source_reliability(self.pose_tabular_meta, 0.72),
        }

    def _video_probs(self, video_path: str | Path) -> Optional[Dict[str, Any]]:
        if self.model is None:
            return None

        self._current_video_path = str(video_path)
        total_frames = get_video_frame_count(video_path)
        clip_starts = self._select_clip_starts(total_frames)
        if not clip_starts:
            return None

        clip_results: List[Dict[str, Any]] = []
        logits_list: List[torch.Tensor] = []
        clip_votes: List[int] = []

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
                if self.video_members:
                    member_probs: List[np.ndarray] = []
                    member_weights: List[float] = []
                    member_logits: List[torch.Tensor] = []
                    for member in self.video_members:
                        member_logit = _tta_logits(member["model"], clip)
                        member_logits.append(member_logit.detach().cpu())
                        member_prob = _softmax_with_temperature(member_logit, float(member.get("temperature", self.model_temperature))).cpu().numpy()[0]
                        member_probs.append(member_prob)
                        member_weights.append(max(0.01, _source_reliability({"reliability": member.get("reliability", 0.7)}, 0.7)))
                    logits = torch.mean(torch.cat(member_logits, dim=0), dim=0, keepdim=True)
                    probs = np.average(np.stack(member_probs, axis=0), axis=0, weights=np.array(member_weights, dtype=np.float32))
                else:
                    logits = _tta_logits(self.model, clip)
                    probs = _softmax_with_temperature(logits, self.model_temperature).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                clip_votes.append(pred_idx)
                logits_list.append(logits.detach().cpu())
                clip_results.append(
                    {
                        "start_frame": int(start),
                        "good_probability": float(probs[1]),
                        "bad_probability": float(probs[0]),
                        "prediction": CLASS_NAMES[pred_idx],
                    }
                )

        if not logits_list:
            return None

        mean_logits = torch.mean(torch.cat(logits_list, dim=0), dim=0, keepdim=True)
        final_probs = _softmax_with_temperature(mean_logits, self.model_temperature).cpu().numpy()[0]
        pred_idx = int(np.argmax(final_probs))

        agreement = 0.0
        if clip_votes:
            agreement = float(sum(1 for vote in clip_votes if vote == pred_idx) / len(clip_votes))

        best_acc = 0.65
        if self.checkpoint is not None:
            best_acc = float(self.checkpoint.get("best_accuracy", best_acc) or best_acc)

        return {
            "name": str(self.checkpoint.get("model_name", "video_model") if self.checkpoint else "video_model"),
            "probabilities": final_probs.tolist(),
            "prediction": CLASS_NAMES[pred_idx],
            "confidence": float(final_probs[pred_idx]),
            "clip_results": clip_results,
            "clips_analyzed": len(clip_results),
            "clip_agreement": agreement,
            "weight_hint": _source_reliability(self.checkpoint, best_acc),
            "total_frames": total_frames,
        }

    def analyze_video(self, video_path: str | Path) -> Dict[str, Any]:
        if not self.is_loaded:
            self._load_model_if_available()
        if not self.is_loaded or (self.model is None and self.pose_gru is None):
            return {"error": "Model not trained or not found. Please train the project first."}

        pose_result = self._pose_probs(video_path)
        pose_tabular_result = self._pose_tabular_probs(video_path)
        video_result = self._video_probs(video_path)

        sources = [src for src in [pose_result, pose_tabular_result, video_result] if src is not None]
        if not sources:
            return {"error": "Could not extract usable features from the uploaded video."}

        weights = [max(0.0, float(src.get("weight_hint", 0.0))) for src in sources]
        if weights:
            best_weight = max(weights)
            min_keep = max(0.18, best_weight - 0.18)
            strong_mask = [weight >= min_keep for weight in weights]
            if best_weight >= 0.65 and sum(strong_mask) > 1:
                strongest_idx = int(np.argmax(np.array(weights, dtype=np.float32)))
                strong_mask = [idx == strongest_idx for idx in range(len(weights))]
            if any(strong_mask):
                sources = [src for src, keep in zip(sources, strong_mask) if keep]
                weights = [weight for weight, keep in zip(weights, strong_mask) if keep]
        weight_sum = float(sum(weights))
        if weight_sum <= 1e-6:
            weights = [max(0.01, float(src.get("confidence", 0.5))) for src in sources]
            weight_sum = float(sum(weights))
        normalized_weights = [w / weight_sum for w in weights]

        final_probs = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        votes: List[str] = []
        for src, weight in zip(sources, normalized_weights):
            final_probs += np.array(src["probabilities"], dtype=np.float32) * weight
            votes.append(str(src["prediction"]))
            if "clip_results" in src:
                votes.extend([str(item["prediction"]) for item in src["clip_results"]])

        pred_idx = int(np.argmax(final_probs))
        pred_label = CLASS_NAMES[pred_idx]
        confidence = float(final_probs[pred_idx])
        margin = float(abs(final_probs[1] - final_probs[0]))
        agreement = float(sum(1 for vote in votes if vote == pred_label) / max(1, len(votes)))
        reliability_score = float(sum(weight * float(src.get("weight_hint", 0.0)) for src, weight in zip(sources, normalized_weights)))
        confidence *= 0.65 + (0.35 * max(0.0, min(1.0, reliability_score)))
        confidence = min(0.99, max(0.01, confidence))
        confidence_level = _confidence_level(confidence, margin, agreement)

        metrics = self.metrics_extractor.extract(video_path)
        feedback = llm_feedback_for_row(
            true_label="unknown",
            pred_label=pred_label,
            confidence=confidence,
            correct=True,
            metrics=metrics,
        )

        model_sources = [src["name"] for src in sources]
        source_summary = []
        for src, weight in zip(sources, normalized_weights):
            source_summary.append(
                {
                    "name": src["name"],
                    "weight": round(float(weight), 3),
                    "reliability": round(float(src.get("weight_hint", 0.0)), 3),
                    "prediction": src["prediction"],
                    "confidence": round(float(src["confidence"]), 4),
                }
            )

        result = {
            "prediction": pred_label,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "agreement": agreement,
            "margin": margin,
            "probabilities": {
                "bad": float(final_probs[0]),
                "good": float(final_probs[1]),
            },
            "video_model": " + ".join(model_sources),
            "model_sources": model_sources,
            "source_summary": source_summary,
            "clips_analyzed": int(video_result.get("clips_analyzed", 0)) if video_result else 0,
            "clip_results": video_result.get("clip_results", []) if video_result else [],
            "clip_agreement": float(video_result.get("clip_agreement", 0.0)) if video_result else 0.0,
            "total_frames": int(video_result.get("total_frames", get_video_frame_count(video_path))) if video_result else get_video_frame_count(video_path),
            "metrics": metrics,
            "llm_keep": feedback["llm_keep"],
            "llm_improve": feedback["llm_improve"],
            "keep_points": feedback.get("keep_points", []),
            "improve_points": feedback.get("improve_points", []),
            "primary_keep_tip": feedback.get("primary_keep_tip", ""),
            "primary_improve_tip": feedback.get("primary_improve_tip", ""),
        }
        return result
