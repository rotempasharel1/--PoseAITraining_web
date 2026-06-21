import cv2
import numpy as np
import threading
import tempfile
import os
import sys
from typing import Optional, Dict, Any, List

try:
    import mediapipe as mp
    import mediapipe.python.solution_base as sb
    import ctypes
    
    # Fix MediaPipe unicode path bug on Windows robustly
    try:
        buffer = ctypes.create_unicode_buffer(256)
        ctypes.windll.kernel32.GetShortPathNameW(sys.executable, buffer, 256)
        short_python = buffer.value
        if short_python:
            venv_dir = os.path.dirname(os.path.dirname(short_python))
            sb_path = os.path.join(venv_dir, "Lib", "site-packages", "mediapipe", "python", "solution_base.py")
            if os.path.exists(sb_path):
                sb.__file__ = sb_path
    except Exception:
        pass
except Exception:
    mp = None


class SquatLiveTracker:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) if mp else None
        
        self.state = "IDLE" 
        self.rep_frames = []
        
        self.baseline_pose = None
        self.max_diff_in_window = 0.0
        self.hip_y_min = 1.0
        self.hip_y_max = 0.0
        
        import time
        self.last_analysis_time = time.time()
        
        self.latest_feedback = None
        self.all_feedbacks = []
        self._analysis_thread = None
        self._lock = threading.Lock()
        
    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.pose:
            return frame_bgr
            
        import time
        now = time.time()
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        self.rep_frames.append(frame_bgr.copy())
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                # Track key points
                hip_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y
                wrist_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y
                knee_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y
                shoulder_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y
                
                curr_pose = np.array([hip_y, wrist_y, knee_y, shoulder_y])
                
                if self.baseline_pose is None:
                    self.baseline_pose = curr_pose
                    
                self.baseline_pose = 0.9 * self.baseline_pose + 0.1 * curr_pose
                
                diffs = np.abs(curr_pose - self.baseline_pose)
                max_diff = np.max(diffs)
                
                if max_diff > self.max_diff_in_window:
                    self.max_diff_in_window = max_diff
                    
                if hip_y < self.hip_y_min: self.hip_y_min = hip_y
                if hip_y > self.hip_y_max: self.hip_y_max = hip_y
                    
            except Exception:
                pass
                
        elapsed = now - self.last_analysis_time
        
        if elapsed >= 3.0:
            frames_to_analyze = self.rep_frames.copy()
            movement = self.max_diff_in_window
            hip_drop = self.hip_y_max - self.hip_y_min
            
            # Reset for next 3 seconds
            self.rep_frames = []
            self.last_analysis_time = now
            self.max_diff_in_window = 0.0
            self.hip_y_min = 1.0
            self.hip_y_max = 0.0
            
            # A squat typically involves a hip drop of > 0.08 normalized units.
            if hip_drop < 0.08:
                with self._lock:
                    self.latest_feedback = {
                        "prediction": "No Squat",
                        "primary_keep_tip": "Waiting for a squat...",
                        "primary_improve_tip": "Bend your knees deeply to perform a squat."
                    }
            else:
                self._trigger_analysis(frames_to_analyze)
                
        # Force state to IDLE so app.py renders the feedback continuously
        self.state = "IDLE"

        cv2.putText(frame_bgr, f"Next feedback in: {max(0, 3.0 - elapsed):.1f}s", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(frame_bgr, (20, 70), 10, (0, 0, 255), -1)
        cv2.putText(frame_bgr, "Recording...", (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
        return frame_bgr

    def _trigger_analysis(self, frames: List[np.ndarray]):
        def analyze_task():
            if not frames:
                return
            h, w, _ = frames[0].shape
            fd, temp_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 10 fps instead of 20 (since we downsampled by ~2.5) -> keeps video playback speed normal-ish
            out = cv2.VideoWriter(temp_path, fourcc, 10.0, (w, h))
            for f in frames:
                out.write(f)
            out.release()
            
            try:
                res = self.analyzer.analyze_video(temp_path)
                with self._lock:
                    self.latest_feedback = res
                    self.all_feedbacks.append(res)
            except Exception as e:
                print("Live analysis error:", e)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        self._analysis_thread = threading.Thread(target=analyze_task, daemon=True)
        self._analysis_thread.start()
        
    def get_latest_feedback(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.latest_feedback
