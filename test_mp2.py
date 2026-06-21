import sys
import os
import ctypes

def _get_safe_mp_path():
    buffer = ctypes.create_unicode_buffer(256)
    ctypes.windll.kernel32.GetShortPathNameW(sys.executable, buffer, 256)
    short_python = buffer.value
    if short_python:
        venv_dir = os.path.dirname(os.path.dirname(short_python))
        sb_path = os.path.join(venv_dir, "Lib", "site-packages", "mediapipe", "python", "solution_base.py")
        return sb_path
    return None

import mediapipe.python.solution_base as sb
safe_path = _get_safe_mp_path()
print("Safe path:", safe_path)
if safe_path and os.path.exists(safe_path):
    sb.__file__ = safe_path
    print("Patched sb.__file__ successfully")
else:
    print("Failed to find path. exists:", os.path.exists(safe_path) if safe_path else False)

import mediapipe as mp
try:
    pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    print("SUCCESS")
except Exception as e:
    print("ERROR:", e)
