import sys
import ctypes

def get_short_path(path):
    buffer = ctypes.create_unicode_buffer(256)
    get_short_path_name = ctypes.windll.kernel32.GetShortPathNameW
    get_short_path_name(path, buffer, 256)
    return buffer.value

try:
    import mediapipe.python.solution_base as sb
    
    # Try to monkey-patch the module path to short path
    sb.__file__ = get_short_path(sb.__file__)

    # Re-initialize or test
    import mediapipe as mp
    pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    print("SUCCESS: Pose initialized with short paths.")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("ERROR:", e)
