"""
core/gesture_detector.py
Real-time local hand gesture recognition using MediaPipe Hands.

Classroom gestures (existing behaviour):
  raised_hand -> student wants to ask a question / interrupt
  thumbs_up   -> "I understood"
  thumbs_down -> "I did not understand"
  palm_stop   -> pause the avatar / TTS immediately

Touchless hologram interaction (new):
  point/index finger -> move hologram with hand position
  two fingers        -> rotate hologram
  thumb + index      -> zoom hologram using finger spread
  pinch              -> select/highlight hologram
  closed fist        -> reset hologram transform

The endpoint remains fully local; camera frames are not sent off-device.
"""
import base64
import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_hands = None


def _get_hands():
    global _hands
    if _hands is None:
        import mediapipe as mp
        # Tracking mode is faster for consecutive webcam frames than treating
        # every frame as an unrelated still image.
        _hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.55,
        )
    return _hands


def _decode_frame(image_b64: str) -> Optional[np.ndarray]:
    try:
        import cv2
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        raw = base64.b64decode(image_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error("Frame decode error: %s", e)
        return None


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _finger_states(landmarks):
    """Return [thumb, index, middle, ring, pinky] extension booleans."""
    # For the four long fingers, tip above PIP works reliably for upright hands.
    long_fingers = [
        landmarks[8][1] < landmarks[6][1],
        landmarks[12][1] < landmarks[10][1],
        landmarks[16][1] < landmarks[14][1],
        landmarks[20][1] < landmarks[18][1],
    ]

    # Thumb is considered extended when its tip is sufficiently separated from
    # the index MCP. This is orientation-agnostic enough for our interaction UI.
    thumb_extended = _dist(landmarks[4], landmarks[5]) > 0.12
    return [thumb_extended, *long_fingers]


def _classify_landmarks(landmarks) -> str:
    """Classify the original classroom gestures."""
    fingers_up = _finger_states(landmarks)
    other_four_down = not any(fingers_up[1:])

    # Preserve the earlier explicit thumb-up/down checks.
    thumb_up = landmarks[4][1] < landmarks[3][1] and landmarks[4][1] < landmarks[5][1]
    thumb_down = landmarks[4][1] > landmarks[3][1] and landmarks[4][1] > landmarks[5][1]

    xs = [p[0] for p in landmarks]
    hand_width = max(xs) - min(xs)

    if all(fingers_up[1:]):
        return "palm_stop" if hand_width > 0.35 else "raised_hand"
    if thumb_up and other_four_down:
        return "thumbs_up"
    if thumb_down and other_four_down:
        return "thumbs_down"
    return "none"


def _interaction_from_landmarks(landmarks, classroom_gesture: str = "none") -> Dict:
    """
    Convert landmarks into a stateless touchless interaction command.

    Coordinates are normalized so the browser can apply them directly to the
    hologram renderer.  The browser owns temporal smoothing/visual transforms.
    """
    center_x = sum(p[0] for p in landmarks) / len(landmarks)
    center_y = sum(p[1] for p in landmarks) / len(landmarks)
    fingers = _finger_states(landmarks)
    thumb, index, middle, ring, pinky = fingers

    thumb_index = _dist(landmarks[4], landmarks[8])
    hand_span = max(_dist(landmarks[5], landmarks[17]), 0.08)
    pinch_ratio = thumb_index / hand_span

    result = {
        "action": "none",
        "x": round(center_x, 4),
        "y": round(center_y, 4),
        "zoom": 1.0,
        "pinch_ratio": round(pinch_ratio, 4),
    }

    # Classroom commands always take priority over manipulation commands.
    if classroom_gesture != "none":
        return result

    # Pinch: select/highlight the current hologram object.
    if pinch_ratio < 0.42:
        result["action"] = "select"
        return result

    # Index only: move object with hand position.
    if index and not middle and not ring and not pinky and not thumb:
        result["action"] = "move"
        return result

    # Index + middle: rotate around Y axis; x controls angle in the browser.
    if index and middle and not ring and not pinky and not thumb:
        result["action"] = "rotate"
        return result

    # Thumb + index only: spread fingers to zoom.
    if thumb and index and not middle and not ring and not pinky:
        result["action"] = "zoom"
        # Normalize a comfortable real-world spread to 0.75x..1.75x.
        zoom = 0.75 + min(max(pinch_ratio - 0.45, 0.0), 1.0)
        result["zoom"] = round(min(max(zoom, 0.75), 1.75), 3)
        return result

    # Closed fist: reset to centered/default transform.
    if not any(fingers):
        result["action"] = "reset"
        return result

    return result


def detect_gesture(image_b64: str) -> Dict:
    """
    Input: base64-encoded frame.

    Output example:
      {
        "gesture": "none",
        "hand_found": true,
        "interaction": {"action":"move", "x":0.42, "y":0.55, ...}
      }
    """
    import cv2

    frame = _decode_frame(image_b64)
    if frame is None:
        return {
            "gesture": "none",
            "hand_found": False,
            "interaction": {"action": "none"},
        }

    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _get_hands().process(rgb)

        if not result.multi_hand_landmarks:
            return {
                "gesture": "none",
                "hand_found": False,
                "interaction": {"action": "none"},
            }

        hand = result.multi_hand_landmarks[0]
        points = [(lm.x, lm.y) for lm in hand.landmark]
        gesture = _classify_landmarks(points)
        interaction = _interaction_from_landmarks(points, gesture)

        return {
            "gesture": gesture,
            "hand_found": True,
            "interaction": interaction,
        }
    except ImportError:
        logger.warning("mediapipe not installed — run: pip install mediapipe")
        return {
            "gesture": "none",
            "hand_found": False,
            "interaction": {"action": "none"},
        }
    except Exception as e:
        logger.error("Gesture detection error: %s", e)
        return {
            "gesture": "none",
            "hand_found": False,
            "interaction": {"action": "none"},
        }
