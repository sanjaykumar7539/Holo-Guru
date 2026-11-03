from core.gesture_detector import _interaction_from_landmarks


def hand_points():
    pts = [(0.5, 0.5) for _ in range(21)]
    # Stable palm width / MCP anchors.
    pts[5] = (0.44, 0.55)
    pts[17] = (0.60, 0.55)
    # PIP joints.
    for idx in (6, 10, 14, 18):
        pts[idx] = (0.5, 0.50)
    # Tips default folded down.
    for idx in (8, 12, 16, 20):
        pts[idx] = (0.5, 0.65)
    pts[3] = (0.47, 0.55)
    pts[4] = (0.48, 0.55)
    return pts


def test_index_only_moves_hologram():
    pts = hand_points()
    pts[8] = (0.46, 0.30)
    out = _interaction_from_landmarks(pts, "none")
    assert out["action"] == "move"


def test_two_fingers_rotate_hologram():
    pts = hand_points()
    pts[8] = (0.46, 0.30)
    pts[12] = (0.53, 0.30)
    out = _interaction_from_landmarks(pts, "none")
    assert out["action"] == "rotate"


def test_thumb_index_spread_zooms_hologram():
    pts = hand_points()
    pts[4] = (0.22, 0.55)  # extended thumb, but neither up nor down
    pts[8] = (0.48, 0.28)
    out = _interaction_from_landmarks(pts, "none")
    assert out["action"] == "zoom"
    assert 0.75 <= out["zoom"] <= 1.75


def test_pinch_selects_hologram():
    pts = hand_points()
    pts[8] = (0.50, 0.30)
    pts[4] = (0.505, 0.305)
    out = _interaction_from_landmarks(pts, "none")
    assert out["action"] == "select"


def test_closed_fist_resets_hologram():
    pts = hand_points()
    out = _interaction_from_landmarks(pts, "none")
    assert out["action"] == "reset"


def test_classroom_gesture_has_priority():
    pts = hand_points()
    pts[8] = (0.46, 0.30)
    out = _interaction_from_landmarks(pts, "raised_hand")
    assert out["action"] == "none"
