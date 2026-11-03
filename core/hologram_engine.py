"""
core/hologram_engine.py
Tony-Stark-inspired hologram scene controller (without copying any franchise
assets). This module keeps a lightweight per-session state for advanced HUD /
scene behaviours used by the frontend.

It does NOT render 3D itself; it provides deterministic scene-state updates so
web UI, local renderers, or future Unity/Three.js clients can stay in sync.

Features (classroom 3D module):
  - Built-in assets + imported mesh
  - Orbit / scan / x-ray / panels / explode / layers / focus / compare
  - Cross-section (slice) plane with axis + position
  - Freeze frame (pause animation)
  - Part labels on/off
  - Orbit speed control
  - Wireframe density
  - Distance measure between two parts
  - Animated explode timeline
  - Multi-layer clipping (surface / structure / internal / labels)
"""
from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import Any, Dict
import logging
import re

logger = logging.getLogger(__name__)

_DEFAULT_STATE: Dict[str, Any] = {
    "mode": "tutor",             # tutor | diagnostics | compare | focus | slice
    "orbit": False,
    "scan": False,
    "xray": False,
    "panels": True,
    "explode": 0,                 # 0..100
    "focus": "auto",
    "selected_part": None,
    "asset": "heart",
    "compare": False,
    "layer_index": 0,
    "layers": ["surface", "structure", "internal", "labels"],
    # --- new 3D classroom features ---
    "section_enabled": False,     # cross-section / slice plane
    "section_axis": "y",          # x | y | z
    "section_pos": 50,            # 0..100 along axis
    "freeze": False,              # pause orbit / scan animation
    "labels": True,               # draw part name callouts
    "orbit_speed": 1.0,           # 0.25 .. 3.0
    "wire_density": 1,            # 1 = normal, 2 = dense edges
    "measure_enabled": False,
    "measure_a": None,
    "measure_b": None,
    "measure_distance": None,
    "explode_anim": False,
    "explode_anim_from": 0,
    "explode_anim_to": 100,
    "explode_anim_ms": 1800,
    "explode_anim_started_at": 0,
    "clip_enabled": False,
    "clip_layers": {"surface": True, "structure": True, "internal": True, "labels": True},
}

_STATE: Dict[str, Dict[str, Any]] = {}
_LOCK = Lock()

_ASSETS = {"heart", "engine", "molecule", "anatomy", "brain", "cell", "solar", "imported"}
_AXES = {"x", "y", "z"}


def _clamp(value: int, lo: int = 0, hi: int = 100) -> int:
    return max(lo, min(hi, int(value)))


def _clamp_float(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _clean_session(session_id: str | None) -> str:
    return (session_id or "default").strip() or "default"


def default_state() -> Dict[str, Any]:
    return deepcopy(_DEFAULT_STATE)


def get_state(session_id: str | None = None) -> Dict[str, Any]:
    sid = _clean_session(session_id)
    with _LOCK:
        if sid not in _STATE:
            _STATE[sid] = default_state()
        # Merge any new default keys into older sessions
        current = _STATE[sid]
        for key, val in _DEFAULT_STATE.items():
            if key not in current:
                current[key] = deepcopy(val)
        return deepcopy(current)


def _save_state(session_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
    with _LOCK:
        _STATE[session_id] = deepcopy(state)
    return deepcopy(state)


def reset(session_id: str | None = None) -> Dict[str, Any]:
    sid = _clean_session(session_id)
    return _save_state(sid, default_state())


def apply_action(session_id: str | None, action: str, value: Any = None) -> Dict[str, Any]:
    sid = _clean_session(session_id)
    state = get_state(sid)
    action = (action or "").strip().lower()

    if action == "toggle_orbit":
        state["orbit"] = not state["orbit"]
    elif action == "toggle_scan":
        state["scan"] = not state["scan"]
    elif action == "toggle_xray":
        state["xray"] = not state["xray"]
    elif action == "toggle_panels":
        state["panels"] = not state["panels"]
    elif action == "set_asset":
        asset = str(value or "heart").strip().lower()
        if asset in _ASSETS:
            state["asset"] = asset
            state["selected_part"] = None
            state["focus"] = "auto"
            state["mode"] = "tutor"
    elif action == "toggle_compare":
        state["compare"] = not state["compare"]
        state["mode"] = "compare" if state["compare"] else ("focus" if state["focus"] != "auto" else "tutor")
    elif action == "explode_more":
        state["explode"] = _clamp(state["explode"] + 20)
    elif action == "explode_less":
        state["explode"] = _clamp(state["explode"] - 20)
    elif action == "set_explode":
        try:
            state["explode"] = _clamp(int(value))
        except (TypeError, ValueError) as exc:
            logger.debug("Invalid explode value %r: %s", value, exc)
    elif action == "next_layer":
        state["layer_index"] = (state["layer_index"] + 1) % len(state["layers"])
    elif action == "prev_layer":
        state["layer_index"] = (state["layer_index"] - 1) % len(state["layers"])
    elif action == "focus_mode":
        focus = str(value or "auto").strip() or "auto"
        state["focus"] = focus
        state["selected_part"] = None if focus == "auto" else focus
        state["mode"] = "focus" if focus != "auto" else "tutor"
    elif action == "diagnostics_mode":
        state["mode"] = "diagnostics"
        state["scan"] = True
        state["panels"] = True
    elif action == "tutor_mode":
        state["mode"] = "tutor"
    elif action == "reset_scene":
        state = default_state()
    elif action == "select_part":
        part = str(value or "").strip()
        if part:
            state["selected_part"] = part
            state["focus"] = part
            state["mode"] = "focus"
    # --- new features ---
    elif action == "toggle_section":
        state["section_enabled"] = not state["section_enabled"]
        if state["section_enabled"]:
            state["mode"] = "slice"
    elif action == "set_section":
        # value can be bool, "on"/"off", or dict
        if isinstance(value, dict):
            if "enabled" in value:
                state["section_enabled"] = bool(value["enabled"])
            if value.get("axis") in _AXES:
                state["section_axis"] = value["axis"]
            if "pos" in value:
                try:
                    state["section_pos"] = _clamp(int(value["pos"]))
                except (TypeError, ValueError):
                    pass
        else:
            token = str(value or "").strip().lower()
            if token in {"1", "true", "on", "yes"}:
                state["section_enabled"] = True
            elif token in {"0", "false", "off", "no"}:
                state["section_enabled"] = False
            else:
                state["section_enabled"] = not state["section_enabled"]
        if state["section_enabled"]:
            state["mode"] = "slice"
    elif action == "set_section_axis":
        axis = str(value or "y").strip().lower()
        if axis in _AXES:
            state["section_axis"] = axis
            state["section_enabled"] = True
            state["mode"] = "slice"
    elif action == "set_section_pos":
        try:
            state["section_pos"] = _clamp(int(value))
            state["section_enabled"] = True
            state["mode"] = "slice"
        except (TypeError, ValueError) as exc:
            logger.debug("Invalid section_pos %r: %s", value, exc)
    elif action == "section_nudge":
        delta = -10 if str(value).lower() in {"-", "down", "less", "back"} else 10
        try:
            if isinstance(value, (int, float)):
                delta = int(value)
        except (TypeError, ValueError):
            pass
        state["section_pos"] = _clamp(state.get("section_pos", 50) + delta)
        state["section_enabled"] = True
        state["mode"] = "slice"
    elif action == "toggle_freeze":
        state["freeze"] = not state["freeze"]
    elif action == "set_freeze":
        token = str(value if value is not None else "").strip().lower()
        if token in {"1", "true", "on", "yes", "freeze", "pause"}:
            state["freeze"] = True
        elif token in {"0", "false", "off", "no", "unfreeze", "play"}:
            state["freeze"] = False
        else:
            state["freeze"] = not state["freeze"]
    elif action == "toggle_labels":
        state["labels"] = not state["labels"]
    elif action == "set_labels":
        token = str(value if value is not None else "").strip().lower()
        if token in {"1", "true", "on", "yes"}:
            state["labels"] = True
        elif token in {"0", "false", "off", "no"}:
            state["labels"] = False
        else:
            state["labels"] = not state["labels"]
    elif action == "set_orbit_speed":
        try:
            state["orbit_speed"] = round(_clamp_float(float(value), 0.25, 3.0), 2)
        except (TypeError, ValueError) as exc:
            logger.debug("Invalid orbit_speed %r: %s", value, exc)
    elif action == "set_wire_density":
        try:
            dens = int(value)
            state["wire_density"] = 2 if dens >= 2 else 1
        except (TypeError, ValueError) as exc:
            logger.debug("Invalid wire_density %r: %s", value, exc)
    elif action == "toggle_wire_density":
        state["wire_density"] = 2 if state.get("wire_density", 1) < 2 else 1
    elif action == "toggle_measure":
        state["measure_enabled"] = not state["measure_enabled"]
        if not state["measure_enabled"]:
            state["measure_a"] = None
            state["measure_b"] = None
            state["measure_distance"] = None
        else:
            state["mode"] = "measure"
    elif action == "set_measure":
        token = str(value if value is not None else "").strip().lower()
        if token in {"0", "false", "off", "no"}:
            state["measure_enabled"] = False
            state["measure_a"] = None
            state["measure_b"] = None
            state["measure_distance"] = None
        elif token in {"1", "true", "on", "yes"}:
            state["measure_enabled"] = True
            state["mode"] = "measure"
        else:
            state["measure_enabled"] = not state["measure_enabled"]
            if not state["measure_enabled"]:
                state["measure_a"] = None
                state["measure_b"] = None
                state["measure_distance"] = None
            else:
                state["mode"] = "measure"
    elif action == "measure_pick":
        part = str(value or "").strip()
        if part:
            state["measure_enabled"] = True
            state["mode"] = "measure"
            if not state.get("measure_a"):
                state["measure_a"] = part
                state["measure_b"] = None
                state["measure_distance"] = None
            elif state.get("measure_a") == part or not state.get("measure_b"):
                if state.get("measure_a") == part:
                    state["measure_b"] = None
                    state["measure_distance"] = None
                else:
                    state["measure_b"] = part
            else:
                state["measure_a"] = part
                state["measure_b"] = None
                state["measure_distance"] = None
    elif action == "set_measure_distance":
        try:
            state["measure_distance"] = round(max(0.0, float(value)), 4)
        except (TypeError, ValueError) as exc:
            logger.debug("Invalid measure_distance %r: %s", value, exc)
    elif action == "clear_measure":
        state["measure_a"] = None
        state["measure_b"] = None
        state["measure_distance"] = None
        state["measure_enabled"] = False
    elif action == "start_explode_anim":
        state["explode_anim_from"] = int(state.get("explode", 0))
        to, ms = 100, 1800
        if isinstance(value, dict):
            if "from" in value:
                try: state["explode_anim_from"] = _clamp(int(value["from"]))
                except (TypeError, ValueError): pass
            if "to" in value:
                try: to = _clamp(int(value["to"]))
                except (TypeError, ValueError): pass
            if "ms" in value:
                try: ms = max(300, min(8000, int(value["ms"])))
                except (TypeError, ValueError): pass
        elif value is not None:
            try: to = _clamp(int(value))
            except (TypeError, ValueError): pass
        state["explode_anim_to"] = to
        state["explode_anim_ms"] = ms
        state["explode_anim"] = True
        state["explode_anim_started_at"] = 0
    elif action == "stop_explode_anim":
        state["explode_anim"] = False
    elif action == "set_explode_anim_progress":
        try:
            t = max(0.0, min(1.0, float(value)))
            a = int(state.get("explode_anim_from", 0))
            b = int(state.get("explode_anim_to", 100))
            state["explode"] = _clamp(int(round(a + (b - a) * t)))
            if t >= 1.0:
                state["explode_anim"] = False
        except (TypeError, ValueError) as exc:
            logger.debug("Invalid explode anim progress %r: %s", value, exc)
    elif action == "toggle_clip":
        state["clip_enabled"] = not state["clip_enabled"]
        if state["clip_enabled"]:
            state["mode"] = "clip"
    elif action == "set_clip":
        token = str(value if value is not None else "").strip().lower()
        if token in {"0", "false", "off", "no"}:
            state["clip_enabled"] = False
        elif token in {"1", "true", "on", "yes"}:
            state["clip_enabled"] = True
            state["mode"] = "clip"
        else:
            state["clip_enabled"] = not state["clip_enabled"]
            if state["clip_enabled"]:
                state["mode"] = "clip"
    elif action == "toggle_clip_layer":
        layer = str(value or "").strip().lower()
        clips = dict(state.get("clip_layers") or {})
        if layer in {"surface", "structure", "internal", "labels"}:
            clips[layer] = not bool(clips.get(layer, True))
            state["clip_layers"] = clips
            state["clip_enabled"] = True
            state["mode"] = "clip"
    elif action == "set_clip_layers":
        if isinstance(value, dict):
            clips = dict(state.get("clip_layers") or {})
            for k in ("surface", "structure", "internal", "labels"):
                if k in value:
                    clips[k] = bool(value[k])
            state["clip_layers"] = clips
            state["clip_enabled"] = True
            state["mode"] = "clip"
    else:
        state["last_unknown_action"] = action

    state["active_layer"] = state["layers"][state["layer_index"]]
    return _save_state(sid, state)


def parse_command(command: str) -> Dict[str, Any]:
    s = (command or "").strip().lower()
    if not s:
        return {"action": "noop", "value": None}

    if "reset" in s:
        return {"action": "reset_scene", "value": None}
    asset_match = re.search(
        r"(?:show|load|model|display)\s+(heart|engine|molecule|anatomy|brain|cell|solar|imported)", s
    )
    if asset_match:
        return {"action": "set_asset", "value": asset_match.group(1)}
    if s in _ASSETS:
        return {"action": "set_asset", "value": s}
    if "orbit speed" in s or "spin speed" in s or "rotation speed" in s:
        speed_match = re.search(r"(\d+(?:\.\d+)?)", s)
        if speed_match:
            return {"action": "set_orbit_speed", "value": float(speed_match.group(1))}
        return {"action": "set_orbit_speed", "value": 1.0}
    if "orbit" in s:
        return {"action": "toggle_orbit", "value": None}
    if "scan" in s:
        return {"action": "toggle_scan", "value": None}
    if "xray" in s or "x-ray" in s or "inside" in s:
        return {"action": "toggle_xray", "value": None}
    if "panel" in s or "hud" in s:
        return {"action": "toggle_panels", "value": None}
    if "compare" in s:
        return {"action": "toggle_compare", "value": None}
    if "diagnostic" in s:
        return {"action": "diagnostics_mode", "value": None}
    if "tutor" in s or "teaching" in s:
        return {"action": "tutor_mode", "value": None}
    if "next layer" in s or "next view" in s:
        return {"action": "next_layer", "value": None}
    if "previous layer" in s or "prev layer" in s:
        return {"action": "prev_layer", "value": None}

    if "explode anim" in s or "animate explode" in s or "play explode" in s:
        to_match = re.search(r"(\d{1,3})", s)
        to = int(to_match.group(1)) if to_match else 100
        return {"action": "start_explode_anim", "value": to}
    explode_match = re.search(r"explode(?:\s+view)?(?:\s+to)?\s+(\d{1,3})", s)
    if explode_match:
        return {"action": "set_explode", "value": int(explode_match.group(1))}
    if "explode" in s:
        return {"action": "explode_more", "value": None}
    if "implode" in s or "reduce explode" in s:
        return {"action": "explode_less", "value": None}

    # Cross-section / slice
    if "slice" in s or "cross section" in s or "cross-section" in s or "section plane" in s:
        axis_match = re.search(r"\b([xyz])\b(?:\s*-?\s*axis)?", s)
        pos_match = re.search(r"(\d{1,3})\s*%?", s)
        if "off" in s or "disable" in s or "hide" in s:
            return {"action": "set_section", "value": "off"}
        if axis_match or pos_match:
            payload: Dict[str, Any] = {"enabled": True}
            if axis_match:
                payload["axis"] = axis_match.group(1)
            if pos_match:
                payload["pos"] = int(pos_match.group(1))
            return {"action": "set_section", "value": payload}
        return {"action": "toggle_section", "value": None}
    if re.search(r"\b(cut|clip)\b", s) and ("plane" in s or "view" in s or "model" in s):
        return {"action": "toggle_section", "value": None}

    # Freeze / labels / wire
    if "unfreeze" in s or "resume" in s or "unpause" in s:
        return {"action": "set_freeze", "value": "off"}
    if "freeze" in s or "pause hologram" in s or "pause model" in s:
        return {"action": "toggle_freeze", "value": None}
    if "label" in s:
        if "off" in s or "hide" in s:
            return {"action": "set_labels", "value": "off"}
        if "on" in s or "show" in s:
            return {"action": "set_labels", "value": "on"}
        return {"action": "toggle_labels", "value": None}
    if "wire" in s and ("dense" in s or "detail" in s or "density" in s):
        return {"action": "toggle_wire_density", "value": None}
    if "measure" in s or "distance" in s or "how far" in s:
        if "off" in s or "clear" in s or "stop" in s:
            return {"action": "clear_measure", "value": None}
        return {"action": "toggle_measure", "value": None}
    if "stop explode" in s or "stop animation" in s:
        return {"action": "stop_explode_anim", "value": None}
    if "clip layer" in s or "layer clip" in s:
        for layer in ("surface", "structure", "internal", "labels"):
            if layer in s:
                return {"action": "toggle_clip_layer", "value": layer}
        return {"action": "toggle_clip", "value": None}
    if re.search(r"\bclip\b", s) and "clipboard" not in s:
        if "off" in s or "disable" in s:
            return {"action": "set_clip", "value": "off"}
        if "on" in s or "enable" in s:
            return {"action": "set_clip", "value": "on"}
        return {"action": "toggle_clip", "value": None}

    focus_match = re.search(r"(?:focus on|highlight|select)\s+([a-z0-9_\- ]{2,40})", s)
    if focus_match:
        part = focus_match.group(1).strip()
        return {"action": "focus_mode", "value": part}

    return {"action": "noop", "value": s}


def apply_command(session_id: str | None, command: str) -> Dict[str, Any]:
    parsed = parse_command(command)
    if parsed["action"] == "noop":
        return {"parsed": parsed, "state": get_state(session_id)}
    state = apply_action(session_id, parsed["action"], parsed.get("value"))
    return {"parsed": parsed, "state": state}
