from core import hologram_engine as h


def test_toggle_and_explode():
    h.reset("s1")
    s = h.apply_action("s1", "toggle_orbit")
    assert s["orbit"] is True
    s = h.apply_action("s1", "set_explode", 80)
    assert s["explode"] == 80
    s = h.apply_action("s1", "next_layer")
    assert s["active_layer"] in s["layers"]


def test_focus_and_diagnostics():
    h.reset("s2")
    s = h.apply_action("s2", "focus_mode", "aorta")
    assert s["mode"] == "focus"
    assert s["selected_part"] == "aorta"
    s = h.apply_action("s2", "diagnostics_mode")
    assert s["mode"] == "diagnostics"
    assert s["scan"] is True


def test_parse_command_basic():
    out = h.parse_command("explode 60")
    assert out["action"] == "set_explode"
    assert out["value"] == 60
    out = h.parse_command("focus on rotor assembly")
    assert out["action"] == "focus_mode"
    assert out["value"] == "rotor assembly"


def test_asset_switch_and_command():
    h.reset("asset")
    s = h.apply_action("asset", "set_asset", "engine")
    assert s["asset"] == "engine"
    out = h.parse_command("show heart")
    assert out == {"action": "set_asset", "value": "heart"}


def test_new_asset_commands_parse():
    out = h.parse_command("show brain")
    assert out["action"] == "set_asset"
    assert out["value"] == "brain"
    out = h.parse_command("solar")
    assert out["action"] == "set_asset"
    assert out["value"] == "solar"


def test_imported_asset_parse_and_apply():
    out = h.parse_command("show imported")
    assert out["action"] == "set_asset"
    assert out["value"] == "imported"
    state = h.apply_action("s2", "set_asset", "imported")
    assert state["asset"] == "imported"


def test_section_slice_features():
    h.reset("slice")
    s = h.apply_action("slice", "toggle_section")
    assert s["section_enabled"] is True
    assert s["mode"] == "slice"
    s = h.apply_action("slice", "set_section_axis", "z")
    assert s["section_axis"] == "z"
    s = h.apply_action("slice", "set_section_pos", 25)
    assert s["section_pos"] == 25
    s = h.apply_action("slice", "section_nudge", 10)
    assert s["section_pos"] == 35
    out = h.parse_command("slice y 40")
    assert out["action"] == "set_section"
    assert out["value"]["axis"] == "y"
    assert out["value"]["pos"] == 40
    out = h.parse_command("cross section off")
    assert out["action"] == "set_section"
    assert out["value"] == "off"


def test_freeze_labels_orbit_speed():
    h.reset("fx")
    s = h.apply_action("fx", "toggle_freeze")
    assert s["freeze"] is True
    s = h.apply_action("fx", "set_labels", "off")
    assert s["labels"] is False
    s = h.apply_action("fx", "set_orbit_speed", 2.5)
    assert s["orbit_speed"] == 2.5
    s = h.apply_action("fx", "toggle_wire_density")
    assert s["wire_density"] == 2
    assert h.parse_command("freeze")["action"] == "toggle_freeze"
    assert h.parse_command("labels on")["action"] == "set_labels"
    assert h.parse_command("orbit speed 1.5")["value"] == 1.5


def test_measure_between_parts():
    h.reset("m1")
    s = h.apply_action("m1", "toggle_measure")
    assert s["measure_enabled"] is True
    s = h.apply_action("m1", "measure_pick", "aorta")
    assert s["measure_a"] == "aorta"
    s = h.apply_action("m1", "measure_pick", "ventricle")
    assert s["measure_b"] == "ventricle"
    s = h.apply_action("m1", "set_measure_distance", 1.25)
    assert s["measure_distance"] == 1.25
    s = h.apply_action("m1", "clear_measure")
    assert s["measure_enabled"] is False
    assert h.parse_command("measure")["action"] == "toggle_measure"


def test_explode_animation_timeline():
    h.reset("anim")
    h.apply_action("anim", "set_explode", 10)
    s = h.apply_action("anim", "start_explode_anim", 90)
    assert s["explode_anim"] is True
    assert s["explode_anim_from"] == 10
    assert s["explode_anim_to"] == 90
    s = h.apply_action("anim", "set_explode_anim_progress", 0.5)
    assert 40 <= s["explode"] <= 60
    s = h.apply_action("anim", "set_explode_anim_progress", 1.0)
    assert s["explode"] == 90
    assert s["explode_anim"] is False
    assert h.parse_command("explode anim 80")["value"] == 80


def test_multi_layer_clipping():
    h.reset("clip")
    s = h.apply_action("clip", "toggle_clip")
    assert s["clip_enabled"] is True
    s = h.apply_action("clip", "toggle_clip_layer", "internal")
    assert s["clip_layers"]["internal"] is False
    assert h.parse_command("clip layer internal")["action"] == "toggle_clip_layer"
