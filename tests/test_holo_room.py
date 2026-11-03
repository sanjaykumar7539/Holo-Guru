from core import holo_room as r


def test_join_and_scene_sync():
    snap = r.join_room('r1', 'u1', 'Sanjay')
    assert snap['room_id'] == 'r1'
    assert snap['participants'][0]['name'] == 'Sanjay'
    state = {'asset': 'heart', 'focus': 'aorta'}
    snap = r.update_scene('r1', 'u1', state)
    assert snap['scene']['state']['focus'] == 'aorta'


def test_annotation_roundtrip_and_clear():
    r.join_room('r2', 'u1', 'A')
    snap = r.update_annotation('r2', 'u1', pointer={'x': 0.5, 'y': 0.4})
    assert snap['annotation']['pointer']['x'] == 0.5
    snap = r.update_annotation('r2', 'u1', stroke={'points': [{'x': 0.1, 'y': 0.2}, {'x': 0.2, 'y': 0.3}]})
    assert snap['annotation']['strokes']
    snap = r.update_annotation('r2', 'u1', clear=True)
    assert snap['annotation']['strokes'] == []
