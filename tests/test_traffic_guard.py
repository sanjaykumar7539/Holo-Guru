from core.traffic_guard import TrafficGuard

def test_guard_rejects_when_full_without_waiting_room():
    g = TrafficGuard(max_concurrent=1, max_waiting=0, wait_timeout=0.01)
    token = g.acquire()
    assert token is not None
    assert g.acquire() is None
    snap = g.snapshot()
    assert snap["active"] == 1
    assert snap["rejected"] == 1
    g.release(token)
    assert g.snapshot()["active"] == 0
