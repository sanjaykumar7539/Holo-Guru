from core import realtime_voice as rv


def test_partial_rate_limit():
    rv.mark_final("sess-rt")
    ok, wait = rv.allow_partial("sess-rt", min_interval=10.0)
    assert ok is True
    ok2, wait2 = rv.allow_partial("sess-rt", min_interval=10.0)
    assert ok2 is False
    assert wait2 > 0
    rv.mark_final("sess-rt")
    ok3, _ = rv.allow_partial("sess-rt", min_interval=0.01)
    assert ok3 is True


def test_short_audio_heuristic():
    assert rv.is_short_audio_bytes(100) is True
    assert rv.is_short_audio_bytes(50_000) is False
