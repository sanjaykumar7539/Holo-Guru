from core import observability as o


def test_record_and_snapshot_percentiles():
    o.reset()
    for ms in (10, 20, 30, 40, 50):
        o.record("unit.op", ms, slow_ms=45)
    snap = o.snapshot()
    m = snap["metrics"]["unit.op"]
    assert m["count"] == 5
    assert m["p50_ms"] == 30
    assert m["p95_ms"] == 50
    assert snap["recent_slow_operations"][0]["duration_ms"] == 50


def test_span_records_errors_and_success():
    o.reset()
    with o.span("unit.ok"):
        pass
    try:
        with o.span("unit.fail"):
            raise ValueError("x")
    except ValueError:
        pass
    snap = o.snapshot()["metrics"]
    assert snap["unit.ok"]["errors"] == 0
    assert snap["unit.fail"]["errors"] == 1
