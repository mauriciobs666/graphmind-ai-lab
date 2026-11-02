from fx_svc.api.health import health


def test_health_reports_ok():
    assert health()["status"] == "ok"
