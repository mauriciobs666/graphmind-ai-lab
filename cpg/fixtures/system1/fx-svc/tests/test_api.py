from api.health import build_router


def test_health_reports_ok():
    route = build_router().routes[0]
    assert route.endpoint()["status"] == "ok"
