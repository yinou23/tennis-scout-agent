"""Tests for the CORS middleware in scripts/api.py.

Run from the project root ~/TSA:
    python -m pytest tests/test_api_cors.py -v

No network, no database, no API key. The app is built with
create_app() and exercised through TestClient. Regression test:
the Week 9 Step 3 rewrite of api.py was based on the pre-CORS
version and dropped the Week 5 Step 2 CORSMiddleware, so the
browser's preflight OPTIONS to /report returned 405 and the
web page could not POST a report. The middleware must allow
only http://127.0.0.1:3000 and http://localhost:3000.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import api
from fastapi.testclient import TestClient


def _client():
    return TestClient(api.create_app())


def _preflight(client, origin):
    return client.options(
        "/report",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )


def test_preflight_from_localhost_3000_allowed():
    res = _preflight(_client(), "http://localhost:3000")
    assert res.status_code == 200
    assert res.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_preflight_from_127_0_0_1_3000_allowed():
    res = _preflight(_client(), "http://127.0.0.1:3000")
    assert res.status_code == 200
    assert res.headers["access-control-allow-origin"] == "http://127.0.0.1:3000"


def test_preflight_from_other_origin_rejected():
    res = _preflight(_client(), "http://evil.example")
    assert res.status_code == 400
    assert "access-control-allow-origin" not in res.headers


def test_health_response_carries_allow_origin_for_localhost():
    res = _client().get("/health", headers={"Origin": "http://localhost:3000"})
    assert res.status_code == 200
    assert res.headers["access-control-allow-origin"] == "http://localhost:3000"
