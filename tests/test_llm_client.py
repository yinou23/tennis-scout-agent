"""Unit tests for Week 9 Step 1 shared Gemini client.

Run from the project root ~/TSA:
    python -m pytest tests/test_llm_client.py -v

No test here makes a real API call. Tests inject a fake urlopen
that returns a fixed JSON body or raises a fake HTTPError /
URLError, and a fake sleep function that records waits instead
of really sleeping, so they run everywhere, with or without a
key, and in well under a second.
"""

import io
import json
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest

from llm_client import (
    DEFAULT_MODEL,
    FALLBACK_MODEL,
    call_gemini_text,
)


def _success_body(text="hello"):
    return json.dumps(
        {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    ).encode("utf-8")


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _http_error(code, retry_after=None):
    headers = {}
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError(
        "https://example.invalid/", code, "error", headers, io.BytesIO(b"{}")
    )


def _make_fake_urlopen(actions):
    """actions is a list, one per call, of ("ok", text) or ("error", exc).

    Returns (urlopen, calls), where calls records the model in
    each request URL, in order.
    """
    calls = []
    state = {"i": 0}

    def urlopen(request, timeout=60):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        calls.append(url)
        idx = state["i"]
        state["i"] += 1
        kind, value = actions[min(idx, len(actions) - 1)]
        if kind == "ok":
            return _FakeResponse(_success_body(value))
        raise value

    return urlopen, calls


def _make_sleep():
    waits = []

    def sleep_fn(seconds):
        waits.append(seconds)

    return sleep_fn, waits


def test_primary_model_is_3_5_flash_lite():
    assert DEFAULT_MODEL == "gemini-3.5-flash-lite"


def test_fallback_model_is_3_1_flash_lite_not_preview():
    assert FALLBACK_MODEL == "gemini-3.1-flash-lite"
    assert not FALLBACK_MODEL.endswith("-preview")


def test_success_first_try_no_sleep():
    urlopen, calls = _make_fake_urlopen([("ok", "hello")])
    sleep_fn, waits = _make_sleep()
    text, model_used = call_gemini_text(
        "prompt", "key", sleep_fn=sleep_fn, urlopen=urlopen
    )
    assert text == "hello"
    assert model_used == "gemini-3.5-flash-lite"
    assert len(calls) == 1
    assert calls[0].count("gemini-3.5-flash-lite") == 1
    assert waits == []


def test_503_once_then_success_on_primary():
    urlopen, calls = _make_fake_urlopen(
        [("error", _http_error(503)), ("ok", "hello")]
    )
    sleep_fn, waits = _make_sleep()
    text, model_used = call_gemini_text(
        "prompt", "key", sleep_fn=sleep_fn, urlopen=urlopen
    )
    assert text == "hello"
    assert model_used == "gemini-3.5-flash-lite"
    assert len(calls) == 2
    assert waits == [2.0]


def test_503_on_primary_twice_then_success_on_fallback():
    urlopen, calls = _make_fake_urlopen(
        [
            ("error", _http_error(503)),
            ("error", _http_error(503)),
            ("ok", "hello from fallback"),
        ]
    )
    sleep_fn, waits = _make_sleep()
    text, model_used = call_gemini_text(
        "prompt", "key", sleep_fn=sleep_fn, urlopen=urlopen
    )
    assert text == "hello from fallback"
    assert model_used == "gemini-3.1-flash-lite"
    assert len(calls) == 3
    assert "gemini-3.5-flash-lite" in calls[0]
    assert "gemini-3.5-flash-lite" in calls[1]
    assert "gemini-3.1-flash-lite" in calls[2]
    # One sleep between the two primary attempts, none after the
    # last primary attempt, none before the fallback first try.
    assert waits == [2.0]


def test_404_on_primary_goes_straight_to_fallback_no_retry():
    urlopen, calls = _make_fake_urlopen(
        [("error", _http_error(404)), ("ok", "hello from fallback")]
    )
    sleep_fn, waits = _make_sleep()
    text, model_used = call_gemini_text(
        "prompt", "key", sleep_fn=sleep_fn, urlopen=urlopen
    )
    assert text == "hello from fallback"
    assert model_used == "gemini-3.1-flash-lite"
    assert len(calls) == 2
    assert waits == []


def test_401_fails_at_once_no_fallback_no_sleep():
    urlopen, calls = _make_fake_urlopen([("error", _http_error(401))])
    sleep_fn, waits = _make_sleep()
    with pytest.raises(RuntimeError, match="llm_http_error:401"):
        call_gemini_text("prompt", "key", sleep_fn=sleep_fn, urlopen=urlopen)
    assert len(calls) == 1
    assert waits == []


def test_503_on_both_models_exhausts_and_raises_503():
    urlopen, calls = _make_fake_urlopen(
        [("error", _http_error(503))]
    )
    sleep_fn, waits = _make_sleep()
    with pytest.raises(RuntimeError, match="llm_http_error:503"):
        call_gemini_text("prompt", "key", sleep_fn=sleep_fn, urlopen=urlopen)
    # 2 attempts per model, 2 models.
    assert len(calls) == 4
    assert waits == [2.0, 2.0]


def test_retry_after_header_is_used_instead_of_default():
    urlopen, calls = _make_fake_urlopen(
        [("error", _http_error(503, retry_after=5)), ("ok", "hello")]
    )
    sleep_fn, waits = _make_sleep()
    text, _model = call_gemini_text(
        "prompt", "key", sleep_fn=sleep_fn, urlopen=urlopen
    )
    assert text == "hello"
    assert waits == [5.0]


def test_url_error_is_retryable_then_success():
    urlopen, calls = _make_fake_urlopen(
        [("error", urllib.error.URLError("timed out")), ("ok", "hello")]
    )
    sleep_fn, waits = _make_sleep()
    text, model_used = call_gemini_text(
        "prompt", "key", sleep_fn=sleep_fn, urlopen=urlopen
    )
    assert text == "hello"
    assert model_used == "gemini-3.5-flash-lite"
    assert waits == [2.0]
