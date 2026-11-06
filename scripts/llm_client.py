"""Week 9 Step 1: shared Gemini client with retry and fallback model.

Purpose: fix the failure the local demo hit on 2026-11-05, where one
/report call failed with llm_http_error:503 and the user had to
curl the same question again by hand. That failure was transient,
the model was overloaded, not a code bug, not a bad key, and not a
bad question. This module makes the server do the retry itself,
instead of making the user do it.

What this file does, and only this:
  1. Retry the same model on transient errors only.
     Retryable: HTTP 429, 500, 502, 503, 504, and network errors
     (URLError, including timeouts). Not retryable: 400, 401, 403,
     and any other 4xx, which a retry will not fix.
     Per model, at most 2 attempts (1 call plus 1 retry), with a
     2 second wait before the retry. If the server sends a
     Retry-After header, that value is used instead, capped at
     10 seconds. The wait is injectable in tests, so tests do not
     really sleep.
  2. Fall back to a second, pre-verified model when the first
     model is exhausted or is not found.
     Primary: gemini-3.5-flash-lite, the model verified working
     on the Mac in Week 3 and used by the Week 4 writer and the
     Week 5 API demo.
     Fallback: gemini-3.1-flash-lite, exact model id, no
     -preview suffix. The preview variant shut down on
     2026-05-25 and is a different id, do not use it.
     A 404 on the primary goes straight to the fallback, with
     no retry on the primary, because a missing model will not
     appear on a retry. A 401 or 403 fails at once, with no
     fallback, because the same key would fail on the fallback
     too. A retryable error retries the primary first, then
     tries the fallback with the same 2 attempt rule.
     Total worst case per LLM call: 4 calls, 2 per model, with
     at most 2 seconds of sleep per model unless Retry-After
     says otherwise. That bound is deliberate. More attempts
     would make a report take a minute, and the demo showed
     that one retry is usually enough.
  3. Report which model actually succeeded.
     call_gemini_text returns (text, model_used). The writer
     records model_used in its result, so the API llm_model
     field shows the fallback when the fallback wrote the
     report, not the primary. The planner returns text only,
     for backward compatibility, because the graph records
     planner_source, not a planner model name.

Fallback model verification, checked 2026-11-05 / 2026-11-06,
before this file was written, per the standing rule that a new
model must be verified from at least two independent sources
covering exact name, availability to new keys, free-tier quota,
and current serving status in the last 7 days:
  - Exact name and free tier, primary source: Google Gemini
    Developer API pricing page, crawled 2026-11-06.
    gemini-3.1-flash-lite, input Free of charge, output Free of
    charge, paid $0.25 / $1.50 per 1M tokens. Described as a
    cost-efficient model, optimized for high-volume agentic
    tasks, translation, and simple data processing, which is
    what the planner and the writer do.
  - Free-tier quota, independent sources: scriptbyai and
    iimagined, both reading the Google rate limits page in
    October 2026, list gemini-3.1-flash-lite at 10 requests
    per minute, 250,000 input tokens per minute, 500 requests
    per day, the same quota as gemini-3.5-flash-lite. Google
    does not publish a fixed per-model table now, and the
    active limits for a project are shown in AI Studio, so
    500 per day is a guide, not a guarantee.
  - Availability to new keys: Gemini 2.5 models are excluded
    as a fallback, because Google and independent sources say
    their API access is limited to developers who previously
    used them, and new projects should use current Gemini 3.x
    models. gemini-3.1-flash-lite is a 3.x model, so it is in
    the current line. gemini-2.5-flash returned 404 for new
    keys in October 2026, which is why it is not used here.
  - Current serving, last 7 days, independent observed data:
    infraone, observed 2026-09-06 to 2026-11-05, 7,636
    successful Gemini free-tier calls, with gemini-3.1-flash-lite
    answering 2,196 of them and gemini-3.5-flash-lite 2,844.
    Overloaded 503 responses in that set had a median time to
    next success of 0.0 hours, p90 0.8 hours, which is why a
    short retry is the right first step. A separate project
    decision record dated 2026-09-24 also records
    gemini-3.1-flash-lite as ok in 4.98 seconds, with a retry
    policy of 3 attempts total on 429 / 503 / timeouts.
  - Sunset warning, not hidden: Google lists May 7, 2027 as
    the earliest shutdown date for gemini-3.1-flash-lite, and
    recommends gemini-3.5-flash-lite as the replacement. That
    is why it is only a fallback, never the primary, and why
    the primary stays gemini-3.5-flash-lite. If the fallback
    is still in use in spring 2027, it must be re-verified
    or replaced before that date.

What this file does NOT do:
  It does not change the locked probability model, any tool,
  any verifier rule, or any writer rule. It does not fall
  back to the template report. A failed LLM call after all
  retries and the fallback still returns an error, and the
  API still returns 422 with no final_text, exactly as before.
  It does not retry verification failures. A verification
  failure means the LLM wrote a report that missed a required
  number, as in the 2026-11-05 demo, and retrying the same
  writer call is a separate step, not covered here. That step
  would re-run the writer node, not just the HTTP call, and
  it is not in this change.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python -m pytest tests/test_llm_client.py -v
  No test here makes a real API call. Tests inject a fake
  urlopen, so they run everywhere, with or without a key.
  A real fallback demo is a manual run on the Mac with a key
  in .env, and it must show llm_model as the fallback only
  when the primary actually failed, otherwise the primary.
"""

import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.1-flash-lite"
RETRYABLE_CODES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS_PER_MODEL = 2
RETRY_DELAY_SECONDS = 2.0
MAX_RETRY_AFTER_SECONDS = 10.0


def _retry_after_seconds(exc):
    """Return the Retry-After wait in seconds, capped, or None."""
    headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("Retry-After")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        value = float(str(raw).strip())
    except ValueError:
        return None
    if value < 0:
        return None
    return min(value, MAX_RETRY_AFTER_SECONDS)


def _call_once(prompt, api_key, model, urlopen):
    """Do one generateContent call. Return text, or raise RuntimeError."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err = RuntimeError(f"llm_http_error:{exc.code}")
        err.retry_after = _retry_after_seconds(exc)
        raise err from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"llm_url_error:{type(exc).__name__}") from exc
    try:
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("llm_bad_response") from exc
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("llm_empty_response")
    return text.strip()


def _http_code(exc):
    """Extract the HTTP code from a llm_http_error RuntimeError, or None."""
    message = str(exc)
    prefix = "llm_http_error:"
    if not message.startswith(prefix):
        return None
    try:
        return int(message[len(prefix):].strip())
    except ValueError:
        return None


def _models_to_try(model, fallback_model):
    """Return the deduplicated model list, primary first, then fallback."""
    primary = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
    if fallback_model is None:
        fallback = os.environ.get("LLM_FALLBACK_MODEL") or FALLBACK_MODEL
    else:
        fallback = fallback_model
    models = [primary]
    if fallback and fallback != primary:
        models.append(fallback)
    return models


def call_gemini_text(prompt, api_key, model=None, fallback_model=None,
                     sleep_fn=None, urlopen=None):
    """Call Gemini with retry and fallback. Return (text, model_used).

    prompt is the full prompt text. api_key is the developer key.
    model is the primary model, default gemini-3.5-flash-lite.
    fallback_model is the fallback model, default
    gemini-3.1-flash-lite, or "" to disable the fallback.
    sleep_fn defaults to time.sleep, urlopen defaults to
    urllib.request.urlopen. Tests inject both.
    On failure after all attempts, raises the last RuntimeError,
    with the same llm_http_error:<code> message as before, so
    existing callers that catch Exception still work.
    """
    if sleep_fn is None:
        sleep_fn = time.sleep
    if urlopen is None:
        urlopen = urllib.request.urlopen
    models = _models_to_try(model, fallback_model)
    last_error = None
    for model_name in models:
        for attempt in range(1, MAX_ATTEMPTS_PER_MODEL + 1):
            try:
                text = _call_once(prompt, api_key, model_name, urlopen)
                return text, model_name
            except RuntimeError as exc:
                last_error = exc
                code = _http_code(exc)
                is_url_error = str(exc).startswith("llm_url_error:")
                # 404 on this model: do not retry it, go to fallback.
                if code == 404:
                    break
                # Non-retryable HTTP error, and non-URL errors other
                # than a bad or empty response that came from a
                # transient failure, fail at once, with no fallback.
                # 401 and 403 fail on every model with the same key.
                if code is not None and code not in RETRYABLE_CODES:
                    raise
                if code is None and not is_url_error:
                    raise
                # Retryable, and this was not the last attempt for
                # this model: wait, then try the same model again.
                if attempt < MAX_ATTEMPTS_PER_MODEL:
                    wait = getattr(exc, "retry_after", None)
                    if wait is None:
                        wait = RETRY_DELAY_SECONDS
                    sleep_fn(wait)
                    continue
                # Last attempt for this model: move to the fallback,
                # if there is one. No sleep after the last attempt.
                break
    raise last_error
