from __future__ import annotations

import json
import os
import re

import requests

_ALLOWED_MODELS = {
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
}
# Safety cap is per Google project/key AND per model, not global per model.
# With a 10-minute workflow cadence this is at most 144 requests/day to any
# single key for each Lite model, comfortably below a 500 RPD model limit.
_MAX_REAL_REQUESTS_PER_KEY_MODEL_PER_RUN = 1
_KEY_NAMES = ["GEMINI_API_KEY", *[f"GEMINI_API_KEY_{i}" for i in range(2, 11)]]
# Temporarily disabled because Google returns 401 ACCOUNT_STATE_INVALID
# (bound service account/account is deleted or disabled). Keep the GitHub Secrets
# untouched so they can be re-enabled after the Google accounts are restored.
_DISABLED_KEY_NAMES = {"GEMINI_API_KEY_3", "GEMINI_API_KEY_4"}
_real_calls: dict[tuple[str, str], int] = {}
_original_post = requests.post


def _run_number() -> int:
    try:
        return int(os.getenv("GITHUB_RUN_NUMBER", "0") or 0)
    except ValueError:
        return 0


def _disable_blocked_keys() -> None:
    for name in sorted(_DISABLED_KEY_NAMES):
        if os.environ.pop(name, None):
            print(f"Gemini quota guard: temporarily disabled {name}")


def _rotate_configured_keys() -> None:
    populated = [(name, os.getenv(name, "").strip()) for name in _KEY_NAMES]
    populated = [(name, value) for name, value in populated if value]
    if populated:
        values = [value for _, value in populated]
        shift = _run_number() % len(values)
        values = values[shift:] + values[:shift]
        for (name, _), value in zip(populated, values):
            os.environ[name] = value
        print(f"Gemini quota guard: rotating {len(values)} named key(s); starting key slot {shift + 1}/{len(values)}")

    raw = os.getenv("GEMINI_API_KEYS", "").strip()
    if raw:
        values = [value for value in re.split(r"[\s,;]+", raw) if value]
        if values:
            shift = _run_number() % len(values)
            values = values[shift:] + values[:shift]
            os.environ["GEMINI_API_KEYS"] = "\n".join(values)
            if not populated:
                print(f"Gemini quota guard: rotating {len(values)} GEMINI_API_KEYS value(s); starting slot {shift + 1}/{len(values)}")


def _blocked_response(url: str, model: str, reason: str) -> requests.Response:
    response = requests.Response()
    response.status_code = 429
    response.url = url
    response.headers["Content-Type"] = "application/json"
    response._content = json.dumps({
        "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "message": f"Local Gemini quota guard: {reason} for {model}",
        }
    }).encode("utf-8")
    return response


def _request_api_key(kwargs: dict) -> str:
    headers = kwargs.get("headers") or {}
    for name, value in headers.items():
        if str(name).casefold() == "x-goog-api-key":
            return str(value or "").strip()
    return ""


def _key_slot(api_key: str) -> str:
    if not api_key:
        return "unknown"
    for name in _KEY_NAMES:
        if os.getenv(name, "").strip() == api_key:
            return name
    raw = [value for value in re.split(r"[\s,;]+", os.getenv("GEMINI_API_KEYS", "")) if value]
    for index, value in enumerate(raw, start=1):
        if value == api_key:
            return f"GEMINI_API_KEYS[{index}]"
    return "unmapped"


def guarded_post(*args, **kwargs):
    url = str(args[0] if args else kwargs.get("url", ""))
    if "generativelanguage.googleapis.com" not in url:
        return _original_post(*args, **kwargs)

    # The Interactions endpoint carries the model in the JSON body rather than URL.
    body = kwargs.get("json") or {}
    match = re.search(r"/models/([^/:]+)", url)
    model = str(body.get("model") or (match.group(1) if match else "") or os.getenv("GEMINI_MODEL", ""))
    if model not in _ALLOWED_MODELS:
        print(f"Gemini quota guard blocked disallowed model: {model}")
        return _blocked_response(url, model or "unknown", "model is not allowed")

    if os.getenv("DRY_RUN", "false").lower() in {"1", "true", "yes", "on"}:
        print(f"Gemini quota guard: DRY_RUN, no real request sent for {model}")
        return _blocked_response(url, model, "dry run does not spend quota")

    api_key = _request_api_key(kwargs)
    slot = _key_slot(api_key)
    counter_key = (api_key or slot, model)
    count = _real_calls.get(counter_key, 0)
    if count >= _MAX_REAL_REQUESTS_PER_KEY_MODEL_PER_RUN:
        print(f"Gemini quota guard: per-key cap reached for {model} on {slot}; request not sent")
        return _blocked_response(url, model, f"per-key safety cap reached on {slot}")

    _real_calls[counter_key] = count + 1
    print(
        f"Gemini quota guard: real request {count + 1}/{_MAX_REAL_REQUESTS_PER_KEY_MODEL_PER_RUN} "
        f"for {model} on {slot}"
    )
    return _original_post(*args, **kwargs)


_disable_blocked_keys()
_rotate_configured_keys()
requests.post = guarded_post
print(
    "Gemini quota guard active: max 1 real request per key per Lite model per workflow run; "
    "GEMINI_API_KEY_3 and GEMINI_API_KEY_4 disabled; DRY_RUN uses zero Gemini quota"
)
