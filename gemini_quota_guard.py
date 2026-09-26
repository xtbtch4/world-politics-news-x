from __future__ import annotations

import json
import os
import re

import requests

_ALLOWED_MODELS = {
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
}
_MAX_REAL_REQUESTS_PER_MODEL_PER_RUN = 1
_KEY_NAMES = ["GEMINI_API_KEY", *[f"GEMINI_API_KEY_{i}" for i in range(2, 11)]]
_real_calls: dict[str, int] = {}
_original_post = requests.post


def _run_number() -> int:
    try:
        return int(os.getenv("GITHUB_RUN_NUMBER", "0") or 0)
    except ValueError:
        return 0


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


def guarded_post(*args, **kwargs):
    url = str(args[0] if args else kwargs.get("url", ""))
    if "generativelanguage.googleapis.com" not in url:
        return _original_post(*args, **kwargs)

    match = re.search(r"/models/([^/:]+)", url)
    model = match.group(1) if match else os.getenv("GEMINI_MODEL", "")
    if model not in _ALLOWED_MODELS:
        print(f"Gemini quota guard blocked disallowed model: {model}")
        return _blocked_response(url, model or "unknown", "model is not allowed")

    if os.getenv("DRY_RUN", "false").lower() in {"1", "true", "yes", "on"}:
        print(f"Gemini quota guard: DRY_RUN, no real request sent for {model}")
        return _blocked_response(url, model, "dry run does not spend quota")

    count = _real_calls.get(model, 0)
    if count >= _MAX_REAL_REQUESTS_PER_MODEL_PER_RUN:
        print(f"Gemini quota guard: local per-run cap reached for {model}; request not sent")
        return _blocked_response(url, model, "per-run safety cap reached")

    _real_calls[model] = count + 1
    print(
        f"Gemini quota guard: real request {count + 1}/{_MAX_REAL_REQUESTS_PER_MODEL_PER_RUN} "
        f"for {model}"
    )
    return _original_post(*args, **kwargs)


_rotate_configured_keys()
requests.post = guarded_post
print(
    "Gemini quota guard active: max 1 real request per model per workflow run; "
    "DRY_RUN uses zero Gemini quota"
)
