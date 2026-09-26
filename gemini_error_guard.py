from __future__ import annotations

import runner

_original_run_gemini_once = runner.run_gemini_once


def run_gemini_once_safe(*args, **kwargs):
    try:
        return _original_run_gemini_once(*args, **kwargs)
    except RuntimeError as exc:
        status = getattr(runner, "_gemini_last_status", None)
        body = getattr(runner, "_gemini_last_body", "") or str(exc)
        text = str(exc).casefold()
        gemini_error = (
            "gemini api error" in text
            or "account_state_invalid" in text
            or "resource_exhausted" in text
            or status in {401, 403, 429, 500, 502, 503, 504}
        )
        if not gemini_error:
            raise

        runner.bot.LOG.warning(
            "Gemini key/model attempt failed without stopping workflow: status=%s; moving to next key/model",
            status,
        )
        return None, status, body


runner.run_gemini_once = run_gemini_once_safe
