from __future__ import annotations

import os
import re
import sys

import requests

GEMINI_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)

# Only these two Gemini models are allowed. 3.5 Flash Lite has priority.
os.environ["GEMINI_MODEL"] = GEMINI_MODELS[0]
os.environ["GEMINI_FALLBACK_MODEL"] = GEMINI_MODELS[1]
os.environ["DISABLE_GEMINI"] = "false"

import bot


# Canonicalize common headline wording so the same event reported by different
# outlets is deduplicated even when wording or demonyms differ.
_EVENT_TOKEN_ALIASES = {
    "australian": "australia",
    "australians": "australia",
    "ukrainian": "ukraine",
    "ukrainians": "ukraine",
    "russian": "russia",
    "russians": "russia",
    "iranian": "iran",
    "iranians": "iran",
    "israeli": "israel",
    "israelis": "israel",
    "palestinian": "palestine",
    "palestinians": "palestine",
    "american": "usa",
    "americans": "usa",
    "british": "uk",
    "german": "germany",
    "germans": "germany",
    "french": "france",
    "chinese": "china",
    "canadian": "canada",
    "canadians": "canada",
    "european": "europe",
    "hacked": "hack",
    "hacks": "hack",
    "hacking": "hack",
    "hackers": "hack",
    "breach": "hack",
    "breached": "hack",
    "breaches": "hack",
    "website": "website",
    "websites": "website",
    "site": "website",
    "sites": "website",
    "governmental": "government",
    "attacked": "attack",
    "attacks": "attack",
    "strikes": "strike",
    "struck": "strike",
    "sanctions": "sanction",
    "sanctioned": "sanction",
    "elections": "election",
    "tariffs": "tariff",
}

_EVENT_NOISE_TOKENS = set(getattr(bot, "EVENT_KEY_STOPWORDS", set())) | {
    "says", "said", "tells", "told", "calls", "called", "calling",
    "warns", "warned", "urges", "urged", "remarks", "remark",
    "criticises", "criticise", "criticizes", "criticized", "slams",
    "expresses", "expressed", "concern", "concerns", "amid",
}


def _canonical_event_tokens(value: str) -> list[str]:
    result: list[str] = []
    for raw in re.findall(r"[a-z0-9]{3,}", (value or "").casefold()):
        token = _EVENT_TOKEN_ALIASES.get(raw, raw)
        if token in _EVENT_NOISE_TOKENS:
            continue
        if token not in result:
            result.append(token)
    return result


def stronger_normalize_event_key(value: str) -> str:
    return " ".join(_canonical_event_tokens(value))[:240]


def stronger_event_similarity(a: str, b: str) -> float:
    tokens_a = set(_canonical_event_tokens(a))
    tokens_b = set(_canonical_event_tokens(b))
    if not tokens_a or not tokens_b:
        return 0.0

    common = tokens_a & tokens_b
    score = len(common) / min(len(tokens_a), len(tokens_b))

    # Cross-source headlines often add a person's name or a reaction phrase.
    # Four matching canonical event tokens are strong evidence of the same
    # underlying event, while still requiring substantial factual overlap.
    if len(common) >= 4:
        score = max(score, 0.65)
    return score


bot.normalize_event_key = stronger_normalize_event_key
bot.event_similarity = stronger_event_similarity


# bot.py has its own retry loop. We allow only one real Gemini HTTP request per
# model/key combination; internal retries reuse the same failure locally.
_original_requests_post = bot.requests.post
_original_make_post = bot.make_post
_gemini_attempted = False
_gemini_cached_response = None
_gemini_cached_exception = None
_gemini_last_status: int | None = None
_gemini_last_body = ""


def reset_gemini_attempt_state() -> None:
    global _gemini_attempted, _gemini_cached_response, _gemini_cached_exception
    global _gemini_last_status, _gemini_last_body
    _gemini_attempted = False
    _gemini_cached_response = None
    _gemini_cached_exception = None
    _gemini_last_status = None
    _gemini_last_body = ""


def post_with_gemini_error_logging(*args, **kwargs):
    global _gemini_attempted, _gemini_cached_response, _gemini_cached_exception
    global _gemini_last_status, _gemini_last_body

    url = str(args[0] if args else kwargs.get("url", ""))
    is_gemini = "generativelanguage.googleapis.com" in url

    if is_gemini and _gemini_attempted:
        if _gemini_cached_exception is not None:
            raise _gemini_cached_exception
        if _gemini_cached_response is not None:
            return _gemini_cached_response

    if is_gemini:
        _gemini_attempted = True

    try:
        response = _original_requests_post(*args, **kwargs)
    except requests.RequestException as exc:
        if is_gemini:
            _gemini_cached_exception = exc
            _gemini_last_status = None
            _gemini_last_body = str(exc)
        raise

    if is_gemini:
        _gemini_last_status = response.status_code
        _gemini_last_body = bot.clean_text(response.text)
        if response.status_code not in {200, 201}:
            bot.LOG.warning(
                "Gemini API response body (HTTP %s): %s",
                response.status_code,
                _gemini_last_body[:3000] or "<empty body>",
            )
            # Cache every failed response so bot.py cannot spend more quota on
            # its internal retries for the same key/model combination.
            _gemini_cached_response = response

    return response


bot.requests.post = post_with_gemini_error_logging


def gemini_api_keys() -> list[str]:
    values: list[str] = []

    # Existing key plus optional individually named keys.
    for name in ["GEMINI_API_KEY", *[f"GEMINI_API_KEY_{i}" for i in range(2, 11)]]:
        value = os.getenv(name, "").strip()
        if value:
            values.append(value)

    # Optional scalable secret containing extra keys separated by newlines,
    # commas, semicolons or spaces.
    raw = os.getenv("GEMINI_API_KEYS", "").strip()
    if raw:
        values.extend(value for value in re.split(r"[\s,;]+", raw) if value)

    # Deduplicate without exposing keys in logs.
    return list(dict.fromkeys(values))


def run_gemini_once(
    story: bot.Story,
    api_key: str,
    model: str,
    key_index: int,
    key_count: int,
) -> tuple[bot.RenderedPost | None, int | None, str]:
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GEMINI_MODEL"] = model
    os.environ["GEMINI_FALLBACK_MODEL"] = model
    reset_gemini_attempt_state()

    bot.LOG.info(
        "Gemini attempt: model=%s key=%d/%d",
        model,
        key_index,
        key_count,
    )

    original_sleep = bot.time.sleep
    bot.time.sleep = lambda _seconds: None
    try:
        post = _original_make_post(story)
    finally:
        bot.time.sleep = original_sleep

    return post, _gemini_last_status, _gemini_last_body


def gemini_translation_with_rotation(story: bot.Story) -> bot.RenderedPost | None:
    keys = gemini_api_keys()
    if not keys:
        bot.LOG.warning("No Gemini API keys are configured")
        return None

    bot.LOG.info(
        "Gemini rotation configured with %d key(s); model priority: %s -> %s",
        len(keys),
        GEMINI_MODELS[0],
        GEMINI_MODELS[1],
    )

    for model in GEMINI_MODELS:
        for index, api_key in enumerate(keys, start=1):
            post, status, body = run_gemini_once(
                story,
                api_key,
                model,
                index,
                len(keys),
            )
            if post is not None:
                bot.LOG.info(
                    "Gemini succeeded: model=%s key=%d/%d",
                    model,
                    index,
                    len(keys),
                )
                return post

            body_lower = (body or "").lower()
            key_or_quota_problem = (
                status in {401, 403, 429}
                or "resource_exhausted" in body_lower
                or "quota" in body_lower
                or "rate limit" in body_lower
                or "api_key_invalid" in body_lower
                or "invalid api key" in body_lower
            )

            if key_or_quota_problem:
                if index < len(keys):
                    bot.LOG.warning(
                        "Gemini key %d/%d exhausted or unavailable for %s; switching to next key",
                        index,
                        len(keys),
                        model,
                    )
                continue

            # Capacity/server failures are normally model-wide, so do not burn
            # the same request against every key. Move to the second allowed model.
            if status in {500, 502, 503, 504} or status is None:
                bot.LOG.warning(
                    "Gemini model %s temporarily unavailable; trying next allowed model",
                    model,
                )
                break

            # HTTP 200 with unusable output or another non-quota error is not
            # improved by changing keys; try the second allowed model instead.
            bot.LOG.warning(
                "Gemini model %s returned unusable output/status %s; trying next allowed model",
                model,
                status,
            )
            break

    return None


def openai_output_text(payload: dict) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    parts: list[str] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts).strip()


def openai_translation_fallback(story: bot.Story) -> bot.RenderedPost | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        bot.LOG.info("OPENAI_API_KEY is not configured; OpenAI fallback skipped")
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna").strip() or "gpt-5.6-luna"
    evidence, image_url, video_url = bot.fetch_article_context(story)
    source_text = bot.clean_text(evidence or story.summary)
    if not source_text:
        source_text = story.summary

    prompt = f"""Translate this news item into natural, fluent Russian for a Telegram news channel.
Preserve the meaning, names, numbers and attribution exactly. Do not invent facts, opinions or context.
Use neutral news style. Do not repeat the headline at the start of the summary.
Return exactly these two tags and nothing else:
<TITLE>Russian headline</TITLE>
<SUMMARY>Russian summary, concise but informative</SUMMARY>

SOURCE TITLE:
{story.title}

SOURCE TEXT:
{source_text[:5500]}
"""

    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "input": prompt,
                "max_output_tokens": 1200,
            },
            timeout=40,
        )
    except requests.RequestException as exc:
        bot.LOG.warning("OpenAI fallback network error: %s", str(exc).splitlines()[0])
        return None

    if response.status_code not in {200, 201}:
        body = bot.clean_text(response.text)
        bot.LOG.warning(
            "OpenAI fallback error HTTP %s: %s",
            response.status_code,
            body[:1200] or "<empty body>",
        )
        return None

    try:
        payload = response.json()
    except ValueError:
        bot.LOG.warning("OpenAI fallback returned invalid JSON")
        return None

    text = openai_output_text(payload)
    title = bot.tagged_value(text, "TITLE")
    summary = bot.tagged_value(text, "SUMMARY")
    if not title or not summary or not bot.has_cyrillic(title) or not bot.has_cyrillic(summary):
        bot.LOG.warning("OpenAI fallback returned unusable Russian output")
        return None

    event_key = bot.normalize_event_key(story.title)
    if len(event_key.split()) < 3:
        event_key = f"source-story-{story.fingerprint}"

    bot.LOG.info("OpenAI Russian translation fallback succeeded: %s", story.title)
    return bot.RenderedPost(
        title=title[:500].rstrip(),
        summary=summary[:2200].rstrip(),
        quote_line="",
        source_line=f"Источник: {story.source}\n{story.url}",
        event_key=event_key,
        image_url=image_url or story.image_url,
        video_url=video_url or story.video_url,
    )


def make_post_with_source_fallback(story: bot.Story) -> bot.RenderedPost:
    post = gemini_translation_with_rotation(story)
    if post is not None:
        return post

    # All Gemini model/key combinations failed: try OpenAI once, then original.
    openai_post = openai_translation_fallback(story)
    if openai_post is not None:
        return openai_post

    evidence, image_url, video_url = bot.fetch_article_context(story)
    summary = bot.clean_text(story.summary or evidence)
    if not summary:
        summary = "Full details are available at the source link below."

    event_key = bot.normalize_event_key(story.title)
    if len(event_key.split()) < 3:
        event_key = f"source-story-{story.fingerprint}"

    bot.LOG.warning(
        "Gemini rotation and OpenAI unavailable or unusable; publishing source language: %s",
        story.title,
    )

    return bot.RenderedPost(
        title=story.title[:500].rstrip(),
        summary=summary[:2200].rstrip(),
        quote_line="",
        source_line=f"Источник: {story.source}\n{story.url}",
        event_key=event_key,
        image_url=image_url or story.image_url,
        video_url=video_url or story.video_url,
    )


bot.make_post = make_post_with_source_fallback


if __name__ == "__main__":
    sys.exit(bot.main())