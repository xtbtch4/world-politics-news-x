from __future__ import annotations

import os
import sys

import requests

# Temporary Gemini configuration: use Flash Lite and keep Gemini enabled.
os.environ["GEMINI_MODEL"] = "gemini-3.5-flash-lite"
os.environ["GEMINI_FALLBACK_MODEL"] = "gemini-3.5-flash-lite"
os.environ["DISABLE_GEMINI"] = "false"

import bot


# Keep Gemini failures visible in Actions logs and allow only one real Gemini
# request per story. bot.py may retry internally, but repeated attempts reuse the
# first failure locally so they do not consume additional Gemini quota.
_original_requests_post = bot.requests.post
_gemini_attempted = False
_gemini_cached_response = None
_gemini_cached_exception = None


def post_with_gemini_error_logging(*args, **kwargs):
    global _gemini_attempted, _gemini_cached_response, _gemini_cached_exception

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
        raise

    if is_gemini:
        if response.status_code not in {200, 201}:
            body = bot.clean_text(response.text)
            bot.LOG.warning(
                "Gemini API response body (HTTP %s): %s",
                response.status_code,
                body[:3000] or "<empty body>",
            )
        if response.status_code in {429, 500, 502, 503, 504}:
            _gemini_cached_response = response

    return response


bot.requests.post = post_with_gemini_error_logging

_original_make_post = bot.make_post


def split_text(text: str, limit: int) -> list[str]:
    text = bot.clean_text(text)
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split_at = max(
            remaining.rfind(". ", 0, limit),
            remaining.rfind("! ", 0, limit),
            remaining.rfind("? ", 0, limit),
        )
        if split_at < max(120, limit // 3):
            split_at = remaining.rfind(" ", 0, limit)
        if split_at < 1:
            split_at = limit
        else:
            split_at += 1
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    return chunks


def translate_google(text: str) -> str:
    translated_parts: list[str] = []
    for chunk in split_text(text, 1400):
        response = requests.get(
            "https://translate.googleapis.com/translate_a/single",
            params={
                "client": "gtx",
                "sl": "auto",
                "tl": "ru",
                "dt": "t",
                "q": chunk,
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        translated = "".join(
            part[0]
            for part in (payload[0] or [])
            if isinstance(part, list) and part and isinstance(part[0], str)
        )
        translated = bot.clean_text(translated)
        if not translated:
            raise RuntimeError("Google Translate returned empty text")
        translated_parts.append(translated)
    return bot.clean_text(" ".join(translated_parts))


def translate_mymemory(text: str) -> str:
    translated_parts: list[str] = []
    for chunk in split_text(text, 420):
        response = requests.get(
            "https://api.mymemory.translated.net/get",
            params={"q": chunk, "langpair": "en|ru"},
            headers={"User-Agent": "WorldPoliticsNewsBot/1.0"},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        status = payload.get("responseStatus", 200)
        if str(status) != "200":
            raise RuntimeError(
                f"MyMemory error {status}: {payload.get('responseDetails', '')}"
            )
        translated = bot.clean_text(
            str((payload.get("responseData") or {}).get("translatedText") or "")
        )
        if not translated:
            raise RuntimeError("MyMemory returned empty text")
        translated_parts.append(translated)
    return bot.clean_text(" ".join(translated_parts))


def translate_to_russian(text: str) -> str:
    text = bot.clean_text(text)
    if not text or bot.has_cyrillic(text):
        return text

    errors: list[str] = []
    for provider_name, provider in (
        ("Google Translate", translate_google),
        ("MyMemory", translate_mymemory),
    ):
        try:
            translated = provider(text)
            if translated and bot.has_cyrillic(translated):
                bot.LOG.info("Fallback translation succeeded via %s", provider_name)
                return translated
            errors.append(f"{provider_name}: no Cyrillic output")
        except (requests.RequestException, ValueError, TypeError, RuntimeError) as exc:
            errors.append(f"{provider_name}: {str(exc).splitlines()[0]}")

    raise RuntimeError("; ".join(errors))


def make_post_with_source_fallback(story: bot.Story) -> bot.RenderedPost:
    global _gemini_attempted, _gemini_cached_response, _gemini_cached_exception

    gemini_disabled = os.getenv("DISABLE_GEMINI", "false").lower() in {"1", "true", "yes", "on"}
    if gemini_disabled:
        bot.LOG.info("Gemini disabled by DISABLE_GEMINI; using translation fallback")
        post = None
    else:
        _gemini_attempted = False
        _gemini_cached_response = None
        _gemini_cached_exception = None
        original_sleep = bot.time.sleep
        bot.time.sleep = lambda _seconds: None
        try:
            post = _original_make_post(story)
        finally:
            bot.time.sleep = original_sleep
            _gemini_attempted = False
            _gemini_cached_response = None
            _gemini_cached_exception = None

    if post is not None:
        return post

    # If Gemini is disabled, rate-limited or temporarily unavailable, translate
    # source material with independent translation services instead of dropping it.
    evidence, image_url, video_url = bot.fetch_article_context(story)
    source_summary = bot.clean_text(story.summary or evidence)
    if not source_summary:
        source_summary = "Full details are available at the source link below."

    title = story.title
    summary = source_summary
    translated = False
    try:
        translated_title = translate_to_russian(story.title)
        translated_summary = translate_to_russian(source_summary)
        if bot.has_cyrillic(translated_title) and bot.has_cyrillic(translated_summary):
            title = translated_title
            summary = translated_summary
            translated = True
    except (ValueError, TypeError, RuntimeError) as exc:
        bot.LOG.warning(
            "Fallback translation failed; publishing source language: %s (%s)",
            story.title,
            str(exc).splitlines()[0],
        )

    event_key = bot.normalize_event_key(story.title)
    if len(event_key.split()) < 3:
        event_key = f"source-story-{story.fingerprint}"

    if translated:
        if gemini_disabled:
            bot.LOG.info("Publishing Russian translation with Gemini disabled: %s", story.title)
        else:
            bot.LOG.warning(
                "Gemini unavailable or unusable; publishing Russian translation fallback: %s",
                story.title,
            )
    else:
        bot.LOG.warning(
            "Translation unavailable; publishing source-language fallback: %s",
            story.title,
        )

    return bot.RenderedPost(
        title=title[:500].rstrip(),
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
