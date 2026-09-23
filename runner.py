from __future__ import annotations

import os
import sys

import requests

# Use Gemini Flash Lite as the only translator/rewriter.
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


def make_post_with_source_fallback(story: bot.Story) -> bot.RenderedPost:
    global _gemini_attempted, _gemini_cached_response, _gemini_cached_exception

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

    # Do not use low-quality machine-translation fallbacks. If Gemini is
    # unavailable, publish the original source-language title and summary.
    evidence, image_url, video_url = bot.fetch_article_context(story)
    summary = bot.clean_text(story.summary or evidence)
    if not summary:
        summary = "Full details are available at the source link below."

    event_key = bot.normalize_event_key(story.title)
    if len(event_key.split()) < 3:
        event_key = f"source-story-{story.fingerprint}"

    bot.LOG.warning(
        "Gemini unavailable or unusable; publishing source language without translation: %s",
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
