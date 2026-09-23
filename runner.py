from __future__ import annotations

import sys

import requests

import bot


_original_make_post = bot.make_post


def translate_to_russian(text: str) -> str:
    text = bot.clean_text(text)
    if not text or bot.has_cyrillic(text):
        return text

    translated_parts: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= 1500:
            chunk = remaining
            remaining = ""
        else:
            split_at = max(
                remaining.rfind(". ", 0, 1500),
                remaining.rfind("! ", 0, 1500),
                remaining.rfind("? ", 0, 1500),
            )
            if split_at < 500:
                split_at = remaining.rfind(" ", 0, 1500)
            if split_at < 1:
                split_at = 1500
            else:
                split_at += 1
            chunk = remaining[:split_at].strip()
            remaining = remaining[split_at:].strip()

        response = requests.get(
            "https://translate.googleapis.com/translate_a/single",
            params={
                "client": "gtx",
                "sl": "auto",
                "tl": "ru",
                "dt": "t",
                "q": chunk,
            },
            timeout=20,
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
            raise RuntimeError("translation service returned empty text")
        translated_parts.append(translated)

    return bot.clean_text(" ".join(translated_parts))


def make_post_with_source_fallback(story: bot.Story) -> bot.RenderedPost:
    post = _original_make_post(story)
    if post is not None:
        return post

    # Gemini may be rate-limited or temporarily unavailable. In that case,
    # translate the source material with a separate translation service instead
    # of dropping the story. If translation also fails, publish the original.
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
    except (requests.RequestException, ValueError, TypeError, RuntimeError) as exc:
        bot.LOG.warning(
            "Fallback translation failed; publishing source language: %s (%s)",
            story.title,
            str(exc).splitlines()[0],
        )

    event_key = bot.normalize_event_key(story.title)
    if len(event_key.split()) < 3:
        event_key = f"source-story-{story.fingerprint}"

    if translated:
        bot.LOG.warning(
            "Gemini unavailable or unusable; publishing Russian translation fallback: %s",
            story.title,
        )
    else:
        bot.LOG.warning(
            "Gemini and fallback translator unavailable; publishing source-language fallback: %s",
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
