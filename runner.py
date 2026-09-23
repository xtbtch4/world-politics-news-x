from __future__ import annotations

import sys

import bot


_original_make_post = bot.make_post


def make_post_with_source_fallback(story: bot.Story) -> bot.RenderedPost:
    post = _original_make_post(story)
    if post is not None:
        return post

    # Gemini may be rate-limited or temporarily unavailable. In that case,
    # publish the source material unchanged instead of dropping the story.
    evidence, image_url, video_url = bot.fetch_article_context(story)
    summary = bot.clean_text(story.summary or evidence)
    if not summary:
        summary = "Full details are available at the source link below."

    event_key = bot.normalize_event_key(story.title)
    if len(event_key.split()) < 3:
        event_key = f"source-story-{story.fingerprint}"

    bot.LOG.warning(
        "Gemini unavailable or unusable; publishing source-language fallback: %s",
        story.title,
    )

    return bot.RenderedPost(
        title=story.title,
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
