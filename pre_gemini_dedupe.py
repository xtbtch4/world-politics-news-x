from __future__ import annotations

import bot
import runner


_original_select_stories = bot.select_stories


def select_stories_before_gemini(stories, state):
    """Drop obvious already-known events before spending a Gemini request.

    The full post-generation event key remains the final dedupe authority. This
    prefilter is intentionally stricter (0.68) to avoid suppressing genuinely
    new developments that merely share a few names with an older story.
    """
    selected = _original_select_stories(stories, state)
    seen_event_keys = [
        str(value)
        for value in state.get("posted_event_keys", [])
        if str(value).strip()
    ]

    filtered = []
    candidate_limit = max(bot.MAX_POSTS + 1, bot.MAX_POSTS)

    for story in selected:
        probe_key = runner.stronger_normalize_event_key(story.title)
        if probe_key and any(
            bot.event_similarity(probe_key, previous_key) >= 0.68
            for previous_key in seen_event_keys
        ):
            bot.LOG.info(
                "Pre-Gemini duplicate skipped to save quota: %s",
                story.title,
            )
            continue

        filtered.append(story)
        if len(filtered) >= candidate_limit:
            break

    bot.LOG.info(
        "Pre-Gemini candidate budget: %d story/stories for up to %d publication(s)",
        len(filtered),
        bot.MAX_POSTS,
    )
    return filtered


bot.select_stories = select_stories_before_gemini
