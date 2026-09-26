from __future__ import annotations

import runner


def make_post_gemini_only(story: runner.bot.Story) -> runner.bot.RenderedPost | None:
    """Publish only posts successfully translated by Gemini.

    OpenAI, Google Translate, and source-language fallbacks stay unreachable.
    If every configured Gemini model/key attempt fails, the story is skipped and
    can be reconsidered on a later workflow run.
    """
    post = runner.gemini_translation_with_rotation(story)
    if post is None:
        runner.bot.LOG.warning(
            "Gemini-only translation failed; skipping story without publishing source language: %s",
            story.title,
        )
        return None
    return post


# entrypoint.py reads runner.make_post_with_source_fallback dynamically after this
# module is imported, so replacing it here also preserves the existing entity and
# deduplication wrappers while removing every non-Gemini translation path.
runner.make_post_with_source_fallback = make_post_gemini_only
runner.bot.make_post = make_post_gemini_only
