from __future__ import annotations

import re

import runner
import bot


# Extra cross-source deduplication layer. Gemini can describe the same event
# with very different EVENT_KEY wording, so we enrich the key with named
# entities from the original RSS item and compare a small set of event families.
# Translation is intentionally Gemini-only: 3.5 Flash Lite first, then 3.1 Flash Lite.
# If every configured Gemini key/model attempt fails, the story is skipped.
_original_make_post = runner.gemini_translation_with_rotation
_original_event_similarity = bot.event_similarity

_ENTITY_STOPWORDS = {
    "the", "this", "that", "with", "from", "after", "before", "while",
    "prime", "minister", "president", "government", "leader", "official",
    "world", "news", "latest", "live", "new", "york",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}

_SPEECH_WORDS = {
    "speech", "address", "speaks", "speak", "speaking", "deliver", "delivers",
    "delivered", "delivering",
}


def _source_entity_tokens(story: bot.Story) -> list[str]:
    text = f"{story.title} {story.summary[:1200]}"
    result: list[str] = []
    # Keep capitalized names/places/organisations from the source language.
    for raw in re.findall(r"\b[A-Z][A-Za-z'’-]{2,}\b", text):
        token = raw.casefold().replace("’", "'")
        token = runner._EVENT_TOKEN_ALIASES.get(token, token)
        if token in _ENTITY_STOPWORDS or token in runner._EVENT_NOISE_TOKENS:
            continue
        if token not in result:
            result.append(token)
        if len(result) >= 14:
            break
    return result


def _enrich_post_event_key(story: bot.Story, post: bot.RenderedPost) -> bot.RenderedPost:
    extra = _source_entity_tokens(story)
    if not extra:
        return post
    enriched = runner.stronger_normalize_event_key(
        f"{post.event_key} {' '.join(extra)}"
    )
    return bot.RenderedPost(
        title=post.title,
        summary=post.summary,
        quote_line=post.quote_line,
        source_line=post.source_line,
        event_key=enriched,
        image_url=post.image_url,
        video_url=post.video_url,
    )


def make_post_with_entity_context(story: bot.Story) -> bot.RenderedPost | None:
    post = _original_make_post(story)
    if post is None:
        return None
    return _enrich_post_event_key(story, post)


def _event_families(tokens: set[str]) -> set[str]:
    families: set[str] = set()
    has_speech = bool(tokens & _SPEECH_WORDS)
    has_general_assembly = {"general", "assembly"}.issubset(tokens)
    has_united_nations = {"united", "nations"}.issubset(tokens)
    if (
        "unga" in tokens
        or has_general_assembly
        or has_united_nations
        or (has_speech and "york" in tokens)
    ):
        families.add("un_speech")
    return families


def semantic_event_similarity(a: str, b: str) -> float:
    score = _original_event_similarity(a, b)
    if score >= 0.60:
        return score

    tokens_a = set(runner._canonical_event_tokens(a))
    tokens_b = set(runner._canonical_event_tokens(b))
    if not tokens_a or not tokens_b:
        return score

    families = _event_families(tokens_a) & _event_families(tokens_b)
    if not families:
        return score

    generic = {
        "general", "assembly", "united", "nations", "unga", "speech", "address",
        "speak", "speaks", "speaking", "deliver", "delivers", "delivered", "delivering",
        "prime", "minister", "president", "leader", "government", "election",
        "york", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december",
    }
    entities_a = {token for token in tokens_a if token not in generic and not token.isdigit()}
    entities_b = {token for token in tokens_b if token not in generic and not token.isdigit()}
    common_entities = entities_a & entities_b

    # Same named entity/country plus the same high-level event family is enough
    # to treat differently worded outlet headlines as the same event.
    if common_entities:
        return max(score, 0.68)
    return score


bot.make_post = make_post_with_entity_context
bot.event_similarity = semantic_event_similarity


if __name__ == "__main__":
    raise SystemExit(bot.main())
