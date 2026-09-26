from __future__ import annotations

import json
import os
import re
import subprocess

import runner
import bot


# Translation order is strict:
# 1) gemini-3.5-flash-lite across every configured Gemini key
# 2) gemini-3.1-flash-lite across every configured Gemini key
# 3) OpenAI
# 4) free translator
# 5) source language only if every translation path fails
#
# runner.py already implements OpenAI/free/source fallbacks. This wrapper makes
# Gemini exhaust every available key for each allowed model before moving on.
def gemini_all_keys_rotation(story: bot.Story) -> bot.RenderedPost | None:
    keys = runner.gemini_api_keys()
    if not keys:
        bot.LOG.warning("No Gemini API keys are configured")
        return None

    bot.LOG.info(
        "Gemini strict rotation configured with %d key(s); model priority: %s -> %s",
        len(keys),
        runner.GEMINI_MODELS[0],
        runner.GEMINI_MODELS[1],
    )

    for model in runner.GEMINI_MODELS:
        for index, api_key in enumerate(keys, start=1):
            post, status, body = runner.run_gemini_once(
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

            bot.LOG.warning(
                "Gemini attempt failed: model=%s key=%d/%d status=%s; trying next key/model",
                model,
                index,
                len(keys),
                status,
            )

        bot.LOG.warning(
            "All %d Gemini key(s) failed for %s; moving to next allowed model/fallback",
            len(keys),
            model,
        )

    return None


# runner.make_post_with_source_fallback resolves this function dynamically, so
# replace the Gemini stage with the strict all-keys implementation above.
runner.gemini_translation_with_rotation = gemini_all_keys_rotation


# Extra cross-source deduplication layer. Gemini can describe the same event
# with very different EVENT_KEY wording, so we enrich the key with named
# entities from the original RSS item and normalize common event word variants.
_original_make_post = runner.make_post_with_source_fallback
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

# These aliases deliberately cover wording changes that commonly made the same
# disaster story look different across outlets (for example France 24 vs BBC).
_EVENT_SEMANTIC_ALIASES = {
    "flooding": "flood",
    "floods": "flood",
    "flooded": "flood",
    "downpour": "rain",
    "downpours": "rain",
    "rainfall": "rain",
    "rains": "rain",
    "raining": "rain",
    "declares": "declare",
    "declared": "declare",
    "declaration": "declare",
    "emergency": "disaster",
    "thai": "thailand",
}


def _semantic_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for token in runner._canonical_event_tokens(value):
        tokens.add(_EVENT_SEMANTIC_ALIASES.get(token, token))
    return tokens


def _source_entity_tokens(story: bot.Story) -> list[str]:
    text = f"{story.title} {story.summary[:1200]}"
    result: list[str] = []
    # Keep capitalized names/places/organisations from the source language.
    for raw in re.findall(r"\b[A-Z][A-Za-z'’-]{2,}\b", text):
        token = raw.casefold().replace("’", "'")
        token = runner._EVENT_TOKEN_ALIASES.get(token, token)
        token = _EVENT_SEMANTIC_ALIASES.get(token, token)
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

    tokens_a = _semantic_tokens(a)
    tokens_b = _semantic_tokens(b)
    if not tokens_a or not tokens_b:
        return score

    # Recalculate after semantic normalization. This closes cases such as
    # flooding/flood, declared/declaration and downpour/rainfall without using a
    # broad city-wide rule that might suppress a genuinely new event later.
    normalized_score = len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))
    score = max(score, normalized_score)
    if score >= 0.60:
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


# A queued GitHub Actions run can carry an older event SHA. actions/checkout may
# therefore give it an older data/posted.json even though the previous serialized
# run already published and pushed newer history. Fetch only the latest history
# from remote main before bot.main(), without replacing the code checkout.
def refresh_history_from_remote_main() -> None:
    if os.getenv("GITHUB_ACTIONS", "").casefold() != "true":
        return

    live_ref = "refs/remotes/origin/live-main-history"
    try:
        subprocess.run(
            ["git", "fetch", "origin", f"main:{live_ref}"],
            check=True,
            text=True,
            capture_output=True,
            timeout=45,
        )
        shown = subprocess.run(
            ["git", "show", f"{live_ref}:data/posted.json"],
            check=True,
            text=True,
            capture_output=True,
            timeout=30,
        )
        remote = json.loads(shown.stdout)
        try:
            local = json.loads(bot.STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            local = {}

        def merged_values(key: str) -> list[str]:
            return list(dict.fromkeys(remote.get(key, []) + local.get(key, [])))[-2000:]

        merged = {
            "posted_urls": merged_values("posted_urls"),
            "posted_fingerprints": merged_values("posted_fingerprints"),
            "posted_event_keys": merged_values("posted_event_keys"),
            "updated_at": remote.get("updated_at") or local.get("updated_at"),
        }
        bot.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        bot.STATE_PATH.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        bot.LOG.info(
            "Refreshed publication history from latest main: %d urls, %d event keys",
            len(merged["posted_urls"]),
            len(merged["posted_event_keys"]),
        )
    except (OSError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        # Do not block the news run if GitHub briefly fails; normal local history
        # is still safer than publishing nothing at all.
        bot.LOG.warning("Could not refresh latest remote publication history: %s", exc)


# bot.main() writes data/posted.json immediately after every successful Telegram
# publication. Persist that exact checkpoint to GitHub immediately as well, so a
# later workflow timeout cannot erase the fact that the URL was sent.
_original_save_state = bot.save_state


def _history_signature(state: dict) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(state.get("posted_urls", [])),
        tuple(state.get("posted_fingerprints", [])),
        tuple(state.get("posted_event_keys", [])),
    )


def _saved_history_signature() -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    try:
        saved = json.loads(bot.STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ((), (), ())
    return _history_signature(saved)


def save_state_with_checkpoint(state: dict) -> None:
    before = _saved_history_signature()
    _original_save_state(state)
    after = _history_signature(state)

    # DRY_RUN never publishes, and updated_at-only changes do not need an
    # immediate checkpoint. The normal final workflow step can handle those.
    if bot.DRY_RUN or before == after:
        return
    if os.getenv("GITHUB_ACTIONS", "").casefold() != "true":
        return

    try:
        subprocess.run(
            ["git", "config", "user.name", "github-actions[bot]"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "config",
                "user.email",
                "41898282+github-actions[bot]@users.noreply.github.com",
            ],
            check=True,
        )
        subprocess.run(["git", "add", str(bot.STATE_PATH)], check=True)
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", str(bot.STATE_PATH)]
        )
        if staged.returncode == 0:
            return
        if staged.returncode != 1:
            raise subprocess.CalledProcessError(staged.returncode, staged.args)

        subprocess.run(
            ["git", "commit", "-m", "Checkpoint publication history"],
            check=True,
        )
        pushed = subprocess.run(
            ["git", "push", "origin", "HEAD:main"],
            text=True,
            capture_output=True,
        )
        if pushed.returncode != 0:
            bot.LOG.warning(
                "Immediate history push failed; retrying after rebase: %s",
                (pushed.stderr or pushed.stdout).strip()[:800],
            )
            subprocess.run(["git", "pull", "--rebase", "origin", "main"], check=True)
            subprocess.run(["git", "push", "origin", "HEAD:main"], check=True)

        bot.LOG.info("Publication history checkpoint pushed immediately")
    except (OSError, subprocess.SubprocessError) as exc:
        # Keep the news run alive; the workflow's final history step remains as
        # a second persistence layer if an immediate checkpoint ever fails.
        bot.LOG.error("Failed to checkpoint publication history immediately: %s", exc)


bot.save_state = save_state_with_checkpoint


if __name__ == "__main__":
    refresh_history_from_remote_main()
    raise SystemExit(bot.main())
