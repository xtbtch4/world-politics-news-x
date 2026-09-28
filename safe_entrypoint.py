from __future__ import annotations

import runpy

import gemini_quota_guard  # noqa: F401 - installs quota guard before bot imports
import gemini_error_guard  # noqa: F401 - converts Gemini key failures into rotation misses
import pre_gemini_dedupe  # noqa: F401 - skips known events before spending Gemini quota

# Do NOT import gemini_only_guard here. runner.py owns the intended fallback chain:
# Gemini 3.5 Flash Lite -> Gemini 3.1 Flash Lite -> OpenAI -> free translator -> source.
runpy.run_path("entrypoint.py", run_name="__main__")
