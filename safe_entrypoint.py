from __future__ import annotations

import runpy

import gemini_quota_guard  # noqa: F401 - installs quota guard before bot imports
import gemini_error_guard  # noqa: F401 - converts Gemini key failures into rotation misses

runpy.run_path("entrypoint.py", run_name="__main__")
