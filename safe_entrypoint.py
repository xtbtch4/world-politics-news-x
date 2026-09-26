from __future__ import annotations

import runpy

import gemini_quota_guard  # noqa: F401 - installs quota guard before bot imports

runpy.run_path("entrypoint.py", run_name="__main__")
