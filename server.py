from __future__ import annotations

# Backwards-compatible shim for `uvicorn server:app`.
from laauditbot.server import app

__all__ = ["app"]
