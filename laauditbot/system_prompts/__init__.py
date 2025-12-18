from __future__ import annotations

from pathlib import Path


_DIR = Path(__file__).resolve().parent


def load_system_prompt(name: str) -> str:
    path = _DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8").rstrip()


def load_default_system_prompts() -> dict[str, str]:
    return {
        "generation": load_system_prompt("generation"),
        "merge": load_system_prompt("merge"),
        "format": load_system_prompt("format"),
    }

