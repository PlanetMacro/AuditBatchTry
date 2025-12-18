from __future__ import annotations

from laauditbot.system_prompts import load_system_prompt

SYSTEM_PROMPT = load_system_prompt("generation")
MERGE_SYSTEM_PROMPT = load_system_prompt("merge")
ISSUE_FORMAT_SYSTEM_PROMPT = load_system_prompt("format")

