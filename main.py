#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

from openai import OpenAI  # pip install --upgrade openai

# custom
from api_key_crypto import get_api_key

# ================== Configuration ==================
LOG_NUMBER_OF_BATCHES = 2          # total completions = 2**LOG_NUMBER_OF_BATCHES
MODEL = "gpt-5"                    # GPT-5 reasoning model
REASONING_EFFORT = "high"          # minimal | low | medium | high
VERBOSITY = "high"                 # low | medium | high (steers visible length)
MAX_CONCURRENCY = 8                # throttle to respect RPM/TPM
PROMPT_FILE = "PUT_PROMPT_HERE.txt"
# ====================================================

@dataclass
class CompletionResult:
    text: str
    reasoning_tokens: int
    output_tokens: int
    total_tokens: int

def _make_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

def _extract_text(resp) -> str:
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt

    parts: List[str] = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for content in getattr(item, "content", []) or []:
                t = getattr(content, "text", None)
                if t:
                    parts.append(t)
    return "".join(parts).strip()

def _extract_usage(resp) -> Tuple[int, int, int]:
    usage = getattr(resp, "usage", None)
    rt = getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0) or 0
    ot = getattr(usage, "output_tokens", 0) or 0
    tt = getattr(usage, "total_tokens", 0) or 0
    return int(rt), int(ot), int(tt)

def _one_call(client: OpenAI, prompt: str):
    return client.responses.create(
        model=MODEL,
        input=prompt,
        reasoning={"effort": REASONING_EFFORT},
        text={"verbosity": VERBOSITY, "format": {"type": "text"}},
    )

def _complete_once(client: OpenAI, prompt: str) -> CompletionResult:
    resp = _one_call(client, prompt)
    text = _extract_text(resp)
    rt, ot, tt = _extract_usage(resp)
    return CompletionResult(text=text or "[empty]", reasoning_tokens=rt, output_tokens=ot, total_tokens=tt)

async def _run_parallel(client: OpenAI, prompt: str, n: int) -> List[CompletionResult]:
    sem = asyncio.Semaphore(min(MAX_CONCURRENCY, n))

    async def _one():
        async with sem:
            return await asyncio.to_thread(_complete_once, client, prompt)

    tasks = [asyncio.create_task(_one()) for _ in range(n)]
    return await asyncio.gather(*tasks)

def _load_prompt() -> str:
    if not os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "w", encoding="utf-8") as f:
            f.write("")
        print(f"Prompt file '{PROMPT_FILE}' created. Please add your prompt and rerun.")
        sys.exit(1)

    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    if not prompt:
        print(f"Prompt file '{PROMPT_FILE}' is empty. Please add your prompt and rerun.")
        sys.exit(1)

    return prompt

def main() -> None:
    prompt = _load_prompt()
    api_key = get_api_key()
    print("API key successfully loaded.")

    client = _make_client(api_key)
    n = 1 << LOG_NUMBER_OF_BATCHES
    print(f"MODEL={MODEL} | completions={n} | reasoning={REASONING_EFFORT} | verbosity={VERBOSITY}")

    results = asyncio.run(_run_parallel(client, prompt, n))

    for i, r in enumerate(results, 1):
        print(f"\n===== COMPLETION {i} of {n} =====")
        print(r.text.strip() or "[empty]")
        print(f"[usage] reasoning_tokens={r.reasoning_tokens}  output_tokens={r.output_tokens}  total_tokens={r.total_tokens}")

if __name__ == "__main__":
    main()

