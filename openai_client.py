from __future__ import annotations

import asyncio
import os
import random
import string
import sys
from dataclasses import dataclass
from typing import List, Tuple

from openai import OpenAI  # pip install --upgrade openai

# ----------------------
# System prompt to define the shape of the issues found to make merging easier
# ----------------------
SYSTEM_PROMPT = (
    "You are an assistant that has a high degree of freedom in answering from multiple angles"
    "Issues should have the exact following format: Issue: <NUMBER> <LINEBREAK>, Location: <COPY_OF_LINE>, Description: <TEXT>"
)

def _random_string(min_len: int = 5, max_len: int = 50) -> str:
    """Generate a random alphanumeric string with length in [min_len, max_len]."""
    n = random.randint(min_len, max_len)
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(n))


@dataclass
class CompletionResult:
    text: str
    reasoning_tokens: int
    output_tokens: int
    total_tokens: int


def make_client(api_key: str) -> OpenAI:
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


def _one_call(
    client: OpenAI,
    prompt: str,
    *,
    model: str,
    reasoning_effort: str,
    verbosity: str,
):
    return client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": reasoning_effort},
        text={"verbosity": verbosity, "format": {"type": "text"}},
    )


def _complete_once(
    client: OpenAI,
    user_prompt: str,
    *,
    model: str,
    reasoning_effort: str,
    verbosity: str,
) -> CompletionResult:
    # Compose the final prompt per API call:
    #   [SYSTEM_PROMPT]
    #   ' IGNORE THIS COMMENT: X        ## SINCE THE API HAS NO TEMPERATURE THIS BREAKS STRICT DETERMINISM
    #   [user-loaded prompt]
    comment = _random_string(5, 50)
    final_prompt = f"{SYSTEM_PROMPT}\n\n' IGNORE THIS COMMENT: {comment}\n\n{user_prompt}"

    resp = _one_call(
        client,
        final_prompt,
        model=model,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
    )
    text = _extract_text(resp)
    rt, ot, tt = _extract_usage(resp)
    return CompletionResult(text=text or "[empty]", reasoning_tokens=rt, output_tokens=ot, total_tokens=tt)


async def run_parallel(
    *,
    client: OpenAI,
    prompt: str,
    n: int,
    max_concurrency: int,
    model: str,
    reasoning_effort: str,
    verbosity: str,
) -> List[CompletionResult]:
    """
    Execute n completions in parallel, throttled by max_concurrency.
    Each call gets its own random IGNORE comment string.
    """
    sem = asyncio.Semaphore(min(max_concurrency, n))

    async def _one():
        async with sem:
            return await asyncio.to_thread(
                _complete_once,
                client,
                prompt,  # pass the user-loaded prompt; _complete_once builds the final prompt
                model=model,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
            )

    tasks = [asyncio.create_task(_one()) for _ in range(n)]
    return await asyncio.gather(*tasks)


def load_prompt(prompt_file: str) -> str:
    """
    Read the prompt from prompt_file. If the file does not exist, create it and exit.
    If the file exists but is empty, exit with an instruction to fill it.

    Note: This returns the *user* prompt only. The final prompt is constructed per API
    call inside _complete_once to ensure a fresh random string each time.
    """
    if not os.path.exists(prompt_file):
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write("")
        print(f"Prompt file '{prompt_file}' created. Please add your prompt and rerun.")
        sys.exit(1)

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    if not prompt:
        print(f"Prompt file '{prompt_file}' is empty. Please add your prompt and rerun.")
        sys.exit(1)

    return prompt

