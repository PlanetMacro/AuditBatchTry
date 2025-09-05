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
# System prompt used for the *generation* step (kept as-is)
# ----------------------
SYSTEM_PROMPT = (
    "You are an assistant that has a high degree of freedom in answering from multiple angles"
    "Issues should have the exact following format: Issue: <NUMBER> <LINEBREAK>, Location: <COPY_OF_LINE>, Description: <TEXT>"
)

# ----------------------
# System prompt used for the *merging* step
# ----------------------
MERGE_SYSTEM_PROMPT = (
    "You are an assistant that merges two audit reports of issues into a single deduplicated list.\n"
    "Output MUST contain only issues in exactly this format:\n"
    "Issue: <NUMBER>\nLocation: <COPY_OF_LINE>\nDescription: <TEXT>\n"
    "Rules:\n"
    " - Compute the set UNION of issues from both reports.\n"
    " - Treat two issues as duplicates if they refer to the same underlying problem OR the same Location (case/whitespace-insensitive), even with wording differences.\n"
    " - When merging duplicates, keep the clearest Description and pick one Location verbatim from either input.\n"
    " - Renumber issues consecutively starting from 1 in the final output.\n"
    " - Place a single blank line between issues. Do NOT add any commentary, headings, code fences, or extra text before/after the list."
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
        tools=[
            {"type": "web_search"},
            {"type": "code_interpreter", "container": {"type": "auto"}},
        ],
        
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
    #   ' IGNORE THIS COMMENT: X        ## breaks strict determinism a bit
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


# =========================
# Merging support
# =========================

def _build_merge_prompt(a: str, b: str) -> str:
    """Construct the deterministic merge instruction that takes two reports A and B."""
    comment = _random_string(5, 50)
    return (
        f"{MERGE_SYSTEM_PROMPT}\n\n"
        f"' IGNORE THIS COMMENT: {comment}\n\n"
        "REPORT A:\n"
        f"{(a or '').strip()}\n\n"
        "REPORT B:\n"
        f"{(b or '').strip()}\n\n"
        "Return ONLY the merged list in the exact required format."
    )


def _merge_two(
    client: OpenAI,
    a: str,
    b: str,
    *,
    model: str,
    reasoning_effort: str,
    verbosity: str,
) -> CompletionResult:
    """Merge two issue lists via the LLM, returning the deduplicated union."""
    prompt = _build_merge_prompt(a, b)
    resp = _one_call(
        client,
        prompt,
        model=model,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
    )
    text = _extract_text(resp)
    rt, ot, tt = _extract_usage(resp)
    return CompletionResult(text=text or "[empty]", reasoning_tokens=rt, output_tokens=ot, total_tokens=tt)


async def hierarchical_merge(
    *,
    client: OpenAI,
    texts: List[str],
    max_concurrency: int,
    model: str,
    reasoning_effort: str,
    verbosity: str,
) -> CompletionResult:
    """
    Hierarchically merge a list of issue lists by repeatedly merging pairs (binary tree reduction)
    until a single deduplicated union remains. This bounds prompt size per call.
    """
    # Trivial cases
    if not texts:
        return CompletionResult(text="", reasoning_tokens=0, output_tokens=0, total_tokens=0)
    if len(texts) == 1:
        # Nothing to merge; return as-is (counts = 0 for merging phase)
        return CompletionResult(text=texts[0], reasoning_tokens=0, output_tokens=0, total_tokens=0)

    agg_rt = 0
    agg_ot = 0
    agg_tt = 0
    current: List[str] = [t or "" for t in texts]

    level = 0
    while len(current) > 1:
        level += 1
        pairs: List[Tuple[str, str]] = []
        i = 0
        while i < len(current):
            a = current[i]
            b = current[i + 1] if i + 1 < len(current) else ""
            pairs.append((a, b))
            i += 2

        sem = asyncio.Semaphore(min(max_concurrency, max(1, len(pairs))))

        async def _merge_job(a: str, b: str) -> CompletionResult:
            async with sem:
                return await asyncio.to_thread(
                    _merge_two,
                    client,
                    a,
                    b,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    verbosity=verbosity,
                )

        # Launch merges only for actual pairs; carry forward unpaired items
        tasks = []
        carried_forward: List[str] = []
        for a, b in pairs:
            if b.strip():
                tasks.append(asyncio.create_task(_merge_job(a, b)))
            else:
                # No need to invoke the model if there's nothing to merge with
                carried_forward.append(a)

        # Collect merged results
        merged_texts: List[str] = []
        if tasks:
            results: List[CompletionResult] = await asyncio.gather(*tasks)
            for r in results:
                merged_texts.append(r.text)
                agg_rt += r.reasoning_tokens
                agg_ot += r.output_tokens
                agg_tt += r.total_tokens

        # Next level
        current = merged_texts + carried_forward

    # Single final text remains
    return CompletionResult(text=current[0], reasoning_tokens=agg_rt, output_tokens=agg_ot, total_tokens=agg_tt)

