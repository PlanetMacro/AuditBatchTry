# openai_client.py
from __future__ import annotations

import asyncio
import os
import random
import string
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

import httpx
import openai  # for typed exceptions
from openai import OpenAI, AsyncOpenAI

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

# ====================================================
# Utilities & data structures
# ====================================================

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


# ====================================================
# Prompt I/O helpers
# ====================================================

def load_prompt(prompt_file: str) -> str:
    """
    Read the prompt from prompt_file. If the file does not exist, create it and exit.
    If the file exists but is empty, exit with an instruction to fill it.

    Note: This returns the *user* prompt only. The final prompt is constructed per API
    call to ensure a fresh random string each time.
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


# ====================================================
# Client factories (sync + async)
# ====================================================

def make_client(api_key: Optional[str] = None, *, timeout_s: float = 600.0, max_retries: int = 4) -> OpenAI:
    """
    Synchronous client (kept for compatibility). Prefer make_async_client for parallel runs.
    """
    return OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        timeout=timeout_s,
        max_retries=max_retries,
    )


def make_async_client(
    api_key: Optional[str] = None,
    *,
    timeout_s: float = 600.0,
    max_retries: int = 4,
    use_aiohttp: bool = True,
) -> AsyncOpenAI:
    """
    Create a single AsyncOpenAI client to be reused across all concurrent tasks.
    Set use_aiohttp=True if you installed: pip install "openai[aiohttp]"
    """
    if use_aiohttp:
        from openai import DefaultAioHttpClient  # optional backend
        return AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            timeout=timeout_s,
            max_retries=max_retries,
            http_client=DefaultAioHttpClient(),
        )
    else:
        return AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            timeout=timeout_s,
            max_retries=max_retries,
        )


# ====================================================
# Response extraction
# ====================================================

def _extract_text(resp) -> str:
    """
    Extract the unified text from a Responses API result.
    """
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
    """
    Extract (reasoning_tokens, output_tokens, total_tokens) if present.
    """
    usage = getattr(resp, "usage", None)
    rt = getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0) or 0
    ot = getattr(usage, "output_tokens", 0) or 0
    tt = getattr(usage, "total_tokens", 0) or 0
    return int(rt), int(ot), int(tt)


# ====================================================
# Single-call wrappers (sync + async)
# ====================================================

def _one_call(
    client: OpenAI,
    prompt: str,
    *,
    model: str,
    reasoning_effort: str,
    verbosity: str,
):
    """
    Synchronous Responses API call (kept for completeness).
    """
    return client.responses.create(
        model=model,
        reasoning={"effort": reasoning_effort},
        text={"verbosity": verbosity, "format": {"type": "text"}},
        input=prompt,
        # tools=[{"type": "web_search"}, {"type": "code_interpreter", "container": {"type": "auto"}}],
    )


def _complete_once(
    client: OpenAI,
    user_prompt: str,
    *,
    model: str,
    reasoning_effort: str,
    verbosity: str,
) -> CompletionResult:
    """
    Synchronous single completion (not used by the async runner).
    """
    comment = _random_string(5, 50)
    final_prompt = f"{SYSTEM_PROMPT}\n\n' IGNORE THIS COMMENT: {comment}\n\n{user_prompt}"

    resp = _one_call(
        client,
        final_prompt,
        model=model,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
    )
    text = _extract_text(resp) or "[empty]"
    rt, ot, tt = _extract_usage(resp)
    return CompletionResult(text=text, reasoning_tokens=rt, output_tokens=ot, total_tokens=tt)


async def _one_call_async(
    client: AsyncOpenAI,
    prompt: str,
    *,
    model: str,
    reasoning_effort: str,
    verbosity: str,
    per_request_timeout: Optional[float] = None,
):
    """
    Asynchronous Responses API call with optional per-request timeout override.
    """
    c = client.with_options(timeout=per_request_timeout) if per_request_timeout else client
    return await c.responses.create(
        model=model,
        reasoning={"effort": reasoning_effort},
        text={"verbosity": verbosity, "format": {"type": "text"}},
        input=prompt,
        # tools=[{"type": "web_search"}, {"type": "code_interpreter", "container": {"type": "auto"}}],
    )


async def _complete_once_async(
    client: AsyncOpenAI,
    user_prompt: str,
    *,
    model: str,
    reasoning_effort: str,
    verbosity: str,
    per_request_timeout: Optional[float] = None,
) -> CompletionResult:
    """
    Asynchronous single completion.
    """
    comment = _random_string(5, 50)
    final_prompt = f"{SYSTEM_PROMPT}\n\n' IGNORE THIS COMMENT: {comment}\n\n{user_prompt}"

    resp = await _one_call_async(
        client,
        final_prompt,
        model=model,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        per_request_timeout=per_request_timeout,
    )
    text = _extract_text(resp) or "[empty]"
    rt, ot, tt = _extract_usage(resp)
    print("Single request completed")
    return CompletionResult(text=text, reasoning_tokens=rt, output_tokens=ot, total_tokens=tt)


# ====================================================
# Parallel generation with bounded concurrency + backoff
# ====================================================

async def run_parallel(
    *,
    client: AsyncOpenAI,
    prompt: str,
    n: int,
    max_concurrency: int,
    model: str,
    reasoning_effort: str = "medium",
    verbosity: str = "medium",
    per_request_timeout: Optional[float] = None,
    app_retries: int = 2,
    base_backoff_s: float = 1.0,
) -> List[CompletionResult]:
    """
    Execute n completions concurrently, bounded by max_concurrency.
    Adds application-level retries for 429s and timeouts with jittered exponential backoff.
    """
    sem = asyncio.Semaphore(max(1, min(max_concurrency, n)))

    async def _worker(k: int) -> CompletionResult:
        attempt = 0
        while True:
            try:
                async with sem:
                    return await _complete_once_async(
                        client,
                        prompt,
                        model=model,
                        reasoning_effort=reasoning_effort,
                        verbosity=verbosity,
                        per_request_timeout=per_request_timeout,
                    )
            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
                httpx.TimeoutException,
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
            ) as e:
                if attempt >= app_retries:
                    raise
                delay = base_backoff_s * (2 ** attempt) + random.uniform(0.0, 0.5)
                await asyncio.sleep(delay)
                attempt += 1

    tasks = [asyncio.create_task(_worker(i)) for i in range(n)]
    return await asyncio.gather(*tasks)


# ====================================================
# Merging support (async)
# ====================================================

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


async def _merge_two_async(
    client: AsyncOpenAI,
    a: str,
    b: str,
    *,
    model: str,
    reasoning_effort: str,
    verbosity: str,
    per_request_timeout: Optional[float] = None,
) -> CompletionResult:
    """Merge two issue lists via the LLM, returning the deduplicated union."""
    prompt = _build_merge_prompt(a, b)
    resp = await _one_call_async(
        client,
        prompt,
        model=model,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        per_request_timeout=per_request_timeout,
    )
    text = _extract_text(resp) or "[empty]"
    rt, ot, tt = _extract_usage(resp)
    return CompletionResult(text=text, reasoning_tokens=rt, output_tokens=ot, total_tokens=tt)


async def hierarchical_merge(
    *,
    client: AsyncOpenAI,
    texts: List[str],
    max_concurrency: int,
    model: str,
    reasoning_effort: str,
    verbosity: str,
    per_request_timeout: Optional[float] = None,
) -> CompletionResult:
    """
    Hierarchically merge a list of issue lists by repeatedly merging pairs (binary tree reduction)
    until a single deduplicated union remains. This bounds prompt size per call.
    """
    if not texts:
        return CompletionResult(text="", reasoning_tokens=0, output_tokens=0, total_tokens=0)
    if len(texts) == 1:
        return CompletionResult(text=texts[0], reasoning_tokens=0, output_tokens=0, total_tokens=0)

    agg_rt = 0
    agg_ot = 0
    agg_tt = 0
    current: List[str] = [t or "" for t in texts]
    sem = asyncio.Semaphore(max(1, max_concurrency))

    while len(current) > 1:
        pairs: List[Tuple[str, str]] = []
        i = 0
        while i < len(current):
            a = current[i]
            b = current[i + 1] if i + 1 < len(current) else ""
            pairs.append((a, b))
            i += 2

        tasks = []
        carried_forward: List[str] = []
        for a, b in pairs:
            if b.strip():
                async def job(a=a, b=b):
                    async with sem:
                        return await _merge_two_async(
                            client,
                            a,
                            b,
                            model=model,
                            reasoning_effort=reasoning_effort,
                            verbosity=verbosity,
                            per_request_timeout=per_request_timeout,
                        )
                tasks.append(asyncio.create_task(job()))
            else:
                carried_forward.append(a)

        merged_texts: List[str] = []
        if tasks:
            results: List[CompletionResult] = await asyncio.gather(*tasks)
            for r in results:
                merged_texts.append(r.text)
                agg_rt += r.reasoning_tokens
                agg_ot += r.output_tokens
                agg_tt += r.total_tokens

        current = merged_texts + carried_forward

    return CompletionResult(text=current[0], reasoning_tokens=agg_rt, output_tokens=agg_ot, total_tokens=agg_tt)

