# openai_client.py
from __future__ import annotations

import asyncio
import os
import random
import re
import string
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

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

RUNS_DIR = "RUNS"
AUDIT_BASENAME_RE = re.compile(r"^audit_(\d+)$")  # files with no extension

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
    use_aiohttp: bool = False,
) -> AsyncOpenAI:
    """
    Create a single AsyncOpenAI client to be reused across all concurrent tasks.
    Prefer the default httpx backend (use_aiohttp=False). If you *must* use aiohttp,
    pass use_aiohttp=True and ensure the client is properly awaited/closed by caller.
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
# Filesystem helpers (RUNS/)
# ====================================================

def _ensure_runs_dir(path: str = RUNS_DIR) -> None:
    """Create RUNS directory if missing."""
    os.makedirs(path, exist_ok=True)

def _list_audit_files(path: str = RUNS_DIR) -> List[Tuple[int, str]]:
    """
    Return list of (index, fullpath) for files named audit_{i} with i >= 1, sorted by i.
    Ignores files that do not match the naming convention.
    """
    if not os.path.isdir(path):
        return []
    out: List[Tuple[int, str]] = []
    for name in os.listdir(path):
        m = AUDIT_BASENAME_RE.match(name)
        if m:
            i = int(m.group(1))
            out.append((i, os.path.join(path, name)))
    out.sort(key=lambda t: t[0])
    return out

def _next_run_index_and_remaining(n_desired: int, path: str = RUNS_DIR) -> Tuple[int, int, int]:
    """
    Given desired total runs (n_desired), returns:
      (next_index, already_existing, to_run_now)
    next_index is 1 if empty, else max existing index + 1.
    """
    files = _list_audit_files(path)
    already_existing = len(files)
    next_index = files[-1][0] + 1 if files else 1
    to_run_now = max(0, n_desired - already_existing)
    return next_index, already_existing, to_run_now

def _write_audit_file(run_index: int, text: str, path: str = RUNS_DIR) -> str:
    """
    Write the raw audit text to RUNS/audit_{run_index} (no extension). Returns the file path.
    Uses an atomic-style write: write to temp then rename.
    """
    _ensure_runs_dir(path)
    target = os.path.join(path, f"audit_{run_index}")
    tmp = target + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text or "")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)
    return target

def _read_all_audit_texts(path: str = RUNS_DIR) -> List[str]:
    """Read all RUNS/audit_{i} files (sorted by i) and return their contents as a list of strings."""
    files = _list_audit_files(path)
    texts: List[str] = []
    for _, fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts

def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0

def _pad_to_power_of_two(texts: List[str]) -> List[str]:
    """
    Ensure len(texts) is a power of two by duplicating the last element.
    This preserves 'merge everything from RUNS' without dropping any run.
    """
    n = len(texts)
    if n == 0:
        return texts
    if _is_power_of_two(n):
        return texts
    # next power-of-two >= n
    k = 1
    while k < n:
        k <<= 1
    last = texts[-1]
    return texts + [last] * (k - n)


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
        # tools=[{"type": "web_search"}],  # generation-only; sync path rarely used
    )


async def _one_call_async(
    client: AsyncOpenAI,
    prompt: str,
    *,
    model: str,
    reasoning_effort: str,
    verbosity: str,
    per_request_timeout: Optional[float] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    stream: bool = False,
):
    """
    Asynchronous Responses API call with optional per-request timeout override, optional tools,
    and optional streaming. We prefer streaming for long outputs to avoid proxy buffering stalls. :contentReference[oaicite:2]{index=2}
    """
    c = client.with_options(timeout=per_request_timeout) if per_request_timeout else client
    kwargs: Dict[str, Any] = dict(
        model=model,
        reasoning={"effort": reasoning_effort},
        text={"verbosity": verbosity, "format": {"type": "text"}},
        input=prompt,
    )
    if tools is not None:
        kwargs["tools"] = tools

    if not stream:
        return await c.responses.create(**kwargs)

    # Streaming path: iterate SSE events and then fetch the final consolidated response
    # Note: event types like "response.output_text.delta" and "response.completed" are expected. :contentReference[oaicite:3]{index=3}
    try:
        async with c.responses.stream(**kwargs) as stream_obj:
            # Optional: surface the response id when created (helps correlate with platform logs)
            async for event in stream_obj:
                et = getattr(event, "type", "")
                if et == "response.created":
                    try:
                        rid = getattr(getattr(event, "response", None), "id", None)
                        if rid:
                            print(f"[stream] response created id={rid}")
                    except Exception:
                        pass
                elif et == "response.error":
                    # The SDK will raise on get_final_response, but we can note early
                    print("[stream] response.error event observed")
                    # continue to let get_final_response raise a structured error
                elif et == "response.completed":
                    # No-op; final response will be obtained below
                    pass

            final = await stream_obj.get_final_response()
            return final
    except Exception:
        # Fall back to non-streaming in case the installed SDK doesn't support .stream (older versions)
        return await c.responses.create(**kwargs)


async def _complete_once_async(
    client: AsyncOpenAI,
    user_prompt: str,
    *,
    model: str,
    reasoning_effort: str,
    verbosity: str,
    per_request_timeout: Optional[float] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    stream: bool = False,
) -> CompletionResult:
    """
    Asynchronous single completion. If stream=True, uses SSE and then composes the final result.
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
        tools=tools,
        stream=stream,
    )
    text = _extract_text(resp) or "[empty]"
    rt, ot, tt = _extract_usage(resp)
    print("Single request completed")
    return CompletionResult(text=text, reasoning_tokens=rt, output_tokens=ot, total_tokens=tt)


# ====================================================
# Parallel generation with bounded concurrency + backoff
# (with RUNS/ persistence and run-count adjustment)
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
    runs_dir: str = RUNS_DIR,
    generation_tools: Optional[List[Dict[str, Any]]] = None,
    use_stream: bool = True,
) -> List[CompletionResult]:
    """
    Execute up to n total completions, but subtract any existing RUNS/audit_* files first.
    Each newly produced completion is written to RUNS/audit_{i}.

    Returns a list of CompletionResult for the completions performed in THIS call
    (length can be less than n if RUNS/ already contains audits).
    """
    _ensure_runs_dir(runs_dir)
    start_index, already_existing, to_run_now = _next_run_index_and_remaining(n, runs_dir)

    if to_run_now <= 0:
        print(f"[gen] Nothing to do. {already_existing} runs already in '{runs_dir}'.")
        return []

    sem = asyncio.Semaphore(max(1, min(max_concurrency, to_run_now)))

    async def _worker(k: int) -> CompletionResult:
        """
        k is 0-based index among the new runs that will be executed in THIS invocation.
        """
        attempt = 0
        run_index = start_index + k  # 1-based absolute run id across invocations
        while True:
            try:
                async with sem:
                    print(f"[gen] run {run_index} starting")
                    res = await _complete_once_async(
                        client,
                        prompt,
                        model=model,
                        reasoning_effort=reasoning_effort,
                        verbosity=verbosity,
                        per_request_timeout=per_request_timeout,
                        tools=generation_tools,
                        stream=use_stream,
                    )
                    _write_audit_file(run_index, res.text, runs_dir)
                    print(f"[gen] run {run_index} completed -> {os.path.join(runs_dir, f'audit_{run_index}')}")
                    return res
            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
                httpx.TimeoutException,
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
            ):
                if attempt >= app_retries:
                    raise
                delay = base_backoff_s * (2 ** attempt) + random.uniform(0.0, 0.5)
                await asyncio.sleep(delay)
                attempt += 1

    tasks = [asyncio.create_task(_worker(i)) for i in range(to_run_now)]
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
        tools=None,      # no web_search needed for merging
        stream=False,    # non-streaming is fine (short response)
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


# ====================================================
# Merge everything from RUNS/ (pad to power-of-two)
# ====================================================

async def merge_all_runs(
    *,
    client: AsyncOpenAI,
    max_concurrency: int,
    model: str,
    reasoning_effort: str = "medium",
    verbosity: str = "medium",
    per_request_timeout: Optional[float] = None,
    runs_dir: str = RUNS_DIR,
    write_merged_to: Optional[str] = os.path.join(RUNS_DIR, "merged.txt"),
) -> CompletionResult:
    """
    Load all RUNS/audit_{i} files, pad to a power-of-two by duplicating the last entry,
    then perform hierarchical_merge. Optionally write the merged output to RUNS/merged.txt.
    """
    _ensure_runs_dir(runs_dir)
    texts = _read_all_audit_texts(runs_dir)
    if not texts:
        return CompletionResult(text="", reasoning_tokens=0, output_tokens=0, total_tokens=0)

    texts_p2 = _pad_to_power_of_two(texts)
    result = await hierarchical_merge(
        client=client,
        texts=texts_p2,
        max_concurrency=max_concurrency,
        model=model,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        per_request_timeout=per_request_timeout,
    )

    if write_merged_to:
        # Atomic write
        tmp = write_merged_to + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(result.text or "")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, write_merged_to)

    return result

