from __future__ import annotations

import os
import random
import re
import string
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import requests

# ----------------------
# System prompt used for the *generation* step (kept as-is)
# ----------------------
SYSTEM_PROMPT = (
    "You are an assistant that has a high degree of freedom in answering from multiple angles"
    "Issues should have the exact following format: Issue: <NUMBER> <LINEBREAK> Location: <COPY_OF_LINE> <LINEBREAK> Description: <TEXT>"
)

# ----------------------
# System prompt used for the *merging* step
# ----------------------
MERGE_SYSTEM_PROMPT = (
    "You are an assistant that merges two audit reports of issues into a single deduplicated list.\n"
    "Output MUST contain only issues in exactly this format:\n"
    "Issue: <NUMBER>\nLocation: <COPY_OF_LINE>\nDescription: <TEXT>\n"
    "Rules:\n"
    " - Compute the semantic set UNION of issues from both reports.\n"
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

API_URL = "https://api.openai.com/v1/responses"


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
# HTTP + Responses API helpers (sync, response-loop style)
# ====================================================

class OpenAIResponseError(Exception):
    """Raised when the OpenAI Responses API returns an error or the network request fails."""


def _make_headers(api_key: str) -> Dict[str, str]:
    if not api_key:
        raise OpenAIResponseError("OpenAI API key is missing.")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _safe_request(method: str, url: str, headers: Dict[str, str], **kwargs) -> Dict[str, Any]:
    """
    Perform an HTTP request and return parsed JSON on success.
    Raises OpenAIResponseError on any HTTP or network error.
    """
    try:
        resp = requests.request(method, url, headers=headers, **kwargs)
    except requests.exceptions.RequestException as e:
        raise OpenAIResponseError(f"Request failed: {e}") from e

    if resp.status_code != 200:
        try:
            data = resp.json()
        except ValueError:
            data = None

        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict) and err.get("message"):
                raise OpenAIResponseError(str(err["message"]))
        raise OpenAIResponseError(f"{resp.status_code} {resp.reason}")

    try:
        return resp.json()
    except ValueError as e:
        raise OpenAIResponseError("Failed to decode JSON from Responses API") from e


def _extract_output_text(response_json: Dict[str, Any]) -> str:
    """
    Extract assistant text from the Responses API JSON structure.
    """
    out = response_json.get("output")
    if not isinstance(out, list):
        return ""
    texts: List[str] = []
    for item in out:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "output_text":
                t = part.get("text")
                if isinstance(t, str):
                    texts.append(t)
    return "\n".join(texts).strip()


def _extract_usage(response_json: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Extract (reasoning_tokens, output_tokens, total_tokens) from the Responses API JSON.
    """
    usage = response_json.get("usage") or {}
    details = usage.get("output_tokens_details") or {}
    rt = int(details.get("reasoning_tokens") or 0)
    ot = int(usage.get("output_tokens") or 0)
    tt = int(usage.get("total_tokens") or 0)
    return rt, ot, tt


def _extract_error_message(response_json: Dict[str, Any]) -> str:
    """
    Try to extract a meaningful error/incomplete message from a Responses API JSON.
    """
    err = response_json.get("error")
    if isinstance(err, dict):
        msg = err.get("message")
        if isinstance(msg, str) and msg:
            return msg

    inc = response_json.get("incomplete_details")
    if isinstance(inc, dict):
        msg = inc.get("message") or inc.get("reason")
        if isinstance(msg, str) and msg:
            return msg

    status = response_json.get("status")
    if isinstance(status, str):
        return f"Request finished with status={status!r} but no error message was provided."
    return "Unknown error"


def _start_response_job(headers: Dict[str, str], payload: Dict[str, Any]) -> str:
    """
    Create a Responses job (background=True) and return the response ID.
    """
    body = dict(payload)
    body.setdefault("background", True)
    body.setdefault("store", True)

    data = _safe_request("POST", API_URL, headers=headers, json=body)
    resp_id = data.get("id")
    if not isinstance(resp_id, str) or not resp_id:
        raise OpenAIResponseError("No response ID returned from Responses API create call.")
    return resp_id


def _poll_response(headers: Dict[str, str], resp_id: str, poll_interval_sec: float = 4.0) -> Dict[str, Any]:
    """
    Poll GET /responses/{id} until a terminal status is reached. Returns final JSON on success.
    Raises OpenAIResponseError if the job fails, is cancelled, or is incomplete.
    """
    terminal_statuses = {"completed", "failed", "cancelled", "incomplete"}
    transient_statuses = {"queued", "in_progress"}

    while True:
        data = _safe_request("GET", f"{API_URL}/{resp_id}", headers=headers)
        status = data.get("status") or "unknown"

        if status in transient_statuses:
            time.sleep(poll_interval_sec)
            continue

        if status == "completed":
            return data

        if status in terminal_statuses:
            msg = _extract_error_message(data)
            raise OpenAIResponseError(f"Response {resp_id} ended with status={status}: {msg}")

        raise OpenAIResponseError(f"Response {resp_id} has unknown terminal status={status!r}.")


def _create_and_poll(headers: Dict[str, str], payload: Dict[str, Any], poll_interval_sec: float = 4.0) -> Dict[str, Any]:
    """
    Utility for merge calls where we don't need to start multiple jobs before polling.
    """
    resp_id = _start_response_job(headers, payload)
    return _poll_response(headers, resp_id, poll_interval_sec=poll_interval_sec)


# ====================================================
# Generation: get N audit runs (response-loop, no local async)
# ====================================================

def generate_runs(
    *,
    api_key: str,
    prompt: str,
    n: int,
    model: str,
    reasoning_effort: str,
    verbosity: str,
    runs_dir: str = RUNS_DIR,
    generation_tools: Optional[List[Dict[str, Any]]] = None,
    poll_interval_sec: float = 4.0,
) -> List[CompletionResult]:
    """
    Generate up to n total audit runs for the given prompt.

    - Uses Responses API with background=True + polling.
    - Writes each run to RUNS/audit_{i} as soon as that response completes.
    - Returns CompletionResult entries for the runs created in THIS invocation
      (existing audit_* files are detected and not regenerated).
    """
    _ensure_runs_dir(runs_dir)
    start_index, already_existing, to_run_now = _next_run_index_and_remaining(n, runs_dir)

    if to_run_now <= 0:
        print(f"[gen] Nothing to do. {already_existing} runs already in '{runs_dir}'.")
        return []

    headers = _make_headers(api_key)

    # 1) Start all jobs (background=True), collecting IDs
    jobs: List[Tuple[int, str]] = []  # (run_index, resp_id)
    for k in range(to_run_now):
        run_index = start_index + k
        comment = _random_string(5, 50)
        final_prompt = f"{SYSTEM_PROMPT}\n\n' IGNORE THIS COMMENT: {comment}\n\n{prompt}"

        payload: Dict[str, Any] = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": final_prompt},
                    ],
                }
            ],
            "reasoning": {"effort": reasoning_effort},
            "text": {
                "format": {"type": "text"},
                "verbosity": verbosity,
            },
        }
        if generation_tools:
            payload["tools"] = generation_tools

        resp_id = _start_response_job(headers, payload)
        print(f"[gen] started run {run_index} -> response_id={resp_id}")
        jobs.append((run_index, resp_id))

    # 2) Poll all jobs; write each audit file as soon as it completes
    pending = {resp_id for (_, resp_id) in jobs}
    id_to_index = {resp_id: run_index for (run_index, resp_id) in jobs}
    results_by_id: Dict[str, CompletionResult] = {}

    terminal_statuses = {"completed", "failed", "cancelled", "incomplete"}
    transient_statuses = {"queued", "in_progress"}

    while pending:
        for resp_id in list(pending):
            data = _safe_request("GET", f"{API_URL}/{resp_id}", headers=headers)
            status = data.get("status") or "unknown"

            if status in transient_statuses:
                continue

            if status == "completed":
                run_index = id_to_index[resp_id]
                text = _extract_output_text(data) or "[empty]"
                rt, ot, tt = _extract_usage(data)

                # Persist immediately so this run is safe even if later jobs fail
                _write_audit_file(run_index, text, runs_dir)
                print(f"[gen] run {run_index} completed -> {os.path.join(runs_dir, f'audit_{run_index}')}")

                results_by_id[resp_id] = CompletionResult(
                    text=text,
                    reasoning_tokens=rt,
                    output_tokens=ot,
                    total_tokens=tt,
                )
                pending.remove(resp_id)
                continue

            if status in terminal_statuses:
                msg = _extract_error_message(data)
                # At this point, any runs already written to disk stay there.
                raise OpenAIResponseError(
                    f"[gen] run {id_to_index.get(resp_id)} failed with status={status}: {msg}"
                )

            raise OpenAIResponseError(
                f"[gen] run {id_to_index.get(resp_id)} has unknown terminal status={status!r}"
            )

        if pending:
            time.sleep(poll_interval_sec)

    # 3) Build the result list in run-index order from what we stored per ID
    results: List[CompletionResult] = []
    for run_index, resp_id in jobs:
        res = results_by_id.get(resp_id)
        if res is None:
            # Should not happen if we only exit the loop when pending is empty,
            # but keep a defensive check.
            raise OpenAIResponseError(f"No completion result recorded for response_id={resp_id}")
        results.append(res)

    return results


# ====================================================
# Merging support (sync, via Responses API)
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


def _merge_two(
    *,
    headers: Dict[str, str],
    a: str,
    b: str,
    model: str,
    reasoning_effort: str,
    verbosity: str,
    poll_interval_sec: float = 4.0,
) -> CompletionResult:
    """Merge two issue lists via the LLM, returning the deduplicated union."""
    prompt = _build_merge_prompt(a, b)
    payload: Dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
        "reasoning": {"effort": reasoning_effort},
        "text": {
            "format": {"type": "text"},
            "verbosity": verbosity,
        },
    }
    resp_json = _create_and_poll(headers, payload, poll_interval_sec=poll_interval_sec)
    text = _extract_output_text(resp_json) or "[empty]"
    rt, ot, tt = _extract_usage(resp_json)
    return CompletionResult(text=text, reasoning_tokens=rt, output_tokens=ot, total_tokens=tt)


def hierarchical_merge(
    *,
    api_key: str,
    texts: List[str],
    model: str,
    reasoning_effort: str,
    verbosity: str,
    poll_interval_sec: float = 4.0,
) -> CompletionResult:
    """
    Hierarchically merge a list of issue lists by repeatedly merging pairs (binary tree reduction)
    until a single deduplicated union remains.

    Prints progress information because this can be long-running.
    """
    if not texts:
        return CompletionResult(text="", reasoning_tokens=0, output_tokens=0, total_tokens=0)
    if len(texts) == 1:
        return CompletionResult(text=texts[0], reasoning_tokens=0, output_tokens=0, total_tokens=0)

    headers = _make_headers(api_key)

    agg_rt = 0
    agg_ot = 0
    agg_tt = 0
    current: List[str] = [t or "" for t in texts]

    total_merges = max(0, len(current) - 1)
    completed_merges = 0
    level = 1

    print(f"[merge] Starting hierarchical merge: {len(current)} reports, {total_merges} pairwise merges needed.")

    while len(current) > 1:
        level_input_count = len(current)
        print(f"[merge] Level {level}: {level_input_count} partial reports to merge.")

        next_round: List[str] = []
        merges_this_level = 0

        i = 0
        while i < len(current):
            a = current[i]
            b = current[i + 1] if i + 1 < len(current) else ""
            i += 2

            if not b.strip():
                # Odd one out, carry forward unchanged.
                next_round.append(a)
                continue

            # Perform one merge
            result = _merge_two(
                headers=headers,
                a=a,
                b=b,
                model=model,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
                poll_interval_sec=poll_interval_sec,
            )
            next_round.append(result.text)

            merges_this_level += 1
            completed_merges += 1
            agg_rt += result.reasoning_tokens
            agg_ot += result.output_tokens
            agg_tt += result.total_tokens

            print(
                f"[merge]   completed merge {completed_merges}/{total_merges} "
                f"(level {level}, this level: {merges_this_level})"
            )

        current = next_round
        print(
            f"[merge] Level {level} done: {len(current)} partial reports remain, "
            f"{total_merges - completed_merges} merges left."
        )
        level += 1

    print("[merge] Hierarchical merge complete.")
    return CompletionResult(text=current[0], reasoning_tokens=agg_rt, output_tokens=agg_ot, total_tokens=agg_tt)


# ====================================================
# Merge everything from RUNS/ (pad to power-of-two)
# ====================================================

def merge_all_runs(
    *,
    api_key: str,
    model: str,
    reasoning_effort: str = "medium",
    verbosity: str = "medium",
    runs_dir: str = RUNS_DIR,
    write_merged_to: Optional[str] = os.path.join(RUNS_DIR, "merged.txt"),
    poll_interval_sec: float = 4.0,
) -> CompletionResult:
    """
    Load all RUNS/audit_{i} files, pad to a power-of-two by duplicating the last entry,
    then perform hierarchical_merge. Optionally write the merged output to RUNS/merged.txt.
    """
    _ensure_runs_dir(runs_dir)
    texts = _read_all_audit_texts(runs_dir)
    if not texts:
        print("[merge] No audit_* files found; nothing to merge.")
        return CompletionResult(text="", reasoning_tokens=0, output_tokens=0, total_tokens=0)

    print(f"[merge] Loaded {len(texts)} audit reports from '{runs_dir}'.")
    texts_p2 = _pad_to_power_of_two(texts)

    if len(texts_p2) != len(texts):
        print(f"[merge] Padded reports to power-of-two: {len(texts)} -> {len(texts_p2)} entries.")

    result = hierarchical_merge(
        api_key=api_key,
        texts=texts_p2,
        model=model,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        poll_interval_sec=poll_interval_sec,
    )

    if write_merged_to:
        tmp = write_merged_to + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(result.text or "")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, write_merged_to)
        print(f"[merge] Final merged report written to '{write_merged_to}'.")

    return result

