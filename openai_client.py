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
    "- You are a senior software security researcher analyzing code for software security issues.\n"
    "- You analyze the given problem from multiple angles.\n"
    "Rules:\n"
    "- Issues MUST have the exact following format: Issue: <NUMBER> <LINEBREAK> Location: <COPY_OF_LINE> <LINEBREAK> Description: <TEXT>\n"
    "- If you can not open given links to the codebase or provided context, you MUST include a notification and abort."
    "- if the provided information is not enough to reason about a potential security issue, you should TRY to gather the information online."
    " if you can not access the required information online, you MUST add a list including the missing information in the Issue's description.\n"
    "- Output ONLY the formatted issues, exactly in the structure above, with a single blank line "
    " between issues and no leading or trailing commentary.\n"
    "- Do NOT invent issues that are not clearly implied by the input.\n"
    "- Do NOT include any meta-explanations about this template or your reasoning process.\n"
    "- You MUST include the location as a correctly formatted link to the exact line of code in the codebase where the issue occurs.\n"
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
    " - When merging duplicates, combine the ideas from both issues, but make sure to not repeat content\n" 
    " - When merged issues have different locations, list both\n"
    " - Renumber issues consecutively starting from 1 in the final output.\n"
    " - Place a single blank line between issues. Do NOT add any commentary, headings, code fences, or extra text before/after the list."
)

# ----------------------
# System prompt used for the *final formatting / triage* step
# ----------------------
ISSUE_FORMAT_SYSTEM_PROMPT = (
    "You are a senior software security researcher. You receive a merged flat list of potential issues, "
    "each in the format:\n"
    "  Issue: <NUMBER>\n"
    "  Location: <COPY_OF_LINE>\n"
    "  Description: <TEXT>\n\n"
    "Your task:\n"
    "  1. Open the link and double check the issue in the provided code behind the link. If the link"
    " does not work, TRY to recover it, if this fails, you MUST RETURN an ERROR and ABORT.\n"
    "  2. Decide for each issue whether it describes an attackable security vulnerability or "
    "     realistic abuse scenario.\n"
    "  3. If an issue is not attackable (e.g., pure style, documentation quality, performance micro-"
    "     optimization, extremely theoretical, or already fully mitigated), DROP IT entirely.\n"
    "  4. For each remaining issue, rewrite it into the following final format:\n\n"
    "Final output format (repeat this whole block for each remaining issue, with a single blank "
    "line between issues and no extra text):\n"
    "Issue <NUMBER>: <DESCRIPTIVE HEADLINE>\n"
    "Location\n"
    "<A link or description of the location in the code / documentation where the issue exists>\n"
    "Synopsis\n"
    "<A concise description of the essential vulnerability, without assuming other components "
    " mitigate it unless clearly stated in the input>\n"
    "Impact\n"
    "<Low | Medium | High>\n"
    "<1–3 sentences describing what benefit an attacker gains by successfully exploiting the "
    " vulnerability, stated conservatively and without overestimating real-world impact>\n"
    "Feasibility\n"
    "<Low | Medium | High>\n"
    "<1–3 sentences estimating how difficult it is to perform the attack in practice, assuming "
    " the preconditions are met>\n"
    "Severity\n"
    "<Low | Medium | High | Critical>\n"
    "<1–3 sentences justifying this severity based on combining impact and feasibility, following "
    " an OWASP-style risk severity matrix>\n"
    "Preconditions\n"
    "<Conditions necessary for the vulnerability to be exploitable (configuration, roles, data, etc.)>\n"
    "Technical Details\n"
    "<Implementation details and a specific process an attacker would use to leverage the issue, "
    " referring to the Location where useful>\n"
    "Mitigation\n"
    "<Immediate steps current users or operators may take to protect themselves>\n"
    "Remediation\n"
    "<Development and architectural changes that will prevent, detect, or otherwise mitigate the "
    " vulnerability in future releases>\n\n"
    "Rules:\n"
    "- Output ONLY the formatted issues, exactly in the structure above, with a single blank line "
    " between issues and no leading or trailing commentary."
    "  DO keep the text as short as possible.\n"
    "  DO use flowing text (except for location, which should be a list).\n\n"
    "  DO NOT repeat content inside a single issue\n"
    "- Do NOT invent issues that are not clearly implied by the input.\n"
    "- Do NOT output non-attackable issues.\n"
    "- Do NOT output issues that have negligible success probabilities.\n"
    "- Do NOT output issues that are only relevant for system changes/updates.\n"
    "  DO NOT use tables, bullet points, lists ect\n"
    "- Do NOT include any meta-explanations about this template or your reasoning process.\n"
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

def is_merged_up_to_date(runs_dir: str = RUNS_DIR, merged_path: Optional[str] = None) -> bool:
    """
    Return True if merged_path exists and is at least as new as every audit_* file.

    This lets the caller skip re-merging when no new audit reports have been produced
    since the last merge.
    """
    if merged_path is None:
        merged_path = os.path.join(runs_dir, "merged.txt")

    if not os.path.exists(merged_path):
        return False
    if not os.path.isdir(runs_dir):
        return False

    merged_mtime = os.path.getmtime(merged_path)

    # If any audit file is newer than merged.txt, the merged output is stale.
    for _, fp in _list_audit_files(runs_dir):
        if os.path.getmtime(fp) > merged_mtime:
            return False

    return True


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
    per_request_timeout: Optional[float] = None,  # not used directly; kept for API compatibility
    app_retries: int = 4,                         # used for start + poll retries
    base_backoff_s: float = 1.0,
    runs_dir: str = RUNS_DIR,
    generation_tools: Optional[List[Dict[str, Any]]] = None,
    poll_interval_sec: float = 4.0,
) -> List[CompletionResult]:
    """
    Generate up to n total audit runs for the given prompt.

    - Uses Responses API with background=True + polling.
    - Writes each run to RUNS/audit_{i} as soon as that response completes.
    - Does NOT raise on per-run errors (failed status, network issues); they are logged and skipped.
    - Returns CompletionResult entries only for runs that completed successfully in THIS invocation.
    """
    _ensure_runs_dir(runs_dir)
    start_index, already_existing, to_run_now = _next_run_index_and_remaining(n, runs_dir)

    if to_run_now <= 0:
        print(f"[gen] Nothing to do. {already_existing} runs already in '{runs_dir}'.")
        return []

    headers = _make_headers(api_key)

    jobs: List[Tuple[int, str]] = []      # (run_index, resp_id)
    failed_to_start: Dict[int, str] = {}  # run_index -> reason

    # 1) Start all jobs (background=True), with retries per job
    for k in range(to_run_now):
        run_index = start_index + k
        comment = _random_string(5, 50)
        final_prompt = f"{SYSTEM_PROMPT}\n\n' IGNORE THIS COMMENT: {comment}\n\n{prompt}"

        payload: Dict[str, Any] = {
            "model": model,
            "tools": [ {"type": "web_search"} ],   # web_search for opening links
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

        attempt = 0
        while True:
            try:
                resp_id = _start_response_job(headers, payload)
                print(f"[gen] started run {run_index} -> response_id={resp_id}")
                jobs.append((run_index, resp_id))
                break
            except OpenAIResponseError as e:
                attempt += 1
                if attempt > app_retries:
                    msg = f"start failed after {app_retries} retries: {e}"
                    print(f"[gen] giving up starting run {run_index}: {msg}")
                    failed_to_start[run_index] = msg
                    break
                delay = base_backoff_s * (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
                print(
                    f"[gen] start error for run {run_index}, "
                    f"retry {attempt}/{app_retries} after {delay:.2f}s: {e}"
                )
                time.sleep(delay)

    if not jobs:
        print("[gen] No runs started successfully; generation produced 0 completions.")
        return []

    # 2) Poll all jobs; write each audit file as soon as it completes
    pending = {resp_id for (_, resp_id) in jobs}
    id_to_index = {resp_id: run_index for (run_index, resp_id) in jobs}
    results_by_id: Dict[str, CompletionResult] = {}
    failures_by_id: Dict[str, str] = {}       # resp_id -> reason
    poll_fail_counts: Dict[str, int] = {}     # resp_id -> number of poll failures

    terminal_statuses = {"completed", "failed", "cancelled", "incomplete"}
    transient_statuses = {"queued", "in_progress"}

    print(f"[gen] Polling {len(pending)} runs until completion.")

    while pending:
        for resp_id in list(pending):
            run_index = id_to_index[resp_id]

            # Network / HTTP errors during polling
            try:
                data = _safe_request("GET", f"{API_URL}/{resp_id}", headers=headers)
            except OpenAIResponseError as e:
                count = poll_fail_counts.get(resp_id, 0) + 1
                poll_fail_counts[resp_id] = count

                if count <= app_retries:
                    delay = base_backoff_s * (2 ** (count - 1)) + random.uniform(0.0, 0.5)
                    print(
                        f"[gen] polling error for run {run_index}, "
                        f"retry {count}/{app_retries} after {delay:.2f}s: {e}"
                    )
                    time.sleep(delay)
                    continue

                msg = f"polling failed after {count} attempts: {e}"
                print(f"[gen] giving up on run {run_index}: {msg}")
                failures_by_id[resp_id] = msg
                pending.remove(resp_id)
                continue

            status = data.get("status") or "unknown"

            if status in transient_statuses:
                # still queued or in_progress
                continue

            if status == "completed":
                text = _extract_output_text(data) or "[empty]"
                rt, ot, tt = _extract_usage(data)

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
                print(f"[gen] run {run_index} ended with status={status}: {msg}")
                failures_by_id[resp_id] = msg
                pending.remove(resp_id)
                continue

            # Unknown terminal-ish status
            msg = f"unknown terminal status={status!r}"
            print(f"[gen] run {run_index} has {msg}; marking as failed.")
            failures_by_id[resp_id] = msg
            pending.remove(resp_id)

        if pending:
            time.sleep(poll_interval_sec)

    # 3) Build result list in run-index order from successful runs
    results: List[CompletionResult] = []
    for run_index, resp_id in jobs:
        res = results_by_id.get(resp_id)
        if res is None:
            continue
        results.append(res)

    total_started = len(jobs)
    total_success = len(results)
    total_failed = len(failures_by_id) + len(failed_to_start)

    print(
        f"[gen] Summary: started={total_started}, "
        f"completed_ok={total_success}, "
        f"failed_or_not_started={total_failed} "
        f"(existing_before={already_existing})."
    )

    if total_failed > 0:
        print("[gen] Some runs did not complete successfully; they will be retried on the next invocation "
              "until the desired total number of RUNS/audit_* files is reached.")

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

def _split_merged_issue_blocks(merged_text: str) -> List[Tuple[int, str]]:
    """
    Split the merged flat list into individual issue blocks.

    Each block starts with:
      Issue: <NUMBER>
    and is separated by at least one blank line.

    Returns a list of (issue_number, block_text), sorted by issue_number.
    """
    if not merged_text or not merged_text.strip():
        return []

    blocks = re.split(r"\n\s*\n(?=Issue:\s*\d+)", merged_text.strip())
    issues: List[Tuple[int, str]] = []

    for b in blocks:
        block = b.strip()
        if not block:
            continue
        m = re.match(r"Issue:\s*(\d+)", block)
        if not m:
            # Block does not start with "Issue: N" – ignore it.
            continue
        try:
            num = int(m.group(1))
        except ValueError:
            continue
        issues.append((num, block))

    issues.sort(key=lambda t: t[0])
    return issues


def _formatted_issue_path(issue_number: int, runs_dir: str = RUNS_DIR) -> str:
    """
    Path where the formatted version of a single merged issue is stored.

    One file per original merged issue:
      RUNS/formatted_issue_<N>.txt
    """
    _ensure_runs_dir(runs_dir)
    return os.path.join(runs_dir, f"formatted_issue_{issue_number}.txt")


def _renumber_issue_block(text: str, new_number: int) -> str:
    """
    Rewrite the heading line of a formatted issue to "Issue <new_number>:".

    Assumes the first line is of the form "Issue <NUMBER>: ...".
    """
    lines = text.splitlines()
    if not lines:
        return ""
    first = lines[0]
    new_first = re.sub(r"^Issue\s+\d+\s*:", f"Issue {new_number}:", first)
    lines[0] = new_first
    return "\n".join(lines)


def _rebuild_final_report_from_issue_files(
    issues: List[Tuple[int, str]],
    runs_dir: str,
    final_output_path: str,
) -> None:
    """
    Rebuild the final report from all existing formatted_issue_* files.

    - Issues are ordered by their original merged issue number.
    - Only non-empty formatted files are included (non-attackable issues
      are represented by empty files and are skipped).
    - Issues are renumbered consecutively from 1 in the final output.
    - Write is atomic via a temporary file + rename.
    """
    tmp = final_output_path + ".tmp"
    first_written = False
    next_number = 1

    with open(tmp, "w", encoding="utf-8") as out:
        for issue_number, _ in sorted(issues, key=lambda t: t[0]):
            path = _formatted_issue_path(issue_number, runs_dir)
            if not os.path.exists(path):
                # Not formatted yet
                continue

            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()

            if not txt:
                # This issue was triaged away as non-attackable.
                continue

            renumbered = _renumber_issue_block(txt, next_number).strip()
            if not renumbered:
                continue

            if first_written:
                out.write("\n\n")
            out.write(renumbered)
            first_written = True
            next_number += 1

        out.flush()
        os.fsync(out.fileno())

    os.replace(tmp, final_output_path)


def format_issues_incremental(
    *,
    api_key: str,
    merged_text: str,
    model: str,
    final_output_path: str,
    reasoning_effort: str = "high",
    verbosity: str = "high",
    runs_dir: str = RUNS_DIR,
    poll_interval_sec: float = 4.0,
) -> CompletionResult:
    """
    Incremental, crash-resilient formatter.

    Behavior:
      - Split merged_text into individual issues (Issue: N / Location / Description).
      - For each issue:
          * If RUNS/formatted_issue_<N>.txt exists, it is reused (skipped).
          * Otherwise, send ONLY that issue to the LLM for triage+formatting.
            The result is written immediately to RUNS/formatted_issue_<N>.txt.
            Empty output means "not an attackable issue"; we still create the file.
      - After each newly formatted issue, rebuild final_output_path by concatenating
        all non-empty formatted_issue_* files, renumbering issues from 1.

    On rerun after a crash:
      - Existing formatted_issue_* files are detected and skipped, so only the
        still-missing issues are reformatted.
    """
    if not merged_text or not merged_text.strip():
        print("[format] Empty merged text; nothing to format.")
        full_text = ""
        if os.path.exists(final_output_path):
            with open(final_output_path, "r", encoding="utf-8") as f:
                full_text = f.read()
        return CompletionResult(text=full_text, reasoning_tokens=0, output_tokens=0, total_tokens=0)

    issues = _split_merged_issue_blocks(merged_text)
    if not issues:
        print("[format] No issues could be parsed from merged text.")
        return CompletionResult(text="", reasoning_tokens=0, output_tokens=0, total_tokens=0)

    headers = _make_headers(api_key)

    agg_rt = 0
    agg_ot = 0
    agg_tt = 0

    print(f"[format] Incremental formatting of {len(issues)} issues.")

    for issue_number, block in issues:
        path = _formatted_issue_path(issue_number, runs_dir)
        if os.path.exists(path):
            print(f"[format] Issue {issue_number} already formatted; skipping.")
            continue

        comment = _random_string(5, 50)
        prompt = (
            f"{ISSUE_FORMAT_SYSTEM_PROMPT}\n\n"
            f"' IGNORE THIS COMMENT: {comment}\n\n"
            "Merged issue list:\n"
            f"{block.strip()}\n\n"
            "You are given exactly one issue above.\n"
            "Return ONLY its reformatted version in the exact output format described above.\n"
            "If the issue is not an attackable security vulnerability, return an empty response.\n"
            "Since there is only a single issue in this request, use 'Issue 1:' as the heading; "
            "the final numbering will be adjusted later."
        )

        payload: Dict[str, Any] = {
            "model": model,
            "tools": [ {"type": "web_search"} ],   # web_search for opening links
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

        try:
            resp_json = _create_and_poll(headers, payload, poll_interval_sec=poll_interval_sec)
        except OpenAIResponseError as e:
            print(f"[format] Error while formatting issue {issue_number}: {e}. Will retry on next run.")
            # Do not create a formatted_issue file; this issue will be retried.
            continue

        text = _extract_output_text(resp_json) or ""
        rt, ot, tt = _extract_usage(resp_json)
        agg_rt += rt
        agg_ot += ot
        agg_tt += tt

        # Persist this issue's result (even if empty) atomically.
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text.strip())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        print(f"[format] Issue {issue_number} formatted and stored at '{path}'.")

        # Rebuild the final report after each new issue.
        _rebuild_final_report_from_issue_files(issues, runs_dir, final_output_path)
        print(f"[format] Final report updated at '{final_output_path}'.")

    # Ensure final report is up to date even if no new issues were formatted.
    if os.path.exists(final_output_path):
        with open(final_output_path, "r", encoding="utf-8") as f:
            full_text = f.read()
    else:
        # Try to build it once from whatever formatted_issue_* files we have.
        _rebuild_final_report_from_issue_files(issues, runs_dir, final_output_path)
        if os.path.exists(final_output_path):
            with open(final_output_path, "r", encoding="utf-8") as f:
                full_text = f.read()
        else:
            full_text = ""

    print("[format] Incremental issue formatting completed.")
    return CompletionResult(text=full_text, reasoning_tokens=agg_rt, output_tokens=agg_ot, total_tokens=agg_tt)



def format_issues(
    *,
    api_key: str,
    merged_text: str,
    model: str,
    reasoning_effort: str = "high",
    verbosity: str = "high",
    poll_interval_sec: float = 4.0,
) -> CompletionResult:
    """
    Take a merged flat list of issues (Issue: N / Location: / Description: ...)
    and turn it into a structured security report.

    The model:
      - Iterates over all issues in merged_text,
      - Drops anything that is not an attackable vulnerability,
      - Rewrites the remaining ones into the detailed security issue template.

    Returns a CompletionResult with token usage for accounting.
    """
    if not merged_text or not merged_text.strip():
        print("[format] Empty merged text; nothing to format.")
        return CompletionResult(text="", reasoning_tokens=0, output_tokens=0, total_tokens=0)

    headers = _make_headers(api_key)

    comment = _random_string(5, 50)
    prompt = (
        f"{ISSUE_FORMAT_SYSTEM_PROMPT}\n\n"
        f"' IGNORE THIS COMMENT: {comment}\n\n"
        "Merged issue list:\n"
        f"{merged_text.strip()}\n\n"
        "Return ONLY the reformatted issues in the exact output format described above."
    )

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
    print("[format] Issue formatting completed.")
    return CompletionResult(text=text, reasoning_tokens=rt, output_tokens=ot, total_tokens=tt)

