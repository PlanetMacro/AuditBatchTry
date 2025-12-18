from __future__ import annotations

import os
import random
import re
import time
from typing import Any, Dict, List, Tuple

from .prompts import ISSUE_FORMAT_SYSTEM_PROMPT
from .utils import (
    API_URL,
    CompletionResult,
    OpenAIResponseError,
    RUNS_DIR,
    _create_and_poll,
    _ensure_runs_dir,
    _extract_error_message,
    _extract_output_text,
    _extract_usage,
    _make_headers,
    _random_string,
    _safe_request,
    _start_response_job,
)


def _split_merged_issue_blocks(merged_text: str) -> List[Tuple[int, str]]:
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
            continue
        try:
            num = int(m.group(1))
        except ValueError:
            continue
        issues.append((num, block))

    issues.sort(key=lambda t: t[0])
    return issues


def _formatted_issue_path(issue_number: int, runs_dir: str = RUNS_DIR) -> str:
    _ensure_runs_dir(runs_dir)
    return os.path.join(runs_dir, f"formatted_issue_{issue_number}.txt")


def _renumber_issue_block(text: str, new_number: int) -> str:
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
    tmp = final_output_path + ".tmp"
    first_written = False
    next_number = 1

    with open(tmp, "w", encoding="utf-8") as out:
        for issue_number, _ in sorted(issues, key=lambda t: t[0]):
            path = _formatted_issue_path(issue_number, runs_dir)
            if not os.path.exists(path):
                continue

            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()

            if not txt:
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
    max_parallel_issues: int = 8,
    issue_format_system_prompt: str = ISSUE_FORMAT_SYSTEM_PROMPT,
) -> CompletionResult:
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

    pending_issues: List[Tuple[int, str]] = []
    for issue_number, block in issues:
        path = _formatted_issue_path(issue_number, runs_dir)
        if os.path.exists(path):
            print(f"[format] Issue {issue_number} already formatted; skipping.")
            continue
        pending_issues.append((issue_number, block))

    total_to_format = len(pending_issues)
    if total_to_format == 0:
        print("[format] All issues already formatted; rebuilding final report.")
        _rebuild_final_report_from_issue_files(issues, runs_dir, final_output_path)
        if os.path.exists(final_output_path):
            with open(final_output_path, "r", encoding="utf-8") as f:
                full_text = f.read()
        else:
            full_text = ""
        return CompletionResult(text=full_text, reasoning_tokens=0, output_tokens=0, total_tokens=0)

    print(f"[format] {total_to_format} issues require formatting in this run.")
    formatted_now = 0

    app_retries = 4
    base_backoff_s = 1.0

    terminal_statuses = {"completed", "failed", "cancelled", "incomplete"}
    transient_statuses = {"queued", "in_progress"}

    idx = 0
    while idx < len(pending_issues):
        chunk = pending_issues[idx : idx + max_parallel_issues]
        idx += max_parallel_issues

        print(
            f"[format] Starting formatting chunk of {len(chunk)} issues "
            f"(formatted so far: {formatted_now}/{total_to_format})."
        )

        jobs: Dict[str, int] = {}  # resp_id -> issue_number
        poll_fail_counts: Dict[str, int] = {}

        for issue_number, block in chunk:
            comment = _random_string(5, 50)
            prompt = (
                f"{issue_format_system_prompt}\n\n"
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
                "tools": [{"type": "web_search"}],
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

            attempt = 0
            while True:
                try:
                    resp_id = _start_response_job(headers, payload)
                    print(f"[format] started formatting issue {issue_number} -> response_id={resp_id}")
                    jobs[resp_id] = issue_number
                    break
                except OpenAIResponseError as e:
                    attempt += 1
                    if attempt > app_retries:
                        print(
                            f"[format] giving up starting formatting for issue {issue_number} "
                            f"after {app_retries} retries: {e}. Will retry on next run."
                        )
                        break
                    delay = base_backoff_s * (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
                    print(
                        f"[format] start error for issue {issue_number}, "
                        f"retry {attempt}/{app_retries} after {delay:.2f}s: {e}"
                    )
                    time.sleep(delay)

        pending_resp_ids = set(jobs.keys())
        if pending_resp_ids:
            print(
                f"[format] Polling {len(pending_resp_ids)} formatting jobs in current chunk "
                f"(max_parallel_issues={max_parallel_issues})."
            )

        while pending_resp_ids:
            for resp_id in list(pending_resp_ids):
                issue_number = jobs[resp_id]

                try:
                    data = _safe_request("GET", f"{API_URL}/{resp_id}", headers=headers)
                except OpenAIResponseError as e:
                    count = poll_fail_counts.get(resp_id, 0) + 1
                    poll_fail_counts[resp_id] = count

                    if count <= app_retries:
                        delay = base_backoff_s * (2 ** (count - 1)) + random.uniform(0.0, 0.5)
                        print(
                            f"[format] polling error for response {resp_id}, "
                            f"retry {count}/{app_retries} after {delay:.2f}s: {e}"
                        )
                        time.sleep(delay)
                        continue

                    print(
                        f"[format] polling failed for response {resp_id} after {count} attempts: {e}. "
                        "Will retry on next run."
                    )
                    pending_resp_ids.remove(resp_id)
                    continue

                status = data.get("status") or "unknown"

                if status in transient_statuses:
                    continue

                path = _formatted_issue_path(issue_number, runs_dir)

                if status == "completed":
                    text = _extract_output_text(data) or ""
                    rt, ot, tt = _extract_usage(data)
                    agg_rt += rt
                    agg_ot += ot
                    agg_tt += tt

                    tmp = path + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        f.write(text.strip())
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp, path)
                    print(f"[format] Issue {issue_number} formatted and stored at '{path}'.")

                    _rebuild_final_report_from_issue_files(issues, runs_dir, final_output_path)
                    print(f"[format] Final report updated at '{final_output_path}'.")

                    formatted_now += 1
                    print(f"[format] Progress: formatted {formatted_now}/{total_to_format} issues in this run.")

                    pending_resp_ids.remove(resp_id)
                    continue

                if status in terminal_statuses:
                    msg = _extract_error_message(data)
                    print(
                        f"[format] Formatting issue {issue_number} ended with status={status}: {msg}. "
                        "Will retry on next run."
                    )
                    pending_resp_ids.remove(resp_id)
                    continue

                print(
                    f"[format] Formatting issue {issue_number} returned unknown terminal status={status!r}. "
                    "Will retry on next run."
                )
                pending_resp_ids.remove(resp_id)

            if pending_resp_ids:
                time.sleep(poll_interval_sec)

    if os.path.exists(final_output_path):
        with open(final_output_path, "r", encoding="utf-8") as f:
            full_text = f.read()
    else:
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
    issue_format_system_prompt: str = ISSUE_FORMAT_SYSTEM_PROMPT,
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
        f"{issue_format_system_prompt}\n\n"
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
