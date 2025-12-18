from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from .prompts import SYSTEM_PROMPT
from .utils import (
    API_URL,
    CompletionResult,
    OpenAIResponseError,
    RUNS_DIR,
    _ensure_runs_dir,
    _extract_error_message,
    _extract_output_text,
    _extract_usage,
    _make_headers,
    _next_run_index_and_remaining,
    _random_string,
    _safe_request,
    _start_response_job,
    _write_audit_file,
)


def generate_runs(
    *,
    api_key: str,
    prompt: str,
    n: int,
    model: str,
    reasoning_effort: str,
    verbosity: str,
    per_request_timeout: Optional[float] = None,  # not used directly; kept for API compatibility
    app_retries: int = 4,  # used for start + poll retries
    base_backoff_s: float = 1.0,
    runs_dir: str = RUNS_DIR,
    generation_tools: Optional[List[Dict[str, Any]]] = None,
    system_prompt: str = SYSTEM_PROMPT,
    poll_interval_sec: float = 4.0,
) -> List[CompletionResult]:
    _ensure_runs_dir(runs_dir)
    start_index, already_existing, to_run_now = _next_run_index_and_remaining(n, runs_dir)

    if to_run_now <= 0:
        print(f"[gen] Nothing to do. {already_existing} runs already in '{runs_dir}'.")
        return []

    headers = _make_headers(api_key)

    jobs: List[Tuple[int, str]] = []  # (run_index, resp_id)
    failed_to_start: Dict[int, str] = {}  # run_index -> reason

    for k in range(to_run_now):
        run_index = start_index + k
        comment = _random_string(5, 50)
        final_prompt = f"{system_prompt}\n\n' IGNORE THIS COMMENT: {comment}\n\n{prompt}"

        payload: Dict[str, Any] = {
            "model": model,
            "tools": [{"type": "web_search"}],  # web_search for opening links
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

    pending = {resp_id for (_, resp_id) in jobs}
    id_to_index = {resp_id: run_index for (run_index, resp_id) in jobs}
    results_by_id: Dict[str, CompletionResult] = {}
    failures_by_id: Dict[str, str] = {}  # resp_id -> reason
    poll_fail_counts: Dict[str, int] = {}  # resp_id -> number of poll failures

    terminal_statuses = {"completed", "failed", "cancelled", "incomplete"}
    transient_statuses = {"queued", "in_progress"}

    print(f"[gen] Polling {len(pending)} runs until completion.")

    while pending:
        for resp_id in list(pending):
            run_index = id_to_index[resp_id]

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

            msg = f"unknown terminal status={status!r}"
            print(f"[gen] run {run_index} has {msg}; marking as failed.")
            failures_by_id[resp_id] = msg
            pending.remove(resp_id)

        if pending:
            time.sleep(poll_interval_sec)

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
        print(
            "[gen] Some runs did not complete successfully; they will be retried on the next invocation "
            "until the desired total number of RUNS/audit_* files is reached."
        )

    return results

