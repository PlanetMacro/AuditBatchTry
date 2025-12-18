from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, List, Optional

from .prompts import MERGE_SYSTEM_PROMPT
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
    _pad_to_power_of_two,
    _random_string,
    _read_all_audit_texts,
    _safe_request,
    _start_response_job,
)


def _build_merge_prompt(a: str, b: str, merge_system_prompt: str = MERGE_SYSTEM_PROMPT) -> str:
    comment = _random_string(5, 50)
    return (
        f"{merge_system_prompt}\n\n"
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
    merge_system_prompt: str = MERGE_SYSTEM_PROMPT,
    poll_interval_sec: float = 4.0,
) -> CompletionResult:
    prompt = _build_merge_prompt(a, b, merge_system_prompt=merge_system_prompt)
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
    merge_system_prompt: str = MERGE_SYSTEM_PROMPT,
    poll_interval_sec: float = 4.0,
) -> CompletionResult:
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

    app_retries = 4
    base_backoff_s = 1.0

    while len(current) > 1:
        level_input_count = len(current)
        print(f"[merge] Level {level}: {level_input_count} partial reports to merge.")

        num_slots = (len(current) + 1) // 2
        next_round: List[Optional[str]] = [None] * num_slots

        respid_to_slot: Dict[str, int] = {}
        pending = set()
        poll_fail_counts: Dict[str, int] = {}

        slot_index = 0
        i = 0

        while i < len(current):
            a = current[i]
            b = current[i + 1] if i + 1 < len(current) else ""
            i += 2

            if not b.strip():
                next_round[slot_index] = a
                slot_index += 1
                continue

            prompt = _build_merge_prompt(a, b, merge_system_prompt=merge_system_prompt)
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

            resp_id = _start_response_job(headers, payload)
            respid_to_slot[resp_id] = slot_index
            pending.add(resp_id)

            print(f"[merge] started level {level} merge for slot {slot_index} -> response_id={resp_id}")

            slot_index += 1

        merges_this_level = 0

        terminal_statuses = {"completed", "failed", "cancelled", "incomplete"}
        transient_statuses = {"queued", "in_progress"}

        if pending:
            print(f"[merge] Polling {len(pending)} merge jobs at level {level} until completion.")

        while pending:
            for resp_id in list(pending):
                try:
                    data = _safe_request("GET", f"{API_URL}/{resp_id}", headers=headers)
                except OpenAIResponseError as e:
                    count = poll_fail_counts.get(resp_id, 0) + 1
                    poll_fail_counts[resp_id] = count

                    if count <= app_retries:
                        delay = base_backoff_s * (2 ** (count - 1)) + random.uniform(0.0, 0.5)
                        print(
                            f"[merge] polling error for response {resp_id}, "
                            f"retry {count}/{app_retries} after {delay:.2f}s: {e}"
                        )
                        time.sleep(delay)
                        continue

                    print(f"[merge] polling failed for response {resp_id} after {count} attempts: {e}")
                    raise OpenAIResponseError(
                        f"[merge] polling failed for response {resp_id} after {count} attempts: {e}"
                    ) from e

                status = data.get("status") or "unknown"

                if status in transient_statuses:
                    continue

                slot = respid_to_slot[resp_id]

                if status == "completed":
                    text = _extract_output_text(data) or "[empty]"
                    rt, ot, tt = _extract_usage(data)
                    agg_rt += rt
                    agg_ot += ot
                    agg_tt += tt

                    next_round[slot] = text
                    pending.remove(resp_id)

                    completed_merges += 1
                    merges_this_level += 1
                    print(
                        f"[merge] completed merge {completed_merges}/{total_merges} "
                        f"(level {level}, slot {slot})"
                    )
                    continue

                if status in terminal_statuses:
                    msg = _extract_error_message(data)
                    print(f"[merge] merge response {resp_id} ended with status={status}: {msg}")
                    raise OpenAIResponseError(
                        f"[merge] merge response {resp_id} ended with status={status}: {msg}"
                    )

                print(f"[merge] merge response {resp_id} has unknown terminal status={status!r}.")
                raise OpenAIResponseError(
                    f"[merge] merge response {resp_id} has unknown terminal status={status!r}."
                )

            if pending:
                time.sleep(poll_interval_sec)

        current = [(t or "") for t in next_round]

        print(
            f"[merge] Level {level} done: {len(current)} partial reports remain, "
            f"{total_merges - completed_merges} merges left."
        )
        level += 1

    print("[merge] Hierarchical merge complete.")
    return CompletionResult(text=current[0], reasoning_tokens=agg_rt, output_tokens=agg_ot, total_tokens=agg_tt)


def merge_all_runs(
    *,
    api_key: str,
    model: str,
    reasoning_effort: str = "medium",
    verbosity: str = "medium",
    runs_dir: str = RUNS_DIR,
    write_merged_to: Optional[str] = os.path.join(RUNS_DIR, "merged.txt"),
    merge_system_prompt: str = MERGE_SYSTEM_PROMPT,
    poll_interval_sec: float = 4.0,
) -> CompletionResult:
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
        merge_system_prompt=merge_system_prompt,
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

