from __future__ import annotations

import os
import random
import re
import string
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

RUNS_DIR = "RUNS"
AUDIT_BASENAME_RE = re.compile(r"^audit_(\d+)$")  # files with no extension

API_URL = "https://api.openai.com/v1/responses"


def _random_string(min_len: int = 5, max_len: int = 50) -> str:
    n = random.randint(min_len, max_len)
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(n))


@dataclass
class CompletionResult:
    text: str
    reasoning_tokens: int
    output_tokens: int
    total_tokens: int


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


def _ensure_runs_dir(path: str = RUNS_DIR) -> None:
    os.makedirs(path, exist_ok=True)


def _list_audit_files(path: str = RUNS_DIR) -> List[Tuple[int, str]]:
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
    if merged_path is None:
        merged_path = os.path.join(runs_dir, "merged.txt")

    if not os.path.exists(merged_path):
        return False
    if not os.path.isdir(runs_dir):
        return False

    merged_mtime = os.path.getmtime(merged_path)

    for _, fp in _list_audit_files(runs_dir):
        if os.path.getmtime(fp) > merged_mtime:
            return False

    return True


def _next_run_index_and_remaining(n_desired: int, path: str = RUNS_DIR) -> Tuple[int, int, int]:
    files = _list_audit_files(path)
    already_existing = len(files)
    next_index = files[-1][0] + 1 if files else 1
    to_run_now = max(0, n_desired - already_existing)
    return next_index, already_existing, to_run_now


def _write_audit_file(run_index: int, text: str, path: str = RUNS_DIR) -> str:
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
    files = _list_audit_files(path)
    texts: List[str] = []
    for _, fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _pad_to_power_of_two(texts: List[str]) -> List[str]:
    n = len(texts)
    if n == 0:
        return texts
    if _is_power_of_two(n):
        return texts
    k = 1
    while k < n:
        k <<= 1
    last = texts[-1]
    return texts + [last] * (k - n)


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
    usage = response_json.get("usage") or {}
    details = usage.get("output_tokens_details") or {}
    rt = int(details.get("reasoning_tokens") or 0)
    ot = int(usage.get("output_tokens") or 0)
    tt = int(usage.get("total_tokens") or 0)
    return rt, ot, tt


def _extract_error_message(response_json: Dict[str, Any]) -> str:
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
    body = dict(payload)
    body.setdefault("background", True)
    body.setdefault("store", True)

    data = _safe_request("POST", API_URL, headers=headers, json=body)
    resp_id = data.get("id")
    if not isinstance(resp_id, str) or not resp_id:
        raise OpenAIResponseError("No response ID returned from Responses API create call.")
    return resp_id


def _poll_response(headers: Dict[str, str], resp_id: str, poll_interval_sec: float = 4.0) -> Dict[str, Any]:
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
    resp_id = _start_response_job(headers, payload)
    return _poll_response(headers, resp_id, poll_interval_sec=poll_interval_sec)

