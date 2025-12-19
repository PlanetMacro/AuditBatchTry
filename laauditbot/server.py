#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import threading
import uuid
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from laauditbot import config as audit_config
from laauditbot import openai_client as oc
from laauditbot.api_key_crypto import blob_to_json, decrypt_api_key, encrypt_api_key, json_to_blob
from laauditbot.system_prompts import load_default_system_prompts as _load_default_system_prompts_from_files
from cryptography.exceptions import InvalidTag
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
DATA_DIR = Path(os.environ.get("AUDIT_DATA_DIR", str(REPO_ROOT / "data")))

LEGACY_PROJECTS_DIR = REPO_ROOT / "projects"
PROJECTS_DIR = Path(os.environ.get("AUDIT_PROJECTS_DIR", str(DATA_DIR / "projects")))

KEY_FILE = Path(os.environ.get("OPENAI_API_KEY_FILE", str(REPO_ROOT / "OPENAI.API_KEY")))

WEB_DIR = PKG_DIR / "web"
STATIC_DIR = WEB_DIR / "static"

META_FILENAME = "meta.json"
PROMPT_FILENAME = "prompt.txt"
RESULT_FILENAME = "result.txt"
LOG_FILENAME = "logs.txt"
CONFIG_FILENAME = "config.json"
SYSTEM_PROMPTS_FILENAME = "system_prompts.json"
AUDIT_PROMPTS_FILENAME = "audit_prompts.json"
RUNS_SUBDIR = "runs"

PROJECT_ID_RE = re.compile(r"^[a-f0-9]{32}$")

STATE_LOCK = threading.Lock()
PROJECT_LOCKS: dict[str, threading.Lock] = {}
RUN_THREADS: dict[str, threading.Thread] = {}
API_TOKENS: set[str] = set()
API_KEY_CACHE: Optional[str] = None

AUDIT_PROMPTS_FILE = DATA_DIR / AUDIT_PROMPTS_FILENAME
AUDIT_PROMPTS_LOCK = threading.Lock()

DEFAULT_AUDIT_PROMPT_ID = "circuit_audit"
DEFAULT_AUDIT_PROMPT_NAME = "circuit audit"
DEFAULT_AUDIT_PROMPT_TEXT = (
    "Audit the following codebase for missing constraints and general security issues"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(_read_text(path))
    except json.JSONDecodeError:
        return {}


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _validate_project_id(project_id: str) -> None:
    if not PROJECT_ID_RE.match(project_id):
        raise HTTPException(status_code=404, detail="Project not found")


def _project_dir(project_id: str) -> Path:
    _validate_project_id(project_id)
    return PROJECTS_DIR / project_id


def _project_paths(project_id: str) -> dict[str, Path]:
    d = _project_dir(project_id)
    return {
        "dir": d,
        "meta": d / META_FILENAME,
        "prompt": d / PROMPT_FILENAME,
        "result": d / RESULT_FILENAME,
        "logs": d / LOG_FILENAME,
        "config": d / CONFIG_FILENAME,
        "system_prompts": d / SYSTEM_PROMPTS_FILENAME,
        "runs": d / RUNS_SUBDIR,
    }


def _get_project_lock(project_id: str) -> threading.Lock:
    with STATE_LOCK:
        lock = PROJECT_LOCKS.get(project_id)
        if lock is None:
            lock = threading.Lock()
            PROJECT_LOCKS[project_id] = lock
        return lock


def _append_log_line(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
        f.flush()


def _maybe_migrate_legacy_projects_dir() -> None:
    if os.environ.get("AUDIT_PROJECTS_DIR"):
        return
    if not LEGACY_PROJECTS_DIR.is_dir():
        return

    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

    moved_any = False
    for child in LEGACY_PROJECTS_DIR.iterdir():
        if not child.is_dir():
            continue
        if not PROJECT_ID_RE.match(child.name):
            continue
        dest = PROJECTS_DIR / child.name
        if dest.exists():
            continue
        shutil.move(str(child), str(dest))
        moved_any = True

    if moved_any:
        try:
            if not any(LEGACY_PROJECTS_DIR.iterdir()):
                LEGACY_PROJECTS_DIR.rmdir()
        except Exception:
            pass


def _try_bootstrap_api_key_from_env() -> None:
    global API_KEY_CACHE

    if API_KEY_CACHE:
        return

    env_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if env_key:
        API_KEY_CACHE = env_key
        return

    pw = (os.environ.get("OPENAI_API_KEY_PASSWORD") or "").strip()
    if pw and KEY_FILE.exists():
        try:
            blob = json_to_blob(_read_text(KEY_FILE))
            API_KEY_CACHE = decrypt_api_key(blob, pw).strip()
        except Exception:
            # Keep locked; UI can unlock interactively.
            API_KEY_CACHE = None


def _get_api_key_or_raise() -> str:
    global API_KEY_CACHE

    if API_KEY_CACHE:
        return API_KEY_CACHE

    pw = (os.environ.get("OPENAI_API_KEY_PASSWORD") or "").strip()
    if pw and KEY_FILE.exists():
        blob = json_to_blob(_read_text(KEY_FILE))
        API_KEY_CACHE = decrypt_api_key(blob, pw).strip()
        return API_KEY_CACHE

    raise RuntimeError(
        "API key is locked. Unlock in the browser (or set OPENAI_API_KEY)."
    )


def _write_secure_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.chmod(tmp, 0o600)
    except Exception:
        pass
    os.replace(tmp, path)


def _new_token() -> str:
    token = uuid.uuid4().hex
    with STATE_LOCK:
        API_TOKENS.add(token)
    return token


def _is_token_valid(token: str) -> bool:
    if not token:
        return False
    with STATE_LOCK:
        return token in API_TOKENS


def _extract_token(request: Request) -> str:
    return (
        (request.headers.get("x-audit-token") or "").strip()
        or (request.query_params.get("token") or "").strip()
    )


def _load_project_meta(project_id: str) -> dict[str, Any]:
    paths = _project_paths(project_id)
    meta = _load_json(paths["meta"])
    if not meta:
        raise HTTPException(status_code=404, detail="Project not found")
    return meta


def _save_project_meta(project_id: str, meta: dict[str, Any]) -> None:
    paths = _project_paths(project_id)
    meta["updated_at"] = _now_iso()
    _atomic_write_json(paths["meta"], meta)


def _is_project_locked(meta: dict[str, Any]) -> bool:
    return bool(meta.get("last_run_started_at"))


def _list_projects() -> list[dict[str, Any]]:
    if not PROJECTS_DIR.exists():
        return []
    prompt_name_by_id: dict[str, str] = {}
    for p in _load_or_init_audit_prompts():
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or "").strip()
        if not pid:
            continue
        prompt_name_by_id[pid] = str(p.get("name") or "").strip() or pid

    projects: list[dict[str, Any]] = []
    for child in PROJECTS_DIR.iterdir():
        if not child.is_dir():
            continue
        project_id = child.name
        if not PROJECT_ID_RE.match(project_id):
            continue
        meta_path = child / META_FILENAME
        meta = _load_json(meta_path)
        if not meta:
            continue
        cfg_path = child / CONFIG_FILENAME
        cfg = _normalize_project_config(_load_json(cfg_path))
        _atomic_write_json(cfg_path, cfg)

        used_cfg = meta.get("last_run_config") or {}
        models = {
            "generation": used_cfg.get("model") or cfg["model"],
            "merge": used_cfg.get("merge_model") or cfg["merge_model"],
            "format": used_cfg.get("format_model") or cfg["format_model"],
        }
        meta["parallel_runs"] = int(used_cfg.get("parallel_runs") or (1 << int(cfg["log_number_of_batches"])))
        meta["format_parallel_issues"] = int(
            used_cfg.get("format_max_parallel_issues") or cfg["format_max_parallel_issues"]
        )
        meta["models"] = models
        meta["audit_prompt_id"] = _normalize_project_audit_prompt_id(meta.get("audit_prompt_id"))
        used_audit_id = _normalize_project_audit_prompt_id(
            used_cfg.get("audit_prompt_id") or meta.get("audit_prompt_id")
        )
        used_audit_name = str(used_cfg.get("audit_prompt_name") or "").strip() or prompt_name_by_id.get(
            used_audit_id, ""
        )
        meta["audit_prompt_used_id"] = used_audit_id
        meta["audit_prompt_used_name"] = used_audit_name
        meta["locked"] = _is_project_locked(meta)
        projects.append(meta)
    projects.sort(key=lambda m: (m.get("updated_at") or "", m.get("created_at") or ""), reverse=True)
    return projects


def _create_project(*, name: Optional[str], prompt: Optional[str]) -> dict[str, Any]:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    project_id = uuid.uuid4().hex
    now = _now_iso()
    display_name = (name or "").strip() or f"Project {project_id[:6]}"

    meta: dict[str, Any] = {
        "id": project_id,
        "name": display_name,
        "status": "idle",
        "phase": None,
        "created_at": now,
        "updated_at": now,
        "last_run_started_at": None,
        "last_run_finished_at": None,
        "last_run_config": None,
        "audit_prompt_id": _fallback_audit_prompt_id(),
        "error": None,
    }

    paths = _project_paths(project_id)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    _atomic_write_json(paths["meta"], meta)
    if not paths["config"].exists():
        _atomic_write_json(paths["config"], _default_project_config())
    if not paths["system_prompts"].exists():
        _atomic_write_json(paths["system_prompts"], _load_default_system_prompts())
    _atomic_write_text(paths["prompt"], (prompt or "").rstrip() + "\n" if prompt is not None else "")
    if not paths["result"].exists():
        _atomic_write_text(paths["result"], "")
    if not paths["logs"].exists():
        _atomic_write_text(paths["logs"], "")
    paths["runs"].mkdir(parents=True, exist_ok=True)
    return meta


def _ensure_seed_project() -> None:
    if _list_projects():
        return

    meta = _create_project(name="Default Project", prompt="")
    paths = _project_paths(meta["id"])
    _append_log_line(
        paths["logs"],
        f"[seed] Created default project {meta['id']} at {_now_iso()}",
    )


class ProjectCreate(BaseModel):
    name: Optional[str] = None
    prompt: Optional[str] = None


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    prompt: Optional[str] = None


class AuthUnlock(BaseModel):
    password: str


class AuthSetup(BaseModel):
    api_key: str
    password: str
    password_confirm: Optional[str] = None


class ProjectConfig(BaseModel):
    log_number_of_batches: int
    model: str
    reasoning_effort: str
    verbosity: str
    merge_model: str
    merge_reasoning_effort: str
    merge_verbosity: str
    format_model: str
    format_reasoning_effort: str
    format_verbosity: str
    format_max_parallel_issues: int


class ProjectConfigUpdate(BaseModel):
    log_number_of_batches: Optional[int] = None
    model: Optional[str] = None
    reasoning_effort: Optional[str] = None
    verbosity: Optional[str] = None
    merge_model: Optional[str] = None
    merge_reasoning_effort: Optional[str] = None
    merge_verbosity: Optional[str] = None
    format_model: Optional[str] = None
    format_reasoning_effort: Optional[str] = None
    format_verbosity: Optional[str] = None
    format_max_parallel_issues: Optional[int] = None


class SystemPromptsUpdate(BaseModel):
    generation: Optional[str] = None
    merge: Optional[str] = None
    format: Optional[str] = None


class AuditPromptCreate(BaseModel):
    name: str
    prompt: Optional[str] = None


class AuditPromptUpdate(BaseModel):
    name: Optional[str] = None
    prompt: Optional[str] = None


class ProjectAuditPromptUpdate(BaseModel):
    audit_prompt_id: Optional[str] = None


@dataclass(frozen=True)
class RunConfig:
    log_number_of_batches: int
    n: int
    model: str
    reasoning_effort: str
    verbosity: str
    merge_model: str
    merge_reasoning_effort: str
    merge_verbosity: str
    format_model: str
    format_reasoning_effort: str
    format_verbosity: str
    format_max_parallel_issues: int


def _default_project_config() -> dict[str, Any]:
    return {
        "log_number_of_batches": int(getattr(audit_config, "LOG_NUMBER_OF_BATCHES", 3)),
        "model": str(getattr(audit_config, "MODEL", "gpt-5-pro")),
        "reasoning_effort": str(getattr(audit_config, "REASONING_EFFORT", "high")),
        "verbosity": str(getattr(audit_config, "VERBOSITY", "high")),
        "merge_model": str(getattr(audit_config, "MERGE_MODEL", getattr(audit_config, "MODEL", "gpt-5-pro"))),
        "merge_reasoning_effort": str(getattr(audit_config, "MERGE_REASONING_EFFORT", "medium")),
        "merge_verbosity": str(getattr(audit_config, "MERGE_VERBOSITY", "high")),
        "format_model": str(getattr(audit_config, "FORMAT_MODEL", "gpt-5-pro")),
        "format_reasoning_effort": str(getattr(audit_config, "FORMAT_REASONING_EFFORT", "high")),
        "format_verbosity": str(getattr(audit_config, "FORMAT_VERBOSITY", "high")),
        "format_max_parallel_issues": int(getattr(audit_config, "FORMAT_MAX_PARALLEL_ISSUES", 8)),
    }


def _normalize_project_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = {**_default_project_config(), **(cfg or {})}

    try:
        out["log_number_of_batches"] = int(out.get("log_number_of_batches", 3))
    except Exception:
        out["log_number_of_batches"] = int(_default_project_config()["log_number_of_batches"])
    out["log_number_of_batches"] = max(0, min(out["log_number_of_batches"], 12))

    for k in (
        "model",
        "reasoning_effort",
        "verbosity",
        "merge_model",
        "merge_reasoning_effort",
        "merge_verbosity",
        "format_model",
        "format_reasoning_effort",
        "format_verbosity",
    ):
        out[k] = str(out.get(k, "")).strip() or str(_default_project_config()[k])

    try:
        out["format_max_parallel_issues"] = int(out.get("format_max_parallel_issues", 8))
    except Exception:
        out["format_max_parallel_issues"] = int(_default_project_config()["format_max_parallel_issues"])
    out["format_max_parallel_issues"] = max(1, min(out["format_max_parallel_issues"], 64))

    return out


def _load_or_init_project_config(project_id: str) -> dict[str, Any]:
    paths = _project_paths(project_id)
    cfg = _normalize_project_config(_load_json(paths["config"]))
    _atomic_write_json(paths["config"], cfg)
    return cfg


def _config_to_run_config(cfg: dict[str, Any]) -> RunConfig:
    logn = int(cfg["log_number_of_batches"])
    n = 1 << logn
    return RunConfig(
        log_number_of_batches=logn,
        n=n,
        model=str(cfg["model"]),
        reasoning_effort=str(cfg["reasoning_effort"]),
        verbosity=str(cfg["verbosity"]),
        merge_model=str(cfg["merge_model"]),
        merge_reasoning_effort=str(cfg["merge_reasoning_effort"]),
        merge_verbosity=str(cfg["merge_verbosity"]),
        format_model=str(cfg["format_model"]),
        format_reasoning_effort=str(cfg["format_reasoning_effort"]),
        format_verbosity=str(cfg["format_verbosity"]),
        format_max_parallel_issues=int(cfg["format_max_parallel_issues"]),
    )


def _normalize_system_prompts(data: dict[str, Any], defaults: dict[str, str]) -> dict[str, str]:
    out = {**defaults, **(data or {})}
    for k in ("generation", "merge", "format"):
        out[k] = str(out.get(k, "")).rstrip() or defaults[k]
    return out  # type: ignore[return-value]


def _default_audit_prompt() -> dict[str, str]:
    return {
        "id": DEFAULT_AUDIT_PROMPT_ID,
        "name": DEFAULT_AUDIT_PROMPT_NAME,
        "prompt": DEFAULT_AUDIT_PROMPT_TEXT,
    }


def _normalize_audit_prompts_record(data: dict[str, Any]) -> dict[str, Any]:
    prompts_in = data.get("prompts")
    if not isinstance(prompts_in, list):
        prompts_in = []

    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in prompts_in:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("id") or "").strip()
        if not pid:
            pid = uuid.uuid4().hex
        if pid in seen:
            continue
        seen.add(pid)

        name = str(item.get("name") or "").strip() or f"Prompt {pid[:6]}"
        prompt = str(item.get("prompt") or "").rstrip()
        normalized.append({"id": pid, "name": name, "prompt": prompt})

    if not normalized:
        normalized = [_default_audit_prompt()]

    return {"version": 1, "prompts": normalized}


def _load_or_init_audit_prompts_record() -> dict[str, Any]:
    with AUDIT_PROMPTS_LOCK:
        record = _normalize_audit_prompts_record(_load_json(AUDIT_PROMPTS_FILE))
        _atomic_write_json(AUDIT_PROMPTS_FILE, record)
        return record


def _load_or_init_audit_prompts() -> list[dict[str, str]]:
    record = _load_or_init_audit_prompts_record()
    prompts = record.get("prompts")
    if isinstance(prompts, list):
        return prompts  # type: ignore[return-value]
    return [_default_audit_prompt()]


def _audit_prompt_exists(prompt_id: str) -> bool:
    if not prompt_id:
        return False
    for p in _load_or_init_audit_prompts():
        if p.get("id") == prompt_id:
            return True
    return False


def _fallback_audit_prompt_id() -> str:
    prompts = _load_or_init_audit_prompts()
    for p in prompts:
        if p.get("id") == DEFAULT_AUDIT_PROMPT_ID:
            return DEFAULT_AUDIT_PROMPT_ID
    return str(prompts[0].get("id") or DEFAULT_AUDIT_PROMPT_ID)


def _normalize_project_audit_prompt_id(value: Any) -> str:
    raw = str(value or "").strip()
    if raw and _audit_prompt_exists(raw):
        return raw
    return _fallback_audit_prompt_id()


def _audit_prompt_text_for_id(prompt_id: Any) -> tuple[str, str]:
    resolved_id = _normalize_project_audit_prompt_id(prompt_id)
    for p in _load_or_init_audit_prompts():
        if p.get("id") == resolved_id:
            return (str(p.get("prompt") or "").rstrip(), resolved_id)
    return ("", resolved_id)


def _audit_prompt_info_for_id(prompt_id: Any) -> tuple[str, str, str]:
    resolved_id = _normalize_project_audit_prompt_id(prompt_id)
    for p in _load_or_init_audit_prompts():
        if p.get("id") == resolved_id:
            text = str(p.get("prompt") or "").rstrip()
            name = str(p.get("name") or "").strip() or resolved_id
            return (text, resolved_id, name)
    return ("", resolved_id, resolved_id)


def _load_default_system_prompts() -> dict[str, str]:
    return _load_default_system_prompts_from_files()


def _load_or_init_project_system_prompts(project_id: str) -> dict[str, str]:
    defaults = _load_default_system_prompts()
    paths = _project_paths(project_id)
    cur = _normalize_system_prompts(_load_json(paths["system_prompts"]), defaults)
    _atomic_write_json(paths["system_prompts"], cur)
    return cur


class _StreamToLog:
    def __init__(self, f) -> None:
        self._f = f

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._f.write(s)
        self._f.flush()
        return len(s)

    def flush(self) -> None:
        self._f.flush()


def _run_project_audit(project_id: str) -> None:
    paths = _project_paths(project_id)
    lock = _get_project_lock(project_id)

    run_cfg: Optional[RunConfig] = None
    sys_prompts: dict[str, str] = {}
    audit_prompt_id = ""
    audit_prompt_text = ""
    audit_prompt_name = ""
    with lock:
        meta = _load_project_meta(project_id)
        cfg = _load_or_init_project_config(project_id)
        run_cfg = _config_to_run_config(cfg)
        sys_prompts = _load_or_init_project_system_prompts(project_id)
        audit_prompt_text, audit_prompt_id, audit_prompt_name = _audit_prompt_info_for_id(meta.get("audit_prompt_id"))
        meta["status"] = "running"
        meta["phase"] = "starting"
        meta["error"] = None
        meta["last_run_started_at"] = _now_iso()
        meta["last_run_finished_at"] = None
        meta["last_run_config"] = {
            **cfg,
            "parallel_runs": run_cfg.n,
            "format_max_parallel_issues": run_cfg.format_max_parallel_issues,
            "audit_prompt_id": audit_prompt_id,
            "audit_prompt_name": audit_prompt_name,
        }
        meta["audit_prompt_id"] = audit_prompt_id
        _save_project_meta(project_id, meta)

    log_path = paths["logs"]

    try:
        # The bottom "Terminal" panel is intended to show the *current* run output,
        # so we truncate the per-project log at the start of each run.
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as logf:
            logger = _StreamToLog(logf)
            with redirect_stdout(logger), redirect_stderr(logger):
                print(f"[run] Starting audit for project={project_id} at {_now_iso()}")

                user_prompt = _read_text(paths["prompt"]).strip()
                if not user_prompt:
                    raise RuntimeError("Code is empty. Paste code in the UI and retry.")

                prompt_prefix = (audit_prompt_text or "").strip()
                prompt = f"{prompt_prefix}\n\n{user_prompt}" if prompt_prefix else user_prompt
                print(f"[run] AUDIT_PROMPT_ID={audit_prompt_id or '-'}")

                api_key = _get_api_key_or_raise()
                print("[run] API key unlocked.")

                config = run_cfg or _config_to_run_config(_load_or_init_project_config(project_id))
                print(
                    "[run] "
                    f"MODEL={config.model} target_completions={config.n} reasoning={config.reasoning_effort} verbosity={config.verbosity}"
                )

                runs_dir = paths["runs"]
                merged_path = runs_dir / "merged.txt"

                with lock:
                    meta = _load_project_meta(project_id)
                    meta["phase"] = "generating"
                    _save_project_meta(project_id, meta)
                print(f"[run] Generating up to {config.n} audit reports into '{runs_dir}'.")
                gen_results = oc.generate_runs(
                    api_key=api_key,
                    prompt=prompt,
                    n=config.n,
                    model=config.model,
                    reasoning_effort=config.reasoning_effort,
                    verbosity=config.verbosity,
                    runs_dir=str(runs_dir),
                    generation_tools=[{"type": "web_search"}],
                    system_prompt=(sys_prompts.get("generation") or oc.SYSTEM_PROMPT),
                )

                gen_reasoning = sum(r.reasoning_tokens for r in gen_results)
                gen_output = sum(r.output_tokens for r in gen_results)
                gen_total = sum(r.total_tokens for r in gen_results)
                print(
                    f"[usage] generation reasoning_tokens={gen_reasoning} output_tokens={gen_output} total_tokens={gen_total}"
                )

                if oc.is_merged_up_to_date(runs_dir=str(runs_dir), merged_path=str(merged_path)):
                    print(f"[run] Merged report at '{merged_path}' is up to date; skipping merge step.")
                    merged_text = _read_text(merged_path)
                    merged = oc.CompletionResult(text=merged_text, reasoning_tokens=0, output_tokens=0, total_tokens=0)
                else:
                    with lock:
                        meta = _load_project_meta(project_id)
                        meta["phase"] = "merging"
                        _save_project_meta(project_id, meta)
                    print(
                        "[run] "
                        f"Merging reports in '{runs_dir}' using MERGE_MODEL={config.merge_model} "
                        f"reasoning={config.merge_reasoning_effort} verbosity={config.merge_verbosity}"
                    )
                    merged = oc.merge_all_runs(
                        api_key=api_key,
                        model=config.merge_model,
                        reasoning_effort=config.merge_reasoning_effort,
                        verbosity=config.merge_verbosity,
                        runs_dir=str(runs_dir),
                        write_merged_to=str(merged_path),
                        merge_system_prompt=(sys_prompts.get("merge") or oc.MERGE_SYSTEM_PROMPT),
                    )

                with lock:
                    meta = _load_project_meta(project_id)
                    meta["phase"] = "formatting"
                    _save_project_meta(project_id, meta)
                print(
                    "[run] "
                    f"Formatting merged issues using FORMAT_MODEL={config.format_model} "
                    f"reasoning={config.format_reasoning_effort} verbosity={config.format_verbosity}"
                )
                formatted = oc.format_issues_incremental(
                    api_key=api_key,
                    merged_text=merged.text or "",
                    original_code=user_prompt,
                    model=config.format_model,
                    final_output_path=str(paths["result"]),
                    reasoning_effort=config.format_reasoning_effort,
                    verbosity=config.format_verbosity,
                    runs_dir=str(runs_dir),
                    max_parallel_issues=config.format_max_parallel_issues,
                    issue_format_system_prompt=(sys_prompts.get("format") or oc.ISSUE_FORMAT_SYSTEM_PROMPT),
                )

                fmt_reasoning = formatted.reasoning_tokens
                fmt_output = formatted.output_tokens
                fmt_total = formatted.total_tokens
                print(
                    f"[usage] formatting reasoning_tokens={fmt_reasoning} output_tokens={fmt_output} total_tokens={fmt_total}"
                )
                print(f"[run] Finished at {_now_iso()}. Output written to '{paths['result']}'.")

        with lock:
            meta = _load_project_meta(project_id)
            meta["status"] = "success"
            meta["phase"] = None
            meta["error"] = None
            meta["last_run_finished_at"] = _now_iso()
            _save_project_meta(project_id, meta)
    except Exception as e:  # noqa: BLE001 - show error in UI
        _append_log_line(log_path, f"[run] ERROR: {e}")
        with lock:
            meta = _load_project_meta(project_id)
            meta["status"] = "error"
            meta["phase"] = None
            meta["error"] = str(e)
            meta["last_run_finished_at"] = _now_iso()
            _save_project_meta(project_id, meta)
    finally:
        with STATE_LOCK:
            RUN_THREADS.pop(project_id, None)


def _start_run_thread(project_id: str) -> dict[str, Any]:
    with STATE_LOCK:
        existing = RUN_THREADS.get(project_id)
        if existing and existing.is_alive():
            raise HTTPException(status_code=409, detail="Project is already running")
        t = threading.Thread(target=_run_project_audit, args=(project_id,), daemon=True)
        RUN_THREADS[project_id] = t
        t.start()
    return _load_project_meta(project_id)


def _tail_lines(path: Path, max_lines: int) -> list[str]:
    text = _read_text(path)
    if not text:
        return []
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return lines
    return lines[-max_lines:]


def _sse_data_line(line: str) -> str:
    safe = line.replace("\r", "")
    return f"data: {safe}\n\n"


app = FastAPI(title="LAAuditBot GUI", version="0.1.0")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def _startup() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _maybe_migrate_legacy_projects_dir()
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    _load_or_init_audit_prompts_record()
    _ensure_seed_project()
    _try_bootstrap_api_key_from_env()


@app.middleware("http")
async def _auth_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    path = request.url.path
    if path.startswith("/api") and not path.startswith("/api/auth"):
        token = _extract_token(request)
        if not _is_token_valid(token):
            return JSONResponse({"detail": "Locked. Unlock with password."}, status_code=401)
    return await call_next(request)


@app.get("/")
def index() -> FileResponse:
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Missing web/index.html")
    return FileResponse(index_path)


@app.get("/api/auth/status")
def api_auth_status() -> dict[str, Any]:
    return {
        "unlocked": bool(API_KEY_CACHE) or bool((os.environ.get("OPENAI_API_KEY") or "").strip()),
        "has_key_file": KEY_FILE.exists(),
    }


@app.post("/api/auth/unlock")
def api_auth_unlock(body: AuthUnlock) -> dict[str, Any]:
    global API_KEY_CACHE

    password = (body.password or "").strip()
    if not password:
        raise HTTPException(status_code=400, detail="Password required")

    if not KEY_FILE.exists():
        raise HTTPException(status_code=404, detail="Missing OPENAI.API_KEY file")

    try:
        blob = json_to_blob(_read_text(KEY_FILE))
        API_KEY_CACHE = decrypt_api_key(blob, password).strip()
    except InvalidTag:
        raise HTTPException(status_code=401, detail="Incorrect password")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid key file format")

    token = _new_token()
    return {"unlocked": True, "token": token}


@app.post("/api/auth/setup")
def api_auth_setup(body: AuthSetup) -> dict[str, Any]:
    global API_KEY_CACHE

    api_key = (body.api_key or "").strip()
    password = (body.password or "").strip()
    confirm = (body.password_confirm or "").strip()

    if not api_key:
        raise HTTPException(status_code=400, detail="API key required")
    if not password:
        raise HTTPException(status_code=400, detail="Password required")
    if body.password_confirm is not None and password != confirm:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    if KEY_FILE.exists():
        raise HTTPException(status_code=409, detail="Key file already exists")

    blob = encrypt_api_key(api_key, password)
    _write_secure_file(KEY_FILE, blob_to_json(blob))
    API_KEY_CACHE = api_key

    token = _new_token()
    return {"unlocked": True, "token": token, "has_key_file": True}


@app.post("/api/auth/token")
def api_auth_token() -> dict[str, Any]:
    _try_bootstrap_api_key_from_env()
    if not API_KEY_CACHE:
        raise HTTPException(status_code=401, detail="Locked")
    token = _new_token()
    return {"unlocked": True, "token": token}


@app.get("/api/projects")
def api_list_projects() -> list[dict[str, Any]]:
    return _list_projects()


@app.post("/api/projects")
def api_create_project(body: ProjectCreate) -> dict[str, Any]:
    meta = _create_project(name=body.name, prompt=body.prompt)
    return meta


@app.get("/api/projects/{project_id}")
def api_get_project(project_id: str) -> dict[str, Any]:
    meta = _load_project_meta(project_id)
    paths = _project_paths(project_id)
    cfg = _load_or_init_project_config(project_id)
    used_cfg = meta.get("last_run_config") or {}
    models = {
        "generation": used_cfg.get("model") or cfg["model"],
        "merge": used_cfg.get("merge_model") or cfg["merge_model"],
        "format": used_cfg.get("format_model") or cfg["format_model"],
    }
    used_audit_name = str(used_cfg.get("audit_prompt_name") or "").strip()
    if used_audit_name:
        used_audit_id = _normalize_project_audit_prompt_id(
            used_cfg.get("audit_prompt_id") or meta.get("audit_prompt_id")
        )
    else:
        _, used_audit_id, used_audit_name = _audit_prompt_info_for_id(
            used_cfg.get("audit_prompt_id") or meta.get("audit_prompt_id")
        )
    return {
        **meta,
        "audit_prompt_id": _normalize_project_audit_prompt_id(meta.get("audit_prompt_id")),
        "audit_prompt_used_id": used_audit_id,
        "audit_prompt_used_name": used_audit_name,
        "locked": _is_project_locked(meta),
        "parallel_runs": int(used_cfg.get("parallel_runs") or (1 << int(cfg["log_number_of_batches"]))),
        "format_parallel_issues": int(
            used_cfg.get("format_max_parallel_issues") or cfg["format_max_parallel_issues"]
        ),
        "models": models,
        "prompt": _read_text(paths["prompt"]),
        "result": _read_text(paths["result"]),
    }


@app.put("/api/projects/{project_id}")
def api_update_project(project_id: str, body: ProjectUpdate) -> dict[str, Any]:
    paths = _project_paths(project_id)
    lock = _get_project_lock(project_id)

    with lock:
        meta = _load_project_meta(project_id)
        if body.name is not None:
            meta["name"] = body.name.strip() or meta.get("name") or f"Project {project_id[:6]}"
        _save_project_meta(project_id, meta)
        if body.prompt is not None:
            existing = _read_text(paths["prompt"]).rstrip("\n")
            incoming = (body.prompt or "").rstrip("\n")
            if _is_project_locked(meta) and incoming != existing:
                raise HTTPException(
                    status_code=409,
                    detail="This project is locked because an audit has already been run. Clone to edit the prompt.",
                )
            if not _is_project_locked(meta):
                _atomic_write_text(paths["prompt"], incoming + "\n" if incoming else "")
    return api_get_project(project_id)


@app.get("/api/projects/{project_id}/config")
def api_get_project_config(project_id: str) -> dict[str, Any]:
    _load_project_meta(project_id)  # 404 if missing
    cfg = _load_or_init_project_config(project_id)
    return {**cfg, "parallel_runs": 1 << int(cfg["log_number_of_batches"])}


@app.put("/api/projects/{project_id}/config")
def api_update_project_config(project_id: str, body: ProjectConfigUpdate) -> dict[str, Any]:
    paths = _project_paths(project_id)
    lock = _get_project_lock(project_id)

    with lock:
        meta = _load_project_meta(project_id)
        if meta.get("status") == "running":
            raise HTTPException(status_code=409, detail="Project is running")
        existing = _load_or_init_project_config(project_id)
        merged = {**existing, **{k: v for k, v in body.model_dump().items() if v is not None}}
        cfg = _normalize_project_config(merged)
        _atomic_write_json(paths["config"], cfg)
        _save_project_meta(project_id, meta)

    return {**cfg, "parallel_runs": 1 << int(cfg["log_number_of_batches"])}


@app.post("/api/projects/{project_id}/config/restore-defaults")
def api_restore_project_config_defaults(project_id: str) -> dict[str, Any]:
    paths = _project_paths(project_id)
    lock = _get_project_lock(project_id)

    with lock:
        meta = _load_project_meta(project_id)
        if meta.get("status") == "running":
            raise HTTPException(status_code=409, detail="Project is running")
        cfg = _normalize_project_config(_default_project_config())
        _atomic_write_json(paths["config"], cfg)
        _save_project_meta(project_id, meta)

    return {**cfg, "parallel_runs": 1 << int(cfg["log_number_of_batches"])}


@app.get("/api/projects/{project_id}/system-prompts")
def api_get_project_system_prompts(project_id: str) -> dict[str, Any]:
    _load_project_meta(project_id)  # 404 if missing
    defaults = _load_default_system_prompts()
    current = _load_or_init_project_system_prompts(project_id)
    return {"current": current, "defaults": defaults}


@app.put("/api/projects/{project_id}/system-prompts")
def api_update_project_system_prompts(project_id: str, body: SystemPromptsUpdate) -> dict[str, Any]:
    paths = _project_paths(project_id)
    lock = _get_project_lock(project_id)

    with lock:
        meta = _load_project_meta(project_id)
        if meta.get("status") == "running":
            raise HTTPException(status_code=409, detail="Project is running")
        defaults = _load_default_system_prompts()
        existing = _load_or_init_project_system_prompts(project_id)
        merged = {**existing, **{k: v for k, v in body.model_dump().items() if v is not None}}
        cur = _normalize_system_prompts(merged, defaults)
        _atomic_write_json(paths["system_prompts"], cur)
        _save_project_meta(project_id, meta)

    return {"current": cur, "defaults": defaults}


@app.post("/api/projects/{project_id}/system-prompts/restore-defaults")
def api_restore_project_system_prompts(project_id: str) -> dict[str, Any]:
    paths = _project_paths(project_id)
    lock = _get_project_lock(project_id)

    with lock:
        meta = _load_project_meta(project_id)
        if meta.get("status") == "running":
            raise HTTPException(status_code=409, detail="Project is running")
        defaults = _load_default_system_prompts()
        _atomic_write_json(paths["system_prompts"], defaults)
        _save_project_meta(project_id, meta)

    return {"current": defaults, "defaults": defaults}


@app.get("/api/audit-prompts")
def api_list_audit_prompts() -> dict[str, Any]:
    record = _load_or_init_audit_prompts_record()
    prompts = record.get("prompts") if isinstance(record, dict) else None
    return {"prompts": prompts or [], "default_id": DEFAULT_AUDIT_PROMPT_ID}


@app.post("/api/audit-prompts")
def api_create_audit_prompt(body: AuditPromptCreate) -> dict[str, Any]:
    name = str(body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name required")
    prompt_text = str(body.prompt or "").rstrip()

    with AUDIT_PROMPTS_LOCK:
        record = _normalize_audit_prompts_record(_load_json(AUDIT_PROMPTS_FILE))
        prompts = record.get("prompts")
        if not isinstance(prompts, list):
            prompts = []

        prompt_id = uuid.uuid4().hex
        created = {"id": prompt_id, "name": name, "prompt": prompt_text}
        prompts.append(created)
        record["prompts"] = prompts
        _atomic_write_json(AUDIT_PROMPTS_FILE, record)
        return created


@app.put("/api/audit-prompts/{prompt_id}")
def api_update_audit_prompt(prompt_id: str, body: AuditPromptUpdate) -> dict[str, Any]:
    prompt_id = str(prompt_id or "").strip()
    if not prompt_id:
        raise HTTPException(status_code=404, detail="Audit prompt not found")

    with AUDIT_PROMPTS_LOCK:
        record = _normalize_audit_prompts_record(_load_json(AUDIT_PROMPTS_FILE))
        prompts = record.get("prompts")
        if not isinstance(prompts, list):
            prompts = []

        updated: Optional[dict[str, Any]] = None
        for p in prompts:
            if not isinstance(p, dict):
                continue
            if str(p.get("id") or "") != prompt_id:
                continue
            if body.name is not None:
                name = str(body.name or "").strip()
                if not name:
                    raise HTTPException(status_code=400, detail="Name required")
                p["name"] = name
            if body.prompt is not None:
                p["prompt"] = str(body.prompt or "").rstrip()
            updated = p
            break

        if updated is None:
            raise HTTPException(status_code=404, detail="Audit prompt not found")

        record["prompts"] = prompts
        _atomic_write_json(AUDIT_PROMPTS_FILE, record)
        return updated


@app.delete("/api/audit-prompts/{prompt_id}")
def api_delete_audit_prompt(prompt_id: str) -> dict[str, Any]:
    prompt_id = str(prompt_id or "").strip()
    if not prompt_id:
        raise HTTPException(status_code=404, detail="Audit prompt not found")

    with AUDIT_PROMPTS_LOCK:
        record = _normalize_audit_prompts_record(_load_json(AUDIT_PROMPTS_FILE))
        prompts = record.get("prompts")
        if not isinstance(prompts, list):
            prompts = []

        if len(prompts) <= 1:
            raise HTTPException(status_code=409, detail="Cannot delete the last audit prompt")

        kept: list[dict[str, Any]] = []
        deleted = False
        for p in prompts:
            if not isinstance(p, dict):
                continue
            if str(p.get("id") or "") == prompt_id:
                deleted = True
                continue
            kept.append(p)

        if not deleted:
            raise HTTPException(status_code=404, detail="Audit prompt not found")

        record["prompts"] = kept
        _atomic_write_json(AUDIT_PROMPTS_FILE, record)

    return {"deleted": True}


@app.put("/api/projects/{project_id}/audit-prompt")
def api_update_project_audit_prompt(project_id: str, body: ProjectAuditPromptUpdate) -> dict[str, Any]:
    lock = _get_project_lock(project_id)

    with lock:
        meta = _load_project_meta(project_id)
        if meta.get("status") == "running":
            raise HTTPException(status_code=409, detail="Project is running")

        requested = str(body.audit_prompt_id or "").strip()
        if requested and not _audit_prompt_exists(requested):
            raise HTTPException(status_code=404, detail="Audit prompt not found")
        meta["audit_prompt_id"] = requested or _fallback_audit_prompt_id()
        _save_project_meta(project_id, meta)

    return api_get_project(project_id)


@app.delete("/api/projects/{project_id}")
def api_delete_project(project_id: str) -> dict[str, Any]:
    paths = _project_paths(project_id)
    lock = _get_project_lock(project_id)

    with lock:
        meta = _load_project_meta(project_id)
        if meta.get("status") == "running":
            raise HTTPException(status_code=409, detail="Project is running")
        if paths["dir"].exists():
            shutil.rmtree(paths["dir"])

    with STATE_LOCK:
        PROJECT_LOCKS.pop(project_id, None)
        RUN_THREADS.pop(project_id, None)

    return {"deleted": True}


@app.post("/api/projects/{project_id}/run")
def api_run_project(project_id: str) -> dict[str, Any]:
    meta = _load_project_meta(project_id)  # 404 if missing
    if _is_project_locked(meta):
        raise HTTPException(
            status_code=409,
            detail="This project already has an audit run. Clone it to run a new audit.",
        )
    try:
        _get_api_key_or_raise()
    except RuntimeError as e:
        raise HTTPException(status_code=401, detail=str(e))
    return _start_run_thread(project_id)


@app.get("/api/projects/{project_id}/prompt", response_class=PlainTextResponse)
def api_get_prompt_text(project_id: str) -> str:
    paths = _project_paths(project_id)
    return _read_text(paths["prompt"])


@app.get("/api/projects/{project_id}/result", response_class=PlainTextResponse)
def api_get_result_text(project_id: str) -> str:
    paths = _project_paths(project_id)
    return _read_text(paths["result"])


@app.get("/api/projects/{project_id}/logs", response_class=PlainTextResponse)
def api_get_logs_text(project_id: str, tail: int = 400) -> str:
    paths = _project_paths(project_id)
    lines = _tail_lines(paths["logs"], max_lines=max(1, min(tail, 5000)))
    return "\n".join(lines) + ("\n" if lines else "")


@app.get("/api/projects/{project_id}/logs/stream")
async def api_stream_logs(project_id: str, tail: int = 200) -> StreamingResponse:
    paths = _project_paths(project_id)
    log_path = paths["logs"]

    async def gen() -> Any:
        yield "retry: 1000\n\n"
        for line in _tail_lines(log_path, max_lines=max(1, min(tail, 2000))):
            yield _sse_data_line(line)

        pos = log_path.stat().st_size if log_path.exists() else 0
        buf = ""

        while True:
            await asyncio.sleep(0.5)
            if not log_path.exists():
                continue
            size = log_path.stat().st_size
            if size < pos:
                pos = 0
                buf = ""

            if size == pos:
                continue

            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()

            if not chunk:
                continue

            buf += chunk.replace("\r", "")
            lines = buf.split("\n")
            buf = lines.pop()  # remainder (possibly partial line)
            for line in lines:
                yield _sse_data_line(line)

    return StreamingResponse(gen(), media_type="text/event-stream")
