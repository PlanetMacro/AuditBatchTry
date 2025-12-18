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

from auditbatchtry import config as audit_config
from auditbatchtry import openai_client as oc
from auditbatchtry.api_key_crypto import blob_to_json, decrypt_api_key, encrypt_api_key, json_to_blob
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

DEFAULTS_DIR = PKG_DIR / "defaults"
DEFAULT_SYSTEM_PROMPTS_PATH = DEFAULTS_DIR / "system_prompts.json"
WEB_DIR = PKG_DIR / "web"
STATIC_DIR = WEB_DIR / "static"

META_FILENAME = "meta.json"
PROMPT_FILENAME = "prompt.txt"
RESULT_FILENAME = "result.txt"
LOG_FILENAME = "logs.txt"
CONFIG_FILENAME = "config.json"
SYSTEM_PROMPTS_FILENAME = "system_prompts.json"
RUNS_SUBDIR = "runs"

PROJECT_ID_RE = re.compile(r"^[a-f0-9]{32}$")

STATE_LOCK = threading.Lock()
PROJECT_LOCKS: dict[str, threading.Lock] = {}
RUN_THREADS: dict[str, threading.Thread] = {}
API_TOKENS: set[str] = set()
API_KEY_CACHE: Optional[str] = None


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


def _maybe_migrate_legacy_runtime_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    legacy_prompt = REPO_ROOT / "PUT_PROMPT_HERE.txt"
    legacy_output = REPO_ROOT / "AUDIT_RESULT.txt"
    legacy_runs = REPO_ROOT / "RUNS"

    data_prompt = DATA_DIR / "PUT_PROMPT_HERE.txt"
    data_output = DATA_DIR / "AUDIT_RESULT.txt"
    data_runs = DATA_DIR / "RUNS"

    if legacy_prompt.exists() and not data_prompt.exists():
        shutil.move(str(legacy_prompt), str(data_prompt))
    if legacy_output.exists() and not data_output.exists():
        shutil.move(str(legacy_output), str(data_output))
    if legacy_runs.is_dir() and not data_runs.exists():
        shutil.move(str(legacy_runs), str(data_runs))


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

    prompt_candidates = [
        DATA_DIR / "PUT_PROMPT_HERE.txt",
        REPO_ROOT / "PUT_PROMPT_HERE.txt",
        REPO_ROOT / "PUT_PROMT_HERE.txt",
        REPO_ROOT / "PUT_PROMPT_HERE",
        REPO_ROOT / "PUT_PROMT_HERE",
    ]
    result_candidates = [
        DATA_DIR / "AUDIT_RESULT.txt",
        REPO_ROOT / "AUDIT_RESULT.txt",
        REPO_ROOT / "AUDIT_RESULTS.txt",
        REPO_ROOT / "AUDIT_RESULTS",
    ]

    prompt = ""
    for p in prompt_candidates:
        if p.exists():
            prompt = _read_text(p).rstrip()
            break

    result = ""
    for r in result_candidates:
        if r.exists():
            result = _read_text(r).rstrip()
            break

    meta = _create_project(name="Default Project", prompt=prompt)
    paths = _project_paths(meta["id"])
    _atomic_write_text(paths["result"], result + "\n" if result else "")
    _append_log_line(
        paths["logs"],
        f"[seed] Created default project {meta['id']} from legacy files at {_now_iso()}",
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


def _default_system_prompts_from_code() -> dict[str, str]:
    return {
        "generation": oc.SYSTEM_PROMPT,
        "merge": oc.MERGE_SYSTEM_PROMPT,
        "format": oc.ISSUE_FORMAT_SYSTEM_PROMPT,
    }


def _normalize_system_prompts(data: dict[str, Any], defaults: dict[str, str]) -> dict[str, str]:
    out = {**defaults, **(data or {})}
    for k in ("generation", "merge", "format"):
        out[k] = str(out.get(k, "")).rstrip() or defaults[k]
    return out  # type: ignore[return-value]


def _load_default_system_prompts() -> dict[str, str]:
    defaults = _default_system_prompts_from_code()
    if DEFAULT_SYSTEM_PROMPTS_PATH.exists():
        data = _load_json(DEFAULT_SYSTEM_PROMPTS_PATH)
        return _normalize_system_prompts(data, defaults)
    return defaults


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
    with lock:
        meta = _load_project_meta(project_id)
        cfg = _load_or_init_project_config(project_id)
        run_cfg = _config_to_run_config(cfg)
        sys_prompts = _load_or_init_project_system_prompts(project_id)
        meta["status"] = "running"
        meta["phase"] = "starting"
        meta["error"] = None
        meta["last_run_started_at"] = _now_iso()
        meta["last_run_finished_at"] = None
        meta["last_run_config"] = {
            **cfg,
            "parallel_runs": run_cfg.n,
            "format_max_parallel_issues": run_cfg.format_max_parallel_issues,
        }
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

                prompt = _read_text(paths["prompt"]).strip()
                if not prompt:
                    raise RuntimeError("Prompt is empty. Fill it in the UI and retry.")

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
    _maybe_migrate_legacy_runtime_files()
    _maybe_migrate_legacy_projects_dir()
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
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
    return {
        **meta,
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
