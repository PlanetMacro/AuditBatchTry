#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
from pathlib import Path

from auditbatchtry import config
from auditbatchtry.api_key_crypto import get_api_key
from auditbatchtry import openai_client as oc


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _data_dir() -> Path:
    return Path(os.environ.get("AUDIT_DATA_DIR", str(_repo_root() / "data")))


def _key_file() -> Path:
    return Path(os.environ.get("OPENAI_API_KEY_FILE", str(_repo_root() / "OPENAI.API_KEY")))


def _maybe_migrate_legacy_runtime(data_dir: Path) -> None:
    repo_root = _repo_root()

    legacy_prompt = repo_root / "PUT_PROMPT_HERE.txt"
    legacy_output = repo_root / "AUDIT_RESULT.txt"
    legacy_runs = repo_root / "RUNS"

    data_prompt = data_dir / "PUT_PROMPT_HERE.txt"
    data_output = data_dir / "AUDIT_RESULT.txt"
    data_runs = data_dir / "RUNS"

    data_dir.mkdir(parents=True, exist_ok=True)

    if legacy_prompt.exists() and not data_prompt.exists():
        shutil.move(str(legacy_prompt), str(data_prompt))
    if legacy_output.exists() and not data_output.exists():
        shutil.move(str(legacy_output), str(data_output))
    if legacy_runs.is_dir() and not data_runs.exists():
        shutil.move(str(legacy_runs), str(data_runs))


def confirm_overwrite(path: str) -> bool:
    if not os.path.exists(path):
        return True
    reply = input(f"{path} already exists. Overwrite? [Y/n]: ").strip().lower()
    return reply in ("", "y", "yes")


def confirm_delete_runs_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    reply = input(f"Folder '{path}' exists. Delete it and start fresh? [y/N]: ").strip().lower()
    return reply in ("y", "yes")


def main() -> None:
    data_dir = _data_dir()
    _maybe_migrate_legacy_runtime(data_dir)

    prompt_file = str(data_dir / "PUT_PROMPT_HERE.txt")
    output_file = str(data_dir / "AUDIT_RESULT.txt")
    runs_dir = str(data_dir / "RUNS")

    if not confirm_overwrite(output_file):
        print("Aborted. Existing file not overwritten.")
        return

    if confirm_delete_runs_dir(runs_dir):
        shutil.rmtree(runs_dir)
        print(f"Deleted '{runs_dir}'.")

    prompt = oc.load_prompt(prompt_file)
    api_key = get_api_key(filename=str(_key_file()))
    print("API key successfully loaded.")

    n = 1 << int(config.LOG_NUMBER_OF_BATCHES)
    print(
        f"MODEL={config.MODEL} | target_completions={n} | "
        f"reasoning={config.REASONING_EFFORT} | verbosity={config.VERBOSITY}"
    )

    print(f"Generating up to {n} audit reports (subtracting any existing runs in '{runs_dir}').")
    gen_results = oc.generate_runs(
        api_key=api_key,
        prompt=prompt,
        n=n,
        model=config.MODEL,
        reasoning_effort=config.REASONING_EFFORT,
        verbosity=config.VERBOSITY,
        runs_dir=runs_dir,
        generation_tools=[{"type": "web_search"}],
    )

    gen_reasoning = sum(r.reasoning_tokens for r in gen_results)
    gen_output = sum(r.output_tokens for r in gen_results)
    gen_total = sum(r.total_tokens for r in gen_results)

    merged_path = os.path.join(runs_dir, "merged.txt")

    if oc.is_merged_up_to_date(runs_dir=runs_dir, merged_path=merged_path):
        print(f"Merged report at '{merged_path}' is up to date; skipping merge step.")
        with open(merged_path, "r", encoding="utf-8") as f:
            merged_text = f.read()
        merged = oc.CompletionResult(text=merged_text, reasoning_tokens=0, output_tokens=0, total_tokens=0)
    else:
        print(
            f"Merging all reports found in '{runs_dir}' using "
            f"MERGE_MODEL={config.MERGE_MODEL}, reasoning={config.MERGE_REASONING_EFFORT}, "
            f"verbosity={config.MERGE_VERBOSITY}"
        )
        merged = oc.merge_all_runs(
            api_key=api_key,
            model=config.MERGE_MODEL,
            reasoning_effort=config.MERGE_REASONING_EFFORT,
            verbosity=config.MERGE_VERBOSITY,
            runs_dir=runs_dir,
            write_merged_to=merged_path,
        )

    print(
        f"Formatting merged issues into final security report using "
        f"FORMAT_MODEL={config.FORMAT_MODEL}, reasoning={config.FORMAT_REASONING_EFFORT}, "
        f"verbosity={config.FORMAT_VERBOSITY}"
    )
    formatted = oc.format_issues_incremental(
        api_key=api_key,
        merged_text=merged.text or "",
        model=config.FORMAT_MODEL,
        final_output_path=output_file,
        reasoning_effort=config.FORMAT_REASONING_EFFORT,
        verbosity=config.FORMAT_VERBOSITY,
        max_parallel_issues=int(config.FORMAT_MAX_PARALLEL_ISSUES),
        runs_dir=runs_dir,
    )

    if (formatted.text or "").strip():
        print(f"\nFinal formatted report written to {output_file}")
    else:
        print(f"\nNo attackable issues; {output_file} is empty or contains only non-attackable findings.")

    fmt_reasoning = formatted.reasoning_tokens
    fmt_output = formatted.output_tokens
    fmt_total = formatted.total_tokens

    print(
        "\n[usage]"
        f" generation: reasoning_tokens={gen_reasoning}  output_tokens={gen_output}  total_tokens={gen_total}"
        f" | merging: reasoning_tokens={merged.reasoning_tokens}  output_tokens={merged.output_tokens}  total_tokens={merged.total_tokens}"
        f" | formatting: reasoning_tokens={fmt_reasoning}  output_tokens={fmt_output}  total_tokens={fmt_total}"
        f" | grand_total={gen_total + merged.total_tokens + fmt_total}"
    )


if __name__ == "__main__":
    main()
