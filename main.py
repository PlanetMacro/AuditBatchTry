#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil

from api_key_crypto import get_api_key
import openai_client as oc

# ================== Configuration ==================
LOG_NUMBER_OF_BATCHES = 3          # total completions = 2**LOG_NUMBER_OF_BATCHES

# Generation config (for individual audits)
MODEL = "gpt-5-pro"                # e.g. "gpt-5-pro"
REASONING_EFFORT = "high"          # minimal | low | medium | high
VERBOSITY = "high"                 # low | medium | high

# Merge config (can be cheaper / less effort)
MERGE_MODEL = "gpt-5"              # or just MODEL if you prefer
MERGE_REASONING_EFFORT = "medium"  # merges are usually easier
MERGE_VERBOSITY = "high"           # often enough for merged output

# Final formatting / triage config (use pro model for security reasoning)
FORMAT_MODEL = "gpt-5-pro"
FORMAT_REASONING_EFFORT = "high"
FORMAT_VERBOSITY = "high"
FORMAT_MAX_PARALLEL_ISSUES = 8     # max number of issues to format in parallel

PROMPT_FILE = "PUT_PROMPT_HERE.txt"
OUTPUT_FILE = "AUDIT_RESULT.txt"
# ====================================================



def confirm_overwrite(path: str) -> bool:
    if not os.path.exists(path):
        return True
    reply = input(f"{path} already exists. Overwrite? [Y/n]: ").strip().lower()
    return reply in ("", "y", "yes")


def confirm_delete_runs_dir(path: str) -> bool:
    """
    Ask whether to delete an existing RUNS directory. Default is No.
    Returns True only if the user explicitly confirms.
    """
    if not os.path.isdir(path):
        return False
    reply = input(f"Folder '{path}' exists. Delete it and start fresh? [y/N]: ").strip().lower()
    return reply in ("y", "yes")


def main() -> None:
    if not confirm_overwrite(OUTPUT_FILE):
        print("Aborted. Existing file not overwritten.")
        return

    # Optional reset of RUNS directory (default No)
    if confirm_delete_runs_dir(oc.RUNS_DIR):
        shutil.rmtree(oc.RUNS_DIR)
        print(f"Deleted '{oc.RUNS_DIR}'.")

    prompt = oc.load_prompt(PROMPT_FILE)
    api_key = get_api_key()
    print("API key successfully loaded.")

    n = 1 << LOG_NUMBER_OF_BATCHES
    print(f"MODEL={MODEL} | target_completions={n} | reasoning={REASONING_EFFORT} | verbosity={VERBOSITY}")

    # Generation: writes each run to RUNS/audit_{i} and auto-subtracts existing files.
    print(f"Generating up to {n} audit reports (subtracting any existing runs in '{oc.RUNS_DIR}').")
    gen_results = oc.generate_runs(
        api_key=api_key,
        prompt=prompt,
        n=n,
        model=MODEL,
        reasoning_effort=REASONING_EFFORT,
        verbosity=VERBOSITY,
        runs_dir=oc.RUNS_DIR,
        generation_tools=[{"type": "web_search"}],  # hosted web_search tool for generation
    )

    # Usage accounting for the generation phase (new runs only)
    gen_reasoning = sum(r.reasoning_tokens for r in gen_results)
    gen_output = sum(r.output_tokens for r in gen_results)
    gen_total = sum(r.total_tokens for r in gen_results)

    # 2) Merge: read ALL RUNS/audit_* files, pad to a power of two, then merge.
    merged_path = os.path.join(oc.RUNS_DIR, "merged.txt")

    if oc.is_merged_up_to_date(runs_dir=oc.RUNS_DIR, merged_path=merged_path):
        # We already have a final, up-to-date merge on disk; reuse it.
        print(f"Merged report at '{merged_path}' is up to date; skipping merge step.")
        with open(merged_path, "r", encoding="utf-8") as f:
            merged_text = f.read()
        # We don't know the original token usage for the previous merge in this run,
        # so we set those counters to 0 for accounting purposes here.
        merged = oc.CompletionResult(
            text=merged_text,
            reasoning_tokens=0,
            output_tokens=0,
            total_tokens=0,
        )
    else:
        # Either no previous merge, or new audit_* files are newer than merged.txt.
        print(
            f"Merging all reports found in '{oc.RUNS_DIR}' "
            f"(padding to next power of two if needed) using "
            f"MERGE_MODEL={MERGE_MODEL}, reasoning={MERGE_REASONING_EFFORT}, verbosity={MERGE_VERBOSITY}"
        )
        merged = oc.merge_all_runs(
            api_key=api_key,
            model=MERGE_MODEL,
            reasoning_effort=MERGE_REASONING_EFFORT,
            verbosity=MERGE_VERBOSITY,
            runs_dir=oc.RUNS_DIR,
            write_merged_to=merged_path,
        )

    # 3) Triage and format: turn merged issues into structured security findings
    print(
        f"Formatting merged issues into final security report using "
        f"FORMAT_MODEL={FORMAT_MODEL}, reasoning={FORMAT_REASONING_EFFORT}, verbosity={FORMAT_VERBOSITY}"
    )
    formatted = oc.format_issues_incremental(
        api_key=api_key,
        merged_text=merged.text or "",
        model=FORMAT_MODEL,
        final_output_path=OUTPUT_FILE,
        reasoning_effort=FORMAT_REASONING_EFFORT,
        verbosity=FORMAT_VERBOSITY,
        max_parallel_issues=FORMAT_MAX_PARALLEL_ISSUES,
    )

    if (formatted.text or "").strip():
        print(f"\nFinal formatted report written to {OUTPUT_FILE}")
    else:
        print(f"\nNo attackable issues; {OUTPUT_FILE} is empty or contains only non-attackable findings.")


    # Usage accounting: generation (this invocation) + merging (all runs) + formatting
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

