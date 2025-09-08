#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import shutil

from api_key_crypto import get_api_key
import openai_client as oc

# ================== Configuration ==================
LOG_NUMBER_OF_BATCHES = 3          # total completions = 2**LOG_NUMBER_OF_BATCHES
MODEL = "gpt-5"                    # GPT-5 reasoning model
REASONING_EFFORT = "high"          # minimal | low | medium | high
VERBOSITY = "high"                 # low | medium | high (steers visible length)
MAX_CONCURRENCY = 8                # throttle to respect RPM/TPM
PER_REQUEST_TIMEOUT = 86400        # limit per request (seconds)
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

    async def _async_main():
        client = oc.make_async_client(
            api_key,
            timeout_s=PER_REQUEST_TIMEOUT,   # SDK-level total timeout per request
            max_retries=4,                   # SDK-level retries
            use_aiohttp=True,                # requires: pip install "openai[aiohttp]"
        )
        try:
            # Generation: oc.run_parallel writes each run to RUNS/audit_{i} and
            # automatically subtracts existing RUNS/audit_* files from 'n'.
            print(f"Generating up to {n} audit reports (subtracting any existing runs in '{oc.RUNS_DIR}').")
            results = await oc.run_parallel(
                client=client,
                prompt=prompt,
                n=n,
                max_concurrency=MAX_CONCURRENCY,
                model=MODEL,
                reasoning_effort=REASONING_EFFORT,
                verbosity=VERBOSITY,
                per_request_timeout=PER_REQUEST_TIMEOUT,
                app_retries=2,
            )

            # Usage accounting for the generation phase (new runs only)
            gen_reasoning = sum(r.reasoning_tokens for r in results)
            gen_output = sum(r.output_tokens for r in results)
            gen_total = sum(r.total_tokens for r in results)

            # Merge: read ALL RUNS/audit_* files, pad to a power of two, then merge.
            print(f"Merging all reports found in '{oc.RUNS_DIR}' (padding to next power of two if needed).")
            merged = await oc.merge_all_runs(
                client=client,
                max_concurrency=MAX_CONCURRENCY,
                model=MODEL,
                reasoning_effort=REASONING_EFFORT,
                verbosity=VERBOSITY,
                per_request_timeout=PER_REQUEST_TIMEOUT,
                runs_dir=oc.RUNS_DIR,
                write_merged_to=os.path.join(oc.RUNS_DIR, "merged.txt"),
            )

            # Final output
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                f.write((merged.text or "").strip() or "[empty]\n")

            print(f"\nMerged result written to {OUTPUT_FILE}")

            # Usage accounting: generation (this invocation) + merging (all runs)
            print(
                "\n[usage]"
                f" generation: reasoning_tokens={gen_reasoning}  output_tokens={gen_output}  total_tokens={gen_total}"
                f" | merging: reasoning_tokens={merged.reasoning_tokens}  output_tokens={merged.output_tokens}  total_tokens={merged.total_tokens}"
                f" | grand_total={gen_total + merged.total_tokens}"
            )
        finally:
            # Properly close the async client to avoid 'coroutine was never awaited' warnings
            await client.close()

    asyncio.run(_async_main())


if __name__ == "__main__":
    main()

