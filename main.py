#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os

from api_key_crypto import get_api_key
import openai_client as oc

# ================== Configuration ==================
LOG_NUMBER_OF_BATCHES = 3          # total completions = 2**LOG_NUMBER_OF_BATCHES
MODEL = "gpt-5"                    # GPT-5 reasoning model
REASONING_EFFORT = "high"          # minimal | low | medium | high
VERBOSITY = "high"                 # low | medium | high (steers visible length)
MAX_CONCURRENCY = 8                # throttle to respect RPM/TPM
PROMPT_FILE = "PUT_PROMPT_HERE.txt"
OUTPUT_FILE = "AUDIT_RESULT.txt"
# ====================================================


def confirm_overwrite(path: str) -> bool:
    if not os.path.exists(path):
        return True
    reply = input(f"{path} already exists. Overwrite? [Y/n]: ").strip().lower()
    return reply in ("", "y", "yes")


def main() -> None:
    if not confirm_overwrite(OUTPUT_FILE):
        print("Aborted. Existing file not overwritten.")
        return

    prompt = oc.load_prompt(PROMPT_FILE)
    api_key = get_api_key()
    print("API key successfully loaded.")

    n = 1 << LOG_NUMBER_OF_BATCHES
    print(f"MODEL={MODEL} | completions={n} | reasoning={REASONING_EFFORT} | verbosity={VERBOSITY}")

    async def _async_main():
        # Create one async client and reuse it for all calls
        client = oc.make_async_client(
            api_key,
            timeout_s=3600.0,     # SDK-level total timeout per request
            max_retries=4,       # SDK-level retries
            use_aiohttp=True,   # set True if you installed: pip install "openai[aiohttp]"
        )

        print(f"Generate {n} initial audit reports. This might take a while...")
        results = await oc.run_parallel(
            client=client,
            prompt=prompt,
            n=n,
            max_concurrency=MAX_CONCURRENCY,
            model=MODEL,
            reasoning_effort=REASONING_EFFORT,
            verbosity=VERBOSITY,
            per_request_timeout=600.0,  # optional per-request override
            app_retries=2,              # app-level retries on 429/timeouts
        )

        print("Merge initial reports into one. This might take a while...")
        # Aggregate usage for the generation phase
        gen_reasoning = sum(r.reasoning_tokens for r in results)
        gen_output = sum(r.output_tokens for r in results)
        gen_total = sum(r.total_tokens for r in results)

        # Hierarchical merge via the async API
        merged = await oc.hierarchical_merge(
            client=client,
            texts=[r.text for r in results],
            max_concurrency=MAX_CONCURRENCY,
            model=MODEL,
            reasoning_effort=REASONING_EFFORT,
            verbosity=VERBOSITY,
            per_request_timeout=600.0,
        )

        # Final output
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(merged.text.strip() or "[empty]\n")

        print(f"\nMerged result written to {OUTPUT_FILE}")

        # Usage accounting: generation + merging
        print(
            "\n[usage]"
            f" generation: reasoning_tokens={gen_reasoning}  output_tokens={gen_output}  total_tokens={gen_total}"
            f" | merging: reasoning_tokens={merged.reasoning_tokens}  output_tokens={merged.output_tokens}  total_tokens={merged.total_tokens}"
            f" | grand_total={gen_total + merged.total_tokens}"
        )

    asyncio.run(_async_main())



if __name__ == "__main__":
    main()

