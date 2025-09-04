#!/usr/bin/env python3
from __future__ import annotations

import asyncio

from api_key_crypto import get_api_key
import openai_client as oc

# ================== Configuration ==================
LOG_NUMBER_OF_BATCHES = 2          # total completions = 2**LOG_NUMBER_OF_BATCHES
MODEL = "gpt-5"                    # GPT-5 reasoning model
REASONING_EFFORT = "high"          # minimal | low | medium | high
VERBOSITY = "high"                 # low | medium | high (steers visible length)
MAX_CONCURRENCY = 8                # throttle to respect RPM/TPM
PROMPT_FILE = "PUT_PROMPT_HERE.txt"
# ====================================================


def main() -> None:
    prompt = oc.load_prompt(PROMPT_FILE)
    api_key = get_api_key()
    print("API key successfully loaded.")

    client = oc.make_client(api_key)
    n = 1 << LOG_NUMBER_OF_BATCHES
    print(f"MODEL={MODEL} | completions={n} | reasoning={REASONING_EFFORT} | verbosity={VERBOSITY}")

    results = asyncio.run(
        oc.run_parallel(
            client=client,
            prompt=prompt,
            n=n,
            max_concurrency=MAX_CONCURRENCY,
            model=MODEL,
            reasoning_effort=REASONING_EFFORT,
            verbosity=VERBOSITY,
        )
    )

    for i, r in enumerate(results, 1):
        print(f"\n===== COMPLETION {i} of {n} =====")
        print(r.text.strip() or "[empty]")
        print(f"[usage] reasoning_tokens={r.reasoning_tokens}  output_tokens={r.output_tokens}  total_tokens={r.total_tokens}")


if __name__ == "__main__":
    main()

