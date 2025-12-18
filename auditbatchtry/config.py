from __future__ import annotations

# Default configuration values used by both the CLI and the GUI (per-project settings
# can override these in the browser).

# Total completions = 2**LOG_NUMBER_OF_BATCHES
LOG_NUMBER_OF_BATCHES = 3

# Generation config (for individual audits)
MODEL = "gpt-5.2-pro"  # e.g. "gpt-5.2-pro"
REASONING_EFFORT = "high"  # minimal | low | medium | high
VERBOSITY = "high"  # low | medium | high

# Merge config (can be cheaper / less effort)
MERGE_MODEL = "gpt-5.2"  # or just MODEL if you prefer
MERGE_REASONING_EFFORT = "medium"  # merges are usually easier
MERGE_VERBOSITY = "high"  # often enough for merged output

# Final formatting / triage config (use pro model for security reasoning)
FORMAT_MODEL = "gpt-5.2-pro"
FORMAT_REASONING_EFFORT = "high"
FORMAT_VERBOSITY = "high"
FORMAT_MAX_PARALLEL_ISSUES = 8  # max number of issues to format in parallel
