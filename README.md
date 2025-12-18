# LAAuditBot

Batch-generate, merge, and format security audits using OpenAI models. Includes a local browser GUI to manage multiple projects.

## Requirements

- Python 3.9+ (3.10+ recommended)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## API key (encrypted)

This project expects an encrypted key file `OPENAI.API_KEY` (gitignored). Create it once:

```bash
python -c "from laauditbot.api_key_crypto import get_api_key; get_api_key()"
```

If the file is missing, the browser GUI also offers a setup dialog to create it.

The browser will ask for the password to decrypt the key (kept only in server memory).

Optional alternatives:
- Set `OPENAI_API_KEY` (plaintext) to skip the password prompt.
- Set `OPENAI_API_KEY_PASSWORD` to auto-decrypt `OPENAI.API_KEY` on startup (no browser prompt).

## Browser GUI

Start the local server:

```bash
python -m uvicorn laauditbot.server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in your browser.

- Backwards-compatible: `python -m uvicorn server:app --host 127.0.0.1 --port 8000`
- Projects live under `data/projects/` by default.
- Per-project settings and system prompts are editable in the UI and apply to the next run.

## Convenience

```bash
python main.py
```

Starts the local GUI server (defaults to `127.0.0.1:8000`).

## Paths / env vars

- `AUDIT_DATA_DIR` (default: `./data`)
- `AUDIT_PROJECTS_DIR` (default: `$AUDIT_DATA_DIR/projects`)
- `OPENAI_API_KEY_FILE` (default: `./OPENAI.API_KEY`)
- `OPENAI_API_KEY` (plaintext key)
- `OPENAI_API_KEY_PASSWORD` (decrypt automatically)

## Notes

- On first run, a legacy `projects/` folder (if present) is migrated into `data/projects/`.
- The server is intended for local use; do not expose it publicly without adding real authentication.
