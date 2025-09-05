# AuditBatchTry

---

## Prerequisites

- **Python:** 3.9+ (3.10 or newer recommended)
- **Git:** to clone and manage the repository
- **OpenAI account & API key** (system will ask for the API key) 

---

## Quick Start

Clone the repository:

```bash
git clone https://github.com/PlanetMacro/AuditBatchTry.git
cd AuditBatchTry
````

Install dependencies inside a virtual environment (see OS-specific instructions below), then:

```bash
pip install -r requirements.txt
```
---

## OS-specific setup

### Linux (Debian/Ubuntu)

1. Ensure Python and venv tools are available:

   ```bash
   sudo apt update
   sudo apt install -y python3 python3-venv python3-pip
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Upgrade pip (recommended) and install deps:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Deactivate when done:

   ```bash
   deactivate
   ```

---

### macOS

1. Verify Python 3 is available (via Xcode tools, Homebrew, or python.org). With Homebrew:

   ```bash
   brew install python
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Upgrade pip (recommended) and install deps:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Deactivate when done:

   ```bash
   deactivate
   ```

---

## Execution

   ```bash
   python3 main.py
   ```

---

## Use

- Change parameters like LOG_NUMBER_OF_BATCHES in main.py
- Put the audit code into PUT_PROMPT_HERE.txt
- Read the result from AUDIT_RESULT.txt


