# Teen Safety Auditor

A developer-focused web tool that lets you test AI application prompts and
conversation flows against OpenAI's teen safety policy guidelines using the
`gpt-4o-mini` model as a safety evaluator.

Paste individual prompts or multi-turn conversations, and the tool:

- **Scores** each response for policy compliance
- **Highlights** risky outputs with colour-coded severity levels (Safe / Caution / Violation)
- **Generates** a downloadable JSON audit report

This helps app builders quickly identify child-safety policy violations before
shipping their AI products to younger audiences.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [Usage](#usage)
4. [API Reference](#api-reference)
5. [Policy Categories](#policy-categories)
6. [Running Tests](#running-tests)
7. [Project Structure](#project-structure)
8. [Policy Reference Links](#policy-reference-links)
9. [License](#license)

---

## Quick Start

### Prerequisites

- Python 3.11 or later
- An [OpenAI API key](https://platform.openai.com/account/api-keys)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/example/teen_safety_auditor.git
cd teen_safety_auditor

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate.bat     # Windows CMD
# .venv\Scripts\Activate.ps1    # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the example environment file and add your API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### Start the server

```bash
python -m teen_safety_auditor.main
# or via the installed script:
teen-safety-auditor
```

Open your browser at **http://localhost:8000**.

---

## Configuration

Create a `.env` file in the project root (copy from `.env.example`):

```dotenv
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional — override defaults
OPENAI_MODEL=gpt-4o-mini          # Model used for safety evaluation
SAFEGUARD_MODEL=gpt-4o-mini       # Alias used internally
HOST=0.0.0.0                      # Bind address (default: 127.0.0.1)
PORT=8000                          # Bind port    (default: 8000)
DEBUG=false                        # Enable Uvicorn reload (default: false)
```

> **Note:** No prompts or conversation content are ever persisted server-side.
> All data lives in memory for the duration of a single request.

---

## Usage

### Single-Prompt Audit

1. Navigate to the home page.
2. Paste your prompt (and optionally the AI response) into the **Single Prompt** tab.
3. Click **Audit** — results appear inline without a page reload (powered by HTMX).

### Multi-Turn Conversation Audit

1. Switch to the **Conversation** tab.
2. Paste a JSON array of message objects following the OpenAI chat format:

```json
[
  { "role": "user",      "content": "Tell me about online safety." },
  { "role": "assistant", "content": "Sure! Here are some tips..." }
]
```

3. Click **Audit Conversation** to receive a per-turn compliance breakdown.

### Downloading the Audit Report

After any audit, click **Download Report** to save a JSON file containing:

- Per-turn scores and risk levels
- Flagged categories with descriptions
- Remediation hints for each violation
- Overall compliance summary

---

## API Reference

All endpoints accept and return JSON.

### `POST /api/audit/prompt`

Audit a single prompt/response pair.

**Request body:**

```json
{
  "prompt": "string",
  "response": "string (optional)"
}
```

**Response:** `AuditResult` object (see `teen_safety_auditor/models.py`).

---

### `POST /api/audit/conversation`

Audit a multi-turn conversation.

**Request body:**

```json
{
  "turns": [
    { "role": "user",      "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

**Response:** `ConversationAuditResult` object.

---

### `POST /api/audit/report`

Generate and return a downloadable JSON audit report from a full conversation.
Returns `Content-Disposition: attachment` so browsers trigger a file save.

---

## Policy Categories

The auditor evaluates content across the following teen-safety categories:

| Category | Description |
|---|---|
| `sexual_content` | Explicit or suggestive sexual material |
| `self_harm` | Content promoting self-injury or suicide |
| `grooming` | Manipulative relationship-building with minors |
| `violence` | Graphic violence or threats |
| `dangerous_activities` | Instructions for illegal or physically dangerous acts |
| `substance_abuse` | Promotion of drug, alcohol, or substance misuse |
| `privacy_violation` | Eliciting or exposing personal identifying information |
| `hate_speech` | Discriminatory or harassing content |

Risk levels and thresholds are defined in `teen_safety_auditor/policy.py`.

---

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run only unit tests
pytest tests/test_auditor.py -v

# Run only API integration tests
pytest tests/test_api.py -v
```

---

## Project Structure

```
teen_safety_auditor/
├── __init__.py          # Package version and key exports
├── main.py              # FastAPI app factory and Uvicorn entrypoint
├── auditor.py           # Core auditing engine (OpenAI calls + scoring)
├── models.py            # Pydantic request/response models
├── policy.py            # Policy categories, thresholds, system prompts
├── templates/
│   ├── index.html       # Main single-page UI
│   └── partials/
│       └── result_card.html  # HTMX partial result card
└── static/
    └── app.js           # Clipboard, export, and UI polish
tests/
├── fixtures.py          # Shared test fixtures
├── test_auditor.py      # Auditor unit tests
└── test_api.py          # FastAPI integration tests
pyproject.toml
requirements.txt
README.md
```

---

## Policy Reference Links

- [OpenAI Usage Policies](https://openai.com/policies/usage-policies)
- [OpenAI Safety Best Practices](https://platform.openai.com/docs/guides/safety-best-practices)
- [OpenAI Moderation Guide](https://platform.openai.com/docs/guides/moderation)
- [Child Safety Policy](https://openai.com/policies/usage-policies#child-safety)

---

## License

MIT — see [LICENSE](LICENSE) for details.
