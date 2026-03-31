# Teen Safety Auditor 🛡️
> Audit AI prompts and conversations against OpenAI teen safety policies — before your app ships.

Teen Safety Auditor is a developer-focused web tool that lets you test AI application prompts and conversation flows against OpenAI's teen safety policy guidelines. Paste individual prompts or multi-turn conversations, and the tool scores each response for policy compliance, highlights risky outputs with color-coded severity levels, and generates a downloadable audit report. Built for app developers who need to identify child-safety policy violations before shipping AI products to younger audiences.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-org/teen_safety_auditor.git
cd teen_safety_auditor

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your environment
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY

# 5. Start the server
uvicorn teen_safety_auditor.main:app --reload
```

Open your browser to **http://127.0.0.1:8000** — the audit UI is ready.

---

## Features

- **Single-prompt & multi-turn auditing** — Evaluate individual prompt/response pairs or full conversation arrays against OpenAI's teen safety policy categories (sexual content, self-harm, grooming, and more).
- **Color-coded risk scoring** — Each turn receives a clear **Safe** 🟢 / **Caution** 🟡 / **Violation** 🔴 classification per policy category, so problem areas are obvious at a glance.
- **Batch conversation mode** — Paste a JSON conversation array and get a per-turn compliance breakdown in a single request.
- **Downloadable audit reports** — Export a structured JSON report summarizing all flagged turns, category scores, and remediation hints.
- **Zero-persistence design** — No prompts are stored server-side. Your test data stays private.

---

## Usage Examples

### Web UI

Navigate to `http://127.0.0.1:8000`, paste your prompt and response into the form, and click **Audit**. Results appear inline with color-coded badges per policy category.

### REST API — Single Prompt

```bash
curl -X POST http://127.0.0.1:8000/api/audit/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the best way to make new friends at school?",
    "response": "Try joining clubs that match your interests!"
  }'
```

**Response:**
```json
{
  "overall_risk_level": "safe",
  "overall_score": 0.05,
  "category_scores": [
    {
      "category": "sexual_content",
      "score": 0.01,
      "risk_level": "safe",
      "label": "Sexual Content"
    },
    {
      "category": "self_harm",
      "score": 0.02,
      "risk_level": "safe",
      "label": "Self-Harm"
    }
  ],
  "flagged_categories": [],
  "remediation_hints": []
}
```

### REST API — Multi-Turn Conversation

```bash
curl -X POST http://127.0.0.1:8000/api/audit/conversation \
  -H "Content-Type: application/json" \
  -d '{
    "turns": [
      {"role": "user",      "content": "Hi, I need some advice."},
      {"role": "assistant", "content": "Of course! What's on your mind?"},
      {"role": "user",      "content": "I've been feeling really down lately."},
      {"role": "assistant", "content": "I'm sorry to hear that. Have you talked to a trusted adult?"}
    ]
  }'
```

### REST API — Download Audit Report

```bash
curl -X POST http://127.0.0.1:8000/api/audit/report \
  -H "Content-Type: application/json" \
  -d '{ "prompt": "...", "response": "..." }' \
  -o audit_report.json
```

### Health Check

```bash
curl http://127.0.0.1:8000/health
# {"status": "ok"}
```

---

## Project Structure

```
teen_safety_auditor/
├── pyproject.toml                          # Project metadata and dependency config
├── requirements.txt                        # Pinned pip dependencies
├── .env.example                            # Environment variable template
├── README.md
│
├── teen_safety_auditor/
│   ├── __init__.py                         # Package init, version, and key exports
│   ├── main.py                             # FastAPI app factory, routes, and server entrypoint
│   ├── auditor.py                          # Core auditing engine (OpenAI calls, scoring, classification)
│   ├── models.py                           # Pydantic request/response models
│   ├── policy.py                           # Policy categories, thresholds, and system prompt templates
│   │
│   ├── templates/
│   │   ├── index.html                      # Main single-page UI
│   │   └── partials/
│   │       └── result_card.html            # HTMX partial for per-turn result cards
│   │
│   └── static/
│       └── app.js                          # Clipboard copy, report export, UI polish
│
└── tests/
    ├── __init__.py
    ├── fixtures.py                         # Shared fixtures and mock data
    ├── test_auditor.py                     # Unit tests for auditing logic
    ├── test_api.py                         # Integration tests for FastAPI endpoints
    ├── test_models.py                      # Unit tests for Pydantic models
    └── test_policy.py                      # Unit tests for policy config
```

---

## Configuration

Copy `.env.example` to `.env` and set your values:

```dotenv
# .env

# Required
OPENAI_API_KEY=sk-your-key-here

# Optional — override defaults
OPENAI_MODEL=gpt-4o-mini       # Model used for safety evaluation
SAFEGUARD_MODEL=gpt-4o-mini    # Safeguard model (can differ from main model)
HOST=127.0.0.1                 # Server bind address
PORT=8000                      # Server port
DEBUG=false                    # Enable FastAPI debug mode
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes | — | Your OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Model used for evaluation |
| `SAFEGUARD_MODEL` | No | `gpt-4o-mini` | Safeguard evaluator model |
| `HOST` | No | `127.0.0.1` | Server bind host |
| `PORT` | No | `8000` | Server port |
| `DEBUG` | No | `false` | Enable debug mode |

---

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_auditor.py
pytest tests/test_api.py
```

All tests mock OpenAI API calls — no real API requests or costs are incurred during testing.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built with [Jitter](https://github.com/jitter-ai) - an AI agent that ships code daily.*
