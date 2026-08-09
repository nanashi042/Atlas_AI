# Atlas AI

Atlas AI is a Telegram-based financial assistant for company research, watchlists, scheduled briefings and price alerts, plus bounded PDF analysis and document Q&A.

## Features

- AI financial assistant backed by Gemini
- Company research using Finnhub
- Persistent personal watchlist
- Personalized scheduled morning briefings
- Scheduled price-change alerts and Telegram notifications
- Financial PDF extraction and analysis
- Conversational document Q&A
- Financial document intelligence with page-aware metrics and risks
- Guided onboarding with `/start`, `/help`, and a Telegram command menu

## Architecture

Telegram updates are routed through the central agent manager to focused services. SQLAlchemy persists conversation memory, watchlists, briefing preferences, and alerts in SQLite by default. APScheduler runs the in-process briefing and alert jobs. PDFs are downloaded to a temporary file, extracted with PyMuPDF, bounded in memory, and then deleted. Gemini receives only the bounded document context when document features are used.

The bot runs as a single Telegram polling process. Natural-language requests are routed to company research, watchlist, alert, briefing, general-chat, or document services. `/start` and `/help` explain the available capabilities to new users.

## Tech stack

- Python 3.12+ for deployment (the local development environment may use a newer Python release)
- python-telegram-bot
- Google Gemini (`google-genai`)
- Finnhub via `httpx`
- FastAPI and Uvicorn (including `/health`)
- SQLAlchemy and SQLite
- APScheduler
- PyMuPDF

## Setup

```powershell
git clone <your-repository-url>
cd atlas-ai
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` using environment-specific values. Never commit it.

```env
APP_NAME=Atlas AI
APP_VERSION=1.0.0
ENVIRONMENT=production
DATABASE_URL=sqlite:///./atlas.db
LOG_LEVEL=INFO

TELEGRAM_BOT_TOKEN=replace_with_your_token
GEMINI_API_KEY=replace_with_your_key
GEMINI_MODEL=gemini-flash-latest
FINNHUB_API_KEY=replace_with_your_key

PRICE_ALERT_INTERVAL_MINUTES=15
DOCUMENT_MAX_CHARACTERS=24000
DOCUMENT_MAX_PAGES=40
TELEGRAM_MAX_MESSAGE_LENGTH=3900
```

The bot initializes the database schema at startup. Start it with:

```powershell
.venv\Scripts\python.exe -m app.run_bot
```

The optional HTTP health server can be run separately:

```powershell
.venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000
```

`GET /health` returns `{"status":"ok"}`.

## Telegram examples

- `What is NVIDIA?`
- `Track NVDA`
- `Alert me if NVDA moves more than 5%`
- `Enable my morning briefing`
- Upload a financial PDF, then ask `What was the revenue?`
- `What are the biggest risks?`
- `Forget this document` or `/document_clear`

Available Telegram commands:

- `/start` — welcome and feature overview
- `/help` — concise usage reference
- `/clear` or `/reset` — clear conversation memory
- `/briefing_on` — enable the daily briefing
- `/briefing_off` — disable the daily briefing
- `/briefing_status` — check briefing status
- `/document_clear` — clear the active PDF context

Watchlists and alerts are intentionally natural-language features; they are not separate `/watchlist` or `/alerts` commands.

## Testing

```powershell
.venv\Scripts\python.exe -m unittest discover tests -v
```

The current suite contains **187 tests**. The command above is the authoritative check after future changes.

## Docker deployment

The image runs the Telegram polling bot by default. Build it with:

```powershell
docker build -t atlas-ai:latest .
```

For SQLite, mount a persistent writable directory and set its container path explicitly:

```powershell
docker run --rm --name atlas-ai `
  --env-file .env `
  -e DATABASE_URL=sqlite:////data/atlas.db `
  -v ${PWD}\data:/data `
  atlas-ai:latest
```

Do not bake `.env` or a local database into the image. To run the optional health service instead of the bot, override the command with `uvicorn app.main:app --host 0.0.0.0 --port 8000` and publish port `8000`.

## Limitations

- APScheduler is in-process: scheduled work runs only while this one bot process is alive. It is not a durable job queue and multiple bot instances would duplicate scheduler work.
- SQLite is suitable for a single-instance deployment. Use a managed server database before scaling to multiple instances.
- Document context is bounded, in-memory, one document per Telegram user, and lost on restart.
- No OCR, RAG, embeddings, vector database, or multi-document search is implemented.
- Finnhub and Gemini availability, quotas, and rate limits can affect responses.
