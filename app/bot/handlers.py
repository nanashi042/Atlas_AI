from telegram import Update
from telegram.ext import ContextTypes

from app.agent.manager import process_message
from app.memory.conversation_memory import conversation_memory
from app.services.documents.document_qa_service import (
    DOCUMENT_CONTEXT_CLEARED_MESSAGE,
    document_context_store,
)


START_MESSAGE = """👋 Welcome to Atlas AI!

Your AI-powered financial research assistant.

📈 Company research
• “What is NVIDIA?”
• “Analyze Tesla”

⭐ Watchlist
• “Add NVDA to my watchlist”
• “Show my watchlist”

🚨 Price alerts
• “Alert me if NVDA moves more than 5%”
• “Show my alerts”

🌅 Daily briefing
• “Enable my morning briefing”
• “Is my daily briefing enabled?”

📄 Financial PDFs
Upload a PDF, then ask:
• “What was the revenue growth?”
• “What are the biggest risks?”

💬 General AI questions
Ask about finance, markets, or investing in plain language.

Try this:
“What is NVIDIA?”
“Add Tesla to my watchlist”
“Alert me if NVDA moves more than 5%”

Financial information is for education only, not financial advice."""


HELP_MESSAGE = """🧭 Atlas AI Help

Company research
“What is NVIDIA?” · “Analyze Tesla”

Watchlist
“Track NVDA” · “Show my watchlist”

Alerts
“Alert me if NVDA moves more than 5%” · “Show my alerts”

Briefings
“Enable my morning briefing” · “/briefing_status”

Documents
Upload a financial PDF, then ask “What was the revenue?” or “What are the risks?”
Use /document_clear to remove the active document.

General questions
Ask any finance or market question naturally.

Informational use only — not financial advice."""


async def briefing_on_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session_id = str(update.effective_user.id)
    reply = await process_message("Enable my morning briefing", session_id=session_id)
    await update.message.reply_text(reply)


async def briefing_off_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session_id = str(update.effective_user.id)
    reply = await process_message("Disable my morning briefing", session_id=session_id)
    await update.message.reply_text(reply)


async def briefing_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session_id = str(update.effective_user.id)
    reply = await process_message("Is my daily briefing enabled?", session_id=session_id)
    await update.message.reply_text(reply)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(START_MESSAGE)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_MESSAGE)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session_id = str(update.effective_user.id)
    conversation_memory.clear_history(session_id)
    await update.message.reply_text("🧹 Your conversation memory has been cleared!")


async def document_clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session_id = str(update.effective_user.id)
    document_context_store.clear_document(session_id)
    await update.message.reply_text(DOCUMENT_CONTEXT_CLEARED_MESSAGE)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    session_id = str(update.effective_user.id) if update.effective_user else "default"
    reply = await process_message(user_message, session_id=session_id)
    await update.message.reply_text(reply)
