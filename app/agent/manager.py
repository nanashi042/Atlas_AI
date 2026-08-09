import logging
import re
from app.agent.intent import (
    detect_intent,
    Intent,
)
from app.memory.conversation_memory import conversation_memory
from app.services.ai_service import chat, research_company_ai
from app.services.briefing_service import briefing_service
from app.services.briefing_preference_service import (
    briefing_preference_service,
    BriefingPreferenceError,
)
from app.scheduler.scheduler import morning_briefing_scheduler
from app.services.finance.company_resolver import resolve_company_ticker
from app.services.finance.company_research import CompanyResearchService
from app.services.finance.finnhub_client import (
    FinnhubAuthError,
    FinnhubRateLimitError,
    FinnhubTimeoutError,
    FinnhubNotFoundError,
    FinnhubAPIError,
)
from app.services.watchlist_service import (
    watchlist_service,
    WatchlistError,
)
from app.services.alert_service import alert_service, AlertError, PRICE_CHANGE
from app.services.documents.document_qa_service import (
    DOCUMENT_CONTEXT_CLEARED_MESSAGE,
    NO_ACTIVE_DOCUMENT_MESSAGE,
    document_context_store,
    document_qa_service,
    is_document_clear_request,
    is_explicit_document_question,
    is_likely_document_question,
)
from app.services.finance.company_resolver import COMPANY_NAME_TO_TICKER, KNOWN_TICKERS
from app.services.documents.financial_document_service import (
    financial_document_service,
    is_financial_document_question,
)

logger = logging.getLogger(__name__)


_CLARIFICATION_MESSAGE = (
    "Which company do you mean? Please provide the company name or ticker symbol."
)


def _mentions_known_company(message: str) -> bool:
    text = (message or "").lower()
    return (
        any(name in text for name in COMPANY_NAME_TO_TICKER)
        or any(word.upper() in KNOWN_TICKERS for word in re.findall(r"\b[A-Za-z]{1,5}\b", message or ""))
    )


def _resolve_company(text: str, history):
    """
    Resolve a ticker from free-form text, optionally using recent
    conversation history. Returns ticker or None.
    """
    try:
        return resolve_company_ticker(text, history=history)
    except Exception as e:
        logger.warning(f"Company resolution failed for '{text}': {e}")
        return None


async def _handle_watchlist_add(user_message: str, session_id: str, history) -> str:
    """Resolve a company from the message and persist it on the user's watchlist."""
    resolved = watchlist_service.resolve_company_for_user(user_message, history=history)
    if not resolved:
        conversation_memory.add_message(session_id, role="user", content=user_message)
        conversation_memory.add_message(session_id, role="model", content=_CLARIFICATION_MESSAGE)
        return _CLARIFICATION_MESSAGE

    symbol, company_name = resolved
    try:
        added, message = watchlist_service.add_to_watchlist(
            user_id=session_id, symbol=symbol, company_name=company_name
        )
    except WatchlistError as e:
        logger.error(f"Watchlist add failed for user '{session_id}': {e}")
        message = "⚠️ I couldn't update your watchlist right now. Please try again shortly."
    else:
        conversation_memory.add_message(session_id, role="user", content=user_message)
        conversation_memory.add_message(session_id, role="model", content=message)
        return message

    conversation_memory.add_message(session_id, role="user", content=user_message)
    conversation_memory.add_message(session_id, role="model", content=message)
    return message


async def _handle_watchlist_remove(user_message: str, session_id: str, history) -> str:
    """Resolve a company from the message and remove it from the watchlist."""
    resolved = watchlist_service.resolve_company_for_user(user_message, history=history)
    if not resolved:
        conversation_memory.add_message(session_id, role="user", content=user_message)
        conversation_memory.add_message(session_id, role="model", content=_CLARIFICATION_MESSAGE)
        return _CLARIFICATION_MESSAGE

    symbol, _ = resolved
    try:
        removed, message = watchlist_service.remove_from_watchlist(
            user_id=session_id, symbol=symbol
        )
    except WatchlistError as e:
        logger.error(f"Watchlist remove failed for user '{session_id}': {e}")
        message = "⚠️ I couldn't update your watchlist right now. Please try again shortly."

    conversation_memory.add_message(session_id, role="user", content=user_message)
    conversation_memory.add_message(session_id, role="model", content=message)
    return message


async def _handle_watchlist_list(user_message: str, session_id: str) -> str:
    """Return the user's current watchlist as a concise Telegram message."""
    try:
        entries = watchlist_service.get_watchlist(session_id)
    except WatchlistError as e:
        logger.error(f"Watchlist list failed for user '{session_id}': {e}")
        message = "⚠️ I couldn't load your watchlist right now. Please try again shortly."
    else:
        if not entries:
            message = (
                "Your watchlist is empty.\n\n"
                "Tell me something like 'Track Nvidia' to add a company."
            )
        else:
            lines = ["Your watchlist:\n"]
            for entry in entries:
                lines.append(f"• {entry['company_name']} ({entry['symbol']})")
            message = "\n".join(lines)

    conversation_memory.add_message(session_id, role="user", content=user_message)
    conversation_memory.add_message(session_id, role="model", content=message)
    return message


def _extract_alert_threshold(user_message: str):
    """Extract a positive percentage from natural alert wording, if supplied."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)", user_message, re.IGNORECASE)
    return float(match.group(1)) if match else None


async def _handle_alert_create(user_message: str, session_id: str, history) -> str:
    threshold = _extract_alert_threshold(user_message)
    if threshold is None:
        return "Please specify a positive percentage, for example: 'Alert me if NVDA moves more than 5%'."

    symbol = _resolve_company(user_message, history)
    if not symbol:
        return "Which stock should I alert you about? Please provide a ticker or company name."

    try:
        created, alert = alert_service.create_alert(
            user_id=session_id,
            symbol=symbol,
            threshold_percentage=threshold,
            alert_type=PRICE_CHANGE,
        )
    except AlertError as exc:
        return f"I couldn't create that alert: {exc}"

    threshold_label = f"{alert.threshold_percentage:g}%"
    if not created:
        return (
            f"Your {alert.symbol} PRICE_CHANGE alert for {threshold_label} is already active."
        )
    return (
        f"Done. I'll alert you when {alert.symbol} changes by {threshold_label} or more "
        f"({alert.alert_type})."
    )


async def _handle_alert_list(session_id: str) -> str:
    try:
        alerts = alert_service.list_alerts(session_id, enabled_only=True)
    except AlertError:
        logger.exception("Could not list alerts for user '%s'.", session_id)
        return "⚠️ I couldn't load your alerts right now. Please try again shortly."

    if not alerts:
        return "You don't have any active price-movement alerts."
    lines = ["Your active alerts:", ""]
    for alert in alerts:
        lines.append(
            f"• {alert['symbol']}: {alert['alert_type']} of "
            f"{alert['threshold_percentage']:g}% or more"
        )
    return "\n".join(lines)


async def _handle_alert_remove(user_message: str, session_id: str, history) -> str:
    symbol = _resolve_company(user_message, history)
    if not symbol:
        return "Which alert should I stop? Please provide its ticker symbol or company name."
    try:
        matches = [
            alert for alert in alert_service.list_alerts(session_id, enabled_only=True)
            if alert["symbol"] == symbol
        ]
        if not matches:
            return f"You don't have an active {symbol} alert."
        if len(matches) > 1:
            return f"You have multiple active {symbol} alerts. Please specify the threshold you want to stop."
        alert_service.disable_alert(session_id, matches[0]["id"])
        return f"Done. Your {symbol} price-movement alert has been disabled."
    except AlertError:
        logger.exception("Could not disable alert for user '%s'.", session_id)
        return "⚠️ I couldn't update your alert right now. Please try again shortly."


def _briefing_time_label(preference) -> str:
    return preference.briefing_time.strftime("%I:%M %p").lstrip("0")


async def _handle_briefing_preference(intent: Intent, session_id: str) -> str:
    """Persist natural-language briefing controls and keep the job in sync."""
    try:
        if intent == Intent.BRIEFING_ENABLE:
            preference = briefing_preference_service.set_enabled(session_id, True)
            morning_briefing_scheduler.schedule_daily_briefing(preference)
            return (
                "Done. I'll send your personalized financial briefing every day at "
                f"{_briefing_time_label(preference)} {preference.timezone}."
            )
        if intent == Intent.BRIEFING_DISABLE:
            preference = briefing_preference_service.set_enabled(session_id, False)
            morning_briefing_scheduler.remove_daily_briefing(session_id)
            return "Done. Your daily briefing has been disabled."
        preference = briefing_preference_service.get_preference(session_id)
        if preference.morning_briefing_enabled:
            return (
                "Your daily briefing is enabled for "
                f"{_briefing_time_label(preference)} {preference.timezone}."
            )
        return "Your daily briefing is currently disabled."
    except BriefingPreferenceError as exc:
        logger.error("Briefing preference operation failed for user '%s': %s", session_id, exc)
        return "⚠️ I couldn't update your briefing settings right now. Please try again shortly."


async def process_message(user_message: str, session_id: str = "default") -> str:
    """
    Central Manager Orchestrator for Atlas AI.
    Routes user messages based on intent classification.
    """
    intent = detect_intent(user_message)
    logger.info(f"Processing message for session '{session_id}' with intent '{intent}'.")

    if intent == Intent.GREETING:
        return (
            "👋 Hello! I'm Atlas AI.\n\n"
            "I can help with:\n"
            "📈 Market research\n"
            "🏢 Company analysis\n"
            "⭐ Your personal watchlist\n"
            "💬 General questions"
        )

    if intent == Intent.DOCUMENT_CLEAR or is_document_clear_request(user_message):
        document_context_store.clear_document(session_id)
        return DOCUMENT_CONTEXT_CLEARED_MESSAGE

    if intent == Intent.COMPANY_RESEARCH:
        history = conversation_memory.get_history(session_id)
        symbol = _resolve_company(user_message, history)

        if not symbol:
            conversation_memory.add_message(session_id, role="user", content=user_message)
            conversation_memory.add_message(session_id, role="model", content=_CLARIFICATION_MESSAGE)
            return _CLARIFICATION_MESSAGE

        research_service = CompanyResearchService()
        try:
            research_result = await research_service.get_company_research(symbol)
            return await research_company_ai(user_message, research_result, session_id=session_id)
        except FinnhubAuthError as e:
            logger.error(f"Finnhub authentication error: {e}")
            return "⚠️ Finnhub API Key is missing or invalid. Please check your configuration."
        except FinnhubRateLimitError as e:
            logger.warning(f"Finnhub rate limit hit: {e}")
            return "⏱️ Finnhub API rate limit reached. Please try your request again in a minute."
        except FinnhubTimeoutError as e:
            logger.error(f"Finnhub timeout error: {e}")
            return "⏳ Finnhub API request timed out. Please try again in a few seconds."
        except FinnhubNotFoundError as e:
            logger.warning(f"Company not found: {e}")
            return f"🔍 Could not find company or market data for ticker symbol '{symbol}'. Please verify the ticker."
        except (FinnhubAPIError, Exception) as e:
            logger.error(f"Unexpected error in company research for '{symbol}': {e}", exc_info=True)
            return "⚠️ Financial market data service is temporarily unavailable. Please try again shortly."

    if intent == Intent.WATCHLIST_ADD:
        history = conversation_memory.get_history(session_id)
        return await _handle_watchlist_add(user_message, session_id, history)

    if intent == Intent.WATCHLIST_REMOVE:
        history = conversation_memory.get_history(session_id)
        return await _handle_watchlist_remove(user_message, session_id, history)

    if intent == Intent.WATCHLIST_LIST:
        return await _handle_watchlist_list(user_message, session_id)

    if intent in {Intent.ALERT_CREATE, Intent.ALERT_LIST, Intent.ALERT_REMOVE}:
        history = conversation_memory.get_history(session_id)
        if intent == Intent.ALERT_CREATE:
            response = await _handle_alert_create(user_message, session_id, history)
        elif intent == Intent.ALERT_LIST:
            response = await _handle_alert_list(session_id)
        else:
            response = await _handle_alert_remove(user_message, session_id, history)
        conversation_memory.add_message(session_id, role="user", content=user_message)
        conversation_memory.add_message(session_id, role="model", content=response)
        return response

    if intent in {
        Intent.BRIEFING_ENABLE,
        Intent.BRIEFING_DISABLE,
        Intent.BRIEFING_STATUS,
    }:
        response = await _handle_briefing_preference(intent, session_id)
        conversation_memory.add_message(session_id, role="user", content=user_message)
        conversation_memory.add_message(session_id, role="model", content=response)
        return response

    if intent == Intent.BRIEFING:
        try:
            response = await briefing_service.generate_briefing(
                user_id=session_id, user_message=user_message
            )
        except Exception as e:
            logger.error(
                f"Briefing routing failed for user '{session_id}': {e}",
                exc_info=True,
            )
            response = (
                "⚠️ I couldn't generate your briefing right now. "
                "Please try again shortly."
            )
        conversation_memory.add_message(session_id, role="user", content=user_message)
        conversation_memory.add_message(session_id, role="model", content=response)
        return response

    # Existing product intents are deliberately handled first so an active
    # document cannot hijack research, watchlist, alert, or briefing flows.
    active_document = document_context_store.get_document(session_id)
    if active_document and not _mentions_known_company(user_message) and is_financial_document_question(user_message):
        return await financial_document_service.answer(user_message, active_document)
    if active_document and not _mentions_known_company(user_message) and is_likely_document_question(user_message):
        return await document_qa_service.answer(user_message, active_document)
    if not active_document and is_explicit_document_question(user_message):
        return NO_ACTIVE_DOCUMENT_MESSAGE

    if intent == Intent.GENERAL_CHAT:
        return await chat(user_message, session_id=session_id)

    # Default fallback to general chat
    return await chat(user_message, session_id=session_id)
