import re
from enum import Enum
from typing import Set

from app.services.finance.company_resolver import COMPANY_NAME_TO_TICKER, KNOWN_TICKERS


class Intent(str, Enum):
    GREETING = "greeting"
    GENERAL_CHAT = "general_chat"

    COMPANY_RESEARCH = "company_research"
    COMPANY_COMPARE = "company_compare"

    MARKET_NEWS = "market_news"

    DOCUMENT_ANALYSIS = "document_analysis"
    DOCUMENT_CLEAR = "document_clear"

    WATCHLIST_ADD = "watchlist_add"
    WATCHLIST_REMOVE = "watchlist_remove"
    WATCHLIST_LIST = "watchlist_list"

    BRIEFING = "briefing"
    BRIEFING_ENABLE = "briefing_enable"
    BRIEFING_DISABLE = "briefing_disable"
    BRIEFING_STATUS = "briefing_status"

    ALERT_CREATE = "alert_create"
    ALERT_LIST = "alert_list"
    ALERT_REMOVE = "alert_remove"

    UNKNOWN = "unknown"


def detect_intent(message: str) -> Intent:
    """
    Rule-based intent detection engine for Atlas AI.
    Classifies user queries into GREETING, COMPANY_RESEARCH, or GENERAL_CHAT.
    """
    if not message:
        return Intent.UNKNOWN

    text = message.lower().strip()
    cleaned_text = re.sub(r"^[^\w\s]+|[^\w\s]+$", "", text)

    greetings: Set[str] = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "greetings",
        "yo",
        "sup",
    }

    if cleaned_text in greetings:
        return Intent.GREETING

    if re.search(r"\b(?:forget|clear|remove|delete)\b.*\b(?:this )?(?:document|pdf|report)\b", text):
        return Intent.DOCUMENT_CLEAR

    # Proactive briefing controls must precede the generic BRIEFING intent.
    briefing_enable_patterns = [
        r"\benable\b.*\b(?:morning|daily)?\s*briefing\b",
        r"\bstart\b.*\b(?:morning|daily)?\s*briefing\b",
        r"\bturn on\b.*\bbriefing\b",
        r"\bsend me\b.*\b(?:morning|daily)\s+briefing\b.*\bevery day\b",
    ]
    briefing_disable_patterns = [
        r"\bdisable\b.*\b(?:morning|daily)?\s*briefing\b",
        r"\bstop\b.*\b(?:morning|daily)?\s*briefing\b",
        r"\bturn off\b.*\bbriefing\b",
    ]
    briefing_status_patterns = [
        r"\bis my\b.*\b(?:morning|daily)\s+briefing\b.*\benabled\b",
        r"\bam i getting\b.*\b(?:morning|daily)\s+briefing\b",
        r"\b(?:morning|daily)\s+briefing\b.*\bstatus\b",
    ]
    if any(re.search(pattern, text) for pattern in briefing_enable_patterns):
        return Intent.BRIEFING_ENABLE
    if any(re.search(pattern, text) for pattern in briefing_disable_patterns):
        return Intent.BRIEFING_DISABLE
    if any(re.search(pattern, text) for pattern in briefing_status_patterns):
        return Intent.BRIEFING_STATUS

    # Price-movement alert controls. These require alert-specific language so
    # existing watchlist and company-research requests retain their routing.
    alert_list_patterns = [
        r"\bwhat alerts do i have\b",
        r"\bshow (?:me )?my alerts\b",
        r"\blist (?:my )?alerts\b",
        r"\bmy alerts\b",
    ]
    alert_remove_patterns = [
        r"\b(?:stop|remove|disable|delete)\b.*\balert\b",
    ]
    alert_create_patterns = [
        r"\balert me\b",
        r"\bcreate (?:an? )?alert\b",
        r"\bnotify me\b.*\b(?:move|moves|change|changes)\b",
        r"\btrack\b.+\bif\b.+\b(?:move|moves|change|changes)\b",
    ]
    if any(re.search(pattern, text) for pattern in alert_list_patterns):
        return Intent.ALERT_LIST
    if any(re.search(pattern, text) for pattern in alert_remove_patterns):
        return Intent.ALERT_REMOVE
    if any(re.search(pattern, text) for pattern in alert_create_patterns):
        return Intent.ALERT_CREATE

    # Explicit research action phrases
    research_action_patterns = [
        r"\btell me about\b",
        r"\bresearch\b",
        r"\banalyze\b",
        r"\bhow is .+ doing\b",
        r"\bwhat is happening with\b",
        r"\bgive me info(?:rmation)?\b",
        r"\bcompany info\b",
        r"\bstock info\b",
        r"\boverview of\b",
        r"\bdetails on\b",
    ]

    for pattern in research_action_patterns:
        if re.search(pattern, text):
            return Intent.COMPANY_RESEARCH

    # ---- Watchlist detection ----
    # Add/tracking patterns take priority over research for explicit phrases.
    # Strong phrases that are unambiguous watchlist_add signals.
    watchlist_add_strong_patterns = [
        r"\badd\b.+\bto\b.+\bwatchlist\b",
        r"\bput\b.+\bon\b.+\bwatchlist\b",
        r"\bstart tracking\b",
        r"\binterested in\b",
        r"\bkeep (?:an? )?eye on\b",
    ]
    # Verbs that suggest watchlist add — must be paired with a known ticker
    # or company name, otherwise fall through to research/general chat.
    watchlist_add_verb_patterns = [
        r"\btrack\b",
        r"\bfollow\b",
        r"\bwatch\b",
    ]
    watchlist_remove_patterns = [
        r"\bremove\b.+\bfrom\b.+\bwatchlist\b",
        r"\bstop tracking\b",
        r"\bstop following\b",
        r"\buntrack\b",
        r"\bdelete\b.+\bfrom\b.+\bwatchlist\b",
    ]
    watchlist_list_patterns = [
        # Direct list/show commands.
        r"\bshow (?:me )?my watchlist\b",
        r"\blist (?:my )?watchlist\b",
        r"\bview (?:my )?watchlist\b",
        # Direct list-style questions about the watchlist itself.
        r"\bwhat'?s on my watchlist\b",
        # Bare reference to "my watchlist" as a noun phrase (without
        # an action verb like "what's happening with").
        r"\bmy watchlist\b",
        r"\bmy (?:tracked|followed|watched) (?:stocks|companies|tickers)\b",
        # "what stocks am I following", "what am I tracking",
        # "what companies am I watching" — true list-style queries.
        r"\bwhat\s+am\s+i\s+(?:tracking|following|watching)\b",
        # "What (stocks/companies) am I following/tracking/watching"
        r"\bwhat\s+(?:stocks|companies|tickers)\s+am\s+i\s+(?:tracking|following|watching)\b",
    ]

    for pattern in watchlist_add_strong_patterns:
        if re.search(pattern, text):
            return Intent.WATCHLIST_ADD
    for pattern in watchlist_remove_patterns:
        if re.search(pattern, text):
            return Intent.WATCHLIST_REMOVE

    # ---- Briefing detection (BEFORE watchlist-list patterns) ----
    # Natural phrases that ask for a personalized watchlist briefing.
    # We intentionally fire BRIEFING ahead of WATCHLIST_LIST so that
    # "What's happening with my watchlist?" becomes a briefing, not
    # a list. The list commands (show/list/view my watchlist, "what's on
    # my watchlist") are protected because none of them match a
    # briefing pattern.
    briefing_patterns = [
        # Explicit "briefing" requests.
        r"\b(?:morning|daily|market|today'?s|evening)\s+briefing\b",
        r"\bbriefing\b",
        # "What's happening with my watchlist / stocks / companies"
        r"\bwhat'?s happening with (?:my )?watchlist\b",
        r"\bwhat'?s happening with (?:my )?(?:stocks|companies|tickers)\b",
        # "What should I know about my stocks today"
        r"\bwhat should i know\b",
        # "What's important today / for my stocks / for my watchlist /
        # for the companies I'm following"
        r"\bwhat'?s important for (?:my |the )?(?:stocks|companies|watchlist|tickers)\b",
        r"\bwhat'?s important today\b",
        # "Give me today's briefing / today's market briefing"
        r"\bgive me (?:today'?s|the)\s+(?:market\s+)?briefing\b",
    ]

    if any(re.search(pattern, text) for pattern in briefing_patterns):
        return Intent.BRIEFING

    for pattern in watchlist_list_patterns:
        if re.search(pattern, text):
            return Intent.WATCHLIST_LIST

    # Verb patterns (track/follow/watch) — fire only when paired with a
    # known company/ticker/cashtag. If the company is unknown, fall through
    # to research/general chat so the user gets a sensible answer rather
    # than a confusing "add to watchlist" flow.
    for pattern in watchlist_add_verb_patterns:
        if re.search(pattern, text):
            if (
                any(name in text for name in COMPANY_NAME_TO_TICKER)
                or re.search(r"\$[A-Za-z]{1,5}\b", text)
                or any(w.upper() in KNOWN_TICKERS for w in re.findall(r"\b[A-Za-z]{1,5}\b", message))
            ):
                return Intent.WATCHLIST_ADD

    # Check for "What is <Company/Ticker>?"
    what_is_match = re.match(r"^what\s+is\s+([a-zA-Z0-9\s\$]+)\??$", text)
    if what_is_match:
        subject = what_is_match.group(1).strip().lower()
        # If subject is a known company, ticker, or cashtag
        if (
            subject in COMPANY_NAME_TO_TICKER
            or subject.upper() in KNOWN_TICKERS
            or subject.startswith("$")
        ):
            return Intent.COMPANY_RESEARCH

    # Check for direct cashtag or ticker research format (e.g. "NVDA", "$TSLA")
    words = re.findall(r"\b[A-Za-z]{1,5}\b", message)
    for word in words:
        if word.startswith("$") or word.upper() in KNOWN_TICKERS:
            # If query is short or ticker-centric, e.g., "NVDA info" or "AAPL"
            if len(message.split()) <= 4:
                return Intent.COMPANY_RESEARCH

    return Intent.GENERAL_CHAT
