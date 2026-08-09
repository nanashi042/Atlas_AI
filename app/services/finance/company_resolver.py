import re
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# Common company name to ticker symbol mapping for quick, reliable resolution
COMPANY_NAME_TO_TICKER: Dict[str, str] = {
    "nvidia": "NVDA",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "amd": "AMD",
    "advanced micro devices": "AMD",
    "netflix": "NFLX",
    "intel": "INTC",
    "palantir": "PLTR",
    "uber": "UBER",
    "disney": "DIS",
    "walt disney": "DIS",
    "boeing": "BA",
    "coca cola": "KO",
    "coke": "KO",
    "pepsi": "PEP",
    "pepsico": "PEP",
}

# Known popular ticker symbols set
KNOWN_TICKERS = {
    "NVDA", "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA",
    "AMD", "NFLX", "INTC", "PLTR", "UBER", "DIS", "BA", "KO", "PEP",
}


def resolve_company_ticker(
    text: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Optional[str]:
    """
    Extracts or resolves a company ticker symbol from user message or recent history context.

    Returns ticker symbol (e.g., 'NVDA') if confidently found, or None if clarification is needed.
    """
    if not text:
        return None

    clean_text = text.strip()
    lowered = clean_text.lower()

    # 1. Direct match for company names in mapping
    for name, ticker in COMPANY_NAME_TO_TICKER.items():
        # Match whole word boundary for company names
        pattern = r"\b" + re.escape(name) + r"\b"
        if re.search(pattern, lowered):
            logger.info(f"Resolved company name '{name}' to ticker '{ticker}' from message.")
            return ticker

    # 2. Match cashtags (e.g., $NVDA, $AAPL)
    cashtag_match = re.search(r"\$([A-Za-z]{1,5})\b", clean_text)
    if cashtag_match:
        ticker = cashtag_match.group(1).upper()
        logger.info(f"Resolved cashtag ticker '{ticker}' from message.")
        return ticker

    # 3. Match uppercase ticker symbols (e.g., NVDA, TSLA, AAPL)
    # Only accept tickers that are in KNOWN_TICKERS to avoid false positives
    # like "TRACK", "STOCK", or other capitalized common nouns.
    words = re.findall(r"\b[A-Z]{1,5}\b", clean_text)
    for word in words:
        if word in KNOWN_TICKERS:
            logger.info(f"Resolved explicit ticker '{word}' from message.")
            return word

    # 4. Check lowercase match if word matches known tickers (e.g., user writes "nvda", "tsla")
    raw_words = re.findall(r"\b[a-zA-Z]{1,5}\b", clean_text)
    for w in raw_words:
        upper_w = w.upper()
        if upper_w in KNOWN_TICKERS:
            logger.info(f"Resolved known ticker '{upper_w}' from lowercase message.")
            return upper_w

    # 5. Check conversation history context if recent turn mentioned a company
    if history:
        for turn in reversed(history[-4:]):
            content = turn.get("parts", [{}])[0].get("text", "") if isinstance(turn.get("parts"), list) else turn.get("content", "")
            if content:
                # Try resolving ticker from recent history
                recent_ticker = resolve_company_ticker(content, history=None)
                if recent_ticker:
                    logger.info(f"Resolved ticker '{recent_ticker}' from recent history context.")
                    return recent_ticker

    logger.info(f"Could not confidently resolve ticker from message: '{text}'")
    return None
