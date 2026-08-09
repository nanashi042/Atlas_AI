import logging
from app.ai.llm import generate_response
from app.memory.conversation_memory import conversation_memory
from app.services.finance.models import CompanyResearchResult

logger = logging.getLogger(__name__)


async def chat(user_message: str, session_id: str = "default") -> str:
    """Handles general conversational chat turns with memory persistence."""
    history = conversation_memory.get_history(session_id)
    response_text = await generate_response(user_message, history=history)

    if response_text and not response_text.startswith("[Error"):
        conversation_memory.add_message(session_id, role="user", content=user_message)
        conversation_memory.add_message(session_id, role="model", content=response_text)

    return response_text


async def research_company_ai(
    user_message: str,
    research_result: CompanyResearchResult,
    session_id: str = "default"
) -> str:
    """
    Synthesizes normalized company research data into an analyst-style response using Gemini.
    Ensures zero hallucination of financial numbers by forcing reliance on supplied Finnhub data.
    """
    history = conversation_memory.get_history(session_id)
    data_context = research_result.to_formatted_context()

    prompt = (
        "You are Atlas AI Financial Copilot.\n"
        "Synthesize the following real financial data retrieved from Finnhub to answer the user's research request.\n\n"
        "CRITICAL RULES:\n"
        "1. Use ONLY the supplied financial/company data below. Do NOT invent numbers, news, dates, prices, or financial facts.\n"
        "2. Clearly distinguish fetched facts from interpretation.\n"
        "3. Be concise and professional in analyst style.\n"
        "4. Explain 'Why it matters' based strictly on the provided data.\n"
        "5. Do NOT give personalized buy, sell, or investment recommendations.\n"
        "6. Mention 'Source: Finnhub' at the end.\n"
        "7. Do not claim data is real-time unless stated.\n\n"
        f"SUPPLIED FINANCIAL DATA (from Finnhub):\n{data_context}\n\n"
        f"USER REQUEST: {user_message}\n"
    )

    response_text = await generate_response(prompt, history=history)

    # Fallback to pre-formatted template if Gemini returns an error code
    if not response_text or response_text.startswith("[Error") or response_text.startswith("⚠️"):
        logger.warning(f"Gemini generation returned error/fallback. Structuring direct template response.")
        price_str = f"${research_result.current_price:.2f}" if research_result.current_price is not None else "N/A"
        pct_str = f"{research_result.percent_change:+.2f}%" if research_result.percent_change is not None else "N/A"
        
        response_text = (
            f"🏢 **{research_result.company_name} ({research_result.symbol})**\n\n"
            f"💰 Price: {price_str} | Today: {pct_str}\n"
            f"🏛️ Exchange: {research_result.exchange} | Industry: {research_result.industry}\n\n"
            f"📌 Website: {research_result.website}\n\n"
            f"Source: Finnhub"
        )

    conversation_memory.add_message(session_id, role="user", content=user_message)
    conversation_memory.add_message(session_id, role="model", content=response_text)

    return response_text