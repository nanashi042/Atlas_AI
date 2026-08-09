import asyncio
from google import genai
from google.genai.errors import ClientError, ServerError
import logging

from app.config.settings import settings

logger = logging.getLogger(__name__)


async def generate_response(user_message: str, history: list = None) -> str:
    if not settings.GEMINI_API_KEY:
        logger.error("Gemini API key is missing in settings.")
        return "[Error] Gemini API key is missing. Please set GEMINI_API_KEY in your .env file."

    # Create the Gemini client once inside generate_response() for each user request
    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    # Build multi-turn contents list for Gemini if history exists
    contents = list(history) if history else []
    contents.append({
        "role": "user",
        "parts": [{"text": user_message}]
    })
    
    # Model fallback hierarchy to ensure ultra-high availability
    candidate_models = [
        settings.GEMINI_MODEL or "gemini-flash-latest",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-flash-latest"
    ]
    # Remove duplicates while preserving order
    models_to_try = list(dict.fromkeys([m for m in candidate_models if m]))

    last_error = None
    for model_name in models_to_try:
        for attempt in range(2):  # Try up to 2 times per model
            try:
                logger.info(f"Generating AI response using model '{model_name}' (history_turns={len(history or [])})...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                )
                if response and response.text:
                    logger.info(f"Successfully received response from model '{model_name}'.")
                    return response.text
            except (ServerError, ClientError, Exception) as e:
                err_msg = str(e)
                last_error = err_msg
                logger.warning(f"Attempt {attempt+1} on model '{model_name}' failed: {err_msg[:120]}")

                # If 503 UNAVAILABLE or 500/504, wait 1s before retrying or switching model
                if "503" in err_msg or "UNAVAILABLE" in err_msg or "500" in err_msg:
                    await asyncio.sleep(1)
                    continue
                # If 429 quota error or 400 invalid key error, break immediately
                elif "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    break
                elif "404" in err_msg or "NOT_FOUND" in err_msg:
                    break  # Model not found, try next candidate model
                else:
                    await asyncio.sleep(0.5)

    # If all candidate models failed, return user-friendly status
    if last_error:
        if "429" in last_error or "RESOURCE_EXHAUSTED" in last_error:
            logger.error("Gemini API Quota Exceeded (429).")
            return "[Error 429] Gemini API Quota Exceeded. Please check your API Key on Google AI Studio (https://aistudio.google.com/app/apikey)."
        if "503" in last_error or "UNAVAILABLE" in last_error:
            logger.error("Gemini service unavailable (503).")
            return "⚠️ Gemini service is currently busy. Please try sending your message again in a few seconds."
        logger.error(f"Failed to generate response after model retries: {last_error[:150]}")
        return f"[Error] Could not get response from Gemini: {last_error[:150]}"

    logger.error("Unknown error in generate_response.")
    return "[Error] Something went wrong while connecting to AI brain."