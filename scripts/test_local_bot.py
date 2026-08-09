#!/usr/bin/env python3
"""Local, non-network test for the bot manager using .env.example.

This script loads `.env.example` (and `.env` if present) and calls
`process_message()` for a few sample messages. It does NOT contact
Telegram or other external services; it's useful to verify routing and
handler logic that doesn't require API keys.

Run:
    python3 scripts/test_local_bot.py

"""
import asyncio
import os


def load_env_file(path: str):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            # Only set if not already in environment to allow overrides
            if k not in os.environ:
                os.environ[k] = v


# Load defaults from .env.example then override from .env if present.
load_env_file(".env.example")
load_env_file(".env")

async def run_tests():
    try:
        from app.agent.manager import process_message
    except Exception as e:
        print("Failed to import process_message:", e)
        raise

    cases = [
        ("/start", "start command"),
        ("hello", "greeting"),
        ("What is NVIDIA?", "company research (may require API keys)"),
    ]

    for text, desc in cases:
        print(f"Test: {desc!s} -> {text}")
        try:
            reply = await process_message(text, session_id="local-test")
            print("Reply:")
            print(reply)
        except Exception as e:
            print("Handler raised:", repr(e))
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(run_tests())
