"""Convenience runner for local development.

Run this from the project root to start the polling bot:

    python run.py

This ensures the project root is on `sys.path` so `import app` works.
"""

from app.run_bot import run_bot


if __name__ == "__main__":
    run_bot()
