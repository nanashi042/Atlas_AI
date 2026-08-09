import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session

from app.database.session import SessionLocal, init_db
from app.models.conversation import ChatMessage
# Importing WatchlistEntry here ensures its table is registered with
# Base.metadata before init_db() runs at module import time.
from app.models.watchlist import WatchlistEntry  # noqa: F401

logger = logging.getLogger(__name__)

# Database initialization should happen at application startup; avoid
# calling `init_db()` at import time to prevent import-time failures on
# serverless hosts. The application startup handler will call `init_db()`.


class ConversationMemory:
    """
    Manages session-isolated conversation history with persistence in SQLAlchemy/Postgres.
    Provides sliding-window context history for multi-turn Gemini conversations.
    """

    def __init__(self, default_limit: int = 10):
        self.default_limit = default_limit

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Stores a message (role: 'user' or 'model') in persistent storage."""
        if not session_id or not content:
            return

        db: Session = SessionLocal()
        try:
            msg = ChatMessage(session_id=str(session_id), role=role, content=content)
            db.add(msg)
            db.commit()
            logger.info(f"Saved {role} message to memory for session '{session_id}'.")
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to save message to memory: {e}")
        finally:
            db.close()

    def get_history(self, session_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Retrieves the last N messages for a session formatted for Gemini contents:
        [
            {"role": "user", "parts": [{"text": "..."}]},
            {"role": "model", "parts": [{"text": "..."}]}
        ]
        """
        if not session_id:
            return []

        fetch_limit = limit or self.default_limit
        db: Session = SessionLocal()
        try:
            # Query the latest fetch_limit messages, then reverse to chronological order
            messages = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == str(session_id))
                .order_by(ChatMessage.created_at.desc())
                .limit(fetch_limit)
                .all()
            )
            messages.reverse()

            formatted_history = []
            for msg in messages:
                # Ensure valid roles for Gemini API ('user' or 'model')
                role = "user" if msg.role == "user" else "model"
                formatted_history.append({
                    "role": role,
                    "parts": [{"text": msg.content}]
                })

            return formatted_history
        except Exception as e:
            logger.error(f"Failed to fetch conversation history for session '{session_id}': {e}")
            return []
        finally:
            db.close()

    def clear_history(self, session_id: str) -> bool:
        """Deletes all conversation history for a given session."""
        if not session_id:
            return False

        db: Session = SessionLocal()
        try:
            db.query(ChatMessage).filter(ChatMessage.session_id == str(session_id)).delete()
            db.commit()
            logger.info(f"Cleared conversation memory for session '{session_id}'.")
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to clear history for session '{session_id}': {e}")
            return False
        finally:
            db.close()


# Global Singleton Instance for Conversation Memory
conversation_memory = ConversationMemory(default_limit=10)
