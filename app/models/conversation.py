from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, DateTime
from app.database.session import Base


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)  # 'user' or 'model'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<ChatMessage(session_id='{self.session_id}', role='{self.role}', created_at='{self.created_at}')>"
