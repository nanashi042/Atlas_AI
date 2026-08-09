import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config.settings import settings

logger = logging.getLogger(__name__)

db_url = settings.DATABASE_URL or "sqlite:///./atlas.db"

# connect_args check_same_thread is needed for SQLite multi-threading
connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}

engine = create_engine(db_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class DatabaseInitializationError(RuntimeError):
    """The configured database is unreachable or its schema cannot be created."""


def init_db(raise_on_error: bool = False):
    """Creates database tables if they do not exist."""
    try:
        # Ensure all model modules have registered their tables with Base.
        import app.models  # noqa: F401
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error("Database initialization failed: %s", e)
        if raise_on_error:
            raise DatabaseInitializationError("Database initialization failed.") from e
        return False
    return True
