import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config.settings import settings

logger = logging.getLogger(__name__)

# Lazily create the engine and sessionmaker so importing this module does
# not attempt to open the database file (important for serverless hosts).
_engine = None
_SessionLocal = None

Base = declarative_base()


class DatabaseInitializationError(RuntimeError):
    """The configured database is unreachable or its schema cannot be created."""


def _ensure_engine():
    global _engine, _SessionLocal
    if _engine is not None and _SessionLocal is not None:
        return

    db_url = settings.DATABASE_URL or "sqlite:///./atlas.db"
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    try:
        _engine = create_engine(db_url, connect_args=connect_args)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    except Exception as e:
        logger.error("Failed to create DB engine: %s", e)
        raise


class _LazySessionFactory:
    def __call__(self):
        _ensure_engine()
        return _SessionLocal()


# Export the callable factory so existing code calling `SessionLocal()` keeps working.
SessionLocal = _LazySessionFactory()


def init_db(raise_on_error: bool = False):
    """Creates database tables if they do not exist.

    The function is tolerant of initialization failures on platforms where a
    writable filesystem is unavailable (serverless). When `raise_on_error` is
    True the function raises `DatabaseInitializationError` on failure.
    """
    try:
        _ensure_engine()
        # Ensure all model modules have registered their tables with Base.
        import app.models  # noqa: F401
        Base.metadata.create_all(bind=_engine)
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error("Database initialization failed: %s", e)
        if raise_on_error:
            raise DatabaseInitializationError("Database initialization failed.") from e
        return False
    return True
