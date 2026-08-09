from fastapi import FastAPI

from app.config.settings import settings

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
)


@app.get("/")
def root():
    return {
        "message": f"Welcome to {settings.APP_NAME} 🚀",
        "environment": settings.ENVIRONMENT,
        "version": settings.APP_VERSION,
    }


@app.get("/health")
def health():
    """Minimal liveness endpoint with no configuration or user-data exposure."""
    return {"status": "ok"}
