from fastapi import FastAPI
from src.api.routes import router
from src.utils.config import settings

app = FastAPI(
    title=settings.app_name,
    description="ML-powered anomaly detection for personal finance transactions",
    version="1.0.0"
)

app.include_router(router, prefix="/api/v1")

@app.get("/health")
def health_check():
    return {"status": "healthy"}
