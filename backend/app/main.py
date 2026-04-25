"""
ShipAI — Main FastAPI Application
The brain of the platform. Single entry point for all services.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import system, models, advisor, templates, license

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("shipai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # Startup
    logger.info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} starting...")
    logger.info(f"📡 Ollama endpoint: {settings.OLLAMA_BASE_URL}")

    # Check Ollama connectivity
    from app.services.ollama_service import ollama_service
    status = await ollama_service.health_check()
    if status["status"] == "healthy":
        logger.info(f"✅ Ollama connected (v{status.get('version', '?')})")
        models_list = await ollama_service.list_models()
        logger.info(f"📦 {len(models_list)} model(s) available locally")
    else:
        logger.warning(f"⚠️  Ollama not available: {status.get('error', 'unknown')}")

    # Log hardware info
    from app.services.hardware_checker import check_hardware
    hw = check_hardware()
    logger.info(f"🖥️  Hardware: {hw.ram_total_gb}GB RAM | GPU: {hw.gpu.name if hw.gpu else 'None'} | Tier: {hw.tier_label}")

    logger.info(f"✅ {settings.APP_NAME} ready! http://{settings.HOST}:{settings.PORT}")

    yield

    # Shutdown
    logger.info(f"👋 {settings.APP_NAME} shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="The AI Engineer You Can Hire at Scale — Build, deploy, and scale AI products with local models.",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(system.router)
app.include_router(models.router)
app.include_router(advisor.router)
app.include_router(templates.router)
app.include_router(license.router)


@app.get("/")
async def root():
    """Root endpoint — platform info."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "tagline": "Ship AI Products in Minutes, Not Months",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/system/health",
            "hardware": "/api/system/hardware",
            "models": "/api/models/",
            "generate": "/api/models/generate",
            "chat": "/api/models/chat",
            "infra_patterns": "/api/system/infra-patterns",
        },
    }
