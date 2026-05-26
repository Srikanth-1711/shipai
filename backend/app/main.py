"""
ShipAI — Main FastAPI Application
The brain of the platform. Single entry point for all services.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.config import settings
from app.routes import system, models, advisor, templates, license, auth, projects, model_suggestions

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

    # Initialize Database
    try:
        from app.core.database import init_db
        from app.db.feedback_db import init_db as init_feedback_db
        await init_db()
        init_feedback_db()
        logger.info("🗄️  Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")

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
app.include_router(auth.router)
app.include_router(projects.router)
app.include_router(model_suggestions.router)

from app.api import ws, feedback
app.include_router(ws.router)
app.include_router(feedback.router, prefix="/api")


@app.get("/api")
async def root():
    """Root API endpoint — platform info."""
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

# Mount Frontend UI for Desktop App
frontend_dir = os.path.join(os.path.dirname(__file__), "../../frontend")
if os.path.exists(frontend_dir):
    css_dir = os.path.join(frontend_dir, "css")
    if os.path.exists(css_dir):
        app.mount("/css", StaticFiles(directory=css_dir), name="css")
        
    js_dir = os.path.join(frontend_dir, "js")
    if os.path.exists(js_dir):
        app.mount("/js", StaticFiles(directory=js_dir), name="js")
        
    assets_dir = os.path.join(frontend_dir, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        @app.get("/")
        async def serve_index():
            return FileResponse(index_path)
