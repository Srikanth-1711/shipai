"""
ShipAI Configuration
Central configuration management using pydantic-settings.
All settings can be overridden via environment variables or .env file.
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # App
    APP_NAME: str = "ShipAI"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_DEFAULT_MODEL: str = "qwen2.5:3b"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_TIMEOUT: int = 120  # seconds

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./shipai.db"

    # Security
    SECRET_KEY: str = "shipai-dev-secret-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # License
    LICENSE_SERVER_URL: Optional[str] = None  # None = offline validation only

    # Gemini API (optional, for model reasoning)
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-flash-latest"
    GEMINI_ENABLE_REASONING: bool = False  # Enable for explanations

    # Paths
    PROJECTS_DIR: str = str(Path.home() / ".shipai" / "projects")
    TEMPLATES_DIR: str = str(Path(__file__).parent / "templates")
    MODELS_DIR: str = str(Path.home() / ".shipai" / "models")

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    RATE_LIMIT_BURST: int = 20

    # Cache
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    CACHE_MAX_SIZE: int = 1000  # max cached items

    class Config:
        env_file = ".env"
        env_prefix = "SHIPAI_"


settings = Settings()
