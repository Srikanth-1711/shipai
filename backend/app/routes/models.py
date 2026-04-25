"""
ShipAI — Model Management Routes
Endpoints for managing local Ollama models.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from app.services.ollama_service import ollama_service

router = APIRouter(prefix="/api/models", tags=["Models"])


class GenerateRequest(BaseModel):
    prompt: str
    model: str | None = None
    system: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2048


class ChatRequest(BaseModel):
    messages: list[dict]
    model: str | None = None
    temperature: float = 0.7


class ModelPullRequest(BaseModel):
    model: str


@router.get("/")
async def list_models():
    """List all locally available models."""
    models = await ollama_service.list_models()
    return {"models": models, "count": len(models)}


@router.get("/running")
async def running_models():
    """Get models currently loaded in memory."""
    models = await ollama_service.get_running_models()
    return {"running_models": models}


@router.post("/pull")
async def pull_model(req: ModelPullRequest):
    """Download a new model from Ollama registry."""
    result = await ollama_service.pull_model(req.model)
    return result


@router.post("/stop")
async def stop_model(req: ModelPullRequest):
    """Unload a model from memory."""
    result = await ollama_service.stop_model(req.model)
    return result


@router.post("/generate")
async def generate(req: GenerateRequest):
    """Generate a response from a local LLM."""
    result = await ollama_service.generate(
        prompt=req.prompt,
        model=req.model,
        system=req.system,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    return result


@router.post("/chat")
async def chat(req: ChatRequest):
    """Chat with a local LLM (multi-turn)."""
    result = await ollama_service.chat(
        messages=req.messages,
        model=req.model,
        temperature=req.temperature,
    )
    return result
