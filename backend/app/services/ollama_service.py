"""
ShipAI — Ollama Service
Manages local LLM models via Ollama API.
Handles: model discovery, pulling, inference, health checks.
"""
import httpx
import logging
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


class OllamaService:
    """Interface to the local Ollama runtime."""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.timeout = settings.OLLAMA_TIMEOUT

    async def health_check(self) -> dict:
        """Check if Ollama is running and responsive."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/api/version", timeout=5)
                if resp.status_code == 200:
                    return {"status": "healthy", "version": resp.json().get("version", "unknown")}
        except httpx.ConnectError:
            return {"status": "not_running", "error": "Ollama is not running. Start it with: ollama serve"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
        return {"status": "error", "error": "Unexpected response from Ollama"}

    async def list_models(self) -> list[dict]:
        """List all locally available models."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/api/tags", timeout=10)
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    return [
                        {
                            "name": m.get("name", ""),
                            "size_gb": round(m.get("size", 0) / (1024**3), 2),
                            "modified": m.get("modified_at", ""),
                            "family": m.get("details", {}).get("family", "unknown"),
                            "parameters": m.get("details", {}).get("parameter_size", "unknown"),
                            "quantization": m.get("details", {}).get("quantization_level", "unknown"),
                        }
                        for m in models
                    ]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    async def get_running_models(self) -> list[dict]:
        """Get models currently loaded in memory."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/api/ps", timeout=10)
                if resp.status_code == 200:
                    return resp.json().get("models", [])
        except Exception as e:
            logger.error(f"Failed to get running models: {e}")
        return []

    async def pull_model(self, model_name: str) -> dict:
        """Pull/download a model from Ollama registry."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model_name, "stream": False},
                    timeout=600,  # 10 min for large models
                )
                if resp.status_code == 200:
                    return {"status": "success", "model": model_name}
                return {"status": "error", "error": resp.text}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def stop_model(self, model_name: str) -> dict:
        """Unload a model from memory."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": model_name, "keep_alive": 0},
                    timeout=30,
                )
                return {"status": "unloaded", "model": model_name}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def generate(
        self,
        prompt: str,
        model: str = None,
        system: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict:
        """Generate a response from a local LLM."""
        model = model or settings.OLLAMA_DEFAULT_MODEL
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "response": data.get("response", ""),
                        "model": model,
                        "total_duration_ms": round(data.get("total_duration", 0) / 1e6, 2),
                        "eval_count": data.get("eval_count", 0),
                        "eval_duration_ms": round(data.get("eval_duration", 0) / 1e6, 2),
                    }
                return {"error": f"Ollama returned {resp.status_code}: {resp.text}"}
        except httpx.TimeoutException:
            return {"error": f"Model '{model}' timed out after {self.timeout}s. Try a smaller model."}
        except httpx.ConnectError:
            return {"error": "Cannot connect to Ollama. Is it running? Start with: ollama serve"}
        except Exception as e:
            return {"error": str(e)}

    async def chat(
        self,
        messages: list[dict],
        model: str = None,
        temperature: float = 0.7,
    ) -> dict:
        """Chat with a local LLM (multi-turn conversation)."""
        model = model or settings.OLLAMA_DEFAULT_MODEL
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": temperature},
                    },
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "message": data.get("message", {}),
                        "model": model,
                        "total_duration_ms": round(data.get("total_duration", 0) / 1e6, 2),
                    }
                return {"error": resp.text}
        except Exception as e:
            return {"error": str(e)}

    async def embeddings(self, text: str, model: str = None) -> dict:
        """Generate embeddings for text (used in RAG)."""
        model = model or settings.OLLAMA_EMBEDDING_MODEL
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": model, "prompt": text},
                    timeout=60,
                )
                if resp.status_code == 200:
                    return resp.json()
                return {"error": resp.text}
        except Exception as e:
            return {"error": str(e)}


# Singleton instance
ollama_service = OllamaService()
