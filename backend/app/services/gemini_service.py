"""
Gemini Service — Optional LLM reasoning for model selection and explanations.
Does NOT replace deterministic scoring, only enhances with reasoning.
"""
import json
import logging
from typing import Optional
import httpx

from app.config import settings

logger = logging.getLogger("shipai.gemini")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = settings.GEMINI_MODEL
        self.enabled = bool(self.api_key)

    async def explain_model_choice(
        self,
        hardware_specs: dict,
        selected_model: str,
        alternatives: list,
        use_case: str = "general",
    ) -> dict:
        """
        Get Gemini's reasoning on why selected_model is best for this hardware.
        Returns: {"explanation": "...", "confidence": 0.8, "concerns": [...]}
        """
        if not self.enabled:
            return {"error": "Gemini API not configured", "explanation": None}

        prompt = f"""You are a hardware-LLM matching expert. Provide BRIEF reasoning (2-3 sentences max).

Hardware: {json.dumps(hardware_specs, indent=2)}
Selected Model: {selected_model}
Alternatives: {', '.join(alternatives)}
Use Case: {use_case}

Explain BRIEFLY why {selected_model} is the best choice here. If there's a concern, mention it.
Be concise."""

        try:
            response = await self._call_gemini(prompt)
            return {
                "explanation": response,
                "confidence": 0.85,
                "concerns": [],
            }
        except Exception as e:
            logger.error(f"Gemini explanation failed: {e}")
            return {"error": str(e), "explanation": None}

    async def validate_user_model_suggestion(
        self,
        suggested_model: str,
        hardware_specs: dict,
        available_models: list,
    ) -> dict:
        """
        User suggests a model. Gemini validates if it's reasonable for this hardware.
        Returns: {"valid": True/False, "reasoning": "...", "concerns": [...]}
        """
        if not self.enabled:
            return {"valid": True, "reasoning": "Gemini validation skipped", "concerns": []}

        prompt = f"""You are a hardware-LLM expert. Validate if a model choice is reasonable.

User suggested: {suggested_model}
Hardware: {json.dumps(hardware_specs, indent=2)}
Available models: {', '.join(available_models)}

Is {suggested_model} a reasonable choice? Consider:
1. Memory requirements (does it fit?)
2. Capability match (will it handle the task?)
3. Alternatives (is there a better fit?)

Respond with:
- VALID or INVALID
- Brief reasoning (1-2 sentences)
- Any concerns (if INVALID, explain why)"""

        try:
            response = await self._call_gemini(prompt)
            is_valid = "VALID" in response.upper()
            concerns = []
            if "concern" in response.lower() or "not recommended" in response.lower():
                concerns.append("See reasoning below")

            return {
                "valid": is_valid,
                "reasoning": response,
                "concerns": concerns,
                "suggested_model": suggested_model,
            }
        except Exception as e:
            logger.error(f"Gemini validation failed: {e}")
            return {
                "valid": False,
                "reasoning": f"Validation error: {str(e)}",
                "concerns": ["Unable to validate"],
            }

    async def rank_models_for_hardware(
        self,
        hardware_specs: dict,
        candidate_models: list,
        node_purpose: str = "general",
    ) -> list:
        """
        Gemini re-ranks models based on context.
        Used as optional second-pass ranking after deterministic scoring.
        Returns: [{"model": "qwen3:4b", "rank": 1, "reasoning": "..."}, ...]
        """
        if not self.enabled:
            return []

        prompt = f"""You are a hardware-LLM expert. Rank models for this hardware.

Hardware: {json.dumps(hardware_specs, indent=2)}
Purpose: {node_purpose}
Candidates: {', '.join(candidate_models)}

Rank these models 1-{len(candidate_models)} where 1 is best fit.
Explain each ranking in 1 sentence max.

Return JSON like:
[
  {{"model": "qwen3:4b", "rank": 1, "reason": "Best balance of speed and quality"}},
  {{"model": "qwen2.5:7b", "rank": 2, "reason": "Higher quality but slower"}}
]"""

        try:
            response = await self._call_gemini(prompt)
            return self._parse_json_response(response)
        except Exception as e:
            logger.error(f"Gemini ranking failed: {e}")
            return []

    async def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API and return text response."""
        if not self.api_key:
            raise ValueError("Gemini API key not configured")

        url = f"{GEMINI_BASE_URL}/models/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 500,
            }
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        # Extract text from response
        try:
            text = data["contents"][0]["parts"][0]["text"]
            return text
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Invalid Gemini response: {e}")

    def _parse_json_response(self, response: str) -> list:
        """Extract JSON array from Gemini response."""
        try:
            # Try direct parse
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
            if match:
                return json.loads(match.group(1))
            raise ValueError("Could not parse JSON from response")


# Singleton instance
gemini_service = GeminiService()
