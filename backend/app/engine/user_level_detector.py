import re
from typing import Dict, Any

class UserLevelDetector:
    def __init__(self):
        # Vocabulary lists for Signal 1
        self.complex_vocab = [
            "vector db", "embeddings", "langgraph", "checkpointer", "peft", 
            "quantization", "rag", "hybrid", "bm25", "dense retrieval", "semantic cache",
            "presidio", "celery", "async", "prometheus", "observability",
            "multi-tenant", "vectors", "multi-hop", "tradeoff", "checkpointing"
        ]
        self.simple_vocab = [
            "chatbot", "ai search", "smart assistant", "ai for my company",
            "chat with my data", "ai helper", "document search"
        ]

    def _score_vocab(self, text: str) -> int:
        """Signal 1: Vocabulary complexity (0-20 points)"""
        text_lower = text.lower()
        score = 0
        
        complex_count = sum(1 for term in self.complex_vocab if term in text_lower)
        score += min(complex_count * 5, 20)
        
        simple_count = sum(1 for term in self.simple_vocab if term in text_lower)
        score -= min(simple_count * 2, score)
        
        return max(score, 0)

    def _score_specificity(self, text: str) -> int:
        """Signal 2: Question specificity (0-20 points)"""
        score = 0
        text_lower = text.lower()
        
        # Length check
        if len(text.split()) > 15:
            score += 5
            
        # Number/Metrics check (e.g. 50-person, 30%, 500M)
        if re.search(r'\d+%|\d+-person|\d+[gmkt]?b|\d+[mk]', text_lower):
            score += 10
            
        # Specific architectural needs
        if any(term in text_lower for term in ["hybrid", "pipeline", "cache", "async", "scale"]):
            score += 5
            
        return min(score, 20)

    def _score_density(self, text: str) -> int:
        """Signal 3: Technical terminology density (0-20 points)"""
        words = text.lower().split()
        if not words: return 0
        
        tech_terms = set([
            "api", "database", "latency", "throughput", "scale", "deployment",
            "architecture", "framework", "model", "token", "context", "prompt",
            "fine-tuning", "rag", "vector", "embedding", "cache", "serverless",
            "langchain", "chromadb", "python", "redis", "celery", "prometheus",
            "llamaindex", "checkpointing", "retrieval", "indexing", "masking"
        ])
        
        term_count = sum(1 for word in words if word.strip(',.') in tech_terms)
        
        if term_count >= 5:
            return 20 # High density -> senior/architect
        elif term_count >= 2:
            return 10 # Medium density -> junior_dev
        else:
            return 0  # Low density -> civilian/business

    def _score_structure(self, text: str) -> int:
        """Signal 4: Question structure (0-20 points)"""
        text_lower = text.lower()
        
        if "tradeoff" in text_lower or "trade-off" in text_lower or ("vs" in text_lower and "system" in text_lower) or "evaluating" in text_lower:
            return 20 # Architect
        elif "need" in text_lower and "," in text_lower and len(text.split()) > 10 and "how" not in text_lower:
            return 15 # Senior (Declarative architecture requests)
        elif "how do i build" in text_lower or "how to implement" in text_lower or "i want to build" in text_lower:
            return 10 # Junior
        elif "automate" in text_lower or "reduce" in text_lower or "increase" in text_lower:
            return 10 # Business structure (focus on ROI/metrics)
        elif "what is" in text_lower or "explain" in text_lower or "i want ai for" in text_lower:
            return 5  # Civilian
        return 0

    def _score_memory(self, session_history: list) -> int:
        """Signal 5: Prior session memory (0-20 points)"""
        # If returning user with history of complex questions
        if not session_history:
            return 0
        
        # Simple heuristic: if they've had more than 5 interactions, assume some baseline
        score = min(len(session_history) * 2, 20)
        return score

    def detect_persona(self, message: str, session_history: list = None) -> str:
        """Detects the user persona based on the 5 signals."""
        session_history = session_history or []
        
        s1 = self._score_vocab(message)
        s2 = self._score_specificity(message)
        s3 = self._score_density(message)
        s4 = self._score_structure(message)
        s5 = self._score_memory(session_history)
        
        total_score = s1 + s2 + s3 + s4 + s5
        
        if total_score <= 15:
            return "civilian"
        elif total_score <= 25:
            return "business"
        elif total_score <= 50:
            return "junior_dev"
        elif total_score <= 65:
            return "senior_engineer"
        else:
            return "architect"

def run_detection(state: Dict[str, Any]) -> Dict[str, Any]:
    """Integration function for the LangGraph pipeline."""
    # Assuming state contains a list of messages
    messages = state.get("messages", [])
    if not messages:
        return state
        
    # Get the latest user message
    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        return state
        
    latest_msg = user_messages[-1].get("content", "")
    history = user_messages[:-1]
    
    # Re-evaluate every 3 messages
    if len(user_messages) % 3 == 1 or "user_level" not in state.get("requirements", {}):
        detector = UserLevelDetector()
        detected_level = detector.detect_persona(latest_msg, history)
        
        reqs = state.get("requirements", {})
        reqs["user_level"] = detected_level
        return {"requirements": reqs}
        
    return {}
