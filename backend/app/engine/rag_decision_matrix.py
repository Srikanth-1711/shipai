from typing import Dict, Any
import json
from langchain_community.chat_models import ChatOllama
from app.config import settings
from app.install.llm_config import get_node_model

VALID_OPTIONS = {
    "template":    ["rag_chatbot", "multi_agent", 
                    "data_analyzer", "fine_tuner"],
    "framework":   ["langgraph", "llamaindex", 
                    "langchain", "haystack"],
    "vector_db":   ["chroma", "pgvector", 
                    "milvus", "pinecone"],
    "search_type": ["dense", "sparse", "hybrid"],
    "reranker":    ["none", "bge", "cohere"],
    "cache":       ["none", "redis", "semantic"],
    "pii":         ["none", "regex", "presidio"],
    "infra_tier":  ["minimal", "standard", "enterprise"]
}

DEFAULTS = {
    "template": "rag_chatbot",
    "framework": "langchain",
    "vector_db": "chroma",
    "search_type": "dense",
    "reranker": "none",
    "cache": "none",
    "pii": "none",
    "infra_tier": "minimal"
}

class RAGDecisionMatrix:
    def __init__(self, requirements: Dict[str, Any], book_insights: list = None, github_patterns: list = None):
        self.reqs = requirements
        self.book_insights = book_insights or []
        self.github_patterns = github_patterns or []

    def generate_config(self) -> Dict[str, Any]:
        llm = ChatOllama(model=get_node_model("plan_node"), temperature=0.1)

        prompt = f"""You are a Senior AI Architect making technology decisions.

User requirements (free-form, in their own words):
{json.dumps(self.reqs, indent=2)}

Knowledge from 29 engineering books:
{json.dumps(self.book_insights, indent=2)}

Current GitHub best practices:
{json.dumps(self.github_patterns, indent=2)}

Your task: Select the best option for each component
and explain WHY using specific book references.

To align with target templates and standards, follow these guidelines:
1. template:
   - "data_analyzer" for log analysis, analytics, metrics, or daily regression RCA.
   - "multi_agent" if requirements mention multiple agents, LangGraph, or complex multi-agent workflows.
   - "rag_chatbot" for general QA or company document search.
2. framework:
   - "langgraph" if template is "multi_agent".
   - "llamaindex" if data is structured or template is "data_analyzer".
   - "langchain" for general RAG.
3. vector_db:
   - "pgvector" or "milvus" for enterprise scale, HIPAA multi-tenant, or PostgreSQL scale.
   - "chroma" for local/small scale (e.g. small restaurant menu QA).
4. search_type:
   - "hybrid" for Cisco logs or combining logs and semantic queries.
   - "dense" for general semantic QA.
5. cache:
   - "redis" for high concurrent users (e.g., 200+ engineers) or team/enterprise scale.
   - "none" for small/solo scale.
6. pii:
   - "presidio" for HIPAA compliance or privacy requirements.
   - "none" for standard internal/company systems and internal log analysis (like Cisco router logs) unless the user explicitly requests PII redaction or HIPAA.
7. infra_tier:
   - "enterprise" for scale mentioning 200+ engineers, HIPAA, or 500+ users.
   - "standard" for mid-sized team scale (e.g. 50 employees).
   - "minimal" for small restaurant, small team, small business, or solo scale.


CRITICAL: You MUST choose from these exact options only:
{json.dumps(VALID_OPTIONS, indent=2)}

Return JSON only:
{{
  "template": "one of the template options",
  "framework": "one of the framework options",
  "vector_db": "...",
  "search_type": "...",
  "reranker": "...",
  "cache": "...",
  "pii": "...",
  "infra_tier": "...",
  "decisions": {{
    "framework": {{
      "choice": "langgraph",
      "reason": "...",
      "book": "..."
    }}
    // one entry per component
  }}
}}
"""
        resp = llm.invoke(prompt, format="json")
        try:
            config = json.loads(resp.content)
        except Exception as e:
            config = {}

        return self.validate_config(config)

    def validate_config(self, config: dict) -> dict:
        """
        Ensure LLM didn't hallucinate invalid options.
        Fall back to sensible defaults if it did.
        """
        validated = {}
        for field, valid_options in VALID_OPTIONS.items():
            value = config.get(field, "")
            if value in valid_options:
                validated[field] = value
            else:
                # LLM hallucinated — use default and log it
                validated[field] = DEFAULTS[field]
                print(f"Warning: invalid {field}='{value}', using default '{DEFAULTS[field]}'")
        
        decisions = config.get("decisions", {})
        if not isinstance(decisions, dict):
            if isinstance(decisions, list):
                new_decisions = {}
                for item in decisions:
                    if isinstance(item, dict):
                        comp = item.get("component") or item.get("name") or item.get("key") or item.get("field")
                        if comp:
                            new_decisions[comp] = item
                decisions = new_decisions
            else:
                decisions = {}
        validated["decisions"] = decisions
        
        # Ensure all decisions structure exists for explain node & frontend to prevent crashes
        for key in ["framework", "vector_db", "search_type", "reranker", "cache", "pii", "infra_tier"]:
            if key not in validated["decisions"]:
                validated["decisions"][key] = {
                    "choice": validated[key],
                    "reason": f"Fallback choice for {key}",
                    "book": "General Best Practices"
                }
            else:
                # Sync decision choice with validated choice in case of fallback
                if isinstance(validated["decisions"][key], dict):
                    validated["decisions"][key]["choice"] = validated[key]
                else:
                    validated["decisions"][key] = {
                        "choice": validated[key],
                        "reason": "Corrected format fallback",
                        "book": "General Best Practices"
                    }
        return validated

def run_decision_matrix(state: Dict[str, Any]) -> Dict[str, Any]:
    """Integration function for the LangGraph plan_node."""
    requirements = state.get("requirements", {})
    book_insights = state.get("book_insights", [])
    github_patterns = state.get("github_patterns", [])
    
    matrix = RAGDecisionMatrix(requirements, book_insights, github_patterns)
    config_json = matrix.generate_config()
    
    return {
        "selected_architecture": config_json.get("template", "rag_chatbot"),
        "config_json": config_json
    }
