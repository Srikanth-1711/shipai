import sys
import os
from pathlib import Path
from pprint import pprint

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.template_engine import ShipAITemplateEngine

def walk_dir(path: Path, prefix: str = ""):
    """Print directory structure."""
    if not path.exists():
        return
    for item in sorted(path.iterdir()):
        print(f"{prefix}|- {item.name}")
        if item.is_dir():
            walk_dir(item, prefix + "   ")

def run_verification():
    engine = ShipAITemplateEngine()
    
    print("=== TEST 1: Enterprise RAG ===")
    config1 = {
      "template": "rag_chatbot",
      "framework": "langgraph", 
      "vector_db": "chroma",
      "search_type": "hybrid",
      "reranker": "bge",
      "cache": "semantic",
      "pii": "presidio",
      "infra_tier": "enterprise",
      "decisions": {
          "search_type": {"choice": "hybrid", "reason": "Best for mixed queries", "book": "Info Retrieval"}
      }
    }
    
    path1 = Path(engine.generate_project(config1))
    print(f"Generated at: {path1}")
    walk_dir(path1)
    
    import time
    time.sleep(1)
    
    print("\n=== TEST 2: Minimal Solo RAG ===")
    config2 = {
      "template": "rag_chatbot",
      "framework": "langchain",
      "vector_db": "chroma", 
      "search_type": "dense",
      "reranker": "none",
      "cache": "none",
      "pii": "none",
      "infra_tier": "minimal",
      "decisions": {
          "infra_tier": {"choice": "minimal", "reason": "Solo dev", "book": "General Practices"}
      }
    }
    
    path2 = Path(engine.generate_project(config2))
    print(f"Generated at: {path2}")
    walk_dir(path2)

if __name__ == "__main__":
    run_verification()
