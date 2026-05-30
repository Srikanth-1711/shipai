import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class ShipAITemplateEngine:
    def __init__(self):
        self.generated_projects_dir = Path.home() / ".shipai" / "projects"
        self.generated_projects_dir.mkdir(parents=True, exist_ok=True)

    def generate_project(self, config: dict, output_dir: str | None = None) -> str:
        """
        Takes config.json from Pattern Matcher.
        Returns path to generated project directory.
        """
        if output_dir:
            project_dir = Path(output_dir)
            project_dir.mkdir(parents=True, exist_ok=True)
            # Core python package structures
            (project_dir / "backend" / "app" / "core").mkdir(parents=True, exist_ok=True)
            (project_dir / "backend" / "app" / "api").mkdir(parents=True, exist_ok=True)
            (project_dir / "backend" / "app" / "workers").mkdir(parents=True, exist_ok=True)
            (project_dir / "eval").mkdir(parents=True, exist_ok=True)
            
            # Make them packages
            (project_dir / "backend" / "__init__.py").touch()
            (project_dir / "backend" / "app" / "__init__.py").touch()
            (project_dir / "backend" / "app" / "core" / "__init__.py").touch()
            (project_dir / "backend" / "app" / "workers" / "__init__.py").touch()
            (project_dir / "eval" / "__init__.py").touch()
        else:
            project_dir = self.create_project_directory(config.get("template", "rag_chatbot"))
        
        # Always generate base
        self.generate_base(project_dir, config)
        
        # Conditionally inject modules
        module_map = {
            "search_type": self.inject_search_module,
            "reranker":    self.inject_reranker_module,
            "pii":         self.inject_pii_module,
            "cache":       self.inject_cache_module,
            "infra_tier":  self.inject_infra_module,
        }
        
        for config_key, injector in module_map.items():
            value = config.get(config_key, "none")
            injector(project_dir, value)
        
        # Always inject evaluation scaffold
        self.inject_eval_scaffold(project_dir, config)
        
        # Generate requirements.txt from active modules
        self.generate_requirements(project_dir, config)
        
        # Generate .env.example from active modules
        self.generate_env_example(project_dir, config)
        
        # Generate README explaining what was built and why
        self.generate_readme(project_dir, config)
        
        # Generate Dockerfile
        self.generate_dockerfile(project_dir, config)
        
        return str(project_dir)

    def generate_dockerfile(self, project_dir: Path, config: dict):
        dockerfile = """
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        self._write_file(project_dir / "Dockerfile", dockerfile)

    def create_project_directory(self, template: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_dir = self.generated_projects_dir / f"{template}_{timestamp}"
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Core python package structures
        (project_dir / "backend" / "app" / "core").mkdir(parents=True, exist_ok=True)
        (project_dir / "backend" / "app" / "api").mkdir(parents=True, exist_ok=True)
        (project_dir / "backend" / "app" / "workers").mkdir(parents=True, exist_ok=True)
        (project_dir / "eval").mkdir(parents=True, exist_ok=True)
        
        # Make them packages
        (project_dir / "backend" / "__init__.py").touch()
        (project_dir / "backend" / "app" / "__init__.py").touch()
        (project_dir / "backend" / "app" / "core" / "__init__.py").touch()
        (project_dir / "backend" / "app" / "workers" / "__init__.py").touch()
        (project_dir / "eval" / "__init__.py").touch()
        
        return project_dir

    def _write_file(self, path: Path, content: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.strip() + "\n")

    def generate_base(self, project_dir: Path, config: dict):
        main_py = """
from fastapi import FastAPI, Request
from app.api import router

app = FastAPI(title="ShipAI Generated Project")
app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}
"""
        self._write_file(project_dir / "backend" / "app" / "main.py", main_py)

        api_py = """
from fastapi import APIRouter
router = APIRouter()
"""
        self._write_file(project_dir / "backend" / "app" / "api" / "__init__.py", api_py)

    def inject_search_module(self, project_dir: Path, value: str):
        content = ""
        if value == "dense":
            content = """
def retrieve(query: str, k: int = 10):
    embedding = embedder.embed_query(query)
    return vectorstore.similarity_search_by_vector(
        embedding, k=k
    )
"""
        elif value == "sparse":
            content = """
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, documents: list[str]):
        tokenized = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        self.documents = documents
    
    def retrieve(self, query: str, k: int = 10):
        scores = self.bm25.get_scores(query.split())
        top_k = scores.argsort()[-k:][::-1]
        return [self.documents[i] for i in top_k]
"""
        elif value == "hybrid":
            content = """
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

class HybridRetriever:
    def __init__(self, vectorstore, documents, 
                 dense_weight=0.6, sparse_weight=0.4):
        dense = vectorstore.as_retriever(
            search_kwargs={"k": 20}
        )
        sparse = BM25Retriever.from_texts(
            documents, k=20
        )
        self.ensemble = EnsembleRetriever(
            retrievers=[dense, sparse],
            weights=[dense_weight, sparse_weight]
        )
    
    def retrieve(self, query: str) -> list:
        return self.ensemble.invoke(query)
"""
        if content:
            self._write_file(project_dir / "backend" / "app" / "core" / "retriever.py", content)

    def inject_reranker_module(self, project_dir: Path, value: str):
        content = ""
        if value == "bge":
            content = """
from sentence_transformers import CrossEncoder

class BGEReranker:
    def __init__(self):
        self.model = CrossEncoder(
            "BAAI/bge-reranker-base",
            max_length=512
        )
    
    def rerank(self, query: str, 
               documents: list[str], 
               top_k: int = 5) -> list[str]:
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1], 
            reverse=True
        )
        return [doc for doc, _ in ranked[:top_k]]
"""
        elif value == "cohere":
            content = """
import cohere
import os

class CohereReranker:
    def __init__(self):
        self.client = cohere.Client(
            os.getenv("COHERE_API_KEY")
        )
    
    def rerank(self, query: str,
               documents: list[str],
               top_k: int = 5) -> list[str]:
        results = self.client.rerank(
            query=query,
            documents=documents,
            top_n=top_k,
            model="rerank-english-v3.0"
        )
        return [r.document["text"] 
                for r in results.results]
"""
        if content:
            self._write_file(project_dir / "backend" / "app" / "core" / "reranker.py", content)

    def inject_pii_module(self, project_dir: Path, value: str):
        content = ""
        if value == "regex":
            content = """
import re

PII_PATTERNS = {
    "email":    r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b',
    "phone":    r'\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b',
    "ssn":      r'\\b\\d{3}-\\d{2}-\\d{4}\\b',
    "credit_card": r'\\b\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}\\b',
}

def redact_pii(text: str) -> str:
    for pii_type, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", text)
    return text
"""
        elif value == "presidio":
            content = """
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

class PresidioPIIGuard:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        # Entities to detect and redact
        self.entities = [
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
            "CREDIT_CARD", "US_SSN", "IBAN_CODE",
            "IP_ADDRESS", "LOCATION", "NRP",
            "DATE_TIME", "MEDICAL_LICENSE", "URL"
        ]
    
    def redact(self, text: str) -> str:
        results = self.analyzer.analyze(
            text=text,
            entities=self.entities,
            language="en"
        )
        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={
                "DEFAULT": OperatorConfig(
                    "replace", 
                    {"new_value": "<REDACTED>"}
                )
            }
        )
        return anonymized.text
    
    def redact_batch(self, texts: list[str]) -> list[str]:
        return [self.redact(t) for t in texts]

# Apply at TWO points — indexing time AND query time
pii_guard = PresidioPIIGuard()
"""
        if content:
            self._write_file(project_dir / "backend" / "app" / "core" / "pii_guard.py", content)

    def inject_cache_module(self, project_dir: Path, value: str):
        content = ""
        if value == "redis":
            content = """
import redis
import hashlib
import json
import os

class RedisCache:
    def __init__(self, ttl: int = 3600):
        self.client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True
        )
        self.ttl = ttl
    
    def get(self, query: str) -> str | None:
        key = hashlib.sha256(query.encode()).hexdigest()
        cached = self.client.get(f"shipai:cache:{key}")
        return json.loads(cached) if cached else None
    
    def set(self, query: str, response: str):
        key = hashlib.sha256(query.encode()).hexdigest()
        self.client.setex(
            f"shipai:cache:{key}",
            self.ttl,
            json.dumps(response)
        )
"""
        elif value == "semantic":
            content = """
from redisvl.extensions.llmcache import SemanticCache
import os

class SemanticCacheLayer:
    def __init__(self, threshold: float = 0.90):
        # threshold: 0.90 = 90% similar queries 
        # get cached response
        self.cache = SemanticCache(
            name="shipai_semantic_cache",
            redis_url=os.getenv(
                "REDIS_URL", "redis://localhost:6379"
            ),
            distance_threshold=threshold
        )
    
    def get(self, query: str) -> str | None:
        results = self.cache.check(prompt=query)
        return results[0]["response"] if results else None
    
    def set(self, query: str, response: str):
        self.cache.store(
            prompt=query,
            response=response
        )
"""
        if content:
            self._write_file(project_dir / "backend" / "app" / "core" / "cache.py", content)

    def inject_infra_module(self, project_dir: Path, value: str):
        docker_compose = "version: '3.8'\nservices:\n"
        docker_compose += """
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
"""
        if value in ["standard", "enterprise"]:
            docker_compose += """
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
"""
        if value == "enterprise":
            docker_compose += """
  celery_worker:
    build: .
    command: celery -A app.workers.indexing_worker worker --loglevel=info
    depends_on:
      - redis
      
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
      
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
      
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
"""
            circuit_breaker = """
from circuitbreaker import circuit

# @circuit(failure_threshold=5, recovery_timeout=30, expected_exception=Exception)
# async def call_llm_with_protection(prompt: str):
#     return await llm.ainvoke(prompt)
"""
            self._write_file(project_dir / "backend" / "app" / "core" / "circuit_breaker.py", circuit_breaker)

            rate_limiter = """
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# @app.get("/query")
# @limiter.limit("30/minute")
# async def query_endpoint(request: Request, q: str):
#     pass
"""
            self._write_file(project_dir / "backend" / "app" / "core" / "rate_limiter.py", rate_limiter)

            indexing_worker = """
import os
from celery import Celery

celery_app = Celery(
    "shipai",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0")
)

@celery_app.task(bind=True, max_retries=3)
def async_index_documents(self, file_paths: list[str]):
    try:
        pass # index_document(path)
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)
"""
            self._write_file(project_dir / "backend" / "app" / "workers" / "indexing_worker.py", indexing_worker)

            observability = """
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "shipai_requests_total",
    "Total requests",
    ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "shipai_request_duration_seconds",
    "Request latency",
    ["endpoint"]
)
LLM_CALL_COUNT = Counter(
    "shipai_llm_calls_total",
    "Total LLM calls",
    ["model", "cache_hit"]
)
RETRIEVAL_SCORE = Histogram(
    "shipai_retrieval_relevance_score",
    "Retrieval relevance scores"
)
"""
            self._write_file(project_dir / "backend" / "app" / "core" / "observability.py", observability)

        self._write_file(project_dir / "docker-compose.yml", docker_compose)

    def inject_eval_scaffold(self, project_dir: Path, config: dict):
        evaluate_py = """
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

def run_evaluation(qa_pairs: list[dict]) -> dict:
    '''
    qa_pairs format:
    [{"question": "...", "ground_truth": "..."}]
    '''
    results = []
    for pair in qa_pairs:
        # response = rag_pipeline.query(pair["question"])
        results.append({
            "question": pair["question"],
            "answer": "answer", # response.answer,
            "contexts": ["context"], # response.source_documents,
            "ground_truth": pair["ground_truth"]
        })
    
    dataset = Dataset.from_list(results)
    scores = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
    )
    return scores
"""
        self._write_file(project_dir / "eval" / "evaluate.py", evaluate_py)
        self._write_file(project_dir / "eval" / "test_retrieval.py", "# Test retrieval quality\n")
        self._write_file(project_dir / "eval" / "test_generation.py", "# Test generation quality\n")
        self._write_file(project_dir / "eval" / "sample_questions.json", "[\n]")
        
        readme = """
# Evaluation Scaffold
Run `python evaluate.py` to evaluate your RAG pipeline using RAGAS.
"""
        self._write_file(project_dir / "eval" / "README.md", readme)

    def generate_requirements(self, project_dir: Path, config: dict):
        deps = ["fastapi", "uvicorn", "pydantic"]
        
        if config.get("framework") == "langchain":
            deps.append("langchain")
        elif config.get("framework") == "langgraph":
            deps.append("langgraph")
            deps.append("langchain")
            
        if config.get("search_type") == "sparse":
            deps.append("rank_bm25")
        elif config.get("search_type") == "hybrid":
            deps.append("rank_bm25")
            deps.append("langchain-community")
            
        if config.get("reranker") == "bge":
            deps.append("sentence_transformers")
        elif config.get("reranker") == "cohere":
            deps.append("cohere")
            
        if config.get("pii") == "presidio":
            deps.append("presidio-analyzer")
            deps.append("presidio-anonymizer")
            
        if config.get("cache") == "redis":
            deps.append("redis")
        elif config.get("cache") == "semantic":
            deps.append("redisvl")
            
        if config.get("infra_tier") in ["standard", "enterprise"]:
            if "redis" not in deps: deps.append("redis")
            deps.append("prometheus-client")
        if config.get("infra_tier") == "enterprise":
            deps.append("celery")
            deps.append("circuitbreaker")
            deps.append("slowapi")
            
        deps.append("ragas")
        deps.append("datasets")
        
        self._write_file(project_dir / "requirements.txt", "\\n".join(deps))

    def generate_env_example(self, project_dir: Path, config: dict):
        env = "ENVIRONMENT=development\\n"
        if config.get("reranker") == "cohere":
            env += "COHERE_API_KEY=your_key_here\\n"
        if config.get("infra_tier") in ["standard", "enterprise"] or config.get("cache") in ["redis", "semantic"]:
            env += "REDIS_URL=redis://localhost:6379/0\\n"
        self._write_file(project_dir / ".env.example", env)

    def generate_readme(self, project_dir: Path, config: dict):
        decisions = config.get("decisions", {})
        readme_content = f"""
# Generated by ShipAI

## Architecture Decisions
"""
        for component, decision in decisions.items():
            if isinstance(decision, dict):
                readme_content += f"""
### {component.replace('_', ' ').title()}
**Choice**: {decision.get('choice', 'N/A')}  
**Reason**: {decision.get('reason', 'N/A')}  
**Source**: {decision.get('book', 'N/A')}
"""
        self._write_file(project_dir / "README.md", readme_content)

def list_templates(tier: str = None) -> list[dict]:
    from app.services.template_registry import REGISTRY

    return REGISTRY.list_templates(tier=tier)

def get_template(template_id: str) -> dict | None:
    from app.services.template_registry import REGISTRY

    return REGISTRY.get_template(template_id)

async def generate_project(template_id: str, config: dict, output_dir: str | None = None, tier: str = "free") -> dict:
    from app.services.template_registry import REGISTRY

    if not REGISTRY.exists(template_id):
        return {"error": f"Template '{template_id}' not found"}
    config["template"] = template_id
    config["infra_tier"] = config.get("infra_tier") or (
        "minimal" if tier == "free" else "standard" if tier == "starter" else "enterprise"
    )
    engine = ShipAITemplateEngine()
    try:
        path = engine.generate_project(config, output_dir)
        
        # Count files generated
        files_generated = 0
        p = Path(path)
        if p.exists():
            files_generated = sum(1 for f in p.rglob("*") if f.is_file())
            
        # Get count of patterns included
        from app.services.infra_generator import get_infra_patterns
        infra_patterns = len(get_infra_patterns(tier))
            
        return {
            "status": "success", 
            "project_path": path, 
            "files_generated": files_generated,
            "infra_patterns_included": infra_patterns
        }
    except Exception as e:
        logger.error(f"Project generation failed: {e}")
        return {"error": str(e)}
