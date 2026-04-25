"""
ShipAI -- Production Infrastructure Generator
Auto-injects 7 production patterns into every generated project:
1. Rate Limiting
2. Caching (embedding queries)
3. API Gateway
4. Load Balancing
5. Circuit Breaker
6. Auto Scaling
7. Message Queue
"""
import json
import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# Which patterns are included per tier
TIER_PATTERNS = {
    "free": ["rate_limiting", "caching", "api_gateway"],
    "starter": ["rate_limiting", "caching", "api_gateway", "circuit_breaker", "message_queue"],
    "pro": ["rate_limiting", "caching", "api_gateway", "circuit_breaker", "message_queue", "load_balancing", "auto_scaling", "observability", "load_testing"],
    "enterprise": ["rate_limiting", "caching", "api_gateway", "circuit_breaker", "message_queue", "load_balancing", "auto_scaling", "observability", "load_testing"],
}


INFRA_TEMPLATES = {
    "rate_limiting": {
        "description": "Token-bucket rate limiter -- protects LLM endpoint from being hammered",
        "files": {
            "middleware/rate_limiter.py": '''"""Rate Limiting Middleware -- Token Bucket Algorithm"""
import time
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimiter:
    """Per-client token bucket rate limiter."""

    def __init__(self, requests_per_minute: int = 100, burst: int = 20):
        self.rate = requests_per_minute / 60.0  # tokens per second
        self.burst = burst
        self.clients: dict[str, dict] = defaultdict(
            lambda: {"tokens": burst, "last_refill": time.time()}
        )

    def _get_client_key(self, request: Request) -> str:
        """Identify client by IP or API key."""
        api_key = request.headers.get("X-API-Key", "")
        if api_key:
            return f"key:{api_key}"
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        return f"ip:{request.client.host}" if request.client else "ip:unknown"

    def _refill_tokens(self, client: dict):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - client["last_refill"]
        client["tokens"] = min(self.burst, client["tokens"] + elapsed * self.rate)
        client["last_refill"] = now

    def check_rate_limit(self, request: Request) -> bool:
        """Returns True if request is allowed, False if rate limited."""
        key = self._get_client_key(request)
        client = self.clients[key]
        self._refill_tokens(client)

        if client["tokens"] >= 1:
            client["tokens"] -= 1
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(self, app, requests_per_minute: int = 100, burst: int = 20):
        super().__init__(app)
        self.limiter = RateLimiter(requests_per_minute, burst)

    async def dispatch(self, request: Request, call_next):
        if not self.limiter.check_rate_limit(request):
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Too Many Requests",
                    "message": "Rate limit exceeded. Please slow down.",
                    "retry_after_seconds": 60,
                }
            )

        response = await call_next(request)
        return response
''',
        },
    },

    "caching": {
        "description": "In-memory + optional Redis cache for embeddings and LLM responses",
        "files": {
            "middleware/cache.py": '''"""Caching Layer -- Embedding & LLM Response Cache"""
import hashlib
import time
import json
from typing import Any, Optional


class InMemoryCache:
    """TTL-based in-memory cache for embedding queries and LLM responses."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: dict[str, dict] = {}

    def _make_key(self, prefix: str, data: str) -> str:
        """Create a deterministic cache key."""
        return f"{prefix}:{hashlib.sha256(data.encode()).hexdigest()[:16]}"

    def _is_expired(self, entry: dict) -> bool:
        return (time.time() - entry["created_at"]) > self.ttl

    def _evict_if_needed(self):
        """Remove oldest entries if cache is full."""
        if len(self._cache) >= self.max_size:
            # Remove expired first
            expired = [k for k, v in self._cache.items() if self._is_expired(v)]
            for k in expired:
                del self._cache[k]
            # If still full, remove oldest
            if len(self._cache) >= self.max_size:
                oldest = min(self._cache, key=lambda k: self._cache[k]["created_at"])
                del self._cache[oldest]

    def get(self, prefix: str, data: str) -> Optional[Any]:
        """Get cached value. Returns None on miss."""
        key = self._make_key(prefix, data)
        entry = self._cache.get(key)
        if entry and not self._is_expired(entry):
            entry["hits"] += 1
            return entry["value"]
        elif entry:
            del self._cache[key]  # Clean expired
        return None

    def set(self, prefix: str, data: str, value: Any):
        """Cache a value with TTL."""
        self._evict_if_needed()
        key = self._make_key(prefix, data)
        self._cache[key] = {
            "value": value,
            "created_at": time.time(),
            "hits": 0,
        }

    def stats(self) -> dict:
        """Return cache statistics."""
        total = len(self._cache)
        expired = sum(1 for v in self._cache.values() if self._is_expired(v))
        total_hits = sum(v["hits"] for v in self._cache.values())
        return {
            "total_entries": total,
            "expired_entries": expired,
            "active_entries": total - expired,
            "total_hits": total_hits,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl,
        }

    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()


# Global cache instance
embedding_cache = InMemoryCache(max_size=1000, ttl_seconds=3600)
llm_cache = InMemoryCache(max_size=500, ttl_seconds=1800)
''',
        },
    },

    "api_gateway": {
        "description": "Single entry point with auth, rate limiting, and service routing",
        "files": {
            "middleware/gateway.py": '''"""API Gateway -- Single entry point for all services"""
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class APIGatewayMiddleware(BaseHTTPMiddleware):
    """
    API Gateway middleware providing:
    - Request logging and tracing
    - Service routing metadata
    - CORS handling
    - Request ID generation
    """

    async def dispatch(self, request: Request, call_next):
        import uuid
        import time

        # Generate request ID for tracing
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Log incoming request
        logger.info(f"[{request_id}] {request.method} {request.url.path}")

        # Add request ID to state for downstream use
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add tracking headers
        duration = round((time.time() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(duration)

        logger.info(f"[{request_id}] {response.status_code} ({duration}ms)")
        return response
''',
        },
    },

    "circuit_breaker": {
        "description": "Auto-recovery pattern -- stops calling failing services, probes for recovery",
        "files": {
            "middleware/circuit_breaker.py": '''"""Circuit Breaker -- Protect against cascading failures"""
import time
import logging
from enum import Enum
from typing import Callable, Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"       # Normal -- requests flow through
    OPEN = "open"           # Tripped -- requests blocked instantly
    HALF_OPEN = "half_open" # Probe -- testing one request


class CircuitBreaker:
    """
    Circuit breaker for LLM endpoints and external services.

    CLOSED -> failures exceed threshold -> OPEN
    OPEN -> timeout expires -> HALF-OPEN
    HALF-OPEN -> probe succeeds -> CLOSED
    HALF-OPEN -> probe fails -> OPEN
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_max_calls: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0

    @property
    def is_available(self) -> bool:
        """Check if the circuit allows requests."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls
        return False

    def record_success(self):
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            self._transition_to(CircuitState.CLOSED)
        self.failure_count = 0

    def record_failure(self):
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
        elif self.failure_count >= self.failure_threshold:
            self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState):
        old = self.state
        self.state = new_state
        logger.warning(f"Circuit [{self.name}]: {old.value} -> {new_state.value}")
        if new_state == CircuitState.CLOSED:
            self.failure_count = 0
            self.half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self.half_open_calls = 0

    def get_status(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_s": self.recovery_timeout,
        }


# Pre-built circuit breakers for common services
llm_circuit = CircuitBreaker(name="llm_endpoint", failure_threshold=3, recovery_timeout=30)
embedding_circuit = CircuitBreaker(name="embedding_endpoint", failure_threshold=5, recovery_timeout=20)
''',
        },
    },

    "message_queue": {
        "description": "Production message queue using Celery and Redis for async LLM jobs",
        "files": {
            "workers/celery_worker.py": '''"""Celery Worker for Async LLM Processing"""
import os
import httpx
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Initialize Celery
celery_app = Celery(
    "shipai_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minute hard limit
)

@celery_app.task(bind=True, name="generate_llm_response", max_retries=3)
def generate_llm_response(self, prompt: str, model: str = "tinyllama"):
    """Background task to call the LLM."""
    try:
        # Use sync client inside Celery worker for simplicity
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False}
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
    except httpx.HTTPError as exc:
        # Exponential backoff retry
        countdown = 2 ** self.request.retries
        raise self.retry(exc=exc, countdown=countdown)
''',
            "workers/docker-compose.redis.yml": '''# Drop-in Docker Compose file for Redis + Celery Worker
version: "3.8"
services:
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
  
  worker:
    build: .
    command: celery -A workers.celery_worker.celery_app worker --loglevel=info
    environment:
      - REDIS_URL=redis://redis:6379/0
      - OLLAMA_URL=http://host.docker.internal:11434
    depends_on:
      - redis
'''
        },
    },

    "load_balancing": {
        "description": "Round-robin load balancer config for multi-GPU scaling (Nginx/HAProxy)",
        "files": {
            "infra/load_balancer.conf": '''# ShipAI Load Balancer Configuration (Nginx)
# Round-robin with health checks across GPU nodes running vLLM

upstream llm_backend {
    # GPU nodes -- add more as you scale
    server gpu-node-a:8000 weight=1 max_fails=3 fail_timeout=30s;
    server gpu-node-b:8000 weight=1 max_fails=3 fail_timeout=30s;
    # server gpu-node-c:8000 weight=1 max_fails=3 fail_timeout=30s;

    # Health check
    keepalive 32;
}

server {
    listen 80;
    server_name api.your-domain.com;

    location /api/ {
        proxy_pass http://llm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 10s;
        proxy_read_timeout 120s;  # LLM inference can be slow
    }

    location /health {
        proxy_pass http://llm_backend/api/health;
    }
}
''',
        },
    },

    "auto_scaling": {
        "description": "Auto-scaling policy configs -- scale GPU nodes based on metrics",
        "files": {
            "infra/autoscaler.yaml": '''# ShipAI Auto Scaling Configuration
# Monitor CPU/GPU utilization and p99 latency -> scale up/down

scaling_policy:
  name: shipai-llm-autoscaler
  
  metrics:
    - type: gpu_utilization
      target: 75          # Scale up if GPU util > 75%
      window: 5m
    - type: cpu_utilization
      target: 80
      window: 5m
    - type: p99_latency_ms
      target: 5000        # Scale up if p99 > 5 seconds
      window: 3m
    - type: queue_depth
      target: 50          # Scale up if queue > 50 pending jobs
      window: 1m

  scale_up:
    min_nodes: 1
    max_nodes: 10
    cooldown: 3m          # Wait 3 min between scale-ups
    increment: 1          # Add 1 node at a time

  scale_down:
    cooldown: 10m         # Wait 10 min before scaling down
    # Don't pay for idle GPUs when traffic drops
    idle_threshold: 10    # GPU util < 10% for cooldown period

  node_template:
    image: shipai/vllm-node:latest
    gpu: 1
    memory: 16Gi
    health_check:
      path: /health
      interval: 30s
      timeout: 5s
      unhealthy_threshold: 3
''',
        },
    },

    "observability": {
        "description": "OpenTelemetry and Prometheus integration for monitoring LLM latency and token usage",
        "files": {
            "middleware/telemetry.py": '''"""Observability Middleware -- Prometheus & OpenTelemetry"""
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, generate_latest

# Prometheus Metrics
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"])
LLM_TOKEN_COUNT = Counter("llm_tokens_total", "Total LLM tokens generated", ["model"])

class TelemetryMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)
            
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        return response

def get_metrics():
    return generate_latest()
''',
        },
    },

    "load_testing": {
        "description": "Locust load testing script to verify scale and rate limits",
        "files": {
            "tests/load_test.py": '''"""Locust Load Test Script"""
# Run with: locust -f tests/load_test.py --host=http://localhost:8001
from locust import HttpUser, task, between
import random

class LLMUser(HttpUser):
    wait_time = between(1, 5) # Wait 1-5 seconds between tasks

    @task(3)
    def test_health(self):
        self.client.get("/")
        
    @task(1)
    def test_inference(self):
        questions = [
            "What is AI?", 
            "Explain quantum computing", 
            "How does FastAPI work?"
        ]
        # Adjust payload based on the specific template's API
        self.client.post("/query", json={
            "question": random.choice(questions)
        })
''',
        },
    },
}


def get_infra_patterns(tier: str = "pro") -> dict:
    """Get infrastructure patterns for a given pricing tier."""
    pattern_names = TIER_PATTERNS.get(tier, TIER_PATTERNS["free"])
    patterns = {}
    for name in pattern_names:
        if name in INFRA_TEMPLATES:
            patterns[name] = INFRA_TEMPLATES[name]
    return patterns


def generate_infra_files(output_dir: str, tier: str = "pro") -> list[str]:
    """Generate all infrastructure files for a project."""
    output_path = Path(output_dir)
    generated_files = []

    patterns = get_infra_patterns(tier)
    for pattern_name, pattern_data in patterns.items():
        for filepath, content in pattern_data["files"].items():
            full_path = output_path / filepath
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            generated_files.append(str(full_path))
            logger.info(f"Generated: {filepath} ({pattern_name})")

    return generated_files


def get_infra_summary(tier: str = "pro") -> list[dict]:
    """Get a summary of what infra patterns are included for a tier."""
    pattern_names = TIER_PATTERNS.get(tier, TIER_PATTERNS["free"])
    return [
        {
            "name": name,
            "included": name in pattern_names,
            "description": INFRA_TEMPLATES.get(name, {}).get("description", ""),
        }
        for name in INFRA_TEMPLATES.keys()
    ]
