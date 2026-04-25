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
    "pro": ["rate_limiting", "caching", "api_gateway", "circuit_breaker", "message_queue", "load_balancing", "auto_scaling"],
    "enterprise": ["rate_limiting", "caching", "api_gateway", "circuit_breaker", "message_queue", "load_balancing", "auto_scaling"],
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
        "description": "Async job queue -- decouples API requests from LLM inference workers",
        "files": {
            "middleware/queue.py": '''"""Message Queue -- Async job processing for LLM inference"""
import asyncio
import uuid
import time
import logging
from enum import Enum
from typing import Any, Callable, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    task_type: str
    payload: dict
    status: JobStatus = JobStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class InMemoryQueue:
    """
    Simple async message queue for LLM inference.
    For production, swap with Celery + Redis.
    """

    def __init__(self, max_workers: int = 2, max_queue_size: int = 100):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.jobs: dict[str, Job] = {}
        self.handlers: dict[str, Callable] = {}
        self._workers: list[asyncio.Task] = []

    def register_handler(self, task_type: str, handler: Callable):
        """Register a handler for a task type."""
        self.handlers[task_type] = handler

    async def enqueue(self, task_type: str, payload: dict) -> str:
        """Add a job to the queue. Returns job ID."""
        job = Job(id=str(uuid.uuid4())[:8], task_type=task_type, payload=payload)
        self.jobs[job.id] = job

        try:
            self.queue.put_nowait(job)
            logger.info(f"Job {job.id} enqueued ({task_type})")
        except asyncio.QueueFull:
            job.status = JobStatus.FAILED
            job.error = "Queue is full. Please try again later."
            logger.warning(f"Queue full -- job {job.id} rejected")

        return job.id

    def get_job_status(self, job_id: str) -> Optional[dict]:
        """Get the status of a job."""
        job = self.jobs.get(job_id)
        if not job:
            return None
        return {
            "id": job.id,
            "status": job.status.value,
            "task_type": job.task_type,
            "result": job.result,
            "error": job.error,
            "duration_ms": (
                round((job.completed_at - job.started_at) * 1000, 2)
                if job.completed_at and job.started_at else None
            ),
        }

    async def _worker(self, worker_id: int):
        """Background worker that processes jobs."""
        logger.info(f"Worker {worker_id} started")
        while True:
            try:
                job = await self.queue.get()
                job.status = JobStatus.PROCESSING
                job.started_at = time.time()

                handler = self.handlers.get(job.task_type)
                if not handler:
                    job.status = JobStatus.FAILED
                    job.error = f"No handler for task type: {job.task_type}"
                else:
                    try:
                        job.result = await handler(job.payload)
                        job.status = JobStatus.COMPLETED
                    except Exception as e:
                        job.status = JobStatus.FAILED
                        job.error = str(e)

                job.completed_at = time.time()
                self.queue.task_done()
                logger.info(f"Worker {worker_id}: Job {job.id} -> {job.status.value}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    async def start_workers(self):
        """Start background workers."""
        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker(i))
            self._workers.append(task)

    async def stop_workers(self):
        """Stop all workers."""
        for task in self._workers:
            task.cancel()
        self._workers.clear()

    def stats(self) -> dict:
        return {
            "queue_size": self.queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "active_workers": len(self._workers),
            "total_jobs": len(self.jobs),
            "pending": sum(1 for j in self.jobs.values() if j.status == JobStatus.PENDING),
            "processing": sum(1 for j in self.jobs.values() if j.status == JobStatus.PROCESSING),
            "completed": sum(1 for j in self.jobs.values() if j.status == JobStatus.COMPLETED),
            "failed": sum(1 for j in self.jobs.values() if j.status == JobStatus.FAILED),
        }


# Global queue instance
job_queue = InMemoryQueue(max_workers=2, max_queue_size=100)
''',
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
