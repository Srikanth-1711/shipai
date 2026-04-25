"""
ShipAI — Template Engine
Generates complete AI project codebases from templates.
Each template = a production-ready AI project that users just configure and deploy.
"""
import json
import shutil
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from jinja2 import Environment, FileSystemLoader
from app.config import settings
from app.services.infra_generator import generate_infra_files

logger = logging.getLogger(__name__)

# Template registry — all available project templates
TEMPLATE_REGISTRY = {
    "rag_chatbot": {
        "name": "RAG Chatbot",
        "description": "Production-ready chatbot powered by RAG. Upload your documents, get a smart chatbot.",
        "category": "rag",
        "icon": "📚",
        "tier": "free",
        "min_hardware": "minimal",
        "estimated_setup_time": "5 minutes",
        "inputs": [
            {"name": "project_name", "type": "string", "required": True, "description": "Name for your chatbot project"},
            {"name": "documents_path", "type": "path", "required": True, "description": "Path to your documents (PDF, TXT, CSV)"},
            {"name": "system_prompt", "type": "text", "required": False, "default": "You are a helpful assistant.", "description": "Chatbot personality and instructions"},
            {"name": "model", "type": "model_select", "required": False, "default": "auto", "description": "LLM model to use (auto = best for your hardware)"},
            {"name": "chunk_size", "type": "integer", "required": False, "default": 512, "description": "Document chunk size for embeddings"},
            {"name": "top_k", "type": "integer", "required": False, "default": 5, "description": "Number of relevant chunks to retrieve"},
        ],
        "outputs": ["FastAPI backend", "Chat WebSocket", "Vector DB", "Document ingestion pipeline", "Production infra"],
        "tech_stack": ["FastAPI", "ChromaDB", "Ollama", "LangChain", "WebSocket"],
    },
    "multi_agent": {
        "name": "Multi-Agent System",
        "description": "Orchestrated multi-agent workflow. Define roles, tools, and goals — get a working agent system.",
        "category": "agent",
        "icon": "🤖",
        "tier": "starter",
        "min_hardware": "basic",
        "estimated_setup_time": "10 minutes",
        "inputs": [
            {"name": "project_name", "type": "string", "required": True, "description": "Name for your agent project"},
            {"name": "agents", "type": "agent_list", "required": True, "description": "List of agents with roles and tools"},
            {"name": "workflow", "type": "workflow", "required": False, "default": "sequential", "description": "Agent workflow type: sequential, parallel, hierarchical"},
            {"name": "model", "type": "model_select", "required": False, "default": "auto", "description": "LLM model for agents"},
        ],
        "outputs": ["Agent orchestrator", "Tool framework", "Monitoring dashboard", "API endpoints", "Production infra"],
        "tech_stack": ["FastAPI", "LangGraph", "Ollama", "WebSocket"],
    },
    "data_analyzer": {
        "name": "AI Data Analyzer",
        "description": "Upload CSV/data → get auto-generated insights, charts, and natural language queries.",
        "category": "ml",
        "icon": "📊",
        "tier": "free",
        "min_hardware": "minimal",
        "estimated_setup_time": "3 minutes",
        "inputs": [
            {"name": "project_name", "type": "string", "required": True, "description": "Name for your analyzer project"},
            {"name": "data_path", "type": "path", "required": True, "description": "Path to CSV or data files"},
            {"name": "analysis_goals", "type": "text", "required": False, "default": "General analysis", "description": "What insights are you looking for?"},
            {"name": "model", "type": "model_select", "required": False, "default": "auto", "description": "LLM for natural language queries"},
        ],
        "outputs": ["Analysis API", "Auto-generated charts", "NL query interface", "Insights dashboard", "Production infra"],
        "tech_stack": ["FastAPI", "Pandas", "Plotly", "Ollama"],
    },
}


def list_templates(tier: str = None) -> list[dict]:
    """List available templates, optionally filtered by tier."""
    templates = []
    tier_order = ["free", "starter", "pro", "enterprise"]

    for key, tmpl in TEMPLATE_REGISTRY.items():
        if tier and tier_order.index(tmpl["tier"]) > tier_order.index(tier):
            continue  # Skip templates above user's tier
        templates.append({
            "id": key,
            **tmpl,
        })
    return templates


def get_template(template_id: str) -> Optional[dict]:
    """Get a specific template's full details."""
    tmpl = TEMPLATE_REGISTRY.get(template_id)
    if tmpl:
        return {"id": template_id, **tmpl}
    return None


async def generate_project(
    template_id: str,
    config: dict,
    output_dir: str = None,
    tier: str = "free",
) -> dict:
    """
    Generate a complete project from a template + user config.
    This is the core of ShipAI — turns a template into a deployable project.
    """
    template = TEMPLATE_REGISTRY.get(template_id)
    if not template:
        return {"error": f"Template '{template_id}' not found"}

    # Determine output directory
    project_name = config.get("project_name", template_id)
    if not output_dir:
        output_dir = str(Path(settings.PROJECTS_DIR) / project_name)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_files = []

    try:
        # 1. Generate project scaffold based on template type
        if template_id == "rag_chatbot":
            generated_files.extend(_generate_rag_chatbot(output_path, config))
        elif template_id == "multi_agent":
            generated_files.extend(_generate_multi_agent(output_path, config))
        elif template_id == "data_analyzer":
            generated_files.extend(_generate_data_analyzer(output_path, config))

        # 2. Inject production infrastructure (7 patterns)
        infra_files = generate_infra_files(output_dir, tier=tier)
        generated_files.extend(infra_files)

        # 3. Generate project metadata
        metadata = {
            "project_name": project_name,
            "template": template_id,
            "template_name": template["name"],
            "created_at": datetime.now().isoformat(),
            "tier": tier,
            "config": config,
            "shipai_version": settings.APP_VERSION,
        }
        meta_path = output_path / "shipai.json"
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        generated_files.append(str(meta_path))

        # 4. Generate README
        readme = _generate_readme(template, config, project_name)
        readme_path = output_path / "README.md"
        readme_path.write_text(readme, encoding="utf-8")
        generated_files.append(str(readme_path))

        # 5. Generate requirements.txt
        reqs = _generate_requirements(template_id)
        reqs_path = output_path / "requirements.txt"
        reqs_path.write_text(reqs, encoding="utf-8")
        generated_files.append(str(reqs_path))

        # 6. Generate Dockerfile
        dockerfile = _generate_dockerfile(project_name)
        docker_path = output_path / "Dockerfile"
        docker_path.write_text(dockerfile, encoding="utf-8")
        generated_files.append(str(docker_path))

        logger.info(f"✅ Generated project '{project_name}' ({len(generated_files)} files) at {output_dir}")

        return {
            "status": "success",
            "project_name": project_name,
            "template": template["name"],
            "output_dir": output_dir,
            "files_generated": len(generated_files),
            "files": generated_files,
            "infra_patterns_included": len(infra_files),
            "next_steps": [
                f"cd {output_dir}",
                "pip install -r requirements.txt",
                "python main.py",
            ],
        }
    except Exception as e:
        logger.error(f"Project generation failed: {e}")
        return {"error": str(e)}


def _generate_rag_chatbot(output_path: Path, config: dict) -> list[str]:
    """Generate a RAG chatbot project."""
    files = []

    # main.py
    main_code = f'''"""
{config.get("project_name", "RAG Chatbot")} — Powered by ShipAI
Production-ready RAG chatbot with vector search and conversation memory.
"""
import os
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import chromadb
from chromadb.config import Settings as ChromaSettings

app = FastAPI(title="{config.get("project_name", "RAG Chatbot")}")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Ollama config
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("MODEL", "{config.get("model", "tinyllama")}")
SYSTEM_PROMPT = """{config.get("system_prompt", "You are a helpful assistant. Answer questions based on the provided context.")}"""

# Vector DB
chroma_client = chromadb.Client(ChromaSettings(anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection("{config.get("project_name", "docs").replace(" ", "_").lower()}")

TOP_K = {config.get("top_k", 5)}


class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K


class IngestRequest(BaseModel):
    text: str
    metadata: dict = {{}}


@app.get("/")
async def root():
    return {{"name": "{config.get("project_name", "RAG Chatbot")}", "status": "running", "docs": "/docs"}}


@app.post("/ingest")
async def ingest_document(req: IngestRequest):
    """Add a document to the knowledge base."""
    doc_id = f"doc_{{collection.count()}}"
    collection.add(documents=[req.text], ids=[doc_id], metadatas=[req.metadata])
    return {{"status": "ingested", "doc_id": doc_id, "total_docs": collection.count()}}


@app.post("/query")
async def query(req: QueryRequest):
    """Ask a question — retrieves relevant docs and generates answer."""
    # Retrieve relevant documents
    results = collection.query(query_texts=[req.question], n_results=req.top_k)
    context = "\\n\\n".join(results["documents"][0]) if results["documents"][0] else "No relevant documents found."

    # Generate answer with context
    prompt = f"""Context:\\n{{context}}\\n\\nQuestion: {{req.question}}\\n\\nAnswer based on the context above:"""

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{{OLLAMA_URL}}/api/generate",
            json={{"model": MODEL, "prompt": prompt, "system": SYSTEM_PROMPT, "stream": False}},
            timeout=120,
        )
        data = resp.json()

    return {{
        "answer": data.get("response", ""),
        "sources": results["documents"][0][:3],
        "model": MODEL,
    }}


@app.get("/stats")
async def stats():
    return {{"total_documents": collection.count(), "model": MODEL}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
'''
    main_path = output_path / "main.py"
    main_path.write_text(main_code, encoding="utf-8")
    files.append(str(main_path))

    return files


def _generate_multi_agent(output_path: Path, config: dict) -> list[str]:
    """Generate a multi-agent system project."""
    files = []

    main_code = f'''"""
{config.get("project_name", "Multi-Agent System")} — Powered by ShipAI
Orchestrated multi-agent workflow with monitoring.
"""
import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
import httpx

app = FastAPI(title="{config.get("project_name", "Multi-Agent System")}")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("MODEL", "{config.get("model", "tinyllama")}")


class AgentTask(BaseModel):
    task: str
    agents: list[str] = ["researcher", "analyzer", "writer"]


async def run_agent(name: str, role: str, task: str, context: str = "") -> str:
    """Run a single agent with its role."""
    prompt = f"""You are the {{name}} agent. Your role: {{role}}
    
Task: {{task}}
Previous context: {{context}}

Provide your output:"""

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{{OLLAMA_URL}}/api/generate",
            json={{"model": MODEL, "prompt": prompt, "stream": False}},
            timeout=120,
        )
    return resp.json().get("response", "")


AGENT_ROLES = {{
    "researcher": "Research and gather information about the topic",
    "analyzer": "Analyze the research findings and identify key insights",
    "writer": "Write a clear, concise summary based on the analysis",
}}


@app.post("/run")
async def run_workflow(req: AgentTask):
    """Execute multi-agent workflow."""
    results = []
    context = ""

    for agent_name in req.agents:
        role = AGENT_ROLES.get(agent_name, f"Handle the {{agent_name}} part of the task")
        output = await run_agent(agent_name, role, req.task, context)
        results.append({{"agent": agent_name, "output": output}})
        context += f"\\n[{{agent_name}}]: {{output}}"

    return {{"task": req.task, "results": results, "final_output": results[-1]["output"] if results else ""}}


@app.get("/")
async def root():
    return {{"name": "{config.get("project_name", "Multi-Agent System")}", "agents": list(AGENT_ROLES.keys())}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
'''
    main_path = output_path / "main.py"
    main_path.write_text(main_code, encoding="utf-8")
    files.append(str(main_path))
    return files


def _generate_data_analyzer(output_path: Path, config: dict) -> list[str]:
    """Generate an AI data analyzer project."""
    files = []

    main_code = f'''"""
{config.get("project_name", "AI Data Analyzer")} — Powered by ShipAI
Upload CSV data → get auto-generated insights + natural language queries.
"""
import os
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import httpx
import json

app = FastAPI(title="{config.get("project_name", "AI Data Analyzer")}")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("MODEL", "{config.get("model", "tinyllama")}")

# In-memory data store
datasets: dict[str, pd.DataFrame] = {{}}


class NLQuery(BaseModel):
    dataset: str
    question: str


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV file for analysis."""
    df = pd.read_csv(file.file)
    name = file.filename.replace(".csv", "")
    datasets[name] = df

    summary = {{
        "name": name,
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {{col: str(dtype) for col, dtype in df.dtypes.items()}},
        "sample": df.head(3).to_dict(),
        "stats": json.loads(df.describe().to_json()),
    }}
    return summary


@app.post("/query")
async def nl_query(req: NLQuery):
    """Ask a natural language question about your data."""
    df = datasets.get(req.dataset)
    if df is None:
        return {{"error": f"Dataset '{{req.dataset}}' not found"}}

    context = f"Dataset: {{req.dataset}}\\nColumns: {{list(df.columns)}}\\nRows: {{len(df)}}\\nSample:\\n{{df.head(5).to_string()}}"
    prompt = f"Data context:\\n{{context}}\\n\\nQuestion: {{req.question}}\\n\\nAnalyze the data and answer:"

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{{OLLAMA_URL}}/api/generate",
            json={{"model": MODEL, "prompt": prompt, "stream": False}},
            timeout=120,
        )
    return {{"answer": resp.json().get("response", ""), "dataset": req.dataset}}


@app.get("/datasets")
async def list_datasets():
    return {{"datasets": [{{
        "name": name, "rows": len(df), "columns": list(df.columns)
    }} for name, df in datasets.items()]}}


@app.get("/")
async def root():
    return {{"name": "{config.get("project_name", "AI Data Analyzer")}", "datasets_loaded": len(datasets)}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
'''
    main_path = output_path / "main.py"
    main_path.write_text(main_code, encoding="utf-8")
    files.append(str(main_path))
    return files


def _generate_readme(template: dict, config: dict, project_name: str) -> str:
    return f"""# {project_name}

> Built with [ShipAI](https://shipai.dev) — The AI Engineer You Can Hire at Scale

## About
{template['description']}

## Quick Start
```bash
pip install -r requirements.txt
python main.py
```

## Tech Stack
{', '.join(template.get('tech_stack', []))}

## API Docs
Once running, visit `http://localhost:8001/docs` for interactive API documentation.

## Production Infrastructure Included
- ✅ Rate Limiting (token bucket)
- ✅ Caching (embedding + LLM response cache)
- ✅ API Gateway (request tracing, CORS)
- ✅ Circuit Breaker (auto-recovery)
- ✅ Message Queue (async inference)

---
*Generated by ShipAI v{settings.APP_VERSION}*
"""


def _generate_requirements(template_id: str) -> str:
    base = "fastapi==0.115.12\nuvicorn==0.34.2\nhttpx==0.28.1\npython-dotenv==1.1.0\n"
    if template_id == "rag_chatbot":
        base += "chromadb==1.0.7\nlangchain==0.3.25\n"
    elif template_id == "data_analyzer":
        base += "pandas==2.2.3\nplotly==6.1.2\n"
    return base


def _generate_dockerfile(project_name: str) -> str:
    return f"""FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8001
CMD ["python", "main.py"]
"""
