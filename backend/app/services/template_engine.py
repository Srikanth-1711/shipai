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
            {"name": "llm_provider", "type": "string", "required": False, "default": "ollama", "description": "LLM Provider: ollama, openai, or anthropic"},
            {"name": "retrieval_mode", "type": "string", "required": False, "default": "vector_chroma", "description": "Retrieval mode: vector_chroma, vector_pinecone, vectorless_bm25, or hybrid"},
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
            {"name": "llm_provider", "type": "string", "required": False, "default": "ollama", "description": "LLM Provider: ollama, openai, or anthropic"},
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
    "llm_finetuner": {
        "name": "LLM Fine-Tuner",
        "description": "Production pipeline for PEFT/LoRA fine-tuning on local hardware. Prepare data, train, and deploy.",
        "category": "fine_tuning",
        "icon": "🧠",
        "tier": "pro",
        "min_hardware": "high",
        "estimated_setup_time": "15 minutes",
        "inputs": [
            {"name": "project_name", "type": "string", "required": True, "description": "Name for your fine-tuning project"},
            {"name": "dataset_path", "type": "path", "required": True, "description": "Path to JSONL training data"},
            {"name": "base_model", "type": "string", "required": False, "default": "unsloth/llama-3-8b-bnb-4bit", "description": "HuggingFace base model"},
            {"name": "target_modules", "type": "string", "required": False, "default": "q_proj,k_proj,v_proj,o_proj", "description": "LoRA target modules"},
        ],
        "outputs": ["Data prep pipeline", "Unsloth training script", "Model export to Ollama/GGUF", "Inference API", "Production infra"],
        "tech_stack": ["Unsloth", "PyTorch", "Transformers", "FastAPI"],
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
        elif template_id == "llm_finetuner":
            generated_files.extend(_generate_llm_finetuner(output_path, config))

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
        reqs = _generate_requirements(template_id, tier, config)
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
    
    retrieval_mode = config.get("retrieval_mode", "vector_chroma")
    imports = "import chromadb\nfrom chromadb.config import Settings as ChromaSettings"
    db_setup = f'''chroma_client = chromadb.Client(ChromaSettings(anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection("{config.get("project_name", "docs").replace(" ", "_").lower()}")'''
    query_logic = '''    results = collection.query(query_texts=[req.question], n_results=req.top_k)
    context = "\\n\\n".join(results["documents"][0]) if results["documents"][0] else "No relevant documents found."
    sources = results["documents"][0][:3] if results["documents"][0] else []'''

    if retrieval_mode == "vector_pinecone":
        imports = "from pinecone import Pinecone"
        db_setup = f'''pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", "your-api-key"))
index = pc.Index("{config.get("project_name", "docs").replace(" ", "_").lower()[:45]}")'''
        query_logic = '''    # Pinecone requires actual embeddings generation here, stubbed for template
    results = index.query(vector=[0.0]*768, top_k=req.top_k, include_metadata=True)
    context = "\\n\\n".join([match.metadata.get('text', '') for match in results.matches]) if results.matches else "No relevant documents found."
    sources = [match.metadata.get('text', '') for match in results.matches][:3]'''
    elif retrieval_mode == "vectorless_bm25":
        imports = "from rank_bm25 import BM25Okapi\nimport numpy as np"
        db_setup = '''bm25_corpus = []
bm25_index = None'''
        query_logic = '''    global bm25_index, bm25_corpus
    if not bm25_index:
        context = "No documents indexed."
        sources = []
    else:
        tokenized_query = req.question.split(" ")
        doc_scores = bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[::-1][:req.top_k]
        sources = [bm25_corpus[i] for i in top_indices if doc_scores[i] > 0]
        context = "\\n\\n".join(sources) if sources else "No relevant documents found."'''

    llm_provider = config.get("llm_provider", "ollama")
    
    if llm_provider == "openai":
        imports += "\nfrom openai import AsyncOpenAI"
        llm_config = '''# OpenAI config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key")
MODEL = os.getenv("MODEL", "gpt-4o-mini")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)'''
        generate_logic = '''    prompt = f"""Context:\\n{context}\\n\\nQuestion: {req.question}\\n\\nAnswer based on the context above:"""
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    answer = resp.choices[0].message.content'''
    elif llm_provider == "anthropic":
        imports += "\nfrom anthropic import AsyncAnthropic"
        llm_config = '''# Anthropic config
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "your-api-key")
MODEL = os.getenv("MODEL", "claude-3-haiku-20240307")
client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)'''
        generate_logic = '''    prompt = f"""Context:\\n{context}\\n\\nQuestion: {req.question}\\n\\nAnswer based on the context above:"""
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    answer = resp.content[0].text'''
    else:
        # Default Ollama
        llm_config = f'''# Ollama config
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("MODEL", "{config.get("model", "tinyllama")}")'''
        generate_logic = '''    prompt = f"""Context:\\n{context}\\n\\nQuestion: {req.question}\\n\\nAnswer based on the context above:"""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL, "prompt": prompt, "system": SYSTEM_PROMPT, "stream": False},
            timeout=120,
        )
        answer = resp.json().get("response", "")'''

    # main.py
    main_code = f'''"""
{config.get("project_name", "RAG Chatbot")} — Powered by ShipAI
Production-ready RAG chatbot with conversation memory.
Retrieval Mode: {retrieval_mode}
"""
import os
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
{imports}

app = FastAPI(title="{config.get("project_name", "RAG Chatbot")}")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

{llm_config}
SYSTEM_PROMPT = """{config.get("system_prompt", "You are a helpful assistant. Answer questions based on the provided context.")}"""

# Database Setup
{db_setup}

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
async def ingest(req: IngestRequest):
    """Ingest text into the knowledge base."""
    # 1. Simple Chunking
    chunks = [req.text[i:i+500] for i in range(0, len(req.text), 450)]
    
    # 2. Add to Knowledge Base
    ids = [f"doc_{{i}}_{{os.urandom(4).hex()}}" for i in range(len(chunks))]
    
    if "{retrieval_mode}" == "vector_chroma":
        collection.add(documents=chunks, ids=ids)
    elif "{retrieval_mode}" == "vectorless_bm25":
        global bm25_index, bm25_corpus
        bm25_corpus.extend(chunks)
        from rank_bm25 import BM25Okapi
        bm25_index = BM25Okapi([doc.split(" ") for doc in bm25_corpus])
    
    return {{"status": "success", "chunks_processed": len(chunks)}}

@app.post("/query")
async def query(req: QueryRequest):
    """Ask a question — retrieves relevant docs and generates answer."""
    # Retrieve relevant documents
{query_logic}

    # Generate answer with context
{generate_logic}

    return {{
        "answer": answer,
        "sources": sources,
        "model": MODEL,
    }}

@app.get("/evaluate")
async def evaluate():
    """Stub for RAG evaluation metrics (Precision/Recall/Hit-Rate)."""
    return {{
        "retrieval_precision": "0.85 (simulated)",
        "faithfulness": "0.92 (simulated)",
        "hit_rate_at_k": "0.78 (simulated)",
        "latency_ms": 120
    }}

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

    llm_provider = config.get("llm_provider", "ollama")
    if llm_provider == "openai":
        llm_init = 'from langchain_openai import ChatOpenAI\nllm = ChatOpenAI(model="gpt-4o-mini", temperature=0)'
    elif llm_provider == "anthropic":
        llm_init = 'from langchain_anthropic import ChatAnthropic\nllm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)'
    else:
        llm_init = f'from langchain_community.chat_models import ChatOllama\nllm = ChatOllama(model="{config.get("model", "llama3")}", temperature=0)'

    # 1. Generate mcp_client.py
    mcp_code = '''"""Model Context Protocol (MCP) Client Stub."""
import asyncio

class MCPClient:
    """Connects to external MCP servers to fetch dynamic tool schemas and context."""
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.connected = False
        
    async def connect(self):
        self.connected = True
        return {"status": "connected", "tools_available": ["read_confluence", "query_jira"]}
        
    async def call_tool(self, tool_name: str, args: dict):
        if not self.connected: raise Exception("MCP Not connected")
        return f"Simulated output from MCP tool {tool_name}"
'''
    mcp_path = output_path / "mcp_client.py"
    mcp_path.write_text(mcp_code, encoding="utf-8")
    files.append(str(mcp_path))

    # 2. Generate tools.py
    tools_code = '''"""Local Python Tools for Agents."""
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Simulated search results for: {query}"

@tool
def calculate_math(expression: str) -> str:
    """Calculate basic math expressions."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error calculating: {e}"

AVAILABLE_TOOLS = [web_search, calculate_math]
'''
    tools_path = output_path / "tools.py"
    tools_path.write_text(tools_code, encoding="utf-8")
    files.append(str(tools_path))

    # 3. Generate main.py
    main_code = f'''"""
{config.get("project_name", "Multi-Agent System")} — Powered by ShipAI
Orchestrated LangGraph workflow with MCP capabilities.
"""
import os
from typing import Annotated, Sequence, TypedDict
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Tools and MCP
from tools import AVAILABLE_TOOLS
from mcp_client import MCPClient

{llm_init}
llm_with_tools = llm.bind_tools(AVAILABLE_TOOLS)

app = FastAPI(title="{config.get("project_name", "Multi-Agent System")}")

# Graph State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "messages"]

# Node logic
def agent_node(state: AgentState):
    """The central agent node that decides to answer or call a tool."""
    response = llm_with_tools.invoke(state["messages"])
    return {{"messages": [response]}}

# Build LangGraph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(AVAILABLE_TOOLS))
workflow.set_entry_point("agent")

# Edges
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
app_graph = workflow.compile()

class AgentTask(BaseModel):
    task: str

@app.post("/run")
async def run_workflow(req: AgentTask):
    """Execute LangGraph multi-agent workflow."""
    final_state = app_graph.invoke({{"messages": [HumanMessage(content=req.task)]}})
    last_message = final_state["messages"][-1]
    return {{"task": req.task, "final_output": last_message.content}}

@app.get("/")
async def root():
    return {{"name": "{config.get("project_name", "Multi-Agent System")}", "status": "running", "mcp_enabled": True}}

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


def _generate_llm_finetuner(output_path: Path, config: dict) -> list[str]:
    """Generate an LLM fine-tuning project (LoRA)."""
    files = []

    # main.py (Training Script)
    train_code = f'''"""
{config.get("project_name", "LLM Fine-Tuner")} — Powered by ShipAI
Unsloth-optimized PEFT/LoRA fine-tuning script.
"""
import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Configuration
BASE_MODEL = "{config.get("base_model", "unsloth/llama-3-8b-bnb-4bit")}"
MAX_SEQ_LENGTH = 2048
DATASET_PATH = "{config.get("dataset_path", "data.jsonl")}"

def main():
    print("🚀 Initializing Unsloth Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    print("🔧 Applying LoRA Adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[{", ".join([f'"{m.strip()}"' for m in config.get("target_modules", "q_proj,k_proj,v_proj,o_proj").split(',')])}],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    print("📊 Loading Dataset...")
    if os.path.exists(DATASET_PATH):
        dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    else:
        print(f"⚠️ Dataset {{DATASET_PATH}} not found. Using sample data.")
        # Fallback sample dataset
        dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
        
    def formatting_prompts_func(examples):
        instructions = examples.get("instruction", examples.get("prompt", []))
        inputs       = examples.get("input", [""] * len(instructions))
        outputs      = examples.get("output", examples.get("response", []))
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = f"Instruction: {{instruction}}\\nInput: {{input}}\\nOutput: {{output}}"
            texts.append(text)
        return {{"text": texts}}
        
    dataset = dataset.map(formatting_prompts_func, batched=True)

    print("🎓 Starting Training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60, # Increase for real training
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    trainer.train()

    print("💾 Saving LoRA Adapters...")
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")
    
    print("✅ Training Complete! You can now merge and export to GGUF for Ollama.")

if __name__ == "__main__":
    main()
'''
    train_path = output_path / "train.py"
    train_path.write_text(train_code, encoding="utf-8")
    files.append(str(train_path))
    
    # Api server for the finetuned model
    api_code = f'''"""
FastAPI Server for {config.get("project_name", "Fine-Tuned LLM")}
"""
from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI(title="Fine-Tuned LLM API")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512

@app.post("/generate")
async def generate(req: GenerateRequest):
    return {{"response": f"Inference pipeline placeholder for: {{req.prompt}}"}}

@app.get("/")
async def root():
    return {{"status": "running", "model": "{config.get("base_model", "unsloth/llama-3-8b-bnb-4bit")}"}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
'''
    api_path = output_path / "main.py"
    api_path.write_text(api_code, encoding="utf-8")
    files.append(str(api_path))

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


def _generate_requirements(template_id: str, tier: str, config: dict) -> str:
    base = "fastapi==0.115.12\nuvicorn==0.34.2\nhttpx==0.28.1\npython-dotenv==1.1.0\n"
    
    # Infrastructure dependencies
    if tier in ["starter", "pro", "enterprise"]:
        base += "celery==5.4.0\nredis==5.2.1\n"
    if tier in ["pro", "enterprise"]:
        base += "prometheus-client==0.21.1\nlocust==2.32.4\n"

    # Provider dependencies
    llm_provider = config.get("llm_provider", "ollama")
    if llm_provider == "openai":
        base += "openai==1.52.2\n"
    elif llm_provider == "anthropic":
        base += "anthropic==0.39.0\n"

    # Template dependencies
    if template_id == "rag_chatbot":
        base += "langchain==0.3.25\n"
        mode = config.get("retrieval_mode", "vector_chroma")
        if mode == "vector_chroma":
            base += "chromadb==1.0.7\n"
        elif mode == "vector_pinecone":
            base += "pinecone-client==5.0.1\n"
        elif mode == "vectorless_bm25":
            base += "rank_bm25==0.2.2\nnumpy==2.0.0\n"
        elif mode == "hybrid":
            base += "chromadb==1.0.7\nrank_bm25==0.2.2\nnumpy==2.0.0\n"
    elif template_id == "multi_agent":
        base += "langchain==0.3.25\nlangchain-community==0.3.2\nlanggraph==0.2.38\nmcp==1.0.0\n"
    elif template_id == "data_analyzer":
        base += "pandas==2.2.3\nplotly==6.1.2\n"
    elif template_id == "llm_finetuner":
        base += "torch==2.3.0\ntransformers==4.40.1\ntrl==0.8.6\ndatasets==2.19.0\nunsloth==2024.4\n"
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
