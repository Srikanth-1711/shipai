  # ShipAI 🚢

  > The AI Engineer You Can Hire at Scale. Ship AI Products in Minutes, Not Months.

  ShipAI is an offline-first AI Engineering platform that democratizes the creation of production-ready AI applications. It leverages local models (Ollama) and Cloud APIs (OpenAI/Anthropic) to generate fully functional AI projects injected with senior-level infrastructure patterns. 

  Zero tokens. Zero cloud bills. 100% private. (Or plug in your own API keys for massive scale).

  ## ✨ Features

  - **3 Ways to Deploy**: Use the CLI (Engineers), the API/SDK (Enterprise), or the completely standalone **1-Click Desktop App** (Non-tech users).
  - **AI Architecture Advisor**: Describe your problem; get expert recommendations from 16 AI patterns (RAG, Agents, Fine-Tuning).
  - **Template Engine**: Instantly generate complete AI codebases:
    - 📚 RAG Chatbot (Vector DB & Smart Vector-less BM25 Search)
    - 🤖 Multi-Agent System
    - 📊 AI Data Analyzer
    - 🧠 LLM Fine-Tuner (Unsloth/PEFT Pipeline)

  ### 🌊 The "RAG for Builders" Difference
  Most tools only give you the surface-level RAG (Beginner). ShipAI handles the **Hidden Complexity** (Builder):
  - ✅ **Query Reformulation**: Automatically rewrites user queries for better retrieval.
  - ✅ **PII Masking**: Built-in regex filters to redact sensitive data before indexing.
  - ✅ **Hybrid Search**: Seamlessly switches between Vector (Chroma) and Sparse (BM25) retrieval.
  - ✅ **Enterprise Infra**: Every project comes pre-configured with 7 patterns:
    1. Rate Limiting & Caching (Redis)
    2. Asynchronous Workers (Celery)
    3. Observability (Prometheus/OpenTelemetry)
    4. Load Testing (Locust)
    5. API Gateway & Load Balancing
    6. Circuit Breakers
    7. Auto-Scaling Scaffolds

  ### 🧊 The "Agentic AI for Builders" Iceberg
  Building an agent is easy. Building a **production agent** is hard. ShipAI handles the complexity below the surface:
  - ✅ **State Persistence**: Built-in stubs for LangGraph checkpointers to recover from failures.
  - ✅ **Trajectory Tracing**: Real-time logging of the agent's "thinking" process and tool selection.
  - ✅ **Loop Protection**: Automatic step-counters to prevent infinite reasoning loops and cost overruns.
  - ✅ **Human-in-the-Loop**: Integrated endpoints for human approval before high-stakes tool calls.
  - ✅ **MCP Native**: Standardized integration with Model Context Protocol for enterprise data access.

  - **100% Offline**: Powered by local LLMs via Ollama. No data leaves your machine.

  ## 🚀 Quick Start

  ### 1. Requirements
  - Python 3.12+
  - [Ollama](https://ollama.ai/) installed and running locally

  ### 2. Installation
  Clone the repository and install the ShipAI CLI:
  ```bash
  git clone https://github.com/yourusername/shipai.git
  cd shipai
  pip install -r backend/requirements.txt
  ```

  ### 3. Setup
  Run the installer to detect hardware and pull recommended models:
  ```bash
  python -m shipai install
  ```

  ### 4. Create Your First Project
  Generate a complete, production-ready AI project:
  ```bash
  python -m shipai create rag_chatbot
  ```

  ### 5. Start the Platform
  Run the ShipAI platform backend:
  ```bash
  python -m shipai serve
  ```

  ## 🛠️ Tech Stack

  - **Backend**: FastAPI, Pydantic, Uvicorn
  - **AI/LLM**: Ollama, LangChain, ChromaDB
  - **Frontend**: Vanilla HTML/CSS/JS (Glassmorphism design)
  - **Deployment**: Docker, Docker Compose

  ## 📦 Deployment

  ShipAI can be deployed using Docker Compose:

  ```bash
  docker-compose up -d
  ```
  - Backend API available at `http://localhost:8000`
  - Frontend UI available at `http://localhost:8080`

  ## 📄 License

  MIT License. See `LICENSE` for details.
