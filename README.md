# ShipAI 🚢

> The AI Engineer You Can Hire at Scale. Ship AI Products in Minutes, Not Months.

ShipAI is an offline-first AI Engineering platform that democratizes the creation of production-ready AI applications. It leverages local models via Ollama to generate fully functional AI projects injected with senior-level infrastructure patterns. 

Zero tokens. Zero cloud bills. 100% private.

## ✨ Features

- **AI Architecture Advisor**: Describe your problem; get expert recommendations from 16 AI patterns (RAG, Agents, Fine-Tuning).
- **Hardware-Aware**: Automatically detects your hardware (CPU/GPU/RAM) to recommend the most optimal local models.
- **Template Engine**: Instantly generate complete AI codebases:
  - 📚 RAG Chatbot
  - 🤖 Multi-Agent System
  - 📊 AI Data Analyzer
- **Production Infrastructure Built-in**: Every project comes pre-configured with 7 enterprise patterns:
  1. Rate Limiting
  2. Caching
  3. API Gateway
  4. Load Balancing
  5. Circuit Breaker
  6. Auto Scaling
  7. Message Queue
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
